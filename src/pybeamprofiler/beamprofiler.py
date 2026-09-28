"""The :class:`BeamProfiler` façade.

This is the object users hold: it owns a camera, runs a frame through
:mod:`pybeamprofiler.fitting`, and turns the result into a figure or a live
stream (the ``pybeamprofiler`` command in :mod:`pybeamprofiler.cli` is a thin
layer over it). The numerical work itself lives in ``fitting.py``; what is here is
the state that has to persist between frames — which camera, which fit
method, and the previous frame's parameters that each new fit warm-starts
from.

Two display paths hang off :meth:`BeamProfiler.plot`, chosen by
environment rather than by argument: a live async loop inside a Jupyter
kernel, and the Dash GUI everywhere else.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import socket
import threading
import time
import webbrowser
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import numpy as np
import plotly.graph_objs as go
from PIL import Image
from plotly.subplots import make_subplots

from . import fitting
from .basler import BaslerCamera
from .camera import Camera
from .constants import (
    D4SIGMA_FACTOR,
    DEFAULT_DASH_PORT,
    FW_1E_FACTOR,
    GAUSSIAN_TO_FWHM,
    MAX_DISPLAY_DIM,
    MAX_FIT_2D_DIM,
)
from .flir import FlirCamera
from .simulated import SimulatedCamera

logger = logging.getLogger(__name__)

# How long one notebook-stream fetch may wait for a frame. Bounding it is what
# keeps stop() prompt: stop() waits for the fetch in flight to finish.
_STREAM_FETCH_TIMEOUT = 1.0

# Consecutive failed frames after which the notebook stream gives up.
_MAX_STREAM_FAILURES = 50

_LOOPBACK_NAMES = frozenset({"127.0.0.1", "::1", "localhost"})


@contextlib.contextmanager
def _without_loopback_reverse_dns() -> Iterator[None]:
    """Answer ``socket.getfqdn`` for the loopback address without asking DNS.

    ``http.server`` looks the address it binds to back up as a host name,
    only to keep it in ``server_name``, which nothing here reads. On some
    Macs that reverse lookup of 127.0.0.1 takes more than 30 s (35 s with
    Homebrew's Python 3.14; on GitHub's macOS runners, long enough that no
    test server came up in 30 s), and all of it passes before the server
    accepts a connection. Any other name still goes to the real lookup.
    """
    real_getfqdn = socket.getfqdn

    def getfqdn(name: str = "") -> str:
        return "localhost" if name in _LOOPBACK_NAMES else real_getfqdn(name)

    socket.getfqdn = getfqdn  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        socket.getfqdn = real_getfqdn


class BeamProfiler:
    """Laser beam profiler with Gaussian fitting capabilities.

    Supports 1D and 2D Gaussian fitting of beam profiles from static images
    or camera streams. Provides beam width measurements in various definitions.

    Args:
        camera: Camera type ('simulated', 'flir', 'basler'); or a
            :class:`~pybeamprofiler.camera.Camera` you built yourself, e.g.
            ``BaslerCamera(cti_file=...)``, which is opened here unless it
            already is, and closed on leaving a ``with`` block like a camera
            the profiler opened itself; or None for the default simulator.
        file: Path to a static image file to analyze
        fit: Fitting method ('1d', '2d', 'linecut')
        definition: Width definition ('gaussian' for 1/e², 'fwhm', 'd4s')
        exposure_time: Camera exposure time in seconds (default: camera default)
        pixel_size: Pixel pitch in micrometers. Required with *file*; with a
            camera it overrides the value the camera reports, which is worth
            doing when binning is on or the camera reports nothing useful.
        serial_number: Open this specific device when more than one camera of
            the requested type is attached. Ignored by the simulated camera,
            and when *camera* is an instance (it already picked its device).

    Attributes:
        width_x: Beam width in x, in the selected definition (μm)
        width_y: Beam width in y, in the selected definition (μm)
        center_x: Beam center x position (pixels)
        center_y: Beam center y position (pixels)
        angle_deg: Beam rotation angle (degrees; 2D fit only, else 0)
        peak_value: Peak intensity of the last analyzed frame

    Widths and centre are NaN before the first frame is analysed, and for
    any frame without a measurable beam; so is the angle when a 2D fit finds
    no beam.
    """

    def __init__(
        self,
        camera: str | Camera | None = None,
        file: str | None = None,
        fit: str = "1d",
        definition: str = "gaussian",
        exposure_time: float | None = None,
        pixel_size: float | None = None,
        serial_number: str | None = None,
    ) -> None:
        """Initialize the beam profiler.

        If neither ``file`` nor ``camera`` is provided, a
        :class:`SimulatedCamera` is used by default.

        Raises:
            ValueError: If ``pixel_size`` is missing (or not positive) for a
                static image file, or if neither camera nor file loaded.
            RuntimeError: If a physical camera (FLIR/Basler) fails to open,
                or a camera instance passed in cannot be opened.
        """
        self.camera: Camera | None = None
        self.fit_method: str = fit
        self.definition: str = definition

        # NaN until a frame has been analysed, and for any frame without a
        # measurable beam.
        self.width_x: float = math.nan
        self.width_y: float = math.nan
        self.center_x: float = math.nan
        self.center_y: float = math.nan
        self.angle_deg: float = 0.0
        self.peak_value: float = 0.0

        # Warm starts: the last *plausible* fit on each path.
        self._last_popt_x: np.ndarray | list[Any] | None = None
        self._last_popt_y: np.ndarray | list[Any] | None = None
        self._last_popt_2d: np.ndarray | list[Any] | None = None
        # Frame geometry the warm starts belong to.
        self._analysis_shape: tuple[int, ...] | None = None
        self._stream_task: asyncio.Task[None] | None = None
        # See _next_frame and stop(): the lock is held for each notebook-stream
        # fetch, the event tells the loop and its worker to stand down.
        self._stream_fetch_lock = threading.Lock()
        self._stream_stopping = threading.Event()
        self._heatmap_only = False

        # What the last analysed frame produced, for drawing it.
        self.last_img: np.ndarray | None = None
        self._last_proj_x: np.ndarray | None = None
        self._last_proj_y: np.ndarray | None = None
        self._ellipse: tuple[float, float, float, float, float] | None = None
        self._linecut_x: int | None = None
        self._linecut_y: int | None = None

        if pixel_size is not None and not (math.isfinite(pixel_size) and pixel_size > 0):
            raise ValueError(f"pixel_size must be a positive number, got {pixel_size}")

        # Kept so :meth:`attach_camera` knows whether the scale was the
        # caller's choice (honour it) or the previous camera's (re-derive it).
        self._pixel_size_override: float | None = pixel_size

        if file:
            self._load_file(file)
            self._mode = "static"
            if pixel_size is None:
                raise ValueError("Pixel size must be provided for static beam image files")
            self.pixel_size = pixel_size
        elif isinstance(camera, Camera):
            self._adopt_camera(camera)
        elif camera:
            self._initialize_camera(camera, serial_number)
        else:
            self.camera = SimulatedCamera()
            self.camera.open()
            self._mode = "camera"

        if self.camera:
            try:
                self.width_pixels = self.camera.width
                self.height_pixels = self.camera.height
                # An explicit pixel_size wins over whatever the camera reports.
                self.pixel_size = pixel_size if pixel_size is not None else self.camera.pixel_size
                if exposure_time is not None:
                    self.camera.set_exposure(exposure_time)
            except Exception:
                # The camera is open by now. A GenICam device stays claimed
                # until closed, so raising without releasing it would keep it
                # busy -- in a notebook, until the kernel restarts.
                self._release_camera()
                raise
        elif file and self.last_img is not None:
            pass
        else:
            raise ValueError("Either camera or file must be provided and successfully loaded")

    def _adopt_camera(self, camera: Camera) -> None:
        """Take over a camera the caller built, opening it unless it already is.

        Passing an instance is how a camera gets anything the names can't
        express -- a particular ``.cti`` file, above all.
        """
        self.camera = camera
        if not camera.is_open:
            try:
                camera.open()
            except Exception as e:
                try:
                    camera.close()
                except Exception:
                    logger.debug("Error closing a camera that failed to open", exc_info=True)
                logger.error(f"Failed to open {type(camera).__name__}: {e}")
                raise RuntimeError(f"Failed to open {type(camera).__name__}: {e}") from e
        self._mode = "camera"

    def _initialize_camera(self, camera: str, serial_number: str | None = None) -> None:
        """Open the named camera type.

        A physical camera that fails to open is an error worth surfacing —
        silently handing back simulated data would look like a working
        measurement. Only an unrecognised name falls back to the simulator.

        Args:
            camera: Camera type string ('flir', 'basler', 'simulated').
            serial_number: Specific device to open when several are attached.

        Raises:
            RuntimeError: If a physical camera fails to open.
        """
        camera_lower = camera.lower()
        if camera_lower == "flir":
            self.camera = FlirCamera(serial_number=serial_number)
        elif camera_lower == "basler":
            self.camera = BaslerCamera(serial_number=serial_number)
        elif camera_lower == "simulated":
            self.camera = SimulatedCamera()
        else:
            logger.warning(f"Unknown camera {camera}, using Simulated.")
            self.camera = SimulatedCamera()

        try:
            self.camera.open()
            self._mode = "camera"
        except Exception as e:
            # A half-finished open() can still hold the device or its
            # producer; close() is safe to call on it either way.
            try:
                self.camera.close()
            except Exception:
                logger.debug("Error closing a camera that failed to open", exc_info=True)
            # Don't fallback to simulated for physical cameras
            if camera_lower in ["flir", "basler"]:
                logger.error(f"Failed to open {camera} camera: {e}")
                raise RuntimeError(f"Failed to open {camera} camera: {e}") from e
            else:
                # Only fallback for unknown/simulated cameras
                logger.error(f"Failed to open camera: {e}")
                self.camera = SimulatedCamera()
                self.camera.open()
                self._mode = "camera"

    def _load_file(self, filename: str) -> None:
        """Load a static image file as a 2D intensity array.

        Colour images are collapsed to a single channel: an alpha channel is
        dropped and RGB is converted with the usual luminance weights, so a
        camera screenshot saved as a colour PNG still analyses correctly rather
        than blowing up on a 3D array later.

        Args:
            filename: Path to the image file.
        """
        try:
            with Image.open(filename) as img:
                data = np.array(img)
        except Exception as e:
            logger.error(f"Error loading image file {filename}: {e}")
            raise

        if data.ndim == 3:
            channels = data.shape[2]
            if channels >= 3:
                logger.info("Converting %d-channel image to grayscale", channels)
                rgb = data[:, :, :3].astype(np.float64)
                data = (rgb @ [0.299, 0.587, 0.114]).astype(data.dtype)
            else:
                # Grayscale + alpha, or a single-channel image stored as 3D.
                data = data[:, :, 0]
        elif data.ndim != 2:
            raise ValueError(f"Expected a 2D or 3D image, got a {data.ndim}D array from {filename}")

        self.last_img = data
        self.height_pixels, self.width_pixels = data.shape

    def __enter__(self) -> BeamProfiler:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Context manager exit: stop any live stream, then release the camera.

        The stream has to go first. Left running, a notebook stream kept
        fetching from the closed camera, failing every frame, for as long as
        the kernel lived.
        """
        try:
            self.stop()
        except Exception:
            logger.warning("Error stopping the stream", exc_info=True)
        self._release_camera()
        return False

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute access to the underlying camera."""
        try:
            camera = object.__getattribute__(self, "camera")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None
        if camera and hasattr(camera, name):
            return getattr(camera, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # The model functions live in :mod:`pybeamprofiler.fitting`; they are
    # re-exposed here because ``BeamProfiler.gaussian`` is part of the public
    # API (and handy for plotting a fit alongside your own data).
    gaussian = staticmethod(fitting.gaussian)
    gaussian_2d = staticmethod(fitting.gaussian_2d)

    @property
    def width(self) -> float:
        """Average beam width (μm)."""
        return (self.width_x + self.width_y) / 2

    @property
    def diameter(self) -> float:
        """Beam diameter, same as width (μm)."""
        return self.width

    @property
    def radius(self) -> float:
        """Beam radius (μm)."""
        return self.width / 2

    def _to_sigma(self, width: float) -> float:
        """Convert a reported width back to sigma (μm) for the current definition.

        ``width_x`` / ``width_y`` are stored in whichever definition the user
        selected.  Normalising through sigma lets every derived property below
        report the right number no matter which definition produced the
        measurement.
        """
        if self.definition == "fwhm":
            return width / GAUSSIAN_TO_FWHM
        # Both 'gaussian' (1/e²) and 'd4s' report 4σ.
        return width / D4SIGMA_FACTOR

    @property
    def fwhm_x(self) -> float:
        """Full Width at Half Maximum in X direction (μm)."""
        return GAUSSIAN_TO_FWHM * self._to_sigma(self.width_x)

    @property
    def fwhm_y(self) -> float:
        """Full Width at Half Maximum in Y direction (μm)."""
        return GAUSSIAN_TO_FWHM * self._to_sigma(self.width_y)

    @property
    def fw_1e_x(self) -> float:
        """Full Width at 1/e of peak intensity in X direction (μm)."""
        return FW_1E_FACTOR * self._to_sigma(self.width_x)

    @property
    def fw_1e_y(self) -> float:
        """Full Width at 1/e of peak intensity in Y direction (μm)."""
        return FW_1E_FACTOR * self._to_sigma(self.width_y)

    @property
    def fw_1e2_x(self) -> float:
        """Full Width at 1/e² in X direction (μm)."""
        return D4SIGMA_FACTOR * self._to_sigma(self.width_x)

    @property
    def fw_1e2_y(self) -> float:
        """Full Width at 1/e² in Y direction (μm)."""
        return D4SIGMA_FACTOR * self._to_sigma(self.width_y)

    @property
    def height_x(self) -> float:
        """Peak image intensity (intensity units)."""
        return self.peak_value

    @property
    def height_y(self) -> float:
        """Peak image intensity (intensity units)."""
        return self.peak_value

    def _measure_fwhm(self, profile: np.ndarray) -> tuple[float, float, float]:
        """Measure FWHM directly from a profile — see :func:`fitting.measure_fwhm`."""
        return fitting.measure_fwhm(profile)

    def _measure_d4s(self, profile: np.ndarray) -> tuple[float, float]:
        """Measure D4σ directly from a profile — see :func:`fitting.measure_d4s`."""
        return fitting.measure_d4s(profile)

    def _fit_1d_gaussian(
        self,
        profile: np.ndarray,
        last_popt: np.ndarray | list[Any] | None = None,
    ) -> np.ndarray | list[Any]:
        """Fit a 1D Gaussian — see :func:`fitting.fit_1d_gaussian`."""
        return fitting.fit_1d_gaussian(profile, last_popt)

    _MAX_FIT_2D_DIM = MAX_FIT_2D_DIM

    def _fit_2d_gaussian(
        self,
        image: np.ndarray,
        sigma_hint: float | tuple[float, float] | None = None,
        center_hint: tuple[float, float] | None = None,
    ) -> np.ndarray | None:
        """Fit a rotated 2D Gaussian, warm-starting from the previous frame.

        Only a plausible fit becomes the next warm start. A failed frame
        leaves the last good one in place: the beam may only have been
        blocked for a moment, and if it has moved instead, the fit's own cold
        retry copes.

        Args:
            image: 2D intensity array.
            sigma_hint: Rough beam sigma in pixels along ``(x, y)``.
            center_hint: Rough beam centre in pixels. With *sigma_hint*, lets
                a small beam be cropped out of a large sensor rather than
                decimated below the fit's resolution.

        Returns:
            ``[amplitude, x0, y0, sigma_x, sigma_y, theta, offset]``, or
            ``None`` if the frame holds no beam the fit could find.
        """
        popt, ok = fitting.fit_2d_gaussian(
            image,
            self._last_popt_2d,
            max_dim=self._MAX_FIT_2D_DIM,
            sigma_hint=sigma_hint,
            center_hint=center_hint,
        )
        if not ok:
            return None
        popt = np.asarray(popt, dtype=float)
        self._last_popt_2d = popt
        return popt

    def beam_ellipse(self) -> tuple[float, float, float, float, float] | None:
        """The beam's outline on the last analysed frame, in pixel coordinates.

        Returns ``(cx, cy, rx, ry, angle_rad)``, the ellipse whose full axes
        are the reported widths in whichever definition is selected: the 1/e²
        contour for ``gaussian``, the half-maximum one for ``fwhm``. The one
        exception is ``2d`` mode with the Gaussian definition, which draws the
        fitted, tilted ellipse itself. Its reported widths are projections
        onto the image axes, and an ellipse built from those would smear a
        tilted beam's outline toward a circle.

        Returns:
            The ellipse, or ``None`` before the first frame and for a frame
            with no measurable beam.
        """
        return self._ellipse

    def _fit_projections(
        self, prof_x: np.ndarray, prof_y: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Fit both axis profiles, warm-starting from the previous frame.

        Returns ``None`` for an axis without a plausible fit. As in 2D, only
        plausible fits are kept as the next warm start.
        """
        popt_x, ok_x = fitting._fit_1d(prof_x, self._last_popt_x)
        popt_y, ok_y = fitting._fit_1d(prof_y, self._last_popt_y)
        if ok_x:
            self._last_popt_x = popt_x
        if ok_y:
            self._last_popt_y = popt_y
        return (popt_x if ok_x else None), (popt_y if ok_y else None)

    def _integrate(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sum the image down each axis and cache the result for plotting."""
        self._last_proj_x = np.sum(image, axis=0)
        self._last_proj_y = np.sum(image, axis=1)
        return self._last_proj_x, self._last_proj_y

    def analyze(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Measure the beam in *image* and update every reported parameter.

        Two things decide what happens here, and they are independent:

        * ``definition`` picks how the width is *measured*.  ``fwhm`` and
          ``d4s`` are read straight off the image with no model, so they also
          override ``fit_method`` — a shape-free measurement and a Gaussian
          fit would disagree, and the definition wins.
        * ``fit_method`` picks what the Gaussian fit is run against:
          ``1d`` the integrated profiles, ``2d`` the whole frame (the only
          mode that recovers a rotation angle), ``linecut`` a single row and
          column through the brightest pixel.

        Either way the axis fits are returned, because the GUI draws them
        alongside the data even when they didn't set the reported width.

        A frame with no measurable beam is reported as such: widths, centre
        and (in ``2d`` mode) angle become NaN and :meth:`beam_ellipse` returns
        ``None``, rather than carrying over the previous frame's numbers or
        presenting a fit to noise as a measurement.

        Args:
            image: 2D intensity array.

        Returns:
            ``(x_fit_params, y_fit_params)`` for the two axis profiles, each
            ``[amplitude, center, sigma, offset]`` or ``None`` if that axis
            could not be fitted.

        Raises:
            ValueError: If image is None, empty, or not 2D.
            TypeError: If image is not a numpy array.
        """
        if image is None:
            raise ValueError("Image cannot be None")

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(image)}")

        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got {image.ndim}D array")

        if image.size == 0:
            raise ValueError("Image cannot be empty")

        if image.shape != self._analysis_shape:
            # A new frame geometry -- an ROI, binning, another camera -- moves
            # the origin every cached parameter is measured from. Warm starts
            # from the old geometry only slow the next fit down, or worse.
            self._forget_warm_starts()
            self._analysis_shape = image.shape

        self.peak_value = float(np.max(image))
        self.angle_deg = 0.0
        self._ellipse = None
        self._linecut_x = self._linecut_y = None

        if self.definition in ("fwhm", "d4s"):
            return self._analyze_model_free(image)
        if self.fit_method == "linecut":
            return self._analyze_linecut(image)
        if self.fit_method == "2d":
            return self._analyze_2d(image)
        popt_x, popt_y = self._fit_projections(*self._integrate(image))
        self._record_gaussian(popt_x, popt_y)
        return popt_x, popt_y

    def _analyze_model_free(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """FWHM or D4σ, measured off the frame itself; the fits only draw curves."""
        proj_x, proj_y = self._integrate(image)
        if self.definition == "fwhm":
            # Band-limited profiles: see measure_fwhm_2d for why a full-frame
            # projection reads narrow on noisy data.
            measured = fitting.measure_fwhm_2d(image)
        else:
            # D4σ needs a 2D integration window (ISO 11146). Taking it from
            # full-frame projections would sum the noise of every beam-free
            # row into each sample.
            measured = fitting.measure_d4s_2d(image)
        self._record(*measured)
        return self._fit_projections(proj_x, proj_y)

    def _analyze_linecut(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Gaussian fits along the row and column through the brightest pixel."""
        peak_y, peak_x = np.unravel_index(int(np.argmax(image)), image.shape)
        # Remembered so the GUI can draw the crosshair it measured along.
        self._linecut_x, self._linecut_y = int(peak_x), int(peak_y)
        row = np.asarray(image[peak_y, :], dtype=float)
        column = np.asarray(image[:, peak_x], dtype=float)
        # The profiles plotted under the fit curves have to be the ones that
        # were fitted, not the full-frame projections.
        self._last_proj_x, self._last_proj_y = row, column
        popt_x, popt_y = self._fit_projections(row, column)
        if popt_x is None and popt_y is None:
            # No beam on either line: the "brightest pixel" is just the
            # loudest noise, and a crosshair there would look like a result.
            self._linecut_x = self._linecut_y = None
        self._record_gaussian(popt_x, popt_y)
        return popt_x, popt_y

    def _analyze_2d(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """A rotated 2D Gaussian over the whole frame."""
        # The projection fits run first: they feed the profile plots, and
        # they are a cheap estimate of the beam's size and position, which
        # decides whether a decimated fit grid can still resolve it.
        popt_x, popt_y = self._fit_projections(*self._integrate(image))
        sigma_hint = center_hint = None
        if popt_x is not None and popt_y is not None:
            sigma_hint = (abs(float(popt_x[2])), abs(float(popt_y[2])))
            center_hint = (float(popt_x[1]), float(popt_y[1]))

        popt = self._fit_2d_gaussian(image, sigma_hint=sigma_hint, center_hint=center_hint)
        if popt is None:
            self._record(math.nan, math.nan, math.nan, math.nan)
            self.angle_deg = math.nan
            return popt_x, popt_y

        _, x0, y0, sigma_x, sigma_y, theta, _ = (float(v) for v in popt)
        # Report widths along the *image* axes, as 1D mode does. The
        # principal-axis sigmas cannot be used directly: (sx, sy, theta) and
        # (sy, sx, theta+90) are the same ellipse, so on a near-round beam the
        # solver flips between them and the reported X and Y widths would
        # swap from frame to frame. The projected widths are invariant to it.
        sx_img, sy_img = fitting.image_axis_sigmas(sigma_x, sigma_y, theta)
        self._record(x0, y0, D4SIGMA_FACTOR * sx_img, D4SIGMA_FACTOR * sy_img)
        self._ellipse = (x0, y0, 2.0 * abs(sigma_x), 2.0 * abs(sigma_y), theta)
        # theta is canonicalised to the major axis in [0, pi).
        self.angle_deg = float(np.degrees(theta) % 180)
        return popt_x, popt_y

    def _record_gaussian(self, popt_x: np.ndarray | None, popt_y: np.ndarray | None) -> None:
        """Record a frame's result from the two axis fits (1/e² = 4σ widths)."""
        nan = (math.nan, math.nan)
        cx, wx = (popt_x[1], D4SIGMA_FACTOR * abs(popt_x[2])) if popt_x is not None else nan
        cy, wy = (popt_y[1], D4SIGMA_FACTOR * abs(popt_y[2])) if popt_y is not None else nan
        self._record(cx, cy, wx, wy)

    def _record(self, cx: float, cy: float, width_x_px: float, width_y_px: float) -> None:
        """Store one frame's result, and the axis-aligned outline that goes with it.

        Centres stay in pixels; widths are converted to μm here, once. Any
        NaN input means that part of the frame had no measurable beam.
        """
        self.center_x, self.center_y = float(cx), float(cy)
        self.width_x = float(width_x_px) * self.pixel_size
        self.width_y = float(width_y_px) * self.pixel_size
        if all(math.isfinite(float(v)) for v in (cx, cy, width_x_px, width_y_px)):
            self._ellipse = (float(cx), float(cy), width_x_px / 2.0, width_y_px / 2.0, 0.0)

    def attach_camera(self, camera: Camera, *, close_previous: bool = True) -> None:
        """Swap in an already-open *camera*, replacing the current one.

        Everything derived from the old camera is dropped, which matters more
        than it looks: the fitter warm-starts each frame from the previous
        frame's parameters, and a centre or sigma measured on a 1024x1024
        sensor is a nonsense starting point for a 1280x1024 one. Carrying it
        over makes the first fits after a switch converge slowly or not at
        all.

        Args:
            camera: An **opened** camera to take over from the current one.
            close_previous: Close the camera being replaced. Leave this on
                unless you intend to keep using it — a GenICam device stays
                claimed until it is closed, so the old one would block any
                attempt to reopen it.
        """
        previous = self.camera
        if previous is camera:
            return

        if previous is not None and close_previous:
            try:
                if previous.is_acquiring:
                    previous.stop_acquisition()
                previous.close()
            except Exception:
                logger.warning("Error closing the previous camera", exc_info=True)

        self.camera = camera
        self._mode = "camera"
        self.width_pixels = camera.width
        self.height_pixels = camera.height
        # A pixel size the caller pinned at construction time stays pinned;
        # otherwise take the new camera's own pitch rather than the old one's.
        self.pixel_size = (
            self._pixel_size_override
            if self._pixel_size_override is not None
            else camera.pixel_size
        )
        self.reset_analysis()

    def reset_analysis(self) -> None:
        """Forget everything measured from previous frames.

        Clears the warm starts, the last frame and everything derived from it,
        so the next :meth:`analyze` starts from a cold estimate. Call it
        whenever the frame's geometry changes in a way :meth:`analyze` can't
        see -- for example an ROI moved without changing its size.

        A loaded file's image is kept: in file mode ``last_img`` is the
        source itself, not a cached frame, and nothing could fetch it again.
        """
        self._forget_warm_starts()
        self._analysis_shape = None
        self._last_proj_x = None
        self._last_proj_y = None
        self._ellipse = None
        self._linecut_x = None
        self._linecut_y = None
        if getattr(self, "_mode", None) != "static":
            self.last_img = None
        self.width_x = math.nan
        self.width_y = math.nan
        self.center_x = math.nan
        self.center_y = math.nan
        self.angle_deg = 0.0
        self.peak_value = 0.0

    def _forget_warm_starts(self) -> None:
        """Drop the cached fit parameters that seed the next frame's fits."""
        self._last_popt_x = None
        self._last_popt_y = None
        self._last_popt_2d = None

    def stop(self) -> None:
        """Stop the notebook live stream, if one is running, and stop acquisition.

        Safe to call from another cell while the stream runs. It waits for a
        fetch already in flight to finish before stopping the camera (at most
        ``_STREAM_FETCH_TIMEOUT``), so the stream can't restart acquisition
        behind its back.
        """
        self._stream_stopping.set()
        task, self._stream_task = self._stream_task, None
        if task is not None:
            task.cancel()
        with self._stream_fetch_lock:
            if self._mode == "camera" and self.camera is not None and self.camera.is_acquiring:
                self.camera.stop_acquisition()

    def plot(
        self,
        num_img: int | None = None,
        heatmap_only: bool = False,
    ) -> asyncio.Task[None] | None:
        """Display beam profile with Gaussian fitting visualization.

        Args:
            num_img: ``1`` for a single shot, ``None`` to stream continuously.
            heatmap_only: Draw only the heatmap, without the profile curves.

        Returns:
            In single-shot or static mode, returns `None`.
            In streaming mode within a Jupyter kernel, returns the background
            `asyncio.Task` powering the live visualization; :meth:`stop` ends
            it. Anywhere else the Dash GUI is served and this blocks until
            Ctrl+C, returning `None`.

        Raises:
            ValueError: For any *num_img* other than 1 or None. Nothing
                captures a fixed number of frames, and quietly streaming
                forever instead is worse than saying so.
        """
        if num_img is not None and num_img != 1:
            raise ValueError(f"num_img must be 1 (a single shot) or None (stream), got {num_img}")

        self._heatmap_only = heatmap_only  # Store for _plot_stream to use

        if num_img == 1 or self._mode == "static":
            self._plot_single()
            return None
        else:
            return self._plot_stream()

    @staticmethod
    def _downsample_for_display(image: np.ndarray, max_dim: int = MAX_DISPLAY_DIM) -> np.ndarray:
        """Downsample an image for browser display — see :func:`fitting.downsample`."""
        return fitting.downsample(image, max_dim)

    def _ellipse_points(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Sample the fitted 1/e² ellipse, in μm, ready to hand to Plotly."""
        ellipse = self.beam_ellipse()
        if ellipse is None:
            return None
        cx, cy, rx, ry, angle = ellipse
        t = np.linspace(0, 2 * np.pi, 100)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        cos_t, sin_t = np.cos(t), np.sin(t)
        xe = cx + rx * cos_t * cos_a - ry * sin_t * sin_a
        ye = cy + rx * cos_t * sin_a + ry * sin_t * cos_a
        return xe * self.pixel_size, ye * self.pixel_size

    def _camera_info_html(self) -> str:
        """Build an HTML snippet summarising the camera and current settings."""
        parts: list[str] = []
        cam = self.camera
        if cam is None:
            return ""

        model = getattr(cam, "device_model", None)
        vendor = getattr(cam, "device_vendor", None)
        serial = getattr(cam, "serial_number", None)
        if vendor and model:
            parts.append(f"{vendor} {model}")
        elif model:
            parts.append(model)
        elif isinstance(cam, SimulatedCamera):
            parts.append("Simulated")

        if serial:
            parts.append(f"S/N: {serial}")

        w = getattr(cam, "width", None) or getattr(cam, "width_pixels", None)
        h = getattr(cam, "height", None) or getattr(cam, "height_pixels", None)
        if w and h:
            parts.append(f"{w}×{h}")

        if cam.exposure_time is not None:
            exp_ms = cam.exposure_time * 1000
            parts.append(f"Exp: {exp_ms:.2f} ms")

        if cam.gain is not None:
            parts.append(f"Gain: {cam.gain:.1f}")

        if not parts:
            return ""
        return "<span style='font-size:12px; color:#888'>" + " | ".join(parts) + "</span><br>"

    def _create_fast_figure(
        self,
        image: np.ndarray,
        popt_x: np.ndarray | list[Any] | None,
        popt_y: np.ndarray | list[Any] | None,
    ) -> go.Figure:
        """Create simplified figure with heatmap only for faster rendering.

        Args:
            image: 2D intensity array
            popt_x: X projection fit parameters
            popt_y: Y projection fit parameters

        Returns:
            Plotly figure with heatmap and ellipse overlay
        """
        if image is None:
            return go.Figure()

        fig = go.Figure()

        h, w = image.shape
        display_img = self._downsample_for_display(image)
        dh, dw = display_img.shape

        x_coords = np.linspace(0, (w - 1) * self.pixel_size, dw)
        y_coords = np.linspace(0, (h - 1) * self.pixel_size, dh)

        fig.add_trace(
            go.Heatmap(
                z=display_img,
                x=x_coords,
                y=y_coords,
                colorscale="Hot",
                showscale=True,
                colorbar=dict(thickness=15, len=0.7),
            )
        )

        # Add linecut crosshair lines if using linecut method
        if (
            self.fit_method == "linecut"
            and self._linecut_x is not None
            and self._linecut_y is not None
        ):
            linecut_x_um = self._linecut_x * self.pixel_size
            linecut_y_um = self._linecut_y * self.pixel_size

            # Vertical line at linecut_x
            fig.add_trace(
                go.Scatter(
                    x=[linecut_x_um, linecut_x_um],
                    y=[0, (h - 1) * self.pixel_size],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                    name="Linecut X",
                    showlegend=False,
                )
            )
            # Horizontal line at linecut_y
            fig.add_trace(
                go.Scatter(
                    x=[0, (w - 1) * self.pixel_size],
                    y=[linecut_y_um, linecut_y_um],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                    name="Linecut Y",
                    showlegend=False,
                )
            )

        ellipse = self._ellipse_points()
        if ellipse is not None:
            fig.add_trace(
                go.Scatter(
                    x=ellipse[0],
                    y=ellipse[1],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name=f"{self.definition} Width",
                    showlegend=False,
                )
            )

        center_x_um = self.center_x * self.pixel_size
        center_y_um = self.center_y * self.pixel_size
        title = "<b>Beam Profile</b><br>"
        title += self._camera_info_html()
        title += (
            f"<span style='font-size:14px'>Width: X={_fmt(self.width_x)}μm, "
            f"Y={_fmt(self.width_y)}μm | "
        )
        title += f"Center: ({_fmt(center_x_um)}, {_fmt(center_y_um)})μm</span><br>"
        title += f"<span style='font-size:12px'>Peak={self.peak_value:.0f}"
        if self.fit_method == "2d":
            title += f" | Angle={_fmt(self.angle_deg)}°"
        title += "</span>"

        h, w = image.shape
        x_range = [0, w * self.pixel_size]
        y_range = [0, h * self.pixel_size]

        fig.update_layout(
            uirevision="constant",
            title_text=title,
            title_font_size=14,
            autosize=True,
            margin=dict(l=40, r=20, t=110, b=40),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                range=y_range,
                title="Y (μm)",
                title_font_size=12,
            ),
            xaxis=dict(
                constrain="domain",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                range=x_range,
                title="X (μm)",
                title_font_size=12,
            ),
            showlegend=False,
            plot_bgcolor="rgba(240,240,240,0.5)",
        )

        return fig

    def _create_figure(
        self,
        image: np.ndarray,
        popt_x: np.ndarray | list[Any] | None,
        popt_y: np.ndarray | list[Any] | None,
    ) -> go.Figure:
        """Create complete figure with beam image and projection plots.

        Args:
            image: 2D intensity array
            popt_x: X projection fit parameters
            popt_y: Y projection fit parameters

        Returns:
            Plotly figure with 2D heatmap and aligned X/Y projection plots
        """
        if image is None:
            return go.Figure()

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.3, 0.7],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "heatmap"}, {"type": "xy"}],
            ],
            subplot_titles=("", "", "", ""),
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )

        # Beam Image (heatmap) — display-only downsampling
        h, w = image.shape
        display_img = self._downsample_for_display(image)
        dh, dw = display_img.shape
        x_coords = np.linspace(0, (w - 1) * self.pixel_size, dw)
        y_coords = np.linspace(0, (h - 1) * self.pixel_size, dh)

        fig.add_trace(
            go.Heatmap(
                z=display_img,
                x=x_coords,
                y=y_coords,
                colorscale="Hot",
                showscale=True,
                colorbar=dict(x=1.15, thickness=15, len=0.5),
            ),
            row=2,
            col=1,
        )

        # Add linecut crosshair lines if using linecut method
        if (
            self.fit_method == "linecut"
            and self._linecut_x is not None
            and self._linecut_y is not None
        ):
            linecut_x_um = self._linecut_x * self.pixel_size
            linecut_y_um = self._linecut_y * self.pixel_size

            # Vertical line at linecut_x
            fig.add_trace(
                go.Scatter(
                    x=[linecut_x_um, linecut_x_um],
                    y=[0, (h - 1) * self.pixel_size],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                    name="Linecut X",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )
            # Horizontal line at linecut_y
            fig.add_trace(
                go.Scatter(
                    x=[0, (w - 1) * self.pixel_size],
                    y=[linecut_y_um, linecut_y_um],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                    name="Linecut Y",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        ellipse = self._ellipse_points()
        if ellipse is not None:
            fig.add_trace(
                go.Scatter(
                    x=ellipse[0],
                    y=ellipse[1],
                    mode="lines",
                    line=dict(color="#FF4444", width=3, dash="dash"),
                    name=f"{self.definition} Width",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        # X Profile (Integrated) - Above beam image
        x = np.arange(len(image[0]))
        x_um = x * self.pixel_size  # Convert to physical dimensions
        # analyze() already summed these for every mode except linecut.
        proj_x = self._last_proj_x if self._last_proj_x is not None else np.sum(image, axis=0)

        fig.add_trace(
            go.Scatter(
                x=x_um,
                y=proj_x,
                mode="markers",
                name="Data X",
                marker=dict(size=3, color="#1f77b4", opacity=0.6),
            ),
            row=1,
            col=1,
        )
        if popt_x is not None:
            fitted_x = BeamProfiler.gaussian(x, *popt_x)
            fig.add_trace(
                go.Scatter(
                    x=x_um,
                    y=fitted_x,
                    mode="lines",
                    name="Fit X",
                    line=dict(color="#FF4444", width=2),
                ),
                row=1,
                col=1,
            )

        # Y Profile (Integrated) - Right of beam image, rotated
        y = np.arange(len(image))
        y_um = y * self.pixel_size  # Convert to physical dimensions
        proj_y = self._last_proj_y if self._last_proj_y is not None else np.sum(image, axis=1)

        fig.add_trace(
            go.Scatter(
                x=proj_y,
                y=y_um,
                mode="markers",
                name="Data Y",
                marker=dict(size=3, color="#2ca02c", opacity=0.6),
            ),
            row=2,
            col=2,
        )
        if popt_y is not None:
            fitted_y = BeamProfiler.gaussian(y, *popt_y)
            fig.add_trace(
                go.Scatter(
                    x=fitted_y,
                    y=y_um,
                    mode="lines",
                    name="Fit Y",
                    line=dict(color="#FF4444", width=2),
                ),
                row=2,
                col=2,
            )

        # Convert center coordinates to physical dimensions
        center_x_um = self.center_x * self.pixel_size
        center_y_um = self.center_y * self.pixel_size

        title = f"<b>Beam Profile Analysis - {self.definition.upper()}</b><br>"
        title += self._camera_info_html()
        title += (
            f"<span style='font-size:14px'>Width: X={_fmt(self.width_x)}μm, "
            f"Y={_fmt(self.width_y)}μm | "
        )
        title += f"Center: ({_fmt(center_x_um)}, {_fmt(center_y_um)})μm | "
        title += f"Peak: {self.peak_value:.0f}"
        if self.fit_method == "2d":
            title += f" | Angle: {_fmt(self.angle_deg)}°"
        title += "</span>"

        fig.update_layout(
            uirevision="constant",
            autosize=True,
            title_text=title,
            title_font_size=14,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            margin=dict(l=40, r=20, t=130, b=60),
            plot_bgcolor="rgba(245,245,245,0.5)",
        )

        # Align X profile's x-axis with beam image's x-axis
        fig.update_xaxes(
            matches="x3",
            row=1,
            col=1,
            showticklabels=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        )

        # Align Y profile's y-axis with beam image's y-axis
        fig.update_yaxes(
            matches="y3",
            row=2,
            col=2,
            showticklabels=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        )

        # Ensure proper aspect ratio for beam image
        # Convert image dimensions to physical coordinates
        h, w = image.shape
        x_range = [0, w * self.pixel_size]
        y_range = [0, h * self.pixel_size]

        fig.update_yaxes(
            scaleanchor="x3",
            scaleratio=1,
            row=2,
            col=1,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            range=y_range,
        )
        fig.update_xaxes(
            constrain="domain",
            row=2,
            col=1,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            range=x_range,
        )

        # Add labels to beam image axes with physical dimensions
        fig.update_xaxes(title_text="X (μm)", title_font_size=12, row=2, col=1)
        fig.update_yaxes(title_text="Y (μm)", title_font_size=12, row=2, col=1)

        return fig

    def _plot_single(self) -> None:
        """Capture one frame (or take the loaded file), analyse it, and show it."""
        if self._mode == "camera":
            camera = self.camera
            if camera is None:
                raise RuntimeError("Camera is not initialized")
            # Leave acquisition as it was found, even if the fetch fails.
            was_acquiring = camera.is_acquiring
            if not was_acquiring:
                camera.start_acquisition()
            try:
                img = camera.get_image()
            finally:
                if not was_acquiring:
                    camera.stop_acquisition()
        else:
            img = self.last_img

        if img is None:
            logger.error("No image available for analysis in _plot_single (img is None).")
            raise ValueError("No image available to analyze or plot (img is None).")
        popt_x, popt_y = self.analyze(img)
        fig = self._create_figure(img, popt_x, popt_y)
        fig.show()

    def _plot_stream(self) -> asyncio.Task[None] | None:
        """Start continuous streaming with live updates.

        Inside a Jupyter kernel this starts a background ``asyncio.Task``
        that redraws the figure in the cell, and returns it. Anywhere else --
        a terminal, a script, a terminal IPython session -- it serves the
        Dash GUI and blocks until Ctrl+C, returning ``None``.
        """
        if self._mode == "camera":
            if self.camera is None:
                raise RuntimeError("Camera is not initialized")
            if not self.camera.is_acquiring:
                self.camera.start_acquisition()

        if _in_notebook():
            return self._start_notebook_stream()
        self._serve_dash()
        return None

    def _next_frame(self) -> np.ndarray | None:
        """The next frame for the notebook stream, or ``None`` if there isn't one yet.

        Runs in a worker thread. It holds ``_stream_fetch_lock`` for the whole
        fetch, which is what lets :meth:`stop` wait out a fetch already in
        flight: cancelling the asyncio task does not stop its thread, and a
        fetch that began after ``stop_acquisition()`` would quietly start
        acquisition again (GenICam cameras restart on demand).
        """
        with self._stream_fetch_lock:
            if self._stream_stopping.is_set():
                return None
            if self._mode == "camera" and self.camera is not None:
                try:
                    return self.camera.get_image(timeout=_STREAM_FETCH_TIMEOUT)
                except TimeoutError:
                    return None
            return self.last_img

    def _start_notebook_stream(self) -> asyncio.Task[None] | None:
        """Run the live figure in the current notebook cell."""
        from IPython.display import clear_output, display

        heatmap_only = self._heatmap_only
        # Re-running the cell must not leave the previous loop fetching from
        # the same camera behind a new one.
        previous, self._stream_task = self._stream_task, None
        if previous is not None and not previous.done():
            previous.cancel()
        self._stream_stopping.clear()

        logger.info("Starting live stream%s...", " (heatmap only)" if heatmap_only else "")
        logger.info("Call profiler.stop() or cancel the returned task to stop\n")

        async def stream() -> None:
            frame_count = 0
            failures = 0
            start_time = time.time()
            try:
                while not self._stream_stopping.is_set():
                    # Yield to the kernel's event loop so interrupts and other
                    # callbacks get their turn between frames.
                    await asyncio.sleep(0)
                    try:
                        img = await asyncio.to_thread(self._next_frame)
                        if img is None:
                            await asyncio.sleep(0.01)
                            continue
                        popt_x, popt_y = await asyncio.to_thread(self.analyze, img)
                        build = self._create_fast_figure if heatmap_only else self._create_figure
                        fig = await asyncio.to_thread(build, img, popt_x, popt_y)
                    except Exception as e:
                        # One bad frame shouldn't end the stream, but a camera
                        # that has gone away fails every frame, and would
                        # otherwise retry forever without a word.
                        failures += 1
                        if failures >= _MAX_STREAM_FAILURES:
                            logger.warning(
                                "Live stream stopped after %d failed frames in a row: %s",
                                failures,
                                e,
                            )
                            break
                        logger.log(
                            logging.WARNING if failures == 1 else logging.DEBUG,
                            "Live stream frame failed: %s",
                            e,
                        )
                        await asyncio.sleep(0.05)
                        continue
                    failures = 0
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    current_title = fig.layout.title.text if fig.layout.title else ""
                    fig.update_layout(
                        title_text=(
                            f"{current_title}<br>"
                            f"<span style='font-size:11px; color:#666'>"
                            f"Frame #{frame_count} | FPS: {fps:.1f}</span>"
                        )
                    )
                    clear_output(wait=True)
                    display(fig)
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            finally:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"\nStream stopped: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps)"
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # A kernel normally runs cells inside its event loop. Without one
            # there is nothing to schedule a task on, so run the stream here
            # until it is interrupted.
            asyncio.run(stream())
            return None
        task = loop.create_task(stream())
        self._stream_task = task
        return task

    def _serve_dash(self) -> None:
        """Serve the Dash GUI on localhost and block until Ctrl+C."""
        from . import dash_app

        app = dash_app.create_app(self)

        url = f"http://127.0.0.1:{DEFAULT_DASH_PORT}"
        print(f"\npyBeamprofiler running at {url}")
        print("Press Ctrl+C to stop.\n", flush=True)
        logger.info(f"Starting Dash server at {url}")

        # Suppress dev-server chatter so the only startup output users see
        # is the two lines above. Werkzeug/Flask still print errors.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        logging.getLogger("dash").setLevel(logging.WARNING)
        logging.getLogger("dash.dash").setLevel(logging.WARNING)
        try:
            import flask.cli as _flask_cli

            _flask_cli.show_server_banner = lambda *a, **kw: None  # ty: ignore[invalid-assignment]
        except ImportError:
            pass

        if os.environ.get("PYBEAMPROFILER_NO_BROWSER") != "1":

            def open_browser() -> None:
                time.sleep(0.5)
                webbrowser.open(url)

            threading.Thread(target=open_browser, daemon=True).start()

        try:
            # The host is pinned. Left out, Dash takes it from $HOST, and with
            # HOST=0.0.0.0 in the environment the GUI -- no authentication,
            # and it writes camera settings -- would be open to the whole
            # network while the line above promises localhost.
            with _without_loopback_reverse_dns():
                app.run(host="127.0.0.1", port=DEFAULT_DASH_PORT, debug=False, use_reloader=False)
        except KeyboardInterrupt:
            pass
        finally:
            # werkzeug swallows the Ctrl+C that stops it, so shutdown happens
            # here rather than in a signal handler. Its request threads can
            # still be in the middle of a tick: pausing the ticks first and
            # then taking their lock means none of them is mid-fetch when the
            # camera closes, and none starts another fetch afterwards.
            logger.info("Stopping Dash server...")
            dash_app._server_paused = True
            with dash_app._callback_lock:
                self._release_camera()
        # Reached only when the server stopped cleanly, which means Ctrl+C:
        # werkzeug swallows the interrupt, so main() never sees it to report.
        print("\nStopped.", flush=True)

    def _release_camera(self) -> None:
        """Stop acquisition and close the camera, logging rather than raising."""
        camera = self.camera
        if camera is None:
            return
        try:
            if camera.is_acquiring:
                camera.stop_acquisition()
            camera.close()
        except Exception:
            logger.warning("Error releasing the camera", exc_info=True)


def _fmt(value: float, spec: str = ".1f") -> str:
    """Format a measured value for a title, with a dash for "no measurement"."""
    return format(value, spec) if math.isfinite(value) else "—"


def _in_notebook() -> bool:
    """Whether this is running inside a Jupyter kernel.

    ``get_ipython()`` alone can't tell: a terminal IPython session has one
    too, but no kernel and nothing to draw into. Sent down the notebook path,
    it ran the loop with ``asyncio.run`` and plotly opened a browser tab for
    every single frame.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and getattr(shell, "kernel", None) is not None


if __name__ == "__main__":
    # ``python -m pybeamprofiler.beamprofiler`` worked before the command moved
    # to cli.py (0.3.0 and earlier), so it still does. Prefer ``python -m
    # pybeamprofiler``: this form makes runpy warn that the package had already
    # imported the module. Every release has printed that warning; it is
    # harmless.
    from .cli import main

    raise SystemExit(main())
