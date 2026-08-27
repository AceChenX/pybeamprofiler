"""The :class:`BeamProfiler` façade and the command-line entry point.

This is the object users hold: it owns a camera, runs a frame through
:mod:`pybeamprofiler.fitting`, and turns the result into a figure or a live
stream. The numerical work itself lives in ``fitting.py``; what is here is
the state that has to persist between frames — which camera, which fit
method, and the previous frame's parameters that each new fit warm-starts
from.

Three display paths hang off :meth:`BeamProfiler.plot`, chosen by
environment rather than by argument: a live async loop inside Jupyter, the
Dash app in a terminal, and a matplotlib animation if Dash is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import threading
import time
import webbrowser
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


class BeamProfiler:
    """Laser beam profiler with Gaussian fitting capabilities.

    Supports 1D and 2D Gaussian fitting of beam profiles from static images
    or camera streams. Provides beam width measurements in various definitions.

    Args:
        camera: Camera type ('simulated', 'flir', 'basler'), or None for default
        file: Path to a static image file to analyze
        fit: Fitting method ('1d', '2d', 'linecut')
        definition: Width definition ('gaussian' for 1/e², 'fwhm', 'd4s')
        exposure_time: Camera exposure time in seconds (default: camera default)
        pixel_size: Pixel pitch in micrometers. Required with *file*; with a
            camera it overrides the value the camera reports, which is worth
            doing when binning is on or the camera reports nothing useful.
        serial_number: Open this specific device when more than one camera of
            the requested type is attached. Ignored by the simulated camera.

    Attributes:
        width_x: Beam width in x, in the selected definition (μm)
        width_y: Beam width in y, in the selected definition (μm)
        center_x: Beam center x position (pixels)
        center_y: Beam center y position (pixels)
        angle_deg: Beam rotation angle (degrees; 2D fit only, else 0)
        peak_value: Peak intensity of the last analyzed frame
    """

    def __init__(
        self,
        camera: str | None = None,
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
            RuntimeError: If a physical camera (FLIR/Basler) fails to open.
        """
        self.camera: Camera | None = None
        self.fit_method: str = fit
        self.definition: str = definition

        self.width_x: float = 0.0
        self.width_y: float = 0.0
        self.center_x: float = 0.0
        self.center_y: float = 0.0
        self.angle_deg: float = 0.0
        self.peak_value: float = 0.0

        self._last_popt_x: np.ndarray | list[Any] | None = None
        self._last_popt_y: np.ndarray | list[Any] | None = None
        self._last_popt_2d: np.ndarray | list[Any] | None = None
        self._stream_task: asyncio.Task[None] | None = None

        self.last_img: np.ndarray | None = None
        self._last_proj_x: np.ndarray | None = None
        self._last_proj_y: np.ndarray | None = None

        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"pixel_size must be greater than zero, got {pixel_size}")

        # Kept so :meth:`attach_camera` knows whether the scale was the
        # caller's choice (honour it) or the previous camera's (re-derive it).
        self._pixel_size_override: float | None = pixel_size

        if file:
            self._load_file(file)
            self._mode = "static"
            if pixel_size is None:
                raise ValueError("Pixel size must be provided for static beam image files")
            self.pixel_size = pixel_size
        elif camera:
            self._initialize_camera(camera, serial_number)
        else:
            self.camera = SimulatedCamera()
            self.camera.open()
            self._mode = "camera"

        if self.camera:
            self.width_pixels = self.camera.width
            self.height_pixels = self.camera.height
            # An explicit pixel_size wins over whatever the camera reports.
            self.pixel_size = pixel_size if pixel_size is not None else self.camera.pixel_size
            if exposure_time is not None:
                self.camera.set_exposure(exposure_time)
        elif file and self.last_img is not None:
            pass
        else:
            raise ValueError("Either camera or file must be provided and successfully loaded")

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
        """Context manager exit — ensures camera is closed."""
        if self.camera:
            try:
                self.camera.close()
            except Exception as e:
                logger.warning(f"Error closing camera: {e}")
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

    def _fit_2d_gaussian(self, image: np.ndarray) -> np.ndarray | list[Any]:
        """Fit a rotated 2D Gaussian, warm-starting from the previous frame.

        Only converged parameters are cached as the next warm start — a failed
        fit returns its initial guess but leaves ``_last_popt_2d`` alone, so one
        bad frame can't poison every frame after it.

        Args:
            image: 2D intensity array.

        Returns:
            ``[amplitude, x0, y0, sigma_x, sigma_y, theta, offset]``.
        """
        popt, converged = fitting.fit_2d_gaussian(
            image, self._last_popt_2d, max_dim=self._MAX_FIT_2D_DIM
        )
        if converged:
            self._last_popt_2d = popt
        return popt

    def beam_ellipse(self) -> tuple[float, float, float, float, float] | None:
        """Return the fitted 1/e² beam ellipse in pixel coordinates.

        Returns ``(cx, cy, rx, ry, angle_rad)`` where *rx* / *ry* are the 1/e²
        semi-axes (2σ).  In ``2d`` mode these come from the rotated 2D fit —
        using the 1D projection widths there would draw a badly wrong ellipse,
        since projecting a tilted beam onto the axes smears both widths toward
        each other.  Otherwise the two independent axis fits are used and the
        angle is zero.

        Returns:
            The ellipse parameters, or ``None`` if nothing has been fitted yet.
        """
        if self.fit_method == "2d" and self._last_popt_2d is not None:
            _, x0, y0, sigma_x, sigma_y, theta, _ = self._last_popt_2d
            return (
                float(x0),
                float(y0),
                2.0 * abs(float(sigma_x)),
                2.0 * abs(float(sigma_y)),
                float(theta),
            )

        popt_x, popt_y = self._last_popt_x, self._last_popt_y
        if popt_x is None or popt_y is None:
            return None
        return (
            float(popt_x[1]),
            float(popt_y[1]),
            2.0 * abs(float(popt_x[2])),
            2.0 * abs(float(popt_y[2])),
            0.0,
        )

    def _fit_projections(
        self, prof_x: np.ndarray, prof_y: np.ndarray
    ) -> tuple[np.ndarray | list[Any], np.ndarray | list[Any]]:
        """Fit both axis profiles, warm-starting from the previous frame."""
        popt_x = self._fit_1d_gaussian(prof_x, self._last_popt_x)
        popt_y = self._fit_1d_gaussian(prof_y, self._last_popt_y)
        self._last_popt_x, self._last_popt_y = popt_x, popt_y
        return popt_x, popt_y

    def _integrate(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sum the image down each axis and cache the result for plotting."""
        self._last_proj_x = np.sum(image, axis=0)
        self._last_proj_y = np.sum(image, axis=1)
        return self._last_proj_x, self._last_proj_y

    def analyze(self, image: np.ndarray) -> tuple[np.ndarray | list[Any], np.ndarray | list[Any]]:
        """Measure the beam in *image* and update every reported parameter.

        Two things decide what happens here, and they are independent:

        * ``definition`` picks how the width is *measured*.  ``fwhm`` and
          ``d4s`` are read straight off the integrated profile with no model,
          so they also override ``fit_method`` — a shape-free measurement and
          a Gaussian fit would disagree, and the definition wins.
        * ``fit_method`` picks what the Gaussian fit is run against:
          ``1d`` the integrated profiles, ``2d`` the whole frame (the only
          mode that recovers a rotation angle), ``linecut`` a single row and
          column through the brightest pixel.

        Either way the axis fits are returned, because the GUI draws them
        alongside the data even when they didn't set the reported width.

        Args:
            image: 2D intensity array.

        Returns:
            ``(x_fit_params, y_fit_params)`` for the two axis profiles.

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

        self.peak_value = float(np.max(image))
        self._last_proj_x = None
        self._last_proj_y = None
        self.angle_deg = 0.0

        # ── Model-free definitions: measure first, fit only for the plot ──
        if self.definition in ("fwhm", "d4s"):
            proj_x, proj_y = self._integrate(image)

            if self.definition == "fwhm":
                center_x, width_x, _ = self._measure_fwhm(proj_x)
                center_y, width_y, _ = self._measure_fwhm(proj_y)
            else:
                center_x, width_x = self._measure_d4s(proj_x)
                center_y, width_y = self._measure_d4s(proj_y)

            self.center_x, self.center_y = center_x, center_y
            self.width_x = width_x * self.pixel_size
            self.width_y = width_y * self.pixel_size
            return self._fit_projections(proj_x, proj_y)

        # ── Gaussian definition: the fit sets the width ──
        if self.fit_method == "linecut":
            peak_y, peak_x = np.unravel_index(int(np.argmax(image)), image.shape)
            # Remembered so the GUI can draw the crosshair it measured along.
            self._linecut_x = peak_x
            self._linecut_y = peak_y

            popt_x, popt_y = self._fit_projections(image[peak_y, :], image[:, peak_x])
            self._update_widths(abs(popt_x[2]), abs(popt_y[2]))
            self.center_x, self.center_y = popt_x[1], popt_y[1]
            return popt_x, popt_y

        if self.fit_method == "2d":
            _, x0, y0, sigma_x, sigma_y, theta, _ = self._fit_2d_gaussian(image)
            self._update_widths(abs(sigma_x), abs(sigma_y))
            self.center_x, self.center_y = x0, y0
            # The fit can't tell a beam from the same beam turned 180°, so
            # wrap into [0, 180) for a stable readout.
            self.angle_deg = np.degrees(theta) % 180
            # The 2D fit owns the width; these are purely for the profile plots.
            return self._fit_projections(*self._integrate(image))

        popt_x, popt_y = self._fit_projections(*self._integrate(image))
        self._update_widths(abs(popt_x[2]), abs(popt_y[2]))
        self.center_x, self.center_y = popt_x[1], popt_y[1]
        return popt_x, popt_y

    def _update_widths(self, sigma_x: float, sigma_y: float) -> None:
        """Record widths from Gaussian sigmas, in the 1/e² (4σ) convention.

        Only the Gaussian-definition paths call this; the model-free paths
        assign ``width_x`` / ``width_y`` from their own measurement.

        Args:
            sigma_x: Gaussian sigma in x (pixels).
            sigma_y: Gaussian sigma in y (pixels).
        """
        self.width_x = D4SIGMA_FACTOR * sigma_x * self.pixel_size
        self.width_y = D4SIGMA_FACTOR * sigma_y * self.pixel_size

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

        Clears the warm-start parameters, the cached projections and the last
        frame, so the next :meth:`analyze` starts from a cold estimate.
        """
        self._last_popt_x = None
        self._last_popt_y = None
        self._last_popt_2d = None
        self._last_proj_x = None
        self._last_proj_y = None
        self.last_img = None
        self.width_x = 0.0
        self.width_y = 0.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.angle_deg = 0.0
        self.peak_value = 0.0
        for attr in ("_linecut_x", "_linecut_y"):
            if hasattr(self, attr):
                delattr(self, attr)

    def stop(self) -> None:
        """Stop any active continuous streams and stop camera acquisition."""
        if hasattr(self, "_stream_task") and self._stream_task is not None:
            self._stream_task.cancel()
            self._stream_task = None

        if self._mode == "camera" and self.camera is not None and self.camera.is_acquiring:
            self.camera.stop_acquisition()

    def plot(
        self,
        num_img: int | None = None,
        heatmap_only: bool = False,
    ) -> asyncio.Task[None] | None:
        """Display beam profile with Gaussian fitting visualization.

        Args:
            num_img: Number of images (1 for single shot, None for continuous streaming)
            heatmap_only: Show only heatmap for faster rendering

        Returns:
            In single-shot or static mode, returns `None`.
            In streaming mode within a Jupyter environment, returns the background `asyncio.Task`
            powering the live visualization. This allows calling `.cancel()` on the task to stop
            the loop programmatically. Outside of Jupyter (terminal Dash/matplotlib), returns `None`
            as it inherently binds to the main process until interrupted (e.g. Ctrl-C).
        """

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
            and hasattr(self, "_linecut_x")
            and hasattr(self, "_linecut_y")
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
            f"<span style='font-size:14px'>Width: X={self.width_x:.1f}μm, Y={self.width_y:.1f}μm | "
        )
        title += f"Center: ({center_x_um:.1f}, {center_y_um:.1f})μm</span><br>"
        title += f"<span style='font-size:12px'>Peak={self.peak_value:.0f}"
        if self.fit_method == "2d":
            title += f" | Angle={self.angle_deg:.1f}°"
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
            and hasattr(self, "_linecut_x")
            and hasattr(self, "_linecut_y")
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
            f"<span style='font-size:14px'>Width: X={self.width_x:.1f}μm, Y={self.width_y:.1f}μm | "
        )
        title += f"Center: ({center_x_um:.1f}, {center_y_um:.1f})μm | "
        title += f"Peak: {self.peak_value:.0f}"
        if self.fit_method == "2d":
            title += f" | Angle: {self.angle_deg:.1f}°"
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
        """Capture and plot single image."""
        if self._mode == "camera":
            if self.camera is None:
                raise RuntimeError("Camera is not initialized")
            self.camera.start_acquisition()
            img = self.camera.get_image()
            self.camera.stop_acquisition()
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

        In a Jupyter environment, returns a background ``asyncio.Task`` driving
        the live loop.  Outside Jupyter, falls back to Dash or matplotlib and
        returns ``None`` (blocks until interrupted).
        """
        if self._mode == "camera":
            if self.camera is None:
                raise RuntimeError("Camera is not initialized")
            if not self.camera.is_acquiring:
                self.camera.start_acquisition()

        heatmap_only = getattr(self, "_heatmap_only", False)

        try:
            from IPython import get_ipython
            from IPython.display import clear_output, display

            if get_ipython() is None:
                raise ImportError("Not in IPython")

            if heatmap_only:
                logger.info("Starting live stream (heatmap only)...")
            else:
                logger.info("Starting live stream...")
            logger.info("Call profiler.stop() or cancel the returned task to stop\n")

            async def jupyter_stream_loop():
                frame_count = 0
                start_time = time.time()
                try:
                    while True:
                        try:
                            # Yield to the kernel event loop so interrupts and
                            # other callbacks can fire promptly.
                            await asyncio.sleep(0)

                            if self._mode == "camera" and self.camera is not None:
                                # Camera fetch can block on the producer; run
                                # it in a thread so the event loop stays free.
                                img = await asyncio.to_thread(self.camera.get_image)
                            else:
                                img = self.last_img
                            if img is None:
                                if (
                                    self._mode == "camera"
                                    and self.camera is not None
                                    and not self.camera.is_acquiring
                                ):
                                    break
                                await asyncio.sleep(0.01)
                                continue

                            popt_x, popt_y = await asyncio.to_thread(self.analyze, img)

                            if heatmap_only:
                                fig = await asyncio.to_thread(
                                    self._create_fast_figure, img, popt_x, popt_y
                                )
                            else:
                                fig = await asyncio.to_thread(
                                    self._create_figure, img, popt_x, popt_y
                                )

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

                        except Exception as e:
                            # Keep the stream alive across transient frame
                            # errors (timeouts, malformed frames, etc.).
                            logger.debug(f"Frame error in stream loop: {e}")
                            await asyncio.sleep(0.01)
                            continue

                except asyncio.CancelledError:
                    pass
                except KeyboardInterrupt:
                    pass
                finally:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"\nStream stopped: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps)"
                    )

            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(jupyter_stream_loop())
                self._stream_task = task
                return task
            except RuntimeError:
                asyncio.run(jupyter_stream_loop())

        except (NameError, ImportError):
            try:
                import dash  # noqa: F401
            except ImportError:
                logger.info("\nDash not available. Using matplotlib fallback.")
                logger.info("Install dash for better performance: pip install dash\n")

                try:
                    import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
                    from matplotlib.animation import FuncAnimation  # ty: ignore[unresolved-import]
                    from matplotlib.patches import Ellipse  # ty: ignore[unresolved-import]

                    fig_plt, axes = plt.subplots(2, 2, figsize=(10, 8))
                    fig_plt.tight_layout(pad=3.0)

                    def update_frame(frame_num):
                        img = (
                            self.camera.get_image()
                            if self._mode == "camera" and self.camera is not None
                            else self.last_img
                        )
                        if img is None:
                            return

                        popt_x, popt_y = self.analyze(img)

                        # Clear all axes
                        for ax in axes.flat:
                            ax.clear()

                        # Beam image
                        axes[1, 0].imshow(img, cmap="viridis")
                        axes[1, 0].set_title("Beam Image")
                        axes[1, 0].set_xlabel("X (pixels)")
                        axes[1, 0].set_ylabel("Y (pixels)")

                        # Add ellipse overlay
                        if popt_x is not None and popt_y is not None:
                            cx, cy = popt_x[1], popt_y[1]
                            width_px, height_px = 4 * abs(popt_x[2]), 4 * abs(popt_y[2])
                            ellipse = Ellipse(
                                (cx, cy),
                                width_px,
                                height_px,
                                fill=False,
                                edgecolor="red",
                                linewidth=2,
                                linestyle="--",
                            )
                            axes[1, 0].add_patch(ellipse)

                        # X profile
                        x = np.arange(img.shape[1])
                        proj_x = np.sum(img, axis=0)
                        axes[0, 0].plot(x, proj_x, "o", markersize=2, label="Data")
                        if popt_x is not None:
                            fitted_x = BeamProfiler.gaussian(x, *popt_x)
                            axes[0, 0].plot(x, fitted_x, "r-", label="Fit")
                        axes[0, 0].set_title("X Profile")
                        axes[0, 0].set_xlabel("X (pixels)")
                        axes[0, 0].legend()

                        # Y profile
                        y = np.arange(img.shape[0])
                        proj_y = np.sum(img, axis=1)
                        axes[1, 1].plot(proj_y, y, "o", markersize=2, label="Data")
                        if popt_y is not None:
                            fitted_y = BeamProfiler.gaussian(y, *popt_y)
                            axes[1, 1].plot(fitted_y, y, "r-", label="Fit")
                        axes[1, 1].set_title("Y Profile")
                        axes[1, 1].set_ylabel("Y (pixels)")
                        axes[1, 1].invert_xaxis()
                        axes[1, 1].legend()

                        # Info panel
                        axes[0, 1].axis("off")
                        info_text = f"Frame: {frame_num}\n\n"
                        info_text += f"Width X: {self.width_x:.1f} μm\n"
                        info_text += f"Width Y: {self.width_y:.1f} μm\n"
                        info_text += f"Center: ({self.center_x:.1f}, {self.center_y:.1f})\n"
                        if self.fit_method == "2d":
                            info_text += f"Angle: {self.angle_deg:.1f}°\n"
                        info_text += f"Peak: {self.peak_value:.0f}"
                        axes[0, 1].text(
                            0.1,
                            0.5,
                            info_text,
                            fontsize=12,
                            verticalalignment="center",
                            family="monospace",
                        )

                    print("\nStarting matplotlib animation. Press Ctrl+C to stop.\n", flush=True)
                    _anim = FuncAnimation(
                        fig_plt, update_frame, interval=50, cache_frame_data=False
                    )
                    plt.show()

                except ImportError:
                    logger.error("ERROR: Neither dash nor matplotlib is installed.")
                    logger.error("   Install one of them:")
                    logger.error("   - pip install dash (recommended for streaming)")
                    logger.error("   - pip install matplotlib")
                    return

                return

            from .dash_app import create_app

            app = create_app(self)

            url = f"http://127.0.0.1:{DEFAULT_DASH_PORT}"
            print(f"\npyBeamprofiler running at {url}")
            print("Press Ctrl+C to stop.\n", flush=True)
            logger.info(f"Starting Dash server at {url}")
            logger.info("Opening browser automatically...")

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
                    webbrowser.open(f"http://127.0.0.1:{DEFAULT_DASH_PORT}")

                threading.Thread(target=open_browser, daemon=True).start()

            def _sigint_handler(signum: int, frame: Any) -> None:
                if self._mode == "camera" and self.camera is not None:
                    try:
                        if self.camera.is_acquiring:
                            self.camera.stop_acquisition()
                        self.camera.close()
                    except Exception:
                        pass
                raise KeyboardInterrupt

            # Signal handlers can only be installed from the main thread; when
            # plot() is driven from a worker thread we just skip the tidy
            # shutdown rather than failing outright.
            prev_handler: Any = None
            try:
                prev_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, _sigint_handler)
            except ValueError:
                logger.debug("Not on the main thread; skipping SIGINT handler")
                prev_handler = None

            try:
                app.run(debug=False, port=DEFAULT_DASH_PORT, use_reloader=False)
            except KeyboardInterrupt:
                logger.info("\nStopping Dash server...")
            finally:
                if prev_handler is not None:
                    signal.signal(signal.SIGINT, prev_handler)


def main() -> None:
    """CLI entry point for pyBeamprofiler."""
    parser = argparse.ArgumentParser(
        description="pyBeamprofiler - Laser beam profiler with Gaussian fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Simulated camera with continuous streaming
        pybeamprofiler

        # FLIR camera, single shot
        pybeamprofiler --camera flir --num-img 1

        # Static image file (pixel size is required — it isn't in the file)
        pybeamprofiler --file beam.png --pixel-size 5.86

        # Basler camera with 2D fitting and FWHM definition
        pybeamprofiler --camera basler --fit 2d --definition fwhm

        # Fast mode (heatmap only)
        pybeamprofiler --heatmap-only
        """,
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="simulated",
        choices=["simulated", "flir", "basler"],
        help="Camera type (default: simulated)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to static image file (overrides --camera)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help=(
            "Sensor pixel pitch in micrometers. Required with --file, since an "
            "image on disk carries no scale. Optional with a camera, where it "
            "overrides the pitch the camera reports."
        ),
    )
    parser.add_argument(
        "--fit",
        type=str,
        default="1d",
        choices=["1d", "2d", "linecut"],
        help="Fitting method: 1d (fastest), 2d (with rotation), linecut (default: 1d)",
    )
    parser.add_argument(
        "--definition",
        type=str,
        default="gaussian",
        choices=["gaussian", "fwhm", "d4s"],
        help="Width definition: gaussian (1/e²), fwhm, d4s (default: gaussian)",
    )
    parser.add_argument(
        "--exposure-time",
        type=float,
        default=None,
        help="Camera exposure time in seconds (set during initialization)",
    )

    parser.add_argument(
        "--num-img",
        type=int,
        default=None,
        help="Number of images: 1 for single shot, None for continuous (default: continuous)",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Show only heatmap for faster display (~8-12 Hz in Jupyter)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.file and args.pixel_size is None:
        parser.error("--pixel-size is required with --file (e.g. --pixel-size 5.86)")
    if args.pixel_size is not None and args.pixel_size <= 0:
        parser.error("--pixel-size must be greater than zero")

    logger.info("Initializing pyBeamprofiler...")
    logger.info(f"   Camera: {args.file if args.file else args.camera}")
    logger.info(f"   Fitting: {args.fit} ({args.definition})")

    bp = BeamProfiler(
        camera=None if args.file else args.camera,
        file=args.file,
        fit=args.fit,
        definition=args.definition,
        exposure_time=args.exposure_time,
        pixel_size=args.pixel_size,
    )

    logger.info(f"   Sensor: {bp.width_pixels}×{bp.height_pixels} pixels")
    logger.info(f"   Pixel size: {bp.pixel_size:.2f} μm")

    if args.num_img == 1:
        logger.info("Single shot acquisition...")
    else:
        logger.info("Starting continuous streaming...")

    try:
        bp.plot(num_img=args.num_img, heatmap_only=args.heatmap_only)
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
    except Exception:
        logger.error("Fatal error during plotting", exc_info=True)
    finally:
        if hasattr(bp, "camera") and bp.camera:
            try:
                if bp.camera.is_acquiring:
                    bp.camera.stop_acquisition()
                bp.camera.close()
                logger.info("Camera closed")
            except Exception:
                logger.debug("Camera cleanup error (already released)", exc_info=True)


if __name__ == "__main__":
    main()
