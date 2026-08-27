"""Simulated cameras: Gaussian beams with no hardware attached.

Several *profiles* are defined so the multi-camera selector has something to
switch between on a developer machine. They differ in sensor size, pixel
pitch and beam shape rather than being clones, so that switching cameras is
visibly doing something and the fitting paths (including the rotated 2D fit)
all get exercised.

Profile one reproduces the historical defaults exactly, so
``SimulatedCamera()`` with no arguments behaves as it always has.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import Camera
from .constants import (
    DEFAULT_EXPOSURE_TIME,
    DEFAULT_GAIN,
    SIMULATED_AMPLITUDE,
    SIMULATED_BACKGROUND,
    SIMULATED_HEIGHT,
    SIMULATED_PIXEL_SIZE,
    SIMULATED_SIGMA_X,
    SIMULATED_SIGMA_Y,
    SIMULATED_WIDTH,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulatedProfile:
    """A distinct fake camera: sensor geometry plus the beam it produces.

    Attributes:
        key: Short stable id used in the GUI dropdown value.
        name: Human-readable model name.
        serial_number: Fake serial, so the selector can key on it like a real
            device.
        width: Sensor width in pixels.
        height: Sensor height in pixels.
        pixel_size: Pixel pitch in micrometers.
        sigma_x: Beam sigma along the first principal axis, in pixels.
        sigma_y: Beam sigma along the second principal axis, in pixels.
        theta_deg: Beam rotation. Non-zero costs a full 2D exponential per
            frame instead of the separable fast path, so it is opt-in.
        amplitude: Peak signal above the background.
        background: Baseline level.
    """

    key: str
    name: str
    serial_number: str
    width: int
    height: int
    pixel_size: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    background: float
    theta_deg: float = 0.0


#: The first entry reproduces the historical defaults exactly; the others
#: exist so the camera selector has more than one thing to offer.
SIMULATED_PROFILES: tuple[SimulatedProfile, ...] = (
    SimulatedProfile(
        key="sim-1",
        name="SimulatedCamera",
        serial_number="SIM-001",
        width=SIMULATED_WIDTH,
        height=SIMULATED_HEIGHT,
        pixel_size=SIMULATED_PIXEL_SIZE,
        sigma_x=SIMULATED_SIGMA_X,
        sigma_y=SIMULATED_SIGMA_Y,
        amplitude=SIMULATED_AMPLITUDE,
        background=SIMULATED_BACKGROUND,
    ),
    SimulatedProfile(
        key="sim-2",
        name="SimulatedCamera Tilted",
        serial_number="SIM-002",
        # A different sensor shape and pitch, so switching is obvious at a
        # glance and the aspect-ratio handling gets a non-square workout.
        width=1280,
        height=1024,
        pixel_size=3.45,
        sigma_x=90.0,
        sigma_y=30.0,
        amplitude=180.0,
        background=8.0,
        theta_deg=35.0,
    ),
)

DEFAULT_PROFILE = SIMULATED_PROFILES[0]


def profile_for(key: str | None) -> SimulatedProfile:
    """Return the profile named by *key*, defaulting to the first one."""
    for profile in SIMULATED_PROFILES:
        if profile.key == key:
            return profile
    return DEFAULT_PROFILE


class _SimulatedNode:
    """Lightweight stand-in for a single GenICam node.

    Provides ``.value``, ``.min``, ``.max`` and ``.symbolics`` so that
    the Dash Setting tab and ``camera.setting()`` can treat the simulated
    camera like a real GenICam device.
    """

    def __init__(
        self,
        value: Any = None,
        *,
        min_val: Any = None,
        max_val: Any = None,
        symbolics: list[str] | None = None,
        readonly: bool = False,
    ) -> None:
        self._value = value
        self.min = min_val
        self.max = max_val
        self.symbolics = symbolics
        self._readonly = readonly

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, v: Any) -> None:
        if self._readonly:
            raise AttributeError("Node is read-only")
        self._value = v


class _SimulatedNodeMap:
    """Minimal GenICam node-map mock for :class:`SimulatedCamera`.

    Only the nodes that are useful for the Dash GUI are exposed.  The
    camera's ``set_exposure`` / ``set_gain`` / ``set_roi`` methods remain
    the primary control path; this object lets the Setting tab *read*
    current values and *write* simple parameters through the same
    ``node_map.Attr.value`` interface used by real cameras.
    """

    def __init__(self, cam: SimulatedCamera) -> None:
        self._cam = cam
        self._init_nodes()

    def _init_nodes(self) -> None:
        cam = self._cam
        self.DeviceModelName = _SimulatedNode(cam.profile.name, readonly=True)
        self.DeviceVendorName = _SimulatedNode("pybeamprofiler", readonly=True)
        self.DeviceSerialNumber = _SimulatedNode(cam.profile.serial_number, readonly=True)

        self.ExposureTime = _SimulatedNode(
            cam.exposure_time * 1_000_000,
            min_val=cam._exposure_min * 1_000_000,
            max_val=cam._exposure_max * 1_000_000,
        )
        self.Gain = _SimulatedNode(
            cam.gain,
            min_val=cam._gain_min,
            max_val=cam._gain_max,
        )

        max_w, max_h = cam.profile.width, cam.profile.height
        self.Width = _SimulatedNode(cam.width, min_val=1, max_val=max_w)
        self.Height = _SimulatedNode(cam.height, min_val=1, max_val=max_h)
        self.OffsetX = _SimulatedNode(0, min_val=0, max_val=max_w - 1)
        self.OffsetY = _SimulatedNode(0, min_val=0, max_val=max_h - 1)
        self.WidthMax = _SimulatedNode(max_w, readonly=True)
        self.HeightMax = _SimulatedNode(max_h, readonly=True)

        self.SensorPixelWidth = _SimulatedNode(cam.pixel_size, readonly=True)
        self.SensorPixelHeight = _SimulatedNode(cam.pixel_size, readonly=True)

        self.GammaEnable = _SimulatedNode(False)
        self.Gamma = _SimulatedNode(1.0, min_val=0.25, max_val=4.0)
        self.BlackLevel = _SimulatedNode(0, min_val=0, max_val=255)
        self.ExposureAuto = _SimulatedNode("Off", symbolics=["Off", "Once", "Continuous"])
        self.GainAuto = _SimulatedNode("Off", symbolics=["Off", "Once", "Continuous"])
        self.AcquisitionFrameRate = _SimulatedNode(30.0, min_val=1.0, max_val=120.0)
        self.PixelFormat = _SimulatedNode("Mono8", symbolics=["Mono8", "Mono12", "Mono16"])
        self.TriggerMode = _SimulatedNode("Off", symbolics=["Off", "On"])
        self.TriggerSource = _SimulatedNode("Software", symbolics=["Software", "Line0", "Line1"])
        self.ReverseX = _SimulatedNode(False)
        self.ReverseY = _SimulatedNode(False)
        self.DeviceTemperature = _SimulatedNode(42.5, min_val=0.0, max_val=100.0, readonly=True)


class SimulatedCamera(Camera):
    """Simulated camera generating dynamic Gaussian beam patterns.

    Produces beam images with frame-to-frame jitter for testing and
    demonstration without hardware. Which sensor and beam it fakes comes from
    a :class:`SimulatedProfile`; the default reproduces the historical
    behaviour, so ``SimulatedCamera()`` is unchanged.

    Supports ROI via :meth:`set_roi` and exposes a minimal GenICam-style
    :attr:`node_map` so that the Dash Setting tab can display controls.

    Args:
        profile: Which fake camera to be. Defaults to :data:`DEFAULT_PROFILE`.
        seed: Seed for the frame-to-frame jitter. Leave as ``None`` for fresh
            randomness; set it when you need the same sequence of frames every
            run, which is what keeps tests that measure the simulated beam from
            failing once in a few hundred runs.
    """

    def __init__(self, profile: SimulatedProfile | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.profile = profile or DEFAULT_PROFILE
        self.width = self.profile.width
        self.height = self.profile.height
        self.width_pixels = self.profile.width
        self.height_pixels = self.profile.height
        self.pixel_size = self.profile.pixel_size
        self.exposure_time = DEFAULT_EXPOSURE_TIME
        self.gain = DEFAULT_GAIN
        # Presented like a real device so the selector and the Camera Info
        # panel have something to key on and display.
        self.device_model = self.profile.name
        self.device_vendor = "pybeamprofiler"
        self.serial_number = self.profile.serial_number

        self._center_x = self.profile.width / 2
        self._center_y = self.profile.height / 2
        self._sigma_x = self.profile.sigma_x
        self._sigma_y = self.profile.sigma_y
        self._theta = math.radians(self.profile.theta_deg)
        self._amplitude = self.profile.amplitude
        self._background = self.profile.background

        self._noise_center = 17.0
        # Width jitter is a *fraction* of each axis, not an absolute pixel
        # count. A fixed 7 px was tuned for the ~50 px axes of the default
        # profile; on a 30 px minor axis it is 23%, which swung the apparent
        # axis ratio anywhere from 1.6 to 8.6 and made a real beam look like
        # it was breathing. Real beams fluctuate proportionally.
        self._noise_sigma_frac = 0.14
        self._noise_amp = 25.0
        self._noise_bg = 10.0
        self._noise_image = 10.0

        self._exposure_min = 0.001
        self._exposure_max = 1.0
        self._gain_min = 0.0
        self._gain_max = 24.0

        # ROI state (full sensor by default)
        self._roi_offset_x = 0
        self._roi_offset_y = 0
        self._roi_width = self.width
        self._roi_height = self.height

        self.node_map: _SimulatedNodeMap | None = None

        # Precompute coordinate axes and a reusable scratch buffer so that
        # get_image() avoids reallocating a full frame every call. An
        # unrotated Gaussian is separable (G(x,y) = Gx(x)·Gy(y)), so the fast
        # path only needs 1D exp() over W+H elements instead of W·H.
        # NB: ``self.width`` / ``self.height`` track the *ROI* and shrink when
        # one is set. The frame is always rendered at full sensor size and then
        # cropped, so everything below is sized from the profile instead.
        self._sensor_w = self.profile.width
        self._sensor_h = self.profile.height
        self._x_axis = np.arange(self._sensor_w, dtype=np.float32)
        self._y_axis = np.arange(self._sensor_h, dtype=np.float32)
        self._frame_buf = np.empty((self._sensor_h, self._sensor_w), dtype=np.float32)
        # The rotated path needs 2D coordinate fields. Rotation is linear, so
        # the grids can be rotated once here and the (jittering) beam centre
        # subtracted as a scalar each frame -- rather than rebuilding a
        # rotated grid, or calling the generic gaussian_2d and paying for its
        # temporaries, every time. Built only for a profile that asks for tilt.
        self._rot: tuple[np.ndarray, np.ndarray] | None = None
        if self._theta:
            grid_y, grid_x = np.mgrid[0 : self._sensor_h, 0 : self._sensor_w].astype(np.float32)
            cos_t, sin_t = math.cos(self._theta), math.sin(self._theta)
            # Matches fitting.gaussian_2d: theta is the counter-clockwise
            # angle of the major (sigma_x) axis in array coordinates.
            self._rot = (
                grid_x * cos_t + grid_y * sin_t,
                -grid_x * sin_t + grid_y * cos_t,
            )
        # Scratch buffer for the rotated path, so the per-frame maths runs
        # in place instead of allocating a full frame per operation.
        self._scratch = (
            np.empty((self._sensor_h, self._sensor_w), dtype=np.float32) if self._theta else None
        )
        self._rng = np.random.default_rng(seed)

    def open(self) -> None:
        """Open the simulated camera and initialize the node map."""
        self.node_map = _SimulatedNodeMap(self)
        logger.info("Simulated camera opened.")

    def close(self) -> None:
        """Release simulated camera resources (no-op)."""
        logger.info("Simulated camera closed.")

    def start_acquisition(self) -> None:
        """Begin simulated image acquisition."""
        self.is_acquiring = True
        logger.info("Simulated acquisition started.")

    def stop_acquisition(self) -> None:
        """Stop simulated image acquisition."""
        self.is_acquiring = False
        logger.info("Simulated acquisition stopped.")

    def get_image(self, timeout: float | None = None) -> np.ndarray:
        """Generate simulated beam image with random fluctuations.

        If an ROI has been set, returns only the cropped region.

        Args:
            timeout: Unused; kept for :class:`~pybeamprofiler.camera.Camera`
                interface compatibility.

        Returns:
            2D numpy array of uint8 intensity values.
        """
        del timeout
        rng = self._rng
        cx = self._center_x + rng.normal(0, self._noise_center)
        cy = self._center_y + rng.normal(0, self._noise_center)
        sx = self._sigma_x * (1.0 + rng.normal(0, self._noise_sigma_frac))
        sy = self._sigma_y * (1.0 + rng.normal(0, self._noise_sigma_frac))
        # Keep the beam from inverting or collapsing on an extreme draw.
        sx = max(sx, 1.0)
        sy = max(sy, 1.0)
        amp = max(1.0, self._amplitude + rng.normal(0, self._noise_amp))
        bg = max(0.0, self._background + rng.normal(0, self._noise_bg))

        buf = self._frame_buf
        if self._rot is None:
            # Separable Gaussian: G(x,y) = amp * Gx(x) * Gy(y). Two 1D
            # exponentials over W and H elements are ~30x cheaper than one 2D
            # exponential over W*H, and avoid meshgrid allocations.
            gx = np.exp(-((self._x_axis - cx) ** 2) / (2 * sx * sx))
            gy = np.exp(-((self._y_axis - cy) ** 2) / (2 * sy * sy))
            np.outer(gy, gx, out=buf)
        else:
            # A tilted beam has a cross term and does not factorise, so one 2D
            # exponential is unavoidable. Everything around it, though, runs
            # in place on two preallocated buffers: rotate the centre into the
            # same frame as the pre-rotated grids and it is a subtract, a
            # square and an add per axis.
            rot_x, rot_y = self._rot
            scratch = self._scratch
            assert scratch is not None  # built together with _rot
            cos_t, sin_t = math.cos(self._theta), math.sin(self._theta)
            u_c = cx * cos_t + cy * sin_t
            v_c = -cx * sin_t + cy * cos_t

            np.subtract(rot_x, u_c, out=buf)
            np.multiply(buf, buf, out=buf)
            buf *= -0.5 / (sx * sx)

            np.subtract(rot_y, v_c, out=scratch)
            np.multiply(scratch, scratch, out=scratch)
            scratch *= -0.5 / (sy * sy)

            buf += scratch
            np.exp(buf, out=buf)

        buf *= amp
        buf += bg
        # standard_normal can emit float32 directly; rng.normal always builds
        # float64 and would then need a full-frame cast.
        noise = rng.standard_normal((self._sensor_h, self._sensor_w), dtype=np.float32)
        noise *= self._noise_image
        buf += noise

        ox, oy = self._roi_offset_x, self._roi_offset_y
        rw, rh = self._roi_width, self._roi_height
        cropped = buf[oy : oy + rh, ox : ox + rw]
        image = np.clip(cropped, 0, 255).astype(np.uint8)

        self.image_buffer = image
        return image

    def _refresh_amplitude(self) -> None:
        """Recompute peak signal from the current exposure *and* gain.

        Both scale the signal on a real sensor, so both have to be folded in
        together — deriving the amplitude from only the setting that changed
        last would quietly undo the other one.
        """
        self._amplitude = (
            SIMULATED_AMPLITUDE
            * (self.exposure_time / DEFAULT_EXPOSURE_TIME)
            * (1 + self.gain / 10)
        )

    def set_exposure(self, exposure_time: float | None) -> None:  # type: ignore[override]
        """Set exposure time and adjust simulated signal amplitude.

        Args:
            exposure_time: Exposure in seconds, or ``None`` to reset to the default.
        """
        if exposure_time is None:
            exposure_time = DEFAULT_EXPOSURE_TIME
        self.exposure_time = exposure_time
        self._refresh_amplitude()
        if self.node_map is not None:
            self.node_map.ExposureTime._value = exposure_time * 1_000_000

    def set_gain(self, gain: float) -> None:
        """Set gain and adjust simulated signal amplitude."""
        self.gain = gain
        self._refresh_amplitude()
        if self.node_map is not None:
            self.node_map.Gain._value = gain

    def set_roi(
        self,
        offset_x: int = 0,
        offset_y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Set the region of interest (ROI).

        Args:
            offset_x: Left edge offset in pixels.
            offset_y: Top edge offset in pixels.
            width: ROI width in pixels (``None`` for full sensor width).
            height: ROI height in pixels (``None`` for full sensor height).
        """
        max_w, max_h = self.profile.width, self.profile.height
        self._roi_offset_x = max(0, min(offset_x, max_w - 1))
        self._roi_offset_y = max(0, min(offset_y, max_h - 1))
        w = width if width is not None else max_w
        h = height if height is not None else max_h
        self._roi_width = max(1, min(w, max_w - self._roi_offset_x))
        self._roi_height = max(1, min(h, max_h - self._roi_offset_y))

        self.width = self._roi_width
        self.height = self._roi_height
        # Keep the ``*_pixels`` aliases in step — the Dash Camera Info panel
        # reads those first and would otherwise keep showing the full sensor.
        self.width_pixels = self._roi_width
        self.height_pixels = self._roi_height

        if self.node_map is not None:
            self.node_map.OffsetX._value = self._roi_offset_x
            self.node_map.OffsetY._value = self._roi_offset_y
            self.node_map.Width._value = self._roi_width
            self.node_map.Height._value = self._roi_height

        logger.info(
            f"ROI set: {self._roi_width}x{self._roi_height} "
            f"at ({self._roi_offset_x}, {self._roi_offset_y})"
        )

    @property
    def roi_info(self) -> dict[str, int]:
        """Current ROI parameters.

        Returns:
            Dictionary with offset_x, offset_y, width, height,
            max_width, max_height.
        """
        return {
            "offset_x": self._roi_offset_x,
            "offset_y": self._roi_offset_y,
            "width": self._roi_width,
            "height": self._roi_height,
            "max_width": self.profile.width,
            "max_height": self.profile.height,
        }

    @property
    def exposure_range(self) -> tuple[float, float]:
        """Exposure time range in seconds ``(min, max)``."""
        return (self._exposure_min, self._exposure_max)

    @property
    def gain_range(self) -> tuple[float, float]:
        """Gain range ``(min, max)``."""
        return (self._gain_min, self._gain_max)
