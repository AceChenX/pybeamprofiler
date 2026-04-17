"""Simulated camera for testing and demonstration."""

from __future__ import annotations

import logging
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
        self.DeviceModelName = _SimulatedNode("SimulatedCamera", readonly=True)
        self.DeviceVendorName = _SimulatedNode("pybeamprofiler", readonly=True)
        self.DeviceSerialNumber = _SimulatedNode("SIM-001", readonly=True)

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

        self.Width = _SimulatedNode(cam.width, min_val=1, max_val=SIMULATED_WIDTH)
        self.Height = _SimulatedNode(cam.height, min_val=1, max_val=SIMULATED_HEIGHT)
        self.OffsetX = _SimulatedNode(0, min_val=0, max_val=SIMULATED_WIDTH - 1)
        self.OffsetY = _SimulatedNode(0, min_val=0, max_val=SIMULATED_HEIGHT - 1)
        self.WidthMax = _SimulatedNode(SIMULATED_WIDTH, readonly=True)
        self.HeightMax = _SimulatedNode(SIMULATED_HEIGHT, readonly=True)

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

    Generates realistic beam images with random fluctuations for testing
    and demonstration purposes without requiring hardware.  Sensor
    dimensions and pixel size default to the values in
    :mod:`pybeamprofiler.constants`.

    Supports ROI via :meth:`set_roi` and exposes a minimal GenICam-style
    :attr:`node_map` so that the Dash Setting tab can display controls.
    """

    def __init__(self) -> None:
        super().__init__()
        self.width = SIMULATED_WIDTH
        self.height = SIMULATED_HEIGHT
        self.width_pixels = SIMULATED_WIDTH
        self.height_pixels = SIMULATED_HEIGHT
        self.pixel_size = SIMULATED_PIXEL_SIZE
        self.exposure_time = DEFAULT_EXPOSURE_TIME
        self.gain = DEFAULT_GAIN
        self._center_x = self.width / 2
        self._center_y = self.height / 2
        self._sigma_x = SIMULATED_SIGMA_X
        self._sigma_y = SIMULATED_SIGMA_Y
        self._amplitude = SIMULATED_AMPLITUDE
        self._background = SIMULATED_BACKGROUND

        self._noise_center = 17.0
        self._noise_sigma = 7.0
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
        self._roi_width = SIMULATED_WIDTH
        self._roi_height = SIMULATED_HEIGHT

        self.node_map: _SimulatedNodeMap | None = None

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
        cx = self._center_x + np.random.normal(0, self._noise_center)
        cy = self._center_y + np.random.normal(0, self._noise_center)
        sx = self._sigma_x + np.random.normal(0, self._noise_sigma)
        sy = self._sigma_y + np.random.normal(0, self._noise_sigma)
        amp = max(1.0, self._amplitude + np.random.normal(0, self._noise_amp))
        bg = max(0.0, self._background + np.random.normal(0, self._noise_bg))
        noise = np.random.normal(0, self._noise_image, (SIMULATED_HEIGHT, SIMULATED_WIDTH))

        x = np.arange(0, SIMULATED_WIDTH)
        y = np.arange(0, SIMULATED_HEIGHT)
        xv, yv = np.meshgrid(x, y)

        gaussian = amp * np.exp(-((xv - cx) ** 2 / (2 * sx**2) + (yv - cy) ** 2 / (2 * sy**2)))

        image = gaussian + bg + noise
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Apply ROI crop
        ox, oy = self._roi_offset_x, self._roi_offset_y
        rw, rh = self._roi_width, self._roi_height
        image = image[oy : oy + rh, ox : ox + rw]

        self.image_buffer = image
        return image

    def set_exposure(self, exposure_time: float | None) -> None:  # type: ignore[override]
        """Set exposure time and adjust simulated signal amplitude.

        Args:
            exposure_time: Exposure in seconds, or ``None`` to reset to the default.
        """
        if exposure_time is None:
            exposure_time = DEFAULT_EXPOSURE_TIME
        self.exposure_time = exposure_time
        self._amplitude = SIMULATED_AMPLITUDE * (exposure_time / DEFAULT_EXPOSURE_TIME)
        if self.node_map is not None:
            self.node_map.ExposureTime._value = exposure_time * 1_000_000

    def set_gain(self, gain: float) -> None:
        """Set gain and adjust simulated signal amplitude."""
        self.gain = gain
        self._amplitude = SIMULATED_AMPLITUDE * (1 + gain / 10)
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
        self._roi_offset_x = max(0, min(offset_x, SIMULATED_WIDTH - 1))
        self._roi_offset_y = max(0, min(offset_y, SIMULATED_HEIGHT - 1))
        w = width if width is not None else SIMULATED_WIDTH
        h = height if height is not None else SIMULATED_HEIGHT
        self._roi_width = max(1, min(w, SIMULATED_WIDTH - self._roi_offset_x))
        self._roi_height = max(1, min(h, SIMULATED_HEIGHT - self._roi_offset_y))

        self.width = self._roi_width
        self.height = self._roi_height

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
            "max_width": SIMULATED_WIDTH,
            "max_height": SIMULATED_HEIGHT,
        }

    @property
    def exposure_range(self) -> tuple[float, float]:
        """Exposure time range in seconds ``(min, max)``."""
        return (self._exposure_min, self._exposure_max)

    @property
    def gain_range(self) -> tuple[float, float]:
        """Gain range ``(min, max)``."""
        return (self._gain_min, self._gain_max)
