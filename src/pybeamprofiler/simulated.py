"""Simulated camera for testing and demonstration."""

import logging
import os
import time

import numpy as np

from .camera import Camera

logger = logging.getLogger(__name__)


class SimulatedCamera(Camera):
    """Simulated camera generating dynamic Gaussian beam patterns.

    Generates realistic beam images with random fluctuations for testing
    and demonstration purposes without requiring hardware.

    Attributes:
        width: Sensor width in pixels (1024)
        height: Sensor height in pixels (1024)
        pixel_size: Pixel pitch in micrometers (5.0)
    """

    def __init__(self):
        super().__init__()
        self.width = 1024
        self.height = 1024
        self.pixel_size = 5.0
        self.exposure_time = 0.01
        self.gain = 0.0
        self._center_x = self.width / 2
        self._center_y = self.height / 2
        self._sigma_x = 150
        self._sigma_y = 140  # Slightly elliptical
        self._amplitude = 250
        self._background = 10

        # Add exposure/gain ranges for compatibility
        self._exposure_min = 0.001  # 1 ms
        self._exposure_max = 1.0  # 1 s
        self._gain_min = 0.0
        self._gain_max = 24.0

    def open(self):
        logger.info("Simulated camera opened.")

    def close(self):
        logger.info("Simulated camera closed.")

    def start_acquisition(self):
        self.is_acquiring = True
        logger.info("Simulated acquisition started.")

    def stop_acquisition(self):
        self.is_acquiring = False
        logger.info("Simulated acquisition stopped.")

    def get_image(self) -> np.ndarray:
        """Generate simulated beam image with random fluctuations.

        Returns:
            2D numpy array of uint8 intensity values
        """
        if "PYTEST_CURRENT_TEST" not in os.environ:
            time.sleep(self.exposure_time if self.exposure_time < 0.1 else 0.1)
        cx = self._center_x + np.random.normal(0, 3)
        cy = self._center_y + np.random.normal(0, 3)
        sx = self._sigma_x + np.random.normal(0, 2)
        sy = self._sigma_y + np.random.normal(0, 2)
        # Ensure amplitude stays positive
        amp = max(1.0, self._amplitude + np.random.normal(0, 5))
        # Ensure background stays positive
        bg = max(0.0, self._background + np.random.normal(0, 1))
        noise = np.random.normal(0, 2, (self.height, self.width))

        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        xv, yv = np.meshgrid(x, y)

        gaussian = amp * np.exp(-((xv - cx) ** 2 / (2 * sx**2) + (yv - cy) ** 2 / (2 * sy**2)))

        image = gaussian + bg + noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        self.image_buffer = image
        return image

    def set_exposure(self, exposure_time: float):
        """Set exposure time and adjust simulated signal amplitude."""
        if exposure_time is None:
            exposure_time = 0.01
        self.exposure_time = exposure_time
        self._amplitude = 250 * (exposure_time / 0.01)

    def set_gain(self, gain: float):
        """Set gain and adjust simulated signal amplitude."""
        self.gain = gain
        self._amplitude = 250 * (1 + gain / 10)

    @property
    def exposure_range(self) -> tuple[float, float]:
        """Get exposure time range in seconds.

        Returns:
            Tuple of (min_exposure, max_exposure) in seconds
        """
        return (self._exposure_min, self._exposure_max)

    @property
    def gain_range(self) -> tuple[float, float]:
        """Get gain range.

        Returns:
            Tuple of (min_gain, max_gain)
        """
        return (self._gain_min, self._gain_max)
