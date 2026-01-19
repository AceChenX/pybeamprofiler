"""GenICam camera wrapper using Harvesters library."""

import logging
import os
from typing import Optional

import numpy as np

try:
    from harvesters.core import Harvester
except ImportError:
    Harvester = None

from .camera import Camera

logger = logging.getLogger(__name__)


class HarvesterCamera(Camera):
    """GenICam camera interface using Harvesters library.

    Provides unified interface for FLIR, Basler, and other GenICam-compliant
    cameras via standard GenTL producers (.cti files).

    Args:
        cti_file: Path to GenTL producer (.cti file). If None, uses
                  GENICAM_GENTL64_PATH environment variable.
        serial_number: Camera serial number to select specific device

    Attributes:
        width_pixels: Sensor width in pixels
        height_pixels: Sensor height in pixels
        pixel_size: Pixel pitch in micrometers
        exposure_time: Current exposure time in seconds
        gain: Current gain value
    """

    def __init__(self, cti_file: Optional[str] = None, serial_number: Optional[str] = None):
        """Initialize Harvester camera."""
        super().__init__()
        if Harvester is None:
            raise ImportError(
                "harvesters package is not installed. Install with: pip install harvesters"
            )
        self.h = Harvester()

        if cti_file:
            if not os.path.exists(cti_file):
                logger.warning(f"CTI file not found: {cti_file}")
            else:
                self.h.add_file(cti_file)
        else:
            gentl_path = os.environ.get("GENICAM_GENTL64_PATH", "")
            if gentl_path:
                logger.info(f"Using GENICAM_GENTL64_PATH: {gentl_path}")
                separator = ";" if os.name == "nt" else ":"
                for path in gentl_path.split(separator):
                    path = path.strip()
                    if path and os.path.exists(path):
                        if os.path.isdir(path):
                            for file in os.listdir(path):
                                if file.endswith(".cti"):
                                    self.h.add_file(os.path.join(path, file))
                        elif path.endswith(".cti"):
                            self.h.add_file(path)
            else:
                logger.info("GENICAM_GENTL64_PATH not set, attempting manual discovery")

        self.serial_number = serial_number
        self.ia = None  # ImageAcquirer
        self.node_map = None
        self._exposure_min = 0.0
        self._exposure_max = 1.0
        self._gain_min = 0.0
        self._gain_max = 24.0

    def open(self) -> None:
        """Open camera connection and retrieve camera properties."""
        self.h.update()

        if len(self.h.device_info_list) == 0:
            raise RuntimeError(
                "No GenICam cameras found. "
                "Ensure camera is connected and GenTL producer (.cti) is installed."
            )

        logger.info(f"Found {len(self.h.device_info_list)} camera(s):")
        for i, device in enumerate(self.h.device_info_list):
            logger.info(f"  [{i}] {device.vendor} {device.model} (S/N: {device.serial_number})")

        # Select camera
        device_to_open = None
        if self.serial_number:
            for device in self.h.device_info_list:
                if self.serial_number in device.serial_number:
                    device_to_open = device
                    break
            if not device_to_open:
                raise RuntimeError(f"Camera with serial number '{self.serial_number}' not found.")
        else:
            device_to_open = self.h.device_info_list[0]
            logger.info(f"Using first camera: {device_to_open.model}")

        self.ia = self.h.create(device_to_open)
        self.node_map = self.ia.remote_device.node_map

        # Get sensor dimensions
        try:
            self.width_pixels = self.node_map.Width.value
            self.height_pixels = self.node_map.Height.value
            logger.info(f"Sensor: {self.width_pixels}×{self.height_pixels} pixels")
        except Exception as e:
            logger.warning(f"Could not get camera dimensions: {e}")
            self.width_pixels = 1024  # Default fallback
            self.height_pixels = 1024

        # Get pixel size (micrometers)
        self._detect_pixel_size()

        # Get exposure and gain ranges
        self._detect_exposure_range()
        self._detect_gain_range()

        logger.info(f"Camera opened successfully: {device_to_open.model}")

    def _detect_pixel_size(self) -> None:
        """Detect pixel size from camera's GenICam features.

        Tries multiple standard feature names and defaults to 1.0 μm if unavailable.
        """
        try:
            # Try standard GenICam feature names
            if hasattr(self.node_map, "PixelSize"):
                # Pixel size in micrometers (some cameras)
                self.pixel_size = self.node_map.PixelSize.value
            elif hasattr(self.node_map, "SensorPixelWidth"):
                # FLIR/EMVA naming
                self.pixel_size = self.node_map.SensorPixelWidth.value
            elif hasattr(self.node_map, "SensorPixelHeight"):
                self.pixel_size = self.node_map.SensorPixelHeight.value
            else:
                self.pixel_size = 1.0
                logger.warning("Pixel size not available from camera, using default 1.0 μm")

            logger.info(f"Pixel size: {self.pixel_size:.2f} μm")
        except Exception as e:
            logger.warning(f"Could not detect pixel size: {e}")
            self.pixel_size = 1.0

    def _detect_exposure_range(self) -> None:
        """Detect exposure time range from camera.

        Tries ExposureTime and ExposureTimeAbs features, converts from microseconds.
        """
        try:
            if hasattr(self.node_map, "ExposureTime"):
                node = self.node_map.ExposureTime
                self._exposure_min = node.min / 1_000_000  # Convert μs to seconds
                self._exposure_max = node.max / 1_000_000
            elif hasattr(self.node_map, "ExposureTimeAbs"):
                node = self.node_map.ExposureTimeAbs
                self._exposure_min = node.min / 1_000_000
                self._exposure_max = node.max / 1_000_000
            logger.info(
                f"Exposure range: {self._exposure_min * 1000:.3f} - "
                f"{self._exposure_max * 1000:.3f} ms"
            )
        except Exception as e:
            logger.warning(f"Could not detect exposure range: {e}")

    def _detect_gain_range(self) -> None:
        """Detect gain range from camera.

        Tries Gain and GainRaw features.
        """
        try:
            if hasattr(self.node_map, "Gain"):
                node = self.node_map.Gain
                self._gain_min = node.min
                self._gain_max = node.max
            elif hasattr(self.node_map, "GainRaw"):
                node = self.node_map.GainRaw
                self._gain_min = float(node.min)
                self._gain_max = float(node.max)
            logger.info(f"Gain range: {self._gain_min:.1f} - {self._gain_max:.1f}")
        except Exception as e:
            logger.warning(f"Could not detect gain range: {e}")

    def close(self) -> None:
        """Close camera connection."""
        if self.ia:
            self.ia.destroy()
        self.h.reset()

    def start_acquisition(self) -> None:
        """Start image acquisition."""
        if self.ia:
            self.ia.start()
            self.is_acquiring = True

    def stop_acquisition(self) -> None:
        """Stop image acquisition."""
        if self.ia:
            self.ia.stop()
            self.is_acquiring = False

    def get_image(self) -> np.ndarray:
        """Retrieve image from camera.

        Returns:
            2D numpy array of image data
        """
        if not self.ia:
            raise RuntimeError("Camera not opened.")

        with self.ia.fetch(timeout=3.0) as buffer:
            component = buffer.payload.components[0]

            if component.data_format == "Mono8":
                image = component.data.reshape(component.height, component.width).copy()
            else:
                image = component.data.reshape(component.height, component.width).copy()

            self.width_pixels = component.width
            self.height_pixels = component.height
            return image

    def set_exposure(self, exposure_time: float) -> None:
        """Set exposure time.

        Args:
            exposure_time: Exposure time in seconds
        """
        if self.node_map:
            try:
                self.node_map.ExposureTime.value = exposure_time * 1_000_000
            except Exception:
                try:
                    self.node_map.ExposureTimeAbs.value = exposure_time * 1_000_000
                except Exception:
                    logger.error("Could not set exposure time.")
        self.exposure_time = exposure_time

    def set_gain(self, gain: float) -> None:
        """Set camera gain.

        Args:
            gain: Gain value
        """
        if self.node_map:
            try:
                self.node_map.Gain.value = gain
            except Exception:
                try:
                    self.node_map.GainRaw.value = int(gain)
                except Exception:
                    logger.error("Could not set gain.")
        self.gain = gain

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
