"""GenICam camera wrapper using Harvesters library."""

import logging
import os

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
        width: Sensor width in pixels
        height: Sensor height in pixels
        pixel_size: Pixel pitch in micrometers
        exposure_time: Current exposure time in seconds
        gain: Current gain value
    """

    def __init__(self, cti_file: str | None = None, serial_number: str | None = None):
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
                logger.info(f"Using CTI file: {cti_file}")
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
                                    cti_path = os.path.join(path, file)
                                    self.h.add_file(cti_path)
                                    logger.info(f"Added CTI: {cti_path}")
                        elif path.endswith(".cti"):
                            self.h.add_file(path)
                            logger.info(f"Added CTI: {path}")
            else:
                logger.info("GENICAM_GENTL64_PATH not set, attempting manual discovery")

        self.serial_number = serial_number
        self.ia = None  # ImageAcquirer
        self.node_map = None
        self._exposure_min = 0.0
        self._exposure_max = 1.0
        self._gain_min = 0.0
        self._gain_max = 24.0
        self._roi_max_width = 0
        self._roi_max_height = 0
        self._roi_offset_x = 0
        self._roi_offset_y = 0
        # Initialize width/height for Camera base class compatibility
        self.width = 0
        self.height = 0

    def open(self) -> None:
        """Open camera connection and retrieve camera properties."""
        # Log which CTI files are loaded
        logger.info(f"Harvester loaded {len(self.h.files)} CTI file(s)")
        for cti in self.h.files:
            logger.info(f"  CTI: {cti}")

        self.h.update()

        if len(self.h.device_info_list) == 0:
            raise RuntimeError(
                f"No GenICam cameras found using {len(self.h.files)} CTI file(s). "
                "Ensure camera is connected and the correct GenTL producer (.cti) is loaded. "
                f"Loaded CTI files: {self.h.files}"
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

        # Configure camera settings first (may affect dimensions)
        self._configure_camera_settings()

        # Get sensor dimensions after configuration
        try:
            self.width_pixels = self.node_map.Width.value
            self.height_pixels = self.node_map.Height.value
            self.width = self.width_pixels  # For Camera base class compatibility
            self.height = self.height_pixels
            logger.info(f"Sensor: {self.width_pixels}×{self.height_pixels} pixels")
        except Exception as e:
            logger.warning(f"Could not get camera dimensions: {e}")
            self.width_pixels = 1024  # Default fallback
            self.height_pixels = 1024
            self.width = 1024
            self.height = 1024

        # Get pixel size (micrometers)
        self._detect_pixel_size()

        # Get exposure and gain ranges
        self._detect_exposure_range()
        self._detect_gain_range()

        # Get ROI information
        self._detect_roi_range()

        logger.info(f"Camera opened successfully: {device_to_open.model}")

    def _detect_pixel_size(self) -> None:
        """Detect pixel size from camera's GenICam features.

        Tries multiple standard feature names, sensor model lookup, and defaults to 1.0 μm.
        """
        try:
            # Try standard GenICam feature names in order of preference
            pixel_size = None

            # FLIR/EMVA standard naming (most reliable)
            try:
                if hasattr(self.node_map, "SensorPixelWidth"):
                    pixel_size = self.node_map.SensorPixelWidth.value
                    logger.debug("Using SensorPixelWidth for pixel size")
            except (AttributeError, ValueError, TypeError):
                pass

            if pixel_size is None:
                try:
                    if hasattr(self.node_map, "SensorPixelHeight"):
                        pixel_size = self.node_map.SensorPixelHeight.value
                        logger.debug("Using SensorPixelHeight for pixel size")
                except (AttributeError, ValueError, TypeError):
                    pass

            # Some cameras may have PixelSize (but verify it's numeric, not a string)
            if pixel_size is None:
                try:
                    if hasattr(self.node_map, "PixelSize"):
                        val = self.node_map.PixelSize.value
                        # Only use if it's a number (not a string like "Bpp8")
                        if isinstance(val, (int, float)):
                            pixel_size = val
                            logger.debug("Using PixelSize for pixel size")
                except (AttributeError, ValueError, TypeError):
                    pass

            # Try to detect from sensor model (common Sony sensors)
            if pixel_size is None:
                pixel_size = self._lookup_sensor_pixel_size()

            if pixel_size is not None:
                self.pixel_size = float(pixel_size)
                logger.info(f"Pixel size: {self.pixel_size:.2f} μm")
            else:
                self.pixel_size = 1.0
                logger.warning("Pixel size not available from camera, using default 1.0 μm")

        except Exception as e:
            logger.warning(f"Could not detect pixel size: {e}")
            self.pixel_size = 1.0

    def _lookup_sensor_pixel_size(self) -> float | None:
        """Look up pixel size from known sensor models.

        Returns:
            Pixel size in micrometers, or None if sensor not recognized
        """
        # Known sensor pixel sizes (in micrometers)
        SENSOR_DATABASE = {
            # Sony sensors (used in FLIR and Basler cameras)
            "IMX174": 5.86,  # Sony IMX174
            "IMX183": 2.4,  # Sony IMX183
            "IMX226": 1.85,  # Sony IMX226
            "IMX249": 5.86,  # Sony IMX249
            "IMX250": 3.45,  # Sony IMX250
            "IMX252": 3.45,  # Sony IMX252
            "IMX253": 1.85,  # Sony IMX253 (Basler ace 4024-8gm)
            "IMX255": 3.45,  # Sony IMX255
            "IMX264": 3.45,  # Sony IMX264
            "IMX265": 3.45,  # Sony IMX265
            "IMX273": 3.45,  # Sony IMX273 (FLIR BFS-PGE-16S2M)
            "IMX287": 6.9,  # Sony IMX287
            "IMX290": 2.9,  # Sony IMX290
            "IMX291": 2.9,  # Sony IMX291
            "IMX304": 3.45,  # Sony IMX304
            "IMX392": 2.9,  # Sony IMX392
            "IMX412": 1.55,  # Sony IMX412
            "IMX477": 1.55,  # Sony IMX477
            "IMX485": 2.9,  # Sony IMX485
            "IMX530": 2.74,  # Sony IMX530
            "IMX531": 2.74,  # Sony IMX531
            "IMX540": 2.5,  # Sony IMX540
            "IMX541": 2.5,  # Sony IMX541
            "IMX542": 2.5,  # Sony IMX542
            "IMX547": 2.74,  # Sony IMX547
            # Basler camera models (direct lookup)
            "acA4024-8gm": 1.85,  # (Sony IMX253)
            "acA4024-29um": 1.85,  # (Sony IMX253)
            "acA1920-155um": 2.74,
            "acA2440-75um": 3.45,
            "acA3800-14um": 1.85,
        }

        try:
            # Try to get sensor description
            if hasattr(self.node_map, "SensorDescription"):
                sensor_desc = str(self.node_map.SensorDescription.value)
                logger.debug(f"Sensor description: {sensor_desc}")

                # Search for sensor model in description
                for model, pixel_size in SENSOR_DATABASE.items():
                    if model in sensor_desc:
                        logger.info(f"Detected sensor {model}, using pixel size {pixel_size} μm")
                        return pixel_size

            # Try DeviceModelName as fallback
            if hasattr(self.node_map, "DeviceModelName"):
                model_name = str(self.node_map.DeviceModelName.value)
                logger.debug(f"Device model: {model_name}")

                for model, pixel_size in SENSOR_DATABASE.items():
                    if model in model_name:
                        logger.info(f"Detected sensor {model}, using pixel size {pixel_size} μm")
                        return pixel_size

        except Exception as e:
            logger.debug(f"Could not lookup sensor pixel size: {e}")

        return None

    def _configure_camera_settings(self) -> None:
        """Configure camera settings for manual control.

        Disables auto-exposure, auto-gain, and gamma correction for consistent imaging.
        Sets ROI to full sensor by default.
        """
        try:
            # Disable auto-exposure
            if hasattr(self.node_map, "ExposureAuto"):
                try:
                    self.node_map.ExposureAuto.value = "Off"
                    logger.info("ExposureAuto: Off")
                except Exception as e:
                    logger.debug(f"Could not set ExposureAuto: {e}")

            # Disable auto-gain
            if hasattr(self.node_map, "GainAuto"):
                try:
                    self.node_map.GainAuto.value = "Off"
                    logger.info("GainAuto: Off")
                except Exception as e:
                    logger.debug(f"Could not set GainAuto: {e}")

            # Disable gamma correction
            if hasattr(self.node_map, "GammaEnable"):
                try:
                    self.node_map.GammaEnable.value = False
                    logger.info("GammaEnable: False")
                except Exception as e:
                    logger.debug(f"Could not set GammaEnable: {e}")

            # Set ROI to full sensor (reset any previous ROI)
            self._reset_roi_to_full_sensor()

        except Exception as e:
            logger.warning(f"Error configuring camera settings: {e}")

    def _reset_roi_to_full_sensor(self) -> None:
        """Reset Region of Interest to full sensor size."""
        try:
            # Get maximum dimensions
            if hasattr(self.node_map, "WidthMax") and hasattr(self.node_map, "HeightMax"):
                width_max = self.node_map.WidthMax.value
                height_max = self.node_map.HeightMax.value

                # Set offsets to 0
                if hasattr(self.node_map, "OffsetX"):
                    self.node_map.OffsetX.value = 0
                if hasattr(self.node_map, "OffsetY"):
                    self.node_map.OffsetY.value = 0

                # Set width and height to maximum
                if hasattr(self.node_map, "Width"):
                    self.node_map.Width.value = width_max
                if hasattr(self.node_map, "Height"):
                    self.node_map.Height.value = height_max

                logger.info(f"ROI set to full sensor: {width_max}×{height_max}")
        except Exception as e:
            logger.debug(f"Could not reset ROI: {e}")

    def _detect_roi_range(self) -> None:
        """Detect ROI (Region of Interest) capabilities."""
        try:
            if hasattr(self.node_map, "WidthMax") and hasattr(self.node_map, "HeightMax"):
                width_max = self.node_map.WidthMax.value
                height_max = self.node_map.HeightMax.value
                logger.info(f"ROI max: {width_max}×{height_max}")

                # Store ROI limits for UI controls
                self._roi_max_width = width_max
                self._roi_max_height = height_max
                self._roi_offset_x = 0
                self._roi_offset_y = 0
        except Exception as e:
            logger.debug(f"Could not detect ROI range: {e}")

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
            except (AttributeError, ValueError, TypeError):
                try:
                    self.node_map.ExposureTimeAbs.value = exposure_time * 1_000_000
                except (AttributeError, ValueError, TypeError):
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
            except (AttributeError, ValueError, TypeError):
                try:
                    self.node_map.GainRaw.value = int(gain)
                except (AttributeError, ValueError, TypeError):
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

    def set_roi(
        self,
        offset_x: int = 0,
        offset_y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Set Region of Interest (ROI).

        Args:
            offset_x: X offset in pixels (default: 0)
            offset_y: Y offset in pixels (default: 0)
            width: ROI width in pixels (default: full width)
            height: ROI height in pixels (default: full height)
        """
        if not self.node_map:
            logger.warning("Camera not opened, cannot set ROI")
            return

        try:
            # Use full sensor if not specified
            if width is None:
                width = self._roi_max_width
            if height is None:
                height = self._roi_max_height

            # Set ROI (order matters: offset -> width/height)
            if hasattr(self.node_map, "OffsetX"):
                self.node_map.OffsetX.value = offset_x
            if hasattr(self.node_map, "OffsetY"):
                self.node_map.OffsetY.value = offset_y
            if hasattr(self.node_map, "Width"):
                self.node_map.Width.value = width
            if hasattr(self.node_map, "Height"):
                self.node_map.Height.value = height

            self.width = width  # Update base class attribute
            self.height = height
            self._roi_offset_x = offset_x
            self._roi_offset_y = offset_y
            self.width_pixels = width
            self.height_pixels = height

            logger.info(f"ROI set: offset=({offset_x}, {offset_y}), size={width}×{height}")
        except Exception as e:
            logger.error(f"Could not set ROI: {e}")

    @property
    def roi_info(self) -> dict:
        """Get current ROI information.

        Returns:
            Dictionary with ROI parameters: offset_x, offset_y, width, height, max_width, max_height
        """
        return {
            "offset_x": self._roi_offset_x,
            "offset_y": self._roi_offset_y,
            "width": self.width_pixels,
            "height": self.height_pixels,
            "max_width": self._roi_max_width,
            "max_height": self._roi_max_height,
        }
