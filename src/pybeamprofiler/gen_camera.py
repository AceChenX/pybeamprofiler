"""GenICam camera wrapper using Harvesters library."""

from __future__ import annotations

import logging
import os
import platform
import time
from typing import Any

import numpy as np

try:
    from harvesters.core import Harvester
    from harvesters.core import TimeoutException as _HarvestersTimeout
except ImportError:
    Harvester = None  # ty:ignore[invalid-assignment]
    _HarvestersTimeout = None  # ty:ignore[invalid-assignment]

from .camera import Camera
from .cti import parse_gentl_path

logger = logging.getLogger(__name__)

# Known sensor pixel sizes in micrometers, used for auto-detection
SENSOR_PIXEL_SIZES: dict[str, float] = {
    # Sony sensors (used in FLIR and Basler cameras)
    "IMX174": 5.86,
    "IMX183": 2.4,
    "IMX226": 1.85,
    "IMX249": 5.86,
    "IMX250": 3.45,
    "IMX252": 3.45,
    "IMX253": 1.85,
    "IMX255": 3.45,
    "IMX264": 3.45,
    "IMX265": 3.45,
    "IMX273": 3.45,
    "IMX287": 6.9,
    "IMX290": 2.9,
    "IMX291": 2.9,
    "IMX304": 3.45,
    "IMX392": 2.9,
    "IMX412": 1.55,
    "IMX477": 1.55,
    "IMX485": 2.9,
    "IMX530": 2.74,
    "IMX531": 2.74,
    "IMX540": 2.5,
    "IMX541": 2.5,
    "IMX542": 2.5,
    "IMX547": 2.74,
    # Basler camera models (direct lookup)
    "acA4024-8gm": 1.85,
    "acA4024-29um": 1.85,
    "acA1920-155um": 2.74,
    "acA2440-75um": 3.45,
    "acA3800-14um": 1.85,
}


class HarvesterCamera(Camera):
    """GenICam camera interface using Harvesters library.

    Provides a unified interface for FLIR, Basler, and other GenICam-compliant
    cameras via standard GenTL producers (``.cti`` files).

    Attributes:
        node_map: GenICam node map for direct feature access, or ``None``.
        device_model: Camera model name (e.g. ``"BFS-PGE-50S5M"``).
        device_vendor: Camera vendor name (e.g. ``"FLIR"``).
        serial_number: Camera serial number string.
        width_pixels: Sensor width in pixels.
        height_pixels: Sensor height in pixels.
    """

    def __init__(
        self,
        cti_file: str | list[str] | None = None,
        serial_number: str | None = None,
    ) -> None:
        """Initialize Harvester camera.

        Args:
            cti_file: Path(s) to GenTL producer (``.cti``) file(s).
                If ``None``, the caller (e.g. :class:`BaslerCamera`) is expected
                to resolve the path via ``GENICAM_GENTL64_PATH`` or platform search.
            serial_number: Camera serial number for device selection.
        """
        super().__init__()
        if Harvester is None:
            raise ImportError(
                "harvesters/genicam is not available. On macOS, install the camera SDK "
                "(Pylon or Spinnaker) and ensure its genicam Python bindings are on the path. "
                "On Linux/Windows: pip install harvesters"
            )
        self.h = Harvester()

        if cti_file:
            files = [cti_file] if isinstance(cti_file, str) else cti_file
            for file_path in files:
                if not os.path.exists(file_path):
                    logger.warning(f"CTI file not found: {file_path}")
                else:
                    self.h.add_file(file_path)
                    logger.info(f"Using CTI file: {file_path}")
        else:
            logger.warning(
                "No CTI file specified. Please provide cti_file parameter or set GENICAM_GENTL64_PATH."
            )

        self.serial_number: str | None = serial_number
        self.device_model: str | None = None
        self.device_vendor: str | None = None
        self.ia: Any = None
        self.node_map: Any = None
        self._exposure_min: float = 1e-6  # safe default (avoids log10(0) in UI)
        self._exposure_max: float = 1.0
        self._gain_min: float = 0.0
        self._gain_max: float = 24.0
        self._roi_max_width: int = 0
        self._roi_max_height: int = 0
        self._roi_offset_x: int = 0
        self._roi_offset_y: int = 0
        self.width: int = 0
        self.height: int = 0
        # Time of the most recent successful frame. Used to detect when the
        # producer has silently stalled (a known issue on some GenTL stacks
        # after long-running acquisition) so ``get_image`` can attempt a
        # stop/start recovery instead of timing out forever.
        self._last_successful_fetch: float = 0.0
        self._stall_recovery_attempted: bool = False

    @staticmethod
    def _parse_gentl_path(gentl_path: str) -> str | list[str] | None:
        """Parse a ``GENICAM_GENTL64_PATH`` value into CTI path(s).

        Delegates the actual expansion to
        :func:`pybeamprofiler.cti.parse_gentl_path` and collapses a single
        result to a bare string, which is what :meth:`__init__` and the
        vendor subclasses expect.

        Args:
            gentl_path: Value of the ``GENICAM_GENTL64_PATH`` environment
                variable.

        Returns:
            One path, several paths, or ``None`` if nothing resolved.
        """
        found = parse_gentl_path(gentl_path)
        if not found:
            return None
        return found if len(found) > 1 else found[0]

    def open(self) -> None:
        """Open camera connection and retrieve camera properties."""
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

        self.device_model = getattr(device_to_open, "model", None)
        self.device_vendor = getattr(device_to_open, "vendor", None)
        self.serial_number = getattr(device_to_open, "serial_number", self.serial_number)

        self.ia = self.h.create(device_to_open)
        self.node_map = self.ia.remote_device.node_map

        self._configure_gige_stream()
        self._configure_camera_settings()

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

        self._detect_pixel_size()
        self._detect_exposure_range()
        self._detect_gain_range()
        self._detect_roi_range()

        logger.info(f"Camera opened successfully: {device_to_open.model}")

    def _detect_pixel_size(self) -> None:
        """Detect pixel size from camera's GenICam features.

        Tries multiple standard feature names, sensor model lookup, and defaults to 1.0 μm.
        """
        try:
            pixel_size = None

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

            if pixel_size is None:
                try:
                    if hasattr(self.node_map, "PixelSize"):
                        val = self.node_map.PixelSize.value
                        if isinstance(val, (int, float)):
                            pixel_size = val
                            logger.debug("Using PixelSize for pixel size")
                except (AttributeError, ValueError, TypeError):
                    pass

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
        try:
            if hasattr(self.node_map, "SensorDescription"):
                sensor_desc = str(self.node_map.SensorDescription.value)
                logger.debug(f"Sensor description: {sensor_desc}")

                for model, pixel_size in SENSOR_PIXEL_SIZES.items():
                    if model in sensor_desc:
                        logger.info(f"Detected sensor {model}, using pixel size {pixel_size} μm")
                        return pixel_size

            if hasattr(self.node_map, "DeviceModelName"):
                model_name = str(self.node_map.DeviceModelName.value)
                logger.debug(f"Device model: {model_name}")

                for model, pixel_size in SENSOR_PIXEL_SIZES.items():
                    if model in model_name:
                        logger.info(f"Detected sensor {model}, using pixel size {pixel_size} μm")
                        return pixel_size

        except Exception as e:
            logger.debug(f"Could not lookup sensor pixel size: {e}")

        return None

    def _configure_gige_stream(self) -> None:
        """Switch GigE Vision streams to SocketDriver on macOS.

        Pylon's default GigEAccelerator transport requires a proprietary kernel
        extension that is unavailable on macOS, resulting in zero received packets.
        The SocketDriver transport uses standard OS UDP sockets and works reliably.
        """
        if platform.system() != "Darwin" or not self.ia.data_streams:
            return
        try:
            ds_nm = self.ia.data_streams[0].node_map
            if getattr(ds_nm, "Type", None) is None:
                return
            current = ds_nm.Type.value
            if current != "SocketDriver" and ds_nm.TypeIsSocketDriverAvailable.value:
                ds_nm.Type.value = "SocketDriver"
                logger.info(f"GigE stream transport: {current} -> SocketDriver")
        except Exception as e:
            logger.debug(f"Could not configure GigE stream transport: {e}")

    def _configure_camera_settings(self) -> None:
        """Configure camera settings for manual control.

        Disables auto-exposure, auto-gain, and gamma correction for consistent imaging.
        Sets ROI to full sensor by default.
        """
        try:
            if hasattr(self.node_map, "ExposureAuto"):
                try:
                    self.node_map.ExposureAuto.value = "Off"
                    logger.info("ExposureAuto: Off")
                except Exception as e:
                    logger.debug(f"Could not set ExposureAuto: {e}")

            if hasattr(self.node_map, "GainAuto"):
                try:
                    self.node_map.GainAuto.value = "Off"
                    logger.info("GainAuto: Off")
                except Exception as e:
                    logger.debug(f"Could not set GainAuto: {e}")

            if hasattr(self.node_map, "GammaEnable"):
                try:
                    self.node_map.GammaEnable.value = False
                    logger.info("GammaEnable: False")
                except Exception as e:
                    logger.debug(f"Could not set GammaEnable: {e}")

            self._reset_roi_to_full_sensor()

        except Exception as e:
            logger.warning(f"Error configuring camera settings: {e}")

    def _reset_roi_to_full_sensor(self) -> None:
        """Reset Region of Interest to full sensor size."""
        try:
            if hasattr(self.node_map, "WidthMax") and hasattr(self.node_map, "HeightMax"):
                width_max = self.node_map.WidthMax.value
                height_max = self.node_map.HeightMax.value

                if hasattr(self.node_map, "OffsetX"):
                    self.node_map.OffsetX.value = 0
                if hasattr(self.node_map, "OffsetY"):
                    self.node_map.OffsetY.value = 0

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
        """Close camera connection and release hardware."""
        try:
            self.stop_acquisition()
        except Exception:
            logger.debug("Error stopping acquisition during close", exc_info=True)
        try:
            if self.ia:
                self.ia.destroy()
        except Exception:
            logger.debug("Error destroying ImageAcquirer", exc_info=True)
        try:
            self.h.reset()
        except Exception:
            logger.debug("Error resetting Harvester", exc_info=True)

    def start_acquisition(self) -> None:
        """Start image acquisition on the GenTL producer."""
        if not self.ia:
            return
        if not self.is_acquiring:
            self.ia.start()
            self.is_acquiring = True
            self._last_successful_fetch = 0.0
            self._stall_recovery_attempted = False

    def stop_acquisition(self) -> None:
        """Stop image acquisition on the GenTL producer."""
        if self.ia and self.is_acquiring:
            try:
                self.ia.stop()
            except Exception:
                logger.debug("Error stopping acquisition", exc_info=True)
            self.is_acquiring = False

    def get_image(self, timeout: float | None = None) -> np.ndarray:
        """Fetch the next frame from the GenTL producer.

        For default short exposures this returns within a few ms. For long
        exposures the caller should pass a small ``timeout`` (e.g. ``0.2``)
        and treat :class:`TimeoutError` as "no new frame yet, try again
        next tick" so the UI stays responsive.

        Args:
            timeout: Maximum seconds to wait for a frame.
                Defaults to ``max(2.0, exposure_time + 2.0)``.

        Returns:
            2D numpy array containing the frame data.

        Raises:
            RuntimeError: If the camera has not been opened.
            TimeoutError: If no frame arrives within ``timeout``.
        """
        if not self.ia:
            raise RuntimeError("Camera not opened.")
        if not self.is_acquiring:
            self.start_acquisition()

        if timeout is None:
            timeout = max(2.0, (self.exposure_time or 0) + 2.0)

        # One-shot stop/start recovery: if the producer has been silent for
        # roughly ``max(5 s, 3× exposure)`` we assume the acquirer has
        # stalled (a known issue on some GenTL stacks after long runs).
        now = time.monotonic()
        if self._last_successful_fetch and not self._stall_recovery_attempted:
            stall_window = max(5.0, 3.0 * (self.exposure_time or 0.0))
            if now - self._last_successful_fetch > stall_window:
                logger.warning(
                    "Acquisition appears stalled (no frame for %.1f s); "
                    "attempting to recover by restarting acquisition.",
                    now - self._last_successful_fetch,
                )
                self._stall_recovery_attempted = True
                try:
                    self.stop_acquisition()
                    self.start_acquisition()
                except Exception:
                    logger.debug("Stall recovery failed", exc_info=True)

        try:
            with self.ia.fetch(timeout=timeout) as buffer:
                component = buffer.payload.components[0]
                self.width_pixels = component.width
                self.height_pixels = component.height
                img = component.data.reshape(component.height, component.width).copy()
            self._last_successful_fetch = time.monotonic()
            self._stall_recovery_attempted = False
            return img
        except Exception as exc:
            # Harvesters re-exports ``_gentl.TimeoutException`` which isn't a
            # subclass of Python's built-in ``TimeoutError`` (they merely share
            # a name), so we catch it explicitly and re-raise as ``TimeoutError``
            # so callers can use a single, standard-library-only except clause.
            is_timeout = isinstance(exc, TimeoutError) or (
                _HarvestersTimeout is not None and isinstance(exc, _HarvestersTimeout)
            )
            if is_timeout:
                # Seed the stall timer on the very first call so we don't
                # falsely trigger recovery for a camera that simply hasn't
                # warmed up yet.
                if not self._last_successful_fetch:
                    self._last_successful_fetch = now
                raise TimeoutError(
                    f"Camera did not deliver a frame within {timeout:.1f} s. "
                    "Check that the camera is connected, powered, and not in "
                    "use by another application."
                ) from exc
            raise

    def set_exposure(self, exposure_time: float) -> None:
        """Set exposure time, restarting acquisition to flush stale buffers.

        Without the stop/start the producer's buffer ring still holds frames
        captured at the old exposure, so the display would show a couple of
        wrongly-exposed frames after every change.

        Args:
            exposure_time: Exposure time in seconds.
        """
        was_acquiring = self.is_acquiring
        if was_acquiring:
            self.stop_acquisition()

        if self.node_map:
            try:
                self.node_map.ExposureTime.value = exposure_time * 1_000_000
            except (AttributeError, ValueError, TypeError):
                try:
                    self.node_map.ExposureTimeAbs.value = exposure_time * 1_000_000
                except (AttributeError, ValueError, TypeError):
                    logger.error("Could not set exposure time.")
        self.exposure_time = exposure_time

        if was_acquiring:
            self.start_acquisition()

    def set_gain(self, gain: float) -> None:
        """Set camera gain, falling back to the legacy ``GainRaw`` feature.

        Args:
            gain: Gain in the camera's own units — dB on most SFNC-compliant
                devices, raw ADC steps on older ones.
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
        """Supported exposure time as ``(min, max)`` in seconds."""
        return (self._exposure_min, self._exposure_max)

    @property
    def gain_range(self) -> tuple[float, float]:
        """Supported gain as ``(min, max)`` in the camera's own units."""
        return (self._gain_min, self._gain_max)

    def set_roi(
        self,
        offset_x: int = 0,
        offset_y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Set the Region of Interest, clamping to what the sensor allows.

        Out-of-range values are clamped rather than rejected, so a too-large
        width simply yields the biggest ROI that fits at the given offset.
        Cameras usually also quantise these to a granularity of their own
        (often 4 or 8 px), so read :attr:`roi_info` back for the real values.

        Args:
            offset_x: X offset in pixels.
            offset_y: Y offset in pixels.
            width: ROI width in pixels (``None`` for full width).
            height: ROI height in pixels (``None`` for full height).
        """
        if not self.node_map:
            logger.warning("Camera not opened, cannot set ROI")
            return

        try:
            if width is None:
                width = self._roi_max_width
            if height is None:
                height = self._roi_max_height

            offset_x = max(0, min(offset_x, self._roi_max_width - 1))
            offset_y = max(0, min(offset_y, self._roi_max_height - 1))
            width = max(1, min(width, self._roi_max_width - offset_x))
            height = max(1, min(height, self._roi_max_height - offset_y))

            # Order matters: set offsets before dimensions
            if hasattr(self.node_map, "OffsetX"):
                self.node_map.OffsetX.value = offset_x
            if hasattr(self.node_map, "OffsetY"):
                self.node_map.OffsetY.value = offset_y
            if hasattr(self.node_map, "Width"):
                self.node_map.Width.value = width
            if hasattr(self.node_map, "Height"):
                self.node_map.Height.value = height

            self.width = width
            self.height = height
            self._roi_offset_x = offset_x
            self._roi_offset_y = offset_y
            self.width_pixels = width
            self.height_pixels = height

            logger.info(f"ROI set: offset=({offset_x}, {offset_y}), size={width}×{height}")
        except Exception as e:
            logger.error(f"Could not set ROI: {e}")

    @property
    def roi_info(self) -> dict[str, int]:
        """Get current ROI information.

        Returns:
            Dict with keys ``offset_x``, ``offset_y``, ``width``, ``height``,
            ``max_width``, ``max_height``.
        """
        return {
            "offset_x": self._roi_offset_x,
            "offset_y": self._roi_offset_y,
            "width": self.width_pixels,
            "height": self.height_pixels,
            "max_width": self._roi_max_width,
            "max_height": self._roi_max_height,
        }
