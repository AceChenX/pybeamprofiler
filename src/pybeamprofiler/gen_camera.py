"""GenICam cameras, driven through Harvesters.

:class:`HarvesterCamera` is the workhorse behind every real device: FLIR and
Basler differ only in where their GenTL producer lives, which
:mod:`pybeamprofiler.cti` already handles, so both vendor classes are thin
subclasses.

Two things here exist because of how the underlying C library behaves rather
than because the GenICam standard asks for them: acquisition is restarted
after an exposure change so the producer's buffer ring cannot deliver
stale-exposure frames, and a silent producer is given one stop/start
recovery attempt before the caller is left waiting forever.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from harvesters.core import Harvester
    from harvesters.core import TimeoutException as _HarvestersTimeout
except ImportError:
    Harvester = None  # ty:ignore[invalid-assignment]
    _HarvestersTimeout = None  # ty:ignore[invalid-assignment]

from .camera import Camera, _roi_pixels
from .cti import parse_gentl_path

logger = logging.getLogger(__name__)


def _genicam_errors() -> tuple[type[Exception], ...]:
    """The base classes of what the GenICam bindings raise, where installed.

    ``genicam.genapi`` (node reads and writes) and ``genicam.gentl``
    (transport) each define their own ``GenericException``. Neither derives
    from ``ValueError`` or ``AttributeError``: an out-of-range write raises
    ``OutOfRangeException``, which an ``except ValueError`` does not catch.
    """
    found: list[type[Exception]] = []
    for name in ("genicam.genapi", "genicam.gentl"):
        try:
            found.append(importlib.import_module(name).GenericException)
        except (ImportError, AttributeError):
            pass
    return tuple(found)


#: What reading or writing a node can raise: ``AttributeError`` for a feature
#: the camera does not have, ``TypeError``/``ValueError`` for a value of the
#: wrong kind, and the GenICam families for everything the device refuses.
_NODE_ERRORS: tuple[type[Exception], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    *_genicam_errors(),
)


def _numeric(value: Any) -> float | None:
    """``value`` as a float if it is a real number, else ``None``.

    Node attributes on a real camera are plain Python numbers. Anything else
    -- a feature that is not implemented, a test double -- is treated as
    unknown rather than coerced into a number that looks meaningful.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value)


def _node(node_map: Any, name: str) -> Any:
    """The node called ``name``, or ``None`` if the camera has no such feature."""
    if node_map is None:
        return None
    try:
        return getattr(node_map, name)
    except Exception:  # absent (AttributeError) or a node map that cannot answer
        return None


def _node_int(node: Any, attr: str = "value") -> int | None:
    """Read ``node.<attr>`` as an int, or ``None`` if it is absent or unreadable."""
    if node is None:
        return None
    try:
        number = _numeric(getattr(node, attr))
    except Exception:  # e.g. AccessException from a node that is not available
        return None
    return None if number is None else int(number)


def _node_error_text(exc: BaseException) -> str:
    """The human half of a GenICam error message.

    GenApi messages read "Value = 100 must be equal or smaller than Max = 0.
    : OutOfRangeException thrown in node 'OffsetX' while calling ... (file
    'IntegerT.h', line 81)"; the part after the separator is for C++
    developers.
    """
    return str(exc).split(" : ")[0].strip() or type(exc).__name__


@dataclass(frozen=True)
class _Axis:
    """What one ROI axis allows right now, read from the live node map.

    Attributes:
        sensor: Full extent, i.e. ``WidthMax``/``HeightMax``. SFNC defines
            those *after* binning and decimation, so this is re-read every
            time rather than remembered from ``open()``.
        size_min: Smallest ``Width``/``Height`` the camera accepts.
        size_inc: Step of ``Width``/``Height``.
        offset_min: Smallest offset (0 on every camera seen so far).
        offset_inc: Step of the offset.
        has_offset: Whether the camera can offset the ROI at all.
    """

    sensor: int
    size_min: int = 1
    size_inc: int = 1
    offset_min: int = 0
    offset_inc: int = 1
    has_offset: bool = True


def _align_down(value: int, minimum: int, step: int) -> int:
    """Round ``value`` down onto the grid ``minimum + k * step``."""
    return minimum + ((value - minimum) // step) * step


def _fit_axis(axis: _Axis, offset: int, size: int | None) -> tuple[int, int]:
    """Clamp and align one axis of a requested ROI to what the camera allows.

    GenApi rejects a value that is off the node's increment instead of
    rounding it, so both numbers are snapped down onto the grid -- down, so
    the ROI never grows past what was asked for or off the sensor edge.

    Args:
        axis: The axis limits.
        offset: Requested offset; clamped into the sensor.
        size: Requested size, or ``None`` for the full sensor.

    Returns:
        ``(offset, size)`` that the camera will accept.
    """
    full = axis.sensor
    size = full if size is None else size
    size = max(axis.size_min, min(size, full))
    if size > axis.size_min:
        size = _align_down(size, axis.size_min, axis.size_inc)
    if not axis.has_offset:
        return 0, size
    offset = max(axis.offset_min, min(offset, full - size))
    offset = _align_down(offset, axis.offset_min, axis.offset_inc)
    return offset, size


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


def _to_mono(component: Any) -> np.ndarray:
    """Reshape one payload component into a 2D intensity array.

    Mono formats arrive as exactly ``height * width`` samples and only need a
    reshape. Colour formats (RGB8, BGR8, YUV...) carry several samples per
    pixel; a beam profiler wants one intensity per pixel, so those are
    collapsed with the usual luminance weights for three channels and a plain
    mean otherwise. Blindly reshaping them — which is what this used to do —
    raises "cannot reshape array of size N" the moment a camera is left in a
    colour pixel format.

    Packed mono formats (Mono10p, Mono12p) do not have a whole number of
    samples per pixel and are rejected with a message that says so, rather
    than producing a silently corrupt image.

    Args:
        component: A Harvesters ``Component2DImage``.

    Returns:
        A fresh 2D array; the payload buffer is reused by the producer as soon
        as the ``fetch`` context exits, so the copy is not optional.

    Raises:
        ValueError: If the payload size is not a whole multiple of the frame.
    """
    height, width = component.height, component.width
    data = component.data
    pixels = height * width
    if pixels == 0:
        raise ValueError("Camera reported a zero-sized frame")

    if data.size == pixels:
        return data.reshape(height, width).copy()

    channels, remainder = divmod(data.size, pixels)
    if remainder or channels < 1:
        fmt = getattr(component, "data_format", "unknown")
        raise ValueError(
            f"Cannot interpret a {data.size}-sample payload as a "
            f"{width}x{height} frame (pixel format {fmt!r}). Packed formats "
            "such as Mono10p/Mono12p are not supported; select Mono8, Mono12 "
            "or Mono16 on the camera."
        )

    planes = data.reshape(height, width, channels)
    if channels == 3:
        return (planes.astype(np.float64) @ [0.299, 0.587, 0.114]).astype(data.dtype)
    return planes.mean(axis=2).astype(data.dtype)


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
        self.width_pixels: int = 0
        self.height_pixels: int = 0
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

        width = _node_int(_node(self.node_map, "Width"))
        height = _node_int(_node(self.node_map, "Height"))
        if width is None or height is None:
            logger.warning("Could not read the camera dimensions; assuming 1024×1024")
            width = height = 1024
        self.width = self.width_pixels = width
        self.height = self.height_pixels = height
        logger.info(f"Sensor: {width}×{height} pixels")

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
        """Start from the full sensor, whatever ROI the last user left behind.

        Only done when the camera states its full size (``WidthMax`` and
        ``HeightMax``); without them "full" is a guess, and an ROI that was
        deliberately configured elsewhere is better left alone.
        """
        try:
            if (
                _node(self.node_map, "WidthMax") is None
                or _node(self.node_map, "HeightMax") is None
            ):
                return
            x, y = self._roi_axes()
            _, width = _fit_axis(x, 0, None)
            _, height = _fit_axis(y, 0, None)
            self._write_roi(x.offset_min, y.offset_min, width, height, x, y)
            logger.info("ROI set to full sensor: %d×%d", width, height)
        except Exception as e:
            logger.debug(f"Could not reset ROI: {e}")

    def _detect_roi_range(self) -> None:
        """Read the ROI limits and the current geometry back from the camera."""
        try:
            self._refresh_roi_cache()
            logger.info(f"ROI max: {self._roi_max_width}×{self._roi_max_height}")
        except Exception as e:
            logger.debug(f"Could not detect ROI range: {e}")

    def _roi_axes(self) -> tuple[_Axis, _Axis]:
        """The live limits of both ROI axes."""
        return (
            self._roi_axis("Width", "OffsetX", "WidthMax", self._roi_max_width or self.width),
            self._roi_axis("Height", "OffsetY", "HeightMax", self._roi_max_height or self.height),
        )

    def _roi_axis(self, size_name: str, offset_name: str, max_name: str, fallback: int) -> _Axis:
        """Limits of one ROI axis, read from the node map on every call.

        Without a ``WidthMax`` node the full extent is recovered from
        ``Width.max``, which SFNC-style descriptions define as ``WidthMax -
        OffsetX``. ``fallback`` (the last extent seen) is used only when the
        camera answers neither.
        """
        node_map = self.node_map
        size_node = _node(node_map, size_name)
        offset_node = _node(node_map, offset_name)
        sensor = _node_int(_node(node_map, max_name))
        if sensor is None:
            size_max = _node_int(size_node, "max")
            if size_max is not None:
                sensor = size_max + (_node_int(offset_node) or 0)
        if sensor is None:
            sensor = max(1, int(fallback or 1))
        return _Axis(
            sensor=sensor,
            size_min=max(1, _node_int(size_node, "min") or 1),
            size_inc=max(1, _node_int(size_node, "inc") or 1),
            offset_min=max(0, _node_int(offset_node, "min") or 0),
            offset_inc=max(1, _node_int(offset_node, "inc") or 1),
            has_offset=_node_int(offset_node) is not None,
        )

    def _write_roi(self, ox: int, oy: int, width: int, height: int, x: _Axis, y: _Axis) -> None:
        """Write an ROI in an order the camera cannot refuse half-way.

        ``OffsetX.max`` is ``WidthMax - Width`` and ``Width.max`` is ``WidthMax
        - OffsetX``, so at full width the only legal offset is 0 and a wide
        ROI does not fit behind a large offset. Moving the offsets to their
        minimum first frees the whole sensor for the size; the final offsets
        then fit behind the new size by construction.

        Nodes already holding their target are not written. Besides saving
        register writes, that lets an offset-only change -- including putting
        back the previous ROI after a refusal -- go through on a camera whose
        size is locked.
        """
        node_map = self.node_map
        cur_ox, cur_oy, cur_w, cur_h = self._read_roi()
        steps: list[tuple[str, int]] = []
        if (width, height) != (cur_w, cur_h):
            if x.has_offset and cur_ox != x.offset_min:
                steps.append(("OffsetX", x.offset_min))
                cur_ox = x.offset_min
            if y.has_offset and cur_oy != y.offset_min:
                steps.append(("OffsetY", y.offset_min))
                cur_oy = y.offset_min
            if width != cur_w:
                steps.append(("Width", width))
            if height != cur_h:
                steps.append(("Height", height))
        if x.has_offset and ox != cur_ox:
            steps.append(("OffsetX", ox))
        if y.has_offset and oy != cur_oy:
            steps.append(("OffsetY", oy))
        for name, value in steps:
            node = _node(node_map, name)
            if node is not None:
                node.value = int(value)

    def _read_roi(self) -> tuple[int, int, int, int]:
        """``(offset_x, offset_y, width, height)`` as the camera reports it now."""
        node_map = self.node_map
        ox = _node_int(_node(node_map, "OffsetX"))
        oy = _node_int(_node(node_map, "OffsetY"))
        width = _node_int(_node(node_map, "Width"))
        height = _node_int(_node(node_map, "Height"))
        return (
            self._roi_offset_x if ox is None else ox,
            self._roi_offset_y if oy is None else oy,
            self.width_pixels if width is None else width,
            self.height_pixels if height is None else height,
        )

    def _refresh_roi_cache(self) -> None:
        """Copy the camera's current ROI and limits into the cached attributes.

        The cache is what :attr:`roi_info` falls back to once the camera is
        closed, and what ``width``/``height`` report between frames.
        """
        x, y = self._roi_axes()
        self._roi_max_width, self._roi_max_height = x.sensor, y.sensor
        ox, oy, width, height = self._read_roi()
        self._roi_offset_x, self._roi_offset_y = ox, oy
        self.width = self.width_pixels = width
        self.height = self.height_pixels = height

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
                img = _to_mono(component)
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
        """Set the region of interest, within what the sensor allows.

        Out-of-range values are clamped, and every value is snapped down onto
        the camera's increment (GenApi rejects off-grid values rather than
        rounding them). Read :attr:`roi_info` for the geometry that stuck.

        Width and height are locked while the camera streams, so acquisition
        is stopped around the write and restored afterwards; callers need
        not do it themselves.

        Args:
            offset_x: X offset in pixels.
            offset_y: Y offset in pixels.
            width: ROI width in pixels (``None`` for the full sensor width).
            height: ROI height in pixels (``None`` for the full sensor height).

        Raises:
            ValueError: A value is not a whole number of pixels, or a size is
                below one pixel.
            RuntimeError: The camera is not open, or refused the ROI. The
                previous ROI is put back first, and the message says what the
                camera is now set to.
        """
        ox = _roi_pixels("offset_x", offset_x)
        oy = _roi_pixels("offset_y", offset_y)
        w = None if width is None else _roi_pixels("width", width, minimum=1)
        h = None if height is None else _roi_pixels("height", height, minimum=1)

        if not self.node_map:
            raise RuntimeError("Camera not opened.")

        x, y = self._roi_axes()
        new_ox, new_w = _fit_axis(x, ox, w)
        new_oy, new_h = _fit_axis(y, oy, h)
        before = self._read_roi()

        was_acquiring = self.is_acquiring
        if was_acquiring:
            self.stop_acquisition()
        try:
            try:
                self._write_roi(new_ox, new_oy, new_w, new_h, x, y)
            except _NODE_ERRORS as exc:
                try:
                    self._write_roi(*before, x, y)
                except _NODE_ERRORS:
                    logger.warning("Could not restore the previous ROI", exc_info=True)
                self._refresh_roi_cache()
                raise RuntimeError(
                    f"The camera refused a {new_w}×{new_h} ROI at ({new_ox}, {new_oy}): "
                    f"{_node_error_text(exc)} It is set to {self.width_pixels}×"
                    f"{self.height_pixels} at ({self._roi_offset_x}, {self._roi_offset_y})."
                ) from exc
            self._refresh_roi_cache()
        finally:
            if was_acquiring:
                self.start_acquisition()

        logger.info(
            "ROI set: offset=(%d, %d), size=%d×%d",
            self._roi_offset_x,
            self._roi_offset_y,
            self.width_pixels,
            self.height_pixels,
        )

    @property
    def roi_info(self) -> dict[str, int]:
        """The current ROI, read back from the camera.

        Falls back to the last geometry seen once the camera is closed or a
        read fails, so it is always safe to call.

        Returns:
            Dict with keys ``offset_x``, ``offset_y``, ``width``, ``height``,
            ``max_width``, ``max_height``. The maxima follow binning and
            decimation, which change them on the camera.
        """
        if self.node_map:
            try:
                self._refresh_roi_cache()
            except Exception:
                logger.debug("Could not read the ROI back from the camera", exc_info=True)
        return {
            "offset_x": self._roi_offset_x,
            "offset_y": self._roi_offset_y,
            "width": self.width_pixels,
            "height": self.height_pixels,
            "max_width": self._roi_max_width,
            "max_height": self._roi_max_height,
        }
