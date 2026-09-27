"""GenICam cameras, driven through Harvesters.

:class:`HarvesterCamera` is the workhorse behind every real device: FLIR and
Basler differ only in where their GenTL producer lives, which
:mod:`pybeamprofiler.cti` already handles, so both vendor classes are thin
subclasses.

Most of what is unusual here follows from how the underlying C libraries
behave rather than from anything the GenICam standard asks for:

* A GenTL producer can be initialised only once per process, so every
  camera and every discovery pass share one Harvester (see
  :class:`_SharedHarvester`).
* Harvesters and the GenICam bindings segfault, rather than raise, when a
  fetched buffer is re-queued after acquisition was stopped under it, or
  when a node of a released device is read. Each camera therefore
  serialises everything that touches its device, and ``close()`` drops
  every handle into it.
* ``ImageAcquirer.fetch()`` is not bounded by its timeout, so frames are
  taken with ``try_fetch`` against a deadline kept here.
* Acquisition is restarted after an exposure change so the producer's
  buffer ring cannot deliver stale-exposure frames, and a producer that goes
  silent gets one stop/start recovery attempt.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import re
import threading
import time
from typing import Any

import numpy as np

try:
    from harvesters.core import Harvester
    from harvesters.core import TimeoutException as _HarvestersTimeout
except ImportError:
    Harvester = None  # ty:ignore[invalid-assignment]
    _HarvestersTimeout = None  # ty:ignore[invalid-assignment]

from .camera import Camera, _align_down, _Axis, _fit_axis, _roi_pixels
from .cti import find_cti_files, parse_gentl_path

logger = logging.getLogger(__name__)

#: Longest single wait inside get_image(). Stop, close and ROI or exposure
#: writes are serialised with fetching, so this also bounds how long a fetch
#: in progress can keep them waiting.
_FETCH_SLICE_S = 0.1

#: The shortest real wait, used for polls: try_fetch(timeout=0) means "wait
#: forever" to Harvesters 1.4, not "don't wait".
_POLL_S = 0.001


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


def _node_float(node: Any, attr: str = "value") -> float | None:
    """Read ``node.<attr>`` as a float, or ``None`` if it is absent or unreadable."""
    if node is None:
        return None
    try:
        return _numeric(getattr(node, attr))
    except Exception:
        return None


def _node_error_text(exc: BaseException) -> str:
    """The human half of a GenICam error message.

    GenApi messages read "Value = 100 must be equal or smaller than Max = 0.
    : OutOfRangeException thrown in node 'OffsetX' while calling ... (file
    'IntegerT.h', line 81)"; the part after the separator is for C++
    developers.
    """
    return str(exc).split(" : ")[0].strip() or type(exc).__name__


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


def _samples_per_pixel(fmt: str) -> float | None:
    """Samples of ``component.data`` per pixel, from the PFNC format name.

    Harvesters unpacks packed formats (Mono12p and the like) before handing
    the data over, so this is about colour layout, not bit packing. ``None``
    for a name not recognised here.
    """
    if fmt.startswith(("Mono", "Bayer", "Coord3D_C")):
        return 1
    if fmt.startswith(("RGBa", "BGRa")):
        return 4
    if fmt.startswith(("RGB", "BGR")):
        return 3
    if fmt.startswith(("YUV", "YCbCr")):
        if "411" in fmt:
            return 1.5
        if "422" in fmt:
            return 2
        return 3
    return None


def _bit_depth_of(fmt: Any) -> int | None:
    """Bits per sample of a PFNC pixel format name, or ``None`` if unknown.

    "Mono12p" is 12, "BayerRG10" 10, "BGRa8" 8. In "YUV422_8" the 422 is
    the chroma sampling, not a depth, so only numbers up to 32 count.
    """
    if not isinstance(fmt, str) or not fmt:
        return None
    for number in re.findall(r"\d+", fmt.removeprefix("Coord3D_")):
        depth = int(number)
        if 0 < depth <= 32:
            return depth
    return None


def _to_mono(component: Any) -> np.ndarray:
    """Turn one payload component into a 2D intensity array.

    The layout is taken from the pixel format name, not guessed from the
    payload size:

    * Mono and Bayer formats are one sample per pixel. Bayer data is the raw
      colour-filter mosaic -- the camera's actual counts, clipping included
      -- and is returned as it is; the caller warns about it.
    * RGB/BGR(a) formats become the brightest colour channel of each pixel,
      alpha ignored. A luminance weighting hid clipping (a red channel at
      full scale came out at 30% of it, so the saturation check never
      fired), gave BGR the weights meant for RGB, and averaged alpha into
      the image as an offset.
    * YUV/YCbCr formats give their luma plane; averaging chroma in added
      half of full scale to a black frame.

    Harvesters 1.4 does not strip line padding (it discards the result of
    its own ``numpy.delete``), so padded rows are cropped here using the
    component's ``x_padding``.

    Args:
        component: A Harvesters ``Component2DImage``.

    Returns:
        A fresh 2D array; the payload buffer is reused by the producer as soon
        as the ``fetch`` context exits, so the copy is not optional.

    Raises:
        ValueError: If the payload cannot be read as the frame it claims to be.
    """
    height, width = int(component.height), int(component.width)
    data = component.data
    pixels = height * width
    if pixels == 0:
        raise ValueError("Camera reported a zero-sized frame")

    raw_format = getattr(component, "data_format", None)
    fmt = raw_format if isinstance(raw_format, str) else ""
    per_pixel = _samples_per_pixel(fmt)
    if per_pixel is None:
        # A format not named above: infer the layout from the size.
        per_pixel = data.size / pixels
    if per_pixel != int(per_pixel) or per_pixel < 1:
        raise ValueError(
            f"Cannot interpret a {data.size}-sample payload as a {width}x{height} "
            f"frame (pixel format {fmt or 'unknown'!r}): it does not hold a whole "
            "number of samples per pixel. Select Mono8, Mono12 or Mono16 on the camera."
        )
    channels = int(per_pixel)
    row = width * channels

    if data.size != pixels * channels:
        padding = _numeric(getattr(component, "x_padding", None)) or 0
        per_row, remainder = divmod(data.size, height)
        if padding > 0 and not remainder and per_row > row:
            data = data.reshape(height, per_row)[:, :row]
        else:
            packed = fmt.endswith("p") or "Packed" in fmt
            hint = (
                "The payload is still packed (Mono10p/Mono12p style), which this "
                "version of Harvesters did not unpack. "
                if packed
                else ""
            )
            raise ValueError(
                f"Cannot interpret a {data.size}-sample payload as a {width}x{height} "
                f"frame (pixel format {fmt or 'unknown'!r}). {hint}"
                "Select Mono8, Mono12 or Mono16 on the camera."
            )

    if channels == 1:
        return data.reshape(height, width).copy()
    planes = data.reshape(height, width, channels)
    if fmt.startswith(("YUV", "YCbCr")):
        luma = 1 if any(order in fmt for order in ("UYV", "CbYCr")) else 0
        return planes[:, :, luma].copy()
    return planes[:, :, :3].max(axis=2)


def _device_field(device: Any, name: str) -> str:
    """One field of a Harvesters ``DeviceInfo``, as a stripped string.

    ``property_dict`` is the snapshot Harvesters takes at enumeration, with
    the fields a producer does not implement already turned into ``None``.
    Reading the live attribute instead raises for those -- which used to
    abort discovery, and every ``open()``, for all cameras because of one.
    """
    props = getattr(device, "property_dict", None)
    if isinstance(props, dict) and name in props:
        value = props[name]
    else:
        try:
            value = getattr(device, name, None)
        except Exception:
            value = None
    return "" if value is None else str(value).strip()


def _device_cti(device: Any) -> str | None:
    """The resolved path of the producer that enumerated ``device``, if known.

    Harvesters links ``DeviceInfo`` -> ``Interface`` -> ``System`` ->
    ``Producer``, and the producer knows the file it was loaded from.
    """
    try:
        path = device.parent.parent.parent.path_name
    except Exception:
        return None
    return os.path.realpath(path) if isinstance(path, str) else None


class _SharedHarvester:
    """The one Harvester this process uses, and who is using it.

    A GenTL producer can be initialised once per process: a second
    Harvester loading the same ``.cti`` gets ``GC_ERR_RESOURCE_IN_USE`` from
    ``GCInitLib``, and Harvesters drops the producer with a log line, so it
    sees no cameras at all. That is what stopped the GUI from switching
    between two cameras on one producer, and made a rescan find nothing
    while a camera was open. Discovery and every camera therefore go through
    this single Harvester.

    ``Harvester.update()`` destroys every ImageAcquirer the Harvester has
    created, so it only runs while no camera is open. While one is, the
    device list from the last enumeration is served instead; cameras
    plugged in since then appear once no camera is open.

    Attributes:
        lock: Guards everything here. Taken inside a camera's own lock,
            never the other way round.
        harvester: The Harvester, or ``None`` while nothing needs one.
        files: Producers added to it, in load order.
        users: ImageAcquirers created from it and not yet destroyed. At zero
            the Harvester is reset and dropped, releasing the producers.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.harvester: Any = None
        self.files: list[str] = []
        self.users = 0


_SHARED = _SharedHarvester()


def _new_harvester() -> Any:
    """A new Harvester, or ImportError if Harvesters is not installed."""
    if Harvester is None:
        raise ImportError("harvesters is not installed")
    return Harvester()


def _drop_shared_harvester() -> None:
    """Reset and forget the shared Harvester. Call with ``_SHARED.lock`` held."""
    h = _SHARED.harvester
    _SHARED.harvester, _SHARED.files, _SHARED.users = None, [], 0
    if h is not None:
        try:
            h.reset()
        except Exception:
            logger.debug("Error resetting Harvester", exc_info=True)


def _reset_shared_harvester() -> None:
    """Drop the shared Harvester whatever holds it. For test isolation."""
    with _SHARED.lock:
        _drop_shared_harvester()


def _acquire_device(files: list[str], choose: Any) -> tuple[Any, Any, Any]:
    """Open a device through the shared Harvester, creating it if need be.

    Args:
        files: Producers the caller needs loaded, in order of preference.
        choose: Called with the device list; returns the ``DeviceInfo`` to
            open or raises ``RuntimeError`` saying why there is none.

    Returns:
        ``(harvester, image_acquirer, device_info)``.
    """
    with _SHARED.lock:
        h = _SHARED.harvester
        if h is None:
            h = _new_harvester()
            _SHARED.harvester = h
        missing = [path for path in files if path not in _SHARED.files]
        if _SHARED.users == 0:
            for path in missing:
                h.add_file(path)
                _SHARED.files.append(path)
            # Nothing is open, so a fresh enumeration is safe, and it finds
            # cameras plugged in since the last one.
            h.update()
        elif missing:
            logger.warning(
                "Not loading %s: another camera is open, and adding a producer means "
                "re-enumerating, which would close it.",
                ", ".join(missing),
            )
        try:
            try:
                device = choose(list(h.device_info_list))
            except RuntimeError as exc:
                if _SHARED.users:
                    raise RuntimeError(
                        f"{exc} (The device list is not rescanned while another camera is open.)"
                    ) from exc
                raise
            ia = h.create(device)
        except BaseException:
            if _SHARED.users == 0:
                _drop_shared_harvester()
            raise
        _SHARED.users += 1
        logger.info(f"Harvester loaded {len(_SHARED.files)} CTI file(s)")
        for path in _SHARED.files:
            logger.info(f"  CTI: {path}")
        return h, ia, device


def _return_device(ia: Any) -> None:
    """Destroy an ImageAcquirer from the shared Harvester and drop a user."""
    with _SHARED.lock:
        try:
            ia.destroy()
        except Exception:
            logger.debug("Error destroying ImageAcquirer", exc_info=True)
        if _SHARED.users > 0:
            _SHARED.users -= 1
        if _SHARED.users == 0:
            _drop_shared_harvester()


def _device_records(devices: list[Any], files: list[str]) -> list[dict[str, str | int]]:
    """Plain-data descriptions of the devices that came from ``files``."""
    wanted = {os.path.realpath(path) for path in files}
    records: list[dict[str, str | int]] = []
    for device in devices:
        cti = _device_cti(device)
        if cti is not None and wanted and cti not in wanted:
            continue
        records.append(
            {
                "vendor": _device_field(device, "vendor"),
                "model": _device_field(device, "model"),
                "serial_number": _device_field(device, "serial_number"),
                "id": _device_field(device, "id_"),
                "index": len(records),
                "cti": cti or "",
            }
        )
    return records


def _list_devices(files: list[str]) -> list[dict[str, str | int]]:
    """Describe every device the given producers can see.

    Never disturbs an open camera: while one is open this answers from the
    last enumeration rather than calling ``update()``. With nothing open it
    enumerates on a scratch Harvester and resets it again, so no producer
    stays loaded behind the caller's back.

    Raises:
        Exception: Whatever enumeration raises; callers treat discovery as
            best-effort.
    """
    with _SHARED.lock:
        if _SHARED.users and _SHARED.harvester is not None:
            missing = [path for path in files if path not in _SHARED.files]
            if missing:
                logger.info(
                    "%d producer(s) are not loaded and cannot be while a camera is open: %s",
                    len(missing),
                    ", ".join(missing),
                )
            return _device_records(list(_SHARED.harvester.device_info_list), files)

        h = _new_harvester()
        try:
            for path in files:
                try:
                    h.add_file(path)
                except Exception as e:
                    logger.warning(f"Could not load {path}: {e}")
            h.update()
            return _device_records(list(h.device_info_list), files)
        finally:
            try:
                h.reset()
            except Exception:
                logger.debug("Error resetting Harvester", exc_info=True)


class HarvesterCamera(Camera):
    """GenICam camera interface using Harvesters library.

    Provides a unified interface for FLIR, Basler, and other GenICam-compliant
    cameras via standard GenTL producers (``.cti`` files).

    Safe to use from several threads: fetching, starting, stopping, closing
    and every node write are serialised on the camera's own lock. A fetch
    waits in slices and steps aside between them, so a stop() or close()
    issued mid-wait runs within one slice (0.1 s) rather than behind the
    whole timeout.

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
        device_id: str | None = None,
    ) -> None:
        """Initialize Harvester camera.

        Nothing is loaded or claimed until :meth:`open`.

        Args:
            cti_file: Path(s) to GenTL producer (``.cti``) file(s).
                If ``None``, the caller (e.g. :class:`BaslerCamera`) is expected
                to resolve the path via ``GENICAM_GENTL64_PATH`` or platform search.
            serial_number: Open the device with exactly this serial number.
            device_id: Open the device with exactly this GenTL device id, for
                producers that report no serial number. Ignored if
                ``serial_number`` is given.
        """
        super().__init__()
        if Harvester is None:
            raise ImportError(
                "harvesters/genicam is not available. On macOS, install the camera SDK "
                "(Pylon or Spinnaker) and ensure its genicam Python bindings are on the path. "
                "On Linux/Windows: pip install harvesters"
            )

        # The producers *this* camera was asked for. Every other installed
        # producer is loaded too (see _harvester_files), but only these
        # decide which device "the first camera" means.
        self._cti_files: list[str] = []
        if cti_file:
            files = [cti_file] if isinstance(cti_file, str) else cti_file
            for file_path in files:
                if not os.path.exists(file_path):
                    logger.warning(f"CTI file not found: {file_path}")
                else:
                    self._cti_files.append(file_path)
                    logger.info(f"Using CTI file: {file_path}")
        else:
            logger.warning(
                "No CTI file specified. Please provide cti_file parameter or set GENICAM_GENTL64_PATH."
            )

        # The Harvester in use while open. It is the process-wide shared one
        # unless a Harvester was assigned here before open(), which is then
        # used as it is and reset by close().
        self.h: Any = None
        self._shared = False
        self.device_id: str | None = device_id
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
        # Pixel format of the most recent frame (or of the camera at open).
        self._pixel_format: str | None = None
        # Seconds between the last two frames. A camera delivering a frame
        # every 8 s is not stalled after 5 s of silence, whatever
        # exposure_time says.
        self._frame_interval: float = 0.0

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
        """Claim the device and read back its capabilities.

        A failure at any point releases whatever was claimed on the way, so
        a failed open() never leaves a producer or device held -- the next
        attempt, in this process or another, starts clean.

        Raises:
            RuntimeError: No matching camera, or the device refused to open
                (most often because another application holds it).
        """
        with self._device():
            if self.ia is not None:
                self._release_device()  # reopening: hand the old device back first
            self._generation += 1
            try:
                if self.h is not None:
                    ia, device = self._open_private()
                else:
                    self.h, ia, device = _acquire_device(
                        self._harvester_files(), self._choose_device
                    )
                    self._shared = True
                self.ia = ia
                self._take_device_identity(device)
                self.node_map = ia.remote_device.node_map
                self._configure_device()
            except BaseException:
                self.close()
                raise
            logger.info(f"Camera opened successfully: {self.device_model}")

    def _configure_device(self) -> None:
        """Put a freshly opened device into a known state and read it back."""
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
        self._sync_exposure_and_gain()
        pixel_format = _node(self.node_map, "PixelFormat")
        try:
            fmt = pixel_format.value if pixel_format is not None else None
        except Exception:
            fmt = None
        self._note_pixel_format(fmt if isinstance(fmt, str) else None)

    def _harvester_files(self) -> list[str]:
        """Producers to load: this camera's own first, then every other one found.

        Loading them all means discovery can list every camera while this one
        is open -- it cannot add a producer later without closing this camera.
        Own producers go first so a camera visible through two of them is
        opened through its vendor's.
        """
        extra = []
        try:
            extra = find_cti_files()
        except Exception:
            logger.debug("Producer search failed", exc_info=True)
        files: list[str] = []
        for path in [*self._cti_files, *extra]:
            if path not in files:
                files.append(path)
        return files

    def _open_private(self) -> tuple[Any, Any]:
        """Open through a Harvester assigned to this camera, as it is."""
        h = self.h
        for path in self._cti_files:
            h.add_file(path)
        h.update()
        device = self._choose_device(list(h.device_info_list))
        return h.create(device), device

    def _choose_device(self, devices: list[Any]) -> Any:
        """Pick the device to open from an enumeration.

        A serial number or device id is matched exactly -- a substring match
        let "4001234" open camera "24001234". Without either, the first
        device found through this camera's own producers is used, so a
        :class:`FlirCamera` does not open a Basler camera just because the
        Pylon producer is loaded too.

        Raises:
            RuntimeError: Nothing matches.
        """
        if not devices:
            raise RuntimeError(
                f"No GenICam cameras found using {len(self._harvester_files())} CTI file(s). "
                "Ensure camera is connected and the correct GenTL producer (.cti) is loaded. "
                f"Loaded CTI files: {self._harvester_files()}"
            )
        logger.info(f"Found {len(devices)} camera(s):")
        for i, device in enumerate(devices):
            logger.info(
                f"  [{i}] {_device_field(device, 'vendor')} {_device_field(device, 'model')} "
                f"(S/N: {_device_field(device, 'serial_number')})"
            )

        own = {os.path.realpath(path) for path in self._cti_files}

        def is_own(device: Any) -> bool:
            cti = _device_cti(device)
            return not own or cti is None or cti in own

        if self.serial_number:
            wanted = str(self.serial_number).strip()
            matches = [d for d in devices if _device_field(d, "serial_number") == wanted]
            if not matches:
                raise RuntimeError(f"Camera with serial number '{wanted}' not found.")
        elif self.device_id:
            wanted = str(self.device_id).strip()
            matches = [d for d in devices if _device_field(d, "id_") == wanted]
            if not matches:
                raise RuntimeError(f"Camera with device id '{wanted}' not found.")
        else:
            matches = [d for d in devices if is_own(d)]
            if not matches:
                raise RuntimeError(
                    f"No GenICam cameras found using {len(self._cti_files)} CTI file(s). "
                    f"Loaded CTI files: {self._cti_files}"
                )
            logger.info(f"Using first camera: {_device_field(matches[0], 'model')}")
        preferred = [d for d in matches if is_own(d)]
        return (preferred or matches)[0]

    def _take_device_identity(self, device: Any) -> None:
        """Record who the opened device is, as the producer reports it."""
        self.device_model = _device_field(device, "model") or None
        self.device_vendor = _device_field(device, "vendor") or None
        self.serial_number = _device_field(device, "serial_number") or self.serial_number
        self.device_id = _device_field(device, "id_") or self.device_id

    def _detect_pixel_size(self) -> None:
        """Detect pixel size from camera's GenICam features.

        Tries multiple standard feature names, sensor model lookup, and defaults to 1.0 μm.
        A feature that exists but refuses to be read (GenICam AccessException,
        which is not a ValueError) moves on to the next source instead of
        abandoning the search.
        """
        try:
            pixel_size = None

            try:
                if hasattr(self.node_map, "SensorPixelWidth"):
                    pixel_size = self.node_map.SensorPixelWidth.value
                    logger.debug("Using SensorPixelWidth for pixel size")
            except _NODE_ERRORS:
                pass

            if pixel_size is None:
                try:
                    if hasattr(self.node_map, "SensorPixelHeight"):
                        pixel_size = self.node_map.SensorPixelHeight.value
                        logger.debug("Using SensorPixelHeight for pixel size")
                except _NODE_ERRORS:
                    pass

            if pixel_size is None:
                try:
                    if hasattr(self.node_map, "PixelSize"):
                        val = self.node_map.PixelSize.value
                        if isinstance(val, (int, float)):
                            pixel_size = val
                            logger.debug("Using PixelSize for pixel size")
                except _NODE_ERRORS:
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
        """Release the device. Safe to call twice, or after a failed open().

        Every handle into Harvesters is dropped here, not just released:
        the node map and acquirer of a closed device point at freed C++
        objects, and reading a node through them segfaults the interpreter
        rather than raising. A closed camera answers "Camera not opened."
        instead, and can be opened again.
        """
        with self._device():
            try:
                self.stop_acquisition()
            except Exception:
                logger.debug("Error stopping acquisition during close", exc_info=True)
            was_shared = self._shared
            self._release_device()
            h, self.h = self.h, None
            if h is not None and not was_shared:
                try:
                    h.reset()
                except Exception:
                    logger.debug("Error resetting Harvester", exc_info=True)

    def _release_device(self) -> None:
        """Give the device back and forget every handle that pointed into it."""
        self._generation += 1
        ia, self.ia = self.ia, None
        self.node_map = None
        self.is_acquiring = False
        self._feature_cache = None
        self._feature_cache_source = None
        if self._shared:
            self._shared = False
            self.h = None
            if ia is not None:
                _return_device(ia)
        elif ia is not None:
            try:
                ia.destroy()
            except Exception:
                logger.debug("Error destroying ImageAcquirer", exc_info=True)

    def start_acquisition(self) -> None:
        """Start image acquisition on the GenTL producer."""
        with self._device():
            if not self.ia or self.is_acquiring:
                return
            self._start_stream()
            # A deliberate start re-arms the stall recovery; see _recover_if_stalled.
            self._last_successful_fetch = 0.0
            self._stall_recovery_attempted = False

    def _start_stream(self) -> None:
        """Start the producer without touching the stall-recovery state."""
        self.ia.start()
        self.is_acquiring = True

    def stop_acquisition(self) -> None:
        """Stop image acquisition on the GenTL producer."""
        with self._device():
            if self.ia and self.is_acquiring:
                try:
                    self.ia.stop()
                except Exception:
                    logger.debug("Error stopping acquisition", exc_info=True)
            self.is_acquiring = False

    def get_image(self, timeout: float | None = None) -> np.ndarray:
        """Fetch the next frame from the GenTL producer.

        Returns within about ``timeout`` seconds whatever the producer does.
        Harvesters' own ``fetch()`` does not: it starts its timeout afresh
        after every incomplete buffer, so a GigE link that drops a packet
        from every frame kept it waiting indefinitely. Frames are therefore
        taken with ``try_fetch`` against a deadline kept here, in slices of
        at most ``_FETCH_SLICE_S``.

        For long exposures pass a small ``timeout`` and treat
        :class:`TimeoutError` as "no new frame yet, try again next tick", so
        the UI stays responsive.

        Args:
            timeout: Maximum seconds to wait for a frame. ``0`` or less
                polls: it returns a frame only if one is ready. Defaults to
                ``max(2.0, exposure_time + 2.0)``.

        Returns:
            2D numpy array containing the frame data.

        Raises:
            RuntimeError: If the camera is not open ("Camera not opened."),
                including when another thread closes it mid-wait.
            TimeoutError: If no frame arrives in time, or another thread
                stops acquisition mid-wait (it is not restarted behind them).
        """
        if timeout is None:
            timeout = max(2.0, (self.exposure_time or 0) + 2.0)
        deadline = time.monotonic() + max(timeout, 0.0)
        first = True
        while True:
            self._yield_to_controls()
            img = self._fetch_slice(deadline, may_start=first)
            first = False
            if img is not None:
                return img
            if time.monotonic() >= deadline:
                if timeout <= 0:
                    raise TimeoutError("No frame was ready.")
                raise TimeoutError(
                    f"Camera did not deliver a frame within {timeout:.1f} s. "
                    "Check that the camera is connected, powered, and not in "
                    "use by another application."
                )

    def _fetch_slice(self, deadline: float, *, may_start: bool) -> np.ndarray | None:
        """Wait for one frame until ``deadline``, but no longer than one slice.

        Args:
            deadline: ``time.monotonic()`` value to give up at.
            may_start: Start acquisition if it is not running. Only the first
                slice of a call may: between slices another thread may have
                stopped acquisition on purpose (``BeamProfiler.stop()``), and
                restarting it would leave the camera streaming after "stop".

        Returns:
            The frame, or ``None`` if none arrived (or it was incomplete).
        """
        with self._lock:
            if not self.ia:
                raise RuntimeError("Camera not opened.")
            if not self.is_acquiring:
                if not may_start:
                    raise TimeoutError("Acquisition was stopped while waiting for a frame.")
                self.start_acquisition()
            self._recover_if_stalled()

            # try_fetch(timeout=0) waits forever in Harvesters 1.4; a poll uses
            # the shortest real wait instead.
            wait = min(max(deadline - time.monotonic(), _POLL_S), _FETCH_SLICE_S)
            try:
                buffer = self.ia.try_fetch(timeout=wait)
            except Exception as exc:
                # Harvesters' TimeoutException shares only a name with the
                # built-in TimeoutError; neither means "the device failed".
                if isinstance(exc, TimeoutError) or (
                    _HarvestersTimeout is not None and isinstance(exc, _HarvestersTimeout)
                ):
                    buffer = None
                else:
                    raise
            if buffer is None:
                # Seed the stall timer on the first empty wait, so a camera that
                # has simply not delivered its first frame yet is not "stalled".
                if not self._last_successful_fetch:
                    self._last_successful_fetch = time.monotonic()
                return None

            with buffer as held:
                component = held.payload.components[0]
                self.width_pixels = component.width
                self.height_pixels = component.height
                fmt = getattr(component, "data_format", None)
                if isinstance(fmt, str) and fmt != self._pixel_format:
                    self._note_pixel_format(fmt)
                img = _to_mono(component)
            now = time.monotonic()
            if self._last_successful_fetch:
                self._frame_interval = now - self._last_successful_fetch
            self._last_successful_fetch = now
            self._stall_recovery_attempted = False
            return img

    def _note_pixel_format(self, fmt: str | None) -> None:
        """Track the pixel format frames arrive in: its bit depth, and Bayer."""
        self._pixel_format = fmt
        self.bit_depth = _bit_depth_of(fmt)
        if fmt and fmt.startswith("Bayer"):
            logger.warning(
                "Frames are %s: a colour-filter mosaic, in which neighbouring pixels "
                "see different colours, so a beam shows a 2x2 checkerboard. Select a "
                "Mono pixel format on the camera for beam profiling.",
                fmt,
            )

    def _recover_if_stalled(self) -> None:
        """Restart acquisition once if the producer has gone quiet.

        Some GenTL stacks stop delivering after long runs until acquisition
        is restarted. The attempt is made once per silence: a successful
        frame, or a deliberate start_acquisition(), re-arms it. The restart
        itself must not re-arm it -- it used to, so a camera that was
        genuinely idle was restarted every five seconds forever, and an
        exposure longer than that was aborted on every attempt and never
        delivered a frame.

        Silence is measured against the longest of 5 s, three exposures and
        three of the intervals the camera has actually been delivering at,
        so a long exposure set behind this object's back (auto-exposure, or
        another tool before open()) is not mistaken for a stall. A camera
        waiting for a trigger is silent by design and is left alone.
        """
        if not self._last_successful_fetch or self._stall_recovery_attempted:
            return
        silent = time.monotonic() - self._last_successful_fetch
        window = max(5.0, 3.0 * (self.exposure_time or 0.0), 3.0 * self._frame_interval)
        if silent <= window or self._waits_for_trigger():
            return
        logger.warning(
            "Acquisition appears stalled (no frame for %.1f s); "
            "attempting to recover by restarting acquisition.",
            silent,
        )
        self._stall_recovery_attempted = True
        try:
            self.stop_acquisition()
            self._start_stream()
        except Exception:
            logger.debug("Stall recovery failed", exc_info=True)

    def _waits_for_trigger(self) -> bool:
        """Is the camera configured to expose only when triggered?"""
        node = _node(self.node_map, "TriggerMode")
        if node is None:
            return False
        try:
            return str(node.value) == "On"
        except Exception:
            return False

    def set_exposure(self, exposure_time: float) -> None:
        """Set the exposure, in seconds, within what the camera allows.

        The value is clamped to the node's range as the camera reports it
        now -- on some cameras the maximum follows the frame rate -- because
        GenApi refuses an out-of-range write instead of clamping it.
        Acquisition is restarted around the write so the producer's buffer
        ring cannot deliver a couple of frames still at the old exposure.

        Afterwards :attr:`exposure_time` is what the camera reports back, not
        what was asked for.

        Args:
            exposure_time: Exposure time in seconds.

        Raises:
            RuntimeError: The camera is not open, or refused the value (for
                example while auto exposure is on).
        """
        with self._device():
            if not self.node_map:
                raise RuntimeError("Camera not opened.")
            node = self._exposure_node()
            if node is None:
                logger.warning("This camera has no ExposureTime feature; exposure unchanged.")
                return
            target_us = self._clamp_to_node(float(exposure_time) * 1_000_000, node)
            lo, hi = _node_float(node, "min"), _node_float(node, "max")
            if lo is not None and hi is not None:
                self._exposure_min, self._exposure_max = lo / 1_000_000, hi / 1_000_000

            was_acquiring = self.is_acquiring
            if was_acquiring:
                self.stop_acquisition()
            try:
                try:
                    node.value = target_us
                except _NODE_ERRORS as exc:
                    raise RuntimeError(
                        f"The camera refused an exposure of {target_us / 1000:.3f} ms: "
                        f"{_node_error_text(exc)}"
                    ) from exc
                readback = _node_float(node)
                self.exposure_time = (target_us if readback is None else readback) / 1_000_000
            finally:
                if was_acquiring:
                    self.start_acquisition()

    def set_gain(self, gain: float) -> None:
        """Set the gain within what the camera allows.

        Clamped to the node's live range like :meth:`set_exposure`. Falls
        back to the legacy ``GainRaw`` feature, whose units are raw ADC
        steps rather than dB, on cameras without ``Gain``. Afterwards
        :attr:`gain` is what the camera reports back.

        Args:
            gain: Gain in the camera's own units -- dB on most SFNC-compliant
                devices, raw ADC steps on older ones.

        Raises:
            RuntimeError: The camera is not open, or refused the value.
        """
        with self._device():
            if not self.node_map:
                raise RuntimeError("Camera not opened.")
            node = _node(self.node_map, "Gain")
            raw = node is None
            if raw:
                node = _node(self.node_map, "GainRaw")
            if node is None:
                logger.warning("This camera has no Gain feature; gain unchanged.")
                return
            target: float = self._clamp_to_node(float(gain), node)
            if raw:
                lo = _node_int(node, "min") or 0
                target = _align_down(int(round(target)), lo, max(1, _node_int(node, "inc") or 1))
            try:
                node.value = target
            except _NODE_ERRORS as exc:
                raise RuntimeError(
                    f"The camera refused a gain of {target:g}: {_node_error_text(exc)}"
                ) from exc
            readback = _node_float(node)
            self.gain = target if readback is None else readback

    def _exposure_node(self) -> Any:
        """``ExposureTime`` (SFNC), or ``ExposureTimeAbs`` on older cameras; both µs."""
        for name in ("ExposureTime", "ExposureTimeAbs"):
            node = _node(self.node_map, name)
            if node is not None:
                return node
        return None

    @staticmethod
    def _clamp_to_node(value: float, node: Any) -> float:
        """``value`` limited to the node's current ``min``/``max``, where readable."""
        lo, hi = _node_float(node, "min"), _node_float(node, "max")
        if lo is not None and value < lo:
            logger.info("%g is below the camera's minimum; using %g", value, lo)
            value = lo
        if hi is not None and value > hi:
            logger.info("%g is above the camera's maximum; using %g", value, hi)
            value = hi
        return value

    def _sync_exposure_and_gain(self) -> None:
        """Take exposure and gain from the device instead of assuming defaults.

        They used to stay at the class defaults (10 ms, 0 dB) whatever the
        camera was set to -- by pylon Viewer, a user set, or a previous
        session -- so the GUI showed the wrong values, and every timeout and
        stall window derived from exposure_time was wrong with them.
        """
        exposure_us = _node_float(self._exposure_node())
        if exposure_us is not None:
            self.exposure_time = exposure_us / 1_000_000
        gain = _node_float(_node(self.node_map, "Gain"))
        if gain is None:
            gain = _node_float(_node(self.node_map, "GainRaw"))
        if gain is not None:
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
        with self._device():
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
        with self._device():
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
