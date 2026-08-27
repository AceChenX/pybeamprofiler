"""Finding cameras: which devices are attached, and how to open one.

Two layers live here.

* :func:`list_cameras` / :func:`print_camera_info` are the long-standing
  informational helpers — they answer "what is plugged in?" as plain data.
* :class:`CameraOption` and :func:`open_camera` are what the GUI drives: a
  stable, serialisable handle for each device that survives a round trip
  through a dropdown and can be turned back into an open camera.

The simulated camera is deliberately offered alongside the real ones so the
GUI has something to show on a machine with no hardware attached.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cti import find_cti_files as _find_cti_files

if TYPE_CHECKING:
    from .camera import Camera

logger = logging.getLogger(__name__)

#: Camera kind for the built-in simulator.
SIMULATED_KEY = "simulated"

#: Prefix for simulated devices; the remainder is the profile key.
SIMULATED_PREFIX = "simulated:"

#: Prefix for real GenICam devices; the remainder is the serial number.
GENICAM_PREFIX = "genicam:"


def find_cti_files() -> list[str]:
    """Search for GenTL producer (.cti) files in platform-specific paths.

    Thin re-export of :func:`pybeamprofiler.cti.find_cti_files`, kept here
    because it is part of the public API.

    Returns:
        List of found .cti file paths
    """
    return _find_cti_files()


def list_cameras(cti_file: str | None = None) -> list[dict[str, str | int]]:
    """List all available GenICam cameras.

    Args:
        cti_file: Path to specific CTI file, or ``None`` to search all.

    Returns:
        List of dicts with keys ``vendor``, ``model``, ``serial_number``
        (all ``str``), ``id`` (GenTL device id string), and ``index`` (``int``).
    """
    try:
        from harvesters.core import Harvester
    except ImportError:
        logger.error("harvesters package not installed")
        return []

    h = Harvester()

    try:
        if cti_file:
            if os.path.exists(cti_file):
                h.add_file(cti_file)
            else:
                logger.error(f"CTI file not found: {cti_file}")
                return []
        else:
            cti_files = find_cti_files()
            if not cti_files:
                # Expected on any machine using only the simulator, and the
                # GUI calls this every time the camera list is refreshed --
                # so this is information, not a problem worth warning about.
                logger.info("No GenTL producers (.cti files) found")
                return []

            for cti in cti_files:
                try:
                    h.add_file(cti)
                except Exception as e:
                    logger.warning(f"Could not load {cti}: {e}")

        try:
            h.update()
        except Exception as e:
            logger.error(f"Error updating Harvester: {e}")
            return []

        cameras = []
        for i, device in enumerate(h.device_info_list):
            cameras.append(
                {
                    "vendor": device.vendor,
                    "model": device.model,
                    "serial_number": device.serial_number,
                    "id": device.id_,
                    "index": i,
                }
            )

        return cameras

    finally:
        h.reset()


def print_camera_info(cti_file: str | None = None) -> None:
    """Print information about all available cameras to stdout.

    Args:
        cti_file: Path to specific CTI file, or None to search all available
    """
    cameras = list_cameras(cti_file)

    if not cameras:
        print("No cameras found.")
        print("\nMake sure:")
        print("  1. Camera is connected")
        print("  2. GenTL producer (.cti) is installed:")
        print("     - FLIR: Spinnaker SDK")
        print("     - Basler: Pylon SDK")
        return

    print(f"Found {len(cameras)} camera(s):\n")
    for cam in cameras:
        print(f"[{cam['index']}] {cam['vendor']} {cam['model']}")
        print(f"    Serial Number: {cam['serial_number']}")
        print(f"    Device ID: {cam['id']}")
        print()


@dataclass(frozen=True)
class CameraOption:
    """One selectable camera, as shown in the GUI dropdown.

    The identity that matters is :attr:`key`: it round-trips through the
    browser as a plain string and is stable across a rescan, so re-selecting
    the same physical camera works even if devices were plugged or unplugged
    in between. Serial numbers are used rather than enumeration indices for
    exactly that reason — indices shuffle the moment the device list changes.
    """

    key: str
    label: str
    kind: str
    vendor: str = ""
    model: str = ""
    serial_number: str = ""

    @property
    def is_simulated(self) -> bool:
        """Is this the built-in simulator rather than real hardware?"""
        return self.kind == SIMULATED_KEY


def simulated_options() -> list[CameraOption]:
    """The built-in fake cameras, one per :data:`SIMULATED_PROFILES` entry.

    More than one is offered deliberately: it means the camera selector can
    be exercised end to end — including the close/reopen path and the
    re-layout that a different sensor size forces — on a machine with no
    hardware attached.
    """
    from .simulated import SIMULATED_PROFILES

    return [
        CameraOption(
            key=f"{SIMULATED_PREFIX}{profile.key}",
            label=f"{profile.name} ({profile.serial_number})",
            kind=SIMULATED_KEY,
            vendor="pybeamprofiler",
            model=profile.name,
            serial_number=profile.serial_number,
        )
        for profile in SIMULATED_PROFILES
    ]


def default_simulated_option() -> CameraOption:
    """The simulated camera used when nothing else is selected."""
    return simulated_options()[0]


def _describe(info: dict[str, str | int]) -> CameraOption:
    """Turn one :func:`list_cameras` entry into a selectable option."""
    vendor = str(info.get("vendor") or "").strip()
    model = str(info.get("model") or "").strip()
    serial = str(info.get("serial_number") or "").strip()
    device_id = str(info.get("id") or "").strip()

    # Serial is the stable handle, but not every producer reports one; the
    # GenTL device id is the next best thing, and the index is a last resort.
    handle = serial or device_id or f"index-{info.get('index', 0)}"

    name = " ".join(part for part in (vendor, model) if part) or "GenICam camera"
    label = f"{name} ({serial})" if serial else name

    return CameraOption(
        key=f"{GENICAM_PREFIX}{handle}",
        label=label,
        kind="genicam",
        vendor=vendor,
        model=model,
        serial_number=serial,
    )


def discover_cameras(
    *, include_simulated: bool = True, cti_file: str | None = None
) -> list[CameraOption]:
    """Enumerate the cameras a user could select right now.

    Discovery is best-effort: a missing SDK, an unreadable producer or a
    camera already claimed by another application must not raise, because
    this runs behind a "Refresh" button in the GUI where an exception would
    be far less useful than a short list.

    Args:
        include_simulated: Append the built-in simulated cameras. Keep this
            on for the GUI; turn it off when you only want real hardware.
        cti_file: Restrict the search to one GenTL producer.

    Returns:
        Real devices first (in enumeration order), then the simulated ones.
    """
    options: list[CameraOption] = []
    try:
        for info in list_cameras(cti_file):
            options.append(_describe(info))
    except Exception:
        logger.warning("Camera discovery failed", exc_info=True)

    # Two producers can enumerate the same physical camera (a Basler USB3
    # device seen through both the GEV and U3V .cti, for instance), so the
    # same serial can legitimately appear twice.
    unique: list[CameraOption] = []
    seen: set[str] = set()
    for option in options:
        if option.key not in seen:
            seen.add(option.key)
            unique.append(option)

    if include_simulated:
        unique.extend(simulated_options())
    return unique


def find_option(key: str | None, options: list[CameraOption]) -> CameraOption | None:
    """Look up the option matching *key*, or ``None``."""
    if not key:
        return None
    for option in options:
        if option.key == key:
            return option
    return None


def open_camera(option: CameraOption) -> Camera:
    """Open the camera *option* describes.

    Real devices are opened through the generic :class:`HarvesterCamera` with
    every discovered producer loaded, rather than through the FLIR/Basler
    subclasses: those exist only to locate a vendor's ``.cti``, and by this
    point discovery has already found them all.

    Args:
        option: A camera from :func:`discover_cameras`.

    Returns:
        An **opened** camera, ready to acquire.

    Raises:
        RuntimeError: If the device cannot be opened — most often because
            another application already holds it.
    """
    if option.is_simulated:
        from .simulated import SimulatedCamera, profile_for

        camera: Camera = SimulatedCamera(profile_for(option.key.removeprefix(SIMULATED_PREFIX)))
        camera.open()
        return camera

    from .gen_camera import HarvesterCamera

    cti_files = find_cti_files()
    camera = HarvesterCamera(
        cti_file=cti_files or None,
        serial_number=option.serial_number or None,
    )
    try:
        camera.open()
    except Exception as e:
        # Release the half-built Harvester handle; leaking it keeps the
        # device claimed and the next attempt fails for the wrong reason.
        try:
            camera.close()
        except Exception:
            logger.debug("Cleanup after failed open also failed", exc_info=True)
        raise RuntimeError(f"Could not open {option.label}: {e}") from e
    return camera


def describe_open_camera(camera: Camera) -> CameraOption:
    """Build the option that represents an already-open *camera*.

    The GUI needs this at start-up: the camera is opened before the dropdown
    exists (by the CLI, or by a caller constructing :class:`BeamProfiler`
    directly), and the selector has to show it as the current choice. It may
    not appear in :func:`discover_cameras` at all — a camera opened from an
    explicit ``.cti`` path on a machine where the standard search finds
    nothing — so the option is derived from the live object rather than
    looked up.

    Args:
        camera: An open camera.

    Returns:
        A matching :class:`CameraOption`.
    """
    from .simulated import SimulatedCamera

    if isinstance(camera, SimulatedCamera):
        profile = camera.profile
        return CameraOption(
            key=f"{SIMULATED_PREFIX}{profile.key}",
            label=f"{profile.name} ({profile.serial_number})",
            kind=SIMULATED_KEY,
            vendor="pybeamprofiler",
            model=profile.name,
            serial_number=profile.serial_number,
        )

    return _describe(
        {
            "vendor": getattr(camera, "device_vendor", "") or "",
            "model": getattr(camera, "device_model", "") or "",
            "serial_number": getattr(camera, "serial_number", "") or "",
            "id": "",
            "index": 0,
        }
    )
