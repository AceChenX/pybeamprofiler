"""Utility functions for pybeamprofiler."""

from __future__ import annotations

import logging
import os

from .cti import find_cti_files as _find_cti_files

logger = logging.getLogger(__name__)


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
                logger.warning("No GenTL producers (.cti files) found")
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
