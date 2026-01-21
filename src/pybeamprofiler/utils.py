"""Utility functions for pybeamprofiler."""

import logging
import os
import platform

logger = logging.getLogger(__name__)


def find_cti_files() -> list[str]:
    """Search for GenTL producer (.cti) files in platform-specific paths.

    Returns:
        List of found .cti file paths
    """
    cti_files = []
    system = platform.system()

    search_paths = []

    if system == "Windows":
        search_paths.extend(
            [
                r"C:\Program Files\FLIR Systems\Spinnaker\cti64",
            ]
        )
        for version in ["7", "6", "5"]:
            search_paths.append(rf"C:\Program Files\Basler\pylon {version}\Runtime\x64")

    elif system == "Linux":
        search_paths.extend(
            [
                "/opt/spinnaker/lib/flir-gentl",
                "/opt/pylon/lib64/gentlproducer/gtl",
                "/opt/pylon5/lib64/gentlproducer/gtl",
            ]
        )

    elif system == "Darwin":
        search_paths.extend(
            [
                "/usr/local/lib",
                "/Library/Application Support/FLIR/Spinnaker/lib",
                "/Library/Frameworks/pylon.framework/Libraries",
            ]
        )

    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue

        # Resolve path to prevent symlink attacks
        try:
            base_path = os.path.realpath(base_path)
        except (OSError, ValueError) as e:
            logger.debug(f"Could not resolve path {base_path}: {e}")
            continue

        # Only search immediate directory, not recursive for security
        try:
            for file in os.listdir(base_path):
                if file.endswith(".cti"):
                    full_path = os.path.join(base_path, file)
                    # Verify the file is actually within base_path
                    if os.path.realpath(full_path).startswith(base_path):
                        cti_files.append(full_path)
        except (OSError, PermissionError) as e:
            logger.debug(f"Could not list directory {base_path}: {e}")

    return cti_files


def list_cameras(cti_file: str | None = None) -> list[dict[str, str]]:
    """List all available GenICam cameras.

    Args:
        cti_file: Path to specific CTI file, or None to search all

    Returns:
        List of camera info dictionaries with keys: vendor, model, serial_number, id
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

        h.update()

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
    """Print information about all available cameras.

    Args:
        cti_file: Path to specific CTI file, or None to search all available
    """
    cameras = list_cameras(cti_file)

    if not cameras:
        logger.info("No cameras found.")
        logger.info("\nMake sure:")
        logger.info("  1. Camera is connected")
        logger.info("  2. GenTL producer (.cti) is installed:")
        logger.info("     - FLIR: Spinnaker SDK")
        logger.info("     - Basler: Pylon SDK")
        return

    logger.info(f"Found {len(cameras)} camera(s):\n")
    for cam in cameras:
        logger.info(f"[{cam['index']}] {cam['vendor']} {cam['model']}")
        logger.info(f"    Serial Number: {cam['serial_number']}")
        logger.info(f"    Device ID: {cam['id']}")
        logger.info("")
