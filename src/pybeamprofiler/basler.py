"""Basler camera interface using Harvesters/GenICam."""

import logging
import os
import platform

from .gen_camera import HarvesterCamera

logger = logging.getLogger(__name__)

PYLON_PRODUCERS = ("ProducerGEV.cti", "ProducerU3V.cti")


class BaslerCamera(HarvesterCamera):
    """Basler camera using Harvesters GenICam interface.

    Automatically locates Basler Pylon GenTL producer (.cti file).
    Requires Pylon SDK. Supports USB3 and GigE cameras.

    Args:
        cti_file: Path to Basler Pylon GenTL producer. If None, searches
                  common installation paths.
        serial_number: Camera serial number to select specific device

    Discovery order:
        1. Explicit cti_file parameter
        2. GENICAM_GENTL64_PATH environment variable
        3. Platform-specific installation paths
    """

    def __init__(self, cti_file: str | None = None, serial_number: str | None = None):
        """Initialize Basler camera with Pylon GenTL.

        Args:
            cti_file: Path to CTI file. If None, searches GENICAM_GENTL64_PATH then platform paths
            serial_number: Camera serial number for device selection
        """
        cti_file_resolved: str | list[str] | None = cti_file
        if cti_file_resolved is None:
            # Try GENICAM_GENTL64_PATH first (user-configured)
            gentl_path = os.environ.get("GENICAM_GENTL64_PATH")
            if gentl_path:
                logger.info(f"Using GENICAM_GENTL64_PATH: {gentl_path}")
                cti_file_resolved = HarvesterCamera._parse_gentl_path(gentl_path)

            # Fall back to platform-specific Pylon SDK paths
            if not cti_file_resolved:
                cti_file_resolved = self._find_basler_cti()
                if cti_file_resolved:
                    if isinstance(cti_file_resolved, list):
                        logger.info(f"Found Basler CTI files: {', '.join(cti_file_resolved)}")
                    else:
                        logger.info(f"Found Basler CTI: {cti_file_resolved}")
                else:
                    logger.warning(
                        "Basler Pylon CTI not found. Please install Pylon SDK or set GENICAM_GENTL64_PATH."
                    )

        super().__init__(cti_file=cti_file_resolved, serial_number=serial_number)

    @staticmethod
    def _find_basler_cti() -> list[str] | None:
        """Search for Basler Pylon CTI files in platform-specific SDK installation paths.

        Basler cameras require multiple CTI producers for different interfaces
        (GigE, USB3). This method finds all available producers.

        Returns:
            List of CTI file paths if found, None otherwise
        """
        system = platform.system()

        if system == "Windows":
            for version in ["7", "6", "5"]:
                base = rf"C:\Program Files\Basler\pylon {version}\Runtime\x64"
                if os.path.isdir(base):
                    found = []
                    for producer in PYLON_PRODUCERS:
                        path = os.path.join(base, producer)
                        if os.path.exists(path):
                            found.append(path)

                    if found:
                        return found

        elif system == "Linux":
            # Check standard pylon (symlink to latest)
            bases = [
                "/opt/pylon/lib64/gentlproducer/gtl",
                "/opt/pylon5/lib64/gentlproducer/gtl",
            ]
            for base in bases:
                if os.path.isdir(base):
                    found = []
                    for producer in PYLON_PRODUCERS:
                        path = os.path.join(base, producer)
                        if os.path.exists(path):
                            found.append(path)
                    if found:
                        return found

        elif system == "Darwin":
            base = "/Library/Frameworks/pylon.framework/Libraries/gentlproducer/gtl"
            if os.path.isdir(base):
                found = []
                for producer in PYLON_PRODUCERS:
                    path = os.path.join(base, producer)
                    if os.path.exists(path):
                        found.append(path)
                if found:
                    return found

        return None
