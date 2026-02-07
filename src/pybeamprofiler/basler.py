"""Basler camera interface using Harvesters/GenICam."""

import logging
import os
import platform

from .gen_camera import HarvesterCamera

logger = logging.getLogger(__name__)


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
        """Initialize Basler camera with Pylon GenTL."""
        if cti_file is None:
            cti_file = self._find_basler_cti()
            if cti_file:
                logger.info(f"Found Basler CTI: {cti_file}")
            else:
                logger.warning(
                    "Basler Pylon CTI not found. Please install Pylon SDK or specify cti_file path."
                )

        super().__init__(cti_file=cti_file, serial_number=serial_number)

    @staticmethod
    def _find_basler_cti() -> str | list[str] | None:
        """Search for Basler Pylon CTI file in platform-specific paths.

        Returns:
            Path(s) to CTI file if found, None otherwise
        """
        if os.environ.get("GENICAM_GENTL64_PATH"):
            return None

        system = platform.system()

        if system == "Windows":
            for version in ["7", "6", "5"]:
                base = rf"C:\Program Files\Basler\pylon {version}\Runtime\x64"
                if os.path.isdir(base):
                    found = []
                    for producer in [
                        "ProducerGEV.cti",
                        "ProducerU3V.cti",
                    ]:
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
                    for producer in ["ProducerGEV.cti", "ProducerU3V.cti"]:
                        path = os.path.join(base, producer)
                        if os.path.exists(path):
                            found.append(path)
                    if found:
                        return found

        elif system == "Darwin":
            base = "/Library/Frameworks/pylon.framework/Libraries/gentlproducer/gtl"
            if os.path.isdir(base):
                found = []
                for producer in ["ProducerGEV.cti", "ProducerU3V.cti"]:
                    path = os.path.join(base, producer)
                    if os.path.exists(path):
                        found.append(path)
                if found:
                    return found

        return None
