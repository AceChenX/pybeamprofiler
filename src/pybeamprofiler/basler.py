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
        cti_file: Path to Basler Pylon GenTL producer. If None, uses
                  GENICAM_GENTL64_PATH or searches common paths (prefers USB3).
        serial_number: Camera serial number to select specific device

    Discovery order:
        1. Explicit cti_file parameter
        2. GENICAM_GENTL64_PATH environment variable
        3. Platform-specific installation paths (USB3 preferred)
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
    def _find_basler_cti() -> str | None:
        """Search for Basler Pylon CTI file in platform-specific paths.

        Prefers USB3 interface over GigE.

        Returns:
            Path to CTI file if found, None otherwise
        """
        system = platform.system()
        search_paths = []

        if system == "Windows":
            for version in ["7", "6", "5"]:
                base = rf"C:\Program Files\Basler\pylon {version}\Runtime\x64"
                search_paths.extend(
                    [
                        os.path.join(base, "ProducerU3V.cti"),
                        os.path.join(base, "ProducerGEV.cti"),
                    ]
                )
        elif system == "Linux":
            search_paths = [
                "/opt/pylon/lib64/gentlproducer/gtl/ProducerU3V.cti",
                "/opt/pylon/lib64/gentlproducer/gtl/ProducerGEV.cti",
                "/opt/pylon5/lib64/gentlproducer/gtl/ProducerU3V.cti",
                "/opt/pylon5/lib64/gentlproducer/gtl/ProducerGEV.cti",
            ]
        elif system == "Darwin":
            search_paths = [
                "/Library/Frameworks/pylon.framework/Libraries/ProducerU3V.cti",
                "/Library/Frameworks/pylon.framework/Libraries/ProducerGEV.cti",
            ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        return None
