"""FLIR camera interface using Harvesters/GenICam."""

import logging
import os
import platform

from .gen_camera import HarvesterCamera

logger = logging.getLogger(__name__)


class FlirCamera(HarvesterCamera):
    """FLIR camera using Harvesters GenICam interface.

    Automatically locates FLIR Spinnaker GenTL producer (.cti file).
    Requires Spinnaker SDK to be installed.

    Args:
        cti_file: Path to FLIR Spinnaker GenTL producer. If None, uses
                  GENICAM_GENTL64_PATH or searches common paths.
        serial_number: Camera serial number to select specific device

    Discovery order:
        1. Explicit cti_file parameter
        2. GENICAM_GENTL64_PATH environment variable
        3. Platform-specific installation paths
    """

    def __init__(self, cti_file: str | None = None, serial_number: str | None = None):
        """Initialize FLIR camera with Spinnaker GenTL."""
        if cti_file is None:
            cti_file = self._find_flir_cti()
            if cti_file:
                logger.info(f"Found FLIR CTI: {cti_file}")
            else:
                logger.warning(
                    "FLIR Spinnaker CTI not found. "
                    "Please install Spinnaker SDK or specify cti_file path."
                )

        super().__init__(cti_file=cti_file, serial_number=serial_number)

    @staticmethod
    def _find_flir_cti() -> str | None:
        """Search for FLIR Spinnaker CTI file in platform-specific paths.

        Returns:
            Path to CTI file if found, None otherwise
        """
        system = platform.system()
        search_paths = []

        if system == "Windows":
            base = r"C:\Program Files\FLIR Systems\Spinnaker"
            search_paths = [
                os.path.join(base, "cti64", "vs2015", "FLIR_GenTL_v140.cti"),
                os.path.join(base, "cti64", "vs2017", "FLIR_GenTL_v141.cti"),
                os.path.join(base, "cti64", "FLIR_GenTL.cti"),
            ]
        elif system == "Linux":
            search_paths = [
                "/opt/spinnaker/lib/flir-gentl/FLIR_GenTL.cti",
                "/usr/lib/flir-gentl/FLIR_GenTL.cti",
            ]
        elif system == "Darwin":
            search_paths = [
                "/usr/local/lib/FLIR_GenTL.cti",
                "/Library/Application Support/FLIR/Spinnaker/lib/FLIR_GenTL.cti",
            ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        return None
