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
        """Initialize FLIR camera with Spinnaker GenTL.

        Args:
            cti_file: Path to CTI file. If None, searches GENICAM_GENTL64_PATH then platform paths
            serial_number: Camera serial number for device selection
        """
        if cti_file is None:
            # Try GENICAM_GENTL64_PATH first (user-configured)
            gentl_path = os.environ.get("GENICAM_GENTL64_PATH")
            if gentl_path:
                logger.info(f"Using GENICAM_GENTL64_PATH: {gentl_path}")
                cti_file = HarvesterCamera._parse_gentl_path(gentl_path)

            # Fall back to platform-specific Spinnaker SDK paths
            if not cti_file:
                cti_file = self._find_flir_cti()
                if cti_file:
                    logger.info(f"Found FLIR CTI: {cti_file}")
                else:
                    logger.warning(
                        "FLIR Spinnaker CTI not found. "
                        "Please install Spinnaker SDK or set GENICAM_GENTL64_PATH."
                    )

        super().__init__(cti_file=cti_file, serial_number=serial_number)

    @staticmethod
    def _find_flir_cti() -> str | None:
        """Search for FLIR Spinnaker CTI in platform-specific SDK installation paths.

        Searches common Spinnaker SDK installation locations by platform.

        Returns:
            Path to first found CTI file, or None if not found
        """
        system = platform.system()
        search_paths = []

        if system == "Windows":
            base = r"C:\Program Files\Teledyne\Spinnaker"
            search_paths = [
                os.path.join(base, "cti64", "vs2015", "Spinnaker_v140.cti"),
                os.path.join(base, "cti64", "vs2017", "Spinnaker_v141.cti"),
            ]
        elif system == "Linux":
            search_paths = [
                "/opt/spinnaker/lib/flir-gentl/Spinnaker_GenTL.cti",
                "/opt/spinnaker/lib/flir-gentl/FLIR_GenTL_v140.cti",
            ]
        elif system == "Darwin":
            search_paths = [
                "/usr/local/lib/spinnaker-gentl/Spinnaker_GenTL.cti",
                "/usr/local/lib/spinnaker-gentl/FLIR_GenTL_v140.cti",
            ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        return None
