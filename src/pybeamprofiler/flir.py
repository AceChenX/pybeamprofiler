"""FLIR camera interface using Harvesters/GenICam."""

import logging
import os
import platform

from .gen_camera import HarvesterCamera

logger = logging.getLogger(__name__)


class FlirCamera(HarvesterCamera):
    """FLIR camera using Harvesters GenICam interface.

    Automatically locates FLIR Spinnaker GenTL producer (``.cti`` file).
    Requires Spinnaker SDK to be installed.

    CTI discovery order: explicit ``cti_file`` parameter →
    ``GENICAM_GENTL64_PATH`` environment variable → platform-specific
    Spinnaker SDK installation paths.
    """

    def __init__(self, cti_file: str | None = None, serial_number: str | None = None) -> None:
        """Initialize FLIR camera with Spinnaker GenTL.

        Args:
            cti_file: Path to FLIR Spinnaker GenTL producer.  If ``None``,
                searches ``GENICAM_GENTL64_PATH`` then platform paths.
            serial_number: Camera serial number for device selection.
        """
        cti_file_resolved: str | list[str] | None = cti_file
        if cti_file_resolved is None:
            gentl_path = os.environ.get("GENICAM_GENTL64_PATH")
            if gentl_path:
                logger.info(f"Using GENICAM_GENTL64_PATH: {gentl_path}")
                cti_file_resolved = HarvesterCamera._parse_gentl_path(gentl_path)

            if not cti_file_resolved:
                cti_file_resolved = self._find_flir_cti()
                if cti_file_resolved:
                    logger.info(f"Found FLIR CTI: {cti_file_resolved}")
                else:
                    logger.warning(
                        "FLIR Spinnaker CTI not found. "
                        "Please install Spinnaker SDK or set GENICAM_GENTL64_PATH."
                    )

        super().__init__(cti_file=cti_file_resolved, serial_number=serial_number)

    @staticmethod
    def _find_flir_cti() -> str | None:
        """Search for FLIR Spinnaker CTI in platform-specific SDK installation paths.

        Searches common Spinnaker SDK installation locations by platform.
        Dynamically scans directories to support any Spinnaker version.

        Returns:
            Path to first found CTI file, or None if not found
        """
        system = platform.system()
        search_dirs = []

        if system == "Windows":
            base = r"C:\Program Files\Teledyne\Spinnaker\cti64"
            if os.path.exists(base):
                try:
                    for subdir in os.listdir(base):
                        dir_path = os.path.join(base, subdir)
                        if os.path.isdir(dir_path):
                            search_dirs.append(dir_path)
                except OSError:
                    pass
        elif system == "Linux":
            search_dirs = ["/opt/spinnaker/lib/flir-gentl"]
        elif system == "Darwin":
            search_dirs = [
                "/usr/local/lib/spinnaker-gentl",
                "/Library/Application Support/FLIR/Spinnaker/lib",
            ]

        for d in search_dirs:
            if os.path.isdir(d):
                try:
                    for f in os.listdir(d):
                        if f.endswith(".cti"):
                            return os.path.join(d, f)
                except OSError:
                    continue

        return None
