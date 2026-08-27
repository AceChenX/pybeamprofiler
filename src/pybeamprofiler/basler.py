"""Basler camera interface using Harvesters/GenICam."""

import logging
import os

from .cti import PYLON, cti_files_for
from .gen_camera import HarvesterCamera

logger = logging.getLogger(__name__)

# The producers Pylon ships today. Kept for reference and for callers that
# want to name one explicitly; discovery scans directories instead.
PYLON_PRODUCERS = ("ProducerGEV.cti", "ProducerU3V.cti")


class BaslerCamera(HarvesterCamera):
    """Basler camera using Harvesters GenICam interface.

    Automatically locates Basler Pylon GenTL producer (``.cti`` file).
    Requires Pylon SDK.  Supports USB3 and GigE cameras.

    CTI discovery order: explicit ``cti_file`` parameter →
    platform-specific Pylon SDK installation paths →
    ``GENICAM_GENTL64_PATH`` environment variable (fallback).
    """

    def __init__(self, cti_file: str | None = None, serial_number: str | None = None) -> None:
        """Initialize Basler camera with Pylon GenTL.

        Args:
            cti_file: Path to Basler Pylon GenTL producer.  If ``None``,
                searches platform SDK paths then ``GENICAM_GENTL64_PATH``.
            serial_number: Camera serial number for device selection.
        """
        cti_file_resolved: str | list[str] | None = cti_file
        if cti_file_resolved is None:
            cti_file_resolved = self._find_basler_cti()
            if cti_file_resolved:
                if isinstance(cti_file_resolved, list):
                    logger.info(f"Found Basler CTI files: {', '.join(cti_file_resolved)}")
                else:
                    logger.info(f"Found Basler CTI: {cti_file_resolved}")
            else:
                gentl_path = os.environ.get("GENICAM_GENTL64_PATH")
                if gentl_path:
                    logger.info(
                        f"Basler CTI not found, falling back to GENICAM_GENTL64_PATH: {gentl_path}"
                    )
                    cti_file_resolved = HarvesterCamera._parse_gentl_path(gentl_path)
                if not cti_file_resolved:
                    logger.warning(
                        "Basler Pylon CTI not found. "
                        "Please install Pylon SDK or set GENICAM_GENTL64_PATH."
                    )

        super().__init__(cti_file=cti_file_resolved, serial_number=serial_number)

    @staticmethod
    def _find_basler_cti() -> list[str] | None:
        """Return every Pylon GenTL producer found, if any.

        Basler splits its producers by transport layer (``ProducerGEV.cti``
        for GigE, ``ProducerU3V.cti`` for USB3) and which one a given camera
        speaks isn't known until it is opened, so all of them are loaded.
        Discovery scans the whole directory rather than matching a fixed name
        list, so a future SDK that adds a producer works without a code change.

        Returns:
            List of ``.cti`` paths, or ``None`` when Pylon isn't installed.
        """
        return cti_files_for(PYLON) or None
