"""Backwards-compatible alias for :mod:`pybeamprofiler.discovery`.

The discovery helpers used to live here under a grab-bag name. They moved to
``discovery.py``, which says what they actually do; this module stays so that
``from pybeamprofiler.utils import list_cameras`` keeps working.
"""

from __future__ import annotations

from .discovery import find_cti_files, list_cameras, print_camera_info

__all__ = ["find_cti_files", "list_cameras", "print_camera_info"]
