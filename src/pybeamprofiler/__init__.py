"""pybeamprofiler — Laser beam profiler with Gaussian fitting."""

from typing import TYPE_CHECKING

from .basler import BaslerCamera
from .beamprofiler import BeamProfiler
from .camera import Camera
from .discovery import (
    CameraOption,
    discover_cameras,
    find_cti_files,
    list_cameras,
    open_camera,
    print_camera_info,
)
from .flir import FlirCamera
from .simulated import SimulatedCamera

if TYPE_CHECKING:
    from .dash_app import create_app

__version__ = "0.3.0"


def __getattr__(name: str) -> object:
    """Import the GUI only when it is asked for (PEP 562).

    ``create_app`` brings in Dash, Flask and the component libraries, a
    quarter of the package's import time (490 -> 366 ms), and a script that
    only analyses frames never needs them.
    """
    if name == "create_app":
        from .dash_app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Camera",
    "SimulatedCamera",
    "FlirCamera",
    "BaslerCamera",
    "BeamProfiler",
    "create_app",
    "list_cameras",
    "print_camera_info",
    "find_cti_files",
    "discover_cameras",
    "open_camera",
    "CameraOption",
    "__version__",
]
