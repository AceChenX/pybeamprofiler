"""Shared pytest fixtures for pybeamprofiler tests."""

import os
import sys
from collections.abc import Iterator

import numpy as np
import pytest
from PIL import Image

from pybeamprofiler import BeamProfiler, cti

# Disable browser auto-opening during tests
os.environ["PYBEAMPROFILER_NO_BROWSER"] = "1"

# The real vendor search tables, captured before anything blanks them.
REAL_VENDOR_DIRS = cti._VENDOR_DIRS

# The GenICam bindings ship as binary wheels and are not available on every
# platform the project supports -- notably Intel Macs, where no genicam after
# 1.3.0 has a wheel and pyproject leaves it out. Tests that genuinely need the
# real enums skip rather than fail there.
try:  # pragma: no cover - depends on what is installed
    import genicam.genapi  # noqa: F401

    HAS_GENICAM = True
except ImportError:  # pragma: no cover
    HAS_GENICAM = False

requires_genicam = pytest.mark.skipif(
    not HAS_GENICAM, reason="genicam bindings are not installed on this platform"
)

try:  # pragma: no cover - depends on what is installed
    import harvesters.core  # noqa: F401

    HAS_HARVESTERS = True
except ImportError:  # pragma: no cover
    HAS_HARVESTERS = False

requires_harvesters = pytest.mark.skipif(
    not HAS_HARVESTERS, reason="harvesters is not importable on this platform"
)


@pytest.fixture(autouse=True)
def _no_installed_sdk(request, monkeypatch):
    """Hide any camera SDK installed on the developer's machine.

    Without this the suite quietly depends on what happens to be installed:
    a test that forgot to patch discovery passes on a laptop with Pylon in
    /Library/Frameworks and fails on CI, where nothing is installed. Blanking
    the search tables by default makes every machine look like CI, so that
    class of bug shows up locally instead of in the pipeline.

    Tests that genuinely exercise the tables opt out with
    ``@pytest.mark.real_cti``.

    ``GENICAM_GENTL64_PATH`` is removed for every test, opted out or not:
    discovery reads it too, and a developer's shell setting would otherwise
    leak their producers into the suite exactly as an installed SDK would.
    """
    monkeypatch.delenv("GENICAM_GENTL64_PATH", raising=False)
    if request.node.get_closest_marker("real_cti"):
        return
    monkeypatch.setattr(
        cti,
        "_VENDOR_DIRS",
        {system: dict.fromkeys(vendors, ()) for system, vendors in REAL_VENDOR_DIRS.items()},
    )


@pytest.fixture(autouse=True)
def _fresh_shared_harvester() -> Iterator[None]:
    """Start and end every test with no process-wide Harvester.

    ``gen_camera`` keeps one Harvester for the whole process, because a
    GenTL producer can only be initialised once. A test that opens a camera
    on a mocked Harvester and never closes it would otherwise hand that mock
    to every later test.
    """
    from pybeamprofiler import gen_camera

    gen_camera._reset_shared_harvester()
    yield
    gen_camera._reset_shared_harvester()


@pytest.fixture
def simulated_image() -> np.ndarray:
    """Generate a synthetic Gaussian beam image for testing."""
    size = 500
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    X, Y = np.meshgrid(x, y)

    x0, y0 = size / 2, size / 2
    sigma = 50
    amplitude = 250
    beam = amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
    beam += 10
    beam = np.clip(beam, 0, 255).astype(np.uint8)

    return beam


@pytest.fixture
def beam_profiler() -> Iterator[BeamProfiler]:
    """Create a BeamProfiler instance with simulated camera."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None
    yield bp
    bp.camera.close()


@pytest.fixture
def test_image_file(tmp_path, simulated_image) -> str:
    """Create a temporary test image file."""
    img_path = tmp_path / "test_beam.png"
    Image.fromarray(simulated_image).save(img_path)
    return str(img_path)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """On GitHub Actions, turn each failure into an annotation.

    A job's log can only be downloaded with admin rights on the repository,
    so a failure that only happens on CI is otherwise invisible to everyone
    else. Annotations appear on the pull request and in the public checks API.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true" or not report.failed or not report.location:
        return
    path, line, _ = report.location
    crash = getattr(report.longrepr, "reprcrash", None)
    message = getattr(crash, "message", None) or str(report.longrepr).splitlines()[-1]
    text = f"{report.nodeid} [{report.when}]: {message}"
    # A workflow command is one line, so %, CR and LF are escaped.
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    # sys.__stdout__ rather than print(): pytest may still be capturing.
    stream = sys.__stdout__
    if stream is not None:
        stream.write(f"\n::error file={path},line={(line or 0) + 1}::{text}\n")
        stream.flush()
