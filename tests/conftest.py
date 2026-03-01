"""Shared pytest fixtures for pybeamprofiler tests."""

import os

import numpy as np
import pytest
from PIL import Image

# Disable browser auto-opening during tests
os.environ["PYBEAMPROFILER_NO_BROWSER"] = "1"


@pytest.fixture
def simulated_image():
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
def beam_profiler():
    """Create a BeamProfiler instance with simulated camera."""
    from pybeamprofiler import BeamProfiler

    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None
    yield bp
    bp.camera.close()


@pytest.fixture
def test_image_file(tmp_path, simulated_image):
    """Create a temporary test image file."""
    img_path = tmp_path / "test_beam.png"
    Image.fromarray(simulated_image).save(img_path)
    return str(img_path)
