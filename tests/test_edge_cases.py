"""Additional tests for camera error handling and edge cases."""

import numpy as np
import pytest

from pybeamprofiler import BeamProfiler
from pybeamprofiler.simulated import SimulatedCamera


class TestCameraErrorHandling:
    """Test error handling for camera operations."""

    def test_simulated_camera_initialization(self):
        """Test simulated camera can be initialized."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        assert isinstance(bp.camera, SimulatedCamera)
        assert bp.width_pixels > 0
        assert bp.height_pixels > 0
        assert bp.pixel_size > 0

    def test_exposure_time_initialization(self):
        """Test exposure time can be set during initialization."""
        exposure = 0.05
        bp = BeamProfiler(camera="simulated", exposure_time=exposure)
        assert bp.exposure_time == pytest.approx(exposure, rel=0.01)

    def test_exposure_time_setter(self):
        """Test exposure time can be set after initialization."""
        bp = BeamProfiler(camera="simulated")
        new_exposure = 0.025
        bp.exposure_time = new_exposure
        assert bp.exposure_time == pytest.approx(new_exposure, rel=0.01)

    def test_gain_setter(self):
        """Test gain can be set after initialization."""
        bp = BeamProfiler(camera="simulated")
        new_gain = 10.0
        bp.gain = new_gain
        assert bp.gain == pytest.approx(new_gain, rel=0.1)

    def test_invalid_camera_type(self):
        """Test invalid camera type raises error."""
        bp = BeamProfiler(camera="nonexistent")
        # Should fall back to simulated
        assert isinstance(bp.camera, SimulatedCamera)


class TestSimulatedCamera:
    """Test simulated camera functionality."""

    def test_positive_peak_values(self):
        """Test simulated camera always produces positive peaks."""
        cam = SimulatedCamera()
        cam.open()
        cam.start_acquisition()

        # Test multiple images to check consistency
        for _ in range(10):
            img = cam.get_image()
            assert img is not None
            assert np.max(img) > 0  # Peak should be positive
            assert np.min(img) >= 0  # No negative values

        cam.stop_acquisition()
        cam.close()

    def test_exposure_range(self):
        """Test simulated camera exposure range."""
        cam = SimulatedCamera()
        cam.open()

        exposure_min, exposure_max = cam.exposure_range
        assert exposure_min > 0
        assert exposure_max > exposure_min
        assert exposure_min == pytest.approx(0.001, rel=0.1)
        assert exposure_max == pytest.approx(1.0, rel=0.1)

        cam.close()

    def test_gain_range(self):
        """Test simulated camera gain range."""
        cam = SimulatedCamera()
        cam.open()

        gain_min, gain_max = cam.gain_range
        assert gain_min >= 0
        assert gain_max > gain_min

        cam.close()

    def test_image_dimensions(self):
        """Test simulated camera produces correct image dimensions."""
        cam = SimulatedCamera()
        cam.open()
        cam.start_acquisition()

        img = cam.get_image()
        assert img.shape == (cam.height, cam.width)

        cam.stop_acquisition()
        cam.close()


class TestWidthDefinitions:
    """Test different width definition calculations."""

    def test_gaussian_definition(self):
        """Test Gaussian (1/e²) width definition."""
        bp = BeamProfiler(camera="simulated", fit="1d", definition="gaussian")
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        bp.analyze(img)
        assert bp.width_x > 0
        assert bp.width_y > 0
        assert bp.definition == "gaussian"

    def test_fwhm_definition(self):
        """Test FWHM width definition."""
        bp = BeamProfiler(camera="simulated", fit="1d", definition="fwhm")
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        bp.analyze(img)
        assert bp.width_x > 0
        assert bp.width_y > 0
        assert bp.definition == "fwhm"

    def test_d4s_definition(self):
        """Test D4σ (ISO 11146) width definition."""
        bp = BeamProfiler(camera="simulated", fit="1d", definition="d4s")
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        bp.analyze(img)
        assert bp.width_x > 0
        assert bp.width_y > 0
        assert bp.definition == "d4s"

    def test_definition_comparison(self):
        """Test that different definitions give different widths."""
        bp = BeamProfiler(camera="simulated", fit="1d")
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        widths = {}
        for definition in ["gaussian", "fwhm", "d4s"]:
            bp.definition = definition
            bp.analyze(img)
            widths[definition] = bp.width_x

        # All should be positive and different
        assert all(w > 0 for w in widths.values())
        assert len(set(widths.values())) > 1  # Not all the same


class TestFittingMethods:
    """Test different fitting methods."""

    def test_1d_fitting(self):
        """Test 1D projection fitting."""
        bp = BeamProfiler(camera="simulated", fit="1d")
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        assert popt_x is not None
        assert popt_y is not None
        assert len(popt_x) == 4  # amplitude, center, sigma, offset
        assert len(popt_y) == 4
        assert bp.angle_deg == 0  # No rotation for 1D

    def test_2d_fitting(self):
        """Test 2D Gaussian fitting with rotation."""
        bp = BeamProfiler(camera="simulated", fit="2d")
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        assert popt_x is not None
        assert popt_y is not None
        assert 0 <= bp.angle_deg < 180  # Angle should be in valid range

    def test_linecut_fitting(self):
        """Test linecut fitting through peak."""
        bp = BeamProfiler(camera="simulated", fit="linecut")
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        assert popt_x is not None
        assert popt_y is not None
        assert bp.angle_deg == 0  # No rotation for linecut


class TestBeamProfilerIntegration:
    """Integration tests for complete workflows."""

    def test_single_shot_workflow(self):
        """Test complete single-shot acquisition workflow."""
        bp = BeamProfiler(camera="simulated", exposure_time=0.01)

        # Should not raise any exceptions
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert bp.width > 0
        assert bp.width_x > 0
        assert bp.width_y > 0
        assert bp.center_x >= 0
        assert bp.center_y >= 0
        assert bp.peak_value > 0

    def test_multiple_acquisitions(self):
        """Test multiple sequential acquisitions."""
        bp = BeamProfiler(camera="simulated")
        bp.camera.start_acquisition()

        widths = []
        for _ in range(5):
            img = bp.camera.get_image()
            bp.analyze(img)
            widths.append(bp.width)

        bp.camera.stop_acquisition()

        # All widths should be positive
        assert all(w > 0 for w in widths)
        # Widths should be relatively consistent for simulated camera
        mean_width = np.mean(widths)
        assert all(abs(w - mean_width) / mean_width < 0.5 for w in widths)

    def test_camera_cleanup(self):
        """Test camera cleanup works correctly."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        bp.camera.open()
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        bp.camera.close()

        assert img is not None

    def test_fit_caching(self):
        """Test that fit parameters are cached for efficiency."""
        bp = BeamProfiler(camera="simulated", fit="1d")
        bp.camera.start_acquisition()

        # First fit
        img1 = bp.camera.get_image()
        popt_x1, popt_y1 = bp.analyze(img1)

        # Cache should be populated
        assert bp._last_popt_x is not None
        assert bp._last_popt_y is not None

        # Second fit should use cached values as initial guess
        img2 = bp.camera.get_image()
        popt_x2, popt_y2 = bp.analyze(img2)

        assert popt_x2 is not None
        assert popt_y2 is not None

        bp.camera.stop_acquisition()
