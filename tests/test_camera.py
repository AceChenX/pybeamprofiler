"""Tests for camera interfaces and control."""

import numpy as np

from pybeamprofiler import BeamProfiler, SimulatedCamera


class TestSimulatedCamera:
    """Test simulated camera functionality."""

    def test_initialization(self):
        """Test camera initialization with correct dimensions."""
        cam = SimulatedCamera()
        cam.open()
        assert cam.width == 1024
        assert cam.height == 1024
        assert cam.pixel_size == 5.0
        cam.close()

    def test_image_acquisition(self):
        """Test image acquisition returns valid data."""
        cam = SimulatedCamera()
        cam.open()
        cam.start_acquisition()

        img = cam.get_image()
        assert isinstance(img, np.ndarray)
        assert img.shape == (cam.height, cam.width)
        assert img.dtype == np.uint8
        assert np.max(img) > 0

        cam.stop_acquisition()
        cam.close()

    def test_exposure_control(self):
        """Test exposure time setting."""
        cam = SimulatedCamera()
        cam.open()

        cam.set_exposure(0.05)
        assert cam.exposure_time == 0.05

        cam.close()

    def test_gain_control(self):
        """Test gain setting."""
        cam = SimulatedCamera()
        cam.open()

        cam.set_gain(2.0)
        assert cam.gain == 2.0

        cam.close()

    def test_none_exposure_handling(self):
        """Test that None exposure defaults to 0.01s."""
        cam = SimulatedCamera()
        cam.open()
        cam.set_exposure(None)
        assert cam.exposure_time == 0.01
        cam.close()

    def test_exposure_affects_amplitude(self):
        """Test that exposure time affects signal amplitude."""
        cam = SimulatedCamera()
        cam.open()

        cam.set_exposure(0.001)
        cam.start_acquisition()
        img1 = cam.get_image()
        cam.stop_acquisition()

        cam.set_exposure(0.1)
        cam.start_acquisition()
        img2 = cam.get_image()
        cam.stop_acquisition()

        assert np.max(img2) > np.max(img1)
        cam.close()


class TestCameraIntegration:
    """Test camera integration with BeamProfiler."""

    def test_camera_type_selection(self):
        """Test camera type string parsing."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        assert isinstance(bp.camera, SimulatedCamera)
        bp.camera.close()

    def test_invalid_camera_fallback(self):
        """Test fallback to simulated for invalid camera type."""
        bp = BeamProfiler(camera="invalid_type")
        assert bp.camera is not None
        assert isinstance(bp.camera, SimulatedCamera)
        bp.camera.close()

    def test_camera_delegation(self):
        """Test that camera methods are accessible via BeamProfiler."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        assert hasattr(bp, "start_acquisition")
        assert hasattr(bp, "set_exposure")
        assert hasattr(bp, "set_gain")

        bp.set_exposure(0.05)
        assert bp.camera.exposure_time == 0.05

        bp.camera.close()

    def test_camera_hardware_imports(self):
        """Test that hardware camera classes can be imported."""
        from pybeamprofiler.basler import BaslerCamera
        from pybeamprofiler.flir import FlirCamera
        from pybeamprofiler.gen_camera import HarvesterCamera

        assert FlirCamera is not None
        assert BaslerCamera is not None
        assert HarvesterCamera is not None
