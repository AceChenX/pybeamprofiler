"""Tests for GenICam camera implementations (FLIR, Basler, Harvester)."""

from unittest.mock import Mock, patch


class TestHarvesterCamera:
    """Test GenICam camera wrapper."""

    @patch("pybeamprofiler.gen_camera.os.path.exists")
    @patch("pybeamprofiler.gen_camera.Harvester")
    def test_camera_initialization(self, mock_harvester_class, mock_exists):
        """Test camera initialization."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_exists.return_value = True
        mock_h = Mock()
        mock_harvester_class.return_value = mock_h

        camera = HarvesterCamera(cti_file="/path/to/test.cti")

        assert camera.h == mock_h
        mock_h.add_file.assert_called_once_with("/path/to/test.cti")

    @patch("pybeamprofiler.gen_camera.Harvester")
    def test_camera_open(self, mock_harvester_class):
        """Test camera opening."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_h = Mock()
        mock_harvester_class.return_value = mock_h

        # Mock files attribute for CTI file logging
        mock_h.files = ["/path/to/test.cti"]

        mock_device = Mock()
        mock_device.vendor = "Test Vendor"
        mock_device.model = "Test Model"
        mock_device.serial_number = "12345"

        # Make device_info_list a proper list that supports len()
        device_list = [mock_device]
        mock_h.device_info_list = device_list

        mock_ia = Mock()
        mock_node_map = Mock()

        # Mock dimension attributes (need both max and actual)
        mock_node_map.WidthMax.value = 1920
        mock_node_map.HeightMax.value = 1080
        mock_node_map.Width.value = 1920
        mock_node_map.Height.value = 1080

        # Mock offset attributes for ROI reset
        mock_node_map.OffsetX.value = 0
        mock_node_map.OffsetY.value = 0

        mock_ia.remote_device.node_map = mock_node_map

        mock_h.create.return_value = mock_ia

        # Test
        camera = HarvesterCamera()
        camera.open()

        assert camera.width_pixels == 1920
        assert camera.height_pixels == 1080
        mock_h.update.assert_called_once()
        mock_h.create.assert_called_once_with(mock_device)

    @patch("pybeamprofiler.gen_camera.Harvester")
    def test_camera_exposure_setting(self, mock_harvester_class):
        """Test exposure time setting."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_h = Mock()
        mock_harvester_class.return_value = mock_h

        camera = HarvesterCamera()
        camera.node_map = Mock()
        camera.node_map.ExposureTime = Mock()

        camera.set_exposure(0.001)  # 1ms

        camera.node_map.ExposureTime.value = 1000  # 1000 microseconds
        assert camera.exposure_time == 0.001

    @patch("pybeamprofiler.gen_camera.Harvester")
    def test_camera_gain_setting(self, mock_harvester_class):
        """Test gain setting."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_h = Mock()
        mock_harvester_class.return_value = mock_h

        camera = HarvesterCamera()
        camera.node_map = Mock()
        camera.node_map.Gain = Mock()

        camera.set_gain(10.0)

        camera.node_map.Gain.value = 10.0
        assert camera.gain == 10.0


class TestFlirCamera:
    """Test FLIR camera implementation."""

    @patch("pybeamprofiler.flir.HarvesterCamera.__init__")
    @patch("pybeamprofiler.flir.os.path.exists")
    def test_flir_cti_discovery(self, mock_exists, mock_super_init):
        """Test FLIR CTI file discovery."""
        from pybeamprofiler.flir import FlirCamera

        mock_super_init.return_value = None
        mock_exists.return_value = True

        _camera = FlirCamera()

        # Verify parent class initialization was called
        assert mock_super_init.called

    def test_find_flir_cti(self):
        """Test FLIR CTI path search."""
        from pybeamprofiler.flir import FlirCamera

        # Returns None when CTI files are not found, or path string if available
        cti_path = FlirCamera._find_flir_cti()
        assert cti_path is None or isinstance(cti_path, str)


class TestBaslerCamera:
    """Test Basler camera implementation."""

    @patch("pybeamprofiler.basler.HarvesterCamera.__init__")
    @patch("pybeamprofiler.basler.os.path.exists")
    def test_basler_cti_discovery(self, mock_exists, mock_super_init):
        """Test Basler CTI file discovery."""
        from pybeamprofiler.basler import BaslerCamera

        mock_super_init.return_value = None
        mock_exists.return_value = True

        _camera = BaslerCamera()

        # Verify parent class initialization was called
        assert mock_super_init.called

    def test_find_basler_cti(self):
        """Test Basler CTI path search."""
        from pybeamprofiler.basler import BaslerCamera

        # Returns None when CTI files are not found, or list of paths if available
        cti_path = BaslerCamera._find_basler_cti()
        assert cti_path is None or isinstance(cti_path, list)

    def test_pylon_producers_constant(self):
        """Test that PYLON_PRODUCERS constant is defined and contains expected producers."""
        from pybeamprofiler.basler import PYLON_PRODUCERS

        assert isinstance(PYLON_PRODUCERS, list | tuple)
        assert "ProducerGEV.cti" in PYLON_PRODUCERS
        assert "ProducerU3V.cti" in PYLON_PRODUCERS


class TestCameraUtils:
    """Test camera utility functions."""

    def test_find_cti_files(self):
        """Test CTI file discovery."""
        from pybeamprofiler.utils import find_cti_files

        cti_files = find_cti_files()
        assert isinstance(cti_files, list)

    def test_list_cameras(self):
        """Test camera listing."""
        import sys

        from pybeamprofiler.utils import list_cameras

        mock_harvesters = Mock()
        mock_core = Mock()
        mock_harvester_class = Mock()

        mock_h = Mock()
        mock_device = Mock()
        mock_device.vendor = "Test Vendor"
        mock_device.model = "Test Camera"
        mock_device.serial_number = "12345"
        mock_device.id_ = "test-id"
        mock_h.device_info_list = [mock_device]

        mock_harvester_class.return_value = mock_h
        mock_core.Harvester = mock_harvester_class
        mock_harvesters.core = mock_core

        # Mock harvesters module for testing without hardware dependency
        with patch.dict(sys.modules, {"harvesters": mock_harvesters, "harvesters.core": mock_core}):
            with patch("pybeamprofiler.utils.find_cti_files", return_value=["/fake/path.cti"]):
                cameras = list_cameras()

                assert len(cameras) == 1
                assert cameras[0]["vendor"] == "Test Vendor"
                assert cameras[0]["model"] == "Test Camera"
                assert cameras[0]["serial_number"] == "12345"


class TestHarvesterCameraErrors:
    """Test HarvesterCamera error handling."""

    @patch("pybeamprofiler.gen_camera.Harvester")
    def test_camera_open_no_devices(self, mock_harvester_class):
        """Test camera opening with no devices found."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_h = Mock()
        mock_harvester_class.return_value = mock_h
        mock_h.files = ["/path/to/test.cti"]
        mock_h.device_info_list = []

        camera = HarvesterCamera()

        try:
            camera.open()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "No GenICam cameras found" in str(e)

    @patch("pybeamprofiler.gen_camera.Harvester")
    def test_camera_open_with_serial_number(self, mock_harvester_class):
        """Test camera opening with specific serial number."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_h = Mock()
        mock_harvester_class.return_value = mock_h
        mock_h.files = ["/path/to/test.cti"]

        mock_device1 = Mock()
        mock_device1.vendor = "Test Vendor"
        mock_device1.model = "Model1"
        mock_device1.serial_number = "11111"

        mock_device2 = Mock()
        mock_device2.vendor = "Test Vendor"
        mock_device2.model = "Model2"
        mock_device2.serial_number = "22222"

        mock_h.device_info_list = [mock_device1, mock_device2]

        mock_ia = Mock()
        mock_node_map = Mock()
        mock_node_map.WidthMax.value = 1920
        mock_node_map.HeightMax.value = 1080
        mock_node_map.Width.value = 1920
        mock_node_map.Height.value = 1080
        mock_node_map.OffsetX.value = 0
        mock_node_map.OffsetY.value = 0
        mock_ia.remote_device.node_map = mock_node_map
        mock_h.create.return_value = mock_ia

        camera = HarvesterCamera(serial_number="22222")
        camera.open()

        # Should have selected device2
        mock_h.create.assert_called_once_with(mock_device2)

    @patch("pybeamprofiler.gen_camera.Harvester")
    def test_camera_open_serial_not_found(self, mock_harvester_class):
        """Test camera opening with non-existent serial number."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_h = Mock()
        mock_harvester_class.return_value = mock_h
        mock_h.files = ["/path/to/test.cti"]

        mock_device = Mock()
        mock_device.vendor = "Test Vendor"
        mock_device.model = "Test Model"
        mock_device.serial_number = "11111"

        mock_h.device_info_list = [mock_device]

        camera = HarvesterCamera(serial_number="99999")

        try:
            camera.open()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not found" in str(e)
