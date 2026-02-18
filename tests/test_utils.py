"""Tests for utility functions."""

import sys
from types import ModuleType
from unittest.mock import Mock, patch

from pybeamprofiler import utils


def _mock_harvesters_core() -> tuple[ModuleType, Mock]:
    """Return a fake harvesters.core module and its Harvester class mock."""
    mock_harvester_class = Mock()
    fake_core = ModuleType("harvesters.core")
    fake_core.Harvester = mock_harvester_class  # type: ignore[attr-defined]
    return fake_core, mock_harvester_class


class TestFindCtiFiles:
    """Test CTI file discovery."""

    @patch("pybeamprofiler.utils.platform.system")
    @patch("pybeamprofiler.utils.os.path.exists")
    @patch("pybeamprofiler.utils.os.path.realpath")
    @patch("pybeamprofiler.utils.os.listdir")
    def test_find_cti_windows(self, mock_listdir, mock_realpath, mock_exists, mock_system):
        """Test CTI file finding on Windows."""
        mock_system.return_value = "Windows"
        mock_exists.return_value = True
        mock_realpath.side_effect = lambda x: x  # Return path as-is
        mock_listdir.return_value = ["FLIR_GenTL.cti", "other_file.txt"]

        cti_files = utils.find_cti_files()

        assert len(cti_files) > 0
        assert any("FLIR_GenTL.cti" in f for f in cti_files)

    @patch("pybeamprofiler.utils.platform.system")
    @patch("pybeamprofiler.utils.os.path.exists")
    @patch("pybeamprofiler.utils.os.path.realpath")
    @patch("pybeamprofiler.utils.os.listdir")
    def test_find_cti_linux(self, mock_listdir, mock_realpath, mock_exists, mock_system):
        """Test CTI file finding on Linux."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_realpath.side_effect = lambda x: x  # Return path as-is
        mock_listdir.return_value = ["FLIR_GenTL_v140.cti"]

        cti_files = utils.find_cti_files()

        assert len(cti_files) > 0
        assert any("FLIR_GenTL_v140.cti" in f for f in cti_files)

    @patch("pybeamprofiler.utils.platform.system")
    @patch("pybeamprofiler.utils.os.path.exists")
    @patch("pybeamprofiler.utils.os.path.realpath")
    @patch("pybeamprofiler.utils.os.listdir")
    def test_find_cti_macos(self, mock_listdir, mock_realpath, mock_exists, mock_system):
        """Test CTI file finding on macOS."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        mock_realpath.side_effect = lambda x: x  # Return path as-is
        mock_listdir.return_value = ["FLIR_GenTL.cti", "ProducerGEV.cti"]

        cti_files = utils.find_cti_files()

        assert len(cti_files) > 0
        # Multiple search paths may find same files
        assert "FLIR_GenTL.cti" in str(cti_files)
        assert "ProducerGEV.cti" in str(cti_files)

    @patch("pybeamprofiler.utils.platform.system")
    @patch("pybeamprofiler.utils.os.path.exists")
    def test_find_cti_no_paths_exist(self, mock_exists, mock_system):
        """Test CTI file finding when no paths exist."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = False

        cti_files = utils.find_cti_files()

        assert cti_files == []


class TestListCameras:
    """Test camera listing functionality."""

    def test_list_cameras_with_cti(self):
        """Test listing cameras with specific CTI file."""
        fake_core, mock_harvester_class = _mock_harvesters_core()
        mock_h = Mock()
        mock_harvester_class.return_value = mock_h

        mock_device = Mock()
        mock_device.vendor = "Test Vendor"
        mock_device.model = "Test Model"
        mock_device.serial_number = "12345"
        mock_device.id_ = "device_id_123"

        mock_h.device_info_list = [mock_device]

        with patch.dict(sys.modules, {"harvesters.core": fake_core}):
            with patch("pybeamprofiler.utils.os.path.exists", return_value=True):
                cameras = utils.list_cameras("/path/to/test.cti")

        assert len(cameras) == 1
        assert cameras[0]["vendor"] == "Test Vendor"
        assert cameras[0]["model"] == "Test Model"
        assert cameras[0]["serial_number"] == "12345"
        assert cameras[0]["id"] == "device_id_123"
        assert cameras[0]["index"] == 0

    def test_list_cameras_cti_not_found(self):
        """Test listing cameras when CTI file doesn't exist."""
        fake_core, mock_harvester_class = _mock_harvesters_core()
        mock_h = Mock()
        mock_harvester_class.return_value = mock_h

        with patch.dict(sys.modules, {"harvesters.core": fake_core}):
            with patch("pybeamprofiler.utils.os.path.exists", return_value=False):
                cameras = utils.list_cameras("/nonexistent/path.cti")

        assert cameras == []

    @patch("pybeamprofiler.utils.find_cti_files")
    def test_list_cameras_no_cti_files(self, mock_find_cti):
        """Test listing cameras when no CTI files found."""
        fake_core, mock_harvester_class = _mock_harvesters_core()
        mock_h = Mock()
        mock_harvester_class.return_value = mock_h
        mock_find_cti.return_value = []

        with patch.dict(sys.modules, {"harvesters.core": fake_core}):
            cameras = utils.list_cameras()

        assert cameras == []

    @patch("pybeamprofiler.utils.find_cti_files")
    def test_list_cameras_multiple_devices(self, mock_find_cti):
        """Test listing multiple cameras."""
        fake_core, mock_harvester_class = _mock_harvesters_core()
        mock_h = Mock()
        mock_harvester_class.return_value = mock_h
        mock_find_cti.return_value = ["/path/to/test.cti"]

        mock_device1 = Mock()
        mock_device1.vendor = "FLIR"
        mock_device1.model = "Camera1"
        mock_device1.serial_number = "11111"
        mock_device1.id_ = "id1"

        mock_device2 = Mock()
        mock_device2.vendor = "Basler"
        mock_device2.model = "Camera2"
        mock_device2.serial_number = "22222"
        mock_device2.id_ = "id2"

        mock_h.device_info_list = [mock_device1, mock_device2]

        with patch.dict(sys.modules, {"harvesters.core": fake_core}):
            cameras = utils.list_cameras()

        assert len(cameras) == 2
        assert cameras[0]["vendor"] == "FLIR"
        assert cameras[1]["vendor"] == "Basler"
        assert cameras[0]["index"] == 0
        assert cameras[1]["index"] == 1

    def test_list_cameras_no_harvesters(self):
        """Test listing cameras when harvesters not installed."""
        import sys

        # Mock the import to fail
        with patch.dict(sys.modules, {"harvesters.core": None}):
            with patch("pybeamprofiler.utils.Harvester", side_effect=ImportError, create=True):
                cameras = utils.list_cameras()
                assert cameras == []


class TestPrintCameraInfo:
    """Test camera info printing."""

    @patch("pybeamprofiler.utils.list_cameras")
    @patch("pybeamprofiler.utils.logger.info")
    def test_print_camera_info_no_cameras(self, mock_logger, mock_list):
        """Test printing when no cameras found."""
        mock_list.return_value = []

        utils.print_camera_info()

        # Check that helpful message is logged
        calls = [str(call) for call in mock_logger.call_args_list]
        assert any("No cameras found" in str(call) for call in calls)
        assert any("Camera is connected" in str(call) for call in calls)

    @patch("pybeamprofiler.utils.list_cameras")
    @patch("pybeamprofiler.utils.logger.info")
    def test_print_camera_info_single_camera(self, mock_logger, mock_list):
        """Test printing info for single camera."""
        mock_list.return_value = [
            {
                "vendor": "FLIR",
                "model": "BFS-U3-123S6M",
                "serial_number": "12345678",
                "id": "device_id",
                "index": 0,
            }
        ]

        utils.print_camera_info()

        # Check that camera info is logged
        calls = [str(call) for call in mock_logger.call_args_list]
        assert any("Found 1 camera" in str(call) for call in calls)
        assert any("FLIR" in str(call) for call in calls)
        assert any("BFS-U3-123S6M" in str(call) for call in calls)

    @patch("pybeamprofiler.utils.list_cameras")
    @patch("pybeamprofiler.utils.logger.info")
    def test_print_camera_info_multiple_cameras(self, mock_logger, mock_list):
        """Test printing info for multiple cameras."""
        mock_list.return_value = [
            {
                "vendor": "FLIR",
                "model": "Camera1",
                "serial_number": "11111",
                "id": "id1",
                "index": 0,
            },
            {
                "vendor": "Basler",
                "model": "Camera2",
                "serial_number": "22222",
                "id": "id2",
                "index": 1,
            },
        ]

        utils.print_camera_info("/path/to/test.cti")

        # Check that both cameras are logged
        calls = [str(call) for call in mock_logger.call_args_list]
        assert any("Found 2 camera" in str(call) for call in calls)  # Matches "camera(s)"
        assert any("FLIR" in str(call) for call in calls)
        assert any("Basler" in str(call) for call in calls)
