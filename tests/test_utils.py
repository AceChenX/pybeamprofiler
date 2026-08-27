"""Tests for utility functions."""

import sys
from types import ModuleType
from unittest.mock import Mock, patch

from pybeamprofiler import utils


def _mock_harvesters_core() -> tuple[ModuleType, Mock]:
    """Return a fake harvesters.core module and its Harvester class mock."""
    mock_harvester_class = Mock()
    fake_core = ModuleType("harvesters.core")
    fake_core.Harvester = mock_harvester_class  # type: ignore
    return fake_core, mock_harvester_class


class TestFindCtiFiles:
    """Test CTI file discovery."""


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

        # Ensure both the parent package and the core submodule are present in sys.modules
        fake_parent = ModuleType("harvesters")
        fake_parent.core = fake_core  # type: ignore

        with patch.dict(sys.modules, {"harvesters": fake_parent, "harvesters.core": fake_core}):
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
        # Mock the import to fail
        with patch.dict(sys.modules, {"harvesters.core": None}):
            with patch("pybeamprofiler.utils.Harvester", side_effect=ImportError, create=True):
                cameras = utils.list_cameras()
                assert cameras == []


class TestPrintCameraInfo:
    """Test camera info printing."""

    @patch("pybeamprofiler.utils.list_cameras")
    def test_print_camera_info_no_cameras(self, mock_list, capsys):
        """Test printing when no cameras found."""
        mock_list.return_value = []

        utils.print_camera_info()

        output = capsys.readouterr().out
        assert "No cameras found" in output
        assert "Camera is connected" in output

    @patch("pybeamprofiler.utils.list_cameras")
    def test_print_camera_info_single_camera(self, mock_list, capsys):
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

        output = capsys.readouterr().out
        assert "Found 1 camera" in output
        assert "FLIR" in output
        assert "BFS-U3-123S6M" in output

    @patch("pybeamprofiler.utils.list_cameras")
    def test_print_camera_info_multiple_cameras(self, mock_list, capsys):
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

        output = capsys.readouterr().out
        assert "Found 2 camera" in output
        assert "FLIR" in output
        assert "Basler" in output


class TestFindCtiEdgeCases:
    """Test edge cases in CTI file discovery."""


class TestListCamerasEdgeCases:
    """Test edge cases in list_cameras."""

    @patch("pybeamprofiler.utils.find_cti_files")
    def test_list_cameras_add_file_exception(self, mock_find_cti):
        """Test that exceptions from add_file are handled."""
        fake_core, mock_harvester_class = _mock_harvesters_core()
        mock_h = Mock()
        mock_harvester_class.return_value = mock_h
        mock_find_cti.return_value = ["/path/to/bad.cti"]
        mock_h.add_file.side_effect = Exception("bad file")
        mock_h.device_info_list = []

        fake_parent = ModuleType("harvesters")
        fake_parent.core = fake_core  # type: ignore

        with patch.dict(sys.modules, {"harvesters": fake_parent, "harvesters.core": fake_core}):
            cameras = utils.list_cameras()

        assert cameras == []

    @patch("pybeamprofiler.utils.find_cti_files")
    def test_list_cameras_update_exception(self, mock_find_cti):
        """Test that exceptions from h.update() are handled."""
        fake_core, mock_harvester_class = _mock_harvesters_core()
        mock_h = Mock()
        mock_harvester_class.return_value = mock_h
        mock_find_cti.return_value = ["/path/to/test.cti"]
        mock_h.update.side_effect = Exception("update failed")

        fake_parent = ModuleType("harvesters")
        fake_parent.core = fake_core  # type: ignore

        with patch.dict(sys.modules, {"harvesters": fake_parent, "harvesters.core": fake_core}):
            cameras = utils.list_cameras()

        assert cameras == []
