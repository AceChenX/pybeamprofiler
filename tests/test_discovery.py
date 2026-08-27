"""Tests for camera discovery: enumerating devices and opening one."""

import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from pybeamprofiler import discovery


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
            with patch("pybeamprofiler.discovery.os.path.exists", return_value=True):
                cameras = discovery.list_cameras("/path/to/test.cti")

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
            with patch("pybeamprofiler.discovery.os.path.exists", return_value=False):
                cameras = discovery.list_cameras("/nonexistent/path.cti")

        assert cameras == []

    @patch("pybeamprofiler.discovery.find_cti_files")
    def test_list_cameras_no_cti_files(self, mock_find_cti):
        """Test listing cameras when no CTI files found."""
        fake_core, mock_harvester_class = _mock_harvesters_core()
        mock_h = Mock()
        mock_harvester_class.return_value = mock_h
        mock_find_cti.return_value = []

        with patch.dict(sys.modules, {"harvesters.core": fake_core}):
            cameras = discovery.list_cameras()

        assert cameras == []

    @patch("pybeamprofiler.discovery.find_cti_files")
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
            cameras = discovery.list_cameras()

        assert len(cameras) == 2
        assert cameras[0]["vendor"] == "FLIR"
        assert cameras[1]["vendor"] == "Basler"
        assert cameras[0]["index"] == 0
        assert cameras[1]["index"] == 1

    def test_list_cameras_no_harvesters(self):
        """Test listing cameras when harvesters not installed."""
        # Mock the import to fail
        with patch.dict(sys.modules, {"harvesters.core": None}):
            with patch("pybeamprofiler.discovery.Harvester", side_effect=ImportError, create=True):
                cameras = discovery.list_cameras()
                assert cameras == []


class TestPrintCameraInfo:
    """Test camera info printing."""

    @patch("pybeamprofiler.discovery.list_cameras")
    def test_print_camera_info_no_cameras(self, mock_list, capsys):
        """Test printing when no cameras found."""
        mock_list.return_value = []

        discovery.print_camera_info()

        output = capsys.readouterr().out
        assert "No cameras found" in output
        assert "Camera is connected" in output

    @patch("pybeamprofiler.discovery.list_cameras")
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

        discovery.print_camera_info()

        output = capsys.readouterr().out
        assert "Found 1 camera" in output
        assert "FLIR" in output
        assert "BFS-U3-123S6M" in output

    @patch("pybeamprofiler.discovery.list_cameras")
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

        discovery.print_camera_info("/path/to/test.cti")

        output = capsys.readouterr().out
        assert "Found 2 camera" in output
        assert "FLIR" in output
        assert "Basler" in output


class TestFindCtiEdgeCases:
    """Test edge cases in CTI file discovery."""


class TestListCamerasEdgeCases:
    """Test edge cases in list_cameras."""

    @patch("pybeamprofiler.discovery.find_cti_files")
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
            cameras = discovery.list_cameras()

        assert cameras == []

    @patch("pybeamprofiler.discovery.find_cti_files")
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
            cameras = discovery.list_cameras()

        assert cameras == []


# ─── Selectable camera options (what the GUI dropdown drives) ──────────────


class TestCameraOption:
    def test_simulated_option_is_flagged(self):
        assert discovery.SIMULATED_OPTION.is_simulated is True

    def test_real_option_is_not(self):
        option = discovery._describe(
            {"vendor": "FLIR", "model": "BFS-U3-51S5M", "serial_number": "12345678", "index": 0}
        )
        assert option.is_simulated is False

    def test_key_is_built_from_the_serial(self):
        option = discovery._describe(
            {"vendor": "FLIR", "model": "BFS", "serial_number": "12345678", "index": 0}
        )
        assert option.key == "genicam:12345678"

    def test_label_carries_vendor_model_and_serial(self):
        option = discovery._describe(
            {"vendor": "Basler", "model": "acA2440-75um", "serial_number": "40012345", "index": 1}
        )
        assert option.label == "Basler acA2440-75um (40012345)"

    def test_missing_serial_falls_back_to_the_device_id(self):
        """Not every producer reports a serial; the GenTL id still identifies it."""
        option = discovery._describe(
            {"vendor": "V", "model": "M", "serial_number": "", "id": "dev-id-7", "index": 3}
        )
        assert option.key == "genicam:dev-id-7"
        assert option.label == "V M"

    def test_missing_serial_and_id_falls_back_to_the_index(self):
        option = discovery._describe({"vendor": "", "model": "", "index": 2})
        assert option.key == "genicam:index-2"
        assert option.label == "GenICam camera"

    def test_options_are_hashable_and_comparable(self):
        """Frozen dataclass — the GUI stores these in sets and compares them."""
        a = discovery._describe({"vendor": "V", "model": "M", "serial_number": "1", "index": 0})
        b = discovery._describe({"vendor": "V", "model": "M", "serial_number": "1", "index": 0})
        assert a == b
        assert len({a, b}) == 1


class TestDiscoverCameras:
    def test_simulated_is_always_offered(self):
        with patch("pybeamprofiler.discovery.list_cameras", return_value=[]):
            options = discovery.discover_cameras()
        assert options == [discovery.SIMULATED_OPTION]

    def test_simulated_can_be_excluded(self):
        with patch("pybeamprofiler.discovery.list_cameras", return_value=[]):
            assert discovery.discover_cameras(include_simulated=False) == []

    def test_real_cameras_come_first(self):
        devices = [
            {"vendor": "FLIR", "model": "BFS", "serial_number": "111", "id": "a", "index": 0},
            {"vendor": "Basler", "model": "acA", "serial_number": "222", "id": "b", "index": 1},
        ]
        with patch("pybeamprofiler.discovery.list_cameras", return_value=devices):
            options = discovery.discover_cameras()

        assert [o.key for o in options] == ["genicam:111", "genicam:222", "simulated"]

    def test_the_same_camera_seen_through_two_producers_appears_once(self):
        """A Basler USB3 device enumerates through both the GEV and U3V .cti."""
        devices = [
            {"vendor": "Basler", "model": "acA", "serial_number": "222", "id": "u3v", "index": 0},
            {"vendor": "Basler", "model": "acA", "serial_number": "222", "id": "gev", "index": 1},
        ]
        with patch("pybeamprofiler.discovery.list_cameras", return_value=devices):
            options = discovery.discover_cameras()

        assert [o.key for o in options] == ["genicam:222", "simulated"]

    def test_discovery_failure_still_offers_the_simulator(self):
        """Behind a Refresh button, a short list beats a traceback."""
        with patch("pybeamprofiler.discovery.list_cameras", side_effect=RuntimeError("no SDK")):
            options = discovery.discover_cameras()
        assert options == [discovery.SIMULATED_OPTION]

    def test_cti_file_is_forwarded(self):
        with patch("pybeamprofiler.discovery.list_cameras", return_value=[]) as mock_list:
            discovery.discover_cameras(cti_file="/x/y.cti")
        mock_list.assert_called_once_with("/x/y.cti")


class TestFindOption:
    def test_finds_by_key(self):
        options = [discovery.SIMULATED_OPTION]
        assert discovery.find_option("simulated", options) is discovery.SIMULATED_OPTION

    def test_unknown_key_is_none(self):
        assert discovery.find_option("genicam:nope", [discovery.SIMULATED_OPTION]) is None

    def test_blank_key_is_none(self):
        assert discovery.find_option(None, [discovery.SIMULATED_OPTION]) is None
        assert discovery.find_option("", [discovery.SIMULATED_OPTION]) is None


class TestOpenCamera:
    def test_opens_the_simulator(self):
        from pybeamprofiler.simulated import SimulatedCamera

        cam = discovery.open_camera(discovery.SIMULATED_OPTION)
        assert isinstance(cam, SimulatedCamera)
        assert cam.node_map is not None, "open() should have built the node map"
        cam.close()

    def test_opens_a_genicam_device_by_serial(self):
        option = discovery._describe(
            {"vendor": "FLIR", "model": "BFS", "serial_number": "12345678", "index": 0}
        )
        with (
            patch("pybeamprofiler.discovery.find_cti_files", return_value=["/x/a.cti"]),
            patch("pybeamprofiler.gen_camera.HarvesterCamera") as mock_cls,
        ):
            discovery.open_camera(option)

        mock_cls.assert_called_once_with(cti_file=["/x/a.cti"], serial_number="12345678")
        mock_cls.return_value.open.assert_called_once()

    def test_no_cti_files_passes_none_rather_than_an_empty_list(self):
        option = discovery._describe(
            {"vendor": "V", "model": "M", "serial_number": "1", "index": 0}
        )
        with (
            patch("pybeamprofiler.discovery.find_cti_files", return_value=[]),
            patch("pybeamprofiler.gen_camera.HarvesterCamera") as mock_cls,
        ):
            discovery.open_camera(option)

        assert mock_cls.call_args.kwargs["cti_file"] is None

    def test_a_failed_open_is_reported_with_the_camera_label(self):
        option = discovery._describe(
            {"vendor": "FLIR", "model": "BFS", "serial_number": "999", "index": 0}
        )
        with (
            patch("pybeamprofiler.discovery.find_cti_files", return_value=[]),
            patch("pybeamprofiler.gen_camera.HarvesterCamera") as mock_cls,
        ):
            mock_cls.return_value.open.side_effect = RuntimeError("device in use")
            with pytest.raises(RuntimeError, match="Could not open FLIR BFS \\(999\\)"):
                discovery.open_camera(option)

    def test_a_failed_open_releases_the_handle(self):
        """Leaking the handle keeps the device claimed, so the next attempt
        fails for the wrong reason."""
        option = discovery._describe(
            {"vendor": "V", "model": "M", "serial_number": "1", "index": 0}
        )
        with (
            patch("pybeamprofiler.discovery.find_cti_files", return_value=[]),
            patch("pybeamprofiler.gen_camera.HarvesterCamera") as mock_cls,
        ):
            mock_cls.return_value.open.side_effect = RuntimeError("in use")
            with pytest.raises(RuntimeError):
                discovery.open_camera(option)

        mock_cls.return_value.close.assert_called_once()

    def test_cleanup_failure_does_not_mask_the_original_error(self):
        option = discovery._describe(
            {"vendor": "V", "model": "M", "serial_number": "1", "index": 0}
        )
        with (
            patch("pybeamprofiler.discovery.find_cti_files", return_value=[]),
            patch("pybeamprofiler.gen_camera.HarvesterCamera") as mock_cls,
        ):
            mock_cls.return_value.open.side_effect = RuntimeError("in use")
            mock_cls.return_value.close.side_effect = RuntimeError("close also broken")
            with pytest.raises(RuntimeError, match="in use"):
                discovery.open_camera(option)
