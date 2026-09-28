"""Tests for camera discovery: enumerating devices and opening one."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from pybeamprofiler import discovery


def _mock_harvesters_core() -> tuple[Any, Mock]:
    """Patch the Harvester class that discovery (via gen_camera) builds.

    Returns the patcher -- use it as a context manager -- and the class mock.
    """
    mock_harvester_class = Mock()
    return patch("pybeamprofiler.gen_camera.Harvester", mock_harvester_class), mock_harvester_class


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

        with fake_core:
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

        with fake_core:
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

        with fake_core:
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

        with fake_core:
            cameras = discovery.list_cameras()

        assert len(cameras) == 2
        assert cameras[0]["vendor"] == "FLIR"
        assert cameras[1]["vendor"] == "Basler"
        assert cameras[0]["index"] == 0
        assert cameras[1]["index"] == 1

    def test_list_cameras_no_harvesters(self):
        """Test listing cameras when harvesters not installed."""
        with patch("pybeamprofiler.gen_camera.Harvester", None):
            with patch("pybeamprofiler.discovery.find_cti_files", return_value=["/x.cti"]):
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

        with fake_core:
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

        with fake_core:
            cameras = discovery.list_cameras()

        assert cameras == []


# ─── Selectable camera options (what the GUI dropdown drives) ──────────────


class TestCameraOption:
    def test_simulated_option_is_flagged(self):
        assert discovery.default_simulated_option().is_simulated is True

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


class TestSimulatedOptions:
    """More than one simulator is offered so the selector can be exercised
    end to end without hardware."""

    def test_one_option_per_profile(self):
        from pybeamprofiler.simulated import SIMULATED_PROFILES

        options = discovery.simulated_options()
        assert len(options) == len(SIMULATED_PROFILES) >= 2

    def test_keys_are_unique_and_prefixed(self):
        options = discovery.simulated_options()
        keys = [o.key for o in options]
        assert len(set(keys)) == len(keys)
        assert all(k.startswith(discovery.SIMULATED_PREFIX) for k in keys)

    def test_all_are_flagged_simulated(self):
        assert all(o.is_simulated for o in discovery.simulated_options())

    def test_labels_carry_the_fake_serial(self):
        for option in discovery.simulated_options():
            assert option.serial_number in option.label

    def test_default_is_the_first_profile(self):
        from pybeamprofiler.simulated import DEFAULT_PROFILE

        assert discovery.default_simulated_option().serial_number == DEFAULT_PROFILE.serial_number

    def test_profiles_differ_in_sensor_geometry(self):
        """Clones would not prove the selector re-laid anything out."""
        from pybeamprofiler.simulated import SIMULATED_PROFILES

        shapes = {(p.width, p.height, p.pixel_size) for p in SIMULATED_PROFILES}
        assert len(shapes) == len(SIMULATED_PROFILES)


class TestDiscoverCameras:
    def test_simulated_is_always_offered(self):
        with patch("pybeamprofiler.discovery.list_cameras", return_value=[]):
            options = discovery.discover_cameras()
        assert options == discovery.simulated_options()

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

        keys = [o.key for o in options]
        assert keys[:2] == ["genicam:111", "genicam:222"]
        assert all(k.startswith(discovery.SIMULATED_PREFIX) for k in keys[2:])

    def test_the_same_camera_seen_through_two_producers_appears_once(self):
        """A Basler USB3 device enumerates through both the GEV and U3V .cti."""
        devices = [
            {"vendor": "Basler", "model": "acA", "serial_number": "222", "id": "u3v", "index": 0},
            {"vendor": "Basler", "model": "acA", "serial_number": "222", "id": "gev", "index": 1},
        ]
        with patch("pybeamprofiler.discovery.list_cameras", return_value=devices):
            options = discovery.discover_cameras()

        assert [o.key for o in options if not o.is_simulated] == ["genicam:222"]

    def test_discovery_failure_still_offers_the_simulator(self):
        """Behind a Refresh button, a short list beats a traceback."""
        with patch("pybeamprofiler.discovery.list_cameras", side_effect=RuntimeError("no SDK")):
            options = discovery.discover_cameras()
        assert options == discovery.simulated_options()

    def test_cti_file_is_forwarded(self):
        with patch("pybeamprofiler.discovery.list_cameras", return_value=[]) as mock_list:
            discovery.discover_cameras(cti_file="/x/y.cti")
        mock_list.assert_called_once_with("/x/y.cti")


class TestFindOption:
    def test_finds_by_key(self):
        options = discovery.simulated_options()
        assert discovery.find_option(options[0].key, options) is options[0]

    def test_unknown_key_is_none(self):
        assert discovery.find_option("genicam:nope", discovery.simulated_options()) is None

    def test_blank_key_is_none(self):
        options = discovery.simulated_options()
        assert discovery.find_option(None, options) is None
        assert discovery.find_option("", options) is None


class TestOpenCamera:
    def test_opens_the_simulator(self):
        from pybeamprofiler.simulated import SimulatedCamera

        cam = discovery.open_camera(discovery.default_simulated_option())
        assert isinstance(cam, SimulatedCamera)
        assert cam.node_map is not None, "open() should have built the node map"
        cam.close()

    @pytest.mark.parametrize("index", range(2))
    def test_each_simulated_option_opens_its_own_profile(self, index):
        from pybeamprofiler.simulated import SimulatedCamera

        option = discovery.simulated_options()[index]
        cam = discovery.open_camera(option)
        assert isinstance(cam, SimulatedCamera)
        assert cam.serial_number == option.serial_number
        assert (cam.width, cam.height) == (cam.profile.width, cam.profile.height)
        cam.close()

    def test_an_unknown_simulated_profile_falls_back_to_the_default(self):
        from pybeamprofiler.simulated import DEFAULT_PROFILE, SimulatedCamera

        option = discovery.CameraOption(
            key=f"{discovery.SIMULATED_PREFIX}does-not-exist",
            label="ghost",
            kind=discovery.SIMULATED_KEY,
        )
        cam = discovery.open_camera(option)
        assert isinstance(cam, SimulatedCamera)
        assert cam.profile is DEFAULT_PROFILE
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

        mock_cls.assert_called_once_with(
            cti_file=["/x/a.cti"], serial_number="12345678", device_id=None
        )
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


class TestOpeningTheCameraThatWasPicked:
    """Options fell back to the device id as their key, but open_camera()
    passed only the serial -- so a serial-less option opened whichever
    device enumerated first -- and serials matched as substrings."""

    def test_a_serial_less_camera_opens_by_device_id(self):
        option = discovery._describe(
            {"vendor": "V", "model": "CamB", "serial_number": "", "id": "devB", "index": 1}
        )
        assert option.key == "genicam:devB"
        assert option.device_id == "devB"
        with (
            patch("pybeamprofiler.discovery.find_cti_files", return_value=["/x/a.cti"]),
            patch("pybeamprofiler.gen_camera.HarvesterCamera") as mock_cls,
        ):
            discovery.open_camera(option)
        assert mock_cls.call_args.kwargs["device_id"] == "devB"
        assert mock_cls.call_args.kwargs["serial_number"] is None

    def test_the_serial_wins_when_there_is_one(self):
        option = discovery._describe(
            {"vendor": "V", "model": "M", "serial_number": "123", "id": "dev", "index": 0}
        )
        assert option.key == "genicam:123"  # keys unchanged by carrying the id
        with (
            patch("pybeamprofiler.discovery.find_cti_files", return_value=[]),
            patch("pybeamprofiler.gen_camera.HarvesterCamera") as mock_cls,
        ):
            discovery.open_camera(option)
        assert mock_cls.call_args.kwargs["serial_number"] == "123"
        assert mock_cls.call_args.kwargs["device_id"] is None

    def test_an_option_that_identifies_nothing_is_refused(self):
        option = discovery._describe({"vendor": "V", "model": "M", "index": 3})
        with patch("pybeamprofiler.gen_camera.HarvesterCamera") as mock_cls:
            with pytest.raises(RuntimeError, match="Could not open V M: .*neither a serial"):
                discovery.open_camera(option)
        mock_cls.assert_not_called()

    def test_describing_an_open_camera_keeps_its_device_id(self):
        from types import SimpleNamespace

        camera = SimpleNamespace(
            device_vendor="V", device_model="M", serial_number="", device_id="devB"
        )
        assert discovery.describe_open_camera(camera).key == "genicam:devB"  # ty: ignore[invalid-argument-type]
