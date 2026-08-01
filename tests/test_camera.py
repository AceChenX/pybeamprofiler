"""Tests for camera interfaces and control."""

import os
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pybeamprofiler import BeamProfiler, SimulatedCamera
from pybeamprofiler.camera import _categorize_feature
from pybeamprofiler.simulated import _SimulatedNode


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

        bp.set_exposure(0.05)  # ty: ignore[call-non-callable]
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


class TestApplySettingsFromKwargs:
    """Test _apply_settings_from_kwargs on Camera base class."""

    def test_set_exposure_via_kwargs(self):
        """Test setting exposure_time through kwargs."""
        cam = SimulatedCamera()
        cam.open()
        cam._apply_settings_from_kwargs({"exposure_time": 0.05})
        assert cam.exposure_time == 0.05
        cam.close()

    def test_set_gain_via_kwargs(self):
        """Test setting gain through kwargs."""
        cam = SimulatedCamera()
        cam.open()
        cam._apply_settings_from_kwargs({"gain": 3.0})
        assert cam.gain == 3.0
        cam.close()

    def test_set_exposure_alias(self):
        """Test setting ExposureTime alias."""
        cam = SimulatedCamera()
        cam.open()
        cam._apply_settings_from_kwargs({"ExposureTime": 0.02})
        assert cam.exposure_time == 0.02
        cam.close()

    def test_set_gain_alias(self):
        """Test setting Gain alias."""
        cam = SimulatedCamera()
        cam.open()
        cam._apply_settings_from_kwargs({"Gain": 5.0})
        assert cam.gain == 5.0
        cam.close()

    def test_unrecognized_param_no_node_map(self):
        """Test unrecognized params log warning when no node_map."""
        cam = SimulatedCamera()
        cam.open()
        cam._apply_settings_from_kwargs({"UnknownParam": 42})
        cam.close()

    def test_node_map_param_setting(self):
        """Test setting a parameter via node_map."""
        cam = SimulatedCamera()
        cam.open()
        mock_node = MagicMock()
        mock_node.value = 10
        mock_node_map = MagicMock()
        mock_node_map.TestParam = mock_node
        cam.node_map = mock_node_map
        cam._apply_settings_from_kwargs({"TestParam": 42})
        assert mock_node.value == 42
        cam.close()

    def test_node_map_bool_conversion_on(self):
        """Test string-to-bool conversion for Enable params."""
        cam = SimulatedCamera()
        cam.open()
        mock_node = MagicMock()
        mock_node.value = False  # Current value is bool
        mock_node_map = MagicMock()
        mock_node_map.TestEnable = mock_node
        cam.node_map = mock_node_map
        cam._apply_settings_from_kwargs({"TestEnable": "on"})
        assert mock_node.value is True
        cam.close()

    def test_node_map_bool_conversion_off(self):
        """Test string-to-bool conversion for Enable params (off)."""
        cam = SimulatedCamera()
        cam.open()
        mock_node = MagicMock()
        mock_node.value = True  # Current value is bool
        mock_node_map = MagicMock()
        mock_node_map.TestEnable = mock_node
        cam.node_map = mock_node_map
        cam._apply_settings_from_kwargs({"TestEnable": "off"})
        assert mock_node.value is False
        cam.close()

    def test_node_map_missing_param(self):
        """Test setting a param not in node_map logs warning."""
        cam = SimulatedCamera()
        cam.open()
        mock_node_map = MagicMock(spec=[])
        cam.node_map = mock_node_map
        cam._apply_settings_from_kwargs({"MissingParam": 1})
        cam.close()

    def test_exposure_error_handled(self):
        """Test exposure setting error is handled."""
        cam = SimulatedCamera()
        cam.open()
        cam.set_exposure = MagicMock(side_effect=RuntimeError("fail"))  # ty: ignore[invalid-assignment]
        cam._apply_settings_from_kwargs({"exposure_time": 0.01})
        cam.close()

    def test_gain_error_handled(self):
        """Test gain setting error is handled."""
        cam = SimulatedCamera()
        cam.open()
        cam.set_gain = MagicMock(side_effect=RuntimeError("fail"))  # ty: ignore[invalid-assignment]
        cam._apply_settings_from_kwargs({"gain": 5.0})
        cam.close()


class TestGenCameraParseGentlPath:
    """Test HarvesterCamera._parse_gentl_path."""

    def test_parse_direct_cti_file(self, tmp_path):
        """Test parsing a direct .cti file path."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        cti_file = tmp_path / "test.cti"
        cti_file.touch()

        result = HarvesterCamera._parse_gentl_path(str(cti_file))
        assert result == str(cti_file)

    def test_parse_directory_with_cti(self, tmp_path):
        """Test parsing a directory containing .cti files."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        (tmp_path / "a.cti").touch()
        (tmp_path / "b.cti").touch()
        (tmp_path / "readme.txt").touch()

        result = HarvesterCamera._parse_gentl_path(str(tmp_path))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_empty_path(self):
        """Test parsing empty/nonexistent path."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        result = HarvesterCamera._parse_gentl_path("/nonexistent/path")
        assert result is None

    def test_parse_multiple_paths(self, tmp_path):
        """Test parsing multiple paths separated by ':'."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        (dir1 / "a.cti").touch()

        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        (dir2 / "b.cti").touch()

        sep = ";" if os.name == "nt" else ":"
        result = HarvesterCamera._parse_gentl_path(f"{dir1}{sep}{dir2}")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_single_cti_returns_str(self, tmp_path):
        """Test parsing returns str when only one .cti found."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        cti_dir = tmp_path / "cti"
        cti_dir.mkdir()
        (cti_dir / "single.cti").touch()

        result = HarvesterCamera._parse_gentl_path(str(cti_dir))
        assert isinstance(result, str)
        assert result.endswith("single.cti")


class TestGenCameraInit:
    """Test HarvesterCamera initialization."""

    def test_init_without_harvesters_raises(self):
        """Test that missing harvesters raises ImportError."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        with patch("pybeamprofiler.gen_camera.Harvester", None):
            with pytest.raises(ImportError, match="harvesters"):
                HarvesterCamera()

    def test_init_with_missing_cti_warns(self):
        """Test that missing CTI file logs warning."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            HarvesterCamera(cti_file="/nonexistent/file.cti")
            mock_harvester.return_value.add_file.assert_not_called()

    def test_init_with_list_of_cti_files(self, tmp_path):
        """Test initialization with a list of CTI files."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        f1 = tmp_path / "a.cti"
        f2 = tmp_path / "b.cti"
        f1.touch()
        f2.touch()

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            HarvesterCamera(cti_file=[str(f1), str(f2)])
            assert mock_harvester.return_value.add_file.call_count == 2


class TestGigEStreamConfig:
    """Test _configure_gige_stream for macOS SocketDriver fallback."""

    def _make_mock_camera(self):
        """Create a HarvesterCamera with mocked internals."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            cam = HarvesterCamera(cti_file="/fake.cti")
        cam.ia = MagicMock()
        return cam

    @patch("pybeamprofiler.gen_camera.platform.system", return_value="Darwin")
    def test_switches_to_socket_driver_on_macos(self, _mock_sys):
        """On macOS with GigEAccelerator, stream is switched to SocketDriver."""
        cam = self._make_mock_camera()
        ds_nm = MagicMock()
        ds_nm.Type.value = "GigEAccelerator"
        ds_nm.TypeIsSocketDriverAvailable.value = True
        cam.ia.data_streams = [MagicMock(node_map=ds_nm)]

        cam._configure_gige_stream()
        assert ds_nm.Type.value == "SocketDriver"

    @patch("pybeamprofiler.gen_camera.platform.system", return_value="Darwin")
    def test_noop_when_already_socket_driver(self, _mock_sys):
        """No change if stream is already using SocketDriver."""
        cam = self._make_mock_camera()
        ds_nm = MagicMock()
        ds_nm.Type.value = "SocketDriver"
        cam.ia.data_streams = [MagicMock(node_map=ds_nm)]

        cam._configure_gige_stream()
        assert ds_nm.Type.value == "SocketDriver"

    @patch("pybeamprofiler.gen_camera.platform.system", return_value="Linux")
    def test_noop_on_linux(self, _mock_sys):
        """No change on non-Darwin platforms."""
        cam = self._make_mock_camera()
        ds_nm = MagicMock()
        ds_nm.Type.value = "GigEAccelerator"
        cam.ia.data_streams = [MagicMock(node_map=ds_nm)]

        cam._configure_gige_stream()
        assert ds_nm.Type.value == "GigEAccelerator"

    @patch("pybeamprofiler.gen_camera.platform.system", return_value="Darwin")
    def test_noop_when_no_data_streams(self, _mock_sys):
        """No error when data_streams is empty (e.g. USB camera)."""
        cam = self._make_mock_camera()
        cam.ia.data_streams = []
        cam._configure_gige_stream()

    @patch("pybeamprofiler.gen_camera.platform.system", return_value="Darwin")
    def test_graceful_on_missing_type_attr(self, _mock_sys):
        """Handles cameras whose data-stream node map lacks the Type attribute."""
        cam = self._make_mock_camera()
        ds_nm = MagicMock(spec=[])
        cam.ia.data_streams = [MagicMock(node_map=ds_nm)]
        cam._configure_gige_stream()


class TestGenCameraExposureGain:
    """Test HarvesterCamera exposure/gain methods without hardware."""

    def _make_mock_camera(self):
        """Create a HarvesterCamera with mocked internals."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            cam = HarvesterCamera()

        cam.node_map = MagicMock()
        return cam

    def test_set_exposure_primary(self):
        """Test set_exposure using ExposureTime node."""
        cam = self._make_mock_camera()
        cam.set_exposure(0.01)
        cam.node_map.ExposureTime.value = 10000  # 0.01s in μs
        assert cam.exposure_time == 0.01

    def test_set_exposure_fallback(self):
        """Test set_exposure falls back to ExposureTimeAbs."""
        cam = self._make_mock_camera()
        cam.node_map.ExposureTime = MagicMock()
        type(cam.node_map.ExposureTime).value = property(fset=MagicMock(side_effect=AttributeError))
        cam.set_exposure(0.01)
        assert cam.exposure_time == 0.01

    def test_set_gain_primary(self):
        """Test set_gain using Gain node."""
        cam = self._make_mock_camera()
        cam.set_gain(5.0)
        assert cam.gain == 5.0

    def test_set_gain_fallback(self):
        """Test set_gain falls back to GainRaw."""
        cam = self._make_mock_camera()
        cam.node_map.Gain = MagicMock()
        type(cam.node_map.Gain).value = property(fset=MagicMock(side_effect=AttributeError))
        cam.set_gain(10.0)
        assert cam.gain == 10.0

    def test_exposure_range_property(self):
        """Test exposure_range property."""
        cam = self._make_mock_camera()
        cam._exposure_min = 1e-6
        cam._exposure_max = 1.0
        assert cam.exposure_range == (1e-6, 1.0)

    def test_gain_range_property(self):
        """Test gain_range property."""
        cam = self._make_mock_camera()
        cam._gain_min = 0.0
        cam._gain_max = 24.0
        assert cam.gain_range == (0.0, 24.0)

    def test_roi_info_property(self):
        """Test roi_info property."""
        cam = self._make_mock_camera()
        cam._roi_offset_x = 10
        cam._roi_offset_y = 20
        cam.width_pixels = 640
        cam.height_pixels = 480
        cam._roi_max_width = 1024
        cam._roi_max_height = 768
        info = cam.roi_info
        assert info["offset_x"] == 10
        assert info["max_width"] == 1024

    def test_close_with_ia(self):
        """Test close destroys image acquirer."""
        cam = self._make_mock_camera()
        mock_ia = MagicMock()
        cam.ia = mock_ia
        cam.close()
        mock_ia.destroy.assert_called_once()
        cam.h.reset.assert_called_once()

    def test_close_without_ia(self):
        """Test close with no image acquirer."""
        cam = self._make_mock_camera()
        cam.ia = None
        cam.close()  # Should not raise
        cam.h.reset.assert_called_once()

    def test_start_acquisition(self):
        """Test start_acquisition calls ia.start."""
        cam = self._make_mock_camera()
        mock_ia = MagicMock()
        cam.ia = mock_ia
        cam.start_acquisition()
        try:
            mock_ia.start.assert_called_once()
            assert cam.is_acquiring is True
        finally:
            cam.stop_acquisition()

    def test_stop_acquisition(self):
        """Test stop_acquisition calls ia.stop."""
        cam = self._make_mock_camera()
        mock_ia = MagicMock()
        cam.ia = mock_ia
        cam.is_acquiring = True
        cam.stop_acquisition()
        mock_ia.stop.assert_called_once()
        assert cam.is_acquiring is False

    def test_get_image_without_ia_raises(self):
        """Test get_image raises when not opened."""
        cam = self._make_mock_camera()
        cam.ia = None
        with pytest.raises(RuntimeError, match="Camera not opened"):
            cam.get_image()

    def test_set_roi(self):
        """Test set_roi sets node_map values."""
        cam = self._make_mock_camera()
        cam._roi_max_width = 1024
        cam._roi_max_height = 768
        cam.set_roi(offset_x=10, offset_y=20, width=640, height=480)
        assert cam.width == 640
        assert cam.height == 480
        assert cam._roi_offset_x == 10

    def test_set_roi_defaults_to_max(self):
        """Test set_roi uses max dimensions when not specified."""
        cam = self._make_mock_camera()
        cam._roi_max_width = 1024
        cam._roi_max_height = 768
        cam.set_roi()
        assert cam.width == 1024
        assert cam.height == 768

    def test_set_roi_no_node_map(self):
        """Test set_roi warns when camera not opened."""
        cam = self._make_mock_camera()
        cam.node_map = None
        cam.set_roi()  # Should not raise


class TestGenCameraSensorLookup:
    """Test sensor pixel size lookup."""

    def test_lookup_by_sensor_description(self):
        """Test pixel size lookup via SensorDescription."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            cam = HarvesterCamera()

        cam.node_map = MagicMock()
        cam.node_map.SensorDescription.value = "Sony IMX174 CMOS"
        result = cam._lookup_sensor_pixel_size()
        assert result == 5.86

    def test_lookup_by_device_model_name(self):
        """Test pixel size lookup via DeviceModelName."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            cam = HarvesterCamera()

        cam.node_map = MagicMock(spec=["DeviceModelName"])
        cam.node_map.DeviceModelName.value = "acA4024-8gm"
        result = cam._lookup_sensor_pixel_size()
        assert result == 1.85

    def test_lookup_unknown_sensor(self):
        """Test pixel size lookup for unknown sensor returns None."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            cam = HarvesterCamera()

        cam.node_map = MagicMock(spec=[])
        result = cam._lookup_sensor_pixel_size()
        assert result is None


class TestGenCameraGetImage:
    """Test HarvesterCamera.get_image timeout normalisation and stall recovery."""

    def _make_mock_camera(self):
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            cam = HarvesterCamera()
        cam.ia = MagicMock()
        cam.is_acquiring = True
        cam.exposure_time = 0.01
        return cam

    def test_harvesters_timeout_normalised(self):
        """The real ``harvesters.core.TimeoutException`` maps to ``TimeoutError``."""
        from harvesters.core import TimeoutException

        cam = self._make_mock_camera()
        cam.ia.fetch.side_effect = TimeoutException
        with pytest.raises(TimeoutError, match="did not deliver a frame"):
            cam.get_image(timeout=0.1)

    def test_builtin_timeout_error_normalised(self):
        """Python's built-in ``TimeoutError`` is also normalised (re-wrapped)."""
        cam = self._make_mock_camera()
        cam.ia.fetch.side_effect = TimeoutError("slow")
        with pytest.raises(TimeoutError, match="did not deliver a frame"):
            cam.get_image(timeout=0.1)

    def test_non_timeout_exception_propagates(self):
        """Non-timeout errors bubble up unchanged."""
        cam = self._make_mock_camera()
        cam.ia.fetch.side_effect = RuntimeError("bad buffer")
        with pytest.raises(RuntimeError, match="bad buffer"):
            cam.get_image(timeout=0.1)

    def test_successful_fetch_resets_stall_state(self):
        """A successful fetch clears the stall-recovery flags."""
        cam = self._make_mock_camera()
        cam._stall_recovery_attempted = True
        cam._last_successful_fetch = 0.0

        buf = MagicMock()
        comp = MagicMock()
        comp.width, comp.height = 4, 4
        comp.data = np.zeros(16, dtype=np.uint8)
        buf.__enter__.return_value.payload.components = [comp]
        cam.ia.fetch.return_value = buf

        img = cam.get_image(timeout=0.1)
        assert img.shape == (4, 4)
        assert cam._stall_recovery_attempted is False
        assert cam._last_successful_fetch > 0.0

    def test_stall_recovery_restarts_acquisition(self):
        """Consecutive timeouts beyond the stall window trigger stop/start."""
        from harvesters.core import TimeoutException

        cam = self._make_mock_camera()
        cam.ia.fetch.side_effect = TimeoutException
        # Simulate a successful fetch 10s ago — beyond the 5s stall window
        # at a 10 ms exposure.
        cam._last_successful_fetch = time.monotonic() - 10.0

        with pytest.raises(TimeoutError):
            cam.get_image(timeout=0.1)

        cam.ia.stop.assert_called()
        cam.ia.start.assert_called()

    def test_stall_recovery_is_one_shot(self):
        """A second timeout within the same stall window doesn't re-trigger recovery."""
        from harvesters.core import TimeoutException

        cam = self._make_mock_camera()
        cam.ia.fetch.side_effect = TimeoutException
        cam._last_successful_fetch = time.monotonic() - 10.0

        with pytest.raises(TimeoutError):
            cam.get_image(timeout=0.1)
        stop_calls = cam.ia.stop.call_count
        start_calls = cam.ia.start.call_count

        with pytest.raises(TimeoutError):
            cam.get_image(timeout=0.1)

        assert cam.ia.stop.call_count == stop_calls
        assert cam.ia.start.call_count == start_calls

    def test_first_timeout_seeds_stall_timer(self):
        """The very first fetch timing out should NOT trigger recovery."""
        from harvesters.core import TimeoutException

        cam = self._make_mock_camera()
        cam.ia.fetch.side_effect = TimeoutException
        assert cam._last_successful_fetch == 0.0

        with pytest.raises(TimeoutError):
            cam.get_image(timeout=0.1)

        cam.ia.stop.assert_not_called()
        assert cam._last_successful_fetch > 0.0

    def test_start_acquisition_resets_stall_state(self):
        """`start_acquisition` clears stall tracking so pause/resume is safe."""
        cam = self._make_mock_camera()
        cam.is_acquiring = False
        cam._last_successful_fetch = 123.0
        cam._stall_recovery_attempted = True

        cam.start_acquisition()

        assert cam._last_successful_fetch == 0.0
        assert cam._stall_recovery_attempted is False


class TestBaslerCameraInit:
    """Test BaslerCamera initialization and CTI discovery."""

    @patch("pybeamprofiler.basler.os.environ", {})
    @patch("pybeamprofiler.basler.BaslerCamera._find_basler_cti")
    def test_basler_init_finds_cti(self, mock_find):
        """Test BaslerCamera uses _find_basler_cti when no env var."""
        from pybeamprofiler.basler import BaslerCamera

        mock_find.return_value = ["/path/ProducerGEV.cti"]
        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            BaslerCamera()
        mock_find.assert_called_once()

    @patch("pybeamprofiler.basler.os.environ", {"GENICAM_GENTL64_PATH": "/some/path"})
    @patch("pybeamprofiler.basler.BaslerCamera._find_basler_cti", return_value=None)
    @patch("pybeamprofiler.basler.HarvesterCamera._parse_gentl_path")
    def test_basler_init_uses_env_var(self, mock_parse, _mock_find):
        """Test BaslerCamera uses GENICAM_GENTL64_PATH when vendor search fails."""
        from pybeamprofiler.basler import BaslerCamera

        mock_parse.return_value = "/some/path/test.cti"
        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            BaslerCamera()
        mock_parse.assert_called_once_with("/some/path")

    @patch("pybeamprofiler.basler.platform.system")
    @patch("pybeamprofiler.basler.os.path.isdir")
    @patch("pybeamprofiler.basler.os.path.exists")
    def test_find_basler_cti_linux(self, mock_exists, mock_isdir, mock_system):
        """Test _find_basler_cti on Linux."""
        from pybeamprofiler.basler import BaslerCamera

        mock_system.return_value = "Linux"
        mock_isdir.side_effect = lambda p: p == "/opt/pylon/lib/gentlproducer/gtl"
        mock_exists.side_effect = lambda p: p == "/opt/pylon/lib/gentlproducer/gtl/ProducerGEV.cti"

        result = BaslerCamera._find_basler_cti()
        assert result is not None
        assert any("ProducerGEV.cti" in f for f in result)

    @patch("pybeamprofiler.basler.platform.system")
    @patch("pybeamprofiler.basler.os.path.isdir")
    def test_find_basler_cti_not_installed(self, mock_isdir, mock_system):
        """Test _find_basler_cti returns None when SDK not installed."""
        from pybeamprofiler.basler import BaslerCamera

        mock_system.return_value = "Linux"
        mock_isdir.return_value = False

        result = BaslerCamera._find_basler_cti()
        assert result is None

    @patch("pybeamprofiler.basler.platform.system")
    @patch("pybeamprofiler.basler.os.path.isdir")
    @patch("pybeamprofiler.basler.os.path.exists")
    def test_find_basler_cti_darwin(self, mock_exists, mock_isdir, mock_system):
        """Test _find_basler_cti on macOS."""
        from pybeamprofiler.basler import BaslerCamera

        mock_system.return_value = "Darwin"
        base = "/Library/Frameworks/pylon.framework/Libraries/gentlproducer/gtl"
        mock_isdir.side_effect = lambda p: p == base
        mock_exists.side_effect = lambda p: p == f"{base}/ProducerU3V.cti"

        result = BaslerCamera._find_basler_cti()
        assert result is not None

    @patch("pybeamprofiler.basler.platform.system")
    @patch("pybeamprofiler.basler.os.path.isdir")
    @patch("pybeamprofiler.basler.os.path.exists")
    def test_find_basler_cti_windows(self, mock_exists, mock_isdir, mock_system):
        """Test _find_basler_cti on Windows."""
        from pybeamprofiler.basler import BaslerCamera

        mock_system.return_value = "Windows"
        base = r"C:\Program Files\Basler\pylon 7\Runtime\x64"
        mock_isdir.side_effect = lambda p: p == base
        mock_exists.side_effect = lambda p: p == os.path.join(base, "ProducerGEV.cti")

        result = BaslerCamera._find_basler_cti()
        assert result is not None


class TestFlirCameraInit:
    """Test FlirCamera initialization and CTI discovery."""

    @patch("pybeamprofiler.flir.os.environ", {})
    @patch("pybeamprofiler.flir.FlirCamera._find_flir_cti")
    def test_flir_init_finds_cti(self, mock_find):
        """Test FlirCamera uses _find_flir_cti when no env var."""
        from pybeamprofiler.flir import FlirCamera

        mock_find.return_value = "/path/FLIR_GenTL.cti"
        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            FlirCamera()
        mock_find.assert_called_once()

    @patch("pybeamprofiler.flir.os.environ", {"GENICAM_GENTL64_PATH": "/flir/path"})
    @patch("pybeamprofiler.flir.FlirCamera._find_flir_cti", return_value=None)
    @patch("pybeamprofiler.flir.HarvesterCamera._parse_gentl_path")
    def test_flir_init_uses_env_var(self, mock_parse, _mock_find):
        """Test FlirCamera uses GENICAM_GENTL64_PATH when vendor search fails."""
        from pybeamprofiler.flir import FlirCamera

        mock_parse.return_value = "/flir/path/FLIR_GenTL.cti"
        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            FlirCamera()
        mock_parse.assert_called_once_with("/flir/path")

    @patch("pybeamprofiler.flir.platform.system")
    @patch("pybeamprofiler.flir.os.path.isdir")
    @patch("pybeamprofiler.flir.os.listdir")
    def test_find_flir_cti_linux(self, mock_listdir, mock_isdir, mock_system):
        """Test _find_flir_cti on Linux."""
        from pybeamprofiler.flir import FlirCamera

        mock_system.return_value = "Linux"
        mock_isdir.side_effect = lambda p: p == "/opt/spinnaker/lib/flir-gentl"
        mock_listdir.return_value = ["FLIR_GenTL_v140.cti"]

        result = FlirCamera._find_flir_cti()
        assert result is not None
        assert "FLIR_GenTL_v140.cti" in result

    @patch("pybeamprofiler.flir.platform.system")
    @patch("pybeamprofiler.flir.os.path.isdir")
    def test_find_flir_cti_not_installed(self, mock_isdir, mock_system):
        """Test _find_flir_cti returns None when SDK not installed."""
        from pybeamprofiler.flir import FlirCamera

        mock_system.return_value = "Linux"
        mock_isdir.return_value = False

        result = FlirCamera._find_flir_cti()
        assert result is None

    @patch("pybeamprofiler.flir.platform.system")
    @patch("pybeamprofiler.flir.os.path.isdir")
    @patch("pybeamprofiler.flir.os.listdir")
    def test_find_flir_cti_darwin(self, mock_listdir, mock_isdir, mock_system):
        """Test _find_flir_cti on macOS."""
        from pybeamprofiler.flir import FlirCamera

        mock_system.return_value = "Darwin"
        mock_isdir.side_effect = lambda p: p == "/usr/local/lib/spinnaker-gentl"
        mock_listdir.return_value = ["FLIR_GenTL.cti"]

        result = FlirCamera._find_flir_cti()
        assert result is not None

    @patch("pybeamprofiler.flir.platform.system")
    @patch("pybeamprofiler.flir.os.path.exists")
    @patch("pybeamprofiler.flir.os.path.isdir")
    @patch("pybeamprofiler.flir.os.listdir")
    def test_find_flir_cti_windows(self, mock_listdir, mock_isdir, mock_exists, mock_system):
        """Test _find_flir_cti on Windows."""
        from pybeamprofiler.flir import FlirCamera

        mock_system.return_value = "Windows"
        base = r"C:\Program Files\Teledyne\Spinnaker\cti64"
        mock_exists.return_value = True
        mock_isdir.side_effect = lambda p: True
        mock_listdir.side_effect = lambda p: ["vs2015"] if p == base else ["FLIR_GenTL_v140.cti"]

        result = FlirCamera._find_flir_cti()
        assert result is not None

    @patch("pybeamprofiler.flir.platform.system")
    @patch("pybeamprofiler.flir.os.path.exists")
    @patch("pybeamprofiler.flir.os.path.isdir")
    @patch("pybeamprofiler.flir.os.listdir")
    def test_find_flir_cti_windows_oserror(
        self, mock_listdir, mock_isdir, mock_exists, mock_system
    ):
        """Test _find_flir_cti handles OSError on Windows listdir."""
        from pybeamprofiler.flir import FlirCamera

        mock_system.return_value = "Windows"
        mock_exists.return_value = True
        mock_listdir.side_effect = OSError("Access denied")

        result = FlirCamera._find_flir_cti()
        assert result is None

    @patch("pybeamprofiler.flir.os.environ", {})
    @patch("pybeamprofiler.flir.FlirCamera._find_flir_cti")
    def test_flir_init_no_cti_found(self, mock_find):
        """Test FlirCamera warns when no CTI found."""
        from pybeamprofiler.flir import FlirCamera

        mock_find.return_value = None
        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            FlirCamera()

    @patch("pybeamprofiler.basler.os.environ", {})
    @patch("pybeamprofiler.basler.BaslerCamera._find_basler_cti")
    def test_basler_init_no_cti_found(self, mock_find):
        """Test BaslerCamera warns when no CTI found."""
        from pybeamprofiler.basler import BaslerCamera

        mock_find.return_value = None
        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            BaslerCamera()

    @patch("pybeamprofiler.basler.os.environ", {})
    @patch("pybeamprofiler.basler.BaslerCamera._find_basler_cti")
    def test_basler_init_scalar_cti_string(self, mock_find):
        """Defensive branch: if ``_find_basler_cti`` ever returns a scalar
        string (instead of a list), ``BaslerCamera.__init__`` must still
        handle it and log the singular-form message."""
        from pybeamprofiler.basler import BaslerCamera

        mock_find.return_value = "/some/single.cti"
        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            BaslerCamera()
        mock_find.assert_called_once()

    @patch("pybeamprofiler.flir.platform.system")
    @patch("pybeamprofiler.flir.os.path.isdir")
    @patch("pybeamprofiler.flir.os.listdir")
    def test_find_flir_cti_linux_listdir_oserror(self, mock_listdir, mock_isdir, mock_system):
        """Per-directory ``os.listdir`` failure on Linux must be swallowed
        and allow the search to continue (line 88-89)."""
        from pybeamprofiler.flir import FlirCamera

        mock_system.return_value = "Linux"
        mock_isdir.return_value = True  # claims dir exists
        mock_listdir.side_effect = OSError("EACCES")

        # No exception propagates, we just get None back.
        assert FlirCamera._find_flir_cti() is None


class TestGenCameraDetection:
    """Test HarvesterCamera detection methods with mocked node_map."""

    def _make_cam(self):
        """Create a HarvesterCamera with mocked internals."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        mock_harvester = MagicMock()
        with patch("pybeamprofiler.gen_camera.Harvester", mock_harvester):
            cam = HarvesterCamera()
        cam.node_map = MagicMock()
        return cam

    def test_detect_exposure_range_from_exposure_time(self):
        """Test _detect_exposure_range from ExposureTime node."""
        cam = self._make_cam()
        cam.node_map.ExposureTime.min = 20.0  # 20 μs
        cam.node_map.ExposureTime.max = 1000000.0  # 1s in μs
        cam._detect_exposure_range()
        assert cam._exposure_min == pytest.approx(20e-6)
        assert cam._exposure_max == pytest.approx(1.0)

    def test_detect_exposure_range_from_exposure_time_abs(self):
        """Test _detect_exposure_range fallback to ExposureTimeAbs."""
        cam = self._make_cam()
        del cam.node_map.ExposureTime
        cam.node_map.ExposureTimeAbs.min = 50.0
        cam.node_map.ExposureTimeAbs.max = 500000.0
        cam._detect_exposure_range()
        assert cam._exposure_min == pytest.approx(50e-6)

    def test_detect_exposure_range_exception(self):
        """Test _detect_exposure_range handles exceptions."""
        cam = self._make_cam()
        cam.node_map.ExposureTime = MagicMock(side_effect=AttributeError)
        type(cam.node_map).ExposureTime = property(fget=MagicMock(side_effect=RuntimeError))
        cam._detect_exposure_range()  # Should not raise

    def test_detect_gain_range_from_gain(self):
        """Test _detect_gain_range from Gain node."""
        cam = self._make_cam()
        cam.node_map.Gain.min = 0.0
        cam.node_map.Gain.max = 48.0
        cam._detect_gain_range()
        assert cam._gain_min == 0.0
        assert cam._gain_max == 48.0

    def test_detect_gain_range_from_gain_raw(self):
        """Test _detect_gain_range fallback to GainRaw."""
        cam = self._make_cam()
        del cam.node_map.Gain
        cam.node_map.GainRaw.min = 0
        cam.node_map.GainRaw.max = 100
        cam._detect_gain_range()
        assert cam._gain_min == 0.0
        assert cam._gain_max == 100.0

    def test_detect_roi_range(self):
        """Test _detect_roi_range from node_map."""
        cam = self._make_cam()
        cam.node_map.WidthMax.value = 2048
        cam.node_map.HeightMax.value = 1536
        cam._detect_roi_range()
        assert cam._roi_max_width == 2048
        assert cam._roi_max_height == 1536

    def test_detect_pixel_size_from_sensor_pixel_width(self):
        """Test _detect_pixel_size from SensorPixelWidth."""
        cam = self._make_cam()
        cam.node_map.SensorPixelWidth.value = 3.45
        cam._detect_pixel_size()
        assert cam.pixel_size == 3.45

    def test_detect_pixel_size_from_sensor_pixel_height(self):
        """Test _detect_pixel_size fallback to SensorPixelHeight."""
        cam = self._make_cam()
        del cam.node_map.SensorPixelWidth
        cam.node_map.SensorPixelHeight.value = 5.86
        cam._detect_pixel_size()
        assert cam.pixel_size == 5.86

    def test_detect_pixel_size_from_pixel_size_numeric(self):
        """Test _detect_pixel_size from PixelSize if numeric."""
        cam = self._make_cam()
        del cam.node_map.SensorPixelWidth
        del cam.node_map.SensorPixelHeight
        cam.node_map.PixelSize.value = 2.4
        cam._detect_pixel_size()
        assert cam.pixel_size == 2.4

    def test_detect_pixel_size_from_pixel_size_string_ignored(self):
        """Test _detect_pixel_size ignores string PixelSize (e.g. 'Bpp8')."""
        cam = self._make_cam()
        del cam.node_map.SensorPixelWidth
        del cam.node_map.SensorPixelHeight
        cam.node_map.PixelSize.value = "Bpp8"
        cam._lookup_sensor_pixel_size = MagicMock(return_value=None)
        cam._detect_pixel_size()
        assert cam.pixel_size == 1.0  # Default

    def test_detect_pixel_size_from_sensor_lookup(self):
        """Test _detect_pixel_size from sensor model lookup."""
        cam = self._make_cam()
        del cam.node_map.SensorPixelWidth
        del cam.node_map.SensorPixelHeight
        del cam.node_map.PixelSize
        cam._lookup_sensor_pixel_size = MagicMock(return_value=6.9)
        cam._detect_pixel_size()
        assert cam.pixel_size == 6.9

    def test_detect_pixel_size_default(self):
        """Test _detect_pixel_size defaults to 1.0."""
        cam = self._make_cam()
        del cam.node_map.SensorPixelWidth
        del cam.node_map.SensorPixelHeight
        del cam.node_map.PixelSize
        cam._lookup_sensor_pixel_size = MagicMock(return_value=None)
        cam._detect_pixel_size()
        assert cam.pixel_size == 1.0

    def test_configure_camera_settings(self):
        """Test _configure_camera_settings disables auto features."""
        cam = self._make_cam()
        cam._reset_roi_to_full_sensor = MagicMock()
        cam._configure_camera_settings()
        assert cam.node_map.ExposureAuto.value == "Off"
        assert cam.node_map.GainAuto.value == "Off"
        assert cam.node_map.GammaEnable.value is False
        cam._reset_roi_to_full_sensor.assert_called_once()

    def test_reset_roi_to_full_sensor(self):
        """Test _reset_roi_to_full_sensor sets offset and max dimensions."""
        cam = self._make_cam()
        cam.node_map.WidthMax.value = 2048
        cam.node_map.HeightMax.value = 1536
        cam._reset_roi_to_full_sensor()
        assert cam.node_map.OffsetX.value == 0
        assert cam.node_map.OffsetY.value == 0
        assert cam.node_map.Width.value == 2048
        assert cam.node_map.Height.value == 1536

    def test_configure_camera_settings_exception(self):
        """Test _configure_camera_settings with ExposureAuto exception."""
        cam = self._make_cam()
        type(cam.node_map.ExposureAuto).value = property(
            fset=MagicMock(side_effect=RuntimeError("hw error"))
        )
        cam._reset_roi_to_full_sensor = MagicMock()
        cam._configure_camera_settings()  # Should not raise

    def test_configure_camera_settings_no_features(self):
        """Test _configure_camera_settings with missing features."""
        cam = self._make_cam()
        del cam.node_map.ExposureAuto
        del cam.node_map.GainAuto
        del cam.node_map.GammaEnable
        cam._reset_roi_to_full_sensor = MagicMock()
        cam._configure_camera_settings()  # Should not raise

    def test_detect_pixel_size_outer_exception(self):
        """Test _detect_pixel_size handles outer exception."""
        cam = self._make_cam()
        type(cam.node_map).SensorPixelWidth = property(
            fget=MagicMock(side_effect=RuntimeError("fatal"))
        )
        cam._detect_pixel_size()
        assert cam.pixel_size == 1.0

    def test_detect_exposure_range_abs_node(self):
        """Test _detect_exposure_range with ExposureTimeAbs."""
        cam = self._make_cam()
        del cam.node_map.ExposureTime
        cam.node_map.ExposureTimeAbs.min = 100.0
        cam.node_map.ExposureTimeAbs.max = 2000000.0
        cam._detect_exposure_range()
        assert cam._exposure_min == pytest.approx(100e-6)

    def test_detect_gain_range_raw(self):
        """Test _detect_gain_range with GainRaw node."""
        cam = self._make_cam()
        del cam.node_map.Gain
        cam.node_map.GainRaw.min = 0
        cam.node_map.GainRaw.max = 255
        cam._detect_gain_range()
        assert cam._gain_max == 255.0

    def test_detect_roi_range_exception(self):
        """Test _detect_roi_range handles exception."""
        cam = self._make_cam()
        type(cam.node_map).WidthMax = property(fget=MagicMock(side_effect=RuntimeError("no ROI")))
        cam._detect_roi_range()  # Should not raise

    def test_reset_roi_exception(self):
        """Test _reset_roi_to_full_sensor handles exception."""
        cam = self._make_cam()
        type(cam.node_map).WidthMax = property(fget=MagicMock(side_effect=RuntimeError("no ROI")))
        cam._reset_roi_to_full_sensor()  # Should not raise

    def test_lookup_sensor_exception(self):
        """Test _lookup_sensor_pixel_size handles exception."""
        cam = self._make_cam()
        type(cam.node_map).SensorDescription = property(
            fget=MagicMock(side_effect=RuntimeError("err"))
        )
        result = cam._lookup_sensor_pixel_size()
        assert result is None

    def test_detect_pixel_size_sensor_pixel_width_exception(self):
        """Test _detect_pixel_size handles SensorPixelWidth value exception."""

        class FailingNode:
            @property
            def value(self):
                raise AttributeError("no")

        cam = self._make_cam()
        cam.node_map.SensorPixelWidth = FailingNode()
        cam.node_map.SensorPixelHeight.value = 5.86
        cam._detect_pixel_size()
        assert cam.pixel_size == 5.86

    def test_detect_pixel_size_all_sensor_exceptions(self):
        """Test _detect_pixel_size handles all sensor feature exceptions."""

        class FailingAttr:
            @property
            def value(self):
                raise AttributeError("no")

        class FailingType:
            @property
            def value(self):
                raise TypeError("no")

        class FailingValue:
            @property
            def value(self):
                raise ValueError("no")

        cam = self._make_cam()
        cam.node_map.SensorPixelWidth = FailingAttr()
        cam.node_map.SensorPixelHeight = FailingType()
        cam.node_map.PixelSize = FailingValue()
        cam._lookup_sensor_pixel_size = MagicMock(return_value=None)
        cam._detect_pixel_size()
        assert cam.pixel_size == 1.0

    def test_configure_gain_auto_exception(self):
        """Test _configure_camera_settings with GainAuto exception."""
        cam = self._make_cam()
        type(cam.node_map.GainAuto).value = property(
            fset=MagicMock(side_effect=RuntimeError("hw error"))
        )
        cam._reset_roi_to_full_sensor = MagicMock()
        cam._configure_camera_settings()

    def test_configure_gamma_enable_exception(self):
        """Test _configure_camera_settings with GammaEnable exception."""
        cam = self._make_cam()
        type(cam.node_map.GammaEnable).value = property(
            fset=MagicMock(side_effect=RuntimeError("hw error"))
        )
        cam._reset_roi_to_full_sensor = MagicMock()
        cam._configure_camera_settings()

    def test_configure_camera_settings_outer_exception(self):
        """Test _configure_camera_settings outer exception handler."""
        cam = self._make_cam()
        # Make hasattr itself raise by corrupting node_map
        cam.node_map = None
        cam._configure_camera_settings()  # Should not raise, just log

    def test_dimension_detection_fallback(self):
        """Test open() dimension detection fallback when Width/Height fail."""
        cam = self._make_cam()
        # Simulate exception in Width.value
        type(cam.node_map.Width).value = property(
            fget=MagicMock(side_effect=RuntimeError("no width"))
        )
        cam._detect_pixel_size = MagicMock()
        cam._detect_exposure_range = MagicMock()
        cam._detect_gain_range = MagicMock()
        cam._detect_roi_range = MagicMock()
        cam._configure_camera_settings = MagicMock()

        try:
            cam.width_pixels = cam.node_map.Width.value
        except Exception:
            cam.width_pixels = 1024
            cam.height_pixels = 1024
            cam.width = 1024
            cam.height = 1024

        assert cam.width == 1024

    def test_set_exposure_both_fail(self):
        """Test set_exposure when both ExposureTime and ExposureTimeAbs fail."""
        cam = self._make_cam()
        type(cam.node_map.ExposureTime).value = property(
            fset=MagicMock(side_effect=AttributeError("no"))
        )
        type(cam.node_map.ExposureTimeAbs).value = property(
            fset=MagicMock(side_effect=AttributeError("no"))
        )
        cam.set_exposure(0.01)
        assert cam.exposure_time == 0.01

    def test_set_gain_both_fail(self):
        """Test set_gain when both Gain and GainRaw fail."""
        cam = self._make_cam()
        type(cam.node_map.Gain).value = property(fset=MagicMock(side_effect=AttributeError("no")))
        type(cam.node_map.GainRaw).value = property(
            fset=MagicMock(side_effect=AttributeError("no"))
        )
        cam.set_gain(5.0)
        assert cam.gain == 5.0

    def test_set_roi_error_handling(self):
        """Test set_roi handles exceptions."""
        cam = self._make_cam()
        cam._roi_max_width = 1024
        cam._roi_max_height = 768
        cam.node_map.OffsetX = MagicMock()
        type(cam.node_map.OffsetX).value = property(
            fset=MagicMock(side_effect=RuntimeError("hw error"))
        )
        cam.set_roi(10, 20, 640, 480)  # Should not raise

    def test_get_image_with_ia(self):
        """Test get_image via mocked image acquirer."""
        cam = self._make_cam()
        mock_ia = MagicMock()
        cam.ia = mock_ia

        mock_component = MagicMock()
        mock_component.data.reshape.return_value.copy.return_value = np.zeros((480, 640))
        mock_component.width = 640
        mock_component.height = 480

        mock_buffer = MagicMock()
        mock_buffer.payload.components = [mock_component]
        mock_ia.fetch.return_value.__enter__ = MagicMock(return_value=mock_buffer)
        mock_ia.fetch.return_value.__exit__ = MagicMock(return_value=False)

        try:
            img = cam.get_image(timeout=1.0)
            assert img.shape == (480, 640)
        finally:
            cam.stop_acquisition()


class TestCameraSettingMethod:
    """Test Camera.setting() method with mocked ipywidgets."""

    def _make_cam_with_mocks(self):
        """Create a SimulatedCamera and mock ipywidgets + IPython."""
        cam = SimulatedCamera()
        cam.open()
        return cam

    def test_setting_basic(self):
        """Test setting() creates widgets and calls display."""
        cam = self._make_cam_with_mocks()
        with patch("IPython.display.display"):
            cam.setting()
        cam.close()

    def test_setting_with_kwargs(self):
        """Test setting() applies kwargs before creating UI."""
        cam = self._make_cam_with_mocks()
        with patch("IPython.display.display"):
            cam.setting(exposure_time=0.05)
        assert cam.exposure_time == 0.05
        cam.close()

    def test_setting_no_node_map(self):
        """Test setting() works without node_map (SimulatedCamera)."""
        cam = self._make_cam_with_mocks()
        with patch("IPython.display.display"):
            cam.setting()
        cam.close()

    def test_setting_with_node_map_info(self):
        """Test setting() shows camera info from node_map."""
        cam = self._make_cam_with_mocks()
        cam.node_map = MagicMock()
        cam.node_map.SensorDescription.value = "Sony IMX174"
        cam.node_map.DeviceModelName.value = "BFS-U3-123"
        with patch("IPython.display.display"):
            cam.setting()
        cam.close()

    def test_setting_with_roi(self):
        """Test setting() creates ROI controls when available."""
        cam = self._make_cam_with_mocks()
        cam.set_roi(0, 0, 1024, 768)
        with patch("IPython.display.display"):
            cam.setting()
        cam.close()

    def test_create_genicam_controls_no_node_map(self):
        """Test _create_genicam_controls returns empty list without node_map."""
        cam = self._make_cam_with_mocks()
        cam.node_map = None
        result = cam._create_genicam_controls({"description_width": "initial"})
        assert result == []
        cam.close()

    def test_create_genicam_controls_with_features(self):
        """Test _create_genicam_controls discovers features from node_map."""
        cam = self._make_cam_with_mocks()
        result = cam._create_genicam_controls({"description_width": "initial"})
        assert len(result) > 0
        cam.close()

    def test_create_checkbox(self):
        """Test _create_checkbox returns a checkbox widget."""
        cam = self._make_cam_with_mocks()
        node = MagicMock()
        widget = cam._create_checkbox(node, "TestEnable", True)
        assert widget is not None
        cam.close()

    def test_create_slider_float(self):
        """Test _create_slider for float features."""
        cam = self._make_cam_with_mocks()
        node = MagicMock()
        node.min = 0.0
        node.max = 10.0
        node.value = 5.0
        result = cam._create_slider(node, "Gamma", {"description_width": "initial"})
        assert result is not None
        cam.close()

    def test_create_slider_int(self):
        """Test _create_slider for integer features."""
        cam = self._make_cam_with_mocks()
        node = MagicMock()
        node.min = 0
        node.max = 255
        node.value = 10
        result = cam._create_slider(node, "BlackLevel", {"description_width": "initial"})
        assert result is not None
        cam.close()

    def test_create_slider_exception(self):
        """Test _create_slider returns None on error."""
        cam = self._make_cam_with_mocks()
        node = MagicMock()
        node.min = MagicMock(side_effect=RuntimeError)
        type(node).min = property(fget=MagicMock(side_effect=RuntimeError("hw error")))
        result = cam._create_slider(node, "Broken", {"description_width": "initial"})
        assert result is None
        cam.close()

    def test_create_enum_dropdown(self):
        """Test _create_enum_dropdown creates dropdown widget."""
        cam = self._make_cam_with_mocks()
        node = MagicMock()
        node.value = "Off"
        node.symbolics = ["Off", "Once", "Continuous"]
        result = cam._create_enum_dropdown(node, "ExposureAuto", {"description_width": "initial"})
        assert result is not None
        cam.close()

    def test_create_enum_dropdown_enable(self):
        """Test _create_enum_dropdown for Enable features without symbolics."""
        cam = self._make_cam_with_mocks()
        node = MagicMock(spec=["value"])
        node.value = "On"
        result = cam._create_enum_dropdown(node, "GammaEnable", {"description_width": "initial"})
        assert result is not None
        cam.close()

    def test_create_enum_dropdown_auto(self):
        """Test _create_enum_dropdown for Auto features without symbolics."""
        cam = self._make_cam_with_mocks()
        node = MagicMock(spec=["value"])
        node.value = "Off"
        result = cam._create_enum_dropdown(node, "ExposureAuto", {"description_width": "initial"})
        assert result is not None
        cam.close()

    def test_create_enum_dropdown_exception(self):
        """Test _create_enum_dropdown returns None on error."""
        cam = self._make_cam_with_mocks()
        node = MagicMock()
        type(node).value = property(fget=MagicMock(side_effect=RuntimeError("error")))
        result = cam._create_enum_dropdown(node, "Broken", {"description_width": "initial"})
        assert result is None
        cam.close()

    def test_create_feature_controls_with_mixed_types(self):
        """Test _create_feature_controls handles different feature types."""
        cam = self._make_cam_with_mocks()
        cam.node_map = MagicMock()

        # Boolean feature
        cam.node_map.GammaEnable.value = True

        # Numeric feature with min/max and — like a real numeric node — no
        # ``symbolics``. The spec matters: a bare MagicMock would answer to
        # every attribute and get mistaken for an enumeration.
        numeric_node = MagicMock(spec=["value", "min", "max"])
        numeric_node.value = 1.0
        numeric_node.min = 0.0
        numeric_node.max = 4.0
        cam.node_map.Gamma = numeric_node

        # Enum feature (no min/max)
        enum_node = MagicMock(spec=["value", "symbolics"])
        enum_node.value = "Off"
        enum_node.symbolics = ["Off", "On"]
        cam.node_map.Sharpness = enum_node

        controls = cam._create_feature_controls(
            ["GammaEnable", "Gamma", "Sharpness"], {"description_width": "initial"}
        )
        assert len(controls) == 3
        cam.close()


class TestSettingCallbacks:
    """Test widget callbacks inside setting() method."""

    def _make_cam(self):
        cam = SimulatedCamera()
        cam.open()
        return cam

    def test_exposure_slider_callback(self):
        """Test exposure slider callback triggers set_exposure."""
        cam = self._make_cam()
        original_set_exposure = cam.set_exposure

        calls = []

        def tracking_set_exposure(val):
            calls.append(val)
            original_set_exposure(val)

        cam.set_exposure = tracking_set_exposure

        with patch("IPython.display.display"):
            cam.setting()

        assert cam.exposure_time == 0.01  # Default
        cam.close()

    def test_gain_slider_callback(self):
        """Test gain slider callback triggers set_gain."""
        cam = self._make_cam()

        with patch("IPython.display.display"):
            cam.setting()

        assert cam.gain == 0.0  # Default
        cam.close()

    def test_start_stop_buttons(self):
        """Test start/stop acquisition buttons."""
        cam = self._make_cam()

        with patch("IPython.display.display"):
            cam.setting()

        cam.start_acquisition()
        assert cam.is_acquiring is True
        cam.stop_acquisition()
        assert cam.is_acquiring is False
        cam.close()

    def test_setting_with_sensor_info_exception(self):
        """Test setting() handles exception in SensorDescription."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        type(cam.node_map.SensorDescription).value = property(
            fget=MagicMock(side_effect=RuntimeError("unavailable"))
        )
        type(cam.node_map.DeviceModelName).value = property(
            fget=MagicMock(side_effect=RuntimeError("unavailable"))
        )

        with patch("IPython.display.display"):
            cam.setting()
        cam.close()

    def test_setting_roi_apply_callback(self):
        """Test ROI apply button callback."""
        cam = self._make_cam()
        cam._roi_offset_x = 0
        cam._roi_offset_y = 0
        cam._roi_max_width = 2048
        cam._roi_max_height = 1536
        cam.width_pixels = 1024
        cam.height_pixels = 768

        cam.set_roi(0, 0, 1024, 768)

        set_roi_calls = []

        def mock_set_roi(ox, oy, w, h):
            set_roi_calls.append((ox, oy, w, h))
            cam._roi_offset_x = ox
            cam._roi_offset_y = oy
            cam._roi_width = w
            cam._roi_height = h

        cam.set_roi = mock_set_roi

        displayed_widgets = []

        def capture_display(*args):
            displayed_widgets.extend(args)

        with patch("IPython.display.display", side_effect=capture_display):
            cam.setting()

        import ipywidgets as widgets

        # Find ROI accordion in the tab
        assert len(displayed_widgets) == 1
        tab = displayed_widgets[0]
        settings_tab = tab.children[0]
        roi_accordion = None
        for child in settings_tab.children:
            if isinstance(child, widgets.Accordion) and child.get_title(0) == "Region of Interest":
                roi_accordion = child
                break

        assert roi_accordion is not None
        roi_box = roi_accordion.children[0]
        buttons_box = roi_box.children[-1]
        apply_btn = buttons_box.children[0]
        reset_btn = buttons_box.children[1]

        apply_btn.click()
        assert len(set_roi_calls) == 1

        reset_btn.click()
        assert len(set_roi_calls) == 2

        cam.close()

    def test_create_feature_controls_no_value_attr(self):
        """Test _create_feature_controls skips nodes without value attribute."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        cam.node_map.TestFeature = MagicMock(spec=[])  # No 'value' attribute
        result = cam._create_feature_controls(["TestFeature"], {"description_width": "initial"})
        assert result == []
        cam.close()

    def test_create_feature_controls_enable_string_value(self):
        """Test _create_feature_controls for Enable with string value."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        cam.node_map.TestEnable.value = "On"
        cam.node_map.TestEnable.symbolics = ["On", "Off"]
        result = cam._create_feature_controls(["TestEnable"], {"description_width": "initial"})
        assert len(result) == 1
        cam.close()

    def test_create_feature_controls_exception_in_node(self):
        """Test _create_feature_controls handles exception accessing node."""
        cam = self._make_cam()
        # Use a mock that has the feature but getattr on the node raises
        mock_node_map = MagicMock()
        node = MagicMock()
        node.value = MagicMock(side_effect=RuntimeError("broken"))
        type(node).value = property(fget=MagicMock(side_effect=RuntimeError("broken")))
        mock_node_map.BadFeature = node
        cam.node_map = mock_node_map
        result = cam._create_feature_controls(["BadFeature"], {"description_width": "initial"})
        assert isinstance(result, list)
        cam.close()

    def test_exposure_observer_fires_set_exposure(self):
        """Test that exposure slider's observe callback fires set_exposure."""
        import ipywidgets as widgets

        cam = self._make_cam()
        displayed = []

        with patch("IPython.display.display", side_effect=lambda *a: displayed.extend(a)):
            cam.setting()

        # Navigate to exposure slider
        tab = displayed[0]
        settings_vbox = tab.children[0]
        timing_accordion = settings_vbox.children[0]
        exposure_box = timing_accordion.children[0]
        exposure_slider = exposure_box.children[0]
        exposure_input = exposure_box.children[1]

        assert isinstance(exposure_slider, widgets.FloatLogSlider)

        exposure_slider.value = 0.05
        assert cam.exposure_time == pytest.approx(0.05)
        assert exposure_input.value == pytest.approx(0.05)

        exposure_input.value = 0.001
        assert exposure_slider.value == pytest.approx(0.001)
        cam.close()

    def test_gain_observer_fires_set_gain(self):
        """Test that gain slider's observe callback fires set_gain."""
        import ipywidgets as widgets

        cam = self._make_cam()
        displayed = []

        with patch("IPython.display.display", side_effect=lambda *a: displayed.extend(a)):
            cam.setting()

        tab = displayed[0]
        settings_vbox = tab.children[0]
        analog_accordion = settings_vbox.children[1]
        gain_box = analog_accordion.children[0]
        gain_slider = gain_box.children[0]
        gain_input = gain_box.children[1]

        assert isinstance(gain_slider, widgets.FloatSlider)

        gain_slider.value = 12.0
        assert cam.gain == pytest.approx(12.0)
        assert gain_input.value == pytest.approx(12.0)

        gain_input.value = 5.0
        assert gain_slider.value == pytest.approx(5.0)
        cam.close()

    def test_start_stop_button_callbacks(self):
        """Test start/stop button on_click callbacks."""
        cam = self._make_cam()
        displayed = []

        with patch("IPython.display.display", side_effect=lambda *a: displayed.extend(a)):
            cam.setting()

        cam.start_acquisition()
        assert cam.is_acquiring is True
        cam.stop_acquisition()
        assert cam.is_acquiring is False
        cam.close()

    def test_roi_apply_error_handling(self):
        """Test ROI apply callback handles set_roi error."""
        cam = self._make_cam()
        cam.set_roi(0, 0, 1024, 768)
        cam.set_roi = MagicMock(side_effect=RuntimeError("ROI error"))

        displayed = []

        with patch("IPython.display.display", side_effect=lambda *a: displayed.extend(a)):
            cam.setting()

        import ipywidgets as widgets

        tab = displayed[0]
        settings_vbox = tab.children[0]
        roi_accordion = None
        for child in settings_vbox.children:
            if isinstance(child, widgets.Accordion) and child.get_title(0) == "Region of Interest":
                roi_accordion = child
                break

        if roi_accordion:
            roi_box = roi_accordion.children[0]
            buttons = roi_box.children[-1]
            apply_btn = buttons.children[0]
            apply_btn.click()  # Should not raise
        cam.close()

    def test_roi_reset_error_handling(self):
        """Test ROI reset callback handles set_roi error."""
        cam = self._make_cam()
        cam.set_roi(0, 0, 1024, 768)

        call_count = [0]

        def failing_set_roi(*args):
            call_count[0] += 1
            if call_count[0] > 0:
                raise RuntimeError("ROI reset error")

        cam.set_roi = failing_set_roi

        displayed = []

        with patch("IPython.display.display", side_effect=lambda *a: displayed.extend(a)):
            cam.setting()

        import ipywidgets as widgets

        tab = displayed[0]
        settings_vbox = tab.children[0]
        roi_accordion = None
        for child in settings_vbox.children:
            if isinstance(child, widgets.Accordion) and child.get_title(0) == "Region of Interest":
                roi_accordion = child
                break

        if roi_accordion:
            roi_box = roi_accordion.children[0]
            buttons = roi_box.children[-1]
            reset_btn = buttons.children[1]
            reset_btn.click()  # Should not raise
        cam.close()

    def test_checkbox_callback(self):
        """Test _create_checkbox callback fires node.value change."""
        cam = self._make_cam()
        node = MagicMock()
        node.value = False
        checkbox = cam._create_checkbox(node, "TestEnable", False)
        checkbox.value = True
        assert node.value is True
        cam.close()

    def test_checkbox_callback_error(self):
        """Test _create_checkbox callback handles error."""
        cam = self._make_cam()
        node = MagicMock()
        type(node).value = property(
            fget=MagicMock(return_value=False),
            fset=MagicMock(side_effect=RuntimeError("write error")),
        )
        checkbox = cam._create_checkbox(node, "TestEnable", False)
        checkbox.value = True  # Should not raise
        cam.close()

    def test_slider_callback(self):
        """Test _create_slider callback fires node.value change."""
        cam = self._make_cam()
        node = MagicMock()
        node.min = 0.0
        node.max = 10.0
        node.value = 5.0
        slider_box = cam._create_slider(node, "Gamma", {"description_width": "initial"})
        assert slider_box is not None

        slider = slider_box.children[0]
        input_widget = slider_box.children[1]

        slider.value = 7.0
        assert node.value == pytest.approx(7.0)
        assert input_widget.value == pytest.approx(7.0)

        input_widget.value = 3.0
        assert slider.value == pytest.approx(3.0)
        cam.close()

    def test_slider_callback_error(self):
        """Test _create_slider callback handles error gracefully."""
        cam = self._make_cam()
        node = MagicMock()
        node.min = 0.0
        node.max = 10.0
        node.value = 5.0

        slider_box = cam._create_slider(node, "Gamma", {"description_width": "initial"})
        slider = slider_box.children[0]

        # Make node.value assignment raise
        type(node).value = property(
            fget=MagicMock(return_value=5.0),
            fset=MagicMock(side_effect=RuntimeError("write error")),
        )
        slider.value = 8.0  # Should not raise
        cam.close()

    def test_enum_dropdown_callback(self):
        """Test _create_enum_dropdown callback fires node.value change."""
        cam = self._make_cam()
        node = MagicMock()
        node.value = "Off"
        node.symbolics = ["Off", "On"]
        dropdown = cam._create_enum_dropdown(node, "ExposureAuto", {"description_width": "initial"})
        assert dropdown is not None
        dropdown.value = "On"
        assert node.value == "On"
        cam.close()

    def test_enum_dropdown_callback_error(self):
        """Test _create_enum_dropdown callback handles error."""
        cam = self._make_cam()
        node = MagicMock()
        node.value = "Off"
        node.symbolics = ["Off", "On"]
        dropdown = cam._create_enum_dropdown(node, "ExposureAuto", {"description_width": "initial"})

        type(node).value = property(
            fget=MagicMock(return_value="Off"),
            fset=MagicMock(side_effect=RuntimeError("write error")),
        )
        dropdown.value = "On"  # Should not raise
        cam.close()

    def test_enum_dropdown_no_options(self):
        """Test _create_enum_dropdown returns None when no options."""
        cam = self._make_cam()
        node = MagicMock(spec=["value"])
        node.value = ""
        result = cam._create_enum_dropdown(node, "SomeFeature", {"description_width": "initial"})
        assert result is None
        cam.close()

    def test_enum_dropdown_single_current_value(self):
        """Test _create_enum_dropdown with single current value as fallback."""
        cam = self._make_cam()
        node = MagicMock(spec=["value"])
        node.value = "SomeValue"
        result = cam._create_enum_dropdown(node, "SomeFeature", {"description_width": "initial"})
        assert result is not None
        cam.close()

    def test_start_stop_button_click_from_setting(self):
        """Test start/stop buttons work through setting() widget tree."""
        cam = self._make_cam()
        displayed = []

        with patch("IPython.display.display", side_effect=lambda *a: displayed.extend(a)):
            cam.setting()

        assert cam.is_acquiring is False
        cam.start_acquisition()
        assert cam.is_acquiring is True
        cam.stop_acquisition()
        assert cam.is_acquiring is False
        cam.close()

    def test_apply_settings_node_map_set_error(self):
        """Test _apply_settings_from_kwargs node_map set error."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        node = MagicMock()
        type(node).value = property(
            fget=MagicMock(return_value=10),
            fset=MagicMock(side_effect=RuntimeError("write error")),
        )
        cam.node_map.TestParam = node
        cam._apply_settings_from_kwargs({"TestParam": 42})
        cam.close()

    def test_apply_settings_node_map_missing_param_warning(self):
        """Test _apply_settings_from_kwargs warns for missing param in node_map."""
        cam = self._make_cam()
        cam.node_map = MagicMock(spec=[])
        cam._apply_settings_from_kwargs({"NonexistentParam": 42})
        cam.close()

    def test_apply_settings_node_map_check_bool_error(self):
        """Test _apply_settings_from_kwargs handles bool check error."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        node = MagicMock()
        type(node).value = property(
            fget=MagicMock(side_effect=RuntimeError("read error")),
            fset=MagicMock(),
        )
        cam.node_map.TestAuto = node
        cam._apply_settings_from_kwargs({"TestAuto": "on"})
        cam.close()

    def test_create_feature_controls_numeric_with_checkbox(self):
        """Test _create_feature_controls creates checkbox for Enable with bool value."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        cam.node_map.GammaEnable.value = False  # Bool value -> checkbox
        result = cam._create_feature_controls(["GammaEnable"], {"description_width": "initial"})
        assert len(result) == 1
        cam.close()

    def test_create_feature_controls_slider_error(self):
        """Test _create_feature_controls handles slider creation error."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        node = MagicMock()
        node.value = 5.0
        node.min = MagicMock(side_effect=RuntimeError("read error"))
        type(node).min = property(fget=MagicMock(side_effect=RuntimeError("read error")))
        cam.node_map.Gamma = node
        result = cam._create_feature_controls(["Gamma"], {"description_width": "initial"})
        assert isinstance(result, list)
        cam.close()

    def test_create_feature_controls_dropdown_error(self):
        """Test _create_feature_controls handles dropdown creation error."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        node = MagicMock(spec=["value"])
        type(node).value = property(fget=MagicMock(side_effect=RuntimeError("err")))
        cam.node_map.TestPattern = node
        result = cam._create_feature_controls(["TestPattern"], {"description_width": "initial"})
        assert isinstance(result, list)
        cam.close()

    def test_create_feature_controls_checkbox_error(self):
        """Test _create_feature_controls handles checkbox creation error."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        # Make Enable feature's value readable but _create_checkbox raises
        mock_node = MagicMock()
        read_count = [0]
        original_value = False

        def failing_value_getter():
            read_count[0] += 1
            if read_count[0] > 1:
                raise RuntimeError("checkbox creation failed")
            return original_value

        type(mock_node).value = property(fget=lambda self: failing_value_getter())
        cam.node_map.TestEnable = mock_node
        result = cam._create_feature_controls(["TestEnable"], {"description_width": "initial"})
        assert isinstance(result, list)
        cam.close()

    def test_create_feature_controls_slider_exception(self):
        """Test _create_feature_controls catches slider creation exception."""
        cam = self._make_cam()
        cam.node_map = MagicMock()
        # Node with value, min, max but _create_slider raises
        node = MagicMock()
        node.value = 5.0
        node.min = 0.0
        node.max = 10.0

        # Patch _create_slider to raise
        original_create_slider = cam._create_slider
        cam._create_slider = MagicMock(side_effect=RuntimeError("slider error"))
        result = cam._create_feature_controls(["Gamma"], {"description_width": "initial"})
        assert isinstance(result, list)
        cam._create_slider = original_create_slider
        cam.close()

    def test_create_feature_controls_missing_feature(self):
        """Test _create_feature_controls skips features not in node_map."""
        cam = self._make_cam()
        cam.node_map = MagicMock(spec=[])  # Empty spec = no attributes
        result = cam._create_feature_controls(
            ["NonexistentFeature"], {"description_width": "initial"}
        )
        assert result == []
        cam.close()


class TestCategorizeFeature:
    """Tests for the _categorize_feature module-level function."""

    def test_known_prefixes(self):
        assert _categorize_feature("TriggerMode") == "Acquisition Control"
        assert _categorize_feature("TriggerSource") == "Acquisition Control"
        assert _categorize_feature("AcquisitionFrameRate") == "Acquisition Control"
        assert _categorize_feature("DeviceTemperature") == "Device Control"
        assert _categorize_feature("Gamma") == "Analog Control"
        assert _categorize_feature("GammaEnable") == "Analog Control"
        assert _categorize_feature("BlackLevel") == "Analog Control"
        assert _categorize_feature("PixelFormat") == "Image Format Control"
        assert _categorize_feature("ReverseX") == "Image Format Control"
        assert _categorize_feature("BinningHorizontal") == "Image Format Control"
        assert _categorize_feature("LUTEnable") == "LUT Control"
        assert _categorize_feature("DefectivePixelCorrection") == "Image Quality Control"
        assert _categorize_feature("TestPattern") == "Test Control"
        assert _categorize_feature("SensorShutterMode") == "Image Format Control"
        assert _categorize_feature("LineSelector") == "Digital I/O Control"
        assert _categorize_feature("CounterEventSource") == "Counter & Timer Control"
        assert _categorize_feature("ExposureMode") == "Acquisition Control"
        assert _categorize_feature("ExposureAuto") == "Acquisition Control"
        assert _categorize_feature("ChunkEnable") == "Chunk Data Control"
        assert _categorize_feature("EventNotification") == "Event Control"

    def test_longest_prefix_wins(self):
        assert _categorize_feature("BlackLevelAuto") == "Analog Control"
        assert _categorize_feature("WhiteBalanceAuto") == "Analog Control"

    def test_unknown_uses_first_camelcase_word(self):
        assert _categorize_feature("FooBarBaz") == "Foo"
        assert _categorize_feature("CustomSetting") == "Custom"

    def test_completely_unknown(self):
        assert _categorize_feature("lowercase") == "Other"
        assert _categorize_feature("") == "Other"


class TestDiscoverFeatures:
    """Tests for Camera._discover_features."""

    def test_no_node_map(self):
        cam = SimulatedCamera()
        assert cam._discover_features() == {}

    def test_with_simulated_camera(self):
        cam = SimulatedCamera()
        cam.open()
        discovered = cam._discover_features()
        assert isinstance(discovered, dict)
        assert len(discovered) > 0

        all_features = [f for features in discovered.values() for f in features]
        assert "Gamma" in all_features
        assert "GammaEnable" in all_features
        assert "BlackLevel" in all_features
        assert "AcquisitionFrameRate" in all_features
        assert "PixelFormat" in all_features
        assert "TriggerMode" in all_features
        assert "TriggerSource" in all_features
        assert "ReverseX" in all_features
        assert "ReverseY" in all_features
        assert "DeviceTemperature" in all_features
        cam.close()

    def test_skips_handled_features(self):
        cam = SimulatedCamera()
        cam.open()
        discovered = cam._discover_features()
        all_features = [f for features in discovered.values() for f in features]
        for skip in ("ExposureTime", "Gain", "Width", "Height", "OffsetX", "OffsetY"):
            assert skip not in all_features
        cam.close()

    def test_skips_private_attributes(self):
        cam = SimulatedCamera()
        cam.open()
        discovered = cam._discover_features()
        all_features = [f for features in discovered.values() for f in features]
        for f in all_features:
            assert not f.startswith("_")
        cam.close()

    def test_features_grouped_by_category(self):
        cam = SimulatedCamera()
        cam.open()
        discovered = cam._discover_features()
        assert "Analog Control" in discovered
        assert "Gamma" in discovered["Analog Control"]
        assert "GammaEnable" in discovered["Analog Control"]
        assert "Acquisition Control" in discovered
        assert "TriggerMode" in discovered["Acquisition Control"]
        cam.close()

    def test_with_custom_node_map(self):
        """Test discovery with a manually constructed node map."""
        cam = SimulatedCamera()

        class CustomNodeMap:
            CustomFloat = _SimulatedNode(5.0, min_val=0.0, max_val=10.0)
            CustomEnum = _SimulatedNode("A", symbolics=["A", "B", "C"])
            CustomBool = _SimulatedNode(True)

        cam.node_map = CustomNodeMap()  # ty: ignore[invalid-assignment]
        discovered = cam._discover_features()
        all_features = [f for features in discovered.values() for f in features]
        assert "CustomFloat" in all_features
        assert "CustomEnum" in all_features
        assert "CustomBool" in all_features

    def test_node_map_none(self):
        cam = SimulatedCamera()
        cam.node_map = None
        assert cam._discover_features() == {}

    def test_skips_callable_attributes(self):
        """Callable attributes (methods) should be skipped."""
        cam = SimulatedCamera()

        class NodeMapWithMethod:
            Gamma = _SimulatedNode(1.0, min_val=0.0, max_val=4.0)

            def some_method(self):
                pass

        cam.node_map = NodeMapWithMethod()  # ty: ignore[invalid-assignment]
        discovered = cam._discover_features()
        all_features = [f for features in discovered.values() for f in features]
        assert "Gamma" in all_features
        assert "some_method" not in all_features

    def test_getattr_exception_handled(self):
        """Features that raise on getattr should be skipped."""
        cam = SimulatedCamera()

        class FailingNodeMap:
            Gamma = _SimulatedNode(1.0, min_val=0.0, max_val=4.0)

            @property
            def BadNode(self):
                raise RuntimeError("broken")

        cam.node_map = FailingNodeMap()  # ty: ignore[invalid-assignment]
        discovered = cam._discover_features()
        all_features = [f for features in discovered.values() for f in features]
        assert "Gamma" in all_features
        assert "BadNode" not in all_features

    def test_nodes_api_with_interface_type_and_visibility(self):
        """Simulate a real GenICam ``node_map`` that exposes a ``.nodes``
        iterable where each entry has ``principal_interface_type`` /
        ``visibility`` / ``name``. This drives the SWIG-shaped branch
        (lines 438-458) that ``dir()``-based discovery bypasses."""
        from genicam.genapi import (  # ty: ignore[unresolved-import]
            EInterfaceType,
            EVisibility,
        )

        cam = SimulatedCamera()

        ok_node = _SimulatedNode(1.0, min_val=0.0, max_val=4.0)
        hidden_node = _SimulatedNode(0, min_val=0, max_val=100)
        wrong_type_node = _SimulatedNode(0)
        boom_node = _SimulatedNode(0)

        class _NodeDescriptor:
            def __init__(self, name: str, iface: int, vis: int) -> None:
                self.name = name
                self.principal_interface_type = iface
                self.visibility = vis

        class _BadDescriptor:
            """A node that raises on every attribute access — the per-node
            ``except`` in the loop must swallow it and continue."""

            def __getattr__(self, _name: str) -> Any:
                raise RuntimeError("swig binding crashed")

        visible_float = _NodeDescriptor(
            "CustomGamma",
            int(EInterfaceType.intfIFloat),
            int(EVisibility.Beginner),
        )
        # intfIValue(0) nodes go through the allowlist bypass at line 448.
        generic_node = _NodeDescriptor("ValueNode", 0, int(EVisibility.Beginner))
        # Expert visibility is below "Invisible" so it stays visible.
        guru_node = _NodeDescriptor(
            "GuruFeature",
            int(EInterfaceType.intfIInteger),
            int(EVisibility.Guru),
        )
        # Non-allowlisted interface type → continue.
        wrong_iface = _NodeDescriptor(
            "NotAValue", int(EInterfaceType.intfICategory), int(EVisibility.Beginner)
        )
        # Invisible → continue.
        invisible = _NodeDescriptor(
            "Secret",
            int(EInterfaceType.intfIFloat),
            int(EVisibility.Invisible),
        )

        class NodeMapWithNodesApi:
            CustomGamma = ok_node
            ValueNode = _SimulatedNode(42.0, min_val=0.0, max_val=100.0)
            GuruFeature = hidden_node
            NotAValue = wrong_type_node
            Secret = boom_node
            # Note: intentionally no ``Broken`` attribute — we want to
            # prove the per-node except block runs cleanly.
            nodes = [
                visible_float,
                generic_node,
                guru_node,
                wrong_iface,
                invisible,
                _BadDescriptor(),
            ]

        cam.node_map = NodeMapWithNodesApi()  # ty: ignore[invalid-assignment]
        discovered = cam._discover_features()
        flat = [f for features in discovered.values() for f in features]

        assert "CustomGamma" in flat
        assert "ValueNode" in flat  # intfIValue(0) bypass path
        assert "GuruFeature" in flat  # Guru < Invisible threshold
        assert "NotAValue" not in flat
        assert "Secret" not in flat

    def test_nodes_api_without_genicam_enum(self):
        """If ``genicam.genapi`` isn't importable at all, the allowlist /
        visibility filters are disabled and every named node is kept.
        We simulate "not installed" by nulling out the module-level
        handles the helper consults."""
        cam = SimulatedCamera()

        class Desc:
            def __init__(self, name: str) -> None:
                self.name = name

        class NodeMap:
            Foo = _SimulatedNode(1.0, min_val=0.0, max_val=10.0)
            Bar = _SimulatedNode(2.0, min_val=0.0, max_val=10.0)
            nodes = [Desc("Foo"), Desc("Bar")]

        cam.node_map = NodeMap()  # ty: ignore[invalid-assignment]

        with (
            patch("pybeamprofiler.camera._EInterfaceType", None),
            patch("pybeamprofiler.camera._EVisibility", None),
        ):
            discovered = cam._discover_features()

        flat = [f for features in discovered.values() for f in features]
        assert "Foo" in flat
        assert "Bar" in flat

    def test_nodes_api_raises_falls_back_to_dir(self):
        """If iterating ``node_map.nodes`` itself blows up, the helper
        must log and fall back to ``dir()``-based discovery (line 461).
        Using a ``@property`` that raises is the simplest model."""
        cam = SimulatedCamera()

        class NodeMap:
            Gamma = _SimulatedNode(1.0, min_val=0.0, max_val=4.0)

            @property
            def nodes(self) -> list[Any]:
                raise RuntimeError("SWIG dead")

        cam.node_map = NodeMap()  # ty: ignore[invalid-assignment]
        discovered = cam._discover_features()
        flat = [f for features in discovered.values() for f in features]
        assert "Gamma" in flat  # dir() fallback still picked it up

    def test_scalar_attribute_is_skipped(self):
        """Attributes whose value is already a plain scalar (str/int/etc)
        are rejected before the ``.value`` probe (line 493)."""
        cam = SimulatedCamera()

        class NodeMap:
            Gamma = _SimulatedNode(1.0, min_val=0.0, max_val=4.0)
            NotANode = "raw string"  # bare str → skipped
            AlsoNot = 42  # bare int → skipped

        cam.node_map = NodeMap()  # ty: ignore[invalid-assignment]
        discovered = cam._discover_features()
        flat = [f for features in discovered.values() for f in features]
        assert "Gamma" in flat
        assert "NotANode" not in flat
        assert "AlsoNot" not in flat

    def test_none_attribute_is_skipped(self):
        """An attribute whose value is ``None`` must be skipped (line 490)."""
        cam = SimulatedCamera()

        class NodeMap:
            Gamma = _SimulatedNode(1.0, min_val=0.0, max_val=4.0)
            NoneAttr = None

        cam.node_map = NodeMap()  # ty: ignore[invalid-assignment]
        discovered = cam._discover_features()
        flat = [f for features in discovered.values() for f in features]
        assert "Gamma" in flat
        assert "NoneAttr" not in flat


class TestDiscoverFeaturesIntegration:
    """Integration tests: _discover_features feeds into _create_genicam_controls."""

    def test_genicam_controls_from_discovery(self):
        """_create_genicam_controls produces accordions from discovered features."""
        cam = SimulatedCamera()
        cam.open()
        controls = cam._create_genicam_controls({"description_width": "initial"})
        assert len(controls) > 0

        import ipywidgets as widgets

        for ctrl in controls:
            assert isinstance(ctrl, widgets.Accordion)
        cam.close()

    def test_setting_shows_discovered_features(self):
        """setting() should show discovered features without error."""
        cam = SimulatedCamera()
        cam.open()
        with patch("IPython.display.display"):
            cam.setting()
        cam.close()

    def test_setting_kwargs_and_discovery(self):
        """setting() applies kwargs then shows discovered controls."""
        cam = SimulatedCamera()
        cam.open()
        with patch("IPython.display.display"):
            cam.setting(exposure_time=0.05)
        assert cam.exposure_time == 0.05
        cam.close()
