"""Tests for defensive branches: fallbacks, guards, and cleanup on failure.

These paths only run when hardware or the environment misbehaves, so they are
driven with mocks. They matter precisely because they are what stands between
a flaky camera and a crashed session.
"""

from __future__ import annotations

import signal
import types
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pybeamprofiler.beamprofiler import BeamProfiler
from pybeamprofiler.gen_camera import HarvesterCamera
from pybeamprofiler.simulated import SimulatedCamera


class _Mocks(NamedTuple):
    """The mocks behind a fake camera.

    Assertions go through these rather than ``cam.h`` / ``cam.ia`` so the type
    checker sees ``MagicMock`` instead of the real Harvester API.
    """

    h: MagicMock
    ia: MagicMock
    node_map: MagicMock


def _mock_harvester_camera() -> tuple[HarvesterCamera, _Mocks]:
    """A HarvesterCamera wired to mocks, built far enough to drive its methods."""
    with patch("pybeamprofiler.gen_camera.Harvester"):
        cam = HarvesterCamera()
    mocks = _Mocks(h=MagicMock(), ia=MagicMock(), node_map=MagicMock())
    cam.h = mocks.h
    cam.ia = mocks.ia
    cam.node_map = mocks.node_map
    return cam, mocks


class TestStaticImageEdgeCases:
    def test_two_channel_image_takes_the_first_channel(self, tmp_path):
        """Grayscale+alpha has no colour to weight — just drop the alpha."""
        la = np.zeros((16, 24, 2), dtype=np.uint8)
        la[..., 0] = 77
        la[..., 1] = 255

        bp = BeamProfiler.__new__(BeamProfiler)
        with patch("pybeamprofiler.beamprofiler.Image.open") as mock_open:
            mock_open.return_value.__enter__.return_value = la
            with patch("pybeamprofiler.beamprofiler.np.array", return_value=la):
                bp._load_file("dummy.png")

        assert bp.last_img is not None
        assert bp.last_img.shape == (16, 24)
        assert bp.last_img.max() == 77

    def test_unsupported_dimensionality_is_rejected(self):
        weird = np.zeros((2, 3, 4, 5), dtype=np.uint8)
        bp = BeamProfiler.__new__(BeamProfiler)
        with patch("pybeamprofiler.beamprofiler.Image.open") as mock_open:
            mock_open.return_value.__enter__.return_value = weird
            with patch("pybeamprofiler.beamprofiler.np.array", return_value=weird):
                with pytest.raises(ValueError, match="2D or 3D"):
                    bp._load_file("dummy.png")

    def test_unreadable_file_propagates(self, tmp_path):
        bp = BeamProfiler.__new__(BeamProfiler)
        with pytest.raises(Exception):
            bp._load_file(str(tmp_path / "does-not-exist.png"))


class TestInitGuards:
    def test_no_camera_and_no_file_is_an_error(self):
        """If camera setup silently yields nothing, say so rather than limp on."""
        with patch.object(BeamProfiler, "_initialize_camera", lambda self, camera: None):
            with pytest.raises(ValueError, match="Either camera or file"):
                BeamProfiler(camera="simulated")


class TestSigintHandlerOffMainThread:
    """``signal.signal`` raises ValueError outside the main thread; the Dash
    branch must degrade gracefully instead of taking the server down."""

    def test_plot_stream_survives_when_handler_cannot_be_installed(self):
        bp = BeamProfiler(camera="simulated")
        fake_app = MagicMock()

        with (
            patch("pybeamprofiler.dash_app.create_app", return_value=fake_app),
            patch("pybeamprofiler.beamprofiler.signal.signal", side_effect=ValueError("not main")),
            patch("pybeamprofiler.beamprofiler.webbrowser.open"),
        ):
            # Force the non-Jupyter path.
            with patch.dict("sys.modules", {"IPython": None}):
                bp._plot_stream()

        fake_app.run.assert_called_once()
        assert bp.camera is not None
        bp.camera.close()

    def test_handler_is_restored_on_the_main_thread(self):
        bp = BeamProfiler(camera="simulated")
        fake_app = MagicMock()
        original = signal.getsignal(signal.SIGINT)

        with (
            patch("pybeamprofiler.dash_app.create_app", return_value=fake_app),
            patch("pybeamprofiler.beamprofiler.webbrowser.open"),
            patch.dict("sys.modules", {"IPython": None}),
        ):
            bp._plot_stream()

        assert signal.getsignal(signal.SIGINT) is original
        assert bp.camera is not None
        bp.camera.close()


class TestHarvesterCameraFallbacks:
    def test_dimension_readback_failure_falls_back_to_1024(self):
        cam, mocks = _mock_harvester_camera()
        device = types.SimpleNamespace(model="M", vendor="V", serial_number="1")
        mocks.h.device_info_list = [device]
        mocks.h.files = []
        mocks.ia.remote_device.node_map = cam.node_map
        type(mocks.node_map).Width = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("no"))
        )

        with (
            patch.object(HarvesterCamera, "_configure_gige_stream"),
            patch.object(HarvesterCamera, "_configure_camera_settings"),
            patch.object(HarvesterCamera, "_detect_pixel_size"),
            patch.object(HarvesterCamera, "_detect_exposure_range"),
            patch.object(HarvesterCamera, "_detect_gain_range"),
            patch.object(HarvesterCamera, "_detect_roi_range"),
        ):
            mocks.h.create.return_value = cam.ia
            cam.open()

        assert (cam.width, cam.height) == (1024, 1024)
        assert (cam.width_pixels, cam.height_pixels) == (1024, 1024)

    def test_configure_settings_swallows_a_broken_node_map(self):
        cam, mocks = _mock_harvester_camera()
        type(mocks.node_map).ExposureAuto = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("bus error"))
        )
        cam._configure_camera_settings()  # must not raise

    def test_close_continues_when_stopping_acquisition_blows_up(self):
        """A wedged producer must not prevent the device from being released."""
        cam, mocks = _mock_harvester_camera()
        with patch.object(cam, "stop_acquisition", side_effect=RuntimeError("wedged")):
            cam.close()
        mocks.ia.destroy.assert_called_once()
        mocks.h.reset.assert_called_once()

    def test_close_tolerates_every_stage_failing(self):
        cam, mocks = _mock_harvester_camera()
        cam.is_acquiring = True
        mocks.ia.stop.side_effect = RuntimeError("stop failed")
        mocks.ia.destroy.side_effect = RuntimeError("destroy failed")
        mocks.h.reset.side_effect = RuntimeError("reset failed")

        cam.close()  # must not raise

        mocks.ia.destroy.assert_called_once()
        mocks.h.reset.assert_called_once()

    def test_start_acquisition_without_an_acquirer_is_a_noop(self):
        cam, mocks = _mock_harvester_camera()
        cam.ia = None  # ty: ignore[invalid-assignment]
        cam.start_acquisition()
        assert cam.is_acquiring is False

    def test_stop_acquisition_swallows_producer_errors(self):
        cam, mocks = _mock_harvester_camera()
        cam.is_acquiring = True
        mocks.ia.stop.side_effect = RuntimeError("already stopped")
        cam.stop_acquisition()
        assert cam.is_acquiring is False

    def test_default_timeout_tracks_exposure(self):
        cam, mocks = _mock_harvester_camera()
        cam.is_acquiring = True
        cam.exposure_time = 3.0

        frame = MagicMock()
        component = MagicMock()
        component.width, component.height = 4, 2
        component.data = np.arange(8, dtype=np.uint8)
        frame.payload.components = [component]
        mocks.ia.fetch.return_value.__enter__.return_value = frame

        cam.get_image()

        assert mocks.ia.fetch.call_args.kwargs["timeout"] == pytest.approx(5.0)

    def test_stall_recovery_failure_does_not_escape(self):
        """If the restart itself fails we still try the fetch — the producer
        may have recovered on its own."""
        cam, mocks = _mock_harvester_camera()
        cam.is_acquiring = True
        cam.exposure_time = 0.01
        cam._last_successful_fetch = 1.0  # long ago

        frame = MagicMock()
        component = MagicMock()
        component.width, component.height = 2, 2
        component.data = np.arange(4, dtype=np.uint8)
        frame.payload.components = [component]
        mocks.ia.fetch.return_value.__enter__.return_value = frame

        with (
            patch("pybeamprofiler.gen_camera.time.monotonic", return_value=1_000.0),
            patch.object(cam, "stop_acquisition", side_effect=RuntimeError("cannot stop")) as stop,
        ):
            img = cam.get_image(timeout=0.1)

        stop.assert_called_once()  # recovery was attempted
        assert img.shape == (2, 2)

    def test_successful_fetch_clears_the_stall_flag(self):
        """One recovery per stall; a good frame re-arms it for the next one."""
        cam, mocks = _mock_harvester_camera()
        cam.is_acquiring = True
        cam.exposure_time = 0.01
        cam._last_successful_fetch = 1.0

        frame = MagicMock()
        component = MagicMock()
        component.width, component.height = 2, 2
        component.data = np.arange(4, dtype=np.uint8)
        frame.payload.components = [component]
        mocks.ia.fetch.return_value.__enter__.return_value = frame

        with patch("pybeamprofiler.gen_camera.time.monotonic", return_value=1_000.0):
            cam.get_image(timeout=0.1)

        mocks.ia.stop.assert_called_once()
        assert cam._stall_recovery_attempted is False
        assert cam._last_successful_fetch > 0

    def test_set_exposure_restarts_acquisition(self):
        cam, mocks = _mock_harvester_camera()
        cam.is_acquiring = True
        with (
            patch.object(cam, "stop_acquisition", wraps=cam.stop_acquisition) as stop,
            patch.object(cam, "start_acquisition") as start,
        ):
            cam.set_exposure(0.05)
        stop.assert_called_once()
        start.assert_called_once()
        assert cam.exposure_time == 0.05

    def test_set_exposure_does_not_start_an_idle_camera(self):
        cam, mocks = _mock_harvester_camera()
        cam.is_acquiring = False
        with patch.object(cam, "start_acquisition") as start:
            cam.set_exposure(0.05)
        start.assert_not_called()


class TestSettingWidgetCallbacks:
    """The acquisition buttons in the Jupyter panel."""

    def _capture_buttons(self, cam):
        """Run setting() and hand back its buttons keyed by description.

        Subclasses the real widget rather than faking one — ipywidgets
        containers reject anything that isn't a genuine Widget.
        """
        import ipywidgets as widgets

        captured: dict[str, Any] = {}

        class RecordingButton(widgets.Button):
            def on_click(self, callback, remove=False):
                self._captured_cb = callback
                captured[self.description] = self
                return super().on_click(callback, remove=remove)

        with (
            patch.object(widgets, "Button", RecordingButton),
            patch("IPython.display.display"),
        ):
            cam.setting()
        return captured

    def test_start_and_stop_buttons_drive_acquisition(self):
        cam = SimulatedCamera()
        cam.open()
        buttons = self._capture_buttons(cam)

        start = buttons["Start Acquisition"]
        stop = buttons["Stop Acquisition"]

        start._captured_cb(start)
        assert cam.is_acquiring is True
        assert start.disabled is True
        assert stop.disabled is False

        stop._captured_cb(stop)
        assert cam.is_acquiring is False
        assert start.disabled is False
        assert stop.disabled is True
        cam.close()


class TestGenicamControlBuilders:
    def test_no_features_yields_no_accordions(self):
        cam = SimulatedCamera()
        cam.open()
        with patch.object(cam, "_discover_features", return_value={}):
            assert cam._create_genicam_controls({"description_width": "initial"}) == []
        cam.close()

    def test_a_node_that_explodes_is_skipped(self):
        """The outer guard in _create_feature_controls catches anything the
        per-widget guards miss."""
        cam = SimulatedCamera()
        cam.open()

        class Exploding:
            def __getattr__(self, name):
                raise RuntimeError("SWIG binding died")

        cam.node_map.Boom = Exploding()  # ty: ignore[invalid-assignment]
        controls = cam._create_feature_controls(["Boom", "Gamma"], {"description_width": "initial"})
        assert len(controls) == 1  # only Gamma survived
        cam.close()

    def test_unknown_parameter_without_a_node_map_warns(self, caplog):
        cam = SimulatedCamera()  # not opened → node_map is None
        with caplog.at_level("WARNING"):
            cam._apply_settings_from_kwargs({"NotAThing": 1})
        assert "not recognized" in caplog.text

    def test_unknown_parameter_with_a_node_map_warns(self, caplog):
        cam = SimulatedCamera()
        cam.open()
        with caplog.at_level("WARNING"):
            cam._apply_settings_from_kwargs({"NotAThing": 1})
        assert "not found in node_map" in caplog.text
        cam.close()

    def test_widget_builder_failure_is_logged_not_raised(self, caplog):
        cam = SimulatedCamera()
        cam.open()
        with (
            patch.object(cam, "_create_enum_dropdown", side_effect=RuntimeError("widget blew up")),
            caplog.at_level("DEBUG"),
        ):
            controls = cam._create_feature_controls(
                ["PixelFormat"], {"description_width": "initial"}
            )
        assert controls == []
        assert "Could not create a control for PixelFormat" in caplog.text
        cam.close()

    def test_unknown_feature_name_is_skipped(self):
        cam = SimulatedCamera()
        cam.open()
        assert cam._create_feature_controls(["NoSuchThing"], {}) == []
        cam.close()

    def test_plain_string_feature_falls_back_to_a_dropdown(self):
        """A bare string node — no symbolics, no range — is the last resort."""
        cam = SimulatedCamera()
        cam.open()

        class PlainNode:
            value = "SomeMode"

        cam.node_map.CustomMode = PlainNode()  # ty: ignore[invalid-assignment]
        controls = cam._create_feature_controls(["CustomMode"], {"description_width": "initial"})
        assert len(controls) == 1
        assert list(controls[0].options) == ["SomeMode"]
        cam.close()

    def test_a_node_whose_value_explodes_is_skipped(self):
        cam = SimulatedCamera()
        cam.open()

        class Detonating:
            @property
            def value(self):
                raise RuntimeError("transport error")

        cam.node_map.Boom = Detonating()  # ty: ignore[invalid-assignment]
        assert cam._create_feature_controls(["Boom"], {"description_width": "initial"}) == []
        cam.close()

    def test_string_valued_auto_feature_becomes_a_dropdown(self):
        cam = SimulatedCamera()
        cam.open()
        controls = cam._create_feature_controls(["ExposureAuto"], {"description_width": "initial"})
        assert len(controls) == 1
        assert list(controls[0].options) == ["Off", "Once", "Continuous"]
        cam.close()

    def test_boolean_enable_feature_becomes_a_checkbox(self):
        import ipywidgets as widgets

        cam = SimulatedCamera()
        cam.open()
        controls = cam._create_feature_controls(["GammaEnable"], {"description_width": "initial"})
        assert len(controls) == 1
        assert isinstance(controls[0], widgets.Checkbox)
        cam.close()

    def test_enum_features_render_even_when_they_expose_min_max(self):
        """A node with symbolics is an enum, not a slider.

        These used to fall into the numeric branch, where the slider builder
        choked on a ``None`` minimum and returned nothing — so PixelFormat,
        TriggerMode and TriggerSource simply vanished from the panel.
        """
        cam = SimulatedCamera()
        cam.open()
        for feature in ("PixelFormat", "TriggerMode", "TriggerSource"):
            node = getattr(cam.node_map, feature)
            assert node.symbolics and hasattr(node, "min")
            controls = cam._create_feature_controls([feature], {"description_width": "initial"})
            assert len(controls) == 1, f"{feature} produced no control"
            assert list(controls[0].options) == list(node.symbolics)
        cam.close()


class TestCliCleanup:
    """``main()`` must release the camera even when plotting blows up."""

    def test_camera_is_closed_after_a_plot_failure(self):
        import sys

        from pybeamprofiler.beamprofiler import main

        def start_then_fail(self, *args, **kwargs):
            self.camera.is_acquiring = True
            raise RuntimeError("render died")

        argv = ["pybeamprofiler", "--camera", "simulated"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(BeamProfiler, "plot", start_then_fail),
            patch.object(SimulatedCamera, "close") as close,
            patch.object(SimulatedCamera, "stop_acquisition") as stop,
        ):
            main()  # the error is logged, not re-raised

        stop.assert_called_once()
        close.assert_called_once()

    def test_cleanup_errors_are_swallowed(self):
        import sys

        from pybeamprofiler.beamprofiler import main

        argv = ["pybeamprofiler", "--camera", "simulated"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(BeamProfiler, "plot"),
            patch.object(SimulatedCamera, "close", side_effect=RuntimeError("already gone")),
        ):
            main()  # must not raise

    def test_keyboard_interrupt_is_reported_cleanly(self, capsys):
        import sys

        from pybeamprofiler.beamprofiler import main

        argv = ["pybeamprofiler", "--camera", "simulated"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(BeamProfiler, "plot", side_effect=KeyboardInterrupt),
        ):
            main()
        assert "Stopped by user" in capsys.readouterr().out
