"""Tests for the Dash camera selector: discover, switch, then play.

These drive the registered callbacks directly rather than through a browser,
which is how the rest of the Dash suite works. The interesting behaviour is
all in the switch: it has to open the new device before releasing the old
one, reset everything derived from the old sensor, and leave the stream
stopped so that Play is a deliberate act.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import dash
import pytest
from dash import html

from pybeamprofiler import dash_app, dash_layout, discovery
from pybeamprofiler.beamprofiler import BeamProfiler
from pybeamprofiler.simulated import SIMULATED_PROFILES, SimulatedCamera


def _callbacks(bp: BeamProfiler) -> dict[str, Any]:
    """Register the app's callbacks and return them keyed by function name."""
    app = dash.Dash(__name__)
    app.layout = html.Div()
    captured: dict[str, Any] = {}
    original = app.callback

    def tracking(*args, **kwargs):
        def decorator(f):
            captured[f.__name__] = f
            return original(*args, **kwargs)(f)

        return decorator

    app.callback = tracking  # ty: ignore[invalid-assignment]
    dash_app._register_callbacks(app, bp)
    return captured


@pytest.fixture
def profiler():
    bp = BeamProfiler(camera="simulated")
    yield bp
    if bp.camera is not None:
        bp.camera.close()


def _other_key(bp: BeamProfiler) -> str:
    """A selectable camera that is not the one currently open."""
    options, current = dash_layout._camera_options(bp)
    return next(o.key for o in options if o.key != current)


class TestCameraOptionsForTheDropdown:
    def test_lists_every_simulated_camera(self, profiler):
        options, _ = dash_layout._camera_options(profiler)
        assert len(options) == len(SIMULATED_PROFILES)

    def test_current_camera_is_preselected(self, profiler):
        options, current = dash_layout._camera_options(profiler)
        assert current in {o.key for o in options}
        assert current == discovery.describe_open_camera(profiler.camera).key

    def test_open_camera_is_added_when_discovery_misses_it(self, profiler):
        """A camera opened from an explicit .cti path won't be enumerated.

        It still has to appear as the selected entry, or the dropdown shows a
        blank box over a running stream.
        """
        with patch.object(dash_layout, "discover_cameras", return_value=[]):
            options, current = dash_layout._camera_options(profiler)

        assert len(options) == 1
        assert options[0].key == current

    def test_no_camera_leaves_the_selection_empty(self, profiler):
        profiler.camera = None
        options, current = dash_layout._camera_options(profiler)
        assert current == ""
        assert options  # the simulated entries are still offered


class TestRefreshCameras:
    def test_repopulates_the_dropdown(self, profiler):
        listed, current, status = _callbacks(profiler)["refresh_cameras"](1)
        assert [o["value"] for o in listed] == [
            o.key for o in dash_layout._camera_options(profiler)[0]
        ]
        assert current

    def test_reports_when_only_simulators_are_available(self, profiler):
        _, _, status = _callbacks(profiler)["refresh_cameras"](1)
        assert "simulated only" in status

    def test_counts_real_hardware(self, profiler):
        found = [
            discovery.CameraOption(key="genicam:1", label="FLIR A", kind="genicam"),
            discovery.CameraOption(key="genicam:2", label="FLIR B", kind="genicam"),
        ]
        with patch.object(dash_layout, "discover_cameras", return_value=found):
            _, _, status = _callbacks(profiler)["refresh_cameras"](1)
        assert status == "2 cameras found"

    def test_singular_wording_for_one_camera(self, profiler):
        found = [discovery.CameraOption(key="genicam:1", label="FLIR A", kind="genicam")]
        with patch.object(dash_layout, "discover_cameras", return_value=found):
            _, _, status = _callbacks(profiler)["refresh_cameras"](1)
        assert status == "1 camera found"


class TestSwitchCamera:
    def test_swaps_in_the_selected_device(self, profiler):
        target = _other_key(profiler)
        result = _callbacks(profiler)["switch_camera"](target)

        assert discovery.describe_open_camera(profiler.camera).key == target
        assert "ready" in result[0]

    def test_geometry_follows_the_new_sensor(self, profiler):
        before = (profiler.width_pixels, profiler.height_pixels, profiler.pixel_size)
        _callbacks(profiler)["switch_camera"](_other_key(profiler))
        after = (profiler.width_pixels, profiler.height_pixels, profiler.pixel_size)

        assert after != before
        assert (profiler.width_pixels, profiler.height_pixels) == (
            profiler.camera.width,
            profiler.camera.height,
        )

    def test_pixel_scale_box_is_updated(self, profiler):
        result = _callbacks(profiler)["switch_camera"](_other_key(profiler))
        assert result[5] == pytest.approx(round(profiler.pixel_size, 4))

    def test_stream_is_left_paused_for_the_play_button(self, profiler):
        result = _callbacks(profiler)["switch_camera"](_other_key(profiler))

        assert result[1] is True, "store-paused should be set"
        assert result[3] == "success", "the button should offer Play"
        assert profiler.camera.is_acquiring is False

    def test_play_after_a_switch_starts_the_new_camera(self, profiler):
        callbacks = _callbacks(profiler)
        callbacks["switch_camera"](_other_key(profiler))

        callbacks["toggle_pause"](1, True)

        assert profiler.camera.is_acquiring is True
        img = profiler.camera.get_image()
        assert img.shape == (profiler.camera.height, profiler.camera.width)

    def test_the_previous_camera_is_released(self, profiler):
        """A GenICam device stays claimed until closed, so a leaked handle
        blocks the next attempt to open it."""
        old = profiler.camera
        with patch.object(SimulatedCamera, "close") as mock_close:
            _callbacks(profiler)["switch_camera"](_other_key(profiler))
        mock_close.assert_called_once()
        assert profiler.camera is not old

    def test_analysis_state_is_reset(self, profiler):
        """Warm-start parameters from the old sensor are a nonsense seed for
        the new one."""
        profiler.analyze(profiler.camera.get_image())
        assert profiler._last_popt_x is not None

        _callbacks(profiler)["switch_camera"](_other_key(profiler))

        assert profiler._last_popt_x is None
        assert profiler._last_popt_y is None
        assert profiler._last_popt_2d is None
        assert profiler.last_img is None
        assert profiler.width_x == 0.0

    def test_zoom_and_frame_buffers_are_cleared(self, profiler):
        dash_app._zoom_range = {"x": [0, 1], "y": [0, 1]}
        dash_app._recent_frame_times.append(1.0)

        _callbacks(profiler)["switch_camera"](_other_key(profiler))

        assert dash_app._zoom_range is None
        assert len(dash_app._recent_frame_times) == 0

    def test_settings_panel_is_rebuilt(self, profiler):
        import dash_bootstrap_components as dbc

        result = _callbacks(profiler)["switch_camera"](_other_key(profiler))
        assert isinstance(result[4], dbc.Accordion)

    def test_reselecting_the_current_camera_is_a_no_op(self, profiler):
        _, current = dash_layout._camera_options(profiler)
        old = profiler.camera

        result = _callbacks(profiler)["switch_camera"](current)

        assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)
        assert profiler.camera is old

    def test_blank_selection_is_ignored(self, profiler):
        result = _callbacks(profiler)["switch_camera"](None)
        assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)

    def test_unknown_key_reports_rather_than_raising(self, profiler):
        old = profiler.camera
        result = _callbacks(profiler)["switch_camera"]("genicam:not-here")

        assert "Unknown camera" in result[0]
        assert profiler.camera is old

    def test_a_camera_that_will_not_open_leaves_the_current_one_running(self, profiler):
        """Opening the new device before releasing the old one means an
        unplugged or already-claimed camera does not strand the user."""
        old = profiler.camera
        old.start_acquisition()
        target = _other_key(profiler)

        with patch.object(dash_app, "open_camera", side_effect=RuntimeError("device in use")):
            result = _callbacks(profiler)["switch_camera"](target)

        assert "Could not open" in result[0]
        assert "device in use" in result[0]
        assert profiler.camera is old
        assert old.is_acquiring is True


class TestUnconditionalCallbackRegistration:
    """Which camera is attached can change at runtime, so callbacks cannot be
    registered based on what the *initial* camera supports."""

    @staticmethod
    def _detached() -> BeamProfiler:
        """A profiler whose camera has been closed and dropped."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.close()
        bp.camera = None
        return bp

    def test_roi_callbacks_exist_without_a_camera(self):
        names = _callbacks(self._detached())
        assert "apply_roi" in names
        assert "reset_roi" in names

    def test_genicam_callbacks_exist_without_a_camera(self):
        names = _callbacks(self._detached())
        assert "set_genicam_numeric" in names
        assert "set_genicam_select" in names
        assert "set_genicam_switch" in names

    def test_roi_callback_handles_a_missing_camera(self):
        assert _callbacks(self._detached())["apply_roi"](1, 0, 0, 10, 10) == "No camera"
