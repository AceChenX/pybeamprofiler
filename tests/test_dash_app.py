"""Tests for the Dash GUI module and refactored SimulatedCamera."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
import pytest
from conftest import requires_genicam
from dash import html

from pybeamprofiler.beamprofiler import BeamProfiler
from pybeamprofiler.dash_app import (
    COLORSCALES,
    _build_setting_items,
    _fitting_tab,
    _format_results,
    _setting_tab,
    build_figure,
    create_app,
)
from pybeamprofiler.simulated import SimulatedCamera, _SimulatedNode, _SimulatedNodeMap

# ─── _SimulatedNode ─────────────────────────────────────────────────────────


class TestSimulatedNode:
    """Tests for _SimulatedNode value handling."""

    def test_get_set_value(self):
        node = _SimulatedNode(42)
        assert node.value == 42
        node.value = 99
        assert node.value == 99

    def test_readonly_raises(self):
        node = _SimulatedNode("locked", readonly=True)
        assert node.value == "locked"
        with pytest.raises(AttributeError, match="read-only"):
            node.value = "new"

    def test_min_max(self):
        node = _SimulatedNode(5.0, min_val=0.0, max_val=10.0)
        assert node.min == 0.0
        assert node.max == 10.0

    def test_symbolics(self):
        node = _SimulatedNode("Off", symbolics=["Off", "Once", "Continuous"])
        assert node.symbolics == ["Off", "Once", "Continuous"]

    def test_defaults_are_none(self):
        node = _SimulatedNode()
        assert node.value is None
        assert node.min is None
        assert node.max is None
        assert node.symbolics is None


# ─── _SimulatedNodeMap ──────────────────────────────────────────────────────


class TestSimulatedNodeMap:
    """Tests for _SimulatedNodeMap initialization with camera ref."""

    def test_init_from_camera(self):
        cam = SimulatedCamera()
        cam.open()
        nm = cam.node_map
        assert nm is not None

        assert nm.DeviceModelName.value == "SimulatedCamera"
        assert nm.DeviceVendorName.value == "pybeamprofiler"
        assert nm.DeviceSerialNumber.value == "SIM-001"

    def test_node_values_match_camera_state(self):
        cam = SimulatedCamera()
        cam.open()
        nm = cam.node_map
        assert nm is not None

        assert nm.ExposureTime.value == cam.exposure_time * 1_000_000
        assert nm.Gain.value == cam.gain
        assert nm.Width.value == cam.width
        assert nm.Height.value == cam.height
        assert nm.WidthMax.value == 1024
        assert nm.HeightMax.value == 1024

    def test_readonly_device_nodes(self):
        cam = SimulatedCamera()
        cam.open()
        nm = cam.node_map
        assert nm is not None

        with pytest.raises(AttributeError):
            nm.DeviceModelName.value = "Changed"
        with pytest.raises(AttributeError):
            nm.WidthMax.value = 9999

    def test_writable_exposure_node(self):
        cam = SimulatedCamera()
        cam.open()
        nm = cam.node_map
        assert nm is not None

        nm.ExposureTime.value = 50000
        assert nm.ExposureTime.value == 50000

    def test_symbolics_on_auto_nodes(self):
        cam = SimulatedCamera()
        cam.open()
        nm = cam.node_map
        assert nm is not None

        assert nm.ExposureAuto.symbolics == ["Off", "Once", "Continuous"]
        assert nm.GainAuto.symbolics == ["Off", "Once", "Continuous"]


# ─── SimulatedCamera open / ROI / exposure / gain ───────────────────────────


class TestSimulatedCameraRefactored:
    """Tests for refactored SimulatedCamera features."""

    def test_open_creates_node_map(self):
        cam = SimulatedCamera()
        assert cam.node_map is None
        cam.open()
        assert cam.node_map is not None
        assert isinstance(cam.node_map, _SimulatedNodeMap)
        cam.close()

    def test_roi_info_defaults(self):
        cam = SimulatedCamera()
        roi = cam.roi_info
        assert roi["offset_x"] == 0
        assert roi["offset_y"] == 0
        assert roi["width"] == 1024
        assert roi["height"] == 1024
        assert roi["max_width"] == 1024
        assert roi["max_height"] == 1024

    def test_set_roi_basic(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_roi(offset_x=100, offset_y=200, width=400, height=300)
        roi = cam.roi_info
        assert roi["offset_x"] == 100
        assert roi["offset_y"] == 200
        assert roi["width"] == 400
        assert roi["height"] == 300
        assert cam.width == 400
        assert cam.height == 300
        cam.close()

    def test_set_roi_clamp_to_bounds(self):
        cam = SimulatedCamera()
        cam.set_roi(offset_x=2000, offset_y=2000, width=5000, height=5000)
        roi = cam.roi_info
        assert roi["offset_x"] <= 1023
        assert roi["offset_y"] <= 1023
        assert roi["offset_x"] + roi["width"] <= 1024
        assert roi["offset_y"] + roi["height"] <= 1024

    def test_set_roi_negative_offset_clamped(self):
        cam = SimulatedCamera()
        cam.set_roi(offset_x=-10, offset_y=-20, width=100, height=100)
        roi = cam.roi_info
        assert roi["offset_x"] == 0
        assert roi["offset_y"] == 0

    def test_set_roi_none_resets_to_full(self):
        cam = SimulatedCamera()
        cam.set_roi(offset_x=100, offset_y=100, width=200, height=200)
        cam.set_roi(offset_x=0, offset_y=0, width=None, height=None)
        roi = cam.roi_info
        assert roi["width"] == 1024
        assert roi["height"] == 1024

    def test_set_roi_syncs_node_map(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_roi(offset_x=50, offset_y=60, width=400, height=300)
        nm = cam.node_map
        assert nm is not None
        assert nm.OffsetX.value == 50
        assert nm.OffsetY.value == 60
        assert nm.Width.value == 400
        assert nm.Height.value == 300
        cam.close()

    def test_set_exposure_syncs_node_map(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_exposure(0.05)
        assert cam.exposure_time == 0.05
        nm = cam.node_map
        assert nm is not None
        assert nm.ExposureTime._value == 0.05 * 1_000_000
        cam.close()

    def test_set_gain_syncs_node_map(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_gain(5.0)
        assert cam.gain == 5.0
        nm = cam.node_map
        assert nm is not None
        assert nm.Gain._value == 5.0
        cam.close()

    def test_get_image_respects_roi(self):
        cam = SimulatedCamera()
        cam.set_roi(offset_x=0, offset_y=0, width=256, height=128)
        img = cam.get_image()
        assert img.shape == (128, 256)
        assert img.dtype == np.uint8

    def test_get_image_full_sensor(self):
        cam = SimulatedCamera()
        img = cam.get_image()
        assert img.shape == (1024, 1024)

    def test_exposure_range_property(self):
        cam = SimulatedCamera()
        assert cam.exposure_range == (0.001, 1.0)

    def test_gain_range_property(self):
        cam = SimulatedCamera()
        assert cam.gain_range == (0.0, 24.0)


# ─── build_figure ───────────────────────────────────────────────────────────


class TestBuildFigure:
    """Tests for the build_figure helper."""

    @pytest.fixture()
    def bp(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        return bp

    def test_none_image_returns_empty_figure(self, bp):
        fig = build_figure(bp, None, None, None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_valid_image_returns_figure_with_traces(self, bp):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        popt_x = [100.0, 32.0, 8.0, 10.0]
        popt_y = [100.0, 32.0, 8.0, 10.0]
        fig = build_figure(bp, img, popt_x, popt_y)
        assert isinstance(fig, go.Figure)
        # Heatmap + ellipse + x-profile + x-fit + y-profile + y-fit = 6
        assert len(fig.data) >= 5

    def test_no_popt_skips_fit_traces(self, bp):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        fig = build_figure(bp, img, None, None)
        # Heatmap + x-profile + y-profile = 3
        assert len(fig.data) == 3

    def test_colorscale_applied(self, bp):
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        fig = build_figure(bp, img, None, None, colorscale="Viridis")
        heatmap = fig.data[0]
        # Plotly expands named colorscales into tuples of (position, color)
        cs = heatmap.colorscale
        assert isinstance(cs, tuple)
        assert cs[0][0] == 0.0
        assert cs[-1][0] == 1.0

    def test_zmin_zmax_applied(self, bp):
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        fig = build_figure(bp, img, None, None, zmin=10.0, zmax=200.0)
        heatmap = fig.data[0]
        assert heatmap.zmin == 10.0
        assert heatmap.zmax == 200.0

    def test_zmin_zmax_none_means_auto(self, bp):
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        fig = build_figure(bp, img, None, None, zmin=None, zmax=None)
        heatmap = fig.data[0]
        assert heatmap.zmin is None
        assert heatmap.zmax is None

    def test_large_image_downsampled(self, bp):
        """Images larger than MAX_DISPLAY_DIM should be downsampled."""
        img = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
        fig = build_figure(bp, img, None, None)
        heatmap = fig.data[0]
        assert heatmap.z.shape[0] <= 1024
        assert heatmap.z.shape[1] <= 1024

    def test_linecut_crosshairs(self, bp):
        bp.fit_method = "linecut"
        bp._linecut_x = 32
        bp._linecut_y = 32
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        fig = build_figure(bp, img, None, None)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 4  # x-profile, y-profile + 2 crosshairs


# ─── _format_results ────────────────────────────────────────────────────────


class TestFormatResults:
    """Tests for _format_results helper."""

    def test_no_fit_data(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        rows = _format_results(bp)
        assert len(rows) == 1
        assert "No fit data" in rows[0].children

    def test_with_fit_data(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        bp.analyze(img)
        if bp.width_x > 0:
            rows = _format_results(bp)
            assert len(rows) > 1
            text = str(rows)
            assert "μm" in text


# ─── _build_setting_items ───────────────────────────────────────────────────


class TestBuildSettingItems:
    """Tests for _build_setting_items."""

    def test_returns_accordion_items_for_simulated(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        items = _build_setting_items(bp)
        assert len(items) >= 2  # Camera Info, Exposure & Gain
        assert all(isinstance(item, dbc.AccordionItem) for item in items)

    def test_roi_controls_in_image_format(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        items = _build_setting_items(bp)
        titles = [item.title for item in items]  # ty: ignore[unresolved-attribute]
        assert "Image Format Control" in titles
        ifc_item = next(i for i in items if i.title == "Image Format Control")  # ty: ignore[unresolved-attribute]
        layout_str = str(ifc_item)
        assert "input-roi-ox" in layout_str

    def test_no_camera_returns_empty(self):
        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        items = _build_setting_items(bp)
        assert items == []


# ─── _fitting_tab / _setting_tab ────────────────────────────────────────────


class TestTabBuilders:
    """Tests for _fitting_tab and _setting_tab."""

    def test_fitting_tab_returns_tab(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        tab = _fitting_tab(bp)
        assert isinstance(tab, dbc.Tab)
        assert tab.tab_id == "tab-fitting"  # ty: ignore[unresolved-attribute]

    def test_setting_tab_returns_tab(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        tab = _setting_tab(bp)
        assert isinstance(tab, dbc.Tab)
        assert tab.tab_id == "tab-setting"  # ty: ignore[unresolved-attribute]

    def test_setting_tab_has_container(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        tab = _setting_tab(bp)
        layout_str = str(tab)
        assert "settings-container" in layout_str

    def test_setting_tab_no_camera(self):
        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        tab = _setting_tab(bp)
        assert isinstance(tab, dbc.Tab)


# ─── create_app ─────────────────────────────────────────────────────────────


class TestCreateApp:
    """Tests for create_app factory."""

    def test_returns_dash_instance(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        app = create_app(bp)
        assert isinstance(app, dash.Dash)

    def test_layout_has_expected_components(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        app = create_app(bp)
        layout = app.layout

        layout_str = str(layout)
        assert "live-graph" in layout_str
        assert "interval" in layout_str
        assert "store-paused" in layout_str
        assert "store-frame" in layout_str

    def test_callbacks_registered(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        app = create_app(bp)
        # Dash stores callbacks in callback_map
        assert len(app.callback_map) > 0

    def test_initial_figure_with_image(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        app = create_app(bp)
        assert isinstance(app, dash.Dash)
        bp.camera.stop_acquisition()

    def test_initial_figure_no_image(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp._mode = "file"
        bp.last_img = None
        app = create_app(bp)
        assert isinstance(app, dash.Dash)


# ─── Callback helpers (extracted from registered callbacks) ─────────────────


class TestCallbackHelpers:
    """Test callback logic by calling the underlying registered functions."""

    @pytest.fixture()
    def app_and_bp(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        app = create_app(bp)
        return app, bp

    def test_toggle_autorange(self, app_and_bp):
        from pybeamprofiler.dash_app import _register_callbacks

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        app = dash.Dash(__name__)
        app.layout = html.Div()

        callback_map: dict = {}
        original_callback = app.callback

        def tracking_callback(*args, **kwargs):
            def decorator(f):
                key = str(args)
                callback_map[key] = f
                return original_callback(*args, **kwargs)(f)

            return decorator

        app.callback = tracking_callback  # ty: ignore[invalid-assignment]
        _register_callbacks(app, bp)

        for key, func in callback_map.items():
            if "input-zmin" in key and "disabled" in key:
                result = func(True)
                assert result == (True, True)
                result = func(False)
                assert result == (False, False)
                break

    def test_colorscales_list(self):
        assert len(COLORSCALES) > 0
        assert "Hot" in COLORSCALES
        assert "Viridis" in COLORSCALES


# ─── Pause stops / Play resumes acquisition ────────────────────────────────


class TestPauseStopsAcquisition:
    """Verify that Pause actually stops acquisition and Play restarts it."""

    @staticmethod
    def _get_toggle_fn(bp: BeamProfiler):
        """Extract the toggle_pause callback.

        ``store-paused`` has two writers now — the camera selector also
        pauses the stream after a switch — so this relies on
        :func:`_extract_callback` preferring the callback that declares the
        output first, which is toggle_pause.
        """
        return _extract_callback(bp, "store-paused")

    def test_pause_stops_acquisition(self):
        bp = BeamProfiler(camera="simulated")
        cam = bp.camera
        assert cam is not None
        cam.start_acquisition()
        assert cam.is_acquiring is True

        toggle = self._get_toggle_fn(bp)
        assert toggle is not None
        result = toggle(1, False)  # paused=False → new_paused=True (pause)
        assert result[0] is True  # store-paused data
        assert cam.is_acquiring is False

    def test_play_restarts_acquisition(self):
        bp = BeamProfiler(camera="simulated")
        cam = bp.camera
        assert cam is not None
        cam.stop_acquisition()
        assert cam.is_acquiring is False

        toggle = self._get_toggle_fn(bp)
        assert toggle is not None
        result = toggle(2, True)  # paused=True → new_paused=False (play)
        assert result[0] is False  # store-paused data
        assert cam.is_acquiring is True

    def test_pause_rebuilds_settings(self):
        bp = BeamProfiler(camera="simulated")
        cam = bp.camera
        assert cam is not None
        cam.start_acquisition()

        toggle = self._get_toggle_fn(bp)
        assert toggle is not None
        result = toggle(1, False)
        settings_body = result[3]
        assert settings_body is not None
        assert isinstance(settings_body, dbc.Accordion)


# ─── GenICam feature controls in Dash ────────────────────────────────────────


class TestBuildGenicamControl:
    """Tests for _build_genicam_control helper."""

    def test_boolean_feature_creates_switch(self):
        from pybeamprofiler.dash_app import _build_genicam_control
        from pybeamprofiler.simulated import _SimulatedNode

        class MockCam:
            class node_map:
                ReverseX = _SimulatedNode(False)

        ctrl = _build_genicam_control(MockCam(), "ReverseX")
        assert ctrl is not None
        layout_str = str(ctrl)
        assert "genicam-sw" in layout_str

    def test_enum_feature_creates_select(self):
        from pybeamprofiler.dash_app import _build_genicam_control
        from pybeamprofiler.simulated import _SimulatedNode

        class MockCam:
            class node_map:
                PixelFormat = _SimulatedNode("Mono8", symbolics=["Mono8", "Mono12"])

        ctrl = _build_genicam_control(MockCam(), "PixelFormat")
        assert ctrl is not None
        layout_str = str(ctrl)
        assert "genicam-sel" in layout_str

    def test_numeric_feature_creates_slider(self):
        from pybeamprofiler.dash_app import _build_genicam_control
        from pybeamprofiler.simulated import _SimulatedNode

        class MockCam:
            class node_map:
                Gamma = _SimulatedNode(1.0, min_val=0.25, max_val=4.0)

        ctrl = _build_genicam_control(MockCam(), "Gamma")
        assert ctrl is not None
        layout_str = str(ctrl)
        assert "genicam-num" in layout_str
        assert "Slider" in layout_str

    def test_readonly_numeric_shows_text(self):
        from pybeamprofiler.dash_app import _build_genicam_control
        from pybeamprofiler.simulated import _SimulatedNode

        class MockCam:
            class node_map:
                DeviceTemperature = _SimulatedNode(42.5, min_val=0.0, max_val=100.0, readonly=True)

        ctrl = _build_genicam_control(MockCam(), "DeviceTemperature")
        assert ctrl is not None
        layout_str = str(ctrl)
        assert "42.5" in layout_str
        assert "genicam-num" not in layout_str

    def test_string_enable_creates_select(self):
        from pybeamprofiler.dash_app import _build_genicam_control
        from pybeamprofiler.simulated import _SimulatedNode

        class MockCam:
            class node_map:
                SomeEnable = _SimulatedNode("On")

        ctrl = _build_genicam_control(MockCam(), "SomeEnable")
        assert ctrl is not None
        layout_str = str(ctrl)
        assert "genicam-sel" in layout_str

    def test_string_auto_creates_select(self):
        from pybeamprofiler.dash_app import _build_genicam_control
        from pybeamprofiler.simulated import _SimulatedNode

        class MockCam:
            class node_map:
                SomeAuto = _SimulatedNode("Off")

        ctrl = _build_genicam_control(MockCam(), "SomeAuto")
        assert ctrl is not None

    def test_no_node_map_returns_none(self):
        from pybeamprofiler.dash_app import _build_genicam_control

        class MockCam:
            node_map = None

        assert _build_genicam_control(MockCam(), "Anything") is None

    def test_missing_feature_returns_none(self):
        from pybeamprofiler.dash_app import _build_genicam_control
        from pybeamprofiler.simulated import _SimulatedNode

        class MockCam:
            class node_map:
                Gamma = _SimulatedNode(1.0, min_val=0.0, max_val=4.0)

        assert _build_genicam_control(MockCam(), "NonexistentFeature") is None

    def test_value_exception_returns_none(self):
        from pybeamprofiler.dash_app import _build_genicam_control

        class FailingNode:
            @property
            def value(self):
                raise RuntimeError("broken")

        class MockCam:
            class node_map:
                Bad = FailingNode()

        assert _build_genicam_control(MockCam(), "Bad") is None


class TestDashGenicamSettingItems:
    """Tests for GenICam features in _build_setting_items."""

    def test_simulated_camera_has_genicam_items(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        items = _build_setting_items(bp)
        titles = [item.title for item in items]  # ty: ignore[unresolved-attribute]
        assert "Camera Info" in titles
        assert "Acquisition Control" in titles
        assert "Analog Control" in titles
        assert "Image Format Control" in titles

    def test_genicam_groups_are_subset_of_discovery(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        discovered = bp.camera._discover_features()
        items = _build_setting_items(bp)
        titles = {item.title for item in items}  # ty: ignore[unresolved-attribute]
        non_discovery_titles = {"Camera Info"}
        from pybeamprofiler.dash_app import _PINNED_CATEGORIES

        non_discovery_titles.update(_PINNED_CATEGORIES)
        genicam_titles = titles - non_discovery_titles
        for title in genicam_titles:
            assert title in discovered

    def test_no_camera_no_genicam_items(self):
        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        items = _build_setting_items(bp)
        assert items == []


class TestDashGenicamCallbacks:
    """Tests for GenICam pattern-matching callbacks registration."""

    def test_callbacks_registered_for_simulated(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        app = create_app(bp)
        callback_keys = list(app.callback_map.keys())
        genicam_callbacks = [k for k in callback_keys if "genicam" in k]
        assert len(genicam_callbacks) > 0

    def test_app_with_genicam_controls_renders(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        app = create_app(bp)
        assert isinstance(app, dash.Dash)
        layout_str = str(app.layout)
        assert "genicam" in layout_str


# ─── _normalize_profile ────────────────────────────────────────────────────


class TestNormalizeProfile:
    """Tests for the _normalize_profile helper."""

    def test_uniform_profile(self):
        from pybeamprofiler.dash_app import _normalize_profile

        data = np.ones(100)
        result = _normalize_profile(data, span=200.0)
        assert np.allclose(result, 0.0)

    def test_range_scaled_to_fraction(self):
        from pybeamprofiler.dash_app import _normalize_profile

        data = np.array([0.0, 50.0, 100.0])
        result = _normalize_profile(data, span=100.0, fraction=0.25)
        assert result.max() == pytest.approx(25.0)
        assert result.min() == pytest.approx(0.0)

    def test_default_fraction(self):
        from pybeamprofiler.dash_app import _normalize_profile

        data = np.array([0.0, 100.0])
        result = _normalize_profile(data, span=100.0)
        assert result.max() > 0.0


# ─── build_figure with cached projections ──────────────────────────────────


class TestBuildFigureCachedProjections:
    """Verify build_figure reuses cached projections from analyze()."""

    def test_cached_projections_used(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        popt_x, popt_y = bp.analyze(img)

        assert bp._last_proj_x is not None
        assert bp._last_proj_y is not None
        assert len(bp._last_proj_x) == img.shape[1]
        assert len(bp._last_proj_y) == img.shape[0]

        fig = build_figure(bp, img, popt_x, popt_y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_no_cached_projections_still_works(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        bp._last_proj_x = None
        bp._last_proj_y = None

        fig = build_figure(bp, img, None, None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3


# ─── _format_results ───────────────────────────────────────────────────────


class TestFormatResultsExtended:
    """Additional tests for _format_results."""

    def test_with_2d_fit(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.fit_method = "2d"
        img = bp.camera.get_image()
        bp.analyze(img)
        rows = _format_results(bp)
        text = str(rows)
        assert "Angle" in text

    def test_zero_width_shows_no_data(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.width_x = 0.0
        rows = _format_results(bp)
        assert len(rows) == 1
        assert "No fit data" in str(rows[0])


# ─── Projection caching in analyze() ──────────────────────────────────────


class TestAnalyzeProjectionCaching:
    """Verify analyze() caches projections for downstream use."""

    def test_1d_fit_caches_projections(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        bp.fit_method = "1d"
        bp.analyze(img)
        assert bp._last_proj_x is not None
        assert bp._last_proj_y is not None
        np.testing.assert_array_equal(bp._last_proj_x, np.sum(img, axis=0))
        np.testing.assert_array_equal(bp._last_proj_y, np.sum(img, axis=1))

    def test_2d_fit_caches_projections(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        bp.fit_method = "2d"
        bp.analyze(img)
        assert bp._last_proj_x is not None
        assert bp._last_proj_y is not None

    def test_fwhm_caches_projections(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        bp.definition = "fwhm"
        bp.analyze(img)
        assert bp._last_proj_x is not None
        assert bp._last_proj_y is not None

    def test_d4s_caches_projections(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        bp.definition = "d4s"
        bp.analyze(img)
        assert bp._last_proj_x is not None
        assert bp._last_proj_y is not None

    def test_linecut_no_projections(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        bp.fit_method = "linecut"
        bp.analyze(img)
        assert bp._last_proj_x is None
        assert bp._last_proj_y is None


# ─── build_figure light theme and 2D rotation ─────────────────────────────


class TestBuildFigureThemeAndRotation:
    """Test build_figure with light theme and 2D ellipse rotation."""

    def test_light_theme(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        popt_x, popt_y = bp.analyze(img)
        fig = build_figure(bp, img, popt_x, popt_y, dark_theme=False)
        assert isinstance(fig, go.Figure)
        assert fig.layout.plot_bgcolor == "#f8f8f8"

    def test_2d_ellipse_rotation(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.fit_method = "2d"
        img = bp.camera.get_image()
        popt_x, popt_y = bp.analyze(img)
        fig = build_figure(bp, img, popt_x, popt_y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 5

    def test_custom_zrange(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = bp.camera.get_image()
        fig = build_figure(bp, img, None, None, zmin=10.0, zmax=200.0)
        assert fig.data[0].zmin == 10.0
        assert fig.data[0].zmax == 200.0


# ─── _is_readonly ──────────────────────────────────────────────────────────


class TestIsReadonly:
    """Tests for _is_readonly helper."""

    def test_readonly_simulated_node(self):
        from pybeamprofiler.dash_app import _is_readonly

        node = _SimulatedNode("locked", readonly=True)
        assert _is_readonly(node) is True

    def test_writable_simulated_node(self):
        from pybeamprofiler.dash_app import _is_readonly

        node = _SimulatedNode(42)
        assert _is_readonly(node) is False

    def test_node_without_access_mode(self):
        from pybeamprofiler.dash_app import _is_readonly

        class PlainNode:
            pass

        assert _is_readonly(PlainNode()) is False

    def test_access_mode_raises(self):
        from pybeamprofiler.dash_app import _is_readonly

        class FailNode:
            def get_access_mode(self):
                raise RuntimeError("broken")

        assert _is_readonly(FailNode()) is False


# ─── _humanize ─────────────────────────────────────────────────────────────


class TestHumanize:
    """Tests for the _humanize helper."""

    def test_camel_case(self):
        from pybeamprofiler.dash_app import _humanize

        assert _humanize("AcquisitionFrameRate") == "Acquisition Frame Rate"

    def test_single_word(self):
        from pybeamprofiler.dash_app import _humanize

        assert _humanize("Gain") == "Gain"

    def test_consecutive_uppercase_unchanged(self):
        from pybeamprofiler.dash_app import _humanize

        assert _humanize("ROIWidth") == "ROIWidth"


# ─── _build_genicam_control edge cases ────────────────────────────────────


class TestBuildGenicamControlEdgeCases:
    """Edge cases for _build_genicam_control."""

    def test_readonly_string_shows_text(self):
        from pybeamprofiler.dash_app import _build_genicam_control

        ctrl = _build_genicam_control(
            type(
                "C",
                (),
                {
                    "node_map": type(
                        "N", (), {"DeviceID": _SimulatedNode("ABC123", readonly=True)}
                    )()
                },
            )(),
            "DeviceID",
        )
        assert ctrl is not None
        layout_str = str(ctrl)
        assert "ABC123" in layout_str
        assert "genicam-sel" not in layout_str

    def test_empty_string_shows_text(self):
        from pybeamprofiler.dash_app import _build_genicam_control

        ctrl = _build_genicam_control(
            type("C", (), {"node_map": type("N", (), {"EmptyVal": _SimulatedNode("")})()})(),
            "EmptyVal",
        )
        assert ctrl is not None
        layout_str = str(ctrl)
        assert "genicam-sel" not in layout_str

    def test_generic_string_creates_select(self):
        from pybeamprofiler.dash_app import _build_genicam_control

        ctrl = _build_genicam_control(
            type("C", (), {"node_map": type("N", (), {"SomeStr": _SimulatedNode("hello")})()})(),
            "SomeStr",
        )
        assert ctrl is not None
        layout_str = str(ctrl)
        assert "genicam-sel" in layout_str


# ─── Dash callback helpers ─────────────────────────────────────────────────


def _fire_slider(fn: Any, slider_id: str, val: float) -> Any:
    """Invoke the slider+input paired callback as if the user dragged the
    slider (trigger = slider id, input value = None). Handles both the
    new tuple return shape and legacy scalar returns."""
    with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
        mock_ctx.triggered_id = slider_id
        return fn(val, None)


def _extract_callback(bp: BeamProfiler, output_id: str) -> Any:
    """Register callbacks on a throwaway Dash app and find one by output id.

    For ``live-graph`` (which has multiple writers — the live update loop,
    plus Auto-fit / Reset that emit a ``Patch``), this returns the
    ``update_live`` callback specifically, identified by its ``interval``
    Input which no other callback uses. For every other id, the callback that
    declares the output *first* wins, since that is the one that owns it.
    """
    from pybeamprofiler.dash_app import _register_callbacks

    app = dash.Dash(__name__)
    app.layout = html.Div()
    captured: dict[str, Any] = {}
    original_callback = app.callback

    def tracking_callback(*args, **kwargs):
        def decorator(f):
            key = str(args)
            captured[key] = f
            return original_callback(*args, **kwargs)(f)

        return decorator

    app.callback = tracking_callback  # ty: ignore[invalid-assignment]
    _register_callbacks(app, bp)

    if output_id == "live-graph":
        # Disambiguate the live update loop from the Patch-emitting
        # Auto-fit / Reset callbacks that also write to live-graph.
        for key, func in captured.items():
            if output_id in key and "interval" in key:
                return func
        return None

    # Several outputs now have more than one writer — the camera selector
    # also drives store-paused, the Play/Pause button, the settings panel and
    # the pixel-scale box. Prefer the callback that *owns* the output, i.e.
    # declares it first, and only then fall back to any writer.
    def _first_output(key: str) -> str:
        """The id.prop of the first Output, from its ``<Output `x.y`>`` repr."""
        marker = "<Output `"
        if marker not in key:
            return ""
        return key.split(marker, 1)[1].split("`", 1)[0]

    for key, func in captured.items():
        if output_id in _first_output(key):
            return func
    for key, func in captured.items():
        if output_id in key:
            return func
    return None


# ─── save_frame callback ──────────────────────────────────────────────────


class TestSaveFrameCallback:
    """Tests for the save-frame callback."""

    def test_save_returns_png_data(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.last_img = bp.camera.get_image()
        fn = _extract_callback(bp, "download-png")
        assert fn is not None
        result = fn(1)
        assert result is not None
        assert result["filename"].startswith("beam_")
        assert result["base64"] is True

    def test_save_no_image(self):
        bp = BeamProfiler(camera="simulated")
        bp.last_img = None
        fn = _extract_callback(bp, "download-png")
        assert fn is not None
        assert fn(1) is None

    def test_save_npy_returns_data(self):
        import base64
        import io

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.last_img = bp.camera.get_image()
        fn = _extract_callback(bp, "download-npy")
        assert fn is not None
        result = fn(1)
        assert result is not None
        assert result["filename"].endswith(".npy")
        assert result["base64"] is True
        # Round-trip verifies the bytes are a valid NumPy file.
        loaded = np.load(io.BytesIO(base64.b64decode(result["content"])))
        assert loaded.shape == bp.last_img.shape

    def test_save_npy_no_image(self):
        bp = BeamProfiler(camera="simulated")
        bp.last_img = None
        fn = _extract_callback(bp, "download-npy")
        assert fn is not None
        assert fn(1) is None


# ─── helper functions ─────────────────────────────────────────────────────


class TestSaturationHelpers:
    """Tests for `_saturation_max` and `_saturation_fraction`."""

    def test_uint8_max(self):
        from pybeamprofiler.dash_app import _saturation_max

        assert _saturation_max(np.zeros((4, 4), dtype=np.uint8)) == 255.0

    def test_uint16_max(self):
        from pybeamprofiler.dash_app import _saturation_max

        assert _saturation_max(np.zeros((4, 4), dtype=np.uint16)) == 65535.0

    def test_float_normalised(self):
        from pybeamprofiler.dash_app import _saturation_max

        img = np.array([[0.0, 0.5], [0.9, 1.0]], dtype=np.float32)
        assert _saturation_max(img) == 1.0

    def test_float_unnormalised(self):
        from pybeamprofiler.dash_app import _saturation_max

        img = np.array([[0.0, 100.0], [50.0, 200.0]], dtype=np.float32)
        assert _saturation_max(img) == 200.0

    def test_fraction_no_saturated_pixels(self):
        from pybeamprofiler.dash_app import _saturation_fraction

        img = np.full((10, 10), 100, dtype=np.uint8)
        assert _saturation_fraction(img) == 0.0

    def test_fraction_some_saturated_pixels(self):
        from pybeamprofiler.dash_app import _saturation_fraction

        img = np.full((10, 10), 100, dtype=np.uint8)
        img[0, :5] = 255
        assert _saturation_fraction(img) == pytest.approx(0.05)

    def test_fraction_empty(self):
        from pybeamprofiler.dash_app import _saturation_fraction

        assert _saturation_fraction(np.array([], dtype=np.uint8)) == 0.0


class TestAveragedImage:
    """Tests for the rolling N-frame averaging buffer."""

    def test_n_one_passes_through(self):
        from pybeamprofiler import dash_app as da

        da._reset_avg_state()
        img = np.full((4, 4), 10, dtype=np.uint8)
        out = da._averaged_image(img, 1)
        assert np.array_equal(out, img)
        assert len(da._avg_buffer) == 0

    def test_running_mean(self):
        from pybeamprofiler import dash_app as da

        da._reset_avg_state()
        a = np.full((4, 4), 0, dtype=np.uint8)
        b = np.full((4, 4), 100, dtype=np.uint8)
        da._averaged_image(a, 2)
        out = da._averaged_image(b, 2)
        assert np.array_equal(out, np.full((4, 4), 50, dtype=np.uint8))

    def test_buffer_resets_on_shape_change(self):
        from pybeamprofiler import dash_app as da

        da._reset_avg_state()
        da._averaged_image(np.zeros((4, 4), dtype=np.uint8), 4)
        da._averaged_image(np.zeros((8, 8), dtype=np.uint8), 4)
        assert len(da._avg_buffer) == 1
        assert da._avg_buffer_shape == (8, 8)

    def test_n_clamped_to_max(self):
        from pybeamprofiler import dash_app as da
        from pybeamprofiler.dash_app import _MAX_AVG_FRAMES

        da._reset_avg_state()
        da._averaged_image(np.zeros((4, 4), dtype=np.uint8), _MAX_AVG_FRAMES + 100)
        assert da._avg_buffer.maxlen == _MAX_AVG_FRAMES

    def test_n_zero_or_negative_treated_as_one(self):
        from pybeamprofiler import dash_app as da

        da._reset_avg_state()
        img = np.full((4, 4), 7, dtype=np.uint8)
        out = da._averaged_image(img, 0)
        assert np.array_equal(out, img)
        out = da._averaged_image(img, -3)
        assert np.array_equal(out, img)

    def test_uses_source_dtype_after_averaging(self):
        """Averaged frame must keep the source dtype so saturation math stays sane."""
        from pybeamprofiler import dash_app as da

        da._reset_avg_state()
        a = np.full((4, 4), 100, dtype=np.uint16)
        b = np.full((4, 4), 200, dtype=np.uint16)
        da._averaged_image(a, 2)
        out = da._averaged_image(b, 2)
        assert out.dtype == np.uint16

    def test_running_mean_evicts_oldest_when_full(self):
        """Verify the incremental sum stays in sync after the deque fills up.

        A naive implementation that forgets to subtract the evicted frame
        would drift; this check would catch that regression.
        """
        from pybeamprofiler import dash_app as da

        da._reset_avg_state()
        shape = (4, 4)
        da._averaged_image(np.full(shape, 10, dtype=np.uint8), 3)
        da._averaged_image(np.full(shape, 20, dtype=np.uint8), 3)
        da._averaged_image(np.full(shape, 30, dtype=np.uint8), 3)
        out = da._averaged_image(np.full(shape, 40, dtype=np.uint8), 3)
        assert np.array_equal(out, np.full(shape, 30, dtype=np.uint8))

    def test_reset_avg_state_clears_running_sum(self):
        from pybeamprofiler import dash_app as da

        da._reset_avg_state()
        da._averaged_image(np.full((4, 4), 10, dtype=np.uint8), 3)
        assert da._avg_running_sum is not None
        da._reset_avg_state()
        assert da._avg_running_sum is None
        assert len(da._avg_buffer) == 0


# ─── _measured_fps and _build_status helpers ──────────────────────────────


class TestMeasuredFps:
    def test_empty_returns_zero(self):
        from pybeamprofiler import dash_app as da

        da._recent_frame_times.clear()
        assert da._measured_fps() == 0.0

    def test_single_sample_returns_zero(self):
        from pybeamprofiler import dash_app as da

        da._recent_frame_times.clear()
        da._recent_frame_times.append(1.0)
        assert da._measured_fps() == 0.0

    def test_zero_span_returns_zero(self):
        from pybeamprofiler import dash_app as da

        da._recent_frame_times.clear()
        da._recent_frame_times.append(5.0)
        da._recent_frame_times.append(5.0)
        assert da._measured_fps() == 0.0

    def test_positive_span_returns_rate(self):
        from pybeamprofiler import dash_app as da

        da._recent_frame_times.clear()
        for t in (0.0, 0.1, 0.2, 0.3):
            da._recent_frame_times.append(t)
        assert da._measured_fps() == pytest.approx(10.0)


class TestBuildStatus:
    def test_no_camera_only_frame(self):
        from pybeamprofiler import dash_app as da

        da._recent_frame_times.clear()
        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        children = da._build_status(bp, np.zeros((4, 4), dtype=np.uint8), 7)
        text = str(children)
        assert "Frame #7" in text
        assert "Exp" not in text and "Gain" not in text

    def test_with_camera_includes_exposure_and_gain(self):
        from pybeamprofiler import dash_app as da

        da._recent_frame_times.clear()
        bp = BeamProfiler(camera="simulated")
        children = da._build_status(bp, np.zeros((4, 4), dtype=np.uint8), 1)
        text = str(children)
        assert "Exp" in text
        assert "Gain" in text

    def test_long_exposure_renders_seconds(self):
        from pybeamprofiler import dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.set_exposure(2.5)
        children = da._build_status(bp, np.zeros((4, 4), dtype=np.uint8), 1)
        text = str(children)
        assert "2.50 s" in text

    def test_saturation_warning_appears(self):
        from pybeamprofiler import dash_app as da

        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:5, :] = 255  # 5% saturated, well above the 0.1% threshold
        children = da._build_status(bp, img, 1)
        text = str(children)
        assert "saturated" in text
        assert "5.0%" in text

    def test_no_saturation_warning_below_threshold(self):
        from pybeamprofiler import dash_app as da

        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        img = np.zeros((100, 100), dtype=np.uint8)
        img[0, 0] = 255  # 0.01%, below the 0.1% threshold
        children = da._build_status(bp, img, 1)
        assert "saturated" not in str(children)


# ─── update_live extra branches ───────────────────────────────────────────


class TestUpdateLiveExtraBranches:
    """Cover the timeout, lock-contention and definition-change branches."""

    @staticmethod
    def _get_update_fn(bp: BeamProfiler):
        return _extract_callback(bp, "live-graph")

    def test_timeout_returns_no_update(self):
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        bp.camera.get_image = MagicMock(  # ty: ignore[invalid-assignment]
            side_effect=TimeoutError("no frame")
        )
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
        assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)

    def test_lock_contention_returns_no_update(self):
        from pybeamprofiler import dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = self._get_update_fn(bp)
        assert fn is not None
        # Hold the lock from another "thread" (simulated via direct acquire).
        assert da._callback_lock.acquire(blocking=False)
        try:
            result = fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
            assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)
        finally:
            da._callback_lock.release()

    def test_definition_change_propagates(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        bp.definition = "gaussian"
        fn = self._get_update_fn(bp)
        assert fn is not None
        fn(1, False, True, "Hot", True, None, None, 0, "1d", "fwhm", True, 1)
        assert bp.definition == "fwhm"

    def test_manual_zmax_uses_user_value(self):
        """When auto-range is off and zmax is provided, it should be honoured."""
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = self._get_update_fn(bp)
        assert fn is not None
        with patch("pybeamprofiler.dash_app.build_figure") as mock_bf:
            mock_bf.return_value = "FIG"
            fn(1, False, True, "Hot", False, 10.0, 200.0, 0, "1d", "gaussian", True, 1)
            kwargs = mock_bf.call_args.kwargs
            assert kwargs["zmin"] == 10.0
            assert kwargs["zmax"] == 200.0

    def test_manual_zmax_falls_back_to_dtype_max(self):
        """When auto-range is off and zmax is None, the dtype max is used."""
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = self._get_update_fn(bp)
        assert fn is not None
        with patch("pybeamprofiler.dash_app.build_figure") as mock_bf:
            mock_bf.return_value = "FIG"
            fn(1, False, True, "Hot", False, None, None, 0, "1d", "gaussian", True, 1)
            assert mock_bf.call_args.kwargs["zmax"] == 255.0


# ─── side-effect callbacks (pause / exposure) clear avg buffer ────────────


class TestPauseClearsAvgBuffer:
    def test_pause_clears_buffer(self):
        from pybeamprofiler import dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        # Prime the buffer with one frame.
        da._averaged_image(np.zeros((4, 4), dtype=np.uint8), 4)
        assert len(da._avg_buffer) == 1
        fn = _extract_callback(bp, "store-paused")
        assert fn is not None
        fn(1, False)
        assert len(da._avg_buffer) == 0

    def test_exposure_change_clears_buffer(self):
        from pybeamprofiler import dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        da._averaged_image(np.zeros((4, 4), dtype=np.uint8), 4)
        assert len(da._avg_buffer) == 1
        fn = _extract_callback(bp, "slider-exposure")
        assert fn is not None
        _fire_slider(fn, "slider-exposure", 50.0)
        assert len(da._avg_buffer) == 0


# ─── create_app handles initial-frame failure gracefully ──────────────────


class TestCreateAppInitialFrame:
    def test_initial_frame_exception_logged(self, caplog):
        """If the first ``get_image`` call raises, ``create_app`` should log
        a warning and still return a working app with an empty figure."""
        import logging
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.get_image = MagicMock(  # ty: ignore[invalid-assignment]
            side_effect=RuntimeError("boom")
        )
        with caplog.at_level(logging.WARNING):
            app = create_app(bp)
        assert app is not None
        assert any("Could not grab initial frame" in r.message for r in caplog.records)


# ─── update_live: img-is-None and outer exception branches ────────────────


class TestUpdateLiveImgPaths:
    @staticmethod
    def _get_update_fn(bp: BeamProfiler):
        return _extract_callback(bp, "live-graph")

    def test_no_image_returns_no_update(self):
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        bp.camera.get_image = MagicMock(  # ty: ignore[invalid-assignment]
            return_value=None
        )
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
        assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)

    def test_static_mode_uses_last_img(self):
        """Non-camera mode should source img from ``bp.last_img``."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.last_img = bp.camera.get_image()
        bp._mode = "file"
        bp.camera = None
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
        assert hasattr(result[0], "data")
        assert result[3] == 1

    def test_outer_exception_returns_no_update(self):
        """Any exception during analyze should be caught and logged."""
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        bp.analyze = MagicMock(  # ty: ignore[invalid-assignment]
            side_effect=RuntimeError("explode")
        )
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
        assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)


# ─── slider restart-on-stopped branches ───────────────────────────────────


class TestSliderRestartBranches:
    """When a setter call stops the camera, the slider callback restarts it."""

    def test_set_exposure_restarts_when_stopped(self):
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        cam = bp.camera
        cam.start_acquisition()
        assert cam.is_acquiring

        original_set = cam.set_exposure

        def stop_then_set(v):
            cam.stop_acquisition()
            original_set(v)

        cam.set_exposure = MagicMock(  # ty: ignore[invalid-assignment]
            side_effect=stop_then_set
        )
        fn = _extract_callback(bp, "slider-exposure")
        assert fn is not None
        _fire_slider(fn, "slider-exposure", 50.0)
        assert cam.is_acquiring

    def test_set_gain_restarts_when_stopped(self):
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        cam = bp.camera
        cam.start_acquisition()

        original_set = cam.set_gain

        def stop_then_set(v):
            cam.stop_acquisition()
            original_set(v)

        cam.set_gain = MagicMock(  # ty: ignore[invalid-assignment]
            side_effect=stop_then_set
        )
        fn = _extract_callback(bp, "slider-gain")
        assert fn is not None
        _fire_slider(fn, "slider-gain", 3.0)
        assert cam.is_acquiring


# ─── ROI extra branches ───────────────────────────────────────────────────


class TestROIExtraBranches:
    @staticmethod
    def _find(bp: BeamProfiler, key: str):
        from pybeamprofiler.dash_app import _register_callbacks

        app = dash.Dash(__name__)
        app.layout = html.Div()
        captured: dict[str, Any] = {}
        original = app.callback

        def tracking(*args, **kwargs):
            def deco(f):
                captured[str(args)] = f
                return original(*args, **kwargs)(f)

            return deco

        app.callback = tracking  # ty: ignore[invalid-assignment]
        _register_callbacks(app, bp)
        for k, fn in captured.items():
            if key in k:
                return fn
        return None

    def test_apply_roi_exception_returns_error_string(self):
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        bp.camera.set_roi = MagicMock(  # ty: ignore[unresolved-attribute]
            side_effect=RuntimeError("bad roi")
        )
        fn = self._find(bp, "div-roi-status")
        assert fn is not None
        result = fn(1, 0, 0, 100, 100)
        assert "Error" in result and "bad roi" in result

    def test_reset_roi_no_camera_returns_no_camera(self):
        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        fn = self._find(bp, "btn-roi-reset")
        # When the camera is missing, the ROI callback isn't even registered
        # (``has_roi`` is False), so we re-register with a real camera then
        # null it before calling.  This exercises the in-callback guard.
        bp.camera = BeamProfiler(camera="simulated").camera
        fn = self._find(bp, "btn-roi-reset")
        assert fn is not None
        bp.camera = None
        assert fn(1) == (0, 0, 0, 0, "No camera")

    def test_reset_roi_restarts_when_was_acquiring(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        assert bp.camera.is_acquiring
        fn = self._find(bp, "btn-roi-reset")
        assert fn is not None
        result = fn(1)
        assert result[4] == "Reset to full sensor"
        assert bp.camera.is_acquiring  # restarted

    def test_reset_roi_exception_returns_error_tuple(self):
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.set_roi = MagicMock(  # ty: ignore[unresolved-attribute]
            side_effect=RuntimeError("bad reset")
        )
        fn = self._find(bp, "btn-roi-reset")
        assert fn is not None
        result = fn(1)
        assert result[:4] == (0, 0, 0, 0)
        assert "Error" in result[4]


# ─── GenICam pattern-matching extra branches ──────────────────────────────


class TestGenicamCallbackBranches:
    @staticmethod
    def _capture(bp: BeamProfiler) -> dict[str, Any]:
        from pybeamprofiler.dash_app import _register_callbacks

        app = dash.Dash(__name__)
        app.layout = html.Div()
        captured: dict[str, Any] = {}
        original = app.callback

        def tracking(*args, **kwargs):
            def deco(f):
                captured[str(args)] = f
                return original(*args, **kwargs)(f)

            return deco

        app.callback = tracking  # ty: ignore[invalid-assignment]
        _register_callbacks(app, bp)
        return captured

    def _find(self, captured: dict[str, Any], key: str):
        for k, fn in captured.items():
            if key in k:
                return fn
        return None

    def test_select_none_value_returns_no_update(self):
        bp = BeamProfiler(camera="simulated")
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-sel")
        assert fn is not None
        assert isinstance(fn(None), dash._no_update.NoUpdate)

    def test_switch_no_camera_returns_no_update(self):
        bp = BeamProfiler(camera="simulated")
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-sw")
        assert fn is not None
        bp.camera = None
        assert isinstance(fn(True), dash._no_update.NoUpdate)

    def test_numeric_set_exception_swallowed(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-num")
        assert fn is not None
        # The simulated node accepts any numeric write, so monkey-patch the
        # node's __setattr__ to raise.
        assert bp.camera is not None
        nm = bp.camera.node_map  # ty: ignore[unresolved-attribute]
        node = getattr(nm, "ExposureTime")

        def boom(_self, _v):
            raise RuntimeError("nope")

        with (
            patch("pybeamprofiler.dash_app.ctx") as mock_ctx,
            patch.object(type(node), "value", property(lambda s: 0.0, boom)),
        ):
            mock_ctx.triggered_id = {"type": "genicam-num", "feature": "ExposureTime"}
            # Slider-triggered → callback mirrors value onto the companion input.
            result = fn(1.0, None)
            assert result[1] == 1.0

    def test_numeric_ignores_a_trigger_without_a_feature(self):
        """Guards against a malformed pattern-matching id reaching the setter
        and raising KeyError inside the callback."""
        bp = BeamProfiler(camera="simulated")
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-num")
        assert fn is not None

        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = {"type": "genicam-num"}  # no "feature"
            result = fn(1.0, None)
        assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)

    def test_numeric_ignores_a_non_dict_trigger(self):
        bp = BeamProfiler(camera="simulated")
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-num")
        assert fn is not None

        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "not-a-pattern-id"
            result = fn(1.0, 2.0)
        assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)

    def test_select_set_exception_swallowed(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-sel")
        assert fn is not None
        assert bp.camera is not None
        nm = bp.camera.node_map  # ty: ignore[unresolved-attribute]
        node = getattr(nm, "ExposureAuto")

        def boom(_self, _v):
            raise RuntimeError("nope")

        with (
            patch("pybeamprofiler.dash_app.ctx") as mock_ctx,
            patch.object(type(node), "value", property(lambda s: "Off", boom)),
        ):
            mock_ctx.triggered_id = {"type": "genicam-sel", "feature": "ExposureAuto"}
            assert fn("Off") == "Off"

    def test_numeric_setter_restarts_when_stopped(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        cam = bp.camera
        cam.start_acquisition()
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-num")
        assert fn is not None

        nm = cam.node_map  # ty: ignore[unresolved-attribute]
        node = getattr(nm, "ExposureTime")

        def stop_then_set(self, v):
            cam.stop_acquisition()
            self._value = v

        with (
            patch("pybeamprofiler.dash_app.ctx") as mock_ctx,
            patch.object(type(node), "value", property(lambda s: s._value, stop_then_set)),
        ):
            mock_ctx.triggered_id = {"type": "genicam-num", "feature": "ExposureTime"}
            fn(123.0, None)
        assert bp.camera.is_acquiring  # restarted

    def test_select_setter_restarts_when_stopped(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        cam = bp.camera
        cam.start_acquisition()
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-sel")
        assert fn is not None

        nm = cam.node_map  # ty: ignore[unresolved-attribute]
        node = getattr(nm, "ExposureAuto")

        def stop_then_set(self, v):
            cam.stop_acquisition()
            self._value = v

        with (
            patch("pybeamprofiler.dash_app.ctx") as mock_ctx,
            patch.object(type(node), "value", property(lambda s: s._value, stop_then_set)),
        ):
            mock_ctx.triggered_id = {"type": "genicam-sel", "feature": "ExposureAuto"}
            fn("Off")
        assert bp.camera.is_acquiring

    def test_switch_setter_restarts_when_stopped(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        cam = bp.camera
        cam.start_acquisition()
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-sw")
        assert fn is not None

        nm = cam.node_map  # ty: ignore[unresolved-attribute]
        node = getattr(nm, "ReverseX", None)
        if node is None:
            pytest.skip("No boolean node available")

        def stop_then_set(self, v):
            cam.stop_acquisition()
            self._value = v

        with (
            patch("pybeamprofiler.dash_app.ctx") as mock_ctx,
            patch.object(type(node), "value", property(lambda s: s._value, stop_then_set)),
        ):
            mock_ctx.triggered_id = {"type": "genicam-sw", "feature": "ReverseX"}
            fn(True)
        assert bp.camera.is_acquiring

    def test_switch_set_exception_swallowed(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        captured = self._capture(bp)
        fn = self._find(captured, "genicam-sw")
        assert fn is not None
        assert bp.camera is not None
        # Pick any boolean-valued node from the simulated map.
        nm = bp.camera.node_map  # ty: ignore[unresolved-attribute]
        node = getattr(nm, "ReverseX", None)
        if node is None:
            pytest.skip("No boolean node available")

        def boom(_self, _v):
            raise RuntimeError("nope")

        with (
            patch("pybeamprofiler.dash_app.ctx") as mock_ctx,
            patch.object(type(node), "value", property(lambda s: False, boom)),
        ):
            mock_ctx.triggered_id = {"type": "genicam-sw", "feature": "ReverseX"}
            assert fn(True) is True


# ─── Pre-existing edge cases (cheap one-liners) ───────────────────────────


class TestPreexistingEdgeCases:
    """Cover marginal `except: pass` defensive paths."""

    def test_saturation_fraction_float_dtype_with_max(self):
        from pybeamprofiler.dash_app import _saturation_fraction

        img = np.array([[0.0, 1.0, 1.0], [0.5, 0.5, 0.5]], dtype=np.float32)
        # Two of six pixels at the (observed) max of 1.0 → 1/3.
        assert _saturation_fraction(img) == pytest.approx(2 / 6)

    def test_exposure_controls_handles_bad_range(self):
        """``exposure_range`` returning a non-iterable shouldn't crash."""
        from pybeamprofiler.dash_app import _exposure_controls

        class FakeCam:
            exposure_range = 5  # not unpackable
            exposure_time = 0.01

        ctrls = _exposure_controls(FakeCam())
        assert ctrls  # didn't crash, fell back to defaults

    def test_gain_controls_handles_bad_range(self):
        from pybeamprofiler.dash_app import _gain_controls

        class FakeCam:
            gain_range = "oops"  # not unpackable to two floats
            gain = 0.0

        ctrls = _gain_controls(FakeCam())
        assert ctrls

    def test_roi_controls_no_roi_info_returns_none(self):
        from pybeamprofiler.dash_app import _roi_controls

        class FakeCam:
            pass

        assert _roi_controls(FakeCam()) is None

    def test_build_genicam_control_unsupported_value_returns_none(self):
        """A node whose value is an exotic type (e.g. tuple) yields no control."""
        from pybeamprofiler.dash_app import _build_genicam_control

        class FakeNode:
            value = (1, 2, 3)  # not bool, str, int, float

        cam = type("C", (), {"node_map": type("N", (), {"Weird": FakeNode()})()})()
        assert _build_genicam_control(cam, "Weird") is None

    def test_build_setting_items_discover_failure_logged(self, caplog):
        """When ``_discover_features`` raises, the panel still builds."""
        import logging
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera._discover_features = MagicMock(  # ty: ignore[invalid-assignment]
            side_effect=RuntimeError("discovery boom")
        )
        with caplog.at_level(logging.WARNING):
            items = _build_setting_items(bp)
        assert items  # camera info accordion still present
        assert any("Feature discovery failed" in r.message for r in caplog.records)


# ─── toggle_colorscale callback ───────────────────────────────────────────


class TestToggleColorscaleCallback:
    def test_color_on_enables(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "dropdown-colorscale")
        assert fn is not None
        assert fn(True) is False

    def test_color_off_disables(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "dropdown-colorscale")
        assert fn is not None
        assert fn(False) is True


# ─── set_pixel_scale callback ─────────────────────────────────────────────


class TestPixelScaleCallback:
    def test_valid_pixel_scale(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "input-pixel-scale")
        assert fn is not None
        fn(1, None, 5.5)
        assert bp.pixel_size == 5.5

    def test_none_pixel_scale_keeps_existing(self):
        bp = BeamProfiler(camera="simulated")
        original = bp.pixel_size
        fn = _extract_callback(bp, "input-pixel-scale")
        assert fn is not None
        fn(1, None, None)
        assert bp.pixel_size == original


# ─── set_exposure callback ────────────────────────────────────────────────


class TestExposureCallback:
    def test_set_exposure_via_callback(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "slider-exposure")
        assert fn is not None
        result = _fire_slider(fn, "slider-exposure", 50.0)
        # Slider-triggered → input mirrors the new value, slider gets no_update.
        assert result[1] == 50.0
        assert bp.camera is not None
        assert abs(bp.camera.exposure_time - 0.05) < 1e-6

    def test_set_exposure_via_input_box(self):
        """Typing a precise value into the companion number input applies
        it to the camera and mirrors back to the slider."""
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "input-exposure")
        assert fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "input-exposure"
            result = fn(None, 123.456)
        assert result[0] == 123.456  # slider is synced
        assert bp.camera is not None
        assert abs(bp.camera.exposure_time - 0.123456) < 1e-6


# ─── set_gain callback ───────────────────────────────────────────────────


class TestGainCallback:
    def test_set_gain_via_callback(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "slider-gain")
        assert fn is not None
        result = _fire_slider(fn, "slider-gain", 5.0)
        assert result[1] == 5.0
        assert bp.camera is not None
        assert bp.camera.gain == 5.0

    def test_set_gain_via_input_box(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "input-gain")
        assert fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "input-gain"
            result = fn(None, 7.25)
        assert result[0] == 7.25
        assert bp.camera is not None
        assert bp.camera.gain == 7.25


# ─── ROI callbacks ────────────────────────────────────────────────────────


class TestROICallbacks:
    def test_apply_roi(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = _extract_callback(bp, "div-roi-status")
        assert fn is not None
        result = fn(1, 0, 0, 512, 512)
        assert "512" in result

    def test_reset_roi(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        from pybeamprofiler.dash_app import _register_callbacks

        app = dash.Dash(__name__)
        app.layout = html.Div()
        captured: dict[str, Any] = {}
        original_callback = app.callback

        def tracking_callback(*args, **kwargs):
            def decorator(f):
                key = str(args)
                captured[key] = f
                return original_callback(*args, **kwargs)(f)

            return decorator

        app.callback = tracking_callback  # ty: ignore[invalid-assignment]
        _register_callbacks(app, bp)

        for key, func in captured.items():
            if "btn-roi-reset" in key:
                result = func(1)
                assert result[4] == "Reset to full sensor"
                return
        pytest.fail("reset_roi callback not found")


# ─── GenicamSetCallbacks ─────────────────────────────────────────────────


class TestGenicamSetCallbacks:
    """Test GenICam pattern-matching callbacks."""

    @staticmethod
    def _extract_genicam_callbacks(bp: BeamProfiler) -> dict[str, Any]:
        from pybeamprofiler.dash_app import _register_callbacks

        app = dash.Dash(__name__)
        app.layout = html.Div()
        captured: dict[str, Any] = {}
        original_callback = app.callback

        def tracking_callback(*args, **kwargs):
            def decorator(f):
                key = str(args)
                captured[key] = f
                return original_callback(*args, **kwargs)(f)

            return decorator

        app.callback = tracking_callback  # ty: ignore[invalid-assignment]
        _register_callbacks(app, bp)
        return captured

    def test_numeric_callback_sets_value(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        captured = self._extract_genicam_callbacks(bp)
        set_fn = None
        for key, func in captured.items():
            if "genicam-num" in key:
                set_fn = func
                break
        assert set_fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = {"type": "genicam-num", "feature": "ExposureTime"}
            result = set_fn(50000.0, None)
            # Slider-triggered → input mirrors, slider returns no_update.
            assert result[1] == 50000.0

    def test_numeric_callback_via_input_box(self):
        """Typing into the companion number input commits to the camera
        and mirrors back to the slider."""
        bp = BeamProfiler(camera="simulated")
        captured = self._extract_genicam_callbacks(bp)
        set_fn = None
        for key, func in captured.items():
            if "genicam-num" in key:
                set_fn = func
                break
        assert set_fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = {"type": "genicam-num-input", "feature": "ExposureTime"}
            result = set_fn(None, 12345.0)
            assert result[0] == 12345.0

    def test_select_callback_sets_value(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        captured = self._extract_genicam_callbacks(bp)
        set_fn = None
        for key, func in captured.items():
            if "genicam-sel" in key:
                set_fn = func
                break
        assert set_fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = {"type": "genicam-sel", "feature": "ExposureAuto"}
            result = set_fn("Off")
            assert result == "Off"

    def test_switch_callback_sets_value(self):
        from unittest.mock import patch

        bp = BeamProfiler(camera="simulated")
        captured = self._extract_genicam_callbacks(bp)
        set_fn = None
        for key, func in captured.items():
            if "genicam-sw" in key:
                set_fn = func
                break
        assert set_fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = {"type": "genicam-sw", "feature": "GammaEnable"}
            result = set_fn(True)
            assert result is True

    def test_numeric_none_returns_no_update(self):
        bp = BeamProfiler(camera="simulated")
        captured = self._extract_genicam_callbacks(bp)
        set_fn = None
        for key, func in captured.items():
            if "genicam-num" in key:
                set_fn = func
                break
        assert set_fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = {"type": "genicam-num", "feature": "ExposureTime"}
            result = set_fn(None, None)
        # Both outputs should be no_update when nothing to commit.
        assert isinstance(result[0], dash._no_update.NoUpdate)
        assert isinstance(result[1], dash._no_update.NoUpdate)


# ─── update_live callback ────────────────────────────────────────────────


class TestUpdateLiveCallback:
    """Test the synchronous update_live callback."""

    @staticmethod
    def _get_update_fn(bp: BeamProfiler):
        fn = _extract_callback(bp, "live-graph")
        return fn

    def test_paused_returns_no_update(self):
        bp = BeamProfiler(camera="simulated")
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, True, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
        assert all(isinstance(r, dash._no_update.NoUpdate) for r in result)

    def test_live_update_returns_figure(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
        assert len(result) == 4
        fig = result[0]
        assert hasattr(fig, "data")
        assert result[3] == 1

    def test_live_update_switches_fit_method(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        bp.fit_method = "1d"
        fn = self._get_update_fn(bp)
        assert fn is not None
        fn(1, False, True, "Hot", True, None, None, 0, "2d", "gaussian", True, 1)
        assert bp.fit_method == "2d"

    def test_live_update_manual_range(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, False, True, "Hot", False, 10.0, 200.0, 0, "1d", "gaussian", True, 1)
        assert len(result) == 4

    def test_live_update_greyscale(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, False, False, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
        assert len(result) == 4

    def test_live_update_averages_frames(self):
        """N>1 averaging returns valid figure and populates the buffer."""
        from pybeamprofiler import dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = self._get_update_fn(bp)
        assert fn is not None
        da._reset_avg_state()
        for _ in range(3):
            fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 4)
        assert len(da._avg_buffer) == 3
        assert da._avg_buffer.maxlen == 4

    def test_live_update_status_includes_exposure_and_gain(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = self._get_update_fn(bp)
        assert fn is not None
        result = fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
        status_text = str(result[2])
        assert "Frame #" in status_text
        assert "Exp" in status_text
        assert "Gain" in status_text


# ─── Exposure/Gain error branches ────────────────────────────────────────


class TestCallbackErrorBranches:
    """Test error handling in exposure/gain/ROI callbacks."""

    def test_exposure_exception_handled(self):
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.set_exposure = MagicMock(  # ty: ignore[invalid-assignment]
            side_effect=RuntimeError("boom")
        )
        fn = _extract_callback(bp, "slider-exposure")
        assert fn is not None
        result = _fire_slider(fn, "slider-exposure", 50.0)
        # Exception is swallowed; the other control still gets mirrored.
        assert result[1] == 50.0

    def test_gain_exception_handled(self):
        from unittest.mock import MagicMock

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.set_gain = MagicMock(  # ty: ignore[invalid-assignment]
            side_effect=RuntimeError("boom")
        )
        fn = _extract_callback(bp, "slider-gain")
        assert fn is not None
        result = _fire_slider(fn, "slider-gain", 5.0)
        assert result[1] == 5.0

    def test_roi_apply_no_camera(self):
        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        from pybeamprofiler.dash_app import _register_callbacks

        app = dash.Dash(__name__)
        app.layout = html.Div()
        captured: dict[str, Any] = {}
        original_callback = app.callback

        def tracking_callback(*args, **kwargs):
            def decorator(f):
                captured[str(args)] = f
                return original_callback(*args, **kwargs)(f)

            return decorator

        app2 = dash.Dash(__name__)
        app2.layout = html.Div()
        bp2 = BeamProfiler(camera="simulated")
        captured2: dict[str, Any] = {}
        original_callback2 = app2.callback

        def tracking_callback2(*args, **kwargs):
            def decorator(f):
                captured2[str(args)] = f
                return original_callback2(*args, **kwargs)(f)

            return decorator

        app2.callback = tracking_callback2  # ty: ignore[invalid-assignment]
        _register_callbacks(app2, bp2)
        for key, func in captured2.items():
            if "btn-roi-apply" in key:
                bp2.camera = None
                result = func(1, 0, 0, 512, 512)
                assert result == "No camera"
                return
        pytest.fail("apply_roi not found")

    def test_pause_no_camera_items(self):
        bp = BeamProfiler(camera="simulated")
        bp.camera = None
        bp._mode = "file"
        fn = _extract_callback(bp, "store-paused")
        if fn is not None:
            result = fn(1, False)
            settings_body = result[3]
            assert "No camera" in str(settings_body) or isinstance(settings_body, dbc.Accordion)


# ─── Small branch coverage for helpers and callbacks ────────────────────────


@requires_genicam
class TestIsReadonlyEAccessMode:
    """``_is_readonly`` branches that depend on GenICam's ``EAccessMode``
    enum — exercised only when a node exposes ``get_access_mode`` AND
    ``genicam.genapi`` is importable (it is, in our dev env)."""

    @pytest.mark.parametrize("mode_name", ["RO", "NA", "NI"])
    def test_access_mode_readonly_states(self, mode_name: str):
        # Import the real enum so the ``mode in (...)`` comparison hits
        # the true branch inside the helper.
        from genicam.genapi import EAccessMode  # ty: ignore[unresolved-import]

        from pybeamprofiler.dash_app import _is_readonly

        mode = getattr(EAccessMode, mode_name)

        class Node:
            def __init__(self, m):
                self._m = m

            def get_access_mode(self):
                return self._m

        assert _is_readonly(Node(mode)) is True

    def test_access_mode_writable(self):
        from genicam.genapi import EAccessMode  # ty: ignore[unresolved-import]

        from pybeamprofiler.dash_app import _is_readonly

        class Node:
            def get_access_mode(self):
                return EAccessMode.RW

        assert _is_readonly(Node()) is False

    def test_no_genicam_installed_returns_false(self):
        """When ``genicam.genapi`` isn't importable the helper short-
        circuits to ``False`` — we can't know whether the node is RO."""
        from pybeamprofiler.dash_app import _is_readonly

        class Node:
            def get_access_mode(self):
                return 3  # would match RO if the enum were loaded

        with patch("pybeamprofiler.dash_app._EAccessMode", None):
            assert _is_readonly(Node()) is False


class TestBuildGenicamControlNumericCoercion:
    """The numeric branch in ``_build_genicam_control`` is wrapped in a
    ``try/except (TypeError, ValueError)`` so malformed min/max still
    produce *something* instead of crashing the whole accordion."""

    def test_non_numeric_min_falls_through(self):
        """Min that can't be coerced to float must hit the
        ``except (TypeError, ValueError): pass`` path (lines 557-558)."""
        from pybeamprofiler.dash_app import _build_genicam_control

        class WeirdNode:
            value = "nope"
            min = "nope"
            max = "still nope"

        class MockCam:
            class node_map:
                WeirdThing = WeirdNode()

        # Should not raise; falls through to string/unknown fallback and
        # ultimately returns *some* component (or None).  The important
        # bit is that the except clause ran without propagating.
        _build_genicam_control(MockCam(), "WeirdThing")


class TestRoiControlsBranch:
    """``_roi_controls`` swallows exceptions from ``cam.roi_info`` so a
    flaky camera doesn't blow up the panel (lines 644-645)."""

    def test_roi_info_raises_returns_none(self):
        from pybeamprofiler.dash_app import _roi_controls

        # Using ``__getattr__`` lets ``hasattr`` succeed on the first probe
        # (returns a stub dict) but the subsequent ``getattr`` in the
        # ``try/except`` block raises — simulating a camera whose state
        # flipped between the two calls.
        class FlakyCam:
            def __init__(self) -> None:
                self._n = 0

            def set_roi(self, **_: Any) -> None:
                pass

            def __getattr__(self, name: str) -> Any:
                if name == "roi_info":
                    self._n += 1
                    if self._n == 1:
                        return {}
                    raise RuntimeError("cable disconnected")
                raise AttributeError(name)

        assert _roi_controls(FlakyCam()) is None


class TestBuildSettingItemsInfoNodeFailure:
    """When a GenICam info node raises on ``.value`` access (e.g. a
    disconnected device), ``_build_setting_items`` must keep going and
    still render the rest of the accordion (lines 780-781)."""

    def test_info_node_value_exception_is_swallowed(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        class Boom:
            @property
            def value(self):
                raise RuntimeError("no response")

        # Wipe the eager device_* attrs so the code falls through into
        # the node_map lookup branch we want to exercise. ``setattr`` is
        # used here so ``ty`` doesn't trip over the narrower ``Camera``
        # base type (the real attributes live on the subclasses).
        cam: Any = bp.camera
        setattr(cam, "device_model", None)
        setattr(cam, "device_vendor", None)
        setattr(cam, "serial_number", None)
        nm: Any = getattr(cam, "node_map")
        nm.DeviceModelName = Boom()
        nm.DeviceVendorName = Boom()
        nm.DeviceSerialNumber = Boom()

        items = _build_setting_items(bp)
        # We still get the accordion items — the info node failure was
        # caught and the rest of rendering proceeded.
        assert len(items) > 0


class TestExposureGainNoneValue:
    """If both slider and input fire with ``None`` (e.g. during initial
    clearing), the paired callbacks must short-circuit to two
    ``dash.no_update`` sentinels (lines 1680, 1711)."""

    def test_set_exposure_both_none(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "slider-exposure")
        assert fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "slider-exposure"
            result = fn(None, None)
        assert result == (dash.no_update, dash.no_update)

    def test_set_gain_both_none(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "slider-gain")
        assert fn is not None
        with patch("pybeamprofiler.dash_app.ctx") as mock_ctx:
            mock_ctx.triggered_id = "slider-gain"
            result = fn(None, None)
        assert result == (dash.no_update, dash.no_update)


class TestApplyRoiMissingFields:
    """``apply_roi`` returns a friendly message when any of the four
    input boxes is empty (line 1744) — the Apply button on an
    uninitialised form should not crash."""

    def test_apply_roi_none_field(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback(bp, "div-roi-status")
        assert fn is not None
        result = fn(1, None, 0, 512, 512)
        assert "Please enter" in result


# ─── Server-managed Auto-fit / Reset zoom ──────────────────────────────────


def _extract_callback_by_input(bp: BeamProfiler, input_id: str) -> Any:
    """Find a registered callback whose Input id matches ``input_id``.

    Several callbacks now write to ``store-zoom-range`` (Auto-fit,
    Reset), so we disambiguate by the trigger they listen to.
    """
    from pybeamprofiler.dash_app import _register_callbacks

    app = dash.Dash(__name__)
    app.layout = html.Div()
    captured: list[tuple[str, Any]] = []
    original_callback = app.callback

    def tracking_callback(*args, **kwargs):
        def decorator(f):
            captured.append((str(args), f))
            return original_callback(*args, **kwargs)(f)

        return decorator

    app.callback = tracking_callback  # ty: ignore[invalid-assignment]
    _register_callbacks(app, bp)

    for key, func in captured:
        if input_id in key:
            return func
    return None


class TestZoomCallbacks:
    """Auto-fit and Reset return a layout-only ``Patch`` for instant
    feedback (works even when paused), and publish to the module-level
    ``_zoom_range`` so ``update_live`` keeps the same view on subsequent
    live ticks."""

    def test_auto_fit_no_clicks_returns_no_update(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback_by_input(bp, "btn-zoom-fit")
        assert fn is not None
        assert isinstance(fn(0), dash._no_update.NoUpdate)

    def test_auto_fit_without_fit_data_returns_no_update(self):
        bp = BeamProfiler(camera="simulated")
        bp._last_popt_x = None
        bp._last_popt_y = None
        fn = _extract_callback_by_input(bp, "btn-zoom-fit")
        assert fn is not None
        assert isinstance(fn(1), dash._no_update.NoUpdate)

    def test_auto_fit_publishes_zoom_and_returns_patch(self):
        """Auto-fit must publish to ``_zoom_range`` (so the next tick of
        ``update_live`` keeps the view) AND return a Patch (so the click
        is reflected immediately, even when paused)."""
        from dash import Patch

        import pybeamprofiler.dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        bp.fit_method = "1d"
        bp.analyze(img)
        assert bp._last_popt_x is not None and bp._last_popt_y is not None
        fn = _extract_callback_by_input(bp, "btn-zoom-fit")
        assert fn is not None
        original = da._zoom_range
        da._zoom_range = None
        try:
            patch = fn(1)
            assert isinstance(patch, Patch)
            assert da._zoom_range is not None
            zoom = da._zoom_range
            cx = bp._last_popt_x[1] * bp.pixel_size
            cy = bp._last_popt_y[1] * bp.pixel_size
            assert abs((zoom["x"][0] + zoom["x"][1]) / 2 - cx) < bp.pixel_size
            assert abs((zoom["y"][0] + zoom["y"][1]) / 2 - cy) < bp.pixel_size
            assert zoom["x"][1] > zoom["x"][0]
            assert zoom["y"][1] > zoom["y"][0]
        finally:
            da._zoom_range = original

    def test_reset_zoom_clears_zoom_and_returns_patch(self):
        from dash import Patch

        import pybeamprofiler.dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        bp.last_img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        fn = _extract_callback_by_input(bp, "btn-zoom-reset")
        assert fn is not None
        original = da._zoom_range
        da._zoom_range = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
        try:
            patch = fn(1)
            assert isinstance(patch, Patch)
            assert da._zoom_range is None
        finally:
            da._zoom_range = original

    def test_reset_zoom_without_image_does_not_crash(self):
        """Reset clicked before any frame arrives must not crash; the
        patch is empty and the next tick will set the proper range."""
        from dash import Patch

        bp = BeamProfiler(camera="simulated")
        bp.last_img = None
        fn = _extract_callback_by_input(bp, "btn-zoom-reset")
        assert fn is not None
        assert isinstance(fn(1), Patch)

    def test_reset_zoom_no_clicks_returns_no_update(self):
        bp = BeamProfiler(camera="simulated")
        fn = _extract_callback_by_input(bp, "btn-zoom-reset")
        assert fn is not None
        assert isinstance(fn(0), dash._no_update.NoUpdate)

    def test_update_live_reads_module_zoom(self):
        """``update_live`` must read ``_zoom_range`` (the live source of
        truth) so a click that fires *after* a tick starts is still
        applied at figure-build time — no one-frame blink."""
        from unittest.mock import patch

        import pybeamprofiler.dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = _extract_callback(bp, "live-graph")
        assert fn is not None
        zoom = {"x": [100.0, 500.0], "y": [50.0, 450.0]}
        original = da._zoom_range
        da._zoom_range = zoom
        try:
            with patch("pybeamprofiler.dash_app.build_figure") as mock_bf:
                mock_bf.return_value = "FIG"
                fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
                kwargs = mock_bf.call_args.kwargs
            assert kwargs["xrange"] == [100.0, 500.0]
            assert kwargs["yrange"] == [50.0, 450.0]
        finally:
            da._zoom_range = original
            bp.camera.stop_acquisition()

    def test_update_live_no_zoom_passes_none(self):
        from unittest.mock import patch

        import pybeamprofiler.dash_app as da

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        fn = _extract_callback(bp, "live-graph")
        assert fn is not None
        original = da._zoom_range
        da._zoom_range = None
        try:
            with patch("pybeamprofiler.dash_app.build_figure") as mock_bf:
                mock_bf.return_value = "FIG"
                fn(1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1)
                kwargs = mock_bf.call_args.kwargs
            assert kwargs["xrange"] is None
            assert kwargs["yrange"] is None
        finally:
            da._zoom_range = original
            bp.camera.stop_acquisition()


class TestBuildFigureZoomRange:
    """``build_figure`` defaults to the full sensor extent and accepts an
    explicit override that wins over the default."""

    def test_default_range_covers_sensor(self):
        from pybeamprofiler.dash_app import build_figure

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        fig = build_figure(bp, img, None, None)
        assert list(fig.layout.xaxis.range) == [0, img.shape[1] * bp.pixel_size]
        assert list(fig.layout.yaxis.range) == [0, img.shape[0] * bp.pixel_size]

    def test_explicit_range_overrides_default(self):
        from pybeamprofiler.dash_app import build_figure

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        fig = build_figure(bp, img, None, None, xrange=[10.0, 200.0], yrange=[20.0, 300.0])
        assert list(fig.layout.xaxis.range) == [10.0, 200.0]
        assert list(fig.layout.yaxis.range) == [20.0, 300.0]
