"""Tests for the Dash GUI module and refactored SimulatedCamera."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
import pytest
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
        fig = build_figure(bp, None, None, None)  # ty: ignore[invalid-argument-type]
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

    def test_roi_section_present(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        items = _build_setting_items(bp)
        titles = [item.title for item in items]  # ty: ignore[unresolved-attribute]
        assert "Region of Interest" in titles

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
