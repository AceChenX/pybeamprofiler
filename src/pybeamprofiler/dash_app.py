"""Dash web application for pyBeamprofiler GUI.

Provides a rich browser-based interface with live camera streaming,
interactive fitting controls, and camera settings management.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from PIL import Image

from .constants import (
    DEFAULT_UPDATE_INTERVAL_MS,
    MAX_DISPLAY_DIM,
)

if TYPE_CHECKING:
    from .beamprofiler import BeamProfiler

logger = logging.getLogger(__name__)

COLORSCALES = list(
    dict.fromkeys(
        [
            "Hot",
            "Inferno",
            "Plasma",
            "Viridis",
            "Magma",
            "Cividis",
            "Turbo",
            "Jet",
            "Rainbow",
            "Portland",
            "Picnic",
            "Electric",
            "Bluered",
            "RdBu",
            "YlOrRd",
            "YlGnBu",
            "Sunset",
            "Sunsetdark",
            "Temps",
            "Thermal",
            "Oryel",
            "Agsunset",
            "Tealrose",
            "Blackbody",
            "Earth",
            "Dense",
            "Deep",
            "Speed",
            "Amp",
            "Matter",
            "Tropic",
            "Cividis",
            "Thermal",
            "YlOrRd",
        ]
    )
)
GRAY_COLORSCALE = "gray"

_PROFILE_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _fitting_tab(bp: BeamProfiler) -> dbc.Tab:
    """Build the **Fitting** tab content."""
    return dbc.Tab(
        label="Fitting",
        tab_id="tab-fitting",
        children=dbc.Card(
            dbc.CardBody(
                [
                    # Row 1 — Play / Pause + Save
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-pause-fill me-1"), "Pause"],
                                    id="btn-play-pause",
                                    color="primary",
                                    size="sm",
                                    className="w-100",
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-download me-1"), "Save"],
                                    id="btn-save",
                                    color="secondary",
                                    size="sm",
                                    className="w-100",
                                ),
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    # Row 2 — Color Scale switch + dropdown
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Color", className="small mb-0"),
                                    dbc.Switch(
                                        id="switch-color",
                                        value=True,
                                        label="",
                                        className="mt-1",
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Select(
                                    id="dropdown-colorscale",
                                    options=[{"label": s, "value": s} for s in COLORSCALES],
                                    value="Hot",
                                    size="sm",
                                ),
                                width=8,
                            ),
                        ],
                        className="mb-3 align-items-center",
                    ),
                    # Row 3 — Auto Range switch + Min / Max inputs
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Auto Range", className="small mb-0"),
                                    dbc.Switch(
                                        id="switch-autorange",
                                        value=True,
                                        label="",
                                        className="mt-1",
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Min", style={"fontSize": "0.8rem"}),
                                        dbc.Input(
                                            id="input-zmin",
                                            type="number",
                                            value=0,
                                            size="sm",
                                            disabled=True,
                                        ),
                                        dbc.InputGroupText("Max", style={"fontSize": "0.8rem"}),
                                        dbc.Input(
                                            id="input-zmax",
                                            type="number",
                                            value=255,
                                            size="sm",
                                            disabled=True,
                                        ),
                                    ],
                                    size="sm",
                                ),
                                width=8,
                            ),
                        ],
                        className="mb-3 align-items-center",
                    ),
                    # Row 4 — Dark / Light theme
                    dbc.Row(
                        [
                            dbc.Col(
                                html.I(className="bi bi-sun-fill text-muted"),
                                width="auto",
                                className="pe-1",
                            ),
                            dbc.Col(
                                dbc.Switch(
                                    id="switch-theme",
                                    value=True,
                                    label="",
                                    className="mb-0",
                                ),
                                width="auto",
                                className="px-0",
                            ),
                            dbc.Col(
                                html.I(className="bi bi-moon-stars-fill text-muted"),
                                width="auto",
                                className="ps-0",
                            ),
                        ],
                        className="mb-3 align-items-center",
                    ),
                    html.Hr(className="my-1"),
                    # Row 5 — Analysis + Fit function
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Analysis", className="small mb-1"),
                                    dbc.Select(
                                        id="dropdown-analysis",
                                        options=[
                                            {"label": "1D Integration", "value": "1d"},
                                            {"label": "2D Gaussian", "value": "2d"},
                                            {"label": "Linecut", "value": "linecut"},
                                        ],
                                        value=bp.fit_method,
                                        size="sm",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Definition", className="small mb-1"),
                                    dbc.Select(
                                        id="dropdown-definition",
                                        options=[
                                            {"label": "Gaussian (1/e²)", "value": "gaussian"},
                                            {"label": "FWHM", "value": "fwhm"},
                                            {"label": "D4σ", "value": "d4s"},
                                        ],
                                        value=bp.definition,
                                        size="sm",
                                    ),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    # Row 5 — Pixel scale
                    dbc.Row(
                        dbc.Col(
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Scale", style={"fontSize": "0.8rem"}),
                                    dbc.Input(
                                        id="input-pixel-scale",
                                        type="number",
                                        value=round(bp.pixel_size, 4),
                                        step=0.01,
                                        size="sm",
                                    ),
                                    dbc.InputGroupText("μm/px", style={"fontSize": "0.8rem"}),
                                ],
                                size="sm",
                            ),
                        ),
                        className="mb-3",
                    ),
                    # Row 6 — Fitted results
                    html.Hr(className="my-2"),
                    html.Div(id="div-results", className="small font-monospace"),
                ],
            ),
            className="border-0",
        ),
    )


def _build_setting_items(bp: BeamProfiler) -> list[dbc.AccordionItem]:
    """Dynamically create accordion items from camera properties."""
    items: list[dbc.AccordionItem] = []
    cam = bp.camera
    if cam is None:
        return items

    # ── Camera Info ──────────────────────────────────────────────
    info_rows = []
    for label, attr in [
        ("Model", "device_model"),
        ("Vendor", "device_vendor"),
        ("Serial", "serial_number"),
    ]:
        val = getattr(cam, attr, None)
        if val is None and hasattr(cam, "node_map") and cam.node_map is not None:
            nm = cam.node_map
            node_attr = {
                "Model": "DeviceModelName",
                "Vendor": "DeviceVendorName",
                "Serial": "DeviceSerialNumber",
            }.get(label)
            if node_attr and hasattr(nm, node_attr):
                try:
                    val = getattr(nm, node_attr).value
                except Exception:
                    pass
        if val:
            info_rows.append(
                html.Tr([html.Td(label, className="pe-3 text-muted"), html.Td(str(val))])
            )

    sensor_w = getattr(cam, "width_pixels", None) or getattr(cam, "width", None)
    sensor_h = getattr(cam, "height_pixels", None) or getattr(cam, "height", None)
    if sensor_w and sensor_h:
        info_rows.append(
            html.Tr(
                [
                    html.Td("Sensor", className="pe-3 text-muted"),
                    html.Td(f"{sensor_w} × {sensor_h}"),
                ]
            )
        )
    ps = getattr(cam, "pixel_size", None)
    if ps:
        info_rows.append(
            html.Tr([html.Td("Pixel size", className="pe-3 text-muted"), html.Td(f"{ps} μm")])
        )

    items.append(
        dbc.AccordionItem(
            html.Table(html.Tbody(info_rows), className="table table-sm table-borderless mb-0"),
            title="Camera Info",
        )
    )

    # ── Exposure & Gain ─────────────────────────────────────────
    exp_min, exp_max = 0.001, 1.0
    er = getattr(cam, "exposure_range", None)
    if er is not None:
        try:
            exp_min, exp_max = er
        except Exception:
            pass
    gain_min, gain_max = 0.0, 24.0
    gr = getattr(cam, "gain_range", None)
    if gr is not None:
        try:
            gain_min, gain_max = gr
        except Exception:
            pass

    exp_ms = (cam.exposure_time or 0.01) * 1000

    items.append(
        dbc.AccordionItem(
            [
                dbc.Label("Exposure (ms)", className="small mb-1"),
                dcc.Slider(
                    id="slider-exposure",
                    min=round(exp_min * 1000, 3),
                    max=round(exp_max * 1000, 3),
                    value=round(exp_ms, 3),
                    step=0.001,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.Div(className="mb-3"),
                dbc.Label("Gain", className="small mb-1"),
                dcc.Slider(
                    id="slider-gain",
                    min=gain_min,
                    max=gain_max,
                    value=cam.gain or 0.0,
                    step=0.1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
            ],
            title="Exposure & Gain",
        )
    )

    # ── ROI ─────────────────────────────────────────────────────
    roi: dict[str, int] | None = None
    if hasattr(cam, "roi_info") and hasattr(cam, "set_roi"):
        try:
            roi = getattr(cam, "roi_info")
        except Exception:
            pass
        if roi is not None:
            items.append(
                dbc.AccordionItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Offset X", className="small"),
                                        dbc.Input(
                                            id="input-roi-ox",
                                            type="number",
                                            value=roi.get("offset_x", 0),
                                            size="sm",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Offset Y", className="small"),
                                        dbc.Input(
                                            id="input-roi-oy",
                                            type="number",
                                            value=roi.get("offset_y", 0),
                                            size="sm",
                                        ),
                                    ],
                                    width=6,
                                ),
                            ],
                            className="mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Width", className="small"),
                                        dbc.Input(
                                            id="input-roi-w",
                                            type="number",
                                            value=roi.get("width", 1024),
                                            size="sm",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Height", className="small"),
                                        dbc.Input(
                                            id="input-roi-h",
                                            type="number",
                                            value=roi.get("height", 1024),
                                            size="sm",
                                        ),
                                    ],
                                    width=6,
                                ),
                            ],
                            className="mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Apply ROI",
                                        id="btn-roi-apply",
                                        color="primary",
                                        size="sm",
                                        className="w-100",
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Full Sensor",
                                        id="btn-roi-reset",
                                        color="outline-secondary",
                                        size="sm",
                                        className="w-100",
                                    ),
                                    width=6,
                                ),
                            ],
                        ),
                        html.Div(id="div-roi-status", className="small text-muted mt-2"),
                    ],
                    title="Region of Interest",
                )
            )

    return items


def _setting_tab(bp: BeamProfiler) -> dbc.Tab:
    """Build the **Setting** tab content."""
    items = _build_setting_items(bp)
    if not items:
        body = html.P("No camera connected.", className="text-muted p-3")
    else:
        body = dbc.Accordion(items, start_collapsed=False, always_open=True)

    return dbc.Tab(
        label="Setting",
        tab_id="tab-setting",
        children=dbc.Card(dbc.CardBody(body), className="border-0"),
    )


# ---------------------------------------------------------------------------
# Figure builder — LaseView-style overlay
# ---------------------------------------------------------------------------


def _normalize_profile(
    data: np.ndarray, span: float, fraction: float = _PROFILE_FRACTION
) -> np.ndarray:
    """Scale a 1-D profile to occupy *fraction* of *span*."""
    lo, hi = float(np.min(data)), float(np.max(data))
    rng = hi - lo if hi != lo else 1.0
    return (data - lo) / rng * span * fraction


def build_figure(
    bp: BeamProfiler,
    image: np.ndarray,
    popt_x: np.ndarray | list[Any] | None,
    popt_y: np.ndarray | list[Any] | None,
    *,
    colorscale: str = "Hot",
    zmin: float | None = None,
    zmax: float | None = None,
    dark_theme: bool = True,
) -> go.Figure:
    """Build a single-plot figure with profiles overlaid on the heatmap.

    The X projection is drawn along the bottom edge and the Y projection
    along the left edge, similar to LaseView.
    """
    if image is None:
        return go.Figure()

    from scipy.ndimage import zoom as _ndimage_zoom

    h, w = image.shape
    if max(h, w) > MAX_DISPLAY_DIM:
        scale = MAX_DISPLAY_DIM / max(h, w)
        display_img = _ndimage_zoom(image, scale, order=1)
    else:
        display_img = image
    dh, dw = display_img.shape

    ps = bp.pixel_size
    x_coords = np.linspace(0, (w - 1) * ps, dw)
    y_coords = np.linspace(0, (h - 1) * ps, dh)
    x_max = w * ps
    y_max = h * ps

    fig = go.Figure()

    # ── Heatmap ─────────────────────────────────────────────────
    heat_kwargs: dict[str, Any] = {
        "z": display_img,
        "x": x_coords,
        "y": y_coords,
        "colorscale": colorscale,
        "showscale": True,
        "colorbar": dict(thickness=12, len=0.7),
    }
    if zmin is not None:
        heat_kwargs["zmin"] = zmin
    if zmax is not None:
        heat_kwargs["zmax"] = zmax
    fig.add_trace(go.Heatmap(**heat_kwargs))

    # ── Linecut crosshairs ──────────────────────────────────────
    if bp.fit_method == "linecut" and hasattr(bp, "_linecut_x") and hasattr(bp, "_linecut_y"):
        lx, ly = bp._linecut_x * ps, bp._linecut_y * ps
        for xs, ys in [([lx, lx], [0, y_max]), ([0, x_max], [ly, ly])]:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="cyan", width=1.5, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # ── Ellipse overlay ─────────────────────────────────────────
    if popt_x is not None and popt_y is not None:
        cx, cy = popt_x[1], popt_y[1]
        rx, ry = 2 * abs(popt_x[2]), 2 * abs(popt_y[2])
        theta = np.linspace(0, 2 * np.pi, 100)

        if bp.fit_method == "2d" and hasattr(bp, "angle_deg"):
            a = np.radians(bp.angle_deg)
            xe = cx + rx * np.cos(theta) * np.cos(a) - ry * np.sin(theta) * np.sin(a)
            ye = cy + rx * np.cos(theta) * np.sin(a) + ry * np.sin(theta) * np.cos(a)
        else:
            xe = cx + rx * np.cos(theta)
            ye = cy + ry * np.sin(theta)

        fig.add_trace(
            go.Scatter(
                x=xe * ps,
                y=ye * ps,
                mode="lines",
                line=dict(color="#FF4444", width=2.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # ── Theme-dependent colors ─────────────────────────────────
    if dark_theme:
        prof_line = "rgba(255,255,255,0.5)"
        fill_x = "rgba(100,160,255,0.15)"
        fill_y = "rgba(100,220,100,0.15)"
        bg_plot = "rgba(0,0,0,1)"
        bg_paper = "rgba(0,0,0,0)"
        fg = "#ccc"
    else:
        prof_line = "rgba(0,0,0,0.45)"
        fill_x = "rgba(50,100,200,0.12)"
        fill_y = "rgba(50,160,50,0.12)"
        bg_plot = "#f8f8f8"
        bg_paper = "#ffffff"
        fg = "#333"

    # ── X profile (bottom edge) ─────────────────────────────────
    x_ax = np.arange(w)
    proj_x = np.sum(image, axis=0).astype(float)
    norm_x = _normalize_profile(proj_x, y_max)

    fig.add_trace(
        go.Scatter(
            x=x_ax * ps,
            y=norm_x,
            mode="lines",
            line=dict(color=prof_line, width=1.5),
            fill="tozeroy",
            fillcolor=fill_x,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if popt_x is not None:
        from .beamprofiler import BeamProfiler as _BP

        fit_x = _BP.gaussian(x_ax, *popt_x).astype(float)
        norm_fit_x = _normalize_profile(fit_x, y_max)
        fig.add_trace(
            go.Scatter(
                x=x_ax * ps,
                y=norm_fit_x,
                mode="lines",
                line=dict(color="#FF4444", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # ── Y profile (left edge) ──────────────────────────────────
    proj_y = np.sum(image, axis=1).astype(float)
    y_ax = np.arange(h)
    norm_y = _normalize_profile(proj_y, x_max)

    fig.add_trace(
        go.Scatter(
            x=norm_y,
            y=y_ax * ps,
            mode="lines",
            line=dict(color=prof_line, width=1.5),
            fill="tozerox",
            fillcolor=fill_y,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if popt_y is not None:
        from .beamprofiler import BeamProfiler as _BP

        fit_y = _BP.gaussian(y_ax, *popt_y).astype(float)
        norm_fit_y = _normalize_profile(fit_y, x_max)
        fig.add_trace(
            go.Scatter(
                x=norm_fit_y,
                y=y_ax * ps,
                mode="lines",
                line=dict(color="#FF4444", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        uirevision="constant",
        autosize=True,
        showlegend=False,
        margin=dict(l=30, r=5, t=5, b=30),
        plot_bgcolor=bg_plot,
        paper_bgcolor=bg_paper,
        font_color=fg,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            range=[0, y_max],
            showgrid=False,
            title="Y (μm)",
            title_font_size=11,
        ),
        xaxis=dict(
            constrain="domain",
            range=[0, x_max],
            showgrid=False,
            title="X (μm)",
            title_font_size=11,
        ),
    )

    return fig


def _format_results(bp: BeamProfiler) -> list[Any]:
    """Format fitted beam parameters for the results panel."""
    rows: list[Any] = []

    def _row(label: str, val: float) -> Any:
        return html.Div(
            [html.Span(f"{label}: ", className="text-muted"), html.Span(f"{val:.1f} μm")],
            className="mb-1",
        )

    if bp.width_x > 0:
        rows.append(_row("FW@1/e² X", bp.fw_1e2_x))
        rows.append(_row("FW@1/e² Y", bp.fw_1e2_y))
        rows.append(_row("FW@1/e X", bp.fw_1e_x))
        rows.append(_row("FW@1/e Y", bp.fw_1e_y))
        rows.append(_row("FWHM X", bp.fwhm_x))
        rows.append(_row("FWHM Y", bp.fwhm_y))

        rows.append(html.Hr(className="my-1"))
        ps = bp.pixel_size
        rows.append(
            html.Div(
                [
                    html.Span("Center: ", className="text-muted"),
                    html.Span(f"({bp.center_x * ps:.1f}, {bp.center_y * ps:.1f}) μm"),
                ],
                className="mb-1",
            )
        )
        rows.append(
            html.Div(
                [html.Span("Peak: ", className="text-muted"), html.Span(f"{bp.peak_value:.0f}")],
                className="mb-1",
            )
        )
        if bp.fit_method == "2d":
            rows.append(
                html.Div(
                    [
                        html.Span("Angle: ", className="text-muted"),
                        html.Span(f"{bp.angle_deg:.1f}°"),
                    ],
                    className="mb-1",
                )
            )
    else:
        rows.append(html.Span("No fit data", className="text-muted"))

    return rows


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(bp: BeamProfiler) -> dash.Dash:
    """Create and configure the Dash application.

    Args:
        bp: Fully initialised :class:`BeamProfiler` instance.

    Returns:
        A ``dash.Dash`` application ready to ``.run()``.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.BOOTSTRAP,
        ],
        title="pyBeamprofiler",
    )

    app.index_string = """<!DOCTYPE html>
<html data-bs-theme="dark"><head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<script>
window.addEventListener('load', function() {
    var fails = 0;
    setInterval(function() {
        fetch('/_dash-component-suites/dash/dcc/async-graph.js',
              {method:'HEAD', cache:'no-cache'})
        .then(function() { fails = 0; })
        .catch(function() { fails++; if (fails >= 2) window.open('','_self').close(); });
    }, 500);
});
</script>
</head><body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>"""

    # ── Initial figure ──────────────────────────────────────────
    if bp._mode == "camera" and bp.camera is not None and not bp.camera.is_acquiring:
        bp.camera.start_acquisition()

    initial_img = (
        bp.camera.get_image() if bp._mode == "camera" and bp.camera is not None else bp.last_img
    )
    if initial_img is not None:
        popt_x, popt_y = bp.analyze(initial_img)
        initial_fig = build_figure(bp, initial_img, popt_x, popt_y)
    else:
        initial_fig = go.Figure()

    # ── Layout ──────────────────────────────────────────────────
    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id="live-graph",
                            figure=initial_fig,
                            style={"height": "100vh"},
                            config={"responsive": True, "displaylogo": False},
                        ),
                        width=9,
                        className="pe-0",
                    ),
                    dbc.Col(
                        [
                            html.H6(
                                "pyBeamprofiler",
                                className="text-center mb-2 mt-1 fw-bold",
                            ),
                            dbc.Tabs(
                                [
                                    _fitting_tab(bp),
                                    _setting_tab(bp),
                                ],
                                id="tabs",
                                active_tab="tab-fitting",
                            ),
                            html.Div(
                                id="status-bar",
                                className="small text-muted text-center mt-2",
                            ),
                        ],
                        width=3,
                        className="ps-1",
                        style={"height": "100vh", "overflowY": "auto"},
                    ),
                ],
                className="g-0",
            ),
            dcc.Interval(id="interval", interval=DEFAULT_UPDATE_INTERVAL_MS, n_intervals=0),
            dcc.Store(id="store-paused", data=False),
            dcc.Store(id="store-frame", data=0),
            dcc.Store(id="store-dark-theme", data=True),
            dcc.Download(id="download-png"),
        ],
        fluid=True,
        className="p-0",
        id="main-container",
        style={"backgroundColor": "#222"},
    )

    _register_callbacks(app, bp)
    return app


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

_callback_lock = threading.Lock()


def _register_callbacks(app: dash.Dash, bp: BeamProfiler) -> None:
    """Wire up all Dash callbacks."""

    # -- Play / Pause toggle --------------------------------------------------
    @app.callback(
        Output("store-paused", "data"),
        Output("btn-play-pause", "children"),
        Output("btn-play-pause", "color"),
        Input("btn-play-pause", "n_clicks"),
        State("store-paused", "data"),
        prevent_initial_call=True,
    )
    def toggle_pause(n: int, paused: bool) -> tuple[bool, list[Any], str]:
        new_paused = not paused
        if new_paused:
            label = [html.I(className="bi bi-play-fill me-1"), "Play"]
            color = "success"
        else:
            label = [html.I(className="bi bi-pause-fill me-1"), "Pause"]
            color = "primary"
        return new_paused, label, color

    # -- Save current frame ---------------------------------------------------
    @app.callback(
        Output("download-png", "data"),
        Input("btn-save", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_frame(_n: int) -> dict[str, Any] | None:
        img = bp.last_img
        if img is None:
            return None
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        ts = time.strftime("%Y%m%d_%H%M%S")
        return {
            "content": b64,
            "filename": f"beam_{ts}.png",
            "base64": True,
        }

    # -- Auto-range toggle disables min/max inputs ----------------------------
    @app.callback(
        Output("input-zmin", "disabled"),
        Output("input-zmax", "disabled"),
        Input("switch-autorange", "value"),
    )
    def toggle_autorange(auto: bool) -> tuple[bool, bool]:
        return auto, auto

    # -- Dark / Light theme toggle --------------------------------------------
    app.clientside_callback(
        """function(isDark) {
            document.documentElement.setAttribute(
                'data-bs-theme', isDark ? 'dark' : 'light');
            var bg = isDark ? '#222' : '#f0f0f0';
            return [isDark, {'backgroundColor': bg}];
        }""",
        Output("store-dark-theme", "data"),
        Output("main-container", "style"),
        Input("switch-theme", "value"),
    )

    # -- Pixel scale override -------------------------------------------------
    @app.callback(
        Output("input-pixel-scale", "value"),
        Input("input-pixel-scale", "value"),
        prevent_initial_call=True,
    )
    def set_pixel_scale(val: float | None) -> float:
        if val is not None and val > 0:
            bp.pixel_size = val
        return round(bp.pixel_size, 4)

    # -- Exposure slider ------------------------------------------------------
    @app.callback(
        Output("slider-exposure", "value"),
        Input("slider-exposure", "value"),
        prevent_initial_call=True,
    )
    def set_exposure(val: float) -> float:
        if bp.camera is not None:
            try:
                was_acquiring = bp.camera.is_acquiring
                bp.camera.set_exposure(val / 1000.0)
                if was_acquiring and not bp.camera.is_acquiring:
                    bp.camera.start_acquisition()
            except Exception as e:
                logger.warning(f"Failed to set exposure: {e}")
        return val

    # -- Gain slider ----------------------------------------------------------
    @app.callback(
        Output("slider-gain", "value"),
        Input("slider-gain", "value"),
        prevent_initial_call=True,
    )
    def set_gain(val: float) -> float:
        if bp.camera is not None:
            try:
                was_acquiring = bp.camera.is_acquiring
                bp.camera.set_gain(val)
                if was_acquiring and not bp.camera.is_acquiring:
                    bp.camera.start_acquisition()
            except Exception as e:
                logger.warning(f"Failed to set gain: {e}")
        return val

    # -- ROI apply (conditional) ----------------------------------------------
    has_roi = bp.camera is not None and hasattr(bp.camera, "set_roi")
    if has_roi:

        @app.callback(
            Output("div-roi-status", "children"),
            Input("btn-roi-apply", "n_clicks"),
            State("input-roi-ox", "value"),
            State("input-roi-oy", "value"),
            State("input-roi-w", "value"),
            State("input-roi-h", "value"),
            prevent_initial_call=True,
        )
        def apply_roi(_n: int, ox: int, oy: int, w: int, h: int) -> str:
            if bp.camera is None:
                return "No camera"
            try:
                was_acquiring = bp.camera.is_acquiring
                if was_acquiring:
                    bp.camera.stop_acquisition()
                getattr(bp.camera, "set_roi")(
                    offset_x=int(ox), offset_y=int(oy), width=int(w), height=int(h)
                )
                if was_acquiring:
                    bp.camera.start_acquisition()
                roi = getattr(bp.camera, "roi_info")
                return (
                    f"ROI: {roi['width']}×{roi['height']} at ({roi['offset_x']},{roi['offset_y']})"
                )
            except Exception as e:
                logger.warning(f"Failed to set ROI: {e}")
                return f"Error: {e}"

        @app.callback(
            Output("input-roi-ox", "value"),
            Output("input-roi-oy", "value"),
            Output("input-roi-w", "value"),
            Output("input-roi-h", "value"),
            Output("div-roi-status", "children", allow_duplicate=True),
            Input("btn-roi-reset", "n_clicks"),
            prevent_initial_call=True,
        )
        def reset_roi(_n: int) -> tuple[int, int, int, int, str]:
            if bp.camera is None:
                return 0, 0, 0, 0, "No camera"
            try:
                was_acquiring = bp.camera.is_acquiring
                if was_acquiring:
                    bp.camera.stop_acquisition()
                getattr(bp.camera, "set_roi")(offset_x=0, offset_y=0, width=None, height=None)
                if was_acquiring:
                    bp.camera.start_acquisition()
                roi = getattr(bp.camera, "roi_info")
                return 0, 0, roi["max_width"], roi["max_height"], "Reset to full sensor"
            except Exception as e:
                logger.warning(f"Failed to reset ROI: {e}")
                return 0, 0, 0, 0, f"Error: {e}"

    # -- Main update loop -----------------------------------------------------
    @app.callback(
        Output("live-graph", "figure"),
        Output("div-results", "children"),
        Output("status-bar", "children"),
        Output("store-frame", "data"),
        Input("interval", "n_intervals"),
        State("store-paused", "data"),
        State("switch-color", "value"),
        State("dropdown-colorscale", "value"),
        State("switch-autorange", "value"),
        State("input-zmin", "value"),
        State("input-zmax", "value"),
        State("store-frame", "data"),
        State("dropdown-analysis", "value"),
        State("dropdown-definition", "value"),
        State("store-dark-theme", "data"),
    )
    async def update_live(
        _n: int,
        paused: bool,
        color_on: bool,
        cs_name: str,
        auto_range: bool,
        zmin_val: float | None,
        zmax_val: float | None,
        frame_count: int,
        analysis: str,
        definition: str,
        dark_theme: bool,
    ) -> tuple[Any, ...]:
        if paused:
            return (dash.no_update,) * 4

        if not _callback_lock.acquire(blocking=False):
            return (dash.no_update,) * 4

        try:
            # Sync settings from GUI state
            if analysis and bp.fit_method != analysis:
                bp.fit_method = analysis
                bp._last_popt_x = None
                bp._last_popt_y = None
                bp._last_popt_2d = None
            if definition and bp.definition != definition:
                bp.definition = definition

            if bp._mode == "camera" and bp.camera is not None:
                img = await asyncio.to_thread(bp.camera.get_image)
            else:
                img = bp.last_img

            if img is None:
                return (dash.no_update,) * 4

            bp.last_img = img

            popt_x, popt_y = await asyncio.to_thread(bp.analyze, img)

            cs = cs_name if color_on else GRAY_COLORSCALE
            zmin = None if auto_range else (zmin_val if zmin_val is not None else 0)
            zmax = None if auto_range else (zmax_val if zmax_val is not None else 255)
            fig = await asyncio.to_thread(
                build_figure,
                bp,
                img,
                popt_x,
                popt_y,
                colorscale=cs,
                zmin=zmin,
                zmax=zmax,
                dark_theme=dark_theme,
            )

            frame_count += 1
            results = _format_results(bp)
            status = f"Frame #{frame_count}"

            return fig, results, status, frame_count
        except Exception as e:
            logger.debug(f"Update error: {e}")
            return (dash.no_update,) * 4
        finally:
            _callback_lock.release()
