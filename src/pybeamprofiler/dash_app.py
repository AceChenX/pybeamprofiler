"""Dash web application for pyBeamprofiler GUI.

Provides a rich browser-based interface with live camera streaming,
interactive fitting controls, and camera settings management.
"""

from __future__ import annotations

import base64
import collections
import io
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import MATCH, Input, Output, Patch, State, ctx, dcc, html
from PIL import Image

from .constants import (
    DEFAULT_UPDATE_INTERVAL_MS,
    MAX_AVG_FRAMES,
    MAX_DISPLAY_DIM,
)
from .dash_layout import (
    GRAY_COLORSCALE,
    _build_setting_items,
    _camera_options,
    _fitting_tab,
    _format_results,
    _setting_tab,
)
from .discovery import (
    CameraOption,
    describe_open_camera,
    find_option,
    open_camera,
)
from .fitting import downsample

# Optional: only present when a real GenTL backend is installed. Used by
# ``_is_readonly`` to map GenICam access modes to our read-only flag.
try:
    from genicam.genapi import EAccessMode as _EAccessMode  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - exercised only on non-GenICam envs
    _EAccessMode = None  # ty: ignore[invalid-assignment]

if TYPE_CHECKING:
    from .beamprofiler import BeamProfiler

logger = logging.getLogger(__name__)


_PROFILE_FRACTION = 0.15

# Saturation warning thresholds.
_SATURATION_PIXEL_FRACTION = 0.001  # 0.1% of pixels at the dtype max ⇒ warn.


def _saturation_max(image: np.ndarray) -> float:
    """Return the saturation level for *image* given its dtype.

    For integer images this is the dtype's largest representable value
    (e.g. 255 for ``uint8``, 65535 for ``uint16``, 4095 for a 12-bit
    sensor packed in ``uint16`` is reported as 65535 — we cannot tell
    "real" bit depth from the dtype alone, so the warning is conservative).
    For floating-point images we assume normalised ``[0, 1]`` data when the
    observed max is ≤ 1, otherwise use the observed max as a heuristic.
    """
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    obs_max = float(image.max()) if image.size else 1.0
    return 1.0 if obs_max <= 1.0 else obs_max


def _saturation_fraction(image: np.ndarray) -> float:
    """Fraction of pixels at or above the dtype's saturation level."""
    if image.size == 0:
        return 0.0
    sat = _saturation_max(image)
    # For integer images, only the exact dtype max counts as saturated.
    # For floating-point images, allow a small epsilon near the inferred max.
    threshold = sat if np.issubdtype(image.dtype, np.integer) else sat - 1e-6
    # A plain max() reduction is several times cheaper than the comparison
    # below, which has to allocate a full-frame boolean temporary. Nothing is
    # saturated far more often than not, so check the cheap way first.
    if float(image.max()) < threshold:
        return 0.0
    return float(np.count_nonzero(image >= threshold)) / image.size


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
    image: np.ndarray | None,
    popt_x: np.ndarray | list[Any] | None,
    popt_y: np.ndarray | list[Any] | None,
    *,
    colorscale: str = "Hot",
    zmin: float | None = None,
    zmax: float | None = None,
    dark_theme: bool = True,
    xrange: list[float] | None = None,
    yrange: list[float] | None = None,
) -> go.Figure:
    """Build a single-plot figure with profiles overlaid on the heatmap.

    The X projection is drawn along the bottom edge and the Y projection
    along the left edge, similar to LaseView.  Returns an empty figure
    when *image* is ``None``.

    Args:
        bp: BeamProfiler instance (used for pixel size and cached projections).
        image: 2-D intensity array, or ``None`` for an empty figure.
        popt_x: X-projection Gaussian fit parameters, or ``None``.
        popt_y: Y-projection Gaussian fit parameters, or ``None``.
        colorscale: Plotly colorscale name.
        zmin: Fixed minimum for the color range (``None`` for auto).
        zmax: Fixed maximum for the color range (``None`` for auto).
        dark_theme: Use dark background when ``True``.

    Returns:
        Plotly ``Figure`` with heatmap, profile overlays, and optional
        fit curves / beam ellipse.
    """
    if image is None:
        return go.Figure()

    h, w = image.shape
    display_img = downsample(image, MAX_DISPLAY_DIM)
    dh, dw = display_img.shape

    ps = bp.pixel_size
    x_coords = np.linspace(0, (w - 1) * ps, dw)
    y_coords = np.linspace(0, (h - 1) * ps, dh)
    x_max = w * ps
    y_max = h * ps

    traces: list[Any] = []

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
    traces.append(go.Heatmap(**heat_kwargs))

    # ── Linecut crosshairs ──────────────────────────────────────
    if bp.fit_method == "linecut" and hasattr(bp, "_linecut_x") and hasattr(bp, "_linecut_y"):
        lx, ly = bp._linecut_x * ps, bp._linecut_y * ps
        for xs, ys in [([lx, lx], [0, y_max]), ([0, x_max], [ly, ly])]:
            traces.append(
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
    ellipse = bp._ellipse_points()
    if ellipse is not None:
        traces.append(
            go.Scatter(
                x=ellipse[0],
                y=ellipse[1],
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
    cached_proj_x = getattr(bp, "_last_proj_x", None)
    proj_x = (cached_proj_x if cached_proj_x is not None else np.sum(image, axis=0)).astype(float)
    norm_x = _normalize_profile(proj_x, y_max)

    traces.append(
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
        fit_x = bp.gaussian(x_ax, *popt_x).astype(float)
        norm_fit_x = _normalize_profile(fit_x, y_max)
        traces.append(
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
    y_ax = np.arange(h)
    cached_proj_y = getattr(bp, "_last_proj_y", None)
    proj_y = (cached_proj_y if cached_proj_y is not None else np.sum(image, axis=1)).astype(float)
    norm_y = _normalize_profile(proj_y, x_max)

    traces.append(
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
        fit_y = bp.gaussian(y_ax, *popt_y).astype(float)
        norm_fit_y = _normalize_profile(fit_y, x_max)
        traces.append(
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
    # Built as a plain dict and handed to the Figure constructor rather than
    # applied with update_layout(). update_layout parses every nested key as
    # a magic-underscore path and re-validates the whole tree, which costs
    # more than everything else in this function combined; the constructor
    # produces byte-identical JSON for ~2.6x less work.
    layout = {
        "uirevision": "constant",
        "autosize": True,
        "showlegend": False,
        "margin": {"l": 30, "r": 5, "t": 5, "b": 30},
        "plot_bgcolor": bg_plot,
        "paper_bgcolor": bg_paper,
        "font_color": fg,
        "yaxis": {
            "scaleanchor": "x",
            "scaleratio": 1,
            "range": yrange if yrange is not None else [0, y_max],
            "showgrid": False,
            "title": "Y (μm)",
            "title_font_size": 11,
        },
        "xaxis": {
            "constrain": "domain",
            "range": xrange if xrange is not None else [0, x_max],
            "showgrid": False,
            "title": "X (μm)",
            "title_font_size": 11,
        },
    }

    return go.Figure(data=traces, layout=layout)


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
        # Don't attach Dash's own StreamHandler; the parent CLI manages logging
        # (otherwise the "Dash is running on..." banner appears twice -- once
        # from Dash's handler and once propagated to the root logger).
        add_log_handler=False,
        # The Setting panel is rebuilt from scratch every time the camera
        # changes, so the ids its callbacks target are not all present in the
        # initial layout. Without this Dash refuses to register them.
        suppress_callback_exceptions=True,
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

    initial_img = None
    if bp._mode == "camera" and bp.camera is not None:
        first_frame_timeout = max(3.0, (bp.camera.exposure_time or 0) + 2.0)
        try:
            initial_img = bp.camera.get_image(timeout=first_frame_timeout)
        except Exception as e:
            logger.warning("Could not grab initial frame: %s", e)
    else:
        initial_img = bp.last_img

    if initial_img is not None:
        popt_x, popt_y = bp.analyze(initial_img)
        initial_fig = build_figure(bp, initial_img, popt_x, popt_y)
    else:
        initial_fig = go.Figure()

    # ── Layout ──────────────────────────────────────────────────
    # One enumeration at start-up, shared by the dropdown and the cache the
    # switch callback resolves against. Scanning twice would double a
    # multi-second GenTL walk on a machine with hardware attached.
    global _known_options  # noqa: PLW0603
    camera_options, current_camera = _camera_options(bp)
    _known_options = camera_options

    # The two-column split is implemented with explicit pixel widths so a
    # draggable divider (``#col-divider``) can resize them on the client.
    # Defaults match the original 75/25% Bootstrap row.
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id="live-graph",
                            figure=initial_fig,
                            style={"height": "100vh"},
                            config={"responsive": True, "displaylogo": False},
                        ),
                        id="col-graph",
                        style={"flex": "1 1 0", "minWidth": "200px", "overflow": "hidden"},
                    ),
                    html.Div(
                        id="col-divider",
                        title="Drag to resize",
                        style={
                            "width": "5px",
                            "cursor": "col-resize",
                            "backgroundColor": "#444",
                            "flex": "0 0 5px",
                        },
                    ),
                    html.Div(
                        [
                            html.H6(
                                "pyBeamprofiler",
                                className="text-center mb-2 mt-1 fw-bold",
                            ),
                            dbc.Tabs(
                                [
                                    _fitting_tab(bp, camera_options, current_camera),
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
                        id="col-side",
                        className="ps-1",
                        style={
                            "flex": "0 0 320px",
                            "minWidth": "240px",
                            "maxWidth": "60%",
                            "height": "100vh",
                            "overflowY": "auto",
                        },
                    ),
                ],
                style={"display": "flex", "width": "100%", "height": "100vh"},
            ),
            dcc.Interval(id="interval", interval=DEFAULT_UPDATE_INTERVAL_MS, n_intervals=0),
            dcc.Store(id="store-paused", data=False),
            dcc.Store(id="store-frame", data=0),
            dcc.Store(id="store-dark-theme", data=True),
            dcc.Download(id="download-png"),
            dcc.Download(id="download-npy"),
        ],
        id="main-container",
        style={"backgroundColor": "#222"},
    )

    _register_callbacks(app, bp)
    return app


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

_callback_lock = threading.Lock()
_server_paused = False

# Rolling FPS tracker for the status bar.
_recent_frame_times: collections.deque[float] = collections.deque(maxlen=20)

# Rolling buffer for N-frame averaging. Recreated when N or shape changes.
# Only the frames are retained here so we can subtract the frame that
# falls out of the window; the mean is kept incrementally in
# ``_avg_running_sum`` so each tick is O(H·W) rather than O(N·H·W).
_avg_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=1)
_avg_buffer_shape: tuple[int, ...] | None = None
_avg_running_sum: np.ndarray | None = None

# The camera list currently shown in the dropdown. Populated when the
# selector is built and refreshed by the rescan button, so switching cameras
# does not have to re-enumerate: a GenTL scan opens every producer and walks
# the network, which takes seconds on a GigE setup and would block the render
# loop for the whole switch.
_known_options: list[CameraOption] = []

# Authoritative zoom state, mutated by Auto-fit / Reset under
# ``_callback_lock`` and read by ``update_live``. Using a module
# variable (rather than a Dash ``State``) avoids a 50–100 ms
# stale-snapshot race that would otherwise blink the previous zoom
# for one frame whenever a click landed mid-tick.
_zoom_range: dict[str, list[float]] | None = None


def _measured_fps() -> float:
    """Compute frames-per-second from the rolling timestamp window."""
    if len(_recent_frame_times) < 2:
        return 0.0
    span = _recent_frame_times[-1] - _recent_frame_times[0]
    if span <= 0:
        return 0.0
    return (len(_recent_frame_times) - 1) / span


def _build_status(bp: BeamProfiler, img: np.ndarray, frame_count: int) -> Any:
    """Build the status bar contents (frame, fps, exposure, gain, saturation)."""
    pieces: list[Any] = [f"Frame #{frame_count}"]

    fps = _measured_fps()
    if fps:
        pieces.append(f"{fps:.1f} fps")

    cam = bp.camera
    if cam is not None:
        exp = getattr(cam, "exposure_time", None)
        if exp:
            pieces.append(f"Exp {exp * 1000:.2f} ms" if exp < 1.0 else f"Exp {exp:.2f} s")
        gain = getattr(cam, "gain", None)
        if gain is not None:
            pieces.append(f"Gain {gain:.1f}")

    children: list[Any] = []
    for i, p in enumerate(pieces):
        if i:
            children.append(html.Span(" · ", className="text-muted mx-1"))
        children.append(html.Span(p))

    sat = _saturation_fraction(img)
    if sat >= _SATURATION_PIXEL_FRACTION:
        children.append(html.Span(" · ", className="text-muted mx-1"))
        children.append(
            html.Span(
                [
                    html.I(className="bi bi-exclamation-triangle-fill me-1"),
                    f"{sat * 100:.1f}% saturated",
                ],
                className="text-danger fw-bold",
                title=(
                    f"{sat * 100:.2f}% of pixels reached the saturation level "
                    f"({_saturation_max(img):.0f}). Reduce exposure or gain."
                ),
            )
        )

    return children


def _reset_avg_state() -> None:
    """Drop any cached averaging state (used on pause/resume, exposure
    changes, ROI changes, etc. where frame contents change shape or
    semantics)."""
    global _avg_running_sum  # noqa: PLW0603
    _avg_buffer.clear()
    _avg_running_sum = None


def _averaged_image(image: np.ndarray, n: int) -> np.ndarray:
    """Return a running mean of the last *n* frames including *image*.

    Uses an incremental sum (one add per new frame, one subtract per
    evicted frame) so the cost stays O(H·W) per call regardless of *n*
    — critical for large sensors where stacking N frames into one array
    would allocate hundreds of megabytes per tick and block the Dash
    callback lock long enough for the camera's buffer ring to overflow.

    Resets the internal buffer when *n* or the frame shape changes, so
    callers don't have to worry about ROI changes mid-stream. Returns
    *image* unchanged when ``n == 1``.
    """
    global _avg_buffer, _avg_buffer_shape, _avg_running_sum  # noqa: PLW0603

    n = max(1, min(int(n), MAX_AVG_FRAMES))
    if n == 1:
        if _avg_buffer.maxlen != 1:
            _avg_buffer = collections.deque(maxlen=1)
        _avg_buffer_shape = image.shape
        _avg_buffer.clear()
        _avg_running_sum = None
        return image

    if _avg_buffer.maxlen != n or _avg_buffer_shape != image.shape or _avg_running_sum is None:
        _avg_buffer = collections.deque(maxlen=n)
        _avg_buffer_shape = image.shape
        # Use float32 — enough precision for N ≤ 32 frames of uint8/uint16
        # pixel values, half the memory of float64.
        _avg_running_sum = np.zeros(image.shape, dtype=np.float32)

    # Evict the frame that the deque will drop before appending, so the
    # running sum stays in sync with the buffer contents.
    if len(_avg_buffer) == n:
        _avg_running_sum -= _avg_buffer[0]

    _avg_buffer.append(image)
    _avg_running_sum += image

    mean = _avg_running_sum / len(_avg_buffer)
    if np.issubdtype(image.dtype, np.integer):
        # Round rather than truncate: casting straight to int would shave a
        # consistent ~0.5 count off every pixel, which shows up as a darker
        # image the moment averaging is switched on.
        mean = np.rint(mean)
    return mean.astype(image.dtype)


def _register_callbacks(app: dash.Dash, bp: BeamProfiler) -> None:
    """Wire up all Dash callbacks."""
    global _known_options, _server_paused, _zoom_range  # noqa: PLW0603
    _server_paused = False
    _zoom_range = None
    _known_options = []

    # -- Camera selection -----------------------------------------------------
    # Rescanning and switching both take ``_callback_lock``: swapping the
    # camera out from under an in-flight ``ia.fetch`` would hand the
    # Harvesters C library a destroyed acquirer, which segfaults rather than
    # raising.

    def _settings_body(items: list[Any]) -> Any:
        """Wrap freshly built accordion items, or say why there are none."""
        if items:
            return dbc.Accordion(items, start_collapsed=False, always_open=True)
        return html.P("No camera connected.", className="text-muted p-3")

    @app.callback(
        Output("dropdown-camera", "options"),
        Output("dropdown-camera", "value"),
        Output("div-camera-status", "children", allow_duplicate=True),
        Input("btn-camera-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_cameras(_n: int | None) -> tuple[Any, Any, Any]:
        """Rescan for connected cameras and repopulate the dropdown.

        Enumeration runs under ``_callback_lock``, so the live view freezes
        for as long as it takes. That is deliberate: a GenTL rescan can take a
        second or two on a GigE network, and doing it concurrently with an
        in-flight fetch is exactly the kind of thing the Harvesters C library
        is unhappy about. A rescan is an explicit click, so a brief pause is
        the right trade against a crash.
        """
        with _callback_lock:
            options, current = _camera_options(bp)
        listed = [{"label": o.label, "value": o.key} for o in options]
        real = sum(1 for o in options if not o.is_simulated)
        if real:
            status = f"{real} camera{'s' if real != 1 else ''} found"
        else:
            status = "No hardware found - simulated only"
        return listed, current, status

    @app.callback(
        Output("div-camera-status", "children"),
        Output("store-paused", "data", allow_duplicate=True),
        Output("btn-play-pause", "children", allow_duplicate=True),
        Output("btn-play-pause", "color", allow_duplicate=True),
        Output("settings-container", "children", allow_duplicate=True),
        Output("input-pixel-scale", "value", allow_duplicate=True),
        Input("dropdown-camera", "value"),
        prevent_initial_call=True,
    )
    def switch_camera(key: str | None) -> tuple[Any, ...]:
        """Open the selected camera and hand the profiler over to it.

        The new camera is opened *before* the old one is closed, so a device
        that is unplugged or already claimed by another application leaves the
        current stream untouched instead of dropping the user into a dead app.

        Streaming is left paused afterwards: the caller picked a camera, and
        starting it is the next deliberate click.
        """
        global _server_paused, _zoom_range  # noqa: PLW0603

        if not key:
            return (dash.no_update,) * 6

        with _callback_lock:
            if bp.camera is not None and describe_open_camera(bp.camera).key == key:
                return (dash.no_update,) * 6

            # Resolve against what the dropdown last offered. Re-running
            # discovery here would put a multi-second GenTL enumeration on the
            # critical path of every switch, with _callback_lock held.
            option = find_option(key, _known_options)
            if option is None:
                option = find_option(key, _camera_options(bp)[0])
            if option is None:
                return (f"Unknown camera: {key}",) + (dash.no_update,) * 5

            try:
                camera = open_camera(option)
            except Exception as e:
                logger.warning("Could not switch to %s: %s", option.label, e)
                return (f"Could not open {option.label}: {e}",) + (dash.no_update,) * 5

            bp.attach_camera(camera)

            # Everything derived from the previous camera is now meaningless:
            # the zoom is in the old sensor's micrometres, the averaging
            # buffer holds frames of the old shape, and the fps window
            # measured a different device.
            _server_paused = True
            _zoom_range = None
            _recent_frame_times.clear()
            _reset_avg_state()

            items = _build_setting_items(bp)
            scale = round(bp.pixel_size, 4)

        return (
            f"{option.label} ready - press Play",
            True,
            [html.I(className="bi bi-play-fill me-1"), "Play"],
            "success",
            _settings_body(items),
            scale,
        )

    # -- Play / Pause toggle --------------------------------------------------
    @app.callback(
        Output("store-paused", "data"),
        Output("btn-play-pause", "children"),
        Output("btn-play-pause", "color"),
        Output("settings-container", "children"),
        Input("btn-play-pause", "n_clicks"),
        State("store-paused", "data"),
        prevent_initial_call=True,
    )
    def toggle_pause(n: int, paused: bool) -> tuple[bool, list[Any], str, Any]:
        global _server_paused  # noqa: PLW0603
        new_paused = not paused

        with _callback_lock:
            _server_paused = new_paused
            if bp._mode == "camera" and bp.camera is not None:
                if new_paused:
                    bp.camera.stop_acquisition()
                else:
                    bp.camera.start_acquisition()
            _recent_frame_times.clear()
            _reset_avg_state()

            items = _build_setting_items(bp)

        if new_paused:
            label = [html.I(className="bi bi-play-fill me-1"), "Play"]
            color = "success"
        else:
            label = [html.I(className="bi bi-pause-fill me-1"), "Pause"]
            color = "primary"

        if items:
            settings_body = dbc.Accordion(items, start_collapsed=False, always_open=True)
        else:
            settings_body = html.P("No camera connected.", className="text-muted p-3")

        return new_paused, label, color, settings_body

    # -- Save current frame as PNG -------------------------------------------
    @app.callback(
        Output("download-png", "data"),
        Input("btn-save-png", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_frame_png(_n: int) -> dict[str, Any] | None:
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

    # -- Save current frame as raw NumPy array -------------------------------
    @app.callback(
        Output("download-npy", "data"),
        Input("btn-save-npy", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_frame_npy(_n: int) -> dict[str, Any] | None:
        img = bp.last_img
        if img is None:
            return None
        buf = io.BytesIO()
        np.save(buf, img, allow_pickle=False)
        b64 = base64.b64encode(buf.getvalue()).decode()
        ts = time.strftime("%Y%m%d_%H%M%S")
        return {
            "content": b64,
            "filename": f"beam_{ts}.npy",
            "base64": True,
        }

    # -- Color switch disables colorscale dropdown ----------------------------
    @app.callback(
        Output("dropdown-colorscale", "disabled"),
        Input("switch-color", "value"),
    )
    def toggle_colorscale(color_on: bool) -> bool:
        return not color_on

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

    # -- Spacebar toggles Play / Pause (clientside) --------------------------
    # Installs a single window-level keydown listener that ignores keystrokes
    # in form fields so it doesn't hijack typing in inputs / sliders.
    app.clientside_callback(
        """function() {
            if (window.__pbpSpaceHooked) { return window.dash_clientside.no_update; }
            window.__pbpSpaceHooked = true;
            document.addEventListener('keydown', function(e) {
                if (e.code !== 'Space') return;
                var t = e.target;
                if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
                          t.tagName === 'SELECT' || t.isContentEditable)) {
                    return;
                }
                var btn = document.getElementById('btn-play-pause');
                if (btn) { e.preventDefault(); btn.click(); }
            });
            return window.dash_clientside.no_update;
        }""",
        Output("btn-play-pause", "title"),
        Input("btn-play-pause", "id"),
    )

    # -- Auto-fit / reset zoom buttons ---------------------------------------
    # The zoom range is held in the module-level ``_zoom_range`` variable
    # (the live source of truth read by ``update_live``) and a ``Patch``
    # is sent to the figure so the change is visible immediately, whether
    # the stream is running or paused.
    @app.callback(
        Output("live-graph", "figure", allow_duplicate=True),
        Input("btn-zoom-fit", "n_clicks"),
        prevent_initial_call=True,
    )
    def auto_fit_zoom(n_clicks: int | None) -> Any:
        global _zoom_range  # noqa: PLW0603
        if not n_clicks:
            return dash.no_update
        popt_x = bp._last_popt_x
        popt_y = bp._last_popt_y
        if popt_x is None or popt_y is None:
            return dash.no_update
        ps = bp.pixel_size
        cx, cy = popt_x[1] * ps, popt_y[1] * ps
        # 1/e² semi-axis = 2σ.  Pad by 1.5× → ±3σ box around the beam.
        rx, ry = 2 * abs(popt_x[2]) * ps, 2 * abs(popt_y[2]) * ps
        pad = 1.5
        zoom = {
            "x": [cx - pad * rx, cx + pad * rx],
            "y": [cy - pad * ry, cy + pad * ry],
        }
        with _callback_lock:
            _zoom_range = zoom
        patch = Patch()
        patch["layout"]["xaxis"]["range"] = zoom["x"]
        patch["layout"]["yaxis"]["range"] = zoom["y"]
        return patch

    @app.callback(
        Output("live-graph", "figure", allow_duplicate=True),
        Input("btn-zoom-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_zoom(n_clicks: int | None) -> Any:
        global _zoom_range  # noqa: PLW0603
        if not n_clicks:
            return dash.no_update
        with _callback_lock:
            _zoom_range = None
        patch = Patch()
        img = bp.last_img
        if img is not None:
            ps = bp.pixel_size
            patch["layout"]["xaxis"]["range"] = [0, img.shape[1] * ps]
            patch["layout"]["yaxis"]["range"] = [0, img.shape[0] * ps]
        return patch

    # -- Draggable column divider (clientside) -------------------------------
    # Keeps the layout state purely in the DOM — no Dash store round-trip
    # while dragging, so it stays smooth even on slower machines.
    app.clientside_callback(
        """function() {
            if (window.__pbpSplitHooked) { return window.dash_clientside.no_update; }
            window.__pbpSplitHooked = true;
            var divider = document.getElementById('col-divider');
            var side = document.getElementById('col-side');
            if (!divider || !side) return window.dash_clientside.no_update;
            var dragging = false;
            divider.addEventListener('mousedown', function(e) {
                dragging = true;
                document.body.style.userSelect = 'none';
                e.preventDefault();
            });
            window.addEventListener('mousemove', function(e) {
                if (!dragging) return;
                var w = Math.max(240, Math.min(window.innerWidth - 240,
                                               window.innerWidth - e.clientX));
                side.style.flex = '0 0 ' + w + 'px';
                if (window.Plotly) {
                    var gd = document.getElementById('live-graph');
                    if (gd) { window.Plotly.Plots.resize(gd); }
                }
            });
            window.addEventListener('mouseup', function() {
                if (!dragging) return;
                dragging = false;
                document.body.style.userSelect = '';
            });
            return window.dash_clientside.no_update;
        }""",
        Output("col-divider", "title"),
        Input("col-divider", "id"),
    )

    # -- Pixel scale override -------------------------------------------------
    @app.callback(
        Output("input-pixel-scale", "value"),
        Input("input-pixel-scale", "n_submit"),
        Input("input-pixel-scale", "n_blur"),
        State("input-pixel-scale", "value"),
        prevent_initial_call=True,
    )
    def set_pixel_scale(_n_submit: int | None, _n_blur: int | None, val: float | None) -> float:
        """Override the pixel pitch used to convert pixels to micrometers."""
        if val is not None and val > 0:
            # build_figure reads pixel_size several times per frame (heatmap
            # extent, ellipse, axis ranges). Changing it mid-render would
            # leave those disagreeing for one frame.
            with _callback_lock:
                bp.pixel_size = val
        return round(bp.pixel_size, 4)

    # -- Exposure slider + input (kept in sync) -------------------------------
    @app.callback(
        Output("slider-exposure", "value"),
        Output("input-exposure", "value"),
        Input("slider-exposure", "value"),
        Input("input-exposure", "value"),
        prevent_initial_call=True,
    )
    def set_exposure(slider_val: float | None, input_val: float | None) -> tuple[Any, Any]:
        trigger = ctx.triggered_id
        val = slider_val if trigger == "slider-exposure" else input_val
        if val is None:
            return dash.no_update, dash.no_update
        if bp.camera is not None:
            with _callback_lock:
                try:
                    was_acquiring = bp.camera.is_acquiring
                    bp.camera.set_exposure(val / 1000.0)
                    if was_acquiring and not bp.camera.is_acquiring:
                        bp.camera.start_acquisition()
                    _recent_frame_times.clear()
                    _reset_avg_state()
                except Exception as e:
                    logger.warning(f"Failed to set exposure: {e}")
        # Mirror the committed value to the *other* control only — echoing
        # the triggering control would cause a pointless second callback
        # round-trip and can jitter the slider thumb while the user drags.
        if trigger == "slider-exposure":
            return dash.no_update, val
        return val, dash.no_update

    # -- Gain slider + input (kept in sync) -----------------------------------
    @app.callback(
        Output("slider-gain", "value"),
        Output("input-gain", "value"),
        Input("slider-gain", "value"),
        Input("input-gain", "value"),
        prevent_initial_call=True,
    )
    def set_gain(slider_val: float | None, input_val: float | None) -> tuple[Any, Any]:
        trigger = ctx.triggered_id
        val = slider_val if trigger == "slider-gain" else input_val
        if val is None:
            return dash.no_update, dash.no_update
        if bp.camera is not None:
            with _callback_lock:
                try:
                    was_acquiring = bp.camera.is_acquiring
                    bp.camera.set_gain(val)
                    if was_acquiring and not bp.camera.is_acquiring:
                        bp.camera.start_acquisition()
                    _recent_frame_times.clear()
                    _reset_avg_state()
                except Exception as e:
                    logger.warning(f"Failed to set gain: {e}")
        if trigger == "slider-gain":
            return dash.no_update, val
        return val, dash.no_update

    # -- ROI apply ------------------------------------------------------------
    # Registered unconditionally: the attached camera can change at runtime,
    # so whether one supports ROI is not a question that can be settled once
    # at start-up. Each callback re-checks the live camera instead.

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
        if ox is None or oy is None or w is None or h is None:
            return "Please enter offset/width/height"
        with _callback_lock:
            try:
                offset_x = int(ox)
                offset_y = int(oy)
                width = int(w)
                height = int(h)
                was_acquiring = bp.camera.is_acquiring
                if was_acquiring:
                    bp.camera.stop_acquisition()
                getattr(bp.camera, "set_roi")(
                    offset_x=offset_x, offset_y=offset_y, width=width, height=height
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
        with _callback_lock:
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

    # -- GenICam feature callbacks (pattern-matching) -------------------------
    # Also unconditional. Pattern-matching callbacks happily target components
    # that appear later, which is exactly what happens when a camera switch
    # rebuilds the Setting panel with a different feature set.

    @app.callback(
        Output({"type": "genicam-num", "feature": MATCH}, "value"),
        Output({"type": "genicam-num-input", "feature": MATCH}, "value"),
        Input({"type": "genicam-num", "feature": MATCH}, "value"),
        Input({"type": "genicam-num-input", "feature": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def set_genicam_numeric(slider_val: float | None, input_val: float | None) -> tuple[Any, Any]:
        trigger = ctx.triggered_id
        source = trigger.get("type") if isinstance(trigger, dict) else None
        value = slider_val if source == "genicam-num" else input_val
        if value is None or bp.camera is None or not isinstance(trigger, dict):
            return dash.no_update, dash.no_update
        feature = trigger.get("feature")
        if feature is None:
            return dash.no_update, dash.no_update
        with _callback_lock:
            nm = getattr(bp.camera, "node_map", None)
            if nm is not None:
                node = getattr(nm, feature, None)
                if node is not None:
                    try:
                        was_acquiring = bp.camera.is_acquiring
                        node.value = value
                        if was_acquiring and not bp.camera.is_acquiring and not _server_paused:
                            bp.camera.start_acquisition()
                    except Exception as e:
                        logger.debug("Failed to set %s: %s", feature, e)
        # Mirror to the other control only (see set_exposure for rationale).
        if source == "genicam-num":
            return dash.no_update, value
        return value, dash.no_update

    @app.callback(
        Output({"type": "genicam-sel", "feature": MATCH}, "value"),
        Input({"type": "genicam-sel", "feature": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def set_genicam_select(value: str | None) -> Any:
        if value is None or bp.camera is None:
            return dash.no_update
        feature = ctx.triggered_id["feature"]
        with _callback_lock:
            nm = getattr(bp.camera, "node_map", None)
            if nm is not None:
                node = getattr(nm, feature, None)
                if node is not None:
                    try:
                        was_acquiring = bp.camera.is_acquiring
                        node.value = value
                        if was_acquiring and not bp.camera.is_acquiring and not _server_paused:
                            bp.camera.start_acquisition()
                    except Exception as e:
                        logger.debug("Failed to set %s: %s", feature, e)
        return value

    @app.callback(
        Output({"type": "genicam-sw", "feature": MATCH}, "value"),
        Input({"type": "genicam-sw", "feature": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def set_genicam_switch(value: bool) -> Any:
        if bp.camera is None:
            return dash.no_update
        feature = ctx.triggered_id["feature"]
        with _callback_lock:
            nm = getattr(bp.camera, "node_map", None)
            if nm is not None:
                node = getattr(nm, feature, None)
                if node is not None:
                    try:
                        was_acquiring = bp.camera.is_acquiring
                        node.value = value
                        if was_acquiring and not bp.camera.is_acquiring and not _server_paused:
                            bp.camera.start_acquisition()
                    except Exception as e:
                        logger.debug("Failed to set %s: %s", feature, e)
        return value

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
        State("input-avg-n", "value"),
    )
    def update_live(
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
        avg_n: int | None,
    ) -> tuple[Any, ...]:
        if paused or _server_paused:
            return (dash.no_update,) * 4

        if not _callback_lock.acquire(blocking=False):
            # A previous tick is still running; skip this one and let the
            # next interval fire. Keeps the UI responsive when a camera
            # fetch takes longer than the tick interval.
            return (dash.no_update,) * 4

        try:
            if analysis and bp.fit_method != analysis:
                bp.fit_method = analysis
                bp._last_popt_x = None
                bp._last_popt_y = None
                bp._last_popt_2d = None
            if definition and bp.definition != definition:
                bp.definition = definition

            if bp._mode == "camera" and bp.camera is not None:
                # Cap fetch at one tick so sliders/buttons (which share
                # ``_callback_lock``) never wait more than ~100 ms. During
                # multi-second exposures this just times out repeatedly
                # until the producer delivers the next frame.
                try:
                    img = bp.camera.get_image(timeout=0.1)
                except TimeoutError:
                    return (dash.no_update,) * 4
            else:
                img = bp.last_img

            if img is None:
                return (dash.no_update,) * 4

            img = _averaged_image(img, avg_n or 1)
            bp.last_img = img
            popt_x, popt_y = bp.analyze(img)

            cs = cs_name if color_on else GRAY_COLORSCALE
            zmin = None if auto_range else (zmin_val if zmin_val is not None else 0)
            zmax_default = _saturation_max(img)
            zmax = None if auto_range else (zmax_val if zmax_val is not None else zmax_default)
            # Read the live source of truth (mutated by Auto-fit / Reset
            # under ``_callback_lock``) instead of capturing it as Dash
            # ``State``: a State snapshot can be 50–100 ms stale if a
            # zoom click fires after this tick started, which would
            # cause a one-frame blink to the previous zoom.
            current_zoom = _zoom_range
            xrange = current_zoom["x"] if current_zoom else None
            yrange = current_zoom["y"] if current_zoom else None
            fig = build_figure(
                bp,
                img,
                popt_x,
                popt_y,
                colorscale=cs,
                zmin=zmin,
                zmax=zmax,
                dark_theme=dark_theme,
                xrange=xrange,
                yrange=yrange,
            )

            _recent_frame_times.append(time.monotonic())
            frame_count += 1
            return (
                fig,
                _format_results(bp),
                _build_status(bp, img, frame_count),
                frame_count,
            )
        except Exception:
            logger.exception("Update error")
            return (dash.no_update,) * 4
        finally:
            _callback_lock.release()
