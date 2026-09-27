"""The live browser GUI: figure building, the app factory, and callbacks.

The page's components are built in :mod:`pybeamprofiler.dash_layout`; what is
here is everything that moves — the heatmap figure rebuilt on each tick, and
the callbacks behind every control.

One rule governs the whole module: anything that touches the camera or the
profiler's state holds ``_callback_lock``. The Harvesters backend is a C
library that will segfault, not raise, if a buffer is fetched while the
acquirer is being destroyed, so the render tick and the controls that stop,
reconfigure or replace the camera are strictly serialised. The render tick
takes the lock without blocking and skips a frame rather than queueing, so a
slow camera fetch cannot make the controls feel stuck.
"""

from __future__ import annotations

import base64
import collections
import functools
import io
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import MATCH, Input, Output, Patch, State, ctx, html
from PIL import Image

from .constants import (
    MAX_AVG_FRAMES,
    MAX_DISPLAY_DIM,
)
from .dash_layout import (
    GRAY_COLORSCALE,
    _build_setting_items,
    _camera_options,
    _format_results,
    _page,
    _play_pause_face,
    _with_open_camera,
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
_SATURATION_PIXEL_FRACTION = 0.001  # 0.1% of pixels at the saturation level ⇒ warn.


def _saturation_max(image: np.ndarray, bit_depth: int | None = None) -> float:
    """Return the saturation level for *image*.

    For integer images this is ``2**bit_depth - 1`` when the camera reports
    its bit depth, and otherwise the dtype's largest value. The dtype alone
    cannot tell a 12-bit sensor packed in ``uint16`` from a 16-bit one, so
    without the bit depth the level there is 65535 and a clipped 12-bit
    frame is never flagged. For floating-point images we assume normalised
    ``[0, 1]`` data when the observed max is ≤ 1, otherwise use the observed
    max as a heuristic.
    """
    if np.issubdtype(image.dtype, np.integer):
        dtype_max = float(np.iinfo(image.dtype).max)
        if bit_depth:
            # A Mono8 stream from a 12-bit sensor still clips at 255.
            return min(float(2**bit_depth - 1), dtype_max)
        return dtype_max
    obs_max = float(image.max()) if image.size else 1.0
    return 1.0 if obs_max <= 1.0 else obs_max


def _saturation_fraction(image: np.ndarray, level: float | None = None) -> float:
    """Fraction of pixels at or above the saturation level.

    Args:
        image: The frame to check.
        level: The saturation level, from :func:`_saturation_max`; derived
            from the dtype when not given.
    """
    if image.size == 0:
        return 0.0
    sat = _saturation_max(image) if level is None else level
    # For integer images, only the exact saturation value counts.
    # For floating-point images, allow a small epsilon near the inferred max.
    threshold = sat if np.issubdtype(image.dtype, np.integer) else sat - 1e-6
    # A plain max() reduction is several times cheaper than the comparison
    # below, which has to allocate a full-frame boolean temporary. Nothing is
    # saturated far more often than not, so check the cheap way first.
    if float(image.max()) < threshold:
        return 0.0
    return float(np.count_nonzero(image >= threshold)) / image.size


def _camera_bit_depth(bp: BeamProfiler) -> int | None:
    """The attached camera's reported bit depth, or ``None`` if unknown."""
    depth = getattr(bp.camera, "bit_depth", None) if bp.camera is not None else None
    # Anything but a plausible int (a mock's attribute, a bool) means unknown.
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 32:
        return None
    return depth


# ---------------------------------------------------------------------------
# Figure builder — LaseView-style overlay
# ---------------------------------------------------------------------------


def _normalize_profile(
    data: np.ndarray,
    span: float,
    fraction: float = _PROFILE_FRACTION,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Scale a 1-D profile to occupy *fraction* of *span*.

    With *reference*, the scale comes from that profile's range instead of
    *data*'s own. A fit curve is drawn this way, against the data it was fitted
    to: scaled to its own range, any fit filled the same height as the data, so
    a wrong amplitude or baseline was invisible on screen.
    """
    ref = data if reference is None else reference
    lo, hi = float(np.min(ref)), float(np.max(ref))
    rng = hi - lo if hi != lo else 1.0
    return (data - lo) / rng * span * fraction


def _profile_traces(
    bp: BeamProfiler,
    image: np.ndarray,
    popt_x: np.ndarray | list[Any] | None,
    popt_y: np.ndarray | list[Any] | None,
    *,
    xrange: list[float] | None,
    yrange: list[float] | None,
    line_colour: str,
    fill_x: str,
    fill_y: str,
) -> list[Any]:
    """The X and Y profiles, and their fits, for :func:`build_figure`."""
    h, w = image.shape
    ps = bp.pixel_size
    x_max, y_max = w * ps, h * ps
    cut_x = getattr(bp, "_linecut_x", None)
    cut_y = getattr(bp, "_linecut_y", None)
    traces: list[Any] = []

    # ── Where the profiles go ───────────────────────────────────
    # Each profile hugs an edge of whatever is in view -- the X projection
    # the bottom, the Y projection the left -- and takes a fraction of the
    # view, not of the sensor. Pinned to the sensor's own edges, as they
    # used to be, they were left behind by any zoom: after Auto-fit none of
    # either curve was on screen. When the view extends past the sensor the
    # profile stays on the sensor's edge, where its fill ends.
    view_x = xrange if xrange is not None else [0.0, x_max]
    view_y = yrange if yrange is not None else [0.0, y_max]
    x_base, x_span = max(view_y[0], 0.0), view_y[1] - view_y[0]
    y_base, y_span = max(view_x[0], 0.0), view_x[1] - view_x[0]

    # ── The profiles to draw ────────────────────────────────────
    # Whatever the fit was run on: the cached projections, or in linecut mode
    # the row and column through the peak. Falling back to the full-frame
    # sums there drew the fit of one row over the projection of the whole
    # frame -- on a tilted beam a 575 um curve under a 327 um fit.
    cached_proj_x = getattr(bp, "_last_proj_x", None)
    cached_proj_y = getattr(bp, "_last_proj_y", None)
    if (
        (cached_proj_x is None or cached_proj_y is None)
        and bp.fit_method == "linecut"
        and cut_x is not None
        and cut_y is not None
        and 0 <= cut_x < w
        and 0 <= cut_y < h
    ):
        cached_proj_x, cached_proj_y = image[int(cut_y), :], image[:, int(cut_x)]
    proj_x = (cached_proj_x if cached_proj_x is not None else np.sum(image, axis=0)).astype(float)
    proj_y = (cached_proj_y if cached_proj_y is not None else np.sum(image, axis=1)).astype(float)

    # ── X profile (bottom edge) ─────────────────────────────────
    x_ax = np.arange(w)
    norm_x = x_base + _normalize_profile(proj_x, x_span)

    traces.append(
        go.Scatter(
            x=x_ax * ps,
            y=norm_x,
            mode="lines",
            line=dict(color=line_colour, width=1.5),
            fill="tozeroy",
            fillcolor=fill_x,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if popt_x is not None:
        fit_x = bp.gaussian(x_ax, *popt_x).astype(float)
        norm_fit_x = x_base + _normalize_profile(fit_x, x_span, reference=proj_x)
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
    norm_y = y_base + _normalize_profile(proj_y, y_span)

    traces.append(
        go.Scatter(
            x=norm_y,
            y=y_ax * ps,
            mode="lines",
            line=dict(color=line_colour, width=1.5),
            fill="tozerox",
            fillcolor=fill_y,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if popt_y is not None:
        fit_y = bp.gaussian(y_ax, *popt_y).astype(float)
        norm_fit_y = y_base + _normalize_profile(fit_y, y_span, reference=proj_y)
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

    return traces


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
    when *image* is ``None``. With the profiler's heatmap-only flag set
    (``--heatmap-only``) the profiles and their fits are left out; the
    heatmap, the beam ellipse and the linecut crosshair stay.

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
    # Only where the last frame was actually cut: the coordinates are absent
    # or None before the first linecut, and after any frame that was not one
    # (another fit method, or FWHM/D4σ, which skip the fit altogether).
    cut_x = getattr(bp, "_linecut_x", None)
    cut_y = getattr(bp, "_linecut_y", None)
    if bp.fit_method == "linecut" and cut_x is not None and cut_y is not None:
        lx, ly = cut_x * ps, cut_y * ps
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

    # ── Profiles ────────────────────────────────────────────────
    # Heatmap-only mode (--heatmap-only) leaves the curves out. The flag used
    # to be stored and then ignored here, so it changed nothing in the GUI.
    # Checked with ``is True`` because an unknown attribute on the profiler
    # is looked up on the camera, and a mock camera's would be truthy.
    if getattr(bp, "_heatmap_only", False) is not True:
        traces.extend(
            _profile_traces(
                bp,
                image,
                popt_x,
                popt_y,
                xrange=xrange,
                yrange=yrange,
                line_colour=prof_line,
                fill_x=fill_x,
                fill_y=fill_y,
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
        bp.analyze(initial_img)
        # Kept so the first page load has a frame to show; the page is built
        # from the profiler's state (see _serve_page).
        bp.last_img = initial_img

    # ── Layout ──────────────────────────────────────────────────
    # One enumeration at start-up, shared by the dropdown and the cache the
    # switch callback resolves against. Scanning twice would double a
    # multi-second GenTL walk on a machine with hardware attached.
    camera_options, _ = _camera_options(bp)

    # A function rather than a component tree, so Dash builds the page for
    # each load. A tree built here kept serving the start-up state: after a
    # camera switch, a reloaded page named the old camera and pixel pitch
    # and offered Pause on a stopped stream -- and merely tabbing out of the
    # Scale box wrote the stale pitch back into the profiler.
    app.layout = functools.partial(_serve_page, bp)

    _register_callbacks(app, bp)
    # Seeded only now: _register_callbacks clears the module state, this
    # cache included, so filling it any earlier leaves it empty and every
    # camera switch falls back to a full rescan with the lock held.
    global _known_options  # noqa: PLW0603
    _known_options = camera_options
    return app


def _serve_page(bp: BeamProfiler) -> Any:
    """Build the page from what is in force right now.

    Runs on every page load, under the lock: the Setting panel reads the
    camera's node map, and the figure and the controls read profiler state
    that a callback might be replacing.
    """
    with _callback_lock:
        options, current = _with_open_camera(bp, _known_options)
        figure: Any = go.Figure()
        results = None
        if bp.last_img is not None:
            xrange, yrange = _zoom_in_um(bp)
            figure = build_figure(
                bp,
                bp.last_img,
                bp._last_popt_x,
                bp._last_popt_y,
                xrange=xrange,
                yrange=yrange,
            )
            results = _format_results(bp)
        return _page(bp, figure, options, current, paused=_server_paused, results=results)


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

# Authoritative zoom state, in sensor *pixels*, mutated by Auto-fit, Reset
# and mouse zooms under ``_callback_lock`` and read by ``update_live``. Using
# a module variable (rather than a Dash ``State``) avoids a 50–100 ms
# stale-snapshot race that would otherwise blink the previous zoom for one
# frame whenever a click landed mid-tick. Pixels rather than micrometres so
# that correcting the pixel scale keeps the same part of the sensor in view;
# a box stored in micrometres framed a different region after the change.
_zoom_range: dict[str, list[float]] | None = None


def _zoom_in_um(bp: BeamProfiler) -> tuple[list[float] | None, list[float] | None]:
    """The current zoom as ``(xrange, yrange)`` in micrometres, or Nones.

    The caller must hold ``_callback_lock``.
    """
    if _zoom_range is None:
        return None, None
    ps = bp.pixel_size
    return [v * ps for v in _zoom_range["x"]], [v * ps for v in _zoom_range["y"]]


def _zoom_after_relayout(
    relayout: dict[str, Any],
    zoom: dict[str, list[float]] | None,
    frame_shape: tuple[int, ...] | None,
    pixel_size: float,
) -> dict[str, list[float]] | None:
    """Fold a Plotly ``relayoutData`` event into the zoom, in pixels.

    A box zoom or pan reports ``xaxis.range[0]``/``[1]`` (sometimes a
    ``xaxis.range`` pair) for the axes it moved, and the modebar's autoscale
    reports ``autorange``. An axis the event doesn't mention keeps its current
    range. Anything else -- ``autosize`` on a resize, a mode change -- leaves
    the zoom as it was.

    Returns:
        The new zoom, or ``None`` for the full frame.
    """

    def axis(name: str) -> list[float] | None:
        pair: Any = relayout.get(f"{name}.range")
        if pair is None:
            pair = (relayout.get(f"{name}.range[0]"), relayout.get(f"{name}.range[1]"))
        if len(pair) != 2 or pair[0] is None or pair[1] is None:
            return None
        return sorted([float(pair[0]) / pixel_size, float(pair[1]) / pixel_size])

    if relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
        return None
    x, y = axis("xaxis"), axis("yaxis")
    if (x is None and y is None) or frame_shape is None:
        return zoom
    full = {"x": [0.0, float(frame_shape[1])], "y": [0.0, float(frame_shape[0])]}
    new = {"x": x or list((zoom or full)["x"]), "y": y or list((zoom or full)["y"])}
    # A double-click reports the full extent as explicit ranges rather than
    # as autorange. Treat that as no zoom, so the view keeps following the
    # frame's own extent.
    if np.allclose(new["x"], full["x"], atol=0.5) and np.allclose(new["y"], full["y"], atol=0.5):
        return None
    return new


# Consecutive ticks on which the camera raised something other than a
# timeout. After _MAX_CAMERA_FAILURES in a row the stream is paused: a device
# that has gone away does not come back for being asked twenty times a
# second, and each of those attempts used to log a full traceback while the
# page went on showing the last good frame as if nothing had happened.
_camera_failures = 0
_MAX_CAMERA_FAILURES = 10

# The last error a render tick logged and when, plus how many repeats have
# been held back since, so a persistent fault is reported every so often
# rather than on every tick.
_last_tick_error: tuple[str, float] | None = None
_suppressed_tick_errors = 0
_TICK_ERROR_LOG_INTERVAL_S = 10.0


def _error_status(message: str) -> Any:
    """A status-bar line that reports *message* as an error."""
    return html.Span(
        [html.I(className="bi bi-exclamation-triangle-fill me-1"), message],
        className="text-danger fw-bold",
    )


def _log_tick_error(message: str, exc: BaseException) -> None:
    """Log a render-tick failure, at most once an interval while it repeats."""
    global _last_tick_error, _suppressed_tick_errors  # noqa: PLW0603
    now = time.monotonic()
    if _last_tick_error is not None:
        last_message, last_time = _last_tick_error
        if last_message == message and now - last_time < _TICK_ERROR_LOG_INTERVAL_S:
            _suppressed_tick_errors += 1
            return
    repeats = _suppressed_tick_errors
    note = f" (repeated {repeats} times since last logged)" if repeats else ""
    logger.error("%s%s", message, note, exc_info=exc)
    _last_tick_error = (message, now)
    _suppressed_tick_errors = 0


def _measured_fps() -> float:
    """Compute frames-per-second from the rolling timestamp window."""
    if len(_recent_frame_times) < 2:
        return 0.0
    span = _recent_frame_times[-1] - _recent_frame_times[0]
    if span <= 0:
        return 0.0
    return (len(_recent_frame_times) - 1) / span


def _build_status(
    bp: BeamProfiler, img: np.ndarray, frame_count: int, raw: np.ndarray | None = None
) -> Any:
    """Build the status bar contents (frame, fps, exposure, gain, saturation).

    Args:
        bp: The profiler, read for the camera's exposure, gain and bit depth.
        img: The frame as displayed.
        frame_count: Frames shown so far.
        raw: The frame as the camera delivered it, if *img* is an average.
            Saturation is judged on this one: the beam jitters, so a core
            clipped in every raw frame rarely stays at the maximum in all of
            them, and the averaged frame hid the clipping entirely.
    """
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

    frame = img if raw is None else raw
    level = _saturation_max(frame, _camera_bit_depth(bp))
    sat = _saturation_fraction(frame, level)
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
                    f"({level:.0f}). Reduce exposure or gain."
                ),
            )
        )

    return children


def _png_bytes(image: np.ndarray) -> bytes:
    """Encode a frame as a greyscale PNG without wrapping or clipping it.

    uint8 and uint16 frames, which is everything a camera delivers (averaged
    or not), are written exactly as 8- and 16-bit greyscale. Anything else
    came from a file: integers that fit 0..65535 are written exactly as 16
    bit, and wider integers and floats are scaled linearly onto the 16-bit
    range -- their shape survives but not their units, which the .npy
    download keeps. Handed straight to Pillow, a float frame could not be
    written at all ("cannot write mode F as PNG") and a 32-bit one was
    silently clipped at 65535.
    """
    if image.dtype in (np.uint8, np.uint16):
        data = image
    elif image.dtype == np.bool_:
        data = image.astype(np.uint8) * 255
    elif (
        np.issubdtype(image.dtype, np.integer)
        and image.size
        and image.min() >= 0
        and image.max() <= np.iinfo(np.uint16).max
    ):
        data = image.astype(np.uint16)
    else:
        values = image.astype(np.float64)
        finite = np.isfinite(values)
        if finite.any():
            lo, hi = float(values[finite].min()), float(values[finite].max())
            values[~finite] = lo
        else:
            lo = hi = 0.0
            values[:] = 0.0
        scale = 65535.0 / (hi - lo) if hi > lo else 0.0
        data = np.rint((values - lo) * scale).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(data).save(buf, format="PNG")
    return buf.getvalue()


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


def _discard_frame_history(bp: BeamProfiler) -> None:
    """Forget everything measured from frames the next frame won't match.

    Needed whenever the coordinate system of the frames changes under us: an
    ROI moves the origin and usually the shape, and a camera switch changes
    both along with the pixel pitch. The fitter warm-starts from the previous
    frame, so a centre measured before the change seeds the next fit outside
    the new frame; the averaging buffer would blend two different windows;
    the zoom box would frame the wrong region; and the fps window would span
    two different frame sizes.

    The caller must hold ``_callback_lock``.
    """
    global _zoom_range  # noqa: PLW0603
    bp.reset_analysis()
    _reset_avg_state()
    _recent_frame_times.clear()
    _zoom_range = None


def _paired_values(requested: Any, actual: Any, *, from_slider: bool) -> tuple[Any, Any]:
    """``(slider, box)`` outputs after a write that may not have stuck as asked.

    The control that was not touched always shows *actual*, the value read
    back from the device. The one that was touched is left alone when the
    device took the value as given -- echoing it back only costs a redraw --
    and corrected when the device clamped, quantised or refused it. Echoing
    the request unconditionally, as this used to, showed values the camera
    did not have.
    """
    if actual is None:
        actual = requested
    took = actual == requested or (
        isinstance(actual, (int, float))
        and isinstance(requested, (int, float))
        and np.isclose(actual, requested, rtol=1e-9, atol=1e-9)
    )
    touched = dash.no_update if took else actual
    return (touched, actual) if from_slider else (actual, touched)


def _register_callbacks(app: dash.Dash, bp: BeamProfiler) -> None:
    """Wire up all Dash callbacks, and reset the state they share.

    The state is module-level, so every app built in this process sees it.
    Without the reset a second ``create_app`` -- a test, or a notebook that
    relaunches the GUI -- would inherit the previous session's pause flag,
    zoom, fps window and averaged frames.
    """
    global _known_options, _server_paused, _zoom_range, _camera_failures  # noqa: PLW0603
    global _last_tick_error, _suppressed_tick_errors  # noqa: PLW0603
    _server_paused = False
    _zoom_range = None
    _known_options = []
    _reset_avg_state()
    _recent_frame_times.clear()
    _camera_failures = 0
    _last_tick_error = None
    _suppressed_tick_errors = 0

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
        global _known_options  # noqa: PLW0603
        with _callback_lock:
            options, current = _camera_options(bp)
            # What the dropdown now offers is what a switch resolves against.
            _known_options = options
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
        Output("dropdown-camera", "value", allow_duplicate=True),
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

        When the switch fails, the selection goes back to the camera that is
        still open. Left on the one that failed, the dropdown named the wrong
        camera, and picking it again to retry could not fire at all: Dash only
        calls back when the value changes.
        """
        global _server_paused, _camera_failures  # noqa: PLW0603
        nothing = (dash.no_update,) * 7

        if not key:
            return nothing

        with _callback_lock:
            current = describe_open_camera(bp.camera).key if bp.camera is not None else ""
            if key == current:
                return nothing

            def refuse(message: str) -> tuple[Any, ...]:
                # Back to the open camera -- or to no selection, when a file
                # is being shown and there is no camera open.
                return (message, *(dash.no_update,) * 5, current)

            # Resolve against what the dropdown last offered. Re-running
            # discovery here would put a multi-second GenTL enumeration on the
            # critical path of every switch, with _callback_lock held.
            option = find_option(key, _known_options)
            if option is None:
                option = find_option(key, _camera_options(bp)[0])
            if option is None:
                return refuse(f"Unknown camera: {key}")

            try:
                camera = open_camera(option)
            except Exception as e:
                logger.warning("Could not switch to %s: %s", option.label, e)
                # open_camera's message already names the camera.
                return refuse(str(e) or f"Could not open {option.label}")

            bp.attach_camera(camera)
            _discard_frame_history(bp)
            _server_paused = True
            _camera_failures = 0

            items = _build_setting_items(bp)
            scale = round(bp.pixel_size, 4)

        button_children, button_color = _play_pause_face(True)
        return (
            f"{option.label} ready - press Play",
            True,
            button_children,
            button_color,
            _settings_body(items),
            scale,
            dash.no_update,
        )

    # -- Play / Pause toggle --------------------------------------------------
    @app.callback(
        Output("store-paused", "data"),
        Output("btn-play-pause", "children"),
        Output("btn-play-pause", "color"),
        Output("settings-container", "children"),
        Output("status-bar", "children", allow_duplicate=True),
        Input("btn-play-pause", "n_clicks"),
        State("store-paused", "data"),
        prevent_initial_call=True,
    )
    def toggle_pause(n: int, paused: bool) -> tuple[Any, ...]:
        """Start or stop streaming, and relabel the button to match.

        Also rebuilds the Setting panel: values the camera changed on its
        own while running (auto-exposure, temperature) are only worth
        re-reading when the stream is not competing for the lock.

        A camera that refuses to start leaves the stream paused, with the
        reason in the status bar, rather than failing the callback and
        leaving the button and the server disagreeing about the state.
        """
        global _server_paused, _camera_failures  # noqa: PLW0603
        new_paused = not paused
        status: Any = dash.no_update

        with _callback_lock:
            _server_paused = new_paused
            if bp._mode == "camera" and bp.camera is not None:
                if new_paused:
                    bp.camera.stop_acquisition()
                else:
                    try:
                        bp.camera.start_acquisition()
                    except Exception as e:
                        logger.warning("Could not start the camera: %s", e)
                        new_paused = _server_paused = True
                        status = _error_status(
                            f"Could not start the camera: {str(e) or type(e).__name__}"
                        )
            # Play is also the retry after the stream paused itself, so it
            # starts a fresh count of camera failures.
            _camera_failures = 0
            _recent_frame_times.clear()
            _reset_avg_state()

            items = _build_setting_items(bp)

        label, color = _play_pause_face(new_paused)
        return new_paused, label, color, _settings_body(items), status

    # -- Save current frame as PNG -------------------------------------------
    @app.callback(
        Output("download-png", "data"),
        Input("btn-save-png", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_frame_png(_n: int) -> dict[str, Any] | None:
        """Download the current frame as a PNG."""
        img = bp.last_img
        if img is None:
            return None
        b64 = base64.b64encode(_png_bytes(img)).decode()
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
        """Download the current frame as a raw ``.npy`` array.

        Unlike the PNG this keeps the original dtype, so 12- and 16-bit
        sensor data survives for later analysis.
        """
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
        """Grey out the colorscale picker when colour is switched off."""
        return not color_on

    # -- Auto-range toggle disables min/max inputs ----------------------------
    @app.callback(
        Output("input-zmin", "disabled"),
        Output("input-zmax", "disabled"),
        Input("switch-autorange", "value"),
    )
    def toggle_autorange(auto: bool) -> tuple[bool, bool]:
        """Grey out the manual min/max boxes while auto-range is on."""
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
        """Zoom to a +/-3 sigma box around the fitted beam centre.

        The fit is read under the lock along with the write. Reading it
        outside let a camera switch land in between, leaving the new camera
        zoomed onto where the old one's beam had been.
        """
        global _zoom_range  # noqa: PLW0603
        if not n_clicks:
            return dash.no_update
        with _callback_lock:
            popt_x, popt_y = bp._last_popt_x, bp._last_popt_y
            if popt_x is None or popt_y is None:
                return dash.no_update
            cx, cy = float(popt_x[1]), float(popt_y[1])
            # 1/e² semi-axis = 2σ.  Pad by 1.5× → ±3σ box around the beam.
            rx, ry = 2 * abs(float(popt_x[2])), 2 * abs(float(popt_y[2]))
            if not np.all(np.isfinite([cx, cy, rx, ry])) or rx == 0 or ry == 0:
                return dash.no_update
            pad = 1.5
            _zoom_range = {
                "x": [cx - pad * rx, cx + pad * rx],
                "y": [cy - pad * ry, cy + pad * ry],
            }
            xrange, yrange = _zoom_in_um(bp)
        patch = Patch()
        patch["layout"]["xaxis"]["range"] = xrange
        patch["layout"]["yaxis"]["range"] = yrange
        return patch

    @app.callback(
        Output("live-graph", "figure", allow_duplicate=True),
        Input("btn-zoom-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_zoom(n_clicks: int | None) -> Any:
        """Zoom back out to the full sensor."""
        global _zoom_range  # noqa: PLW0603
        if not n_clicks:
            return dash.no_update
        with _callback_lock:
            _zoom_range = None
            img = bp.last_img
            ps = bp.pixel_size
        patch = Patch()
        if img is not None:
            patch["layout"]["xaxis"]["range"] = [0, img.shape[1] * ps]
            patch["layout"]["yaxis"]["range"] = [0, img.shape[0] * ps]
        return patch

    # -- Mouse zoom and pan ---------------------------------------------------
    # No output: the zoom becomes part of the state every tick draws from.
    @app.callback(
        Input("live-graph", "relayoutData"),
        prevent_initial_call=True,
    )
    def follow_mouse_zoom(relayout: dict[str, Any] | None) -> None:
        """Adopt a zoom or pan made with the mouse, so the next frame keeps it.

        The layout's ``uirevision`` is supposed to preserve this across figure
        updates, and in Dash 4 it doesn't: right after the user zooms, the
        Graph component re-plots its own figure with the zoomed range, Plotly
        takes that as the app setting the range and forgets the user's edit,
        and the next tick's explicit full-sensor range wins. A mouse zoom
        lasted one frame. Recording it server-side makes it behave like
        Auto-fit.
        """
        global _zoom_range  # noqa: PLW0603
        if not relayout:
            return
        with _callback_lock:
            shape = bp.last_img.shape if bp.last_img is not None else None
            _zoom_range = _zoom_after_relayout(relayout, _zoom_range, shape, bp.pixel_size)

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
        """Apply an exposure change from either the slider or the box.

        Both controls end up showing the exposure the camera reports after
        the write, which is its clamped, quantised value -- or the old one,
        if the write failed.
        """
        trigger = ctx.triggered_id
        from_slider = trigger == "slider-exposure"
        val = slider_val if from_slider else input_val
        if val is None:
            return dash.no_update, dash.no_update
        actual = None
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
                    logger.warning("Failed to set exposure: %s", e)
                exposure = bp.camera.exposure_time
                # Rounded to the controls' 1 us step, so float noise in the
                # read-back doesn't count as the camera changing the value.
                actual = None if exposure is None else round(exposure * 1000.0, 3)
        return _paired_values(val, actual, from_slider=from_slider)

    # -- Gain slider + input (kept in sync) -----------------------------------
    @app.callback(
        Output("slider-gain", "value"),
        Output("input-gain", "value"),
        Input("slider-gain", "value"),
        Input("input-gain", "value"),
        prevent_initial_call=True,
    )
    def set_gain(slider_val: float | None, input_val: float | None) -> tuple[Any, Any]:
        """Apply a gain change from either the slider or the box, showing
        the gain the camera reports afterwards (see set_exposure)."""
        trigger = ctx.triggered_id
        from_slider = trigger == "slider-gain"
        val = slider_val if from_slider else input_val
        if val is None:
            return dash.no_update, dash.no_update
        actual = None
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
                    logger.warning("Failed to set gain: %s", e)
                actual = bp.camera.gain
        return _paired_values(val, actual, from_slider=from_slider)

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
        """Apply the requested region of interest and report what stuck.

        Cameras quantise ROI values to their own granularity, so the status
        line reports what the device accepted, not what was asked for.
        Stopping and restarting acquisition around the change is the
        camera's job (``set_roi`` knows whether its device needs it), and a
        rejection comes back as an exception whose message is shown as is.
        """
        if bp.camera is None:
            return "No camera"
        if ox is None or oy is None or w is None or h is None:
            return "Please enter offset/width/height"
        with _callback_lock:
            try:
                getattr(bp.camera, "set_roi")(
                    offset_x=int(ox), offset_y=int(oy), width=int(w), height=int(h)
                )
                roi = getattr(bp.camera, "roi_info")
            except Exception as e:
                logger.warning("ROI not applied: %s", e)
                return str(e) or type(e).__name__
            finally:
                # Even a rejected ROI may have been half applied (an offset
                # accepted before the width was refused), so the old frames'
                # coordinates can't be trusted either way.
                _discard_frame_history(bp)
        return f"ROI: {roi['width']}×{roi['height']} at ({roi['offset_x']},{roi['offset_y']})"

    @app.callback(
        Output("input-roi-ox", "value"),
        Output("input-roi-oy", "value"),
        Output("input-roi-w", "value"),
        Output("input-roi-h", "value"),
        Output("div-roi-status", "children", allow_duplicate=True),
        Input("btn-roi-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_roi(_n: int) -> tuple[Any, ...]:
        """Restore the full sensor and refresh the ROI boxes.

        On failure the boxes are left alone. Writing zeros into them, as this
        used to, set up the next Apply to request a 0×0 ROI.
        """
        unchanged = (dash.no_update,) * 4
        if bp.camera is None:
            return (*unchanged, "No camera")
        with _callback_lock:
            try:
                getattr(bp.camera, "set_roi")(offset_x=0, offset_y=0, width=None, height=None)
                roi = getattr(bp.camera, "roi_info")
            except Exception as e:
                logger.warning("Could not restore the full sensor: %s", e)
                return (*unchanged, str(e) or type(e).__name__)
            finally:
                _discard_frame_history(bp)
        return (
            roi["offset_x"],
            roi["offset_y"],
            roi["width"],
            roi["height"],
            "Reset to full sensor",
        )

    # -- GenICam feature callbacks (pattern-matching) -------------------------
    # Also unconditional. Pattern-matching callbacks happily target components
    # that appear later, which is exactly what happens when a camera switch
    # rebuilds the Setting panel with a different feature set.

    def _write_node(feature: str, value: Any) -> Any:
        """Write a GenICam feature and return what it holds afterwards.

        The read-back is what the controls show: a write can be refused (many
        features are locked while the camera streams) or clamped, and
        showing the request instead left the control claiming a setting the
        camera did not have. Refusals are logged as warnings; they used to go
        to DEBUG, where nobody sees them. Returns ``None`` when there is no
        such node, or it cannot be read back. The caller holds the lock.
        """
        camera = bp.camera
        nm = getattr(camera, "node_map", None)
        node = getattr(nm, feature, None) if nm is not None else None
        if camera is None or node is None:
            return None
        try:
            was_acquiring = camera.is_acquiring
            node.value = value
            if was_acquiring and not camera.is_acquiring and not _server_paused:
                camera.start_acquisition()
        except Exception as e:
            logger.warning("Camera did not accept %s = %r: %s", feature, value, e)
        try:
            return node.value
        except Exception:
            logger.debug("Could not read %s back", feature, exc_info=True)
            return None

    @app.callback(
        Output({"type": "genicam-num", "feature": MATCH}, "value"),
        Output({"type": "genicam-num-input", "feature": MATCH}, "value"),
        Input({"type": "genicam-num", "feature": MATCH}, "value"),
        Input({"type": "genicam-num-input", "feature": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def set_genicam_numeric(slider_val: float | None, input_val: float | None) -> tuple[Any, Any]:
        """Write a numeric GenICam feature from its slider or box."""
        trigger = ctx.triggered_id
        source = trigger.get("type") if isinstance(trigger, dict) else None
        value = slider_val if source == "genicam-num" else input_val
        if value is None or bp.camera is None or not isinstance(trigger, dict):
            return dash.no_update, dash.no_update
        feature = trigger.get("feature")
        if feature is None:
            return dash.no_update, dash.no_update
        with _callback_lock:
            actual = _write_node(feature, value)
        return _paired_values(value, actual, from_slider=source == "genicam-num")

    @app.callback(
        Output({"type": "genicam-sel", "feature": MATCH}, "value"),
        Input({"type": "genicam-sel", "feature": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def set_genicam_select(value: str | None) -> Any:
        """Write an enumerated GenICam feature from its dropdown."""
        if value is None or bp.camera is None:
            return dash.no_update
        feature = ctx.triggered_id["feature"]
        with _callback_lock:
            actual = _write_node(feature, value)
        return value if actual is None else str(actual)

    @app.callback(
        Output({"type": "genicam-sw", "feature": MATCH}, "value"),
        Input({"type": "genicam-sw", "feature": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def set_genicam_switch(value: bool) -> Any:
        """Write a boolean GenICam feature from its switch."""
        if bp.camera is None:
            return dash.no_update
        feature = ctx.triggered_id["feature"]
        with _callback_lock:
            actual = _write_node(feature, value)
        return value if actual is None else bool(actual)

    # -- Main update loop -----------------------------------------------------
    # Registered last: other test suites find this callback as the final one.
    @app.callback(
        Output("live-graph", "figure"),
        Output("div-results", "children"),
        Output("status-bar", "children"),
        Output("store-frame", "data"),
        # The tick can stop the stream (a failing camera) or find it stopped
        # from elsewhere (another tab, a camera switch), and the button has
        # to say so, or its first click would only repeat the pause.
        Output("store-paused", "data", allow_duplicate=True),
        Output("btn-play-pause", "children", allow_duplicate=True),
        Output("btn-play-pause", "color", allow_duplicate=True),
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
        prevent_initial_call="initial_duplicate",
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
        """Grab a frame, fit it, and redraw — once per interval tick.

        Almost every control on the page arrives here as ``State`` rather than
        ``Input``: they should change what the *next* frame looks like, not
        force an extra redraw of their own.

        The tick is skipped rather than queued if the previous one is still
        running. Queueing would let a camera slower than the interval build an
        unbounded backlog of stale frames, and would leave the controls (which
        share the lock) waiting behind all of them.
        """
        global _camera_failures, _server_paused  # noqa: PLW0603
        # Outputs: figure, results, status, frame count, pause flag, and the
        # Play/Pause button's children and colour.
        nothing = (dash.no_update,) * 7

        def show_status(status: Any) -> tuple[Any, ...]:
            return (dash.no_update, dash.no_update, status, *(dash.no_update,) * 4)

        def show_paused(status: Any = dash.no_update) -> tuple[Any, ...]:
            children, color = _play_pause_face(True)
            return (dash.no_update, dash.no_update, status, dash.no_update, True, children, color)

        if paused:
            return nothing
        if _server_paused:
            # Stopped by something this page didn't see: bring its button
            # into line.
            return show_paused()

        if not _callback_lock.acquire(blocking=False):
            # A previous tick is still running; skip this one and let the
            # next interval fire. Keeps the UI responsive when a camera
            # fetch takes longer than the tick interval.
            return nothing

        try:
            if _server_paused:
                # A Pause or a camera switch completed between the check
                # above and taking the lock. Fetching now would restart the
                # acquisition it just stopped: HarvesterCamera.get_image
                # starts a stopped stream by itself.
                return show_paused()

            # Either change starts the fits from scratch. A new fit method
            # fits different data, so the old warm start is meaningless; a
            # model-free definition (FWHM, D4σ) skips the 2D fit and the
            # linecut altogether, so without the reset their last results
            # stayed on screen, frozen, as the ellipse and the crosshair.
            if analysis and bp.fit_method != analysis:
                bp.fit_method = analysis
                bp.reset_analysis()
            if definition and bp.definition != definition:
                bp.definition = definition
                bp.reset_analysis()

            if bp._mode == "camera" and bp.camera is not None:
                # Cap fetch at one tick so sliders/buttons (which share
                # ``_callback_lock``) never wait more than ~100 ms. During
                # multi-second exposures this just times out repeatedly
                # until the producer delivers the next frame.
                try:
                    img = bp.camera.get_image(timeout=0.1)
                except TimeoutError:
                    return nothing
                except Exception as exc:
                    # Anything else is the device failing, or gone.
                    _camera_failures += 1
                    message = f"Camera error: {str(exc) or type(exc).__name__}"
                    _log_tick_error(message, exc)
                    if _camera_failures < _MAX_CAMERA_FAILURES:
                        return show_status(_error_status(message))
                    logger.warning(
                        "Pausing the stream after %d camera errors in a row", _camera_failures
                    )
                    _server_paused = True
                    _recent_frame_times.clear()
                    try:
                        bp.camera.stop_acquisition()
                    except Exception:
                        logger.debug("Stopping a failed camera also failed", exc_info=True)
                    return show_paused(_error_status(f"{message} - paused; press Play to retry"))
            else:
                img = bp.last_img

            if img is None:
                return nothing
            _camera_failures = 0

            raw = img
            img = _averaged_image(img, avg_n or 1)
            bp.last_img = img
            popt_x, popt_y = bp.analyze(img)

            cs = cs_name if color_on else GRAY_COLORSCALE
            zmin = None if auto_range else (zmin_val if zmin_val is not None else 0)
            zmax_default = _saturation_max(raw, _camera_bit_depth(bp))
            zmax = None if auto_range else (zmax_val if zmax_val is not None else zmax_default)
            # Read the live source of truth (mutated by Auto-fit, Reset and
            # mouse zooms under ``_callback_lock``) instead of capturing it as
            # Dash ``State``: a State snapshot can be 50–100 ms stale if a
            # zoom click fires after this tick started, which would cause a
            # one-frame blink to the previous zoom.
            xrange, yrange = _zoom_in_um(bp)
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
                _build_status(bp, img, frame_count, raw=raw),
                frame_count,
                *(dash.no_update,) * 3,
            )
        except Exception as exc:
            message = f"Update error: {str(exc) or type(exc).__name__}"
            _log_tick_error(message, exc)
            return show_status(_error_status(message))
        finally:
            _callback_lock.release()
