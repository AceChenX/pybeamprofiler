"""Dash web application for pyBeamprofiler GUI.

Provides a rich browser-based interface with live camera streaming,
interactive fitting controls, and camera settings management.
"""

from __future__ import annotations

import base64
import collections
import io
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import MATCH, Input, Output, Patch, State, ctx, dcc, html
from PIL import Image
from scipy.ndimage import zoom as _ndimage_zoom

from .constants import (
    DEFAULT_UPDATE_INTERVAL_MS,
    MAX_DISPLAY_DIM,
)

# Optional: only present when a real GenTL backend is installed. Used by
# ``_is_readonly`` to map GenICam access modes to our read-only flag.
try:
    from genicam.genapi import EAccessMode as _EAccessMode  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - exercised only on non-GenICam envs
    _EAccessMode = None  # ty: ignore[invalid-assignment]

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

# Saturation warning thresholds.
_SATURATION_PIXEL_FRACTION = 0.001  # 0.1% of pixels at the dtype max ⇒ warn.

# Maximum frames in the rolling-average buffer (memory cap: N×frame size).
_MAX_AVG_FRAMES = 32


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
    if np.issubdtype(image.dtype, np.integer):
        return float(np.count_nonzero(image >= sat)) / image.size
    return float(np.count_nonzero(image >= sat - 1e-6)) / image.size


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
                    # Row 1 — Play / Pause (full-width) with Spacebar shortcut hint.
                    dbc.Row(
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="bi bi-pause-fill me-1"), "Pause"],
                                id="btn-play-pause",
                                color="primary",
                                size="sm",
                                className="w-100",
                                title="Toggle Play / Pause (Spacebar)",
                            ),
                        ),
                        className="mb-2",
                    ),
                    # Row 1b — view + save controls.
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-aspect-ratio me-1"), "Auto-fit"],
                                    id="btn-zoom-fit",
                                    color="secondary",
                                    size="sm",
                                    className="w-100",
                                    title="Zoom to ±3σ around the fit center",
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-arrows-fullscreen me-1"), "Reset"],
                                    id="btn-zoom-reset",
                                    color="secondary",
                                    size="sm",
                                    className="w-100",
                                    title="Reset zoom to full sensor",
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                dbc.DropdownMenu(
                                    [
                                        dbc.DropdownMenuItem(
                                            [html.I(className="bi bi-image me-2"), "PNG"],
                                            id="btn-save-png",
                                        ),
                                        dbc.DropdownMenuItem(
                                            [
                                                html.I(className="bi bi-filetype-raw me-2"),
                                                "NumPy (.npy)",
                                            ],
                                            id="btn-save-npy",
                                        ),
                                    ],
                                    label=[html.I(className="bi bi-download me-1"), "Save"],
                                    color="secondary",
                                    size="sm",
                                    # ``className`` only styles the outer
                                    # ``.dropdown`` wrapper; the toggle <button>
                                    # stays content-sized unless we target it
                                    # explicitly.
                                    className="w-100",
                                    toggle_class_name="w-100",
                                    align_end=True,
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3 g-1",
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
                    # Row 5b — Frame averaging (running mean over N frames).
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText(
                                            "Average",
                                            style={"fontSize": "0.8rem"},
                                        ),
                                        dbc.Input(
                                            id="input-avg-n",
                                            type="number",
                                            value=1,
                                            min=1,
                                            max=_MAX_AVG_FRAMES,
                                            step=1,
                                            size="sm",
                                        ),
                                        dbc.InputGroupText("frames", style={"fontSize": "0.8rem"}),
                                    ],
                                    size="sm",
                                ),
                                title="Running mean over the last N frames (1 = off)",
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


def _is_readonly(node: Any) -> bool:
    """Return ``True`` if the GenICam node is read-only."""
    if getattr(node, "_readonly", False):
        return True
    if _EAccessMode is None:
        return False
    try:
        access = getattr(node, "get_access_mode", None)
        if access is not None:
            mode = access()
            if mode in (_EAccessMode.RO, _EAccessMode.NA, _EAccessMode.NI):
                return True
    except Exception:
        pass
    return False


def _humanize(name: str) -> str:
    """``"AcquisitionFrameRate"`` → ``"Acquisition Frame Rate"``."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def _slider_with_input(
    *,
    slider_id: Any,
    input_id: Any,
    min_val: float,
    max_val: float,
    value: float,
    step: float,
    input_width: str = "110px",
) -> html.Div:
    """Render a slider paired with a narrow numeric input box.

    The slider is for quick drag-to-set, the input on the right lets the
    user type a precise value (debounced — only fires on Enter / blur).
    Both are kept in sync by a matching callback elsewhere; this helper
    just lays them out consistently.
    """
    return html.Div(
        dbc.Row(
            [
                dbc.Col(
                    dcc.Slider(
                        id=slider_id,
                        min=min_val,
                        max=max_val,
                        value=value,
                        step=step,
                        tooltip={"placement": "bottom", "always_visible": False},
                        marks=None,
                    ),
                    className="pe-2",
                ),
                dbc.Col(
                    dbc.Input(
                        id=input_id,
                        type="number",
                        value=value,
                        min=min_val,
                        max=max_val,
                        step=step,
                        size="sm",
                        debounce=True,
                    ),
                    width="auto",
                    style={"width": input_width},
                ),
            ],
            className="g-0 align-items-center",
        ),
        className="mb-2",
    )


def _build_genicam_control(cam: Any, feature_name: str) -> html.Div | None:
    """Build a Dash control for a single GenICam ``node_map`` feature.

    Returns a ``html.Div`` wrapping a label and the appropriate input widget
    (slider for numeric, select for enums, switch for booleans), styled
    consistently with the built-in Exposure & Gain controls.  Returns
    ``None`` when the feature cannot be rendered.
    """
    node_map = getattr(cam, "node_map", None)
    if node_map is None:
        return None

    node = getattr(node_map, feature_name, None)
    if node is None:
        return None

    try:
        current_val = node.value
    except Exception:
        return None

    readonly = _is_readonly(node)
    label = dbc.Label(_humanize(feature_name), className="small mb-1")

    def _readonly_row(value_text: str) -> html.Div:
        """Render a label/value pair as a flex row so the value is
        right-aligned with a clear gap from the label (avoids text
        running together when both fit on one line)."""
        return html.Div(
            [
                dbc.Label(_humanize(feature_name), className="small text-muted mb-0 me-2"),
                html.Span(
                    value_text,
                    className="small text-end",
                    style={"wordBreak": "break-word"},
                ),
            ],
            className="mb-2 d-flex justify-content-between align-items-baseline",
        )

    # ── Boolean ───────────────────────────────────────────────────
    if isinstance(current_val, bool):
        return html.Div(
            [
                label,
                dbc.Switch(
                    id={"type": "genicam-sw", "feature": feature_name},
                    value=current_val,
                    label="",
                    disabled=readonly,
                ),
            ],
            className="mb-2",
        )

    # ── Enumeration (has symbolics) ───────────────────────────────
    symbolics = getattr(node, "symbolics", None)
    if symbolics:
        val_str = str(current_val)
        return html.Div(
            [
                label,
                dbc.Select(
                    id={"type": "genicam-sel", "feature": feature_name},
                    options=[{"label": s, "value": s} for s in symbolics],
                    value=val_str if val_str in symbolics else symbolics[0],
                    size="sm",
                    disabled=readonly,
                ),
            ],
            className="mb-2",
        )

    # ── Numeric with range ────────────────────────────────────────
    node_min = getattr(node, "min", None)
    node_max = getattr(node, "max", None)
    if node_min is not None and node_max is not None:
        try:
            fmin, fmax, fval = float(node_min), float(node_max), float(current_val)
            if readonly:
                return _readonly_row(f"{fval:g}")
            is_int = isinstance(current_val, int) and isinstance(node_min, int)
            step = 1 if is_int else round((fmax - fmin) / 1000, 6) or 0.001
            return html.Div(
                [
                    label,
                    _slider_with_input(
                        slider_id={"type": "genicam-num", "feature": feature_name},
                        input_id={"type": "genicam-num-input", "feature": feature_name},
                        min_val=fmin,
                        max_val=fmax,
                        value=fval,
                        step=step,
                    ),
                ],
                className="mb-2",
            )
        except (TypeError, ValueError):
            pass

    # ── Read-only / unknown string → text display ─────────────────
    if readonly or (isinstance(current_val, str) and not current_val):
        return _readonly_row(str(current_val))

    # ── String fallback (Enable / Auto without symbolics) ─────────
    if isinstance(current_val, str):
        if feature_name.endswith("Enable"):
            opts = [{"label": s, "value": s} for s in ["On", "Off"]]
        elif feature_name.endswith("Auto"):
            opts = [{"label": s, "value": s} for s in ["Off", "Once", "Continuous"]]
        else:
            opts = [{"label": current_val, "value": current_val}]
        return html.Div(
            [
                label,
                dbc.Select(
                    id={"type": "genicam-sel", "feature": feature_name},
                    options=opts,  # ty: ignore[invalid-argument-type]
                    value=current_val,
                    size="sm",
                ),
            ],
            className="mb-2",
        )

    return None


# ---------------------------------------------------------------------------
# Dedicated control builders (Exposure / Gain / ROI)
# ---------------------------------------------------------------------------


def _exposure_controls(cam: Any) -> list[Any]:
    """Build the Exposure (ms) slider + numeric input — always present."""
    exp_min, exp_max = 0.001, 1.0
    er = getattr(cam, "exposure_range", None)
    if er is not None:
        try:
            exp_min, exp_max = er
        except Exception:
            pass
    exp_ms = (cam.exposure_time or 0.01) * 1000
    return [
        dbc.Label("Exposure (ms)", className="small mb-1"),
        _slider_with_input(
            slider_id="slider-exposure",
            input_id="input-exposure",
            min_val=round(exp_min * 1000, 3),
            max_val=round(exp_max * 1000, 3),
            value=round(exp_ms, 3),
            step=0.001,
        ),
    ]


def _gain_controls(cam: Any) -> list[Any]:
    """Build the Gain slider + numeric input — always present."""
    gain_min, gain_max = 0.0, 24.0
    gr = getattr(cam, "gain_range", None)
    if gr is not None:
        try:
            gain_min, gain_max = gr
        except Exception:
            pass
    return [
        dbc.Label("Gain", className="small mb-1"),
        _slider_with_input(
            slider_id="slider-gain",
            input_id="input-gain",
            min_val=gain_min,
            max_val=gain_max,
            value=cam.gain or 0.0,
            step=0.1,
        ),
    ]


def _roi_controls(cam: Any) -> list[Any] | None:
    """Build the ROI panel, or ``None`` if the camera has no ROI."""
    if not (hasattr(cam, "roi_info") and hasattr(cam, "set_roi")):
        return None
    try:
        roi: dict[str, int] = getattr(cam, "roi_info")
    except Exception:
        return None

    return [
        html.Hr(className="my-2"),
        dbc.Label("Region of Interest", className="small fw-bold mb-1"),
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
    ]


# ---------------------------------------------------------------------------
# Main settings builder
# ---------------------------------------------------------------------------

# Categories that always appear (in this order) because they contain
# dedicated Exposure / Gain / ROI controls even when no GenICam features
# are discovered.
_PINNED_CATEGORIES = ["Acquisition Control", "Analog Control", "Image Format Control"]


def _build_setting_items(bp: BeamProfiler) -> list[dbc.AccordionItem]:
    """Build the full Setting accordion driven by GenICam feature discovery.

    The layout mirrors official camera GUIs (SpinView, pylon Viewer):

    * **Camera Info** — device metadata summary
    * **Acquisition Control** — Exposure slider + discovered features
    * **Analog Control** — Gain slider + discovered features
    * **Image Format Control** — ROI panel + discovered features
    * *remaining categories* — auto-discovered features only
    """
    items: list[dbc.AccordionItem] = []
    cam = bp.camera
    if cam is None:
        return items

    # ── Camera Info (always first) ────────────────────────────────
    info_rows: list[Any] = []
    for lbl, attr in [
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
            }.get(lbl)
            if node_attr and hasattr(nm, node_attr):
                try:
                    val = getattr(nm, node_attr).value
                except Exception:
                    pass
        if val:
            info_rows.append(
                html.Tr([html.Td(lbl, className="pe-3 text-muted"), html.Td(str(val))])
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

    # ── Discover features ─────────────────────────────────────────
    discovered: dict[str, list[str]] = {}
    if hasattr(cam, "_discover_features"):
        try:
            discovered = cam._discover_features()
        except Exception:
            logger.warning("Feature discovery failed", exc_info=True)
    logger.debug(
        "Setting panel: discovered %d features in %d categories: %s",
        sum(len(v) for v in discovered.values()),
        len(discovered),
        list(discovered.keys()),
    )

    # Build ordered category list: pinned first, then the rest alphabetically
    category_order: list[str] = list(_PINNED_CATEGORIES)
    for cat in sorted(discovered):
        if cat not in category_order:
            category_order.append(cat)

    # ── Build one accordion item per category ─────────────────────
    for cat in category_order:
        children: list[Any] = []

        # Inject dedicated controls into the appropriate category
        if cat == "Acquisition Control":
            children.extend(_exposure_controls(cam))
        elif cat == "Analog Control":
            children.extend(_gain_controls(cam))

        # Auto-discovered features for this category
        for fname in discovered.get(cat, []):
            ctrl = _build_genicam_control(cam, fname)
            if ctrl is not None:
                children.append(ctrl)

        # ROI panel at the bottom of Image Format Control
        if cat == "Image Format Control":
            roi = _roi_controls(cam)
            if roi is not None:
                children.extend(roi)

        if children:
            items.append(dbc.AccordionItem(children, title=cat))

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
        children=dbc.Card(
            dbc.CardBody(html.Div(body, id="settings-container")),
            className="border-0",
        ),
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
    cached_proj_x = getattr(bp, "_last_proj_x", None)
    proj_x = (cached_proj_x if cached_proj_x is not None else np.sum(image, axis=0)).astype(float)
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
        fit_x = bp.gaussian(x_ax, *popt_x).astype(float)
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
    y_ax = np.arange(h)
    cached_proj_y = getattr(bp, "_last_proj_y", None)
    proj_y = (cached_proj_y if cached_proj_y is not None else np.sum(image, axis=1)).astype(float)
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
        fit_y = bp.gaussian(y_ax, *popt_y).astype(float)
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
            range=yrange if yrange is not None else [0, y_max],
            showgrid=False,
            title="Y (μm)",
            title_font_size=11,
        ),
        xaxis=dict(
            constrain="domain",
            range=xrange if xrange is not None else [0, x_max],
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
        # Don't attach Dash's own StreamHandler; the parent CLI manages logging
        # (otherwise the "Dash is running on..." banner appears twice -- once
        # from Dash's handler and once propagated to the root logger).
        add_log_handler=False,
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

    n = max(1, min(int(n), _MAX_AVG_FRAMES))
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
    return mean.astype(image.dtype)


def _register_callbacks(app: dash.Dash, bp: BeamProfiler) -> None:
    """Wire up all Dash callbacks."""
    global _server_paused, _zoom_range  # noqa: PLW0603
    _server_paused = False
    _zoom_range = None

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
        if val is not None and val > 0:
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
                    return f"ROI: {roi['width']}×{roi['height']} at ({roi['offset_x']},{roi['offset_y']})"
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
    has_genicam = (
        bp.camera is not None
        and hasattr(bp.camera, "_discover_features")
        and bp.camera._discover_features()
    )
    if has_genicam:

        @app.callback(
            Output({"type": "genicam-num", "feature": MATCH}, "value"),
            Output({"type": "genicam-num-input", "feature": MATCH}, "value"),
            Input({"type": "genicam-num", "feature": MATCH}, "value"),
            Input({"type": "genicam-num-input", "feature": MATCH}, "value"),
            prevent_initial_call=True,
        )
        def set_genicam_numeric(
            slider_val: float | None, input_val: float | None
        ) -> tuple[Any, Any]:
            trigger = ctx.triggered_id or {}
            source = trigger.get("type") if isinstance(trigger, dict) else None
            value = slider_val if source == "genicam-num" else input_val
            if value is None or bp.camera is None:
                return dash.no_update, dash.no_update
            feature = trigger["feature"]
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
