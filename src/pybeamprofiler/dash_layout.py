"""What the pyBeamprofiler page looks like.

Everything here builds Dash components: the two side-panel tabs, the camera
selector, and the widgets generated from a camera's GenICam feature set. None
of it reacts to anything -- the callbacks that do live in
:mod:`pybeamprofiler.dash_app`, which imports these builders.

The split is worth having because the two halves change for different
reasons: layout changes when the UI is redesigned, callbacks when behaviour
does, and they were previously interleaved across two thousand lines.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from .constants import MAX_AVG_FRAMES
from .discovery import CameraOption, describe_open_camera, discover_cameras

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

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _camera_options(bp: BeamProfiler) -> tuple[list[CameraOption], str]:
    """Discover selectable cameras and the key of the one already open.

    The open camera is folded in even when discovery misses it — it can be
    opened from an explicit ``.cti`` path on a machine where the standard
    search finds nothing — so the dropdown always has something selected
    rather than showing a blank box over a running stream.

    Returns:
        ``(options, current_key)``.
    """
    options = discover_cameras()
    current = ""
    if bp.camera is not None:
        open_option = describe_open_camera(bp.camera)
        current = open_option.key
        if not any(o.key == current for o in options):
            options.insert(0, open_option)
    return options, current


def _camera_controls(options: list[CameraOption], current: str) -> Any:
    """Build the camera selector: a dropdown plus a rescan button.

    Args:
        options: Cameras to offer, from :func:`_camera_options`.
        current: Key of the one already open.
    """
    return dbc.Row(
        [
            dbc.Col(
                dbc.Select(
                    id="dropdown-camera",
                    options=[{"label": o.label, "value": o.key} for o in options],
                    value=current,
                    size="sm",
                ),
            ),
            dbc.Col(
                dbc.Button(
                    html.I(className="bi bi-arrow-clockwise"),
                    id="btn-camera-refresh",
                    color="secondary",
                    size="sm",
                    title="Rescan for connected cameras",
                ),
                width="auto",
                className="ps-1",
            ),
        ],
        className="mb-2 g-0",
    )


def _fitting_tab(bp: BeamProfiler, options: list[CameraOption], current: str) -> dbc.Tab:
    """Build the **Fitting** tab content.

    Args:
        bp: The profiler, read for its current fit method and pixel size.
        options: Cameras to offer in the selector.
        current: Key of the camera already open.
    """
    return dbc.Tab(
        label="Fitting",
        tab_id="tab-fitting",
        children=dbc.Card(
            dbc.CardBody(
                [
                    # Row 0 — which camera to stream from.
                    dbc.Label("Camera", className="small mb-1"),
                    _camera_controls(options, current),
                    html.Div(id="div-camera-status", className="small text-muted mb-2"),
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
                                            max=MAX_AVG_FRAMES,
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
