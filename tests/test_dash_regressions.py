"""Regression tests for defects found reviewing the Dash GUI.

One class per defect, each saying what used to go wrong, in the style of
``test_regressions.py``. The callbacks are driven directly, as in the rest of
the Dash suite; the few behaviours that only a browser can show (the page
served on reload, mouse zoom) are covered here at the level the server
controls, and were checked end to end in headless Chrome as well.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import dash
import numpy as np
import plotly.graph_objs as go
import pytest
from dash import html
from dash.development.base_component import Component

from pybeamprofiler import dash_app, dash_layout, discovery
from pybeamprofiler.beamprofiler import BeamProfiler
from pybeamprofiler.simulated import SimulatedCamera, profile_for


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


def _registered(app: dash.Dash, name: str) -> Any:
    """The undecorated function behind one of a real app's callbacks."""
    for spec in app.callback_map.values():
        fn = spec.get("callback")
        if fn is not None and getattr(fn, "__wrapped__", fn).__name__ == name:
            return fn.__wrapped__
    raise KeyError(name)


def _by_id(page: Any) -> dict[str, Any]:
    """Every component in *page* that has a plain string id, keyed by it."""
    found: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, Component):
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                found[cid] = node
            walk(getattr(node, "children", None))
            walk(getattr(node, "label", None))
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(page)
    return found


def _page_load(app: dash.Dash) -> Any:
    """What Dash serves for one page load, whether the layout is a tree or a
    function that builds one."""
    return app._layout_value()


def _tick(
    cbs: dict[str, Any],
    *,
    analysis: str = "1d",
    definition: str = "gaussian",
    avg_n: int = 1,
    paused: bool = False,
    color_on: bool = True,
    auto_range: bool = True,
    zmin: float | None = None,
    zmax: float | None = None,
) -> tuple[Any, ...]:
    """One render tick, with the State bundle the browser would send."""
    return cbs["update_live"](
        1, paused, color_on, "Hot", auto_range, zmin, zmax, 0, analysis, definition, True, avg_n
    )


def _profiler(profile: str = "sim-1", seed: int = 11) -> BeamProfiler:
    """A profiler streaming from a seeded simulator, so beam shapes repeat."""
    bp = BeamProfiler(camera="simulated")
    camera = SimulatedCamera(profile_for(profile), seed=seed)
    camera.open()
    bp.attach_camera(camera)
    camera.start_acquisition()
    return bp


@pytest.fixture(autouse=True)
def _quiet_fits() -> Iterator[None]:
    """Cold fits on cropped frames warn about covariance; that is noise here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class TestRoiChangeForgetsTheOldFrames:
    """Applying an ROI used to leave every frame-derived value in place.

    The ROI moves the frame's origin, so the fitter's warm start pointed
    outside the new frame. In 2D mode every ROI tried -- including one
    centred on the beam -- left the reported centre stuck outside the frame
    for as long as anyone watched; in 1D a 200x200 window pinned it at about
    3e10 px. The averaging buffer blended two windows into a ghost beam, and
    the zoom box framed the wrong region.
    """

    @staticmethod
    def _primed(method: str = "2d") -> tuple[BeamProfiler, dict[str, Any]]:
        bp = _profiler()
        cbs = _callbacks(bp)
        for _ in range(3):
            _tick(cbs, analysis=method, avg_n=4)
        cbs["auto_fit_zoom"](1)
        assert bp._last_popt_x is not None
        assert len(dash_app._avg_buffer) == 3
        assert dash_app._zoom_range is not None
        return bp, cbs

    @pytest.mark.parametrize("callback", ["apply_roi", "reset_roi"])
    def test_every_frame_derived_value_is_dropped(self, callback):
        bp, cbs = self._primed()
        if callback == "apply_roi":
            cbs["apply_roi"](1, 412, 412, 200, 200)
        else:
            cbs["reset_roi"](1)

        assert bp._last_popt_x is None
        assert bp._last_popt_y is None
        assert bp._last_popt_2d is None
        assert len(dash_app._avg_buffer) == 0
        assert dash_app._zoom_range is None
        assert len(dash_app._recent_frame_times) == 0

    @pytest.mark.parametrize("method", ["1d", "2d"])
    def test_the_fit_follows_the_beam_into_the_new_window(self, method):
        bp, cbs = self._primed(method)
        # A 200x200 window with its corner at (412, 412): the beam, jittering
        # around (512, 512) on the sensor, sits near (100, 100) in it.
        assert "200×200" in cbs["apply_roi"](1, 412, 412, 200, 200)

        centres = []
        for _ in range(10):
            _tick(cbs, analysis=method)
            centres.append((bp.center_x, bp.center_y))

        assert bp.last_img is not None and bp.last_img.shape == (200, 200)
        for cx, cy in centres:
            assert abs(cx - 100) < 60 and abs(cy - 100) < 60, centres

    def test_a_same_size_move_does_not_blend_the_two_windows(self):
        bp = _profiler(seed=5)
        cbs = _callbacks(bp)
        cbs["apply_roi"](1, 312, 312, 400, 400)  # a window on the beam
        for _ in range(8):
            _tick(cbs, avg_n=8)
        assert bp.last_img is not None and bp.last_img.max() > 150

        cbs["apply_roi"](1, 0, 0, 400, 400)  # same shape, an empty corner
        _tick(cbs, avg_n=8)

        # Background plus noise only. The blend used to show a beam at 175.
        assert bp.last_img.max() < 100

    def test_the_camera_does_its_own_stop_and_restart(self):
        """``set_roi`` knows whether the device needs acquisition stopped;
        a second stop/start from the GUI would only restart it twice."""
        bp = _profiler()
        cbs = _callbacks(bp)
        assert bp.camera is not None
        with (
            patch.object(bp.camera, "stop_acquisition") as stop,
            patch.object(bp.camera, "start_acquisition") as start,
        ):
            cbs["apply_roi"](1, 0, 0, 512, 512)
            cbs["reset_roi"](1)
        stop.assert_not_called()
        start.assert_not_called()

    def test_a_rejected_roi_shows_the_cameras_reason(self):
        bp = _profiler()
        cbs = _callbacks(bp)
        assert bp.camera is not None
        reason = "Width 333 is not a multiple of the 16 px increment"
        with patch.object(bp.camera, "set_roi", side_effect=ValueError(reason)):
            assert cbs["apply_roi"](1, 0, 0, 333, 333) == reason
            boxes_and_status = cbs["reset_roi"](1)

        # The boxes keep what the user typed rather than being zeroed, which
        # used to turn the next Apply into a request for a 0x0 ROI.
        assert all(v is dash.no_update for v in boxes_and_status[:4])
        assert boxes_and_status[4] == reason
        # Half-applied or not, the old frames are no longer trusted.
        assert bp._last_popt_x is None


class TestTheCameraListIsCachedForSwitching:
    """The cache the camera switch resolves against was always empty.

    ``create_app`` filled it and ``_register_callbacks`` emptied it again
    straight afterwards, and the rescan button never refilled it. Every
    switch therefore fell back to a full GenTL enumeration with the lock
    held -- seconds of frozen stream on a GigE setup, the stall an earlier
    commit had fixed.
    """

    def test_the_startup_scan_survives_create_app(self):
        bp = BeamProfiler(camera="simulated")
        dash_app.create_app(bp)
        expected = [o.key for o in dash_layout._camera_options(bp)[0]]
        assert [o.key for o in dash_app._known_options] == expected

    def test_a_switch_resolves_without_rescanning(self):
        bp = BeamProfiler(camera="simulated")
        app = dash_app.create_app(bp)
        assert bp.camera is not None
        current = discovery.describe_open_camera(bp.camera).key
        target = next(o.key for o in dash_app._known_options if o.key != current)

        with patch.object(dash_layout, "discover_cameras", side_effect=AssertionError("rescan")):
            status = _registered(app, "switch_camera")(target)[0]

        assert "ready" in status

    def test_a_rescan_refreshes_what_a_switch_can_find(self):
        bp = BeamProfiler(camera="simulated")
        app = dash_app.create_app(bp)
        new = discovery.CameraOption(key="genicam:4242", label="Acme 4242", kind="genicam")
        found = [new, *discovery.simulated_options()]
        with patch.object(dash_layout, "discover_cameras", return_value=found):
            _registered(app, "refresh_cameras")(1)
        assert new in dash_app._known_options

        opened = SimulatedCamera(profile_for("sim-2"))
        opened.open()
        with (
            patch.object(dash_layout, "discover_cameras", side_effect=AssertionError("rescan")),
            patch.object(dash_app, "open_camera", return_value=opened) as open_camera,
        ):
            _registered(app, "switch_camera")(new.key)
        open_camera.assert_called_once_with(new)


class TestANewAppStartsFromScratch:
    """Building a second app in one process kept the first one's frames.

    ``_register_callbacks`` reset the pause flag and the zoom but not the
    averaging buffer or the fps window, so a relaunched GUI averaged its
    first frames with the previous session's and reported a frame rate
    measured across the gap.
    """

    def test_the_previous_sessions_frames_are_gone(self):
        dash_app._averaged_image(np.zeros((4, 4), dtype=np.uint8), 4)
        dash_app._recent_frame_times.extend([1.0, 2.0])

        _callbacks(BeamProfiler(camera="simulated"))

        assert len(dash_app._avg_buffer) == 0
        assert dash_app._avg_running_sum is None
        assert len(dash_app._recent_frame_times) == 0


class TestAPageLoadShowsWhatIsInForce:
    """The page used to be one component tree, built at start-up.

    Dash served that same tree on every load, so after a camera switch a
    reloaded page (or a second tab) named the old camera, showed its pixel
    pitch in the Scale box and offered Pause on a stopped stream. Checked in
    Chrome: merely clicking into the Scale box and out again wrote the stale
    5.0 um/px over the 3.45 um/px camera, inflating every width by 45%.
    """

    @staticmethod
    def _switched() -> tuple[BeamProfiler, dash.Dash, str]:
        bp = BeamProfiler(camera="simulated")
        app = dash_app.create_app(bp)
        target = f"{discovery.SIMULATED_PREFIX}sim-2"
        assert "ready" in _registered(app, "switch_camera")(target)[0]
        return bp, app, target

    def test_the_page_follows_a_camera_switch(self):
        bp, app, target = self._switched()
        page = _by_id(_page_load(app))

        assert page["dropdown-camera"].value == target
        assert page["input-pixel-scale"].value == 3.45
        assert page["store-paused"].data is True
        assert "Play" in str(page["btn-play-pause"].children)
        assert page["input-roi-w"].value == 1280  # the new sensor's panel

    def test_leaving_the_scale_box_keeps_the_real_pitch(self):
        bp, app, _ = self._switched()
        page = _by_id(_page_load(app))

        # What the browser sends on blur: whatever the box shows.
        _registered(app, "set_pixel_scale")(None, 1, page["input-pixel-scale"].value)

        assert bp.pixel_size == pytest.approx(3.45)

    def test_analysis_settings_come_from_the_profiler(self):
        bp = BeamProfiler(camera="simulated")
        app = dash_app.create_app(bp)
        bp.fit_method, bp.definition = "2d", "d4s"

        page = _by_id(_page_load(app))

        assert page["dropdown-analysis"].value == "2d"
        assert page["dropdown-definition"].value == "d4s"

    def test_a_paused_page_shows_the_last_frame_and_its_numbers(self):
        bp = BeamProfiler(camera="simulated")
        app = dash_app.create_app(bp)
        _registered(app, "update_live")(
            1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True, 1
        )
        _registered(app, "toggle_pause")(1, False)
        assert bp.last_img is not None

        page = _by_id(_page_load(app))

        heatmap = page["live-graph"].figure.data[0]
        assert np.array_equal(heatmap.z, bp.last_img)
        assert "μm" in str(page["div-results"].children)
        assert page["store-paused"].data is True


def _ellipse_inside_view(fig: Any) -> float:
    """Fraction of the drawn beam ellipse that lies inside the axis ranges."""
    trace = next(t for t in fig.data if t.type == "scatter" and t.line.dash == "dash")
    x, y = np.asarray(trace.x), np.asarray(trace.y)
    (x0, x1), (y0, y1) = fig.layout.xaxis.range, fig.layout.yaxis.range
    return float(np.mean((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)))


class TestAMouseZoomSurvivesTheNextFrame:
    """A box zoom or pan with the mouse lasted exactly one frame.

    The figure's ``uirevision`` is meant to preserve it, but in Dash 4 the
    Graph re-plots its own figure with the zoomed range as soon as the user
    lets go. Plotly takes that as the app setting the range, drops its record
    of the user's edit (``_preGUI``), and the next tick's explicit
    full-sensor range wins. Seen in Chrome: zoomed at frame 9, full sensor
    again by frame 11. The zoom now becomes server state, like Auto-fit.
    """

    @staticmethod
    def _streaming() -> tuple[BeamProfiler, dict[str, Any]]:
        bp = _profiler()
        cbs = _callbacks(bp)
        _tick(cbs)
        return bp, cbs

    def test_a_box_zoom_is_kept_by_the_following_frames(self):
        bp, cbs = self._streaming()
        cbs["follow_mouse_zoom"](
            {
                "xaxis.range[0]": 1000.0,
                "xaxis.range[1]": 2000.0,
                "yaxis.range[0]": 1500.0,
                "yaxis.range[1]": 2500.0,
            }
        )
        for _ in range(3):
            fig = _tick(cbs)[0]
            assert list(fig.layout.xaxis.range) == pytest.approx([1000.0, 2000.0])
            assert list(fig.layout.yaxis.range) == pytest.approx([1500.0, 2500.0])

    def test_a_drag_along_one_axis_keeps_the_other(self):
        bp, cbs = self._streaming()
        cbs["follow_mouse_zoom"]({"xaxis.range": [1000.0, 2000.0]})
        fig = _tick(cbs)[0]
        assert list(fig.layout.xaxis.range) == pytest.approx([1000.0, 2000.0])
        assert list(fig.layout.yaxis.range) == pytest.approx([0.0, 1024 * bp.pixel_size])

    @pytest.mark.parametrize(
        "event",
        [
            {"xaxis.autorange": True, "yaxis.autorange": True},
            # What a double-click sends: the full extent as explicit ranges.
            {
                "xaxis.range[0]": 0.0,
                "xaxis.range[1]": 5120.0,
                "yaxis.range[0]": 0.0,
                "yaxis.range[1]": 5120.0,
            },
        ],
    )
    def test_autoscale_and_double_click_return_to_the_full_sensor(self, event):
        bp, cbs = self._streaming()
        cbs["follow_mouse_zoom"]({"xaxis.range": [1000.0, 2000.0]})
        cbs["follow_mouse_zoom"](event)
        assert dash_app._zoom_range is None

    @pytest.mark.parametrize("event", [None, {}, {"autosize": True}, {"dragmode": "pan"}])
    def test_events_that_move_no_axis_change_nothing(self, event):
        bp, cbs = self._streaming()
        cbs["follow_mouse_zoom"]({"xaxis.range": [1000.0, 2000.0]})
        before = dash_app._zoom_range
        cbs["follow_mouse_zoom"](event)
        assert dash_app._zoom_range == before


class TestTheZoomStaysOnTheSamePixels:
    """The zoom box used to be stored in micrometres.

    Correcting the pixel scale while zoomed kept the box's micrometre values
    while the image's extent changed, so it framed a different part of the
    sensor: at 5 -> 2.5 um/px none of the beam was left in view.
    """

    def test_correcting_the_scale_keeps_the_beam_in_view(self):
        bp = _profiler()
        cbs = _callbacks(bp)
        _tick(cbs)
        cbs["auto_fit_zoom"](1)
        # The simulated beam jitters by ~17 px a frame, so a later frame's
        # ellipse can graze the edge of a box fitted to an earlier one.
        assert _ellipse_inside_view(_tick(cbs)[0]) > 0.9

        cbs["set_pixel_scale"](1, None, 2.5)

        fig = _tick(cbs)[0]
        assert _ellipse_inside_view(fig) > 0.9  # was 0.0
        assert dash_app._zoom_range is not None
        assert list(fig.layout.xaxis.range) == pytest.approx(
            [v * 2.5 for v in dash_app._zoom_range["x"]]
        )


class _InterleavingLock:
    """Stands in for ``_callback_lock`` and runs *interloper* at the moment a
    callback asks for the lock the first time, before it gets it."""

    def __init__(self, interloper: Any) -> None:
        import threading

        self._lock = threading.Lock()
        self._interloper = interloper

    def _maybe_interlope(self) -> None:
        interloper, self._interloper = self._interloper, None
        if interloper is not None:
            interloper()

    def __enter__(self) -> _InterleavingLock:
        self._maybe_interlope()
        self._lock.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        self._maybe_interlope()
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class TestAutoFitReadsTheFitUnderTheLock:
    """Auto-fit read the fit and the pixel size before taking the lock and
    only wrote the zoom under it. A camera switch landing in between was
    overwritten: the new camera started zoomed onto the old camera's beam."""

    def test_a_switch_in_between_is_not_overwritten(self, monkeypatch):
        bp = _profiler()
        cbs = _callbacks(bp)
        _tick(cbs)
        target = f"{discovery.SIMULATED_PREFIX}sim-2"
        switch = lambda: cbs["switch_camera"](target)  # noqa: E731
        monkeypatch.setattr(dash_app, "_callback_lock", _InterleavingLock(switch))

        cbs["auto_fit_zoom"](1)

        assert bp.camera is not None
        assert discovery.describe_open_camera(bp.camera).key == target
        assert dash_app._zoom_range is None


def _profiles(fig: Any) -> dict[str, Any]:
    """The four profile traces, told apart from the dashed/dotted overlays."""
    plain = [t for t in fig.data if t.type == "scatter" and t.line.dash is None]
    return dict(zip(["x_data", "x_fit", "y_data", "y_fit"], plain, strict=True))


class TestTheProfilesFollowTheZoom:
    """The projections and their fits were drawn at the sensor's edges, in
    data coordinates. Any zoom away from those edges -- Auto-fit, a mouse
    zoom -- left them off screen: after Auto-fit, 0% of either projection or
    fit curve was inside the view."""

    def test_after_auto_fit_every_profile_hugs_the_view(self):
        bp = _profiler()
        cbs = _callbacks(bp)
        _tick(cbs)
        cbs["auto_fit_zoom"](1)

        fig = _tick(cbs)[0]
        (x0, x1), (y0, y1) = fig.layout.xaxis.range, fig.layout.yaxis.range
        assert x0 > 0 and y0 > 0  # genuinely zoomed away from both edges

        for name in ("x_data", "x_fit"):
            trace = _profiles(fig)[name]
            y = np.asarray(trace.y)
            assert y.min() == pytest.approx(y0, abs=0.01 * (y1 - y0)), name
            assert y.max() == pytest.approx(y0 + 0.15 * (y1 - y0), rel=0.05), name
        for name in ("y_data", "y_fit"):
            trace = _profiles(fig)[name]
            x = np.asarray(trace.x)
            assert x.min() == pytest.approx(x0, abs=0.01 * (x1 - x0)), name
            assert x.max() == pytest.approx(x0 + 0.15 * (x1 - x0), rel=0.05), name

    def test_the_full_view_is_unchanged(self):
        bp = _profiler()
        cbs = _callbacks(bp)
        fig = _tick(cbs)[0]
        full = 1024 * bp.pixel_size
        x_data = np.asarray(_profiles(fig)["x_data"].y)
        y_data = np.asarray(_profiles(fig)["y_data"].x)
        assert x_data.min() == pytest.approx(0.0)
        assert x_data.max() == pytest.approx(0.15 * full)
        assert y_data.min() == pytest.approx(0.0)
        assert y_data.max() == pytest.approx(0.15 * full)


def _ellipse_centre(fig: Any) -> tuple[float, float] | None:
    """Centre of the drawn beam ellipse, or ``None`` if none is drawn."""
    traces = [t for t in fig.data if t.type == "scatter" and t.line.dash == "dash"]
    if not traces:
        return None
    x, y = np.asarray(traces[0].x), np.asarray(traces[0].y)
    return (float(x.max() + x.min()) / 2, float(y.max() + y.min()) / 2)


def _crosshair(fig: Any) -> tuple[float, float] | None:
    """``(x, y)`` of the linecut crosshair, or ``None`` if none is drawn."""
    lines = [t for t in fig.data if t.type == "scatter" and t.line.dash == "dot"]
    if not lines:
        return None
    vertical, horizontal = lines
    return float(vertical.x[0]), float(horizontal.y[0])


class TestOverlaysStayLiveUnderModelFreeDefinitions:
    """FWHM and D4σ are read straight off the profiles, so analyze() skips
    the 2D fit and the linecut. Their last results stayed on screen anyway:
    on the tilted simulator the 2D ellipse sat at one point for as long as
    FWHM was selected while the measured centre moved ~100 um a frame, drawn
    tilted beside "Angle 0.0°", and the crosshair stayed on a peak the beam
    had left."""

    def test_the_ellipse_does_not_freeze_after_switching_to_fwhm(self):
        bp = _profiler("sim-2", seed=7)
        cbs = _callbacks(bp)
        for _ in range(3):
            fig = _tick(cbs, analysis="2d")[0]
        last_2d = _ellipse_centre(fig)
        assert last_2d is not None

        centres = [
            _ellipse_centre(_tick(cbs, analysis="2d", definition="fwhm")[0]) for _ in range(4)
        ]

        drawn = [c for c in centres if c is not None]
        assert last_2d not in drawn
        assert len(set(drawn)) == len(drawn), "the ellipse froze"

    def test_the_crosshair_is_only_drawn_through_this_frames_peak(self):
        bp = _profiler("sim-2", seed=7)
        cbs = _callbacks(bp)
        for _ in range(2):
            fig = _tick(cbs, analysis="linecut")[0]
        assert _crosshair(fig) is not None

        for _ in range(3):
            fig = _tick(cbs, analysis="linecut", definition="d4s")[0]
            cross = _crosshair(fig)
            if cross is not None:
                assert bp.last_img is not None
                py, px = np.unravel_index(int(np.argmax(bp.last_img)), bp.last_img.shape)
                assert cross == pytest.approx((px * bp.pixel_size, py * bp.pixel_size))

    def test_no_angle_is_reported_unless_the_2d_fit_ran(self):
        bp = _profiler("sim-2", seed=7)
        cbs = _callbacks(bp)
        assert "Angle" in str(_tick(cbs, analysis="2d")[1])
        assert "Angle" not in str(_tick(cbs, analysis="2d", definition="fwhm")[1])


def _feed(bp: BeamProfiler, frames: list[np.ndarray]) -> None:
    """Make the attached camera deliver *frames*, round and round."""
    import itertools

    assert bp.camera is not None
    source = itertools.cycle(frames)
    bp.camera.get_image = lambda timeout=None: next(source)  # ty: ignore[invalid-assignment]


def _saturated(status: Any) -> bool:
    return "saturated" in str(status)


class TestAveragingDoesNotHideSaturation:
    """The saturation warning was computed on the averaged frame. A jittering
    beam's clipped core rarely sits at the maximum in every frame of the
    window, so with averaging on the warning vanished: on the simulator at 14
    ms it showed in 0 of 20 ticks at N=4 and N=16, while 13-14 of the 20 raw
    frames were clipped."""

    def test_a_clipped_raw_frame_is_flagged_through_the_average(self):
        bp = BeamProfiler(camera="simulated")
        # Two frames clipped in different places: no pixel is at 255 in both,
        # so their average never reaches it.
        left = np.full((64, 64), 20, dtype=np.uint8)
        left[:, :8] = 255
        right = np.full((64, 64), 20, dtype=np.uint8)
        right[:, -8:] = 255
        _feed(bp, [left, right])
        cbs = _callbacks(bp)

        statuses = [_tick(cbs, avg_n=4)[2] for _ in range(4)]

        assert bp.last_img is not None and bp.last_img.max() < 255
        assert all(_saturated(s) for s in statuses)


class TestSaturationUsesTheSensorsBitDepth:
    """With only the dtype to go on, a 12-bit sensor packed in uint16 was
    judged against 65535 and never flagged, however clipped. The camera's
    reported bit depth now sets the level."""

    @staticmethod
    def _twelve_bit() -> BeamProfiler:
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.bit_depth = 12  # ty: ignore[unresolved-attribute]
        frame = np.full((64, 64), 300, dtype=np.uint16)
        frame[:4, :] = 4095  # 6% of the sensor clipped
        _feed(bp, [frame])
        return bp

    def test_a_clipped_12_bit_frame_is_flagged(self):
        status = _tick(_callbacks(self._twelve_bit()))[2]
        assert _saturated(status)
        assert "(4095)" in str(status)

    def test_the_manual_colour_range_tops_out_at_the_sensor_maximum(self):
        cbs = _callbacks(self._twelve_bit())
        with patch("pybeamprofiler.dash_app.build_figure", return_value=go.Figure()) as build:
            _tick(cbs, auto_range=False)
        assert build.call_args.kwargs["zmax"] == 4095

    def test_a_narrower_stream_still_clips_at_its_own_maximum(self):
        """Mono8 from a 12-bit sensor clips at 255, not 4095."""
        frame = np.full((8, 8), 255, dtype=np.uint8)
        assert dash_app._saturation_max(frame, 12) == 255

    @pytest.mark.parametrize("depth", [None, True, 0, 64, "12", 12.0])
    def test_anything_but_a_plausible_integer_is_ignored(self, depth):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.bit_depth = depth  # ty: ignore[unresolved-attribute]
        assert dash_app._camera_bit_depth(bp) is None
