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
