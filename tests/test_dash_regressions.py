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
import pytest
from dash import html

from pybeamprofiler import dash_app
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
