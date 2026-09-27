"""What the Jupyter figures actually draw, checked against the image itself.

Swapping an ellipse trace's x and y, flipping the heatmap, or swapping the
crosshair coordinates all used to pass the whole suite: the tests sampled the
profiler's own ellipse helper, never the traces a figure is built from. These
read the traces back out of each figure and compare them with the pixels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pybeamprofiler.beamprofiler import BeamProfiler

BUILDERS = ["_create_figure", "_create_fast_figure"]
PIXEL_SIZE = 2.5  # not 1, so a trace left in pixels can't pass as micrometres


def _beam(h: int, w: int, cx: float, cy: float, sx: float, sy: float, theta_deg: float):
    y, x = np.mgrid[0:h, 0:w].astype(float)
    t = np.radians(theta_deg)
    u = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    v = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return 200.0 * np.exp(-u * u / (2 * sx * sx) - v * v / (2 * sy * sy))


def _profiler(fit: str) -> BeamProfiler:
    bp = BeamProfiler(camera="simulated", fit=fit)
    bp.pixel_size = PIXEL_SIZE
    return bp


def _trace(fig: Any, name: str) -> Any:
    return next(t for t in fig.data if t.name == name)


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("theta_deg", [25.0, 115.0])
def test_the_drawn_ellipse_sits_on_the_beam_contour(builder, theta_deg):
    img = _beam(240, 260, 130.0, 110.0, 30.0, 10.0, theta_deg)
    bp = _profiler("2d")
    popt_x, popt_y = bp.analyze(img)
    fig = getattr(bp, builder)(img, popt_x, popt_y)

    ellipse = _trace(fig, "gaussian Width")
    xs = np.asarray(ellipse.x) / PIXEL_SIZE
    ys = np.asarray(ellipse.y) / PIXEL_SIZE
    sampled = img[np.round(ys).astype(int), np.round(xs).astype(int)] / img.max()
    # The 1/e² contour sits at 13.5% of the peak all the way round.
    assert sampled.min() > 0.11, "part of the drawn ellipse is off the beam"
    assert sampled.max() < 0.17, "part of the drawn ellipse cuts through the core"


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_heatmap_puts_row_zero_at_y_zero(builder):
    img = np.zeros((40, 60))
    img[0, :] = 100.0
    bp = _profiler("1d")
    fig = getattr(bp, builder)(img, None, None)

    heatmap = next(t for t in fig.data if t.type == "heatmap")
    z, y = np.asarray(heatmap.z), np.asarray(heatmap.y)
    assert y[0] == 0.0 and y[-1] == pytest.approx(39 * PIXEL_SIZE)
    assert z[0].max() == 100.0 and z[-1].max() == 0.0


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_crosshair_crosses_at_the_measured_pixel(builder):
    img = _beam(48, 64, 37.0, 21.0, 4.0, 3.0, 0.0)
    bp = _profiler("linecut")
    popt_x, popt_y = bp.analyze(img)
    fig = getattr(bp, builder)(img, popt_x, popt_y)

    assert set(_trace(fig, "Linecut X").x) == {37 * PIXEL_SIZE}
    assert set(_trace(fig, "Linecut Y").y) == {21 * PIXEL_SIZE}


def test_linecut_profiles_show_the_fitted_row_and_column():
    """The plot used to show the full-frame projections under a single-row
    fit: two curves 150 times apart in scale."""
    img = _beam(48, 64, 37.0, 21.0, 4.0, 3.0, 20.0)
    bp = _profiler("linecut")
    popt_x, popt_y = bp.analyze(img)
    fig = bp._create_figure(img, popt_x, popt_y)

    np.testing.assert_allclose(_trace(fig, "Data X").y, img[21, :])
    np.testing.assert_allclose(_trace(fig, "Data Y").x, img[:, 37])
    fit_x = np.asarray(_trace(fig, "Fit X").y)
    assert fit_x.max() == pytest.approx(img[21, :].max(), rel=0.05)


@pytest.mark.parametrize("builder", BUILDERS)
def test_a_frame_without_a_beam_says_so(builder):
    rng = np.random.default_rng(0)
    img = rng.normal(20.0, 3.0, (60, 80))
    bp = _profiler("1d")
    popt_x, popt_y = bp.analyze(img)
    assert popt_x is None and popt_y is None
    fig = getattr(bp, builder)(img, popt_x, popt_y)

    title = fig.layout.title.text
    assert "—" in title and "nan" not in title
    assert not [t for t in fig.data if t.name in ("gaussian Width", "Fit X", "Fit Y")]
