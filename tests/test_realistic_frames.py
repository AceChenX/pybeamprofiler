"""The profiler on frames that look like a real camera's.

Most tests elsewhere feed clean synthetic beams. These add what a sensor
adds -- a pedestal, read noise, 8-bit clipping -- and then do what users do
between frames: move the beam, block it, change the ROI, switch definitions.
Each of these once produced a confident wrong number rather than an error:

* D4σ measured over the whole noisy frame came out many times too wide.
* FWHM read systematically narrow once the peak was noisy.
* A warm start left behind by a moved beam kept the fit on stale or
  nonsense parameters (a centre at 3e10 px, a "beam" fitted to a noise
  bump) for as long as the stream ran.

The frame generator here is deliberately independent of the package, so a
convention shared by the fit and its reference can't hide a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybeamprofiler import fitting
from pybeamprofiler.beamprofiler import BeamProfiler


def _frame(
    rng: np.random.Generator,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    theta_deg: float = 0.0,
    *,
    h: int = 480,
    w: int = 640,
    amp: float = 200.0,
    bg: float = 12.0,
    noise: float = 4.0,
) -> np.ndarray:
    """An 8-bit frame: rotated Gaussian + pedestal + read noise, clipped."""
    y, x = np.mgrid[0:h, 0:w].astype(float)
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    u = (x - cx) * c + (y - cy) * s
    v = -(x - cx) * s + (y - cy) * c
    img = amp * np.exp(-u * u / (2 * sx * sx) - v * v / (2 * sy * sy)) + bg
    img += rng.normal(0.0, noise, (h, w))
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _truth(sx: float, sy: float, theta_deg: float) -> tuple[float, float]:
    """1/e² widths along the image axes, in pixels (4 x the projected sigma)."""
    c2, s2 = np.cos(np.radians(theta_deg)) ** 2, np.sin(np.radians(theta_deg)) ** 2
    return 4 * np.sqrt(sx * sx * c2 + sy * sy * s2), 4 * np.sqrt(sx * sx * s2 + sy * sy * c2)


def _profiler(fit: str = "1d", definition: str = "gaussian") -> BeamProfiler:
    bp = BeamProfiler(camera="simulated", fit=fit, definition=definition)
    bp.pixel_size = 1.0
    return bp


def _error(bp: BeamProfiler, truth: tuple[float, float]) -> float:
    """Worst relative width error of the two axes (inf if nothing measured)."""
    ex = bp.fw_1e2_x / truth[0] - 1
    ey = bp.fw_1e2_y / truth[1] - 1
    if not (np.isfinite(ex) and np.isfinite(ey)):
        return float("inf")
    return max(abs(ex), abs(ey))


# How close each definition can get at this signal-to-noise: the fits use
# every pixel, FWHM only the half-maximum crossings, and D4σ weights the
# noise by distance squared.
_TOLERANCE = {"gaussian": 0.02, "fwhm": 0.04, "d4s": 0.08}


class TestAccuracyOnNoisyFrames:
    @pytest.mark.parametrize("definition", ["gaussian", "fwhm", "d4s"])
    @pytest.mark.parametrize("fit", ["1d", "2d"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_widths_match_the_truth(self, fit, definition, seed):
        rng = np.random.default_rng(seed)
        bp = _profiler(fit, definition)
        bp.analyze(_frame(rng, 300.4, 250.7, 24.0, 16.0, 30.0))
        assert _error(bp, _truth(24.0, 16.0, 30.0)) < _TOLERANCE[definition]
        assert bp.center_x == pytest.approx(300.4, abs=1.0)
        assert bp.center_y == pytest.approx(250.7, abs=1.0)

    def test_d4s_ignores_the_noise_of_the_empty_sensor(self):
        """Taken over the whole frame, D4σ summed the noise of every
        beam-free pixel, weighted by distance squared: a 6 px beam measured
        dozens of times too wide. The ISO window keeps it out."""
        rng = np.random.default_rng(4)
        bp = _profiler(definition="d4s")
        bp.analyze(_frame(rng, 320, 240, 6.0, 6.0, h=1024, w=1024, noise=10.0, bg=10.0))
        assert _error(bp, _truth(6.0, 6.0, 0.0)) < 0.25

    def test_fwhm_of_a_small_beam_is_steady(self):
        """Off a full-frame projection, the noise of every row made the
        half-maximum crossings of a 6 px beam wander by 7% frame to frame.
        The band the beam occupies carries a fraction of that noise."""
        rng = np.random.default_rng(5)
        widths = []
        for _ in range(20):
            bp = _profiler(definition="fwhm")
            bp.analyze(_frame(rng, 480.3, 530.7, 6.0, 6.0, h=1024, w=1024, noise=10.0, bg=10.0))
            widths.append(bp.fw_1e2_x / _truth(6.0, 6.0, 0.0)[0])
        assert np.std(widths) < 0.03
        assert abs(np.mean(widths) - 1) < 0.03

    def test_a_clipped_black_level_does_not_bias_the_background(self):
        """With the black level near zero, the noise floor clips at 0 and
        its median sits below its mean. Moments need the mean."""
        rng = np.random.default_rng(6)
        img = _frame(rng, 320, 240, 20.0, 20.0, bg=6.0, noise=8.0)
        assert np.mean(img == 0) > 0.1  # the scenario really is clipped
        bp = _profiler(definition="d4s")
        bp.analyze(img)
        assert _error(bp, _truth(20.0, 20.0, 0.0)) < 0.08

    def test_a_small_beam_on_a_large_sensor(self):
        """Decimating a 1 MPix frame to the fit grid would shrink this beam
        below a pixel; it has to be cropped out instead."""
        rng = np.random.default_rng(7)
        bp = _profiler("2d")
        bp.analyze(_frame(rng, 700.3, 300.6, 3.0, 2.5, h=1024, w=1024))
        assert _error(bp, _truth(3.0, 2.5, 0.0)) < 0.03
        assert bp.center_x == pytest.approx(700.3, abs=0.3)

    def test_an_elongated_small_beam_is_not_cropped_short(self):
        """The crop used to size both axes from the smaller sigma, keeping
        barely one sigma of this beam's long axis."""
        rng = np.random.default_rng(8)
        bp = _profiler("2d")
        bp.analyze(_frame(rng, 500.0, 400.0, 16.0, 2.0, h=1024, w=1024))
        assert _error(bp, _truth(16.0, 2.0, 0.0)) < 0.03

    def test_a_cold_2d_fit_survives_a_high_pedestal(self):
        """The cold seed took moments of the whole frame above its minimum,
        and a 30-count pedestal over a megapixel outweighs the beam: the seed
        came out ten times too wide and the fit ran out of evaluations."""
        rng = np.random.default_rng(9)
        img = _frame(rng, 640, 512, 90.0, 30.0, 35.0, h=1024, w=1280, amp=180, bg=30, noise=10)
        popt, ok = fitting.fit_2d_gaussian(img)
        assert ok
        major, minor, theta = fitting.canonical_ellipse(popt[3], popt[4], popt[5])
        assert major == pytest.approx(90.0, rel=0.03)
        assert minor == pytest.approx(30.0, rel=0.03)
        assert np.degrees(theta) == pytest.approx(35.0, abs=1.0)


class TestNoBeam:
    @pytest.mark.parametrize("definition", ["gaussian", "fwhm", "d4s"])
    @pytest.mark.parametrize("fit", ["1d", "2d", "linecut"])
    def test_pure_noise_is_not_a_beam(self, fit, definition):
        rng = np.random.default_rng(10)
        bp = _profiler(fit, definition)
        bp.analyze(_frame(rng, 320, 240, 20.0, 20.0, amp=0.0))
        assert np.isnan(bp.width_x) and np.isnan(bp.width_y)
        assert np.isnan(bp.center_x)
        assert bp.beam_ellipse() is None


class TestAcrossFrames:
    """What happens on the *first* frame after something changes."""

    @pytest.mark.parametrize("fit", ["1d", "2d", "linecut"])
    def test_recovers_when_the_beam_jumps(self, fit):
        rng = np.random.default_rng(11)
        bp = _profiler(fit)
        for _ in range(3):
            bp.analyze(_frame(rng, 150, 150, 12.0, 12.0))
        bp.analyze(_frame(rng, 480, 330, 12.0, 12.0))  # ~27 sigma away
        assert bp.center_x == pytest.approx(480, abs=1.0)
        assert bp.center_y == pytest.approx(330, abs=1.0)
        assert _error(bp, _truth(12.0, 12.0, 0.0)) < 0.03

    @pytest.mark.parametrize("definition", ["gaussian", "fwhm", "d4s"])
    @pytest.mark.parametrize("fit", ["1d", "2d", "linecut"])
    def test_a_blocked_beam_reports_nothing_then_recovers(self, fit, definition):
        rng = np.random.default_rng(12)
        bp = _profiler(fit, definition)
        bp.analyze(_frame(rng, 320, 240, 20.0, 15.0))
        bp.analyze(_frame(rng, 320, 240, 20.0, 15.0, amp=0.0))
        assert np.isnan(bp.width_x), "a blank frame must not repeat the last result"
        assert bp.beam_ellipse() is None
        bp.analyze(_frame(rng, 200, 300, 20.0, 15.0))
        assert bp.center_x == pytest.approx(200, abs=1.5)
        assert _error(bp, _truth(20.0, 15.0, 0.0)) < _TOLERANCE[definition]

    @pytest.mark.parametrize("fit", ["1d", "2d", "linecut"])
    def test_an_roi_moved_without_resizing(self, fit):
        """Same frame shape, new origin: the warm start now points at the
        wrong place, and nothing about the frame says so."""
        rng = np.random.default_rng(13)
        scene = _frame(rng, 500, 400, 14.0, 14.0, h=800, w=1000)
        bp = _profiler(fit)
        bp.analyze(scene[100:500, 200:700])  # beam at (300, 300) in this ROI
        bp.analyze(scene[250:650, 450:950])  # the same beam, now at (50, 150)
        assert bp.center_x == pytest.approx(50, abs=1.0)
        assert bp.center_y == pytest.approx(150, abs=1.0)

    def test_a_new_frame_shape_drops_the_warm_starts(self):
        rng = np.random.default_rng(14)
        bp = _profiler("2d")
        bp.analyze(_frame(rng, 320, 240, 20.0, 15.0))
        assert bp._last_popt_2d is not None
        bp.analyze(_frame(rng, 100, 80, 10.0, 8.0, h=200, w=240))
        assert bp._analysis_shape == (200, 240)
        assert bp.center_x == pytest.approx(100, abs=1.0)
        assert _error(bp, _truth(10.0, 8.0, 0.0)) < 0.03

    def test_switching_definition_redraws_the_overlay(self):
        """FWHM skips the 2D fit, so the tilted 2D ellipse from the frame
        before must not survive into it."""
        rng = np.random.default_rng(15)
        img = _frame(rng, 320, 240, 30.0, 10.0, 40.0)
        bp = _profiler("2d")
        bp.analyze(img)
        ellipse = bp.beam_ellipse()
        assert ellipse is not None and np.degrees(ellipse[4]) == pytest.approx(40.0, abs=1.0)

        bp.definition = "fwhm"
        bp.analyze(img)
        ellipse = bp.beam_ellipse()
        assert ellipse is not None
        cx, cy, rx, ry, angle = ellipse
        assert angle == 0.0
        # Semi-axes are half the reported widths, in the reported definition.
        assert rx == pytest.approx(bp.width_x / 2)
        assert ry == pytest.approx(bp.width_y / 2)

    def test_the_crosshair_belongs_to_linecut_frames_only(self):
        rng = np.random.default_rng(16)
        img = _frame(rng, 320, 240, 20.0, 15.0)
        bp = _profiler("linecut")
        bp.analyze(img)
        assert bp._linecut_x is not None and bp._linecut_y is not None
        assert bp._linecut_x == pytest.approx(320, abs=3)
        bp.fit_method = "1d"
        bp.analyze(img)
        assert bp._linecut_x is None and bp._linecut_y is None


class TestEllipseScaling:
    """Decimation stretches x and y by slightly different factors."""

    @pytest.mark.parametrize("theta_deg", [0.0, 30.0, 45.0, 110.0])
    @pytest.mark.parametrize("kx,ky", [(1.0, 1.0), (10.07, 10.13), (2.0, 0.5)])
    def test_matches_an_explicit_covariance_transform(self, theta_deg, kx, ky):
        sx, sy, t = 12.0, 4.0, np.radians(theta_deg)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        cov = rot @ np.diag([sx * sx, sy * sy]) @ rot.T
        stretch = np.diag([kx, ky])
        values, vectors = np.linalg.eigh(stretch @ cov @ stretch)
        major, minor = np.sqrt(values[1]), np.sqrt(values[0])
        angle = np.arctan2(vectors[1, 1], vectors[0, 1]) % np.pi

        got = fitting._stretch_ellipse(sx, sy, t, kx, ky)
        assert got[0] == pytest.approx(major, rel=1e-9)
        assert got[1] == pytest.approx(minor, rel=1e-9)
        assert got[2] % np.pi == pytest.approx(angle, abs=1e-9)


class TestCoreVarianceCorrection:
    @pytest.mark.parametrize("cut", [0.05, 0.2, 0.5])
    def test_matches_brute_force(self, cut):
        x = np.linspace(-12, 12, 200_001)
        g = np.exp(-x * x / 2)
        keep = g > cut
        brute_1d = (x[keep] ** 2 * g[keep]).sum() / g[keep].sum()
        assert fitting._kept_variance(cut, ndim=1) == pytest.approx(brute_1d, rel=1e-4)

        grid = np.linspace(-9, 9, 1501)
        gx, gy = np.meshgrid(grid, grid)
        g2 = np.exp(-(gx * gx + gy * gy) / 2)
        keep2 = g2 > cut
        brute_2d = (gx[keep2] ** 2 * g2[keep2]).sum() / g2[keep2].sum()
        assert fitting._kept_variance(cut, ndim=2) == pytest.approx(brute_2d, rel=1e-3)
