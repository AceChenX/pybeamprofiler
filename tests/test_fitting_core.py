"""Tests for the pure numerical routines in :mod:`pybeamprofiler.fitting`."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from pybeamprofiler import fitting
from pybeamprofiler.constants import D4SIGMA_FACTOR, FW_1E_FACTOR, GAUSSIAN_TO_FWHM


def _gaussian_profile(n: int = 200, center: float = 100.0, sigma: float = 20.0) -> np.ndarray:
    x = np.arange(n, dtype=float)
    return 250.0 * np.exp(-((x - center) ** 2) / (2 * sigma**2))


class TestConversionFactors:
    """The width factors are physics, not tuning knobs — pin them down."""

    def test_fwhm_factor(self):
        assert GAUSSIAN_TO_FWHM == pytest.approx(2.354820045, rel=1e-9)

    def test_fw_1e_factor(self):
        assert FW_1E_FACTOR == pytest.approx(2.828427125, rel=1e-9)

    def test_ordering(self):
        """Lower threshold ⇒ wider: FWHM < 1/e < 1/e²."""
        assert GAUSSIAN_TO_FWHM < FW_1E_FACTOR < D4SIGMA_FACTOR

    @pytest.mark.parametrize(
        ("factor", "threshold"),
        [(GAUSSIAN_TO_FWHM, 0.5), (FW_1E_FACTOR, 1 / math.e), (D4SIGMA_FACTOR, 1 / math.e**2)],
    )
    def test_factor_matches_its_threshold(self, factor, threshold):
        """Each factor must land exactly on its intensity threshold."""
        sigma = 1.0
        half_width = factor * sigma / 2.0
        assert math.exp(-(half_width**2) / (2 * sigma**2)) == pytest.approx(threshold)


class TestGaussian:
    def test_peak_and_baseline(self):
        x = np.arange(101, dtype=float)
        y = fitting.gaussian(x, 100.0, 50.0, 10.0, 5.0)
        assert y[50] == pytest.approx(105.0)
        assert y[0] == pytest.approx(5.0, abs=1e-3)

    def test_fwhm_of_the_model_matches_the_constant(self):
        sigma = 12.0
        x = np.linspace(0, 200, 20001)
        y = fitting.gaussian(x, 1.0, 100.0, sigma, 0.0)
        above = x[y >= 0.5]
        assert (above[-1] - above[0]) == pytest.approx(GAUSSIAN_TO_FWHM * sigma, rel=1e-3)


class TestGaussian2D:
    def test_unrotated_matches_separable_product(self):
        y, x = np.mgrid[0:40, 0:40]
        got = fitting.gaussian_2d((x, y), 100.0, 20.0, 15.0, 5.0, 8.0, 0.0, 3.0)
        expected = (
            100.0
            * np.exp(-((x - 20.0) ** 2) / (2 * 5.0**2))
            * np.exp(-((y - 15.0) ** 2) / (2 * 8.0**2))
            + 3.0
        )
        np.testing.assert_allclose(got, expected.ravel(), rtol=1e-9)

    def test_quarter_turn_swaps_the_axes(self):
        y, x = np.mgrid[0:40, 0:40]
        straight = fitting.gaussian_2d((x, y), 100.0, 20.0, 20.0, 4.0, 10.0, 0.0, 0.0)
        turned = fitting.gaussian_2d((x, y), 100.0, 20.0, 20.0, 10.0, 4.0, np.pi / 2, 0.0)
        np.testing.assert_allclose(straight, turned, atol=1e-9)

    def test_output_is_flat(self):
        y, x = np.mgrid[0:13, 0:17]
        assert fitting.gaussian_2d((x, y), 1.0, 8.0, 6.0, 2.0, 2.0, 0.0, 0.0).shape == (13 * 17,)


class TestMeasureFwhm:
    def test_recovers_gaussian_width(self):
        sigma = 20.0
        center, fwhm, peak = fitting.measure_fwhm(_gaussian_profile(sigma=sigma))
        assert center == pytest.approx(100.0, abs=0.5)
        assert fwhm == pytest.approx(GAUSSIAN_TO_FWHM * sigma, rel=0.01)
        assert peak == pytest.approx(250.0, rel=0.01)

    def test_subtracts_baseline(self):
        """A DC offset must not change the measured width."""
        profile = _gaussian_profile()
        _, plain, _ = fitting.measure_fwhm(profile)
        _, offset, _ = fitting.measure_fwhm(profile + 500.0)
        assert offset == pytest.approx(plain, rel=1e-9)

    def test_flat_profile_has_zero_width(self):
        center, fwhm, peak = fitting.measure_fwhm(np.full(50, 7.0))
        assert fwhm == 0.0
        assert peak == 0.0
        assert 0 <= center < 50

    def test_interpolates_between_samples(self):
        """A triangle with known flanks pins the sub-pixel interpolation."""
        profile = np.array([0.0, 0.0, 2.0, 4.0, 2.0, 0.0, 0.0])
        center, fwhm, peak = fitting.measure_fwhm(profile)
        # Half max is 2.0, reached exactly at indices 2 and 4.
        assert peak == 4.0
        assert center == pytest.approx(3.0)
        assert fwhm == pytest.approx(2.0)

    def test_beam_clipped_at_the_edge(self):
        """A profile that never falls below half max stops at the array end."""
        profile = np.linspace(0.0, 100.0, 20)
        center, fwhm, _ = fitting.measure_fwhm(profile)
        assert fwhm >= 0.0
        assert 0 <= center <= 19

    def test_single_sample(self):
        center, fwhm, peak = fitting.measure_fwhm(np.array([42.0]))
        assert (center, fwhm, peak) == (0.0, 0.0, 0.0)

    def test_unsigned_input_does_not_wrap(self):
        """uint8 arithmetic must not underflow during baseline removal."""
        x = np.arange(64)
        profile = (200 * np.exp(-((x - 32) ** 2) / (2 * 8.0**2)) + 30).astype(np.uint8)
        _, fwhm, _ = fitting.measure_fwhm(profile)
        assert fwhm == pytest.approx(GAUSSIAN_TO_FWHM * 8.0, rel=0.05)


class TestMeasureD4s:
    def test_recovers_gaussian_width(self):
        sigma = 20.0
        center, d4s = fitting.measure_d4s(_gaussian_profile(sigma=sigma))
        assert center == pytest.approx(100.0, abs=0.5)
        assert d4s == pytest.approx(4.0 * sigma, rel=0.02)

    def test_blank_profile_returns_midpoint(self):
        center, d4s = fitting.measure_d4s(np.zeros(80))
        assert center == 40.0
        assert d4s == 1.0

    def test_flat_top_is_wider_than_its_gaussian_fit_would_suggest(self):
        """D4σ is shape-free, which is the whole point of offering it."""
        profile = np.zeros(200)
        profile[60:140] = 100.0
        _, d4s = fitting.measure_d4s(profile)
        # Uniform slab of width W has sigma = W/sqrt(12) ⇒ D4σ = 4W/sqrt(12).
        assert d4s == pytest.approx(4 * 80 / math.sqrt(12), rel=0.02)

    def test_off_center_beam(self):
        center, _ = fitting.measure_d4s(_gaussian_profile(n=300, center=210.0, sigma=15.0))
        assert center == pytest.approx(210.0, abs=0.5)


class TestFit1DGaussian:
    def test_recovers_known_parameters(self):
        x = np.arange(200, dtype=float)
        truth = (180.0, 90.0, 25.0, 12.0)
        popt = fitting.fit_1d_gaussian(fitting.gaussian(x, *truth))
        assert popt[0] == pytest.approx(truth[0], rel=1e-3)
        assert popt[1] == pytest.approx(truth[1], rel=1e-3)
        assert abs(popt[2]) == pytest.approx(truth[2], rel=1e-3)
        assert popt[3] == pytest.approx(truth[3], abs=1e-3)

    def test_empty_profile(self):
        assert fitting.fit_1d_gaussian(np.array([])) == [0.0, 0.0, 1.0, 0.0]

    def test_profile_shorter_than_parameter_count(self):
        """curve_fit raises TypeError here, not RuntimeError — must be caught."""
        popt = fitting.fit_1d_gaussian(np.array([1.0, 5.0, 2.0]))
        assert len(popt) == 4
        assert np.all(np.isfinite(np.asarray(popt, dtype=float)))

    def test_warm_start_is_used(self):
        x = np.arange(200, dtype=float)
        profile = fitting.gaussian(x, 180.0, 90.0, 25.0, 12.0)
        warm = fitting.fit_1d_gaussian(profile, last_popt=[180.0, 90.0, 25.0, 12.0])
        assert warm[1] == pytest.approx(90.0, rel=1e-3)

    def test_stale_warm_start_recovers_via_cold_retry(self):
        """A beam that jumps far from the cached guess must not get stuck.

        The warm fit is forced to fail; the retry from a fresh estimate has to
        find the real center instead of echoing the stale parameters back.
        """
        x = np.arange(400, dtype=float)
        profile = fitting.gaussian(x, 200.0, 320.0, 15.0, 5.0)
        stale = [200.0, 40.0, 15.0, 5.0]

        real_curve_fit = fitting.curve_fit
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Optimal parameters not found")
            return real_curve_fit(*args, **kwargs)

        with patch("pybeamprofiler.fitting.curve_fit", side_effect=flaky):
            popt = fitting.fit_1d_gaussian(profile, last_popt=stale)

        assert calls["n"] == 2, "cold retry should have run"
        assert popt[1] == pytest.approx(320.0, rel=1e-2)

    def test_both_attempts_failing_returns_a_guess(self):
        profile = _gaussian_profile()
        with patch("pybeamprofiler.fitting.curve_fit", side_effect=RuntimeError("nope")):
            popt = fitting.fit_1d_gaussian(profile, last_popt=[1.0, 2.0, 3.0, 4.0])
        assert len(popt) == 4

    def test_cold_failure_is_not_retried(self):
        """Without a warm start there is nothing better to retry from."""
        profile = _gaussian_profile()
        with patch(
            "pybeamprofiler.fitting.curve_fit", side_effect=RuntimeError("nope")
        ) as mock_fit:
            fitting.fit_1d_gaussian(profile)
        assert mock_fit.call_count == 1


class TestFit2DGaussian:
    @staticmethod
    def _synthetic(h=120, w=120, x0=60.0, y0=55.0, sx=18.0, sy=9.0, theta=0.6):
        y, x = np.mgrid[0:h, 0:w]
        return fitting.gaussian_2d((x, y), 200.0, x0, y0, sx, sy, theta, 5.0).reshape(h, w)

    def test_recovers_rotated_beam(self):
        popt, converged = fitting.fit_2d_gaussian(self._synthetic())
        assert converged
        assert popt[1] == pytest.approx(60.0, abs=0.5)
        assert popt[2] == pytest.approx(55.0, abs=0.5)
        assert abs(popt[3]) == pytest.approx(18.0, rel=0.02)
        assert abs(popt[4]) == pytest.approx(9.0, rel=0.02)
        assert popt[5] == pytest.approx(0.6, abs=0.05)

    def test_warm_start_converges(self):
        img = self._synthetic()
        cold, _ = fitting.fit_2d_gaussian(img)
        warm, converged = fitting.fit_2d_gaussian(img, last_popt=cold)
        assert converged
        np.testing.assert_allclose(np.asarray(warm)[1:5], np.asarray(cold)[1:5], rtol=0.02)

    def test_large_image_is_downsampled_but_reported_full_scale(self):
        img = self._synthetic(h=600, w=600, x0=300.0, y0=280.0, sx=60.0, sy=40.0, theta=0.0)
        popt, converged = fitting.fit_2d_gaussian(img, max_dim=128)
        assert converged
        assert popt[1] == pytest.approx(300.0, rel=0.02)
        assert popt[2] == pytest.approx(280.0, rel=0.02)
        assert abs(popt[3]) == pytest.approx(60.0, rel=0.05)

    def test_failure_reports_not_converged(self):
        with patch("pybeamprofiler.fitting.curve_fit", side_effect=RuntimeError("nope")):
            popt, converged = fitting.fit_2d_gaussian(self._synthetic(h=40, w=40))
        assert not converged
        assert len(popt) == 7

    def test_failure_on_downsampled_image_returns_full_scale_guess(self):
        img = self._synthetic(h=512, w=512, x0=256.0, y0=256.0, sx=30.0, sy=30.0, theta=0.0)
        with patch("pybeamprofiler.fitting.curve_fit", side_effect=RuntimeError("nope")):
            popt, converged = fitting.fit_2d_gaussian(img, max_dim=128)
        assert not converged
        # The guess must come back in original coordinates, near the real peak —
        # not left in the downsampled frame the fit actually ran in.
        assert popt[1] == pytest.approx(256.0, rel=0.05)
        assert popt[2] == pytest.approx(256.0, rel=0.05)

    def test_diverging_warm_lm_falls_back_to_bounded(self):
        """A degenerate LM result must trigger the bounded solver, not escape."""
        img = self._synthetic(h=80, w=80, x0=40.0, y0=40.0, sx=10.0, sy=10.0, theta=0.0)
        bad_warm = [200.0, 40.0, 40.0, 1e-3, 1e-3, 0.0, 5.0]
        popt, converged = fitting.fit_2d_gaussian(img, last_popt=bad_warm)
        assert converged
        assert abs(popt[3]) > 0.1
        assert abs(popt[4]) > 0.1


class TestDownsample:
    def test_small_image_is_returned_untouched(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        assert fitting.downsample(img, 1024) is img

    def test_boundary_is_inclusive(self):
        img = np.zeros((1024, 768), dtype=np.uint8)
        assert fitting.downsample(img, 1024) is img

    @pytest.mark.parametrize(("h", "w"), [(3000, 4000), (4000, 2000), (2000, 4000)])
    def test_longest_edge_hits_the_cap(self, h, w):
        out = fitting.downsample(np.zeros((h, w), dtype=np.uint8), 1024)
        assert max(out.shape) == 1024

    def test_aspect_ratio_preserved(self):
        out = fitting.downsample(np.zeros((2000, 4000), dtype=np.uint8), 1024)
        assert out.shape[0] / out.shape[1] == pytest.approx(0.5, abs=0.02)

    def test_dtype_and_values_are_preserved(self):
        """Decimation must report real sensor counts, not blended ones."""
        img = np.random.randint(0, 256, (2048, 2048), dtype=np.uint8)
        out = fitting.downsample(img, 512)
        assert out.dtype == np.uint8
        assert np.isin(out, img).all()

    def test_extreme_reduction_keeps_at_least_one_row(self):
        out = fitting.downsample(np.zeros((4000, 4), dtype=np.uint8), 8)
        assert out.shape[0] == 8
        assert out.shape[1] >= 1
