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


class TestCanonicalEllipse:
    """The rotated 2D Gaussian is degenerate: ``(sx, sy, th)`` and
    ``(sy, sx, th + 90deg)`` are the same ellipse. Left alone the solver picks
    between them arbitrarily, which made a near-round beam's reported X and Y
    widths swap from frame to frame."""

    def test_major_axis_comes_first(self):
        major, minor, _ = fitting.canonical_ellipse(10.0, 40.0, 0.0)
        assert (major, minor) == (40.0, 10.0)

    def test_already_ordered_is_untouched(self):
        major, minor, theta = fitting.canonical_ellipse(40.0, 10.0, 0.3)
        assert (major, minor) == (40.0, 10.0)
        assert theta == pytest.approx(0.3)

    def test_swapping_rotates_by_a_quarter_turn(self):
        _, _, theta = fitting.canonical_ellipse(10.0, 40.0, 0.0)
        assert theta == pytest.approx(np.pi / 2)

    def test_theta_is_wrapped_into_half_a_turn(self):
        for theta in (-3.0, 0.0, 3.5, 7.0, 100.0):
            _, _, out = fitting.canonical_ellipse(5.0, 3.0, theta)
            assert 0.0 <= out < np.pi

    def test_negative_sigmas_are_normalised(self):
        """The model is even in sigma, so the solver may return either sign."""
        major, minor, _ = fitting.canonical_ellipse(-30.0, -12.0, 0.2)
        assert (major, minor) == (30.0, 12.0)

    def test_both_parameterisations_land_on_the_same_answer(self):
        a = fitting.canonical_ellipse(40.0, 10.0, 0.4)
        b = fitting.canonical_ellipse(10.0, 40.0, 0.4 + np.pi / 2)
        assert a[0] == pytest.approx(b[0])
        assert a[1] == pytest.approx(b[1])
        assert a[2] == pytest.approx(b[2])


class TestImageAxisSigmas:
    """Widths projected onto the sensor's own axes — what 1D mode reports."""

    def test_unrotated_beam_passes_through(self):
        sx, sy = fitting.image_axis_sigmas(30.0, 10.0, 0.0)
        assert (sx, sy) == pytest.approx((30.0, 10.0))

    def test_quarter_turn_swaps_them(self):
        sx, sy = fitting.image_axis_sigmas(30.0, 10.0, np.pi / 2)
        assert (sx, sy) == pytest.approx((10.0, 30.0))

    def test_invariant_under_the_axis_swap(self):
        """This is the property that makes the reported widths stable."""
        a = fitting.image_axis_sigmas(90.0, 30.0, np.radians(35))
        b = fitting.image_axis_sigmas(30.0, 90.0, np.radians(125))
        assert a == pytest.approx(b)

    def test_matches_what_the_1d_projection_fits_measure(self):
        h = w = 512
        y, x = np.mgrid[0:h, 0:w].astype(float)
        sx, sy, theta = 60.0, 20.0, np.radians(35)
        img = fitting.gaussian_2d((x, y), 200.0, 256.0, 256.0, sx, sy, theta, 0.0).reshape(h, w)

        expected = fitting.image_axis_sigmas(sx, sy, theta)
        measured = (
            abs(fitting.fit_1d_gaussian(img.sum(axis=0))[2]),
            abs(fitting.fit_1d_gaussian(img.sum(axis=1))[2]),
        )
        assert measured == pytest.approx(expected, rel=1e-3)

    def test_a_round_beam_is_round_from_every_angle(self):
        for theta in (0.0, 0.7, 1.9, 3.0):
            sx, sy = fitting.image_axis_sigmas(25.0, 25.0, theta)
            assert sx == pytest.approx(25.0)
            assert sy == pytest.approx(25.0)


class TestFitRegionSelection:
    """The fit runs on a reduced grid for speed. A large beam is decimated;
    a small one is *cropped* instead, because decimating a tightly focused
    beam below about a pixel of sigma makes the fit unreliable and then
    non-convergent — and fitting the whole frame at full resolution to avoid
    that took two seconds, which is worse than the problem."""

    @staticmethod
    def _beam(sigma, h=1024, w=1024, cx=None, cy=None):
        cx = w / 2 if cx is None else cx
        cy = h / 2 if cy is None else cy
        y, x = np.mgrid[0:h, 0:w].astype(float)
        return fitting.gaussian_2d((x, y), 200.0, cx, cy, sigma * 1.5, sigma, 0.4, 5.0).reshape(
            h, w
        )

    def test_a_large_beam_is_left_whole_and_decimated(self):
        img = self._beam(50.0)
        region, ox, oy = fitting._fit_region(img, 128, 50.0, (512.0, 512.0))
        assert region.shape == img.shape
        assert (ox, oy) == (0, 0)

    def test_a_small_beam_is_cropped_around_itself(self):
        img = self._beam(4.0)
        region, ox, oy = fitting._fit_region(img, 128, 4.0, (512.0, 512.0))
        assert max(region.shape) < max(img.shape)
        assert ox > 0 and oy > 0

    def test_the_crop_is_centred_on_the_beam(self):
        img = self._beam(4.0, cx=300.0, cy=700.0)
        region, ox, oy = fitting._fit_region(img, 128, 4.0, (300.0, 700.0))
        h, w = region.shape
        assert ox <= 300 <= ox + w
        assert oy <= 700 <= oy + h

    def test_the_crop_never_goes_below_a_usable_size(self):
        """Seven free parameters need more than a handful of pixels."""
        from pybeamprofiler.constants import MIN_FIT_2D_WINDOW_PX

        region, _, _ = fitting._fit_region(self._beam(0.5), 128, 0.5, (512.0, 512.0))
        assert min(region.shape) >= MIN_FIT_2D_WINDOW_PX

    def test_the_crop_is_clamped_to_the_frame(self):
        """A beam near the edge must not produce an out-of-bounds window."""
        img = self._beam(4.0, cx=3.0, cy=1020.0)
        region, ox, oy = fitting._fit_region(img, 128, 4.0, (3.0, 1020.0))
        assert ox >= 0 and oy >= 0
        assert ox + region.shape[1] <= img.shape[1]
        assert oy + region.shape[0] <= img.shape[0]
        assert region.size > 0

    def test_a_window_covering_the_whole_frame_is_not_cropped(self):
        """On a tiny ROI the minimum window already spans the sensor, so the
        frame is handed back whole rather than sliced into an identical copy."""
        img = self._beam(0.5, h=30, w=30, cx=15.0, cy=15.0)
        region, ox, oy = fitting._fit_region(img, 128, 0.5, (15.0, 15.0))
        assert region is img
        assert (ox, oy) == (0, 0)

    def test_no_hint_means_no_crop(self):
        img = self._beam(50.0)
        for hint, centre in ((None, (512.0, 512.0)), (0.0, (512.0, 512.0)), (4.0, None)):
            region, ox, oy = fitting._fit_region(img, 128, hint, centre)
            assert region.shape == img.shape
            assert (ox, oy) == (0, 0)

    @pytest.mark.parametrize("sigma", [200.0, 50.0, 12.0, 6.0, 3.0, 1.5])
    def test_beams_of_every_size_still_fit(self, sigma):
        """A 3 px beam previously did not converge at all."""
        centre = (512.0, 500.0)
        img = self._beam(sigma, cx=centre[0], cy=centre[1])
        popt, converged = fitting.fit_2d_gaussian(img, None, sigma_hint=sigma, center_hint=centre)
        assert converged
        major, minor, _ = fitting.canonical_ellipse(popt[3], popt[4], popt[5])
        assert major == pytest.approx(sigma * 1.5, rel=0.1)
        assert minor == pytest.approx(sigma, rel=0.1)


class TestDecimatedCoordinates:
    """``ndimage.zoom`` aligns the first and last pixel *centres*, so a fitted
    coordinate maps back by ``(n_in-1)/(n_out-1)``, not by ``1/ds``. The naive
    ratio biased the reported beam centre low — 3.4 px decimating 1024 to 128,
    7.4 px from 2048, which is 37 um on a 5 um pitch."""

    @staticmethod
    def _beam(n, cx, cy, sigma):
        y, x = np.mgrid[0:n, 0:n].astype(float)
        return fitting.gaussian_2d((x, y), 200.0, cx, cy, sigma * 1.5, sigma, 0.4, 5.0).reshape(
            n, n
        )

    @pytest.mark.parametrize("n", [512, 1024, 2048])
    def test_centre_survives_decimation(self, n):
        # Deliberately not a round number, so an off-by-half-pixel shows.
        cx, cy = n * 0.4901, n * 0.5137
        sigma = n / 20  # large enough that the frame is decimated, not cropped
        img = self._beam(n, cx, cy, sigma)

        popt, converged = fitting.fit_2d_gaussian(img, None, sigma_hint=sigma)
        assert converged
        assert popt[1] == pytest.approx(cx, abs=0.5)
        assert popt[2] == pytest.approx(cy, abs=0.5)

    @pytest.mark.parametrize("n", [512, 1024, 2048])
    def test_sigma_survives_decimation(self, n):
        sigma = n / 20
        img = self._beam(n, n * 0.4901, n * 0.5137, sigma)
        popt, _ = fitting.fit_2d_gaussian(img, None, sigma_hint=sigma)
        major, minor, _ = fitting.canonical_ellipse(popt[3], popt[4], popt[5])
        assert major == pytest.approx(sigma * 1.5, rel=0.02)
        assert minor == pytest.approx(sigma, rel=0.02)

    def test_a_cropped_fit_reports_full_frame_coordinates(self):
        """The crop offset has to be added back, or every small beam reads as
        being near the origin."""
        cx, cy = 300.0, 700.0
        img = self._beam(1024, cx, cy, 4.0)
        popt, converged = fitting.fit_2d_gaussian(img, None, sigma_hint=4.0, center_hint=(cx, cy))
        assert converged
        assert popt[1] == pytest.approx(cx, abs=0.5)
        assert popt[2] == pytest.approx(cy, abs=0.5)

    def test_a_warm_start_round_trips_through_the_crop(self):
        """The cached parameters are in full-frame coordinates and have to be
        mapped into the region and back again without drifting."""
        cx, cy = 300.0, 700.0
        img = self._beam(1024, cx, cy, 4.0)
        popt, _ = fitting.fit_2d_gaussian(img, None, sigma_hint=4.0, center_hint=(cx, cy))
        for _ in range(5):
            popt, converged = fitting.fit_2d_gaussian(
                img, popt, sigma_hint=4.0, center_hint=(cx, cy)
            )
            assert converged
            assert popt[1] == pytest.approx(cx, abs=0.5)
            assert popt[2] == pytest.approx(cy, abs=0.5)


class TestFitEvaluationCap:
    def test_the_bounded_solver_gets_a_real_cap(self):
        """``maxfev`` is an lm-only name; the bounded path needs ``max_nfev``
        or scipy silently falls back to its own 700-evaluation default."""
        import inspect

        from pybeamprofiler.constants import MAX_FIT_2D_EVALS

        source = inspect.getsource(fitting.fit_2d_gaussian)
        assert "max_nfev=MAX_FIT_2D_EVALS" in source
        assert "maxfev" not in source.split("bounds=bounds")[1]
        assert MAX_FIT_2D_EVALS > 0


def _image_major_axis(img):
    """Major/minor sigma and orientation straight from image moments.

    Deliberately independent of anything in ``fitting`` — this is the
    reference the fit is checked against.
    """
    a = img.astype(float) - img.min()
    h, w = a.shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    total = a.sum()
    cx, cy = (a * x).sum() / total, (a * y).sum() / total
    vxx = (a * (x - cx) ** 2).sum() / total
    vyy = (a * (y - cy) ** 2).sum() / total
    vxy = (a * (x - cx) * (y - cy)).sum() / total
    vals, vecs = np.linalg.eigh(np.array([[vxx, vxy], [vxy, vyy]]))
    i = int(np.argmax(vals))
    angle = np.degrees(np.arctan2(vecs[1, i], vecs[0, i])) % 180
    return np.sqrt(vals[i]), np.sqrt(vals[1 - i]), angle


def _angle_error(a: float, b: float) -> float:
    """Difference between two orientations, accounting for the 180 wrap."""
    return abs(((a - b + 90) % 180) - 90)


class TestRotationConvention:
    """``theta`` is the counter-clockwise angle of the sigma_x axis in the
    array's own (x, y) coordinates.

    The textbook form of the rotated-Gaussian exponent is written for an
    image drawn with y pointing *down* and puts sigma_x at *minus* theta in
    array coordinates. Using it as-is left the drawn ellipse mirrored about
    the x-axis: for any tilted beam it crossed the beam instead of tracing
    it, and the reported angle had the wrong sign.
    """

    @staticmethod
    def _beam(theta_deg, n=400, sx=60.0, sy=20.0):
        y, x = np.mgrid[0:n, 0:n].astype(float)
        return fitting.gaussian_2d(
            (x, y), 200.0, n / 2, n / 2 - 10, sx, sy, np.radians(theta_deg), 0.0
        ).reshape(n, n)

    @pytest.mark.parametrize("theta_deg", [0, 20, 45, 70, 90, 120, 155])
    def test_the_model_puts_the_major_axis_where_theta_says(self, theta_deg):
        _, _, angle = _image_major_axis(self._beam(theta_deg))
        assert _angle_error(angle, theta_deg) < 0.5

    @pytest.mark.parametrize("theta_deg", [20, 45, 70, 120, 155])
    def test_the_fit_recovers_that_same_angle(self, theta_deg):
        img = self._beam(theta_deg)
        popt, converged = fitting.fit_2d_gaussian(img, None, sigma_hint=20.0)
        assert converged
        _, _, theta = fitting.canonical_ellipse(popt[3], popt[4], popt[5])
        assert _angle_error(np.degrees(theta), theta_deg) < 1.5

    def test_a_positive_angle_tilts_towards_increasing_y(self):
        """Pins the *sign*, which is the part that was wrong. A mirrored
        convention would still satisfy every magnitude check above."""
        img = self._beam(30.0)
        h, w = img.shape
        # Sample along the major axis in both directions from the centre.
        cy, cx = h / 2 - 10, w / 2
        r = 40.0
        col = int(cx + r * np.cos(np.radians(30)))
        along = img[int(cy + r * np.sin(np.radians(30))), col]
        mirrored = img[int(cy - r * np.sin(np.radians(30))), col]
        # ~3.7x on this beam; a mirrored convention would invert the ratio.
        assert along > mirrored * 2, "a +30 deg beam must extend towards larger y as x grows"


class TestMomentSeed:
    """The cold fit is seeded from image moments.

    Guessing theta = 0 and a tenth of the frame for each sigma left the solver
    across a ridge from the answer on a tilted beam: a 3:1 beam at 34 degrees
    came back as 14 x 11 at 0 degrees, reporting convergence.
    """

    @staticmethod
    def _beam(n=120, cx=60.0, cy=55.0, sx=18.0, sy=9.0, theta=0.6, offset=5.0):
        y, x = np.mgrid[0:n, 0:n].astype(float)
        return fitting.gaussian_2d((x, y), 200.0, cx, cy, sx, sy, theta, offset).reshape(n, n)

    def test_the_seed_is_already_close(self):
        seed = fitting._moment_seed(self._beam())
        assert seed is not None
        assert seed[1] == pytest.approx(60.0, abs=0.5)
        assert seed[2] == pytest.approx(55.0, abs=0.5)
        assert seed[3] == pytest.approx(18.0, rel=0.02)
        assert seed[4] == pytest.approx(9.0, rel=0.02)
        assert _angle_error(np.degrees(seed[5]), np.degrees(0.6)) < 1.0

    def test_the_seed_recovers_the_offset(self):
        seed = fitting._moment_seed(self._beam(offset=42.0))
        assert seed is not None
        assert seed[6] == pytest.approx(42.0, abs=1.0)

    def test_a_tilted_beam_no_longer_fits_to_an_axis_aligned_compromise(self):
        img = self._beam()
        popt, converged = fitting.fit_2d_gaussian(img, None)
        assert converged
        major, minor, theta = fitting.canonical_ellipse(popt[3], popt[4], popt[5])
        assert major == pytest.approx(18.0, rel=0.02)
        assert minor == pytest.approx(9.0, rel=0.02)
        assert _angle_error(np.degrees(theta), np.degrees(0.6)) < 1.5

    def test_a_blank_image_has_no_seed(self):
        assert fitting._moment_seed(np.zeros((32, 32))) is None

    def test_a_non_finite_image_has_no_seed(self):
        img = np.zeros((32, 32))
        img[4, 4] = np.inf
        assert fitting._moment_seed(img) is None

    def test_a_blank_image_still_fits_without_raising(self):
        """Falls back to the old peak-based guess."""
        popt, converged = fitting.fit_2d_gaussian(np.zeros((32, 32)), None)
        assert len(popt) == 7
        del converged


class TestEllipseTracesTheBeam:
    """The strongest end-to-end check available: sample the image where the
    overlay is drawn. On the 1/e^2 contour every sample must be at 13.5% of
    peak. A mirrored ellipse crosses the beam instead, swinging from the core
    to the background."""

    @pytest.mark.parametrize("theta_deg", [0, 20, 45, 70, 120, 155])
    def test_the_overlay_sits_on_the_contour(self, theta_deg):
        from pybeamprofiler.beamprofiler import BeamProfiler

        n = 400
        y, x = np.mgrid[0:n, 0:n].astype(float)
        img = fitting.gaussian_2d(
            (x, y), 200.0, 200.0, 190.0, 60.0, 20.0, np.radians(theta_deg), 0.0
        ).reshape(n, n)

        bp = BeamProfiler(camera="simulated", fit="2d")
        bp.pixel_size = 1.0
        bp.analyze(img)
        points = bp._ellipse_points()
        assert points is not None
        xs, ys = points

        xi = np.clip(np.round(xs).astype(int), 0, n - 1)
        yi = np.clip(np.round(ys).astype(int), 0, n - 1)
        sampled = img[yi, xi] / img.max()

        assert sampled.min() > 0.11, "part of the overlay is off the beam"
        assert sampled.max() < 0.17, "part of the overlay cuts through the core"
        assert bp.camera is not None
        bp.camera.close()
