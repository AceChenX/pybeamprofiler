"""Tests for Gaussian fitting methods (1D, 2D, linecut)."""

from unittest.mock import patch

import numpy as np

from pybeamprofiler import BeamProfiler, fitting
from pybeamprofiler.simulated import SimulatedCamera


def _seeded_camera(seed: int) -> SimulatedCamera:
    camera = SimulatedCamera(seed=seed)
    camera.open()
    return camera


class TestOneDimensionalFitting:
    """Test 1D projection-based Gaussian fitting."""

    def test_1d_fitting_basic(self, beam_profiler):
        """Test 1D fitting produces valid results."""
        bp = beam_profiler
        assert bp.camera is not None
        bp.fit_method = "1d"

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert popt_x is not None
        assert popt_y is not None
        assert len(popt_x) == 4
        assert len(popt_y) == 4

        assert bp.width_x > 0
        assert bp.width_y > 0
        assert 0 < bp.center_x < bp.width_pixels
        assert 0 < bp.center_y < bp.height_pixels
        assert bp.peak_value > 0

    def test_1d_fit_speed(self, beam_profiler, simulated_image):
        """Test 1D fitting is fast enough for real-time (< 100ms)."""
        import time

        bp = beam_profiler
        bp.fit_method = "1d"

        start = time.time()
        bp.analyze(simulated_image)
        elapsed = time.time() - start

        assert elapsed < 0.1, f"1D fit took {elapsed * 1000:.1f}ms"

    def test_1d_fit_caching(self, beam_profiler):
        """Test fit parameter caching improves convergence."""
        bp = beam_profiler
        assert bp.camera is not None
        bp.fit_method = "1d"

        bp.camera.start_acquisition()
        img1 = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x1, _ = bp.analyze(img1)
        assert bp._last_popt_x is not None

        bp.camera.start_acquisition()
        img2 = bp.camera.get_image()
        bp.camera.stop_acquisition()

        # The second frame must start the solver from the first frame's
        # result. Checked on the solver itself: the fit converging either
        # way would hide a warm start that was never used.
        cached = np.asarray(bp._last_popt_x, dtype=float).copy()
        starts: list[np.ndarray] = []
        real_curve_fit = fitting.curve_fit

        def spy(*args, **kwargs):
            starts.append(np.asarray(kwargs["p0"], dtype=float))
            return real_curve_fit(*args, **kwargs)

        with patch("pybeamprofiler.fitting.curve_fit", side_effect=spy):
            bp.analyze(img2)
        np.testing.assert_allclose(starts[0], cached)


class TestTwoDimensionalFitting:
    """Test 2D Gaussian fitting with rotation."""

    def test_2d_fitting_basic(self, beam_profiler):
        """Test 2D fitting with angle detection."""
        bp = beam_profiler
        assert bp.camera is not None
        bp.fit_method = "2d"

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert popt_x is not None
        assert popt_y is not None
        assert bp.width_x > 0
        assert bp.width_y > 0
        assert 0 <= bp.angle_deg < 180

    def test_2d_vs_1d_comparison(self, beam_profiler):
        """Compare 2D and 1D fitting results on same image."""
        bp = beam_profiler
        # Seeded, so a failure reproduces instead of showing up once in 400 runs.
        bp.attach_camera(_seeded_camera(7))
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        bp.fit_method = "1d"
        bp.analyze(img)
        width_1d = bp.width

        bp.fit_method = "2d"
        bp.analyze(img)
        width_2d = bp.width

        assert abs(width_1d - width_2d) / width_1d < 0.1


class TestLinecutFitting:
    """Test linecut fitting through beam peak."""

    def test_linecut_fitting_basic(self):
        """Test linecut fitting produces valid results."""
        bp = BeamProfiler(camera="simulated", fit="linecut")
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert popt_x is not None
        assert popt_y is not None
        assert bp.width_x > 0
        assert bp.width_y > 0
        assert bp.angle_deg == 0.0

        bp.camera.close()

    def test_linecut_vs_1d_comparison(self, simulated_image):
        """Compare linecut and 1D projection methods."""
        bp_linecut = BeamProfiler(camera="simulated", fit="linecut")
        bp_1d = BeamProfiler(camera="simulated", fit="1d")

        bp_linecut.analyze(simulated_image)
        width_linecut = bp_linecut.width

        bp_1d.analyze(simulated_image)
        width_1d = bp_1d.width

        assert width_linecut > 0
        assert width_1d > 0
        assert abs(width_1d - width_linecut) / width_1d < 0.5

        bp_linecut.camera.close()  # ty:ignore[unresolved-attribute]
        bp_1d.camera.close()  # ty:ignore[unresolved-attribute]


class TestFittingEdgeCases:
    """Test fitting with edge cases and invalid data."""

    def test_empty_image(self, beam_profiler):
        """A blank frame has no beam, and says so instead of raising."""
        bp = beam_profiler
        popt_x, popt_y = bp.analyze(np.zeros((100, 100)))
        assert popt_x is None and popt_y is None
        assert np.isnan(bp.width_x) and np.isnan(bp.width_y)

    def test_noisy_image(self, beam_profiler):
        """Pure noise is not a beam, however well a Gaussian can be fitted to it."""
        bp = beam_profiler
        rng = np.random.default_rng(3)
        noisy_img = rng.integers(0, 50, (100, 100), dtype=np.uint8)

        popt_x, popt_y = bp.analyze(noisy_img)
        assert popt_x is None and popt_y is None
        assert np.isnan(bp.width_x) and np.isnan(bp.width_y)
        assert bp.beam_ellipse() is None

    def test_multiple_consecutive_fits(self, beam_profiler):
        """Test multiple consecutive fits produce consistent results."""
        bp = beam_profiler
        assert bp.camera is not None
        bp.fit_method = "1d"

        bp.camera.start_acquisition()

        widths = []
        for _ in range(10):
            img = bp.camera.get_image()
            bp.analyze(img)
            widths.append(bp.width)

        bp.camera.stop_acquisition()

        assert len(widths) == 10
        assert all(w > 0 for w in widths)
        assert np.std(widths) > 0
