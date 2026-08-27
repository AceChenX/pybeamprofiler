"""Regression tests for bugs found during review.

Each test here maps to a specific defect and states what used to go wrong, so
a future change that reintroduces it fails with an explanation rather than a
bare assertion.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from pybeamprofiler import fitting
from pybeamprofiler.beamprofiler import BeamProfiler, main
from pybeamprofiler.dash_app import _averaged_image, _reset_avg_state, _saturation_fraction
from pybeamprofiler.simulated import SimulatedCamera


def _rotated_beam(h=200, w=200, x0=100.0, y0=100.0, sx=30.0, sy=10.0, theta_deg=40.0):
    """An elongated beam at an angle — the case that exposes ellipse bugs."""
    y, x = np.mgrid[0:h, 0:w]
    flat = fitting.gaussian_2d((x, y), 200.0, x0, y0, sx, sy, np.deg2rad(theta_deg), 0.0)
    return flat.reshape(h, w).astype(np.uint8)


class TestBeamEllipseUsesTheRightFit:
    """The 2D overlay used to be drawn from the 1D projection sigmas.

    Projecting a tilted beam onto the axes pulls both widths toward each
    other, so a 3:1 beam at 40° was drawn as a nearly circular ellipse that
    merely happened to be rotated 40°.
    """

    def test_2d_ellipse_matches_the_2d_fit_not_the_projections(self):
        bp = BeamProfiler(camera="simulated", fit="2d")
        img = _rotated_beam()
        popt_x, popt_y = bp.analyze(img)

        ellipse = bp.beam_ellipse()
        assert ellipse is not None
        cx, cy, rx, ry, angle = ellipse

        assert cx == pytest.approx(100.0, abs=1.0)
        assert cy == pytest.approx(100.0, abs=1.0)
        # 1/e² semi-axes are 2 sigma.
        assert rx == pytest.approx(2 * 30.0, rel=0.05)
        assert ry == pytest.approx(2 * 10.0, rel=0.05)
        assert np.degrees(angle) % 180 == pytest.approx(40.0, abs=2.0)

        # And it is genuinely different from what the projections would give.
        proj_rx, proj_ry = 2 * abs(popt_x[2]), 2 * abs(popt_y[2])
        assert abs(rx - proj_rx) > 5.0
        assert abs(ry - proj_ry) > 5.0

    def test_2d_ellipse_is_actually_elongated(self):
        """The projection-based ellipse was nearly circular; this one isn't."""
        bp = BeamProfiler(camera="simulated", fit="2d")
        bp.analyze(_rotated_beam())
        ellipse = bp.beam_ellipse()
        assert ellipse is not None
        _, _, rx, ry, _ = ellipse
        assert max(rx, ry) / min(rx, ry) == pytest.approx(3.0, rel=0.1)

    def test_1d_mode_uses_the_axis_fits(self):
        bp = BeamProfiler(camera="simulated", fit="1d")
        popt_x, popt_y = bp.analyze(_rotated_beam())
        ellipse = bp.beam_ellipse()
        assert ellipse is not None
        cx, cy, rx, ry, angle = ellipse
        assert angle == 0.0
        assert cx == pytest.approx(popt_x[1])
        assert rx == pytest.approx(2 * abs(popt_x[2]))
        assert cy == pytest.approx(popt_y[1])
        assert ry == pytest.approx(2 * abs(popt_y[2]))

    def test_no_fit_yet_returns_none(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.beam_ellipse() is None
        assert bp._ellipse_points() is None

    def test_2d_mode_without_a_2d_fit_falls_back(self):
        """FWHM/D4σ skip the 2D fit entirely; the ellipse must still resolve."""
        bp = BeamProfiler(camera="simulated", fit="2d", definition="d4s")
        bp.analyze(_rotated_beam())
        assert bp._last_popt_2d is None
        ellipse = bp.beam_ellipse()
        assert ellipse is not None
        assert ellipse[4] == 0.0

    def test_ellipse_points_are_in_micrometers(self):
        bp = BeamProfiler(camera="simulated", fit="1d")
        bp.pixel_size = 5.0
        bp.analyze(_rotated_beam())
        ellipse = bp.beam_ellipse()
        points = bp._ellipse_points()
        assert ellipse is not None and points is not None
        cx, _, rx, _, _ = ellipse
        xs, ys = points
        assert xs.max() == pytest.approx((cx + rx) * 5.0, rel=1e-6)
        assert len(xs) == len(ys) == 100


class TestFailedFitDoesNotPoisonTheCache:
    """A non-converging 2D fit used to leave its own bad guess cached."""

    def test_last_popt_2d_unchanged_after_failure(self):
        bp = BeamProfiler(camera="simulated", fit="2d")
        img = _rotated_beam()
        bp.analyze(img)
        good = np.array(bp._last_popt_2d, dtype=float).copy()

        with patch("pybeamprofiler.fitting.curve_fit", side_effect=RuntimeError("nope")):
            bp._fit_2d_gaussian(img)

        np.testing.assert_allclose(np.asarray(bp._last_popt_2d, dtype=float), good)


class TestStaticImageLoading:
    """``--file`` used to reject colour images with a 3D-array error."""

    def _write(self, tmp_path, array, name="beam.png"):
        path = tmp_path / name
        Image.fromarray(array).save(path)
        return str(path)

    def test_rgb_image_is_converted_to_grayscale(self, tmp_path):
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        y, x = np.mgrid[0:64, 0:96]
        blob = (200 * np.exp(-((x - 48) ** 2 + (y - 32) ** 2) / (2 * 12.0**2))).astype(np.uint8)
        rgb[..., 0] = blob
        rgb[..., 1] = blob
        rgb[..., 2] = blob

        bp = BeamProfiler(file=self._write(tmp_path, rgb), pixel_size=5.0)
        assert bp.last_img is not None
        assert bp.last_img.ndim == 2
        assert bp.last_img.shape == (64, 96)
        assert bp.width_pixels == 96
        assert bp.height_pixels == 64
        bp.analyze(bp.last_img)  # must not raise
        assert bp.width_x > 0

    def test_rgba_image_drops_alpha(self, tmp_path):
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., :3] = 120
        rgba[..., 3] = 255
        bp = BeamProfiler(file=self._write(tmp_path, rgba), pixel_size=1.0)
        assert bp.last_img is not None
        assert bp.last_img.ndim == 2
        assert bp.last_img.max() == pytest.approx(120, abs=1)

    def test_luminance_weights_are_applied(self, tmp_path):
        """Pure green must read brighter than pure blue, per ITU-R weights."""
        green = np.zeros((16, 16, 3), dtype=np.uint8)
        green[..., 1] = 255
        blue = np.zeros((16, 16, 3), dtype=np.uint8)
        blue[..., 2] = 255

        g = BeamProfiler(file=self._write(tmp_path, green, "g.png"), pixel_size=1.0)
        b = BeamProfiler(file=self._write(tmp_path, blue, "b.png"), pixel_size=1.0)
        assert g.last_img is not None and b.last_img is not None
        assert g.last_img.max() > b.last_img.max()

    def test_grayscale_image_is_untouched(self, tmp_path):
        gray = np.random.randint(0, 255, (40, 50), dtype=np.uint8)
        bp = BeamProfiler(file=self._write(tmp_path, gray), pixel_size=2.0)
        np.testing.assert_array_equal(bp.last_img, gray)

    def test_pixel_size_still_required(self, tmp_path):
        gray = np.zeros((16, 16), dtype=np.uint8)
        with pytest.raises(ValueError, match="Pixel size must be provided"):
            BeamProfiler(file=self._write(tmp_path, gray))

    def test_pixel_size_must_be_positive(self, tmp_path):
        gray = np.zeros((16, 16), dtype=np.uint8)
        with pytest.raises(ValueError, match="greater than zero"):
            BeamProfiler(file=self._write(tmp_path, gray), pixel_size=0)


class TestCliPixelSize:
    """``pybeamprofiler --file beam.png`` used to crash unconditionally.

    ``__init__`` requires a pixel size for static files, but the parser had no
    way to supply one — so the documented example was guaranteed to raise.
    """

    def test_file_without_pixel_size_exits_with_guidance(self, tmp_path, capsys):
        img = tmp_path / "beam.png"
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(img)

        with patch.object(sys, "argv", ["pybeamprofiler", "--file", str(img)]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 2
        assert "--pixel-size is required" in capsys.readouterr().err

    def test_file_with_pixel_size_runs(self, tmp_path):
        y, x = np.mgrid[0:64, 0:64]
        blob = (200 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 10.0**2))).astype(np.uint8)
        img = tmp_path / "beam.png"
        Image.fromarray(blob).save(img)

        argv = ["pybeamprofiler", "--file", str(img), "--pixel-size", "5.86"]
        with patch.object(sys, "argv", argv), patch.object(BeamProfiler, "plot") as plot:
            main()
        plot.assert_called_once()

    def test_non_positive_pixel_size_is_rejected(self, tmp_path, capsys):
        img = tmp_path / "beam.png"
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(img)
        argv = ["pybeamprofiler", "--file", str(img), "--pixel-size", "0"]
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                main()
        assert "greater than zero" in capsys.readouterr().err

    def test_pixel_size_overrides_the_camera(self):
        """A camera reports its own pitch, but the flag wins when given."""
        seen = {}

        def capture(self, *args, **kwargs):
            seen["pixel_size"] = self.pixel_size

        argv = ["pybeamprofiler", "--camera", "simulated", "--pixel-size", "99"]
        with patch.object(sys, "argv", argv), patch.object(BeamProfiler, "plot", capture):
            main()
        assert seen["pixel_size"] == 99.0

    def test_camera_pitch_is_used_when_the_flag_is_absent(self):
        seen = {}

        def capture(self, *args, **kwargs):
            seen["pixel_size"] = self.pixel_size
            seen["camera_pixel_size"] = self.camera.pixel_size

        argv = ["pybeamprofiler", "--camera", "simulated"]
        with patch.object(sys, "argv", argv), patch.object(BeamProfiler, "plot", capture):
            main()
        assert seen["pixel_size"] == seen["camera_pixel_size"]


class TestPixelSizeOverride:
    """An explicit pixel_size used to be silently discarded for cameras."""

    def test_override_wins_over_the_camera_value(self):
        bp = BeamProfiler(camera="simulated", pixel_size=3.45)
        assert bp.pixel_size == 3.45
        assert bp.camera is not None
        assert bp.camera.pixel_size != 3.45  # camera itself untouched

    def test_camera_value_used_when_not_overridden(self):
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        assert bp.pixel_size == bp.camera.pixel_size

    def test_override_feeds_through_to_widths(self):
        img = _rotated_beam(sx=20.0, sy=20.0, theta_deg=0.0)
        one = BeamProfiler(camera="simulated", pixel_size=1.0)
        two = BeamProfiler(camera="simulated", pixel_size=2.0)
        one.analyze(img)
        two.analyze(img)
        assert two.width_x == pytest.approx(2 * one.width_x, rel=1e-9)


class TestTinyImages:
    """Profiles shorter than the parameter count raise TypeError from
    curve_fit, which used to escape ``analyze``."""

    @pytest.mark.parametrize("size", [1, 2, 3, 4])
    def test_analyze_survives_tiny_images(self, size):
        bp = BeamProfiler(camera="simulated")
        img = np.full((size, size), 100, dtype=np.uint8)
        popt_x, popt_y = bp.analyze(img)  # must not raise
        assert len(popt_x) == 4
        assert len(popt_y) == 4

    @pytest.mark.parametrize("definition", ["gaussian", "fwhm", "d4s"])
    def test_tiny_images_for_every_definition(self, definition):
        bp = BeamProfiler(camera="simulated", definition=definition)
        bp.analyze(np.full((2, 2), 50, dtype=np.uint8))
        assert np.isfinite(bp.width_x)

    def test_tiny_image_in_2d_mode(self):
        bp = BeamProfiler(camera="simulated", fit="2d")
        bp.analyze(np.full((3, 3), 50, dtype=np.uint8))
        assert np.isfinite(bp.width_x)

    def test_tiny_image_in_linecut_mode(self):
        bp = BeamProfiler(camera="simulated", fit="linecut")
        bp.analyze(np.full((3, 3), 50, dtype=np.uint8))
        assert np.isfinite(bp.width_x)


class TestSimulatedCameraRoi:
    """set_roi updated ``width``/``height`` but not the ``*_pixels`` aliases,
    so the Dash Camera Info panel kept reporting the full sensor."""

    def test_roi_updates_both_name_pairs(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_roi(offset_x=100, offset_y=50, width=256, height=128)

        assert (cam.width, cam.height) == (256, 128)
        assert (cam.width_pixels, cam.height_pixels) == (256, 128)
        cam.close()

    def test_roi_matches_the_frame_actually_returned(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_roi(offset_x=10, offset_y=20, width=200, height=100)
        img = cam.get_image()
        assert img.shape == (cam.height_pixels, cam.width_pixels)
        cam.close()

    def test_reset_to_full_sensor(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_roi(width=64, height=64)
        cam.set_roi()
        assert cam.width_pixels == cam.roi_info["max_width"]
        assert cam.height_pixels == cam.roi_info["max_height"]
        cam.close()


class TestSimulatedCameraAmplitude:
    """Exposure and gain each recomputed amplitude from scratch, so whichever
    was set last silently undid the other."""

    def test_gain_does_not_undo_exposure(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_exposure(0.05)  # 5x default
        after_exposure = cam._amplitude
        cam.set_gain(10.0)  # doubles amplitude
        assert cam._amplitude == pytest.approx(after_exposure * 2.0)
        cam.close()

    def test_exposure_does_not_undo_gain(self):
        cam = SimulatedCamera()
        cam.open()
        cam.set_gain(10.0)
        cam.set_exposure(0.02)  # 2x default
        base = cam._amplitude
        cam.set_exposure(0.04)  # 4x default
        assert cam._amplitude == pytest.approx(base * 2.0)
        cam.close()

    def test_order_of_application_does_not_matter(self):
        a, b = SimulatedCamera(), SimulatedCamera()
        a.open()
        b.open()
        a.set_exposure(0.03)
        a.set_gain(6.0)
        b.set_gain(6.0)
        b.set_exposure(0.03)
        assert a._amplitude == pytest.approx(b._amplitude)
        a.close()
        b.close()

    def test_brighter_settings_brighten_the_frame(self):
        cam = SimulatedCamera()
        cam.open()
        cam._noise_image = 0.0
        cam._noise_amp = 0.0
        cam.set_exposure(0.001)
        dim = int(cam.get_image().max())
        cam.set_gain(20.0)
        assert int(cam.get_image().max()) > dim
        cam.close()


class TestFrameAveraging:
    """The running mean was truncated toward zero, darkening every frame by
    about half a count the moment averaging was switched on."""

    def setup_method(self):
        _reset_avg_state()

    def teardown_method(self):
        _reset_avg_state()

    def test_mean_is_rounded_not_truncated(self):
        # Alternating 10 / 11 averages to 10.5, which must round to 10 or 11
        # rather than always flooring to 10.
        a = np.full((4, 4), 10, dtype=np.uint8)
        b = np.full((4, 4), 11, dtype=np.uint8)
        _averaged_image(a, 2)
        out = _averaged_image(b, 2)
        assert out.max() >= 10
        assert np.rint(10.5) == out[0, 0]

    def test_constant_input_is_preserved_exactly(self):
        """Truncation used to shave a count off a perfectly steady signal."""
        frame = np.full((8, 8), 200, dtype=np.uint8)
        for _ in range(6):
            out = _averaged_image(frame, 4)
        np.testing.assert_array_equal(out, frame)

    def test_float_frames_are_not_rounded(self):
        a = np.full((4, 4), 10.0, dtype=np.float32)
        b = np.full((4, 4), 11.0, dtype=np.float32)
        _averaged_image(a, 2)
        out = _averaged_image(b, 2)
        assert out[0, 0] == pytest.approx(10.5)

    def test_n_of_one_is_a_passthrough(self):
        frame = np.full((4, 4), 33, dtype=np.uint8)
        assert _averaged_image(frame, 1) is frame

    def test_window_slides(self):
        """Old frames must drop out once the window is full."""
        low = np.full((4, 4), 0, dtype=np.uint8)
        high = np.full((4, 4), 100, dtype=np.uint8)
        _averaged_image(low, 2)
        _averaged_image(high, 2)
        out = _averaged_image(high, 2)
        np.testing.assert_array_equal(out, high)


class TestSaturationFastPath:
    """The saturation scan short-circuits on a cheap max() reduction; it must
    give the same answer as the full comparison."""

    def test_clean_frame_reports_nothing(self):
        assert _saturation_fraction(np.full((64, 64), 200, dtype=np.uint8)) == 0.0

    def test_fully_saturated_frame(self):
        assert _saturation_fraction(np.full((10, 10), 255, dtype=np.uint8)) == 1.0

    def test_partial_saturation_is_exact(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[:2, :] = 255  # 20 of 100 pixels
        assert _saturation_fraction(img) == pytest.approx(0.2)

    def test_one_hot_pixel_is_not_missed(self):
        """The max() shortcut must not skip a single saturated pixel."""
        img = np.zeros((256, 256), dtype=np.uint8)
        img[123, 45] = 255
        assert _saturation_fraction(img) == pytest.approx(1 / 65536)

    def test_uint16_uses_its_own_ceiling(self):
        img = np.full((8, 8), 255, dtype=np.uint16)
        assert _saturation_fraction(img) == 0.0
        img[0, 0] = 65535
        assert _saturation_fraction(img) == pytest.approx(1 / 64)

    def test_normalised_float_frames(self):
        img = np.zeros((10, 10), dtype=np.float64)
        img[0, :] = 1.0
        assert _saturation_fraction(img) == pytest.approx(0.1)

    def test_empty_frame(self):
        assert _saturation_fraction(np.array([], dtype=np.uint8).reshape(0, 0)) == 0.0


class TestFeatureDiscoveryCache:
    """Discovery probes ``.value`` on every node, which is a register read on
    real hardware. It is memoised per node map."""

    def test_second_call_is_served_from_cache(self):
        cam = SimulatedCamera()
        cam.open()
        first = cam._discover_features()
        assert first
        assert cam._discover_features() is first
        cam.close()

    def test_no_node_map_is_not_cached(self):
        cam = SimulatedCamera()
        assert cam._discover_features() == {}
        assert cam._feature_cache is None

    def test_replacing_the_node_map_invalidates(self):
        cam = SimulatedCamera()
        cam.open()
        first = cam._discover_features()
        cam.open()  # builds a fresh node map
        second = cam._discover_features()
        assert second is not first
        assert set(second) == set(first)
        cam.close()

    def test_refresh_forces_a_rebuild(self):
        cam = SimulatedCamera()
        cam.open()
        first = cam._discover_features()
        second = cam._discover_features(refresh=True)
        assert second is not first
        assert second == first
        cam.close()

    def test_cache_does_not_leak_across_instances(self):
        a, b = SimulatedCamera(), SimulatedCamera()
        a.open()
        a._discover_features()
        assert b._feature_cache is None
        a.close()


class TestSimulatedProfiles:
    """Multiple simulated cameras exist so the selector can be exercised
    without hardware. Each must behave like its own device."""

    def test_default_matches_the_historical_constants(self):
        """``SimulatedCamera()`` with no argument must not have changed."""
        from pybeamprofiler import constants
        from pybeamprofiler.simulated import SimulatedCamera

        cam = SimulatedCamera()
        assert (cam.width, cam.height) == (constants.SIMULATED_WIDTH, constants.SIMULATED_HEIGHT)
        assert cam.pixel_size == constants.SIMULATED_PIXEL_SIZE
        assert cam._sigma_x == constants.SIMULATED_SIGMA_X
        assert cam._sigma_y == constants.SIMULATED_SIGMA_Y
        cam.close()

    @pytest.mark.parametrize("profile_index", [0, 1])
    def test_frame_shape_follows_the_profile(self, profile_index):
        from pybeamprofiler.simulated import SIMULATED_PROFILES, SimulatedCamera

        profile = SIMULATED_PROFILES[profile_index]
        cam = SimulatedCamera(profile)
        cam.open()
        assert cam.get_image().shape == (profile.height, profile.width)
        cam.close()

    @pytest.mark.parametrize("profile_index", [0, 1])
    def test_roi_still_crops_correctly(self, profile_index):
        """The frame is rendered at full sensor size and then cropped.

        ``width``/``height`` track the ROI while the render buffer stays at
        sensor size — mixing the two up broadcasts a full-sensor noise array
        against an ROI-sized buffer and raises.
        """
        from pybeamprofiler.simulated import SIMULATED_PROFILES, SimulatedCamera

        cam = SimulatedCamera(SIMULATED_PROFILES[profile_index])
        cam.open()
        cam.set_roi(offset_x=10, offset_y=20, width=64, height=48)

        img = cam.get_image()
        assert img.shape == (48, 64)
        assert img.shape == (cam.height_pixels, cam.width_pixels)
        cam.close()

    @pytest.mark.parametrize("profile_index", [0, 1])
    def test_roi_clamps_to_this_profile_sensor(self, profile_index):
        from pybeamprofiler.simulated import SIMULATED_PROFILES, SimulatedCamera

        profile = SIMULATED_PROFILES[profile_index]
        cam = SimulatedCamera(profile)
        cam.open()
        cam.set_roi(offset_x=0, offset_y=0, width=99_999, height=99_999)

        assert cam.roi_info["max_width"] == profile.width
        assert cam.roi_info["max_height"] == profile.height
        assert cam.get_image().shape == (profile.height, profile.width)
        cam.close()

    def test_tilted_profile_produces_a_rotated_beam(self):
        """The 2D fit must recover the tilt the profile asked for."""
        from pybeamprofiler.simulated import SIMULATED_PROFILES, SimulatedCamera

        tilted = next(p for p in SIMULATED_PROFILES if p.theta_deg)
        cam = SimulatedCamera(tilted)
        cam.open()

        bp = BeamProfiler(camera="simulated", fit="2d")
        bp.attach_camera(cam)
        for _ in range(4):  # let the warm-started fit settle
            bp.analyze(cam.get_image())

        assert bp.angle_deg == pytest.approx(tilted.theta_deg, abs=5.0)
        assert max(bp.width_x, bp.width_y) / min(bp.width_x, bp.width_y) > 2.0
        cam.close()

    def test_node_map_reports_this_profile(self):
        from pybeamprofiler.simulated import SIMULATED_PROFILES, SimulatedCamera

        profile = SIMULATED_PROFILES[1]
        cam = SimulatedCamera(profile)
        cam.open()
        nm = cam.node_map
        assert nm is not None
        assert nm.DeviceModelName.value == profile.name
        assert nm.DeviceSerialNumber.value == profile.serial_number
        assert nm.WidthMax.value == profile.width
        assert nm.HeightMax.value == profile.height
        cam.close()

    def test_profiles_have_unique_keys_and_serials(self):
        from pybeamprofiler.simulated import SIMULATED_PROFILES

        assert len({p.key for p in SIMULATED_PROFILES}) == len(SIMULATED_PROFILES)
        assert len({p.serial_number for p in SIMULATED_PROFILES}) == len(SIMULATED_PROFILES)
