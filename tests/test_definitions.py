"""Tests for width definitions: Gaussian (1/e²), FWHM, D4σ."""

import numpy as np
import pytest

from pybeamprofiler import BeamProfiler


class TestGaussianDefinition:
    """Test Gaussian (1/e²) width definition."""

    def test_gaussian_basic(self, beam_profiler):
        """Test default gaussian width definition."""
        bp = beam_profiler
        assert bp.camera is not None
        bp.fit_method = "1d"
        bp.definition = "gaussian"

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert popt_x is not None
        assert popt_y is not None
        assert bp.width_x > 0
        assert bp.width_y > 0

        sigma_x = popt_x[2]
        expected_width_x = 4.0 * sigma_x * bp.pixel_size
        assert abs(bp.width_x - expected_width_x) < 0.1


class TestFWHMDefinition:
    """Test Full Width at Half Maximum definition."""

    def test_fwhm_basic(self, beam_profiler):
        """Test FWHM width definition."""
        bp = beam_profiler
        assert bp.camera is not None
        bp.fit_method = "1d"
        bp.definition = "fwhm"

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert popt_x is not None
        assert popt_y is not None
        assert bp.width_x > 0
        assert bp.width_y > 0
        assert 500 < bp.width_x < 2000


class TestD4SigmaDefinition:
    """Test D4σ (ISO 11146 second moment) definition."""

    def test_d4s_basic(self, beam_profiler):
        """Test D4σ width definition."""
        bp = beam_profiler
        assert bp.camera is not None
        bp.fit_method = "1d"
        bp.definition = "d4s"

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        bp.analyze(img)

        assert bp.width_x > 0
        assert bp.width_y > 0
        assert 1000 < bp.width_x < 3100


class TestDefinitionComparisons:
    """Test relationships between different width definitions."""

    def test_definition_ordering(self, simulated_image):
        """Test that D4σ > Gaussian > FWHM."""
        bp_gaussian = BeamProfiler(camera="simulated", fit="1d", definition="gaussian")
        bp_fwhm = BeamProfiler(camera="simulated", fit="1d", definition="fwhm")
        bp_d4s = BeamProfiler(camera="simulated", fit="1d", definition="d4s")

        bp_gaussian.analyze(simulated_image)
        bp_fwhm.analyze(simulated_image)
        bp_d4s.analyze(simulated_image)

        # D4σ and Gaussian should be similar (both ~4σ), FWHM smallest (~2.355σ)
        assert abs(bp_d4s.width_x - bp_gaussian.width_x) / bp_gaussian.width_x < 0.05
        assert bp_gaussian.width_x > bp_fwhm.width_x

        bp_gaussian.camera.close()  # ty:ignore[unresolved-attribute]
        bp_fwhm.camera.close()  # ty:ignore[unresolved-attribute]
        bp_d4s.camera.close()  # ty:ignore[unresolved-attribute]

    def test_definition_ratios(self, simulated_image):
        """Test approximate ratios between definitions."""
        bp_gaussian = BeamProfiler(camera="simulated", fit="1d", definition="gaussian")
        bp_fwhm = BeamProfiler(camera="simulated", fit="1d", definition="fwhm")
        bp_d4s = BeamProfiler(camera="simulated", fit="1d", definition="d4s")

        bp_gaussian.analyze(simulated_image)
        bp_fwhm.analyze(simulated_image)
        bp_d4s.analyze(simulated_image)

        ratio_gauss_fwhm = bp_gaussian.width_x / bp_fwhm.width_x
        expected_ratio = 4.0 / 2.355
        assert abs(ratio_gauss_fwhm - expected_ratio) < 0.2

        bp_gaussian.camera.close()  # ty:ignore[unresolved-attribute]
        bp_fwhm.camera.close()  # ty:ignore[unresolved-attribute]
        bp_d4s.camera.close()  # ty:ignore[unresolved-attribute]

    def test_definition_switching(self, beam_profiler):
        """Test switching definitions on the fly."""
        bp = beam_profiler
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        bp.definition = "gaussian"
        bp.analyze(img)
        width_gaussian = bp.width_x

        bp.definition = "fwhm"
        bp.analyze(img)
        width_fwhm = bp.width_x

        bp.definition = "d4s"
        bp.analyze(img)
        width_d4s = bp.width_x

        assert width_gaussian != width_fwhm
        assert width_fwhm != width_d4s
        assert width_gaussian != width_d4s


class TestDefinitionsWithFittingMethods:
    """Test all definitions work with all fitting methods."""

    @pytest.mark.parametrize("fit_method", ["1d", "2d", "linecut"])
    @pytest.mark.parametrize("definition", ["gaussian", "fwhm", "d4s"])
    def test_definition_fit_combinations(self, fit_method, definition):
        """Test all definition/fit method combinations."""
        bp = BeamProfiler(camera="simulated", fit=fit_method, definition=definition)
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert bp.width_x > 0
        assert bp.width_y > 0

        bp.camera.close()


class TestDefinitionEdgeCases:
    """Test definitions with edge cases."""

    @pytest.mark.parametrize("definition", ["gaussian", "fwhm", "d4s"])
    def test_empty_image_all_definitions(self, definition):
        """Test empty images don't crash with any definition."""
        bp = BeamProfiler(camera="simulated", definition=definition)
        assert bp.camera is not None
        empty_img = np.zeros((100, 100))

        popt_x, popt_y = bp.analyze(empty_img)
        assert popt_x is not None
        assert popt_y is not None

        bp.camera.close()


class TestWidthDefinitions:
    """Test different width definitions: gaussian (1/e²), FWHM, D4σ."""

    def test_gaussian_definition(self):
        """Test default gaussian (1/e²) width definition."""
        bp = BeamProfiler(camera="simulated", fit="1d", definition="gaussian")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert popt_x is not None
        assert popt_y is not None
        assert bp.width_x > 0
        assert bp.width_y > 0

        # For gaussian definition, width should be 4*sigma (1/e²)
        # Verify the relationship
        sigma_x = popt_x[2]
        expected_width_x = 4.0 * sigma_x * bp.pixel_size
        assert abs(bp.width_x - expected_width_x) < 0.1

        bp.camera.close()

    def test_fwhm_definition(self):
        """Test FWHM (Full Width at Half Maximum) definition."""
        bp = BeamProfiler(camera="simulated", fit="1d", definition="fwhm")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)

        assert popt_x is not None
        assert popt_y is not None
        assert bp.width_x > 0
        assert bp.width_y > 0

        # FWHM is measured directly from half-maximum points, not from Gaussian fit
        # Just verify it's in a reasonable range
        assert 500 < bp.width_x < 2000  # Typical beam width range
        assert 500 < bp.width_y < 2000
        assert bp.width_x > 0
        assert bp.width_y > 0

        # D4σ is measured using second moment method, not from Gaussian fit
        # Just verify it's in a reasonable range
        assert 1000 < bp.width_x < 3000  # D4σ should be larger than FWHM
        assert 1000 < bp.width_y < 3000

        """Test that different definitions give different widths for same beam."""
        bp_gaussian = BeamProfiler(camera="simulated", fit="1d", definition="gaussian")
        bp_fwhm = BeamProfiler(camera="simulated", fit="1d", definition="fwhm")
        bp_d4s = BeamProfiler(camera="simulated", fit="1d", definition="d4s")

        # Use same image for all
        bp_gaussian.camera.start_acquisition()  # ty:ignore[unresolved-attribute]
        img = bp_gaussian.camera.get_image()  # ty:ignore[unresolved-attribute]
        bp_gaussian.camera.stop_acquisition()  # ty:ignore[unresolved-attribute]

        bp_gaussian.analyze(img)
        bp_fwhm.analyze(img)
        bp_d4s.analyze(img)

        # D4σ and Gaussian should be similar (both ~4σ), FWHM smallest (~2.355σ)
        # D4σ = 4σ ≈ 4.0
        # Gaussian = 4σ ≈ 4.0 (1/e²)
        # FWHM = 2.355σ ≈ 2.355

        # D4σ and Gaussian should be close (within 5%)
        assert abs(bp_d4s.width_x - bp_gaussian.width_x) / bp_gaussian.width_x < 0.05
        # Gaussian should be significantly larger than FWHM
        assert bp_gaussian.width_x > bp_fwhm.width_x

        # Check approximate ratios (Gaussian vs FWHM)
        ratio_gauss_fwhm = bp_gaussian.width_x / bp_fwhm.width_x
        expected_ratio = 4.0 / 2.355  # ≈ 1.7
        assert abs(ratio_gauss_fwhm - expected_ratio) < 0.2

        bp_gaussian.camera.close()  # ty:ignore[unresolved-attribute]
        bp_fwhm.camera.close()  # ty:ignore[unresolved-attribute]
        bp_d4s.camera.close()  # ty:ignore[unresolved-attribute]

    def test_2d_fitting_with_definitions(self):
        """Test 2D fitting works with all width definitions."""
        for definition in ["gaussian", "fwhm", "d4s"]:
            bp = BeamProfiler(camera="simulated", fit="2d", definition=definition)
            assert bp.camera is not None
            bp.camera.start_acquisition()
            img = bp.camera.get_image()
            bp.camera.stop_acquisition()

            popt_x, popt_y = bp.analyze(img)

            assert bp.width_x > 0, f"Failed for definition={definition}"
            assert bp.width_y > 0, f"Failed for definition={definition}"
            assert hasattr(bp, "angle_deg"), f"No angle for definition={definition}"

            bp.camera.close()


class TestExposureHandling:
    """Test exposure time handling fixes."""

    def test_none_exposure_single_shot(self):
        """Test that None exposure_time doesn't crash for single shot."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        # Test exposure handling without calling plot()
        bp.camera.set_exposure(None)  # ty:ignore[invalid-argument-type]
        assert bp.camera.exposure_time == 0.01

        # Get and analyze an image
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        bp.analyze(img)

        assert bp.width > 0
        bp.camera.close()

    def test_none_exposure_continuous(self):
        """Test that None exposure_time works for continuous (uses default)."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        # Should use default 0.01s
        bp.camera.set_exposure(None)  # ty:ignore[invalid-argument-type]
        assert bp.camera.exposure_time == 0.01
        bp.camera.close()

    def test_exposure_affects_amplitude(self):
        """Test that exposure time affects simulated amplitude."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        # Short exposure
        bp.camera.set_exposure(0.001)
        bp.camera.start_acquisition()
        img1 = bp.camera.get_image()
        bp.camera.stop_acquisition()

        # Long exposure
        bp.camera.set_exposure(0.1)
        bp.camera.start_acquisition()
        img2 = bp.camera.get_image()
        bp.camera.stop_acquisition()

        # Long exposure should have higher peak
        assert np.max(img2) > np.max(img1)

        bp.camera.close()


class TestPropertiesWithDefinitions:
    """Test that properties work correctly with different definitions."""

    def test_properties_gaussian(self):
        """Test all properties with gaussian definition."""
        bp = BeamProfiler(camera="simulated", fit="1d", definition="gaussian")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        bp.analyze(img)

        # All properties should be accessible
        assert bp.width > 0
        assert bp.diameter == bp.width
        assert bp.radius == bp.width / 2
        assert bp.fwhm_x > 0
        assert bp.fwhm_y > 0
        assert bp.fw_1e_x > 0
        assert bp.fw_1e_y > 0
        assert bp.fw_1e2_x > 0
        assert bp.fw_1e2_y > 0
        assert bp.height_x > 0
        assert bp.height_y > 0
        assert bp.peak_value > 0

        bp.camera.close()

    def test_properties_fwhm(self):
        """Test that width_x/y change but properties remain consistent with FWHM."""
        bp = BeamProfiler(camera="simulated", fit="1d", definition="fwhm")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        bp.analyze(img)

        # Width should now be FWHM
        assert bp.width > 0
        # But fwhm_x property should still be accessible and give similar value
        # (fwhm_x recalculates from width_x assuming gaussian)
        assert bp.fwhm_x > 0

        bp.camera.close()


class TestHarvesterIntegration:
    """Test Harvesters/GenICam integration."""

    def test_flir_camera_import(self):
        """Test that FlirCamera can be imported and initialized."""
        from pybeamprofiler.flir import FlirCamera

        # Should not crash on import
        # Actual connection will fail without hardware, but class should exist
        assert FlirCamera is not None

    def test_basler_camera_import(self):
        """Test that BaslerCamera can be imported and initialized."""
        from pybeamprofiler.basler import BaslerCamera

        # Should not crash on import
        assert BaslerCamera is not None

    def test_harvester_camera_import(self):
        """Test that HarvesterCamera can be imported."""
        from pybeamprofiler.gen_camera import HarvesterCamera

        assert HarvesterCamera is not None

    def test_camera_type_selection(self):
        """Test that camera type strings work correctly."""
        # Simulated should still work
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.close()

        # FLIR/Basler will fall back to simulated if hardware not available
        # But should not crash
        try:
            bp_flir = BeamProfiler(camera="flir")
            if bp_flir.camera:
                bp_flir.camera.close()
        except Exception:
            # Camera initialization fails without hardware or drivers
            pass

        try:
            bp_basler = BeamProfiler(camera="basler")
            if bp_basler.camera:
                bp_basler.camera.close()
        except Exception:
            # Camera initialization fails without hardware or drivers
            pass


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_image_with_definitions(self):
        """Test that empty images don't crash with different definitions."""
        for definition in ["gaussian", "fwhm", "d4s"]:
            bp = BeamProfiler(definition=definition)
            img = np.zeros((100, 100))
            # Should not crash, even if it returns some fit (scipy can fit zeros)
            popt_x, popt_y = bp.analyze(img)
            # Just verify it doesn't crash and returns something
            assert popt_x is not None
            assert popt_y is not None

    def test_definition_switching(self):
        """Test switching definition on the fly."""
        bp = BeamProfiler(camera="simulated", definition="gaussian")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        bp.analyze(img)
        width_gaussian = bp.width_x

        # Switch to FWHM
        bp.definition = "fwhm"
        bp.analyze(img)
        width_fwhm = bp.width_x

        # Switch to D4σ
        bp.definition = "d4s"
        bp.analyze(img)
        width_d4s = bp.width_x

        # All should be different
        assert width_gaussian != width_fwhm
        assert width_fwhm != width_d4s
        assert width_gaussian != width_d4s

        bp.camera.close()
