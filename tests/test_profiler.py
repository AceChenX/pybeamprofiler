"""Tests for BeamProfiler properties, attributes, and integration."""

from pybeamprofiler import BeamProfiler


class TestBeamProfilerProperties:
    """Test BeamProfiler computed properties."""

    def test_basic_properties(self, beam_profiler):
        """Test basic width and diameter properties."""
        bp = beam_profiler

        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()

        bp.analyze(img)

        assert bp.width > 0
        assert bp.diameter == bp.width
        assert bp.radius == bp.width / 2
        assert bp.width_x > 0
        assert bp.width_y > 0
        assert bp.peak_value > 0

    def test_laseview_properties(self, beam_profiler):
        """Test LaseView-compatible width properties."""
        bp = beam_profiler

        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()

        bp.analyze(img)

        assert bp.fwhm_x > 0
        assert bp.fwhm_y > 0
        assert bp.fw_1e_x > 0
        assert bp.fw_1e_y > 0
        assert bp.fw_1e2_x == bp.width_x
        assert bp.fw_1e2_y == bp.width_y
        assert bp.height_x > 0
        assert bp.height_y > 0

    def test_property_relationships(self, beam_profiler):
        """Test relationships between width properties."""
        bp = beam_profiler

        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()

        bp.analyze(img)

        assert bp.fw_1e_x < bp.fwhm_x < bp.fw_1e2_x
        assert bp.fw_1e_y < bp.fwhm_y < bp.fw_1e2_y


class TestBeamProfilerStaticImages:
    """Test static image loading and analysis."""

    def test_load_from_file(self, test_image_file):
        """Test loading image from file."""
        bp = BeamProfiler(file=test_image_file)

        assert bp.last_img is not None
        assert bp.last_img.shape == (500, 500)
        assert bp.pixel_size == 1.0

    def test_analyze_static_image(self, test_image_file):
        """Test analyzing static image file."""
        bp = BeamProfiler(file=test_image_file)
        bp.fit_method = "1d"
        bp.definition = "gaussian"

        popt_x, popt_y = bp.analyze(bp.last_img)

        assert popt_x is not None
        assert popt_y is not None
        assert len(popt_x) == 4
        assert len(popt_y) == 4
        assert bp.width_x > 0
        assert bp.width_y > 0


class TestBeamProfilerInitialization:
    """Test BeamProfiler initialization options."""

    def test_default_initialization(self):
        """Test default initialization."""
        bp = BeamProfiler()
        assert bp.fit_method == "1d"
        assert bp.definition == "gaussian"
        assert bp._camera is not None
        bp._camera.close()

    def test_initialization_with_camera(self):
        """Test initialization with camera type."""
        bp = BeamProfiler(camera="simulated")
        assert bp._camera is not None
        bp._camera.close()

    def test_initialization_with_file(self, test_image_file):
        """Test initialization with file."""
        bp = BeamProfiler(file=test_image_file)
        assert bp.last_img is not None
        assert bp._mode == "static"

    def test_initialization_with_fit_method(self):
        """Test initialization with fit method."""
        for method in ["1d", "2d", "linecut"]:
            bp = BeamProfiler(camera="simulated", fit=method)
            assert bp.fit_method == method
            bp._camera.close()

    def test_initialization_with_definition(self):
        """Test initialization with definition."""
        for defn in ["gaussian", "fwhm", "d4s"]:
            bp = BeamProfiler(camera="simulated", definition=defn)
            assert bp.definition == defn
            bp._camera.close()


class TestBeamProfilerExposure:
    """Test exposure time handling."""

    def test_none_exposure_single_shot(self):
        """Test None exposure doesn't crash for single shot."""
        bp = BeamProfiler(camera="simulated")

        # This should not crash
        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()
        bp.analyze(img)

        assert bp.width > 0
        bp._camera.close()

    def test_none_exposure_continuous(self):
        """Test None exposure uses default."""
        bp = BeamProfiler(camera="simulated")
        bp._camera.set_exposure(None)
        assert bp._camera.exposure_time == 0.01
        bp._camera.close()


class TestBeamProfilerDimensions:
    """Test sensor dimensions and pixel size."""

    def test_sensor_dimensions(self):
        """Test sensor dimensions are set correctly."""
        bp = BeamProfiler(camera="simulated")

        # Verify camera initialized properly
        assert bp._camera is not None
        assert bp._camera.width == 1024
        assert bp._camera.height == 1024

        # Check dimensions are propagated to BeamProfiler
        assert bp.width_pixels == 1024
        assert bp.height_pixels == 1024
        assert bp.pixel_size == 5.0

        bp._camera.close()

    def test_file_dimensions(self, test_image_file):
        """Test dimensions from loaded file."""
        bp = BeamProfiler(file=test_image_file)

        assert bp.width_pixels == 500
        assert bp.height_pixels == 500
        assert bp.pixel_size == 1.0


class TestBeamProfilerVisualization:
    """Test visualization methods."""

    def test_create_fast_figure_1d(self):
        """Test fast figure creation with 1D fitting."""
        bp = BeamProfiler(camera="simulated", fit="1d")

        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        fig = bp._create_fast_figure(img, popt_x, popt_y)

        assert fig is not None
        assert len(fig.data) > 0
        bp._camera.close()

    def test_create_fast_figure_2d(self):
        """Test fast figure creation with 2D fitting."""
        bp = BeamProfiler(camera="simulated", fit="2d")

        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        fig = bp._create_fast_figure(img, popt_x, popt_y)

        assert fig is not None
        assert len(fig.data) > 0
        bp._camera.close()

    def test_create_fast_figure_none_params(self):
        """Test fast figure creation with None parameters."""
        bp = BeamProfiler(camera="simulated")

        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()

        fig = bp._create_fast_figure(img, None, None)

        assert fig is not None
        assert len(fig.data) > 0
        bp._camera.close()

    def test_create_fast_figure_none_image(self):
        """Test fast figure creation with None image."""
        bp = BeamProfiler(camera="simulated")
        fig = bp._create_fast_figure(None, None, None)

        assert fig is not None
        assert len(fig.data) == 0
        bp._camera.close()


class TestBeamProfilerMethods:
    """Test various BeamProfiler methods."""

    def test_analyze_with_different_definitions(self):
        """Test analyze with different width definitions."""
        bp = BeamProfiler(camera="simulated")

        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()

        for definition in ["gaussian", "fwhm", "d4s"]:
            bp.definition = definition
            popt_x, popt_y = bp.analyze(img)
            assert popt_x is not None
            assert popt_y is not None
            assert bp.width_x > 0
            assert bp.width_y > 0

        bp._camera.close()

    def test_analyze_with_different_fit_methods(self):
        """Test analyze with different fitting methods."""
        bp = BeamProfiler(camera="simulated")

        bp._camera.start_acquisition()
        img = bp._camera.get_image()
        bp._camera.stop_acquisition()

        for method in ["1d", "2d", "linecut"]:
            bp.fit_method = method
            popt_x, popt_y = bp.analyze(img)
            assert popt_x is not None
            assert popt_y is not None

        bp._camera.close()

    def test_getattr_proxy(self):
        """Test attribute proxying to camera."""
        bp = BeamProfiler(camera="simulated")

        # Test that camera attributes are accessible
        assert hasattr(bp, "exposure_time")
        assert hasattr(bp, "gain")

        # Test that non-existent attributes raise AttributeError
        try:
            _ = bp.nonexistent_attribute
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass

        bp._camera.close()
