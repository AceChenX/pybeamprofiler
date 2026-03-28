"""Tests for BeamProfiler properties, attributes, and integration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pybeamprofiler import BeamProfiler


class TestBeamProfilerProperties:
    """Test BeamProfiler computed properties."""

    def test_basic_properties(self, beam_profiler):
        """Test basic width and diameter properties."""
        bp = beam_profiler
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

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
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        # Default definition is gaussian, where fw_1e2 == width
        bp.analyze(img)

        assert bp.fwhm_x > 0
        assert bp.fwhm_y > 0
        assert bp.fw_1e_x > 0
        assert bp.fw_1e_y > 0
        assert bp.fw_1e2_x == pytest.approx(bp.width_x, rel=1e-10)
        assert bp.fw_1e2_y == pytest.approx(bp.width_y, rel=1e-10)
        assert bp.height_x > 0
        assert bp.height_y > 0

    def test_property_relationships(self, beam_profiler):
        """Test relationships between width properties hold for all definitions."""
        bp = beam_profiler
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        for definition in ["gaussian", "fwhm", "d4s"]:
            bp.definition = definition
            bp.analyze(img)

            assert bp.fw_1e_x < bp.fwhm_x < bp.fw_1e2_x, f"Failed for {definition}"
            assert bp.fw_1e_y < bp.fwhm_y < bp.fw_1e2_y, f"Failed for {definition}"


class TestBeamProfilerStaticImages:
    """Test static image loading and analysis."""

    def test_load_from_file(self, test_image_file):
        """Test loading image from file."""
        bp = BeamProfiler(file=test_image_file, pixel_size=1.0)

        assert bp.last_img is not None
        assert bp.last_img.shape == (500, 500)
        assert bp.pixel_size == 1.0

    def test_analyze_static_image(self, test_image_file):
        """Test analyzing static image file."""
        bp = BeamProfiler(file=test_image_file, pixel_size=1.0)
        bp.fit_method = "1d"
        bp.definition = "gaussian"

        popt_x, popt_y = bp.analyze(bp.last_img)  # ty:ignore[invalid-argument-type]

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
        assert bp.camera is not None
        bp.camera.close()

    def test_initialization_with_camera(self):
        """Test initialization with camera type."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.close()

    def test_initialization_with_file(self, test_image_file):
        """Test initialization with file."""
        bp = BeamProfiler(file=test_image_file, pixel_size=1.0)
        assert bp.last_img is not None
        assert bp._mode == "static"

    def test_initialization_with_fit_method(self):
        """Test initialization with fit method."""
        for method in ["1d", "2d", "linecut"]:
            bp = BeamProfiler(camera="simulated", fit=method)
            assert bp.camera is not None
            assert bp.fit_method == method
            bp.camera.close()

    def test_initialization_with_definition(self):
        """Test initialization with definition."""
        for defn in ["gaussian", "fwhm", "d4s"]:
            bp = BeamProfiler(camera="simulated", definition=defn)
            assert bp.camera is not None
            assert bp.definition == defn
            bp.camera.close()


class TestBeamProfilerExposure:
    """Test exposure time handling."""

    def test_none_exposure_single_shot(self):
        """Test None exposure doesn't crash for single shot."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()
        bp.analyze(img)

        assert bp.width > 0
        bp.camera.close()

    def test_none_exposure_continuous(self):
        """Test None exposure uses default."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.set_exposure(None)  # ty:ignore[invalid-argument-type]
        assert bp.camera.exposure_time == 0.01
        bp.camera.close()


class TestBeamProfilerDimensions:
    """Test sensor dimensions and pixel size."""

    def test_sensor_dimensions(self):
        """Test sensor dimensions are set correctly."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        assert bp.camera is not None
        assert bp.camera.width == 1024
        assert bp.camera.height == 1024

        assert bp.width_pixels == 1024
        assert bp.height_pixels == 1024
        assert bp.pixel_size == 5.0

        bp.camera.close()

    def test_file_dimensions(self, test_image_file):
        """Test dimensions from loaded file."""
        bp = BeamProfiler(file=test_image_file, pixel_size=1.0)

        assert bp.width_pixels == 500
        assert bp.height_pixels == 500
        assert bp.pixel_size == 1.0


class TestBeamProfilerVisualization:
    """Test visualization methods."""

    def test_create_fast_figure_1d(self):
        """Test fast figure creation with 1D fitting."""
        bp = BeamProfiler(camera="simulated", fit="1d")
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        fig = bp._create_fast_figure(img, popt_x, popt_y)

        assert fig is not None
        assert len(fig.data) > 0
        bp.camera.close()

    def test_create_fast_figure_2d(self):
        """Test fast figure creation with 2D fitting."""
        bp = BeamProfiler(camera="simulated", fit="2d")
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        fig = bp._create_fast_figure(img, popt_x, popt_y)

        assert fig is not None
        assert len(fig.data) > 0
        bp.camera.close()

    def test_create_fast_figure_none_params(self):
        """Test fast figure creation with None parameters."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        fig = bp._create_fast_figure(img, None, None)

        assert fig is not None
        assert len(fig.data) > 0
        bp.camera.close()

    def test_create_fast_figure_none_image(self):
        """Test fast figure creation with None image."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        fig = bp._create_fast_figure(None, None, None)  # ty:ignore[invalid-argument-type]

        assert fig is not None
        assert len(fig.data) == 0
        bp.camera.close()


class TestBeamProfilerMethods:
    """Test various BeamProfiler methods."""

    def test_analyze_with_different_definitions(self):
        """Test analyze with different width definitions."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        for definition in ["gaussian", "fwhm", "d4s"]:
            bp.definition = definition
            popt_x, popt_y = bp.analyze(img)
            assert popt_x is not None
            assert popt_y is not None
            assert bp.width_x > 0
            assert bp.width_y > 0

        bp.camera.close()

    def test_analyze_with_different_fit_methods(self):
        """Test analyze with different fitting methods."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        for method in ["1d", "2d", "linecut"]:
            bp.fit_method = method
            popt_x, popt_y = bp.analyze(img)
            assert popt_x is not None
            assert popt_y is not None

        bp.camera.close()

    def test_getattr_proxy(self):
        """Test attribute proxying to camera."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        assert hasattr(bp, "exposure_time")
        assert hasattr(bp, "gain")

        with pytest.raises(AttributeError):
            _ = bp.nonexistent_attribute

        assert bp.camera is not None
        bp.camera.close()


class TestBeamProfilerContextManager:
    """Test context manager support."""

    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with BeamProfiler(camera="simulated") as bp:
            assert bp.camera is not None
            bp.camera.start_acquisition()
            img = bp.camera.get_image()
            bp.camera.stop_acquisition()
            bp.analyze(img)
            assert bp.width > 0

    def test_context_manager_with_exception(self):
        """Test context manager properly closes camera on exception."""
        try:
            with BeamProfiler(camera="simulated") as bp:
                assert bp.camera is not None
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

    def test_context_manager_with_file(self, test_image_file):
        """Test context manager with static file (no camera to close)."""
        with BeamProfiler(file=test_image_file, pixel_size=1.0) as bp:
            assert bp.last_img is not None
            bp.analyze(bp.last_img)


class TestBeamProfilerInputValidation:
    """Test input validation for analyze method."""

    def test_analyze_none_image(self):
        """Test analyze raises error on None input."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        with pytest.raises(ValueError, match="Image cannot be None"):
            bp.analyze(None)  # ty: ignore[invalid-argument-type]
        assert bp.camera is not None
        bp.camera.close()

    def test_analyze_wrong_type(self):
        """Test analyze raises error on wrong type."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        with pytest.raises(TypeError, match="Image must be numpy array"):
            bp.analyze([1, 2, 3])  # ty: ignore[invalid-argument-type]
        assert bp.camera is not None
        bp.camera.close()

    def test_analyze_wrong_dimensions(self):
        """Test analyze raises error on wrong dimensions."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None

        with pytest.raises(ValueError, match="Image must be 2D"):
            bp.analyze(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="Image must be 2D"):
            bp.analyze(np.zeros((10, 10, 3)))

        assert bp.camera is not None
        bp.camera.close()

    def test_analyze_empty_image(self):
        """Test analyze raises error on empty image."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        with pytest.raises(ValueError, match="Image cannot be empty"):
            bp.analyze(np.array([[]]))
        assert bp.camera is not None
        bp.camera.close()


class TestBeamProfilerInit:
    """Test initialization edge cases."""

    def test_file_without_pixel_size_raises(self, tmp_path):
        """Test that file mode requires pixel_size."""
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        img.save(img_path)

        with pytest.raises(ValueError, match="Pixel size must be provided"):
            BeamProfiler(file=str(img_path))

    def test_file_load_error_raises(self, tmp_path):
        """Test error propagation for invalid image files."""
        bad_path = tmp_path / "nonexistent.png"
        with pytest.raises(Exception):
            BeamProfiler(file=str(bad_path), pixel_size=5.0)

    def test_physical_camera_failure_raises(self):
        """Test that physical camera failure raises RuntimeError, not fallback."""
        with patch("pybeamprofiler.beamprofiler.BaslerCamera") as MockBasler:
            MockBasler.return_value.open.side_effect = RuntimeError("No camera")
            with pytest.raises(RuntimeError, match="Failed to open basler camera"):
                BeamProfiler(camera="basler")

    def test_unknown_camera_fallback_on_open_error(self):
        """Test unknown camera type falls back to SimulatedCamera on error."""
        from pybeamprofiler import SimulatedCamera

        with patch("pybeamprofiler.beamprofiler.SimulatedCamera") as MockSim:
            call_count = 0
            instances = []

            def make_instance():
                nonlocal call_count
                call_count += 1
                inst = MagicMock(spec=SimulatedCamera)
                inst.width = 1024
                inst.height = 1024
                inst.pixel_size = 5.0
                if call_count == 1:
                    inst.open.side_effect = RuntimeError("Mock fail")
                else:
                    inst.open.return_value = None
                instances.append(inst)
                return inst

            MockSim.side_effect = make_instance

            bp = BeamProfiler(camera="unknown_brand")
            assert bp.camera is not None
            assert call_count == 2  # First failed, second succeeded

    def test_exposure_time_passed_to_camera(self):
        """Test exposure_time kwarg is forwarded to camera."""
        bp = BeamProfiler(camera="simulated", exposure_time=0.05)
        assert bp.camera is not None
        assert bp.camera.exposure_time == 0.05
        bp.camera.close()


class TestContextManager:
    """Test context manager behavior."""

    def test_exit_with_close_error(self):
        """Test __exit__ handles camera close errors gracefully."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.close = MagicMock(side_effect=RuntimeError("close failed"))  # ty: ignore[invalid-assignment]
        result = bp.__exit__(None, None, None)
        assert result is False

    def test_exit_without_camera(self):
        """Test __exit__ when camera is None."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera = None
        result = bp.__exit__(None, None, None)
        assert result is False


class TestGetattr:
    """Test attribute proxying."""

    def test_getattr_proxies_camera_attribute(self):
        """Test __getattr__ returns camera attributes."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        assert bp.is_acquiring is False
        bp.camera.close()

    def test_getattr_raises_for_missing_attribute(self):
        """Test __getattr__ raises AttributeError for missing attributes."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        with pytest.raises(AttributeError, match="no attribute"):
            _ = bp.totally_nonexistent_attribute
        bp.camera.close()

    def test_getattr_before_camera_init(self):
        """Test __getattr__ when camera attribute doesn't exist yet."""
        bp = object.__new__(BeamProfiler)
        with pytest.raises(AttributeError, match="no attribute"):
            _ = bp.some_attr


class TestFitFailures:
    """Test Gaussian fitting edge cases."""

    def test_1d_fit_empty_profile(self):
        """Test _fit_1d_gaussian with empty profile."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        result = bp._fit_1d_gaussian(np.array([]))
        assert result == [0, 0, 1, 0]
        bp.camera.close()

    def test_1d_fit_runtime_error_fallback(self):
        """Test _fit_1d_gaussian falls back to initial guess on failure."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        noisy = np.random.uniform(0, 255, 10).astype(np.float64)
        with patch("pybeamprofiler.beamprofiler.curve_fit", side_effect=RuntimeError("fit failed")):
            result = bp._fit_1d_gaussian(noisy)
        assert len(result) == 4
        bp.camera.close()

    def test_2d_fit_runtime_error_fallback(self):
        """Test _fit_2d_gaussian falls back to initial guess on failure."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = np.random.uniform(0, 255, (32, 32)).astype(np.float64)
        with patch("pybeamprofiler.beamprofiler.curve_fit", side_effect=RuntimeError("fit failed")):
            result = bp._fit_2d_gaussian(img)
        assert len(result) == 7
        bp.camera.close()

    def test_2d_fit_uses_cached_guess(self):
        """Test _fit_2d_gaussian uses cached initial guess."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        img = np.random.uniform(0, 255, (32, 32)).astype(np.float64)
        bp._last_popt_2d = [100, 16, 16, 5, 5, 0, 10]
        with patch("pybeamprofiler.beamprofiler.curve_fit", side_effect=RuntimeError("fit failed")):
            result = bp._fit_2d_gaussian(img)
        assert result == [100, 16, 16, 5, 5, 0, 10]
        bp.camera.close()


class TestStopMethod:
    """Test the stop() method."""

    def test_stop_cancels_stream_task(self):
        """Test stop() cancels a running stream task."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        mock_task = MagicMock()
        bp._stream_task = mock_task
        bp.camera.is_acquiring = True

        bp.stop()

        mock_task.cancel.assert_called_once()
        assert bp._stream_task is None
        bp.camera.close()

    def test_stop_stops_acquisition(self):
        """Test stop() stops camera acquisition when active."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        assert bp.camera.is_acquiring is True

        bp.stop()

        assert bp.camera.is_acquiring is False
        bp.camera.close()

    def test_stop_noop_when_idle(self):
        """Test stop() is safe when nothing is running."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.stop()  # Should not raise
        bp.camera.close()


class TestPlotSingle:
    """Test _plot_single method."""

    def test_plot_single_static_mode(self, tmp_path):
        """Test _plot_single with static file."""
        from PIL import Image

        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        img_path = tmp_path / "beam.png"
        Image.fromarray(img).save(img_path)

        bp = BeamProfiler(file=str(img_path), pixel_size=5.0)
        with patch.object(bp, "_create_figure") as mock_fig:
            mock_fig.return_value = MagicMock()
            bp._plot_single()
            mock_fig.assert_called_once()

    def test_plot_single_camera_mode(self):
        """Test _plot_single with camera acquires and releases."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        with patch.object(bp, "_create_figure") as mock_fig:
            mock_fig.return_value = MagicMock()
            bp._plot_single()
            mock_fig.assert_called_once()
        bp.camera.close()

    def test_plot_single_no_camera_raises(self):
        """Test _plot_single raises when camera is None in camera mode."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera = None
        bp._mode = "camera"
        with pytest.raises(RuntimeError, match="Camera is not initialized"):
            bp._plot_single()

    def test_plot_single_none_image_raises(self):
        """Test _plot_single raises when image is None."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        bp.camera.get_image = MagicMock(return_value=None)  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError, match="No image available"):
            bp._plot_single()
        bp.camera.close()


class TestCreateFigure:
    """Test figure creation methods."""

    def test_create_figure_none_image(self):
        """Test _create_figure returns empty figure for None image."""
        import plotly.graph_objects as go

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        fig = bp._create_figure(None, None, None)  # ty: ignore[invalid-argument-type]
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        bp.camera.close()

    def test_create_fast_figure_none_image(self):
        """Test _create_fast_figure returns empty figure for None image."""
        import plotly.graph_objects as go

        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        fig = bp._create_fast_figure(None, None, None)  # ty: ignore[invalid-argument-type]
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        bp.camera.close()

    def test_create_figure_with_linecut(self):
        """Test _create_figure with linecut method includes crosshair lines."""
        bp = BeamProfiler(camera="simulated", fit="linecut")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        fig = bp._create_figure(img, popt_x, popt_y)
        trace_names = [t.name for t in fig.data if t.name]
        assert any("Linecut" in n for n in trace_names)
        bp.camera.close()

    def test_create_fast_figure_with_linecut(self):
        """Test _create_fast_figure with linecut method includes crosshair lines."""
        bp = BeamProfiler(camera="simulated", fit="linecut")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        fig = bp._create_fast_figure(img, popt_x, popt_y)
        trace_names = [t.name for t in fig.data if t.name]
        assert any("Linecut" in n for n in trace_names)
        bp.camera.close()

    def test_create_figure_with_2d_angle(self):
        """Test _create_figure with 2D fit includes angle in title."""
        bp = BeamProfiler(camera="simulated", fit="2d")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        fig = bp._create_figure(img, popt_x, popt_y)
        title_text = fig.layout.title.text
        assert "Angle" in title_text
        bp.camera.close()

    def test_create_figure_2d_ellipse_rotation(self):
        """Test _create_figure with 2d fit uses rotated ellipse."""
        bp = BeamProfiler(camera="simulated", fit="2d")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        fig = bp._create_figure(img, popt_x, popt_y)
        width_traces = [t for t in fig.data if t.name and "Width" in t.name]
        assert len(width_traces) > 0
        bp.camera.close()

    def test_create_fast_figure_2d_ellipse_rotation(self):
        """Test _create_fast_figure with 2d fit uses rotated ellipse."""
        bp = BeamProfiler(camera="simulated", fit="2d")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        bp.fit_method = "2d"
        fig = bp._create_fast_figure(img, popt_x, popt_y)
        width_traces = [t for t in fig.data if t.name and "Width" in t.name]
        assert len(width_traces) > 0
        bp.camera.close()


class TestAnalyzeMethod:
    """Test analyze method with various configurations."""

    def test_analyze_none_raises(self):
        """Test analyze raises on None image."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        with pytest.raises(ValueError, match="cannot be None"):
            bp.analyze(None)  # ty: ignore[invalid-argument-type]
        bp.camera.close()

    def test_analyze_wrong_type_raises(self):
        """Test analyze raises on non-ndarray."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        with pytest.raises(TypeError, match="must be numpy array"):
            bp.analyze([[1, 2], [3, 4]])  # ty: ignore[invalid-argument-type]
        bp.camera.close()

    def test_analyze_1d_array_raises(self):
        """Test analyze raises on 1D array."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        with pytest.raises(ValueError, match="must be 2D"):
            bp.analyze(np.array([1, 2, 3]))
        bp.camera.close()

    def test_analyze_fwhm_definition(self):
        """Test analyze with FWHM definition."""
        bp = BeamProfiler(camera="simulated", definition="fwhm")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        assert popt_x is not None
        assert popt_y is not None
        assert bp.width_x > 0
        assert bp.angle_deg == 0.0
        bp.camera.close()

    def test_analyze_d4s_definition(self):
        """Test analyze with D4σ definition."""
        bp = BeamProfiler(camera="simulated", definition="d4s")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        assert popt_x is not None
        assert bp.width_x > 0
        bp.camera.close()

    def test_analyze_linecut_stores_positions(self):
        """Test linecut method stores _linecut_x and _linecut_y."""
        bp = BeamProfiler(camera="simulated", fit="linecut")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        bp.analyze(img)
        assert hasattr(bp, "_linecut_x")
        assert hasattr(bp, "_linecut_y")
        bp.camera.close()

    def test_analyze_2d_returns_projections(self):
        """Test 2D fit returns 1D projection fits."""
        bp = BeamProfiler(camera="simulated", fit="2d")
        assert bp.camera is not None
        bp.camera.start_acquisition()
        img = bp.camera.get_image()
        bp.camera.stop_acquisition()

        popt_x, popt_y = bp.analyze(img)
        assert popt_x is not None
        assert popt_y is not None
        assert len(popt_x) == 4
        assert len(popt_y) == 4
        assert bp.angle_deg >= 0
        bp.camera.close()


class TestMeasureMethods:
    """Test FWHM and D4σ measurement helpers."""

    def test_measure_d4s_zero_intensity(self):
        """Test _measure_d4s with flat (zero intensity) profile."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        center, width = bp._measure_d4s(np.zeros(100))
        assert center == 50.0
        assert width == 1.0
        bp.camera.close()

    def test_measure_fwhm_edge_cases(self):
        """Test _measure_fwhm with very narrow peak."""
        bp = BeamProfiler(camera="simulated")
        assert bp.camera is not None
        profile = np.zeros(100, dtype=np.float64)
        profile[50] = 100
        center, fwhm, peak = bp._measure_fwhm(profile)
        assert peak == 100
        assert abs(center - 50) < 2
        bp.camera.close()


class TestCLI:
    """Test CLI argument parser and main block."""

    def test_cli_argparse_defaults(self):
        """Test CLI argument parser with defaults."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--camera", type=str, default="simulated")
        parser.add_argument("--file", type=str, default=None)
        parser.add_argument("--fit", type=str, default="1d")
        parser.add_argument("--definition", type=str, default="gaussian")
        parser.add_argument("--exposure-time", type=float, default=None)
        parser.add_argument("--num-img", type=int, default=None)
        parser.add_argument("--heatmap-only", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")

        args = parser.parse_args([])
        assert args.camera == "simulated"
        assert args.fit == "1d"
        assert args.definition == "gaussian"
        assert args.num_img is None
        assert not args.heatmap_only

    def test_cli_argparse_custom(self):
        """Test CLI argument parser with custom args."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--camera", type=str, default="simulated")
        parser.add_argument("--fit", type=str, default="1d")
        parser.add_argument("--definition", type=str, default="gaussian")
        parser.add_argument("--num-img", type=int, default=None)
        parser.add_argument("--heatmap-only", action="store_true")
        parser.add_argument("--exposure-time", type=float, default=None)

        args = parser.parse_args(
            [
                "--camera",
                "simulated",
                "--fit",
                "2d",
                "--definition",
                "fwhm",
                "--num-img",
                "1",
                "--heatmap-only",
                "--exposure-time",
                "0.05",
            ]
        )
        assert args.camera == "simulated"
        assert args.fit == "2d"
        assert args.definition == "fwhm"
        assert args.num_img == 1
        assert args.heatmap_only is True
        assert args.exposure_time == 0.05

    def test_cli_plot_single_shot(self):
        """Test CLI-style single shot execution."""
        bp = BeamProfiler(camera="simulated", fit="1d", definition="gaussian")
        assert bp.camera is not None
        with patch.object(bp, "_plot_single") as mock_plot:
            bp.plot(num_img=1)
            mock_plot.assert_called_once()
        bp.camera.close()

    def test_cli_exposure_time_arg(self):
        """Test that exposure_time argument is applied."""
        bp = BeamProfiler(camera="simulated", exposure_time=0.05)
        assert bp.camera is not None
        assert bp.camera.exposure_time == 0.05
        bp.camera.close()

    def test_cli_all_fit_methods(self):
        """Test BeamProfiler with all CLI fit method options."""
        for fit in ["1d", "2d", "linecut"]:
            bp = BeamProfiler(camera="simulated", fit=fit)
            assert bp.camera is not None
            assert bp.fit_method == fit
            bp.camera.close()

    def test_cli_all_definitions(self):
        """Test BeamProfiler with all CLI definition options."""
        for defn in ["gaussian", "fwhm", "d4s"]:
            bp = BeamProfiler(camera="simulated", definition=defn)
            assert bp.camera is not None
            assert bp.definition == defn
            bp.camera.close()
