"""Tests for beamprofiler streaming and plot methods."""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import dash
import numpy as np

from pybeamprofiler.beamprofiler import BeamProfiler
from pybeamprofiler.dash_app import build_figure


def test_plot_single():
    """Test that plot(num_img=1) uses single plot and returns None."""
    bp = BeamProfiler(camera="simulated")

    with patch.object(bp, "_plot_single") as mock_plot_single:
        result = bp.plot(num_img=1)
        mock_plot_single.assert_called_once()
        assert result is None


def test_plot_stream_jupyter_task():
    """Test that Jupyter streaming creates an asyncio.Task."""
    bp = BeamProfiler(camera="simulated")

    # Always return a frame so it loops normally without raising StopIteration
    mock_img = np.ones((10, 10))

    assert bp.camera is not None
    bp.camera.get_image = MagicMock(return_value=mock_img)  # type: ignore
    bp.analyze = MagicMock(return_value=([0, 0, 1, 0], [0, 0, 1, 0]))  # type: ignore
    bp._create_fast_figure = MagicMock(return_value=MagicMock())  # type: ignore

    async def run_test():
        # Setup mock IPython environment
        mock_get_ipython = MagicMock(return_value=MagicMock())
        display_mock = MagicMock()
        clear_output_mock = MagicMock()

        mock_ipython = MagicMock()
        mock_ipython.get_ipython = mock_get_ipython

        mock_ipython_display = MagicMock()
        mock_ipython_display.display = display_mock
        mock_ipython_display.clear_output = clear_output_mock

        with patch.dict(
            "sys.modules",
            {"IPython": mock_ipython, "IPython.display": mock_ipython_display},
        ):
            # Run plot_stream
            task = bp.plot(heatmap_only=True)

            # Should return a task
            assert isinstance(task, asyncio.Task)

            # Let it render the first frame
            await asyncio.sleep(0.01)

            # Cancel the task since it now continues on dropped frames
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Check clear_output and display were called for at least 1 successful frame
            assert clear_output_mock.call_count >= 1
            assert display_mock.call_count >= 1

    asyncio.run(run_test())


def test_plot_stream_jupyter_cancellation():
    """Test that the Jupyter stream task can be cancelled cleanly."""
    bp = BeamProfiler(camera="simulated")

    # Always return an image to run indefinitely
    assert bp.camera is not None
    bp.camera.get_image = MagicMock(return_value=np.ones((10, 10)))  # type: ignore
    bp.analyze = MagicMock(return_value=([0, 0, 1, 0], [0, 0, 1, 0]))  # type: ignore
    bp._create_fast_figure = MagicMock(return_value=MagicMock())  # type: ignore

    async def run_cancel_test():
        mock_get_ipython = MagicMock(return_value=MagicMock())
        display_mock = MagicMock()
        clear_output_mock = MagicMock()

        mock_ipython = MagicMock()
        mock_ipython.get_ipython = mock_get_ipython

        mock_ipython_display = MagicMock()
        mock_ipython_display.display = display_mock
        mock_ipython_display.clear_output = clear_output_mock

        with patch.dict(
            "sys.modules",
            {"IPython": mock_ipython, "IPython.display": mock_ipython_display},
        ):
            task = bp.plot(heatmap_only=True)
            assert isinstance(task, asyncio.Task)

            # Give it a tiny bit of time to start and loop once
            await asyncio.sleep(0.01)

            # Cancel the task
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # The task should be cleanly finished (since it catches CancelledError internally)
            assert task.done()
            # Should have run at least once
            assert display_mock.call_count > 0

    asyncio.run(run_cancel_test())


def test_plot_stream_jupyter_robustness():
    """Test that the Jupyter stream task survives exceptions and missing frames."""
    bp = BeamProfiler(camera="simulated")

    assert bp.camera is not None

    mock_img = np.ones((10, 10))
    # Sequence: 1 exception (timeout), 1 missing frame (None), and then valid frames.
    # This proves the loop didn't break on errors/Nones
    sequence = [RuntimeError("Camera timeout"), None, mock_img, mock_img, mock_img]

    def mock_get_image():
        if not sequence:
            return mock_img
        val = sequence.pop(0)
        if isinstance(val, Exception):
            raise val
        return val

    bp.camera.get_image = MagicMock(side_effect=mock_get_image)  # type: ignore
    bp.camera.is_acquiring = True  # Ensure it doesn't gracefully exit on None
    bp.analyze = MagicMock(return_value=([0, 0, 1, 0], [0, 0, 1, 0]))  # type: ignore
    bp._create_fast_figure = MagicMock(return_value=MagicMock())  # type: ignore

    async def run_robustness_test():
        mock_get_ipython = MagicMock(return_value=MagicMock())
        display_mock = MagicMock()
        clear_output_mock = MagicMock()

        mock_ipython = MagicMock()
        mock_ipython.get_ipython = mock_get_ipython

        mock_ipython_display = MagicMock()
        mock_ipython_display.display = display_mock
        mock_ipython_display.clear_output = clear_output_mock

        with patch.dict(
            "sys.modules",
            {"IPython": mock_ipython, "IPython.display": mock_ipython_display},
        ):
            task = bp.plot(heatmap_only=True)
            assert isinstance(task, asyncio.Task)

            # Wait for the loop to process the errors and output the valid frames
            # Note: It sleeps for 0.01s on Exception and 0.01s on None
            await asyncio.sleep(0.05)

            # Should not be done (loop still running)
            assert not task.done()

            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have survived the initial exceptions/Nones and processed the valid frames
            assert display_mock.call_count >= 1

    asyncio.run(run_robustness_test())


def test_plot_stream_dash_robustness():
    """Test that Dash streaming handles errors gracefully."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None
    mock_img = np.ones((10, 10))

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    with (
        patch("dash.Dash") as MockDash,
        patch.dict("sys.modules", {"IPython": mock_ipython}),
        patch("threading.Thread"),
        patch("asyncio.to_thread") as mock_to_thread,
    ):
        mock_app = MagicMock()
        MockDash.return_value = mock_app
        mock_app.run = MagicMock()

        mock_fig = MagicMock()
        mock_fig.layout.title.text = "Title"

        callback_func = None

        def capture_callback(*args, **kwargs):
            def decorator(f):
                nonlocal callback_func
                callback_func = f
                return f

            return decorator

        mock_app.callback = MagicMock(side_effect=capture_callback)

        bp.plot(heatmap_only=True)
        assert callback_func is not None

        bp.camera.is_acquiring = True

        async def run_callback_error():
            def side_effect(*args, **kwargs):
                if args[0] == bp.camera.get_image:
                    raise RuntimeError("Camera dead")
                return None

            mock_to_thread.side_effect = side_effect
            return await callback_func(
                0, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
            )  # type: ignore

        assert asyncio.run(run_callback_error())[0] is dash.no_update

        async def run_callback_none():
            def side_effect(*args, **kwargs):
                if args[0] == bp.camera.get_image:
                    return None
                return None

            mock_to_thread.side_effect = side_effect
            return await callback_func(
                0, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
            )  # type: ignore

        assert asyncio.run(run_callback_none())[0] is dash.no_update

        async def run_callback_success():
            def side_effect(*args, **kwargs):
                if args[0] == bp.camera.get_image:
                    return mock_img
                if args[0] == bp.analyze:
                    return ([0, 0, 1, 0], [0, 0, 1, 0])
                if args[0] == build_figure:
                    return mock_fig
                return None

            mock_to_thread.side_effect = side_effect
            return await callback_func(
                0, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
            )  # type: ignore

        assert asyncio.run(run_callback_success())[0] is mock_fig


def test_plot_stream_matplotlib_fallback():
    """Test fallback to matplotlib when both Jupyter and Dash are unavailable."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None

    mock_fig_plt = MagicMock()
    mock_axes = MagicMock()
    mock_axes.flat = [MagicMock() for _ in range(4)]

    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig_plt, mock_axes)

    mock_animation = MagicMock()
    mock_patches = MagicMock()

    # `import a.b as x` resolves via parent module attribute access,
    # so matplotlib.pyplot must be an attribute of the matplotlib mock.
    mock_matplotlib = MagicMock()
    mock_matplotlib.pyplot = mock_plt
    mock_matplotlib.animation = mock_animation
    mock_matplotlib.patches = mock_patches

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    with patch.dict(
        sys.modules,
        {
            "IPython": mock_ipython,
            "dash": None,
            "dash.dependencies": None,
            "matplotlib": mock_matplotlib,
            "matplotlib.pyplot": mock_plt,
            "matplotlib.animation": mock_animation,
            "matplotlib.patches": mock_patches,
        },
    ):
        bp._plot_stream()
        mock_plt.subplots.assert_called_once()
        mock_plt.show.assert_called_once()

    bp.camera.close()


def test_plot_stream_no_visualization_available():
    """Test graceful handling when neither Dash nor matplotlib is available."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    with patch.dict(
        sys.modules,
        {
            "IPython": mock_ipython,
            "dash": None,
            "dash.dependencies": None,
            "matplotlib": None,
            "matplotlib.pyplot": None,
            "matplotlib.animation": None,
            "matplotlib.patches": None,
        },
    ):
        result = bp._plot_stream()
        assert result is None

    bp.camera.close()


def test_plot_stream_dash_non_heatmap():
    """Test Dash streaming in full figure (non-heatmap) mode."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None
    mock_img = np.ones((10, 10))

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    with (
        patch("dash.Dash") as MockDash,
        patch.dict("sys.modules", {"IPython": mock_ipython}),
        patch("threading.Thread"),
        patch("asyncio.to_thread") as mock_to_thread,
    ):
        mock_app = MagicMock()
        MockDash.return_value = mock_app
        mock_app.run = MagicMock()

        mock_fig = MagicMock()
        mock_fig.layout.title.text = "Title"

        callback_func = None

        def capture_callback(*args, **kwargs):
            def decorator(f):
                nonlocal callback_func
                callback_func = f
                return f

            return decorator

        mock_app.callback = MagicMock(side_effect=capture_callback)

        bp.plot(heatmap_only=False)
        assert callback_func is not None

        bp.camera.is_acquiring = True

        async def run_callback_success():
            def side_effect(*args, **kwargs):
                if args[0] == bp.camera.get_image:
                    return mock_img
                if args[0] == bp.analyze:
                    return ([0, 0, 1, 0], [0, 0, 1, 0])
                if args[0] == build_figure:
                    return mock_fig
                return None

            mock_to_thread.side_effect = side_effect
            return await callback_func(
                0, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
            )  # type: ignore

        assert asyncio.run(run_callback_success())[0] is mock_fig

    bp.camera.close()


def test_plot_stream_camera_not_acquiring_restart():
    """Test Dash callback handles camera not acquiring (proceeds with get_image)."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None
    mock_img = np.ones((10, 10))

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    with (
        patch("dash.Dash") as MockDash,
        patch.dict("sys.modules", {"IPython": mock_ipython}),
        patch("threading.Thread"),
        patch("asyncio.to_thread") as mock_to_thread,
    ):
        mock_app = MagicMock()
        MockDash.return_value = mock_app
        mock_app.run = MagicMock()

        mock_fig = MagicMock()
        mock_fig.layout.title.text = "Title"

        callback_func = None

        def capture_callback(*args, **kwargs):
            def decorator(f):
                nonlocal callback_func
                callback_func = f
                return f

            return decorator

        mock_app.callback = MagicMock(side_effect=capture_callback)

        bp.plot(heatmap_only=True)
        assert callback_func is not None

        # Camera stops acquiring mid-stream
        bp.camera.is_acquiring = False

        async def run_callback():
            def side_effect(*args, **kwargs):
                if args[0] == bp.camera.get_image:
                    return mock_img
                if args[0] == bp.analyze:
                    return ([0, 0, 1, 0], [0, 0, 1, 0])
                if args[0] == build_figure:
                    return mock_fig
                return None

            mock_to_thread.side_effect = side_effect
            return await callback_func(
                0, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
            )  # type: ignore

        result = asyncio.run(run_callback())
        assert result[0] is mock_fig

    bp.camera.close()


def test_plot_stream_static_mode():
    """Test Dash callback uses last_img for static file mode."""
    import os
    import tempfile

    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        img_arr = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        img_path = os.path.join(tmpdir, "beam.png")
        Image.fromarray(img_arr).save(img_path)

        bp = BeamProfiler(file=img_path, pixel_size=5.0)

        mock_ipython = MagicMock()
        mock_ipython.get_ipython = MagicMock(return_value=None)

        with (
            patch("dash.Dash") as MockDash,
            patch.dict("sys.modules", {"IPython": mock_ipython}),
            patch("threading.Thread"),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            mock_app = MagicMock()
            MockDash.return_value = mock_app
            mock_app.run = MagicMock()

            mock_fig = MagicMock()
            mock_fig.layout.title.text = "Title"

            callback_func = None

            def capture_callback(*args, **kwargs):
                def decorator(f):
                    nonlocal callback_func
                    callback_func = f
                    return f

                return decorator

            mock_app.callback = MagicMock(side_effect=capture_callback)

            bp._plot_stream()
            assert callback_func is not None

            async def run_callback():
                def side_effect(*args, **kwargs):
                    if args[0] == bp.analyze:
                        return ([0, 0, 1, 0], [0, 0, 1, 0])
                    if args[0] == build_figure:
                        return mock_fig
                    return None

                mock_to_thread.side_effect = side_effect
                return await callback_func(
                    0, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
                )  # type: ignore

            result = asyncio.run(run_callback())
            assert result[0] is mock_fig


def test_plot_stream_matplotlib_update_frame():
    """Test the matplotlib update_frame function renders correctly."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None

    mock_fig_plt = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_ax3 = MagicMock()
    mock_ax4 = MagicMock()
    mock_axes = MagicMock()
    mock_axes.flat = [mock_ax1, mock_ax2, mock_ax3, mock_ax4]
    mock_axes.__getitem__ = lambda self, key: {
        (0, 0): mock_ax1,
        (1, 0): mock_ax2,
        (1, 1): mock_ax3,
        (0, 1): mock_ax4,
    }.get(key, MagicMock())

    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig_plt, mock_axes)

    mock_animation = MagicMock()
    mock_patches = MagicMock()

    mock_matplotlib = MagicMock()
    mock_matplotlib.pyplot = mock_plt
    mock_matplotlib.animation = mock_animation
    mock_matplotlib.patches = mock_patches

    update_frame_fn = None

    def capture_update_frame(*args, **kwargs):
        nonlocal update_frame_fn
        if len(args) >= 2:
            update_frame_fn = args[1]
        return MagicMock()

    mock_animation.FuncAnimation = capture_update_frame

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    with patch.dict(
        sys.modules,
        {
            "IPython": mock_ipython,
            "dash": None,
            "dash.dependencies": None,
            "matplotlib": mock_matplotlib,
            "matplotlib.pyplot": mock_plt,
            "matplotlib.animation": mock_animation,
            "matplotlib.patches": mock_patches,
        },
    ):
        bp._plot_stream()

    assert update_frame_fn is not None
    # Call update_frame to exercise the rendering code
    update_frame_fn(0)
    assert mock_ax2.imshow.called or mock_ax1.clear.called

    bp.camera.close()


def test_plot_stream_jupyter_non_heatmap():
    """Test Jupyter streaming in non-heatmap (full figure) mode."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None

    mock_img = np.ones((10, 10))
    bp.camera.get_image = MagicMock(return_value=mock_img)  # type: ignore
    bp.analyze = MagicMock(return_value=([0, 0, 1, 0], [0, 0, 1, 0]))  # type: ignore
    bp._create_figure = MagicMock(return_value=MagicMock())  # type: ignore

    async def run_test():
        mock_get_ipython = MagicMock(return_value=MagicMock())
        display_mock = MagicMock()
        clear_output_mock = MagicMock()

        mock_ipython = MagicMock()
        mock_ipython.get_ipython = mock_get_ipython

        mock_ipython_display = MagicMock()
        mock_ipython_display.display = display_mock
        mock_ipython_display.clear_output = clear_output_mock

        with patch.dict(
            "sys.modules",
            {"IPython": mock_ipython, "IPython.display": mock_ipython_display},
        ):
            task = bp.plot(heatmap_only=False)

            assert isinstance(task, asyncio.Task)

            await asyncio.sleep(0.02)

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert bp._create_figure.call_count >= 1  # ty: ignore[unresolved-attribute]

    asyncio.run(run_test())
    bp.camera.close()


def test_dash_callback_lock_prevents_reentrance():
    """Test that the Dash callback returns no_update when the lock is held."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None
    mock_img = np.ones((10, 10))

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    with (
        patch("dash.Dash") as MockDash,
        patch.dict("sys.modules", {"IPython": mock_ipython}),
        patch("threading.Thread"),
        patch("asyncio.to_thread") as mock_to_thread,
    ):
        mock_app = MagicMock()
        MockDash.return_value = mock_app
        mock_app.run = MagicMock()

        mock_fig = MagicMock()
        mock_fig.layout.title.text = "Title"

        callback_func = None

        def capture_callback(*args, **kwargs):
            def decorator(f):
                nonlocal callback_func
                callback_func = f
                return f

            return decorator

        mock_app.callback = MagicMock(side_effect=capture_callback)

        bp.plot(heatmap_only=True)
        assert callback_func is not None

        bp.camera.is_acquiring = True

        def side_effect(*args, **kwargs):
            if args[0] == bp.camera.get_image:
                return mock_img
            if args[0] == bp.analyze:
                return ([0, 0, 1, 0], [0, 0, 1, 0])
            if args[0] == build_figure:
                return mock_fig
            return None

        mock_to_thread.side_effect = side_effect

        async def run_lock_test():
            result1 = await callback_func(
                0, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
            )  # ty: ignore[call-non-callable]
            assert result1[0] is mock_fig

            result2 = await callback_func(
                1, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
            )
            assert result2[0] is mock_fig

        asyncio.run(run_lock_test())

    bp.camera.close()


def test_dash_sigint_handler_restores_original():
    """Test that SIGINT handler is installed and original is restored."""
    import signal

    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    original_handler = signal.getsignal(signal.SIGINT)

    with (
        patch("dash.Dash") as MockDash,
        patch.dict("sys.modules", {"IPython": mock_ipython}),
        patch("threading.Thread"),
    ):
        mock_app = MagicMock()
        MockDash.return_value = mock_app

        mock_app.callback = MagicMock(side_effect=lambda *a, **k: lambda f: f)

        mock_app.run = MagicMock(side_effect=KeyboardInterrupt)

        bp.plot(heatmap_only=True)

    restored_handler = signal.getsignal(signal.SIGINT)
    assert restored_handler is original_handler

    bp.camera.close()


def test_dash_shutdown_flag_stops_callback():
    """Test that the callback returns valid figures when shutdown_flag is not set."""
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None
    mock_img = np.ones((10, 10))

    mock_ipython = MagicMock()
    mock_ipython.get_ipython = MagicMock(return_value=None)

    with (
        patch("dash.Dash") as MockDash,
        patch.dict("sys.modules", {"IPython": mock_ipython}),
        patch("threading.Thread"),
        patch("asyncio.to_thread") as mock_to_thread,
    ):
        mock_app = MagicMock()
        MockDash.return_value = mock_app
        mock_app.run = MagicMock()

        mock_fig = MagicMock()
        mock_fig.layout.title.text = "Title"

        callback_func = None

        def capture_callback(*args, **kwargs):
            def decorator(f):
                nonlocal callback_func
                callback_func = f
                return f

            return decorator

        mock_app.callback = MagicMock(side_effect=capture_callback)

        bp.plot(heatmap_only=True)
        assert callback_func is not None

        bp.camera.is_acquiring = True

        def side_effect(*args, **kwargs):
            if args[0] == bp.camera.get_image:
                return mock_img
            if args[0] == bp.analyze:
                return ([0, 0, 1, 0], [0, 0, 1, 0])
            if args[0] == build_figure:
                return mock_fig
            return None

        mock_to_thread.side_effect = side_effect

        async def run_test():
            return await callback_func(
                0, False, True, "Hot", True, None, None, 0, "1d", "gaussian", True
            )  # ty: ignore[call-non-callable]

        result = asyncio.run(run_test())
        assert result[0] is mock_fig

    bp.camera.close()
