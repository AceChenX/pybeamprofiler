"""Tests for beamprofiler streaming and plot methods."""

import asyncio
from unittest.mock import MagicMock, patch

import dash
import numpy as np

from pybeamprofiler.beamprofiler import BeamProfiler


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
        with patch("pybeamprofiler.beamprofiler.get_ipython", create=True):
            display_mock = MagicMock()
            clear_output_mock = MagicMock()

            mock_ipython_display = MagicMock()
            mock_ipython_display.display = display_mock
            mock_ipython_display.clear_output = clear_output_mock

            with patch.dict("sys.modules", {"IPython.display": mock_ipython_display}):
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
        with patch("pybeamprofiler.beamprofiler.get_ipython", create=True):
            display_mock = MagicMock()
            clear_output_mock = MagicMock()

            mock_ipython_display = MagicMock()
            mock_ipython_display.display = display_mock
            mock_ipython_display.clear_output = clear_output_mock

            with patch.dict("sys.modules", {"IPython.display": mock_ipython_display}):
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
        with patch("pybeamprofiler.beamprofiler.get_ipython", create=True):
            display_mock = MagicMock()
            clear_output_mock = MagicMock()

            mock_ipython_display = MagicMock()
            mock_ipython_display.display = display_mock
            mock_ipython_display.clear_output = clear_output_mock

            with patch.dict("sys.modules", {"IPython.display": mock_ipython_display}):
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
    bp = BeamProfiler(camera="simulated")
    assert bp.camera is not None
    mock_img = np.ones((10, 10))

    with (
        patch("dash.Dash") as MockDash,
        patch("pybeamprofiler.beamprofiler.get_ipython", side_effect=NameError, create=True),
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
            def side_effect(*args):
                if args[0] == bp.camera.get_image:
                    raise RuntimeError("Camera dead")
                return None

            mock_to_thread.side_effect = side_effect
            return await callback_func(0)  # type: ignore

        assert asyncio.run(run_callback_error()) is dash.no_update

        async def run_callback_none():
            def side_effect(*args):
                if args[0] == bp.camera.get_image:
                    return None
                return None

            mock_to_thread.side_effect = side_effect
            return await callback_func(0)  # type: ignore

        assert asyncio.run(run_callback_none()) is dash.no_update

        async def run_callback_success():
            def side_effect(*args):
                if args[0] == bp.camera.get_image:
                    return mock_img
                if args[0] == bp.analyze:
                    return ([0, 0, 1, 0], [0, 0, 1, 0])
                if args[0] == bp._create_fast_figure:
                    return mock_fig
                if args[0] == bp._create_figure:
                    return mock_fig
                return None

            mock_to_thread.side_effect = side_effect
            return await callback_func(0)  # type: ignore

        assert asyncio.run(run_callback_success()) is mock_fig
