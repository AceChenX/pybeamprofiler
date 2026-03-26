"""Tests for beamprofiler streaming and plot methods."""

import asyncio
from unittest.mock import MagicMock, patch

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

    mock_img = np.ones((10, 10))
    # Return 1 frame then None to break the infinite loop
    get_image_returns = [mock_img, None]

    assert bp.camera is not None
    bp.camera.get_image = MagicMock(side_effect=get_image_returns)  # type: ignore
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

                # Await task completion (it will finish because get_image returns None on 2nd call)
                await task

                # Check clear_output and display were called for the 1 successful frame
                clear_output_mock.assert_called_once_with(wait=True)
                display_mock.assert_called_once()

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
                await asyncio.sleep(0.1)

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
