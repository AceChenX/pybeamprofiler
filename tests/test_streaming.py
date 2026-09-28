"""Tests for the plot() entry points: single shots and the notebook stream.

The Dash GUI's own tick is covered in test_dash_app.py; how plot() starts and
stops the Dash server is in test_error_paths.py.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pybeamprofiler import beamprofiler
from pybeamprofiler.beamprofiler import BeamProfiler, _in_notebook


def _kernel_modules() -> dict[str, MagicMock]:
    """Stand-ins for IPython that look like a running Jupyter kernel."""
    display_module = MagicMock()
    ipython = MagicMock()
    shell = MagicMock()
    shell.kernel = MagicMock()  # a kernel is what makes it a notebook
    ipython.get_ipython = MagicMock(return_value=shell)
    return {"IPython": ipython, "IPython.display": display_module}


async def _wait_for(condition: Any, timeout: float = 3.0) -> None:
    """Poll *condition* until true. The loop hands its work to threads, so a
    fixed sleep would be flaky on a slow machine."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the stream")
        await asyncio.sleep(0.01)


async def _finish(task: asyncio.Task[None]) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.fixture
def bp():
    profiler = BeamProfiler(camera="simulated")
    yield profiler
    profiler.stop()
    assert profiler.camera is not None
    profiler.camera.close()


def test_plot_single(bp):
    with patch.object(bp, "_plot_single") as mock_plot_single:
        assert bp.plot(num_img=1) is None
    mock_plot_single.assert_called_once()


@pytest.mark.parametrize("num_img", [0, 2, 5, -3])
def test_unsupported_frame_counts_are_refused(bp, num_img):
    """Anything but 1 used to stream forever without a word."""
    with pytest.raises(ValueError, match="num_img"):
        bp.plot(num_img=num_img)


class TestNotebookDetection:
    def test_no_ipython_is_not_a_notebook(self):
        with patch.dict("sys.modules", {"IPython": None}):
            assert not _in_notebook()

    def test_plain_python_is_not_a_notebook(self):
        ipython = MagicMock()
        ipython.get_ipython = MagicMock(return_value=None)
        with patch.dict("sys.modules", {"IPython": ipython}):
            assert not _in_notebook()

    def test_a_terminal_ipython_session_is_not_a_notebook(self):
        """It has get_ipython() but no kernel. Treated as a notebook, every
        frame opened a new browser tab."""
        ipython = MagicMock()
        ipython.get_ipython = MagicMock(return_value=object())
        with patch.dict("sys.modules", {"IPython": ipython}):
            assert not _in_notebook()

    def test_a_kernel_is_a_notebook(self):
        with patch.dict("sys.modules", _kernel_modules()):
            assert _in_notebook()

    def test_a_terminal_ipython_session_gets_the_dash_gui(self, bp):
        ipython = MagicMock()
        ipython.get_ipython = MagicMock(return_value=object())
        with (
            patch.dict("sys.modules", {"IPython": ipython}),
            patch.object(bp, "_serve_dash") as serve,
            patch.object(bp, "_start_notebook_stream") as notebook,
        ):
            assert bp.plot() is None
        serve.assert_called_once()
        notebook.assert_not_called()


class TestNotebookStream:
    def test_renders_frames(self, bp):
        bp.camera.get_image = MagicMock(return_value=np.ones((10, 10)))
        bp.analyze = MagicMock(return_value=(None, None))
        bp._create_fast_figure = MagicMock(return_value=MagicMock())
        modules = _kernel_modules()
        display = modules["IPython.display"]

        async def run() -> None:
            with patch.dict("sys.modules", modules):
                task = bp.plot(heatmap_only=True)
                assert isinstance(task, asyncio.Task)
                await _wait_for(lambda: display.display.call_count >= 2)
                await _finish(task)
            assert display.clear_output.call_count >= 1
            assert task.done()

        asyncio.run(run())

    def test_full_figure_mode_uses_the_full_figure(self, bp):
        bp.camera.get_image = MagicMock(return_value=np.ones((10, 10)))
        bp.analyze = MagicMock(return_value=(None, None))
        bp._create_figure = MagicMock(return_value=MagicMock())
        bp._create_fast_figure = MagicMock()

        async def run() -> None:
            with patch.dict("sys.modules", _kernel_modules()):
                task = bp.plot(heatmap_only=False)
                assert isinstance(task, asyncio.Task)
                await _wait_for(lambda: bp._create_figure.call_count >= 1)
                await _finish(task)
            bp._create_fast_figure.assert_not_called()

        asyncio.run(run())

    def test_survives_a_failed_frame_and_a_dropped_one(self, bp):
        frames: list[Any] = [RuntimeError("camera hiccup"), TimeoutError(), np.ones((10, 10))]

        def get_image(*args, **kwargs):
            item = frames.pop(0) if frames else np.ones((10, 10))
            if isinstance(item, BaseException):
                raise item
            return item

        bp.camera.get_image = MagicMock(side_effect=get_image)
        bp.analyze = MagicMock(return_value=(None, None))
        bp._create_fast_figure = MagicMock(return_value=MagicMock())
        modules = _kernel_modules()

        async def run() -> None:
            with patch.dict("sys.modules", modules):
                task = bp.plot(heatmap_only=True)
                assert isinstance(task, asyncio.Task)
                await _wait_for(lambda: modules["IPython.display"].display.call_count >= 1)
                assert not task.done()
                await _finish(task)

        asyncio.run(run())

    def test_a_dead_camera_ends_the_stream_with_a_warning(self, bp, caplog):
        """A camera that has gone away fails every frame. The stream used to
        retry it forever, logging at debug level only."""
        bp.camera.get_image = MagicMock(side_effect=RuntimeError("Camera not opened."))

        async def run() -> None:
            with (
                patch.dict("sys.modules", _kernel_modules()),
                patch.object(beamprofiler, "_MAX_STREAM_FAILURES", 3),
            ):
                task = bp.plot(heatmap_only=True)
                assert isinstance(task, asyncio.Task)
                await asyncio.wait_for(task, timeout=5.0)

        with caplog.at_level(logging.WARNING, logger="pybeamprofiler.beamprofiler"):
            asyncio.run(run())
        assert "stopped after 3 failed frames" in caplog.text

    def test_rerunning_plot_replaces_the_previous_stream(self, bp):
        """Re-running the cell used to leave the first loop fetching from the
        same camera as the second, out of reach of stop()."""
        bp.camera.get_image = MagicMock(return_value=np.ones((10, 10)))
        bp.analyze = MagicMock(return_value=(None, None))
        bp._create_fast_figure = MagicMock(return_value=MagicMock())

        async def run() -> None:
            with patch.dict("sys.modules", _kernel_modules()):
                first = bp.plot(heatmap_only=True)
                second = bp.plot(heatmap_only=True)
                assert isinstance(first, asyncio.Task) and isinstance(second, asyncio.Task)
                await asyncio.sleep(0.05)
                assert first.done()
                assert not second.done()
                assert bp._stream_task is second
                await _finish(second)

        asyncio.run(run())

    def test_stop_waits_for_the_fetch_in_flight(self, bp):
        """Cancelling the task doesn't stop its worker thread. A fetch that
        outlived stop_acquisition() would restart acquisition on a GenICam
        camera, so stop() must wait for it."""
        events: list[str] = []
        entered, release = threading.Event(), threading.Event()

        def slow_get_image(*args, **kwargs):
            events.append("fetch started")
            entered.set()
            release.wait(5.0)
            events.append("fetch returned")
            return np.ones((10, 10))

        bp.camera.get_image = MagicMock(side_effect=slow_get_image)
        real_stop = bp.camera.stop_acquisition

        def recording_stop() -> None:
            events.append("acquisition stopped")
            real_stop()

        bp.camera.stop_acquisition = MagicMock(side_effect=recording_stop)
        bp.analyze = MagicMock(return_value=(None, None))
        bp._create_fast_figure = MagicMock(return_value=MagicMock())

        async def run() -> None:
            with patch.dict("sys.modules", _kernel_modules()):
                task = bp.plot(heatmap_only=True)
                assert isinstance(task, asyncio.Task)
                await _wait_for(entered.is_set)
                stopper = threading.Thread(target=bp.stop)
                stopper.start()
                await asyncio.sleep(0.1)
                assert "acquisition stopped" not in events, "stopped under a live fetch"
                release.set()
                await asyncio.to_thread(stopper.join, 5.0)
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run(run())
        assert events.index("fetch returned") < events.index("acquisition stopped")
        assert events.count("fetch started") == 1, "no new fetch may start after stop()"

    def test_leaving_the_context_manager_stops_the_stream(self):
        """``__exit__`` closed the camera but left the stream fetching from
        it, failing every frame for as long as the kernel lived."""

        async def run() -> None:
            with patch.dict("sys.modules", _kernel_modules()):
                with BeamProfiler(camera="simulated") as bp:
                    assert bp.camera is not None
                    bp.analyze = MagicMock(return_value=(None, None))
                    bp._create_fast_figure = MagicMock(return_value=MagicMock())
                    task = bp.plot(heatmap_only=True)
                    assert isinstance(task, asyncio.Task)
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.05)
                assert task is not None and task.done()
                assert bp.camera is not None and not bp.camera.is_acquiring

        asyncio.run(run())

    def test_without_an_event_loop_the_stream_runs_in_place(self, bp):
        """No running loop (unusual in a kernel): the stream runs until it
        ends, here because the camera keeps failing."""
        bp.camera.get_image = MagicMock(side_effect=RuntimeError("gone"))
        with (
            patch.dict("sys.modules", _kernel_modules()),
            patch.object(beamprofiler, "_MAX_STREAM_FAILURES", 2),
        ):
            assert bp._plot_stream() is None

    def test_starts_acquisition_if_the_camera_is_idle(self, bp):
        assert not bp.camera.is_acquiring
        with (
            patch.dict("sys.modules", {"IPython": None}),
            patch.object(bp, "_serve_dash"),
        ):
            bp.plot()
        assert bp.camera.is_acquiring
