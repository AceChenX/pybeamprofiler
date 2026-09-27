"""HarvesterCamera against a camera whose rules are enforced by GenApi itself.

The fake device in ``_genapi_device`` runs the real GenApi engine on an
SFNC-style description, so a write that a real camera would refuse -- an
offset beyond ``WidthMax - Width``, a width off its increment, a locked node
-- is refused here too, with the same exception. Behaviour that only mocks
had ever checked is pinned against those rules instead.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from _genapi_device import FakeBus, FakeDevice
from conftest import requires_genicam, requires_harvesters

from pybeamprofiler import discovery, gen_camera
from pybeamprofiler.gen_camera import HarvesterCamera

pytestmark = [requires_genicam, requires_harvesters]


@pytest.fixture
def bus(tmp_path, monkeypatch) -> FakeBus:
    """One fake camera on one fake producer, wired in as ``Harvester``."""
    cti = tmp_path / "FakeProducer.cti"
    cti.touch()
    fake = FakeBus([FakeDevice("SN-A", id_="dev-a", cti=str(cti))])
    monkeypatch.setattr(gen_camera, "Harvester", fake.harvester_class)
    return fake


@pytest.fixture
def camera(bus) -> Iterator[HarvesterCamera]:
    cam = HarvesterCamera(cti_file=bus.cti)
    cam.open()
    yield cam
    cam.close()


def _device_roi(cam: HarvesterCamera) -> tuple[int, int, int, int]:
    nm: Any = cam.node_map
    return (nm.OffsetX.value, nm.OffsetY.value, nm.Width.value, nm.Height.value)


class TestRoiUnderGenicamRules:
    """``set_roi`` wrote the offsets before the size and never looked at the
    increments. At full width ``OffsetX.max`` is 0, so every ROI with an
    offset was refused by a real camera -- and the error was swallowed, so
    the GUI reported the unchanged ROI as if it had worked."""

    def test_an_offset_roi_from_full_frame_is_applied(self, camera):
        assert _device_roi(camera) == (0, 0, 2048, 1536)
        camera.set_roi(100, 100, 200, 200)
        assert _device_roi(camera) == (100, 100, 200, 200)

    def test_moving_right_while_shrinking(self, camera):
        """Offset 1500 only fits once the width has shrunk to 400."""
        camera.set_roi(0, 0, 1024, 768)
        camera.set_roi(1500, 0, 400, 300)
        assert _device_roi(camera) == (1500, 0, 400, 300)

    def test_values_snap_down_onto_the_increments(self, camera):
        """Width and OffsetX step in 4s, Height and OffsetY in 2s."""
        camera.set_roi(3, 5, 301, 299)
        assert _device_roi(camera) == (0, 4, 300, 298)

    def test_out_of_range_values_are_clamped(self, camera):
        camera.set_roi(5000, 5000, 99_999, 99_999)
        assert _device_roi(camera) == (0, 0, 2048, 1536)

    def test_offsets_are_clamped_to_fit_the_size(self, camera):
        camera.set_roi(2000, 1500, 400, 300)
        assert _device_roi(camera) == (1648, 1236, 400, 300)

    def test_roi_info_reports_what_the_device_holds(self, camera):
        camera.set_roi(3, 5, 301, 299)
        assert camera.roi_info == {
            "offset_x": 0,
            "offset_y": 4,
            "width": 300,
            "height": 298,
            "max_width": 2048,
            "max_height": 1536,
        }
        assert (camera.width, camera.height) == (300, 298)
        assert (camera.width_pixels, camera.height_pixels) == (300, 298)

    def test_roi_info_follows_a_change_made_behind_its_back(self, camera):
        """The Setting panel can write Width directly."""
        camera.node_map.Width.value = 512
        assert camera.roi_info["width"] == 512

    def test_full_sensor_follows_binning(self, camera):
        """``WidthMax`` halves with 2x binning. It used to be read once at
        open(), so "Full Sensor" then asked for a 2048-wide ROI and failed."""
        camera.node_map.BinningHorizontal.value = 2
        camera.set_roi()
        assert _device_roi(camera) == (0, 0, 1024, 1536)
        assert camera.roi_info["max_width"] == 1024

    def test_a_streaming_camera_is_stopped_and_restarted(self, camera):
        """Width is locked while streaming; set_roi handles that itself."""
        camera.start_acquisition()
        camera.set_roi(0, 0, 512, 512)
        assert _device_roi(camera) == (0, 0, 512, 512)
        assert camera.is_acquiring
        assert camera.ia.calls[-2:] == ["stop", "start"]

    def test_an_idle_camera_is_left_idle(self, camera):
        camera.set_roi(0, 0, 512, 512)
        assert not camera.is_acquiring
        assert "start" not in camera.ia.calls

    def test_a_refusal_restores_the_previous_roi_and_says_so(self, camera):
        camera.set_roi(8, 8, 1024, 768)
        camera.node_map.TLParamsLocked.value = 1  # e.g. another client locked it
        with pytest.raises(RuntimeError, match=r"refused .* It is set to 1024×768 at \(8, 8\)"):
            camera.set_roi(100, 100, 200, 200)
        assert _device_roi(camera) == (8, 8, 1024, 768)
        assert camera.roi_info["offset_x"] == 8

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ((0, 0, 10.5, 100), "width must be a whole number"),
            ((0, 0, 0, 100), "width must be at least 1"),
            ((0, 0, 100, -4), "height must be at least 1"),
            (("left", 0, 100, 100), "offset_x must be a whole number"),
        ],
    )
    def test_meaningless_geometry_is_a_value_error(self, camera, args, message):
        with pytest.raises(ValueError, match=message):
            camera.set_roi(*args)
        assert _device_roi(camera) == (0, 0, 2048, 1536)

    def test_an_unopened_camera_refuses(self, bus):
        cam = HarvesterCamera(cti_file=bus.cti)
        with pytest.raises(RuntimeError, match="not opened"):
            cam.set_roi(0, 0, 100, 100)


class _Node:
    """A plain node with just the attributes a GenApi IInteger exposes."""

    def __init__(self, value: int, lo: int, hi: int, inc: int = 1) -> None:
        self.min, self.max, self.inc = lo, hi, inc
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: int) -> None:
        if not (self.min <= v <= self.max) or (v - self.min) % self.inc:
            raise ValueError(f"{v} outside [{self.min}, {self.max}] step {self.inc}")
        self._value = v


class TestRoiWithoutWidthMax:
    """TLSimu, the GenTL reference producer, has ``Width`` but no
    ``WidthMax``/``OffsetX``. The old clamp used a cached maximum of 0 and so
    asked for a 1-pixel-wide ROI."""

    def test_the_limit_comes_from_the_width_node(self, bus):
        cam = HarvesterCamera(cti_file=bus.cti)
        cam.node_map = type("NodeMap", (), {})()
        cam.node_map.Width = _Node(512, 8, 4096, 4)  # ty: ignore[unresolved-attribute]
        cam.node_map.Height = _Node(512, 8, 4096, 4)  # ty: ignore[unresolved-attribute]

        cam.set_roi(0, 0, 256, 256)

        assert (cam.node_map.Width.value, cam.node_map.Height.value) == (256, 256)  # ty: ignore[unresolved-attribute]
        assert cam.roi_info["max_width"] == 4096

    def test_an_offset_is_dropped_when_the_camera_cannot_offset(self, bus):
        cam = HarvesterCamera(cti_file=bus.cti)
        cam.node_map = type("NodeMap", (), {})()
        cam.node_map.Width = _Node(512, 8, 4096, 4)  # ty: ignore[unresolved-attribute]
        cam.node_map.Height = _Node(512, 8, 4096, 4)  # ty: ignore[unresolved-attribute]

        cam.set_roi(100, 100, 256, 256)

        assert cam.roi_info["offset_x"] == 0
        assert cam.roi_info["width"] == 256


@pytest.fixture
def two_cameras(tmp_path, monkeypatch) -> FakeBus:
    """Two cameras on one producer, which discovery also finds."""
    cti = tmp_path / "FakeProducer.cti"
    cti.touch()
    fake = FakeBus(
        [
            FakeDevice("SN-A", id_="dev-a", cti=str(cti)),
            FakeDevice("SN-B", id_="dev-b", cti=str(cti)),
        ]
    )
    monkeypatch.setattr(gen_camera, "Harvester", fake.harvester_class)
    monkeypatch.setattr(discovery, "find_cti_files", lambda: [str(cti)])
    return fake


def _option(serial: str) -> discovery.CameraOption:
    return discovery._describe({"vendor": "Fake", "model": "FakeCam", "serial_number": serial})


class TestOneHarvesterPerProcess:
    """Every camera and every discovery pass built its own Harvester. A GenTL
    producer initialises once per process, so the second Harvester was
    refused the producer and saw no devices: with a camera open, a rescan
    found nothing and switching to another camera on the same producer
    failed with "No GenICam cameras found"."""

    def test_discovery_lists_every_camera_while_one_is_open(self, two_cameras):
        a = HarvesterCamera(cti_file=two_cameras.cti, serial_number="SN-A")
        a.open()
        try:
            found = [c["serial_number"] for c in discovery.list_cameras()]
            assert found == ["SN-A", "SN-B"]
            # Harvester.update() destroys live acquirers; discovery must not call it.
            assert not a.ia.destroyed
        finally:
            a.close()

    def test_switching_between_two_cameras_on_one_producer(self, two_cameras):
        a = HarvesterCamera(cti_file=two_cameras.cti, serial_number="SN-A")
        a.open()
        b = discovery.open_camera(_option("SN-B"))  # opened before A is released
        assert isinstance(b, HarvesterCamera)
        a.close()
        try:
            assert b.serial_number == "SN-B"
            assert not b.ia.destroyed
            assert b.node_map.Width.value == 2048
        finally:
            b.close()

    def test_the_producer_is_released_when_the_last_camera_closes(self, two_cameras):
        a = HarvesterCamera(cti_file=two_cameras.cti, serial_number="SN-A")
        b = HarvesterCamera(cti_file=two_cameras.cti, serial_number="SN-B")
        a.open()
        b.open()
        a.close()
        assert two_cameras.producer_owner(two_cameras.cti) is not None
        b.close()
        assert two_cameras.producer_owner(two_cameras.cti) is None
        assert gen_camera._SHARED.harvester is None

    def test_a_failed_open_releases_the_producer(self, two_cameras):
        """In Jupyter: mistype the serial, fix it, run the cell again. The
        failed attempt kept its producer, so the corrected one found nothing."""
        with pytest.raises(RuntimeError, match="serial number 'SN-X' not found"):
            HarvesterCamera(cti_file=two_cameras.cti, serial_number="SN-X").open()
        assert two_cameras.producer_owner(two_cameras.cti) is None

        cam = HarvesterCamera(cti_file=two_cameras.cti, serial_number="SN-A")
        cam.open()
        assert cam.serial_number == "SN-A"
        cam.close()

    def test_a_failure_after_the_device_opened_releases_it(self, two_cameras, monkeypatch):
        def broken(self):
            raise RuntimeError("register read failed")

        monkeypatch.setattr(HarvesterCamera, "_detect_pixel_size", broken)
        cam = HarvesterCamera(cti_file=two_cameras.cti, serial_number="SN-A")
        with pytest.raises(RuntimeError, match="register read failed"):
            cam.open()
        assert cam.ia is None
        assert two_cameras.producer_owner(two_cameras.cti) is None


class TestCloseLeavesNothingBehind:
    """close() destroyed the acquirer but kept ``ia`` and ``node_map``. Reading
    a node through them afterwards segfaulted the interpreter -- which is
    what a Jupyter panel still on screen did after the camera was swapped."""

    def test_close_drops_every_handle(self, camera):
        camera.close()
        assert camera.ia is None
        assert camera.node_map is None
        assert camera.h is None
        assert not camera.is_acquiring

    def test_close_is_idempotent(self, camera):
        camera.close()
        camera.close()

    def test_a_closed_camera_says_so(self, camera):
        camera.close()
        with pytest.raises(RuntimeError, match="Camera not opened"):
            camera.get_image(timeout=0.1)

    def test_close_forgets_the_feature_cache(self, camera):
        assert camera._discover_features()
        camera.close()
        assert camera._feature_cache is None
        assert camera._discover_features() == {}

    def test_a_closed_camera_opens_again(self, camera):
        first = camera.ia
        camera.close()
        camera.open()
        assert camera.ia is not None and camera.ia is not first
        assert camera.node_map.Width.value == 2048


def _timed(fn: Any, limit: float = 3.0) -> tuple[bool, float, BaseException | None]:
    """Run ``fn`` in a thread; report whether it finished, how long it took, what it raised."""
    result: dict[str, Any] = {}

    def run() -> None:
        started = time.monotonic()
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            result["exc"] = exc
        result["t"] = time.monotonic() - started

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(limit)
    return (not worker.is_alive(), result.get("t", limit), result.get("exc"))


class TestGetImageTimeoutIsABound:
    """``get_image(timeout)`` called Harvesters' ``fetch()``, which restarts its
    timeout after every incomplete buffer and treats ``timeout=0`` as
    "forever". The Dash tick holds the GUI lock around this call, so a lossy
    GigE link froze every control, Pause included."""

    def test_every_buffer_incomplete_still_times_out(self, camera):
        camera.start_acquisition()
        camera.ia.incomplete = True
        try:
            finished, took, exc = _timed(lambda: camera.get_image(timeout=0.2))
        finally:
            camera.ia.incomplete = False  # lets an old-style fetch() go
        assert finished, "get_image(timeout=0.2) was still waiting after 3 s"
        assert isinstance(exc, TimeoutError)
        assert took < 1.0

    def test_zero_timeout_polls(self, camera):
        camera.start_acquisition()
        camera.ia.frames_ready = False
        try:
            finished, took, exc = _timed(lambda: camera.get_image(timeout=0))
        finally:
            camera.ia.frames_ready = True
        assert finished, "get_image(timeout=0) blocked instead of polling"
        assert isinstance(exc, TimeoutError)
        assert "No frame was ready" in str(exc)
        assert took < 0.5

    def test_a_poll_returns_a_frame_that_is_ready(self, camera):
        img = camera.get_image(timeout=0)
        assert img.shape == (1536, 2048)

    def test_a_long_timeout_returns_as_soon_as_a_frame_arrives(self, camera):
        camera.start_acquisition()
        camera.ia.frames_ready = False
        threading.Timer(0.15, lambda: setattr(camera.ia, "frames_ready", True)).start()
        finished, took, exc = _timed(lambda: camera.get_image(timeout=5.0))
        assert finished and exc is None
        assert 0.1 < took < 1.0


class TestDeviceAccessIsSerialised:
    """Stopping or closing from one thread while another held a fetched
    buffer freed the buffer under it: 7 of 8 runs segfaulted on TLSimu.
    ``BeamProfiler.stop()`` in Jupyter does exactly that to the
    ``asyncio.to_thread(get_image)`` still in flight, and the Dash SIGINT
    handler closes the camera without the GUI lock."""

    @staticmethod
    def _park_inside_fetch(monkeypatch) -> tuple[threading.Event, threading.Event]:
        """Make get_image() stop, holding its buffer, until released."""
        inside, release = threading.Event(), threading.Event()
        real = gen_camera._to_mono

        def slow_copy(component: Any) -> Any:
            inside.set()
            release.wait(2.0)
            return real(component)

        monkeypatch.setattr(gen_camera, "_to_mono", slow_copy)
        return inside, release

    @pytest.mark.parametrize("action", ["stop_acquisition", "close"])
    def test_waits_for_the_frame_being_copied(self, camera, monkeypatch, action):
        camera.start_acquisition()
        inside, release = self._park_inside_fetch(monkeypatch)
        errors: list[BaseException] = []

        def grab() -> None:
            try:
                camera.get_image(timeout=1.0)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        fetcher = threading.Thread(target=grab, daemon=True)
        fetcher.start()
        assert inside.wait(1.0)

        other = threading.Thread(target=getattr(camera, action), daemon=True)
        other.start()
        other.join(0.2)
        still_waiting = other.is_alive()
        release.set()
        fetcher.join(2.0)
        other.join(2.0)

        assert still_waiting, f"{action}() ran while another thread held a buffer"
        assert errors == []

    def test_a_stop_ends_a_wait_in_progress_without_restarting(self, camera):
        """get_image() starts a stopped camera, but only when called: a stop
        from another thread mid-wait must stick."""
        camera.start_acquisition()
        camera.ia.frames_ready = False
        finished, took, exc = _timed(
            lambda: (
                threading.Timer(0.1, camera.stop_acquisition).start(),
                camera.get_image(timeout=2.0),
            )
        )
        camera.ia.frames_ready = True
        assert finished and isinstance(exc, TimeoutError)
        assert took < 1.0
        assert not camera.is_acquiring
        assert camera.ia.calls[-1] == "stop"

    def test_stop_waits_at_most_one_slice(self, camera):
        """Waiting for the lock must not mean waiting out the whole timeout."""
        camera.start_acquisition()
        camera.ia.frames_ready = False

        def grab() -> None:
            with pytest.raises(TimeoutError):
                camera.get_image(timeout=2.0)

        fetcher = threading.Thread(target=grab, daemon=True)
        fetcher.start()
        time.sleep(0.05)
        started = time.monotonic()
        camera.stop_acquisition()
        assert time.monotonic() - started < 0.5
        camera.ia.frames_ready = True
        fetcher.join(3.0)


class TestPanelControlsOfAClosedCamera:
    """A Jupyter panel stays on screen after its camera is swapped out or
    closed. Its controls held nodes of the released device, and touching one
    segfaulted the kernel (reproduced on TLSimu)."""

    def test_a_control_refuses_once_its_camera_is_closed(self, camera, caplog):
        (dropdown,) = camera._create_feature_controls(["TriggerMode"], {})
        old_node_map = camera.node_map
        camera.close()

        with caplog.at_level("WARNING"):
            dropdown.value = "On"

        assert old_node_map.TriggerMode.value == "Off"
        assert "since been closed" in caplog.text

    def test_a_control_refuses_after_a_reopen(self, camera):
        (dropdown,) = camera._create_feature_controls(["TriggerMode"], {})
        old_node_map = camera.node_map
        camera.close()
        camera.open()

        dropdown.value = "On"

        assert old_node_map.TriggerMode.value == "Off"
        assert camera.node_map.TriggerMode.value == "Off"

    def test_a_live_control_still_writes(self, camera):
        (dropdown,) = camera._create_feature_controls(["TriggerMode"], {})
        dropdown.value = "On"
        assert camera.node_map.TriggerMode.value == "On"

    def test_setting_on_a_closed_camera_offers_no_device_controls(self, camera, monkeypatch):
        import IPython.display

        monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
        camera.close()
        camera.setting()  # must not touch the released node map
        assert camera._create_genicam_controls({}) == []


class TestExposureAndGain:
    """Out-of-range exposures escaped as a GenICam OutOfRangeException --
    which is not a ValueError, so nothing caught it -- after acquisition had
    already been stopped. A refused write was recorded anyway. And open()
    never read the device, so exposure and gain showed the class defaults
    whatever the camera was actually doing."""

    def test_open_reads_exposure_and_gain_from_the_device(self, camera):
        assert camera.exposure_time == pytest.approx(0.005)
        assert camera.gain == pytest.approx(1.5)

    def test_open_turns_auto_exposure_off(self, camera):
        assert camera.node_map.ExposureAuto.value == "Off"

    def test_exposure_range_comes_from_the_node(self, camera):
        assert camera.exposure_range == pytest.approx((20e-6, 10.0))

    def test_too_long_an_exposure_is_clamped_and_streaming_resumes(self, camera):
        camera.start_acquisition()
        camera.set_exposure(20.0)
        assert camera.node_map.ExposureTime.value == pytest.approx(10_000_000)
        assert camera.exposure_time == pytest.approx(10.0)
        assert camera.is_acquiring

    def test_too_short_an_exposure_is_clamped(self, camera):
        camera.set_exposure(1e-7)
        assert camera.node_map.ExposureTime.value == pytest.approx(20)
        assert camera.exposure_time == pytest.approx(20e-6)

    def test_a_refused_exposure_raises_and_changes_nothing(self, camera):
        camera.start_acquisition()
        camera.node_map.ExposureAuto.value = "Continuous"  # locks ExposureTime
        with pytest.raises(RuntimeError, match="refused an exposure"):
            camera.set_exposure(0.02)
        assert camera.exposure_time == pytest.approx(0.005)
        assert camera.is_acquiring

    def test_gain_is_clamped_and_read_back(self, camera):
        camera.set_gain(99.0)
        assert camera.gain == pytest.approx(24.0)
        camera.set_gain(-3.0)
        assert camera.gain == pytest.approx(0.0)
        assert camera.node_map.Gain.value == pytest.approx(0.0)

    def test_an_exposure_left_on_the_device_is_picked_up(self, bus):
        """E.g. 8 s set in pylon Viewer before this session. It used to read
        as the 10 ms default, which also sized every fetch timeout."""
        node_map = bus.devices[0].node_map
        node_map.ExposureAuto.value = "Off"
        node_map.ExposureTime.value = 8_000_000
        cam = HarvesterCamera(cti_file=bus.cti)
        cam.open()
        try:
            assert cam.exposure_time == pytest.approx(8.0)
        finally:
            cam.close()


class TestExposurePanel:
    """The Jupyter exposure slider spanned whole decades around the camera's
    range, and its observer let the camera's refusal escape."""

    @staticmethod
    def _slider(camera, monkeypatch) -> Any:
        import IPython.display
        import ipywidgets as widgets

        shown: list[Any] = []
        monkeypatch.setattr(IPython.display, "display", lambda w, *a, **k: shown.append(w))
        camera.setting()

        def walk(w: Any) -> Iterator[Any]:
            yield w
            for child in getattr(w, "children", ()):
                yield from walk(child)

        return next(w for w in walk(shown[0]) if isinstance(w, widgets.FloatLogSlider))

    def test_the_slider_stops_at_the_camera_limits(self, camera, monkeypatch):
        slider = self._slider(camera, monkeypatch)
        assert slider.base**slider.min == pytest.approx(20e-6)
        assert slider.base**slider.max == pytest.approx(10.0)

    def test_a_refusal_from_the_slider_is_logged_not_raised(self, camera, monkeypatch, caplog):
        slider = self._slider(camera, monkeypatch)
        camera.node_map.ExposureAuto.value = "Continuous"  # locks ExposureTime
        with caplog.at_level("ERROR"):
            slider.value = 0.02
        assert "refused an exposure" in caplog.text


def _component(fmt: str, samples: Any, width: int, height: int, padding_x: int = 0) -> Any:
    """A real Harvesters ``Component2DImage`` over the given raw samples."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np
    from harvesters.core import Component2DImage
    from harvesters.util.pfnc import dict_by_names

    raw = np.asarray(samples).tobytes()
    buffer = SimpleNamespace(
        width=width,
        height=height,
        padding_x=padding_x,
        raw_buffer=raw,
        pixel_format=dict_by_names[fmt],
        delivered_image_height=height,
    )
    return Component2DImage(buffer=buffer, part=None, node_map=MagicMock())


class TestPixelFormats:
    """Colour payloads were collapsed with luminance weights in RGB order
    whatever the format, a plain mean otherwise, and Bayer passed as mono.
    Checked with Harvesters' own Component2DImage, so the layouts are the
    ones a camera delivers."""

    W, H = 8, 4

    def test_clipping_in_one_colour_channel_stays_visible(self):
        import numpy as np

        rgb = np.zeros((self.H, self.W, 3), np.uint8)
        rgb[..., 0] = 255  # a red laser saturating the red channel
        img = gen_camera._to_mono(_component("RGB8", rgb, self.W, self.H))
        assert img.max() == 255  # was 76: the saturation check never fired

    def test_bgr_and_rgb_of_the_same_scene_agree(self):
        import numpy as np

        rgb = np.zeros((self.H, self.W, 3), np.uint8)
        rgb[..., 0] = 200
        from_rgb = gen_camera._to_mono(_component("RGB8", rgb, self.W, self.H))
        from_bgr = gen_camera._to_mono(_component("BGR8", rgb[..., ::-1].copy(), self.W, self.H))
        assert np.array_equal(from_rgb, from_bgr)  # was 59 vs 22

    def test_alpha_is_not_part_of_the_image(self):
        import numpy as np

        rgba = np.zeros((self.H, self.W, 4), np.uint8)
        rgba[..., 3] = 255
        img = gen_camera._to_mono(_component("RGBa8", rgba, self.W, self.H))
        assert img.max() == 0  # was 63 everywhere

    def test_yuv_gives_its_luma(self):
        import numpy as np

        yuyv = np.tile(np.array([100, 128], np.uint8), self.W * self.H)  # Y=100, neutral chroma
        img = gen_camera._to_mono(_component("YUV422_8", yuyv, self.W, self.H))
        assert (img == 100).all()  # was 114: chroma averaged in

    def test_uyvy_carries_luma_second(self):
        import numpy as np

        uyvy = np.tile(np.array([128, 100], np.uint8), self.W * self.H)
        img = gen_camera._to_mono(_component("YUV422_8_UYVY", uyvy, self.W, self.H))
        assert (img == 100).all()

    def test_bayer_is_returned_as_the_raw_mosaic(self):
        import numpy as np

        mosaic = np.full((self.H, self.W), 100, np.uint8)
        mosaic[0::2, 0::2] = 200
        img = gen_camera._to_mono(_component("BayerRG8", mosaic, self.W, self.H))
        assert np.array_equal(img, mosaic)

    @pytest.mark.parametrize("padding", [8, 3])
    def test_line_padding_is_cropped(self, padding):
        """Harvesters leaves the padding in (it discards its own numpy.delete)."""
        import numpy as np

        rows = np.zeros((self.H, self.W + padding), np.uint8)
        rows[:, : self.W] = np.arange(self.W, dtype=np.uint8) + 1
        img = gen_camera._to_mono(_component("Mono8", rows, self.W, self.H, padding_x=padding))
        assert img.shape == (self.H, self.W)
        assert (img == np.arange(self.W) + 1).all()

    def test_packed_mono_arrives_unpacked(self):
        """Harvesters 1.4 unpacks Mono12p itself; nothing to reject."""
        import numpy as np

        packed = bytearray()
        for _ in range(self.W * self.H // 2):  # two 12-bit pixels, both 0xABC, per 3 bytes
            packed += bytes([0xBC, 0xCA, 0xAB])
        img = gen_camera._to_mono(
            _component("Mono12p", np.frombuffer(bytes(packed), np.uint8), self.W, self.H)
        )
        assert img.dtype == np.uint16 and (img == 0xABC).all()

    def test_chroma_subsampled_yuv_is_refused_clearly(self):
        import numpy as np

        data = np.zeros(int(self.W * self.H * 1.5), np.uint8)
        with pytest.raises(ValueError, match="whole number of samples per pixel"):
            gen_camera._to_mono(_component("YCbCr411_8", data, self.W, self.H))


class TestBitDepth:
    """get_image() gave a Mono12 frame as uint16 with nothing saying it was
    12-bit, so the dtype-based saturation check waited for 65535 and a frame
    with 9% of its pixels clipped at 4095 raised no warning."""

    @pytest.mark.parametrize(
        ("fmt", "bits"),
        [
            ("Mono8", 8),
            ("Mono10", 10),
            ("Mono10p", 10),
            ("Mono12", 12),
            ("Mono12Packed", 12),
            ("Mono16", 16),
            ("BayerRG12", 12),
            ("RGB8", 8),
            ("BGRa8", 8),
            ("YUV422_8", 8),
            ("YCbCr411_8", 8),
            ("Coord3D_C16", 16),
            ("", None),
            ("Custom", None),
        ],
    )
    def test_from_the_format_name(self, fmt, bits):
        assert gen_camera._bit_depth_of(fmt) == bits

    def test_a_mono12_camera_reports_12_bits(self, bus):
        bus.devices[0].node_map.PixelFormat.value = "Mono12"
        cam = HarvesterCamera(cti_file=bus.cti)
        cam.open()
        try:
            assert cam.bit_depth == 12
            assert cam.get_image(timeout=1.0).dtype.name == "uint16"
            assert cam.bit_depth == 12
        finally:
            cam.close()

    def test_bit_depth_follows_a_format_change(self, camera):
        assert camera.bit_depth == 8
        camera.node_map.PixelFormat.value = "Mono12"
        camera.get_image(timeout=1.0)
        assert camera.bit_depth == 12

    def test_a_bayer_format_is_flagged(self, camera, caplog):
        with caplog.at_level("WARNING"):
            camera._note_pixel_format("BayerRG8")
        assert camera.bit_depth == 8
        assert "Mono pixel format" in caplog.text

    def test_the_simulator_is_8_bit(self):
        from pybeamprofiler.simulated import SimulatedCamera

        assert SimulatedCamera().bit_depth == 8


class _SerialNotImplemented:
    """A DeviceInfo whose producer does not implement DEVICE_INFO_SERIAL_NUMBER.

    Harvesters' property_dict has ``None`` for it; the live attribute raises.
    """

    def __init__(self, base: FakeDevice) -> None:
        self.cti, self.id_, self.model, self.vendor = base.cti, base.id_, base.model, base.vendor
        self.node_map, self.parent = base.node_map, base.parent
        self.property_dict = {
            "serial_number": None,
            "id_": base.id_,
            "model": base.model,
            "vendor": base.vendor,
        }

    @property
    def serial_number(self) -> str:
        from genicam.gentl import NotImplementedException

        raise NotImplementedException("DEVICE_INFO_SERIAL_NUMBER")


class TestDiscoveryFindsWhatWasPicked:
    def _bus(self, tmp_path, monkeypatch, devices: list[Any]) -> FakeBus:
        cti = tmp_path / "FakeProducer.cti"
        cti.touch()
        for device in devices:
            device.cti = str(cti)
            device.__post_init__()
        fake = FakeBus(devices)
        monkeypatch.setattr(gen_camera, "Harvester", fake.harvester_class)
        monkeypatch.setattr(discovery, "find_cti_files", lambda: [str(cti)])
        return fake

    def test_a_serial_less_camera_opens_the_one_picked(self, tmp_path, monkeypatch):
        self._bus(tmp_path, monkeypatch, [FakeDevice("", id_="devA"), FakeDevice("", id_="devB")])
        options = discovery.discover_cameras(include_simulated=False)
        assert [o.key for o in options] == ["genicam:devA", "genicam:devB"]

        cam = discovery.open_camera(options[1])
        try:
            assert isinstance(cam, HarvesterCamera)
            assert cam.device_id == "devB"  # used to be devA, the first enumerated
        finally:
            cam.close()

    def test_serials_match_exactly(self, tmp_path, monkeypatch):
        bus = self._bus(tmp_path, monkeypatch, [FakeDevice("24001234"), FakeDevice("4001234")])
        cam = HarvesterCamera(cti_file=bus.cti, serial_number="4001234")
        cam.open()
        try:
            assert cam.serial_number == "4001234"  # a substring match took 24001234
        finally:
            cam.close()

    def test_one_device_without_a_serial_field_hides_nothing(self, tmp_path, monkeypatch):
        good = FakeDevice("SN-GOOD", id_="good")
        odd = FakeDevice("ignored", id_="odd")
        bus = self._bus(tmp_path, monkeypatch, [odd, good])
        bus.devices[0] = _SerialNotImplemented(odd)

        found = discovery.list_cameras()
        assert [(c["serial_number"], c["id"]) for c in found] == [("", "odd"), ("SN-GOOD", "good")]

        cam = HarvesterCamera(cti_file=bus.cti, serial_number="SN-GOOD")
        cam.open()
        try:
            assert cam.device_id == "good"
        finally:
            cam.close()

    def test_a_camera_found_through_the_env_var_can_be_reopened(self, tmp_path, monkeypatch):
        """FlirCamera/BaslerCamera fall back to GENICAM_GENTL64_PATH, but
        discovery never read it: the camera was missing from the dropdown,
        and switching away from it could not be undone."""
        from pybeamprofiler.basler import BaslerCamera

        producer_dir = tmp_path / "gentl"
        producer_dir.mkdir()
        cti = producer_dir / "Producer.cti"
        cti.touch()
        fake = FakeBus([FakeDevice("SN-ENV", cti=str(cti))])
        monkeypatch.setattr(gen_camera, "Harvester", fake.harvester_class)
        monkeypatch.setenv("GENICAM_GENTL64_PATH", str(producer_dir))

        assert [c["serial_number"] for c in discovery.list_cameras()] == ["SN-ENV"]

        cam = BaslerCamera()
        cam.open()
        option = discovery.describe_open_camera(cam)
        cam.close()  # e.g. the user switched to the simulator

        again = discovery.open_camera(option)
        try:
            assert isinstance(again, HarvesterCamera)
            assert again.serial_number == "SN-ENV"
        finally:
            again.close()
