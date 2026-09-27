"""HarvesterCamera against a camera whose rules are enforced by GenApi itself.

The fake device in ``_genapi_device`` runs the real GenApi engine on an
SFNC-style description, so a write that a real camera would refuse -- an
offset beyond ``WidthMax - Width``, a width off its increment, a locked node
-- is refused here too, with the same exception. Behaviour that only mocks
had ever checked is pinned against those rules instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from _genapi_device import FakeBus, FakeDevice
from conftest import requires_genicam, requires_harvesters

from pybeamprofiler import gen_camera
from pybeamprofiler.gen_camera import HarvesterCamera

pytestmark = [requires_genicam, requires_harvesters]


@pytest.fixture
def bus(tmp_path, monkeypatch) -> FakeBus:
    """One fake camera on one fake producer, wired in as ``Harvester``."""
    cti = tmp_path / "FakeProducer.cti"
    cti.touch()
    fake = FakeBus([FakeDevice("SN-A", id_="dev-a", cti=str(cti))])
    fake.cti = str(cti)  # ty: ignore[unresolved-attribute]
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
