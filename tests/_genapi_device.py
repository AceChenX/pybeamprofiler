"""A fake GenICam camera whose node map is the real GenApi engine.

Mocks cannot tell you whether an ROI write order or an exposure value would be
accepted by a camera: that is decided by GenApi, from the device's XML. So the
node map here *is* GenApi (``genicam.genapi.NodeMap``), loaded from a small
XML description that follows the SFNC conventions real cameras use:

* ``WidthMax`` is the full width *after* binning (``2048 / BinningHorizontal``).
* ``Width`` may not exceed ``WidthMax - OffsetX`` and ``OffsetX`` may not
  exceed ``WidthMax - Width``, so at full width the only legal offset is 0.
* ``Width``/``OffsetX`` step in 4s and ``Height``/``OffsetY`` in 2s; GenApi
  rejects anything off the increment rather than rounding it.
* ``Width``, ``Height``, ``BinningHorizontal`` and ``PixelFormat`` are locked
  while ``TLParamsLocked`` is set, which Harvesters does on every ``start()``.
* ``ExposureTime`` (µs, 20 to 10 000 000) is locked while ``ExposureAuto`` is
  on, and the device starts with it on, as many do out of the box.

Around that sits just enough of Harvesters -- a ``Harvester`` that enumerates
devices and whose ``update()`` destroys every acquirer it created, as the real
one does, and an acquirer with ``start``/``stop``/``try_fetch``/``destroy``.
Everything is in-process: no producer, no SDK, no hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

XML = """<?xml version="1.0" encoding="utf-8"?>
<RegisterDescription ModelName="FakeCam" VendorName="Fake" ToolTip="Fake camera" StandardNameSpace="None"
  SchemaMajorVersion="1" SchemaMinorVersion="1" SchemaSubMinorVersion="0"
  MajorVersion="1" MinorVersion="0" SubMinorVersion="0"
  ProductGuid="1F3C6A72-7842-4edd-9130-E2E90A2058BA" VersionGuid="7645D2A1-A41E-4ac6-B486-1531FB7BECE6"
  xmlns="http://www.genicam.org/GenApi/Version_1_1"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.genicam.org/GenApi/Version_1_1 http://www.genicam.org/GenApi/GenApiSchema_Version_1_1.xsd">
  <Category Name="Root" NameSpace="Standard">
    <pFeature>Width</pFeature><pFeature>Height</pFeature>
    <pFeature>OffsetX</pFeature><pFeature>OffsetY</pFeature>
    <pFeature>WidthMax</pFeature><pFeature>HeightMax</pFeature>
    <pFeature>BinningHorizontal</pFeature><pFeature>TLParamsLocked</pFeature>
    <pFeature>ExposureTime</pFeature><pFeature>Gain</pFeature>
    <pFeature>ExposureAuto</pFeature><pFeature>PixelFormat</pFeature>
    <pFeature>TriggerMode</pFeature>
  </Category>

  <Integer Name="TLParamsLocked" NameSpace="Standard">
    <Value>0</Value><Min>0</Min><Max>1</Max>
  </Integer>

  <Integer Name="BinningHorizontal" NameSpace="Standard">
    <pIsLocked>TLParamsLocked</pIsLocked>
    <Value>1</Value><Min>1</Min><Max>4</Max>
  </Integer>

  <IntSwissKnife Name="WidthMax" NameSpace="Standard">
    <pVariable Name="B">BinningHorizontal</pVariable>
    <Formula>2048 / B</Formula>
  </IntSwissKnife>
  <IntSwissKnife Name="HeightMax" NameSpace="Standard">
    <Formula>1536</Formula>
  </IntSwissKnife>

  <Integer Name="WidthVal"><Value>2048</Value></Integer>
  <Integer Name="HeightVal"><Value>1536</Value></Integer>
  <Integer Name="OffsetXVal"><Value>0</Value></Integer>
  <Integer Name="OffsetYVal"><Value>0</Value></Integer>

  <IntSwissKnife Name="WidthLimit">
    <pVariable Name="WM">WidthMax</pVariable><pVariable Name="OX">OffsetXVal</pVariable>
    <Formula>WM - OX</Formula>
  </IntSwissKnife>
  <IntSwissKnife Name="OffsetXLimit">
    <pVariable Name="WM">WidthMax</pVariable><pVariable Name="W">WidthVal</pVariable>
    <Formula>WM - W</Formula>
  </IntSwissKnife>
  <IntSwissKnife Name="HeightLimit">
    <pVariable Name="HM">HeightMax</pVariable><pVariable Name="OY">OffsetYVal</pVariable>
    <Formula>HM - OY</Formula>
  </IntSwissKnife>
  <IntSwissKnife Name="OffsetYLimit">
    <pVariable Name="HM">HeightMax</pVariable><pVariable Name="H">HeightVal</pVariable>
    <Formula>HM - H</Formula>
  </IntSwissKnife>

  <Integer Name="Width" NameSpace="Standard">
    <pIsLocked>TLParamsLocked</pIsLocked>
    <pValue>WidthVal</pValue><Min>16</Min><pMax>WidthLimit</pMax><Inc>4</Inc>
  </Integer>
  <Integer Name="Height" NameSpace="Standard">
    <pIsLocked>TLParamsLocked</pIsLocked>
    <pValue>HeightVal</pValue><Min>16</Min><pMax>HeightLimit</pMax><Inc>2</Inc>
  </Integer>
  <Integer Name="OffsetX" NameSpace="Standard">
    <pValue>OffsetXVal</pValue><Min>0</Min><pMax>OffsetXLimit</pMax><Inc>4</Inc>
  </Integer>
  <Integer Name="OffsetY" NameSpace="Standard">
    <pValue>OffsetYVal</pValue><Min>0</Min><pMax>OffsetYLimit</pMax><Inc>2</Inc>
  </Integer>

  <Float Name="ExposureTime" NameSpace="Standard">
    <pIsLocked>ExposureAutoVal</pIsLocked>
    <Value>5000</Value><Min>20</Min><Max>10000000</Max><Unit>us</Unit>
  </Float>
  <Float Name="Gain" NameSpace="Standard">
    <Value>1.5</Value><Min>0</Min><Max>24</Max><Unit>dB</Unit>
  </Float>

  <Enumeration Name="ExposureAuto" NameSpace="Standard">
    <EnumEntry Name="Off" NameSpace="Standard"><Value>0</Value></EnumEntry>
    <EnumEntry Name="Continuous" NameSpace="Standard"><Value>2</Value></EnumEntry>
    <pValue>ExposureAutoVal</pValue>
  </Enumeration>
  <Integer Name="ExposureAutoVal"><Value>2</Value></Integer>

  <Enumeration Name="PixelFormat" NameSpace="Standard">
    <pIsLocked>TLParamsLocked</pIsLocked>
    <EnumEntry Name="Mono8" NameSpace="Standard"><Value>17301505</Value></EnumEntry>
    <EnumEntry Name="Mono12" NameSpace="Standard"><Value>17825797</Value></EnumEntry>
    <pValue>PixelFormatVal</pValue>
  </Enumeration>
  <Integer Name="PixelFormatVal"><Value>17301505</Value></Integer>

  <Enumeration Name="TriggerMode" NameSpace="Standard">
    <EnumEntry Name="Off" NameSpace="Standard"><Value>0</Value></EnumEntry>
    <EnumEntry Name="On" NameSpace="Standard"><Value>1</Value></EnumEntry>
    <pValue>TriggerModeVal</pValue>
  </Enumeration>
  <Integer Name="TriggerModeVal"><Value>0</Value></Integer>
</RegisterDescription>
"""


def make_node_map() -> Any:
    """A fresh GenApi node map for one fake camera."""
    from genicam.genapi import NodeMap

    node_map = NodeMap()
    node_map.load_xml_from_string(XML)
    return node_map


@dataclass
class FakeComponent:
    """What ``buffer.payload.components[0]`` looks like to the camera code."""

    width: int
    height: int
    data: np.ndarray
    data_format: str = "Mono8"
    x_padding: int = 0


class FakeBuffer:
    """A fetched buffer; the context manager mirrors Harvesters' re-queue."""

    def __init__(self, component: FakeComponent, owner: FakeAcquirer) -> None:
        self.payload = SimpleNamespace(components=[component])
        self._owner = owner
        self._stops = owner.stops

    def __enter__(self) -> FakeBuffer:
        return self

    def __exit__(self, *exc: object) -> None:
        # Re-queueing a buffer that stop() or destroy() revoked while it was
        # held is the use-after-free that segfaults with a real producer.
        if self._owner.destroyed:
            raise AssertionError(
                "buffer re-queued on a destroyed acquirer (a segfault on hardware)"
            )
        if self._owner.stops != self._stops:
            raise AssertionError(
                "buffer re-queued after stop() revoked it (a segfault on hardware)"
            )


class FakeAcquirer:
    """Stands in for ``harvesters.core.ImageAcquirer``."""

    def __init__(self, node_map: Any) -> None:
        self.remote_device = SimpleNamespace(node_map=node_map)
        self.data_streams: list[Any] = []
        self.acquiring = False
        self.destroyed = False
        self.frames_ready = True
        # Every buffer arrives with packets missing, as on a lossy GigE link.
        self.incomplete = False
        self.stops = 0
        self.calls: list[str] = []

    def _check(self) -> None:
        if self.destroyed:
            raise AssertionError("acquirer used after destroy() (a segfault on hardware)")

    def start(self) -> None:
        self._check()
        self.calls.append("start")
        self.remote_device.node_map.TLParamsLocked.value = 1
        self.acquiring = True

    def stop(self) -> None:
        self._check()
        self.calls.append("stop")
        self.stops += 1
        self.remote_device.node_map.TLParamsLocked.value = 0
        self.acquiring = False

    def try_fetch(self, *, timeout: float) -> FakeBuffer | None:
        """Harvesters 1.4 ``try_fetch``: ``None`` on timeout or an incomplete buffer."""
        self._check()
        if not (self.acquiring and self.frames_ready) or self.incomplete:
            time.sleep(min(timeout, 0.005))
            return None
        node_map = self.remote_device.node_map
        width, height = int(node_map.Width.value), int(node_map.Height.value)
        fmt = str(node_map.PixelFormat.value)
        dtype = np.uint8 if fmt == "Mono8" else np.uint16
        data = np.full(width * height, 7, dtype=dtype)
        return FakeBuffer(FakeComponent(width, height, data, fmt), self)

    def fetch(self, *, timeout: float = 0) -> FakeBuffer:
        """Harvesters 1.4 ``fetch``, including the two ways it never returns.

        ``timeout=0`` means "wait forever", and an incomplete buffer is
        discarded and the wait starts again, so the timeout bounds each
        attempt rather than the call. Clear ``incomplete``/set
        ``frames_ready`` from the test to let a stuck call go.
        """
        while True:
            started = time.monotonic()
            while not (self.acquiring and self.frames_ready) or self.incomplete:
                if self.incomplete:
                    started = time.monotonic()  # a buffer arrived, incomplete: wait anew
                elif timeout > 0 and time.monotonic() - started > timeout:
                    from harvesters.core import TimeoutException

                    raise TimeoutException
                time.sleep(0.002)
            buffer = self.try_fetch(timeout=0.001)
            if buffer is not None:
                return buffer

    def destroy(self) -> None:
        self.calls.append("destroy")
        self.acquiring = False
        self.destroyed = True


@dataclass
class FakeDevice:
    """A ``DeviceInfo`` as far as the camera code reads one."""

    serial_number: str
    id_: str = "dev"
    model: str = "FakeCam"
    vendor: str = "Fake"
    cti: str = "/fake/Producer.cti"
    node_map: Any = None
    parent: Any = field(init=False)

    def __post_init__(self) -> None:
        # Harvesters reaches the producer through DeviceInfo -> Interface ->
        # System -> Producer; the camera code follows the same chain.
        producer = SimpleNamespace(path_name=self.cti)
        self.parent = SimpleNamespace(parent=SimpleNamespace(parent=producer))
        if self.node_map is None:
            self.node_map = make_node_map()


class FakeBus:
    """The devices attached to the machine, and the Harvesters built on them.

    ``harvester_class`` goes where ``pybeamprofiler.gen_camera.Harvester``
    would. Each Harvester it builds sees the devices only through producers
    no *other* live Harvester has initialised -- a GenTL producer's
    ``GCInitLib`` succeeds once per process -- which is exactly the behaviour
    that broke switching between two cameras on the same producer.
    """

    def __init__(self, devices: list[Any]) -> None:
        self.devices: list[Any] = devices
        self.harvesters: list[FakeHarvester] = []
        # The producer the (first) device is on, for tests to pass as cti_file.
        self.cti: str = devices[0].cti if devices else ""

    def harvester_class(self) -> FakeHarvester:
        h = FakeHarvester(self)
        self.harvesters.append(h)
        return h

    def producer_owner(self, cti: str) -> FakeHarvester | None:
        for h in self.harvesters:
            if cti in h.loaded:
                return h
        return None


class FakeHarvester:
    """Stands in for ``harvesters.core.Harvester``."""

    def __init__(self, bus: FakeBus) -> None:
        self._bus = bus
        self.files: list[str] = []
        self.loaded: set[str] = set()
        self.device_info_list: list[FakeDevice] = []
        self.acquirers: list[FakeAcquirer] = []
        self.updates = 0
        self.resets = 0

    def add_file(self, path: str) -> None:
        if path not in self.files:
            self.files.append(path)

    def update(self) -> None:
        # Like Harvesters: every acquirer this object created is destroyed.
        for ia in self.acquirers:
            ia.destroy()
        self.updates += 1
        self.loaded = set()
        for path in self.files:
            owner = self._bus.producer_owner(path)
            if owner is None or owner is self:
                self.loaded.add(path)
        self.device_info_list = [d for d in self._bus.devices if d.cti in self.loaded]

    def create(self, device: FakeDevice) -> FakeAcquirer:
        if device not in self.device_info_list:
            raise RuntimeError(f"device {device.serial_number} is not on this Harvester's list")
        for ia in self.acquirers:
            if not ia.destroyed and ia.remote_device.node_map is device.node_map:
                raise RuntimeError("GenTL: resource in use")
        ia = FakeAcquirer(device.node_map)
        self.acquirers.append(ia)
        return ia

    def reset(self) -> None:
        for ia in self.acquirers:
            ia.destroy()
        self.resets += 1
        self.files.clear()
        self.loaded = set()
        self.device_info_list = []
