"""End to end against TLSimu, the EMVA GenTL reference producer.

TLSimu is a real GenTL producer with four simulated cameras, so these tests
run the genuine stack -- GCInitLib, enumeration, acquisition, buffers --
with no hardware and no vendor SDK. It is how the shared-Harvester, close()
and locking bugs were first reproduced.

genicam 1.6 no longer ships TLSimu. The genicam 1.5 wheels do (next to
``genicam/__init__.py``); point ``PYBEAMPROFILER_TLSIMU_CTI`` at a copy of
``TLSimu.cti``, with its ``libVirtualFG`` library beside it, to run these.
Otherwise they are skipped.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator

import pytest
from conftest import requires_harvesters

from pybeamprofiler import discovery
from pybeamprofiler.gen_camera import HarvesterCamera


def _tlsimu() -> str | None:
    path = os.environ.get("PYBEAMPROFILER_TLSIMU_CTI")
    if path and os.path.isfile(path):
        return path
    try:
        import genicam
    except ImportError:
        return None
    bundled = os.path.join(os.path.dirname(genicam.__file__), "TLSimu.cti")
    return bundled if os.path.isfile(bundled) else None


TLSIMU = _tlsimu()

pytestmark = [
    requires_harvesters,
    pytest.mark.skipif(TLSIMU is None, reason="set PYBEAMPROFILER_TLSIMU_CTI to a TLSimu.cti"),
]

# TLSimu's first interface carries a mono and a colour camera.
MONO, COLOUR = "SN_InterfaceA_0", "SN_InterfaceA_1"


@pytest.fixture
def mono() -> Iterator[HarvesterCamera]:
    cam = HarvesterCamera(cti_file=TLSIMU, serial_number=MONO)
    cam.open()
    yield cam
    cam.close()


def test_discovery_sees_every_camera_while_one_is_open(mono):
    serials = [c["serial_number"] for c in discovery.list_cameras(TLSIMU)]
    assert MONO in serials and COLOUR in serials and len(serials) == 4


def test_switching_between_cameras_on_one_producer(mono, monkeypatch):
    monkeypatch.setattr(discovery, "find_cti_files", lambda: [TLSIMU])
    option = discovery._describe(
        {"vendor": "EMVA_D", "model": "TLSimuColor", "serial_number": COLOUR}
    )
    colour = discovery.open_camera(option)  # opened while the mono camera is still open
    mono.close()
    try:
        assert colour.get_image(timeout=3.0).ndim == 2
    finally:
        colour.close()


def test_a_corrected_retry_after_a_failed_open_works():
    with pytest.raises(RuntimeError, match="not found"):
        HarvesterCamera(cti_file=TLSIMU, serial_number="SN_typo").open()
    cam = HarvesterCamera(cti_file=TLSIMU, serial_number=MONO)
    cam.open()
    try:
        assert cam.get_image(timeout=3.0).shape == (512, 512)
    finally:
        cam.close()


def test_a_closed_camera_refuses_and_reopens(mono):
    mono.close()
    assert mono.node_map is None  # reading a node here used to segfault
    with pytest.raises(RuntimeError, match="Camera not opened"):
        mono.get_image(timeout=0.5)
    mono.open()
    assert mono.get_image(timeout=3.0).shape == (512, 512)


def test_a_poll_does_not_block(mono):
    mono.node_map.TriggerMode.value = "On"  # no trigger will come
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        mono.get_image(timeout=0)
    assert time.monotonic() - started < 0.5


def test_stopping_from_another_thread_while_fetching(mono):
    """Unserialised, this segfaulted in most runs of a few hundred stops."""
    mono.start_acquisition()
    errors: list[BaseException] = []
    done = threading.Event()

    def fetch() -> None:
        while not done.is_set():
            try:
                mono.get_image(timeout=0.2)
            except TimeoutError:
                pass
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
                return

    worker = threading.Thread(target=fetch, daemon=True)
    worker.start()
    for _ in range(4):  # TLSimu takes about half a second per stop
        time.sleep(0.01)
        mono.stop_acquisition()
    done.set()
    worker.join(5.0)
    assert errors == []


def test_a_colour_frame_becomes_one_plane():
    cam = HarvesterCamera(cti_file=TLSIMU, serial_number=COLOUR)
    cam.open()
    try:
        img = cam.get_image(timeout=3.0)
        assert img.ndim == 2 and img.dtype.name == "uint8"
        assert cam.bit_depth == 8
    finally:
        cam.close()
