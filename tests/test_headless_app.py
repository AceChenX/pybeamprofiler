"""End-to-end tests that drive a real Dash server over HTTP.

No browser involved. Dash's callback endpoint takes a JSON body naming the
callback and its inputs, and all of that can be reconstructed from
``app.callback_map`` — so a plain ``requests`` session is enough to act like
one.

This is the only layer that exercises the app as it actually runs: serving,
routing, callback dispatch and JSON serialisation of the figure. The rest of
the Dash suite calls callback functions directly, which cannot catch a
callback that is registered wrongly, an output id that does not exist, or a
figure that fails to serialise.

Timing is deliberately not asserted on — CI machines are too noisy for that.
Cost regressions are guarded in ``test_regressions.py`` against the analysis
step alone, which is measurable without a server in the loop.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import pytest
import requests

from pybeamprofiler import dash_app
from pybeamprofiler.beamprofiler import BeamProfiler

SERVER_START_TIMEOUT = 30.0
REQUEST_TIMEOUT = 60.0


def _free_port() -> int:
    """Ask the OS for a port nothing else is using."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class DashClient:
    """A running Dash app plus the means to poke it like a browser would."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.session = requests.Session()

    def start(self) -> DashClient:
        threading.Thread(
            target=lambda: self.app.run(
                debug=False, port=self.port, use_reloader=False, threaded=True
            ),
            daemon=True,
        ).start()

        deadline = time.monotonic() + SERVER_START_TIMEOUT
        while time.monotonic() < deadline:
            try:
                if self.session.get(self.base, timeout=2).status_code == 200:
                    return self
            except requests.RequestException:
                time.sleep(0.1)
        raise RuntimeError("Dash server did not start")

    def _lookup(self, output_id: str, input_id: str) -> tuple[str, Any]:
        for key, spec in self.app.callback_map.items():
            if output_id in key and any(i["id"] == input_id for i in spec["inputs"]):
                return key, spec
        raise KeyError(f"no callback writes {output_id} from {input_id}")

    def fire(
        self,
        output_id: str,
        input_id: str,
        value: Any,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke one callback over HTTP and return the parsed response."""
        key, spec = self._lookup(output_id, input_id)
        raw = spec["output"]
        outs = [
            {"id": o.component_id, "property": o.component_property}
            for o in (raw if isinstance(raw, list) else [raw])
        ]
        body = {
            "output": key,
            "outputs": outs if len(outs) > 1 else outs[0],
            "inputs": [
                {
                    "id": i["id"],
                    "property": i["property"],
                    "value": value if i["id"] == input_id else None,
                }
                for i in spec["inputs"]
            ],
            "changedPropIds": [f"{input_id}.{spec['inputs'][0]['property']}"],
            "state": [
                {"id": s["id"], "property": s["property"], "value": (state or {}).get(s["id"])}
                for s in spec["state"]
            ],
        }
        response = self.session.post(
            f"{self.base}/_dash-update-component", json=body, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()


def _tick_state(analysis: str = "1d") -> dict[str, Any]:
    """The State bundle the render callback reads on every interval tick."""
    return {
        "store-paused": False,
        "switch-color": True,
        "dropdown-colorscale": "Hot",
        "switch-autorange": True,
        "input-zmin": None,
        "input-zmax": None,
        "store-frame": 0,
        "dropdown-analysis": analysis,
        "dropdown-definition": "gaussian",
        "store-dark-theme": True,
        "input-avg-n": 1,
    }


@pytest.fixture(scope="module")
def live_app():
    """One server for the whole module; starting it is the slow part."""
    bp = BeamProfiler(camera="simulated")
    client = DashClient(dash_app.create_app(bp)).start()
    yield client, bp
    if bp.camera is not None:
        bp.camera.close()


class TestServing:
    def test_index_page_is_served(self, live_app):
        client, _ = live_app
        page = client.session.get(client.base, timeout=REQUEST_TIMEOUT)
        assert page.status_code == 200
        assert "pyBeamprofiler" in page.text

    def test_layout_endpoint_is_valid(self, live_app):
        client, _ = live_app
        layout = client.session.get(f"{client.base}/_dash-layout", timeout=REQUEST_TIMEOUT)
        assert layout.status_code == 200

    def test_dependencies_endpoint_lists_the_callbacks(self, live_app):
        client, _ = live_app
        deps = client.session.get(
            f"{client.base}/_dash-dependencies", timeout=REQUEST_TIMEOUT
        ).json()
        outputs = " ".join(d["output"] for d in deps)
        assert "live-graph.figure" in outputs
        assert "dropdown-camera" in outputs


class TestRenderTick:
    """Every analysis mode must survive a full round trip, including the JSON
    serialisation of the figure — which is where a bad trace shows up."""

    @pytest.mark.parametrize("analysis", ["1d", "2d", "linecut"])
    def test_a_tick_returns_a_figure(self, live_app, analysis):
        client, _ = live_app
        body = client.fire("live-graph.figure", "interval", 1, _tick_state(analysis))
        figure = body["response"]["live-graph"]["figure"]
        assert figure["data"], f"{analysis} produced no traces"
        assert figure["layout"]["xaxis"]["range"]

    @pytest.mark.parametrize("analysis", ["1d", "2d", "linecut"])
    def test_results_panel_is_populated(self, live_app, analysis):
        client, _ = live_app
        body = client.fire("live-graph.figure", "interval", 2, _tick_state(analysis))
        assert body["response"]["div-results"]["children"]

    def test_switching_to_2d_keeps_serving_frames(self, live_app):
        """The regression this file was written for: 2D fitting used to cost
        more per frame than the interval, and the display stopped keeping up.
        """
        client, _ = live_app
        for n in range(3):
            client.fire("live-graph.figure", "interval", n, _tick_state("1d"))
        for n in range(6):
            body = client.fire("live-graph.figure", "interval", 10 + n, _tick_state("2d"))
            assert body["response"]["live-graph"]["figure"]["data"]

    def test_a_paused_stream_does_not_redraw(self, live_app):
        client, _ = live_app
        state = _tick_state()
        state["store-paused"] = True
        body = client.fire("live-graph.figure", "interval", 3, state)
        assert body["response"] == {} or "live-graph" not in body.get("response", {})


class TestControls:
    def test_play_pause_round_trips(self, live_app):
        client, bp = live_app
        body = client.fire("store-paused.data", "btn-play-pause", 1, {"store-paused": False})
        assert body["response"]["store-paused"]["data"] is True

        body = client.fire("store-paused.data", "btn-play-pause", 2, {"store-paused": True})
        assert body["response"]["store-paused"]["data"] is False

    def test_camera_refresh_returns_options(self, live_app):
        client, _ = live_app
        body = client.fire("dropdown-camera.options", "btn-camera-refresh", 1)
        options = body["response"]["dropdown-camera"]["options"]
        assert options
        assert all("label" in o and "value" in o for o in options)

    def test_switching_camera_over_http(self, live_app):
        client, bp = live_app
        from pybeamprofiler import dash_layout, discovery

        options, current = dash_layout._camera_options(bp)
        target = next(o for o in options if o.key != current)

        body = client.fire("div-camera-status.children", "dropdown-camera", target.key)
        assert "ready" in body["response"]["div-camera-status"]["children"]
        assert discovery.describe_open_camera(bp.camera).key == target.key

        # And the stream still renders on the newly attached camera.
        client.fire("store-paused.data", "btn-play-pause", 3, {"store-paused": True})
        tick = client.fire("live-graph.figure", "interval", 20, _tick_state("2d"))
        assert tick["response"]["live-graph"]["figure"]["data"]

    def test_saving_a_frame_returns_a_download(self, live_app):
        client, _ = live_app
        body = client.fire("download-png.data", "btn-save-png", 1)
        payload = body["response"]["download-png"]["data"]
        assert payload["filename"].endswith(".png")
        assert payload["base64"] is True
