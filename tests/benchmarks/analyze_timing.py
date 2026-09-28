"""How long ``BeamProfiler.analyze`` and the figure builds take per frame.

Not a test: pytest collects only ``test_*.py``, and timings mean something
only on a quiet machine, not on a shared CI runner. From the repository root::

    uv run python tests/benchmarks/analyze_timing.py
    uv run python tests/benchmarks/analyze_timing.py --frames 200

and to see where the time goes::

    uv run python -m cProfile -s cumulative tests/benchmarks/analyze_timing.py | head -60

Frames come from both simulator profiles (``sim-1``: 1024x1024; ``sim-2``:
1280x1024 with a tilted beam), made before any timing starts so the
simulator's own cost is left out. Each mode first analyses a few frames
untimed, which fills the fit's warm start the way a running stream does.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

from pybeamprofiler import BeamProfiler, SimulatedCamera
from pybeamprofiler.dash_app import build_figure
from pybeamprofiler.simulated import profile_for

FITS = ("1d", "linecut", "2d")
DEFINITIONS = ("gaussian", "fwhm", "d4s")


def _ms_per_call(
    fn: Callable[[np.ndarray], object], frames: list[np.ndarray], warmup: int
) -> float:
    for img in frames[:warmup]:
        fn(img)
    timed = frames[warmup:]
    start = time.perf_counter()
    for img in timed:
        fn(img)
    return (time.perf_counter() - start) / len(timed) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=60, help="frames per mode (default 60)")
    parser.add_argument("--warmup", type=int, default=5, help="untimed frames first (default 5)")
    parser.add_argument("--seed", type=int, default=5, help="simulator seed (default 5)")
    args = parser.parse_args()
    if args.frames <= args.warmup:
        parser.error("--frames must be larger than --warmup")

    for key in ("sim-1", "sim-2"):
        camera = SimulatedCamera(profile_for(key), seed=args.seed)
        frames = [camera.get_image() for _ in range(args.frames)]
        h, w = frames[0].shape
        print(f"\n{key} ({w}x{h}), analyze() per frame:")
        print(f"  {'fit':8} " + "".join(f"{d:>11}" for d in DEFINITIONS))
        for fit in FITS:
            cells = []
            for definition in DEFINITIONS:
                bp = BeamProfiler(
                    camera=SimulatedCamera(profile_for(key)), fit=fit, definition=definition
                )
                cells.append(_ms_per_call(bp.analyze, frames, args.warmup))
                bp._release_camera()
            print(f"  {fit:8} " + "".join(f"{ms:8.2f} ms" for ms in cells))

        # Each figure is drawn right after its own frame is analysed, as the
        # GUI and the notebook stream do: the builders reuse the projections
        # analyze() cached, which belong to the last frame analysed.
        bp = BeamProfiler(camera=SimulatedCamera(profile_for(key)))
        builds: dict[str, Callable[..., object]] = {
            "GUI figure (build_figure)": lambda img, px, py: build_figure(bp, img, px, py),
            "GUI figure + to_json()": lambda img, px, py: build_figure(bp, img, px, py).to_json(),
            "notebook figure": bp._create_figure,
            "notebook heatmap-only figure": bp._create_fast_figure,
        }
        print(f"{key}, figure per frame (analysis not included):")
        for name, build in builds.items():
            spent = 0.0
            for i, img in enumerate(frames):
                popt_x, popt_y = bp.analyze(img)
                start = time.perf_counter()
                build(img, popt_x, popt_y)
                if i >= args.warmup:
                    spent += time.perf_counter() - start
            ms = spent / (len(frames) - args.warmup) * 1e3
            print(f"  {name:30} {ms:8.2f} ms")
        bp._release_camera()


if __name__ == "__main__":
    main()
