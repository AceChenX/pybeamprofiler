"""How far each fit and definition is from the true width, across beams.

Not a test: pytest collects only ``test_*.py``. The tests pin the behaviour
this measures to tolerances; this prints the whole picture, which is what to
look at after changing anything in ``fitting.py``. From the repository root::

    uv run python tests/benchmarks/accuracy_matrix.py
    uv run python tests/benchmarks/accuracy_matrix.py --seed 3

Each frame is a Gaussian beam drawn by the generator below, which is
independent of the package so that a convention shared by the fit and its
reference cannot hide a bug: a pedestal of 10 counts, Gaussian read noise,
rounding and clipping to 8 bits, as a camera delivers. The beam is 1.5 times
wider along its major axis than its minor one, and either upright or tilted
by 35 degrees.

For a Gaussian every definition should report the same thing once converted
to 1/e², namely four times the beam's sigma along the image axis (a tilted
beam's projection is itself a Gaussian, of the sigma ``_axis_sigmas``
gives). A cell is the larger of the X and Y errors in percent, and ``fail``
means the frame was reported as having no beam. Every cell starts cold, as
the first frame of a stream does.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from pybeamprofiler import BeamProfiler, SimulatedCamera

SIGMAS = (6.0, 12.0, 24.0, 50.0, 100.0)  # minor-axis sigma, px
NOISES = (0.0, 2.0, 10.0)  # read noise, counts rms
TILTS = (0.0, 35.0)  # degrees
COMBOS = [(fit, d) for fit in ("1d", "2d") for d in ("gaussian", "fwhm", "d4s")]


def _frame(
    rng: np.random.Generator,
    sx: float,
    sy: float,
    theta_deg: float,
    noise: float,
    *,
    h: int = 1024,
    w: int = 1024,
    cx: float = 480.3,
    cy: float = 530.7,
    amp: float = 200.0,
    bg: float = 10.0,
) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(float)
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    u = (x - cx) * c + (y - cy) * s
    v = -(x - cx) * s + (y - cy) * c
    img = amp * np.exp(-(u * u) / (2 * sx * sx) - (v * v) / (2 * sy * sy)) + bg
    img += rng.normal(0.0, noise, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _axis_sigmas(sx: float, sy: float, theta_deg: float) -> tuple[float, float]:
    """Sigma of the beam's projection onto the image x and y axes."""
    c2 = math.cos(math.radians(theta_deg)) ** 2
    s2 = 1.0 - c2
    return math.sqrt(sx * sx * c2 + sy * sy * s2), math.sqrt(sx * sx * s2 + sy * sy * c2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="noise seed (default 0)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # Widths in pixels, so they compare with the generator's sigmas directly.
    profilers = {
        (fit, d): BeamProfiler(camera=SimulatedCamera(), fit=fit, definition=d, pixel_size=1.0)
        for fit, d in COMBOS
    }
    worst = dict.fromkeys(COMBOS, 0.0)

    labels = "".join(f"{fit + '/' + d:>12}" for fit, d in COMBOS)
    print(f"{'sigma':>5} {'noise':>5} {'tilt':>4} |{labels}")
    for sigma in SIGMAS:
        for noise in NOISES:
            for tilt in TILTS:
                sx, sy = 1.5 * sigma, sigma
                img = _frame(rng, sx, sy, tilt, noise)
                true_x, true_y = _axis_sigmas(sx, sy, tilt)
                cells = []
                for combo, bp in profilers.items():
                    bp.reset_analysis()
                    bp.analyze(img)
                    error = 100 * max(
                        abs(bp.fw_1e2_x / (4 * true_x) - 1), abs(bp.fw_1e2_y / (4 * true_y) - 1)
                    )
                    if math.isfinite(error):
                        worst[combo] = max(worst[combo], error)
                        cells.append(f"{error:11.1f}%")
                    else:
                        worst[combo] = math.inf
                        cells.append(f"{'fail':>12}")
                print(f"{sigma:5.0f} {noise:5.0f} {tilt:4.0f} |" + "".join(cells))

    print("\nWorst error per fit/definition:")
    for (fit, d), error in worst.items():
        print(f"  {fit}/{d:9} {error:6.1f}%")
    for bp in profilers.values():
        bp._release_camera()


if __name__ == "__main__":
    main()
