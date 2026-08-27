# pyBeamprofiler

Real-time laser beam profiler with Gaussian fitting for GenICam cameras.

[![PyPI](https://img.shields.io/pypi/v/pybeamprofiler.svg)](https://pypi.org/project/pybeamprofiler/)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Real-time Gaussian fitting** — 1D projections (230+ fps), 2D with rotation (50+ fps), linecut (300+ fps)
- **Browser-based GUI** — Dash web interface with live heatmap, fits, dark/light theme, 30+ color scales, and camera controls
- **Multi-camera** — discover attached cameras, pick one from the dropdown, press Play; switch devices without restarting
- **Multiple width definitions** — Gaussian (1/e²), FWHM, D4σ (ISO 11146)
- **GenICam hardware support** — FLIR (Spinnaker) and Basler (Pylon) via [Harvesters](https://github.com/genicam/harvesters); auto-discovered features grouped by SFNC category
- **Jupyter support** — Live streaming with `bp.setting()` interactive widgets
- **Simulated camera** — Gaussian beam with noise, no hardware needed
- **Auto-configuration** — Pixel size detection for 40+ sensor models

## Quick Start

```bash
pip install pybeamprofiler

# Run the browser GUI with a simulated camera (no hardware needed)
pybeamprofiler --camera simulated

# Or connect to real hardware
pybeamprofiler --camera flir       # FLIR / Spinnaker
pybeamprofiler --camera basler     # Basler / Pylon
```

The browser opens automatically at http://127.0.0.1:8050. Press **Ctrl+C** in
the terminal to stop the server and exit.

### Switching cameras

The **Camera** dropdown at the top of the Fitting tab lists every camera found
on the machine, plus the built-in simulators. Press ⟳ to rescan after plugging
something in. Selecting a camera opens it and leaves the stream stopped — press
**Play** (or the spacebar) to start acquiring.

Switching releases the previous device, so nothing stays claimed. If the camera
you pick cannot be opened — unplugged, or held by Spinnaker/pylon Viewer — the
current stream keeps running and the reason is shown under the dropdown.

Two simulated cameras are always offered. They have different sensor sizes,
pixel pitches and beam shapes (the second is tilted 35°), so the selector and
the 2D fit can both be exercised with no hardware attached.

## Installation

```bash
pip install pybeamprofiler
```

For real hardware (FLIR or Basler), also install the corresponding SDK and set the CTI path:

| Vendor | SDK | `GENICAM_GENTL64_PATH` |
|--------|-----|------------------------|
| FLIR | [Spinnaker](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/) | `/usr/local/lib/spinnaker-gentl` (macOS/Linux) |
| Basler (macOS) | [Pylon](https://www.baslerweb.com/en-us/software/pylon/sdk/) | `/Library/Frameworks/pylon.framework/Libraries/gentlproducer/gtl` |
| Basler (Linux) | [Pylon](https://www.baslerweb.com/en-us/software/pylon/sdk/) | `/opt/pylon/lib64/gentlproducer/gtl` |
| Basler (Windows) | [Pylon](https://www.baslerweb.com/en-us/software/pylon/sdk/) | `C:\Program Files\Basler\pylon\Runtime\x64` |

```bash
export GENICAM_GENTL64_PATH=/path/to/cti/files
```

> **Note:** GenICam cameras allow only one application to connect at a time — close Spinnaker GUI / Pylon Viewer first.

## Usage

### Command line

```bash
pybeamprofiler --camera simulated               # simulated camera
pybeamprofiler --camera flir                    # FLIR / Spinnaker
pybeamprofiler --camera basler                  # Basler / Pylon
pybeamprofiler --file beam.png --pixel-size 5.86  # static image
pybeamprofiler --fit 2d --definition fwhm
pybeamprofiler --help                           # all options
```

`--pixel-size` is required with `--file`: an image on disk carries no scale, so
without it every width would be in pixels pretending to be micrometers. Cameras
report their own pitch, but you can still pass `--pixel-size` to override it
(useful with binning, or when the camera reports nothing).

Press **Ctrl+C** in the terminal to stop streaming and exit.

### Python API

```python
from pybeamprofiler import BeamProfiler, discover_cameras, print_camera_info

print_camera_info()                    # list connected cameras
discover_cameras()                     # the same list the GUI dropdown shows

bp = BeamProfiler(camera="simulated")
bp = BeamProfiler(camera="flir", serial_number="12345678")   # pick a device
bp.plot()                              # open browser GUI
bp.plot(num_img=1)                     # single shot

print(f"Width X: {bp.width_x:.1f} μm, Y: {bp.width_y:.1f} μm")
print(f"Center: ({bp.center_x:.1f}, {bp.center_y:.1f}) px")

# Change settings
bp.setting(exposure_time=0.05, Gain=10.0)

# In Jupyter: open interactive control panel
bp.setting()
```

See [the API docs](https://github.com/acechenx/pybeamprofiler) for fitting methods (`1d`, `2d`, `linecut`), width definitions (`gaussian`, `fwhm`, `d4s`), ROI control, and more.

### Width definitions

`width_x` / `width_y` are reported in whichever definition you select, and the
derived properties convert between them from the fitted Gaussian sigma:

| Property | Threshold | Multiple of σ |
|----------|-----------|---------------|
| `fwhm_x` / `fwhm_y` | 1/2 | 2·√(2·ln2) ≈ 2.3548 |
| `fw_1e_x` / `fw_1e_y` | 1/e | 2·√2 ≈ 2.8284 |
| `fw_1e2_x` / `fw_1e2_y` | 1/e² | 4 |

A lower threshold is crossed further out on the flanks, so the widths always
order as FWHM < FW@1/e < FW@1/e². `d4s` is the ISO 11146 second-moment width,
which equals 4σ for a Gaussian but — unlike a fit — stays meaningful for
flat-top and multi-lobed beams.

Choosing `fwhm` or `d4s` measures directly off the integrated profile with no
model, so it takes precedence over `fit`; the Gaussian fit still runs, but only
to draw the overlay.

## Troubleshooting

- **Camera not found** — verify SDK install, `GENICAM_GENTL64_PATH`, and run `print_camera_info()`.
- **Access denied** — close other camera software (Spinnaker GUI, Pylon Viewer).
- **Jupyter camera stuck** — restart the kernel to release the hardware lock.
- **Basler USB3** — pass the USB3 CTI explicitly: `BaslerCamera(cti_file="/path/to/ProducerU3V.cti")`.

---

## For Developers

### Setup

```bash
git clone https://github.com/acechenx/pybeamprofiler.git
cd pybeamprofiler
uv sync --extra dev
uv run pre-commit install
```

### Code quality

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run ty check src tests
```

### Testing

```bash
uv run pytest                          # all tests + coverage
uv run pytest --cov-report=html        # HTML coverage report
uv run pytest tests/test_profiler.py   # single file
```

Current suite: **815 tests, 99% coverage**. The suite is hermetic — an
autouse fixture hides any camera SDK installed on the machine, so results
do not depend on what happens to be plugged in.

### Architecture

- `fitting.py` — Gaussian models, direct width measurements, curve fits, decimation (pure functions over arrays; no camera or plotting state)
- `cti.py` — one table of GenTL producer (`.cti`) search paths, shared by every vendor
- `discovery.py` — enumerating cameras and opening the selected one (`utils.py` re-exports it)
- `beamprofiler.py` — `BeamProfiler` class, figure building, streaming, CLI entry point
- `camera.py` — abstract `Camera` base class + Jupyter widget builder
- `gen_camera.py` — `HarvesterCamera` for GenICam devices
- `flir.py` / `basler.py` — vendor-specific subclasses
- `simulated.py` — `SimulatedCamera` and its beam profiles (no hardware)
- `dash_app.py` — browser GUI with live updates, settings panel, pattern-matching callbacks
- `constants.py` — shared constants and conversion factors

### Contributing

1. Fork and create a feature branch
2. Add tests for new functionality
3. Ensure `uv run pytest`, `uv run ruff check`, and `uv run ty check` pass
4. Submit a pull request

### Performance notes

- 2D fits downsample to 256 px on the longest edge and warm-start Levenberg-Marquardt from the previous frame; a failed fit is retried from a fresh estimate rather than left stuck on stale parameters
- Display decimation is nearest-neighbour fancy indexing (~4× faster than interpolated zoom at 1024 px, and it shows real sensor counts rather than blended ones)
- Projection profiles are cached between `analyze()` and figure rendering
- GenICam feature discovery is memoised per node map — the `.value` probe it relies on is a register read per node, which costs hundreds of round trips on a GigE camera
- The saturation check short-circuits on a `max()` reduction, so a clean frame skips the full-frame comparison
- In the current Dash implementation, live updates perform `bp.camera.get_image(timeout=0.1)` and `bp.analyze(img)` synchronously while holding a `threading.Lock` to reduce segfault risk in the Harvesters C library

### Supported sensors

Auto-detected pixel sizes (40+ models) — Sony IMX174, IMX183, IMX226, IMX249, IMX250, IMX252, IMX253, IMX255, IMX264, IMX265, IMX273, IMX287, IMX290, IMX291, IMX304, IMX392, IMX412, IMX477, IMX485, IMX530, IMX531, IMX540, IMX541, IMX542, IMX547, and direct Basler model lookups (acA4024-8gm, acA4024-29um, acA1920-155um, acA2440-75um, acA3800-14um).

---

## License

MIT — see [LICENSE](LICENSE).

## Author

C.-A. Chen (acechen@cirx.org)

## Acknowledgments

- Built on [Harvesters](https://github.com/genicam/harvesters), [Plotly/Dash](https://plotly.com/dash/)
- Inspired by LaseView (old freeware), [ptomato/Beams](https://github.com/ptomato/Beams), [jordens/bullseye](https://github.com/jordens/bullseye)
- FLIR and Basler cameras loaned from [Atom Computing](https://atom-computing.com/) for testing
