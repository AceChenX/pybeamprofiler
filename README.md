# pyBeamprofiler

Real-time laser beam profiler with Gaussian fitting for GenICam cameras.

[![Python 3.9+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

* **Fast Gaussian fitting:** 850+ fps for 1D, 95 fps for 2D
* **Live streaming:** Dash web interface (10 Hz) or Jupyter notebooks (6-10 Hz)
* **Multiple fitting methods:** 1D projections, 2D Gaussian, linecut
* **Multiple width definitions:** Gaussian (1/e²), FWHM, D4σ (ISO 11146)
* **Interactive controls:** Jupyter widgets for camera settings
* **Flexible inputs:** Static images or camera streams
* **Hardware support:** FLIR, Basler cameras via GenICam/[Harvesters](https://github.com/genicam/harvesters)
* **Simulated camera:** Included for testing without hardware
* **Auto-configuration:** Pixel size detection, auto-exposure/gain disabled by default
* **ROI support:** Region of Interest with full sensor default

## Quick Start

```bash
# Install (creates/updates a reproducible environment from the lockfile)
uv sync

# Run with simulated camera (no hardware needed)
uv run python -m pybeamprofiler.beamprofiler --camera simulated

# Browser opens automatically at http://127.0.0.1:8050
```

## Installation

### Basic Installation

```bash
uv sync
```

`uv sync` installs the exact versions recorded in `uv.lock`, giving you a reproducible environment. This is the recommended path for both users and CI.

> **Installing into an existing environment?** If you are managing your own virtualenv or conda environment and do not want to use the lockfile, you can run `uv pip install .` (or plain `pip install .`) instead. Be aware that this resolves dependencies independently and may produce a different set of package versions than the lockfile.

### Development Installation

```bash
uv sync --extra dev
pre-commit install  # Optional: enable git hooks
```

### Optional Dependencies

```bash
uv sync --extra matplotlib  # Matplotlib fallback for CLI
uv sync --extra test        # Testing tools only
uv sync --extra dev         # All development tools
```

## Usage

### Command Line Interface

```bash
# Simulated camera (no hardware needed)
python -m pybeamprofiler.beamprofiler --camera simulated

# Real cameras (requires SDK installation)
python -m pybeamprofiler.beamprofiler --camera flir    # FLIR/Spinnaker
python -m pybeamprofiler.beamprofiler --camera basler  # Basler/Pylon

# Single shot acquisition
python -m pybeamprofiler.beamprofiler --num-img 1

# Static image analysis
python -m pybeamprofiler.beamprofiler --file beam.png

# Custom fitting and definitions
python -m pybeamprofiler.beamprofiler --fit 2d --definition fwhm

# Set exposure time (in seconds)
python -m pybeamprofiler.beamprofiler --exposure-time 0.05

# Fast display mode (heatmap only)
python -m pybeamprofiler.beamprofiler --heatmap-only

# See all options
python -m pybeamprofiler.beamprofiler --help
```

**Continuous streaming** automatically opens a browser with live beam profile and fitting results at http://127.0.0.1:8050.

### Python API

#### Basic Usage

```python
from pybeamprofiler import BeamProfiler

# Initialize with simulated camera
bp = BeamProfiler(camera="simulated")
bp.plot()  # Opens Dash web interface

# For real hardware
bp = BeamProfiler(camera="flir")      # or camera="basler"
bp.plot()
```

#### Single Measurement

```python
from pybeamprofiler import BeamProfiler

bp = BeamProfiler(camera="simulated")
bp.plot(num_img=1)

# Access results
print(f"Beam width: {bp.width:.1f} μm")
print(f"Width X: {bp.width_x:.1f} μm, Y: {bp.width_y:.1f} μm")
print(f"Center: ({bp.center_x:.1f}, {bp.center_y:.1f}) pixels")
print(f"Peak intensity: {bp.peak_value:.0f}")
```

#### Static Image Analysis

```python
from pybeamprofiler import BeamProfiler

bp = BeamProfiler(file="beam_image.png")
bp.plot(num_img=1)

print(f"Width X: {bp.width_x:.1f} μm")
print(f"Width Y: {bp.width_y:.1f} μm")
```

#### Camera Control

```python
# List available cameras
from pybeamprofiler import print_camera_info
print_camera_info()

# Set exposure time (three ways)
bp = BeamProfiler(camera="simulated", exposure_time=0.05)  # 1. During init

bp.exposure_time = 0.01  # 2. Direct attribute

bp.setting(exposure_time=0.05)  # 3. Via setting() method

# Set multiple parameters
bp.setting(
    exposure_time=0.025,
    Gain=10.0,
    ExposureAuto=False,
    GammaEnable=False
)

# Interactive widget (Jupyter only)
bp.setting()  # Opens interactive control panel
```

#### Fitting Methods

```python
# 1D Gaussian fitting (fastest, 850+ fps)
bp.fit_method = '1d'
bp.plot(num_img=1)

# 2D Gaussian fitting with rotation
bp.fit_method = '2d'
bp.plot(num_img=1)
print(f"Rotation angle: {bp.angle_deg:.1f}°")

# Linecut through peak
bp.fit_method = 'linecut'
bp.plot(num_img=1)
```

#### Width Definitions

```python
# Gaussian (1/e²) - standard laser beam width
bp.definition = 'gaussian'
bp.plot(num_img=1)

# Full Width at Half Maximum
bp.definition = 'fwhm'
bp.plot(num_img=1)

# D4σ (ISO 11146 second moment)
bp.definition = 'd4s'
bp.plot(num_img=1)

# Access all width metrics
print(f"1/e² width: {bp.fw_1e2_x:.1f} μm")
print(f"FWHM: {bp.fwhm_x:.1f} μm")
```

#### Region of Interest (ROI)

```python
# Get ROI info
roi = bp.camera.roi_info
print(f"ROI: {roi['width']}×{roi['height']} at ({roi['offset_x']}, {roi['offset_y']})")

# Set ROI (GenICam cameras only)
bp.camera.set_roi(offset_x=100, offset_y=100, width=800, height=600)

# Reset to full sensor
bp.camera.set_roi(offset_x=0, offset_y=0, width=None, height=None)
```

## Jupyter Notebook

See [pybeamprofiler_demo.ipynb](pybeamprofiler_demo.ipynb) for interactive examples with:
- Camera discovery and initialization
- Interactive controls with widgets
- Single-shot and continuous acquisition
- Multiple fitting methods and width definitions
- Programmatic camera control

## Hardware Setup

### FLIR Cameras ([Spinnaker SDK](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/))

**macOS/Linux:**
```bash
# Install Spinnaker SDK from FLIR website
# Then set environment variable
export GENICAM_GENTL64_PATH=/usr/local/lib/spinnaker-gentl
```

**Windows:**
```cmd
set GENICAM_GENTL64_PATH=C:\Program Files\FLIR Systems\Spinnaker\cti64\vs2015
```

### Basler Cameras ([Pylon SDK](https://www.baslerweb.com/en-us/software/pylon/sdk/))

**macOS:**
```bash
export GENICAM_GENTL64_PATH=/Library/Frameworks/pylon.framework/Libraries/gentlproducer/gtl
```

**Linux:**
```bash
export GENICAM_GENTL64_PATH=/opt/pylon/lib64/gentlproducer/gtl
```

**Windows:**
```cmd
set GENICAM_GENTL64_PATH=C:\Program Files\Basler\pylon\Runtime\x64
```

**Note:** GenICam cameras can only be accessed by one application at a time. Close other camera software before using pyBeamprofiler.

## Supported Cameras

### FLIR
- Blackfly S (BFS-PGE, BFS-U3, etc.)
- Grasshopper3 (GS3)
- Auto-detected sensors: Sony IMX273, IMX174, IMX183, IMX250, IMX252, etc.

### Basler
- ace (acA series) - USB3, GigE
- ace 2 (a2A series)
- Auto-detected sensors: Sony IMX253, IMX226, IMX249, IMX255, etc.

### Pixel Size Auto-Detection
Automatic pixel size detection for 40+ sensor models including:
- Sony IMX series (IMX174, IMX183, IMX226, IMX249, IMX250, IMX252, IMX253, IMX255, IMX264, IMX265, IMX273, IMX287, IMX290, IMX291, IMX304, IMX392, IMX412, IMX477, IMX485, IMX530, IMX531, IMX540, IMX541, IMX542, IMX547)
- Direct Basler model lookups (acA4024-8gm, acA4024-29um, acA1920-155um, acA2440-75um, acA3800-14um)

## Performance

- **Gaussian fitting:** 850+ fps (1D), 95 fps (2D) - Not the bottleneck!
- **Display rates:**
  - Jupyter notebook: 6-10 Hz (standard), 25-30 Hz (heatmap only)
  - Dash web interface: 10 Hz
  - Matplotlib fallback: ~5 Hz

## Dependencies

**Core:**
- numpy, scipy - Numerical computing and optimization
- plotly, dash - Interactive visualization and web interface
- ipywidgets - Jupyter notebook controls
- Pillow - Image file loading
- harvesters - GenICam camera interface

**Optional:**
- matplotlib - Fallback plotting (CLI only)

**Development:**
- pytest, pytest-cov - Testing framework
- ruff - Fast linter and formatter
- ty - Static type checking
- pre-commit - Git hooks for code quality

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=pybeamprofiler --cov-report=html

# Run specific test file
uv run pytest tests/test_fitting.py

# Run with verbose output
uv run pytest -v
```

## Development

```bash
# Install in development mode
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run linter
uv run ruff check src tests

# Run formatter
uv run ruff format src tests

# Run type checker
uv run ty check src tests
```

## Troubleshooting

### Camera Not Found

1. **Check SDK installation:**
   - FLIR: Install [Spinnaker SDK](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/)
   - Basler: Install [Pylon SDK](https://www.baslerweb.com/en-us/software/pylon/sdk/)

2. **Set GENICAM_GENTL64_PATH:**
   ```bash
   export GENICAM_GENTL64_PATH=/path/to/cti/files
   ```

3. **Check camera connection:**
   ```python
   from pybeamprofiler import print_camera_info
   print_camera_info()  # Lists all detected cameras
   ```

4. **Access denied error:**
   - Close other camera software (Spinnaker GUI, Pylon Viewer, etc.)
   - GenICam cameras allow only one connection at a time

### GigE vs USB3
- Basler cameras: Code auto-detects and prefers GigE over USB3
- For USB3 cameras, explicitly pass the USB3 CTI file path:
  ```python
  from pybeamprofiler.basler import BaslerCamera
  cam = BaslerCamera(cti_file="/path/to/ProducerU3V.cti")
  ```

### Jupyter Kernel Restart
When re-initializing cameras in Jupyter, restart the kernel first:
- Kernel → Restart Kernel
- This releases the camera hardware lock

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

C.-A. Chen (acechen@cirx.org)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Run code quality checks: `ruff check` and `ruff format`
6. Submit a pull request

## Acknowledgments

- Built on [Harvesters](https://github.com/genicam/harvesters) for GenICam camera interface
- Uses [Plotly/Dash](https://plotly.com/dash/) for interactive visualization
- Inspired by various beam profiling tools: LaseView (old freeware version), [ptomato/Beams](https://github.com/ptomato/Beams), [jordens/bullseye](https://github.com/jordens/bullseye)
- FLIR and Basler cameras loaned from [Atom Computing](https://atom-computing.com/) for testing
