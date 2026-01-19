# pyBeamprofiler

Real-time laser beam profiler using GenICam cameras with Gaussian fitting.

## Features

* **Fast Gaussian fitting:** 850+ fps for 1D, 95 fps for 2D
* **Live streaming:** Dash web interface (10 Hz) or Jupyter notebooks (6-10 Hz)
* **Multiple fitting methods:** 1D projections, 2D Gaussian, linecut
* **Multiple width definitions:** Gaussian (1/e²), FWHM, D4σ
* **Interactive controls:** Jupyter widgets for camera settings
* **Flexible inputs:** Static images or camera streams
* **Hardware support:** FLIR, Basler cameras via GenICam/[Harvesters](https://github.com/genicam/harvesters)
* **Simulated camera:** Included for testing without hardware
* **Auto browser launch:** Opens browser automatically for Dash streaming

## Quick Start

```bash
# Install
pip install .

# Test with simulated camera
python src/pybeamprofiler/beamprofiler.py --camera simulated

# Browser opens automatically at http://127.0.0.1:8050
```

## Installation

### Basic Installation

```bash
pip install .
```

### With Optional Dependencies

```bash
pip install .[matplotlib]  # For matplotlib fallback
pip install .[dev]         # For development (includes ruff, pyright, pre-commit)
pip install .[test]        # For testing only
```

### Dependencies

Core (automatically installed):
- `numpy`, `scipy` - Numerical computing
- `plotly`, `dash` - Interactive visualization and web interface
- `ipywidgets` - Jupyter widgets for camera controls
- `Pillow` - Image loading
- `harvesters` - GenICam camera interface

Optional:
- `matplotlib` - Fallback plotting when Dash unavailable

Development tools:
- `ruff` - Fast Python linter and formatter
- `pyright` - Static type checker
- `pre-commit` - Git hooks for code quality

## Usage

### Command Line Interface

```bash
# Simulated camera (no hardware needed)
python src/pybeamprofiler/beamprofiler.py --camera simulated

# Real cameras
python src/pybeamprofiler/beamprofiler.py --camera flir
python src/pybeamprofiler/beamprofiler.py --camera basler

# Single shot acquisition
python src/pybeamprofiler/beamprofiler.py --camera simulated --num-img 1

# Static image analysis
python src/pybeamprofiler/beamprofiler.py --file beam.png

# Custom fitting and definitions
python src/pybeamprofiler/beamprofiler.py --fit 2d --definition fwhm

# Faster display (heatmap only)
python src/pybeamprofiler/beamprofiler.py --heatmap-only

# See all options
python src/pybeamprofiler/beamprofiler.py --help
```

**Continuous streaming** automatically opens browser at http://127.0.0.1:8050 with live beam profile and fitting.

**Command-line options:**
- `--camera`: `simulated`, `flir`, or `basler`
- `--file`: Path to image file (overrides camera)
- `--fit`: `1d` (fastest), `2d` (with rotation), or `linecut`
- `--definition`: `gaussian` (1/e²), `fwhm`, or `d4s`
- `--exposure-time`: Exposure in seconds (default: 0.01)
- `--num-img`: Number of images (1 for single shot, None for continuous)
- `--heatmap-only`: Show only heatmap for faster updates
- `--verbose`: Enable detailed logging

### Python API

#### Quick Start

```python
from pybeamprofiler import BeamProfiler

# Simulated camera
bp = BeamProfiler(camera="simulated")
bp.plot()  # Opens Dash web interface

# Single measurement
bp = BeamProfiler(camera="simulated")
bp.plot(num_img=1)
print(f"Beam width: {bp.width:.1f} μm")
```

#### Static Image Analysis

```python
from pybeamprofiler import BeamProfiler

bp = BeamProfiler(file="beam_image.png")
bp.plot()

# Access results
print(f"Width X: {bp.width_x:.1f} μm")
print(f"Width Y: {bp.width_y:.1f} μm")
print(f"Center: ({bp.center_x:.1f}, {bp.center_y:.1f})")
```

#### Camera Control

```python
# List available cameras
from pybeamprofiler import print_camera_info
print_camera_info()

# Real cameras
bp = BeamProfiler(camera="flir")  # or "basler"
bp.plot()

# Select specific camera by serial number
bp = BeamProfiler(camera="flir", serial_number="12345678")
```

#### Fitting Options

```python
# Different fitting methods
bp = BeamProfiler(camera="simulated", fit="1d")      # Fastest
bp = BeamProfiler(camera="simulated", fit="2d")      # With rotation
bp = BeamProfiler(camera="simulated", fit="linecut") # Peak linecut

# Different width definitions
bp = BeamProfiler(camera="simulated", definition="gaussian")  # 1/e²
bp = BeamProfiler(camera="simulated", definition="fwhm")      # Full Width Half Max
bp = BeamProfiler(camera="simulated", definition="d4s")       # D4σ (ISO 11146)
```

#### Result Properties

```python
# Beam parameters
bp.width          # Average width (μm)
bp.width_x        # X width (μm)
bp.width_y        # Y width (μm)
bp.center_x       # X center (pixels)
bp.center_y       # Y center (pixels)
bp.angle_deg      # Rotation angle (degrees, 2D fit only)
bp.peak_value     # Peak intensity

# Alternative width definitions
bp.fwhm_x         # FWHM in X (μm)
bp.fwhm_y         # FWHM in Y (μm)
bp.diameter       # Beam diameter (μm)
bp.radius         # Beam radius (μm)
```

### Jupyter Notebook

```python
from pybeamprofiler import BeamProfiler

# Initialize with simulated camera
bp = BeamProfiler(camera="simulated")

# Interactive camera controls
bp.setting()  # Shows Jupyter widgets for exposure, gain, etc.

# Live streaming in notebook
bp.plot()  # Standard: Full display (~6-10 Hz)
bp.plot(heatmap_only=True)  # Fast: Heatmap only (~8-12 Hz)
```

**Stopping continuous acquisition:** Press Jupyter's interrupt button or Kernel → Interrupt

## Hardware Setup

### Quick Setup (Recommended)

Set the `GENICAM_GENTL64_PATH` environment variable and cameras will be auto-discovered:

**Windows:**
```bash
# FLIR
set GENICAM_GENTL64_PATH=C:\Program Files\FLIR Systems\Spinnaker\cti64\vs2015

# Basler
set GENICAM_GENTL64_PATH=C:\Program Files\Basler\pylon 6\Runtime\x64
```

**Linux/macOS:**
```bash
# FLIR
export GENICAM_GENTL64_PATH=/opt/spinnaker/lib/flir-gentl

# Basler
export GENICAM_GENTL64_PATH=/opt/pylon/lib64/gentlproducer/gtl

# Multiple cameras (use : on Unix, ; on Windows)
export GENICAM_GENTL64_PATH=/opt/spinnaker/lib/flir-gentl:/opt/pylon/lib64/gentlproducer/gtl
```

### FLIR Cameras

1. **Install [Spinnaker SDK](https://www.flir.com/products/spinnaker-sdk/)**
   - Windows: `C:\Program Files\FLIR Systems\Spinnaker\`
   - Linux: `/opt/spinnaker/`
   - macOS: `/Library/Application Support/FLIR/`

2. **Verify:**
   ```python
   from pybeamprofiler import print_camera_info
   print_camera_info()
   ```

3. **Use:**
   ```python
   bp = BeamProfiler(camera="flir")
   ```

### Basler Cameras

1. **Install [Pylon SDK](https://www.baslerweb.com/en/downloads/software-downloads/)**
   - Windows: `C:\Program Files\Basler\pylon 6\`
   - Linux: `/opt/pylon/`
   - macOS: `/Library/Frameworks/pylon.framework/`

2. **Verify:**
   ```python
   from pybeamprofiler import print_camera_info
   print_camera_info()
   ```

3. **Use:**
   ```python
   bp = BeamProfiler(camera="basler")
   ```

### Common Issues

| Issue | Solution |
|-------|----------|
| "No cameras found" | Check camera connection and power |
| "CTI file not found" | Install SDK (Spinnaker or Pylon) |
| Permission denied (Linux) | Add user to video group: `sudo usermod -a -G video $USER` |
| Multiple cameras | Use `serial_number` parameter |

### Manual CTI Path (Advanced)

If auto-discovery fails, specify CTI file manually:

```python
from pybeamprofiler import FlirCamera, BaslerCamera

# FLIR
camera = FlirCamera(cti_file="/path/to/Spinnaker_GenTL.cti")
bp = BeamProfiler(camera=camera)

# Basler USB3
camera = BaslerCamera(cti_file="/path/to/ProducerU3V.cti")
bp = BeamProfiler(camera=camera)

# Basler GigE
camera = BaslerCamera(cti_file="/path/to/ProducerGEV.cti")
bp = BeamProfiler(camera=camera)
```

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install
```

### Code Quality Tools

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests --fix

# Type checking
pyright src tests

# Run all pre-commit hooks
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_camera.py

# Run with verbose output
pytest -vv
```

## Examples

### Basic Measurement
```python
from pybeamprofiler import BeamProfiler

# Quick measurement
bp = BeamProfiler(camera="simulated")
bp.plot(num_img=1)
print(f"Beam diameter: {bp.diameter:.1f} μm")
```

### Continuous Monitoring
```python
# Web-based streaming
bp = BeamProfiler(camera="flir")
bp.plot()  # Browser opens automatically
```

### Analysis with Different Definitions
```python
# Compare width definitions
for definition in ["gaussian", "fwhm", "d4s"]:
    bp = BeamProfiler(camera="simulated", definition=definition)
    bp.plot(num_img=1)
    print(f"{definition}: {bp.width:.1f} μm")
```

### 2D Fitting for Elliptical Beams
```python
# Get rotation angle
bp = BeamProfiler(camera="simulated", fit="2d")
bp.plot(num_img=1)
print(f"Width: {bp.width_x:.1f} x {bp.width_y:.1f} μm")
print(f"Rotation: {bp.angle_deg:.1f}°")
```

## License

MIT License - see [LICENSE](/LICENSE) file for details.
