# Test Suite Organization

## Overview

The test suite is organized into focused modules for better maintainability and clarity.

## Test Files

### Core Functionality

- **`conftest.py`** - Shared pytest fixtures
  - `simulated_image`: Synthetic Gaussian beam for testing
  - `beam_profiler`: BeamProfiler instance with simulated camera
  - `test_image_file`: Temporary test image file

- **`test_camera.py`** - Camera interfaces and control
  - Simulated camera functionality
  - Exposure and gain control
  - Camera integration with BeamProfiler
  - Hardware camera imports

- **`test_fitting.py`** - Gaussian fitting methods
  - 1D projection fitting
  - 2D Gaussian fitting with rotation
  - Linecut fitting
  - Fit caching and performance
  - Edge cases (empty images, noise)

- **`test_definitions.py`** - Width definitions
  - Gaussian (1/e²) definition
  - FWHM (Full Width at Half Maximum)
  - D4σ (ISO 11146 second moment)
  - Definition comparisons and ratios
  - Switching definitions

- **`test_profiler.py`** - BeamProfiler properties and integration
  - Properties (width, diameter, radius)
  - LaseView-compatible properties
  - Static image loading
  - Initialization options
  - Exposure handling
  - Sensor dimensions

### Hardware Tests

- **`test_genicam.py`** - GenICam camera implementations
  - HarvesterCamera wrapper
  - FLIR camera implementation
  - Basler camera implementation
  - CTI file discovery
  - Camera utilities

## Running Tests

### Setup Environment

**First, activate the conda environment:**
```bash
conda activate pybeamprofiler-env
```

### All tests
```bash
conda activate pybeamprofiler-env
pytest tests/
```

### Specific module
```bash
conda activate pybeamprofiler-env
pytest tests/test_camera.py -v
pytest tests/test_fitting.py -v
pytest tests/test_definitions.py -v
```

### With coverage
```bash
conda activate pybeamprofiler-env
pytest tests/ --cov=pybeamprofiler --cov-report=term-missing
```

### Specific test class or function
```bash
conda activate pybeamprofiler-env
pytest tests/test_fitting.py::TestOneDimensionalFitting -v
pytest tests/test_definitions.py::TestDefinitionComparisons::test_definition_ordering -v
```

## Test Structure Guidelines

- **Class-based organization**: Tests are grouped into classes by functionality
- **Descriptive names**: Test functions clearly state what they test
- **Fixtures**: Shared setup code is in `conftest.py`
- **Isolation**: Each test is independent and cleans up resources
- **Parametrization**: Use `@pytest.mark.parametrize` for testing multiple inputs
- **Mocking**: Hardware tests use mocks to avoid requiring actual cameras

## Migration from Old Structure

### Removed Files
- `test_basic.py` - Merged into `test_camera.py` and `test_profiler.py`
- `test_comprehensive.py` - Split into focused modules

### New Organization Benefits
- **Faster test discovery**: Smaller, focused files
- **Better maintainability**: Clear separation of concerns
- **Easier debugging**: Know exactly where to look for test failures
- **Reusable fixtures**: Centralized test setup
- **Clearer coverage**: See what functionality is tested where
