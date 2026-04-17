"""Constants for pybeamprofiler."""

# Camera defaults
DEFAULT_EXPOSURE_TIME = 0.01  # seconds
DEFAULT_GAIN = 0.0
DEFAULT_PIXEL_SIZE = 1.0  # micrometers

# Simulated camera parameters
SIMULATED_WIDTH = 1024  # pixels
SIMULATED_HEIGHT = 1024  # pixels
SIMULATED_PIXEL_SIZE = 5.0  # micrometers
SIMULATED_SIGMA_X = 50  # pixels, ~1 mm 1/e² width at 5 µm/px (typical lab laser)
SIMULATED_SIGMA_Y = 45  # pixels, slightly elliptical (~0.9 mm)
SIMULATED_AMPLITUDE = 200  # intensity units (below 255 to avoid saturation)
SIMULATED_BACKGROUND = 10  # baseline intensity

# Fitting parameters
GAUSSIAN_SIGMA_ESTIMATE_FACTOR = 0.1  # Initial guess: 10% of sensor width
MAX_FIT_ITERATIONS = 100
FIT_CONVERGENCE_TOLERANCE = 1e-6

# Conversion factors
GAUSSIAN_TO_FWHM = 2.355  # FWHM = 2.355 * sigma
D4SIGMA_FACTOR = 4.0  # D4σ = 4 * sigma

# Web interface
DEFAULT_DASH_PORT = 8050
DEFAULT_UPDATE_INTERVAL_MS = 100  # milliseconds

# Display
MAX_DISPLAY_DIM = 1024
