"""Tunable constants and physical conversion factors."""

import math

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
MAX_FIT_ITERATIONS = 100
# Longest edge a 2D fit is allowed to see. curve_fit over a megapixel grid
# takes hundreds of ms; 256² keeps it interactive with no measurable loss of
# accuracy for beams that span more than a handful of pixels.
MAX_FIT_2D_DIM = 256

# Width conversion factors, all relative to the Gaussian sigma of an
# *intensity* profile I(x) = A·exp(-(x-x₀)²/2σ²).  Solving for the full
# width at each threshold:
#
#   I/A = 1/2   → 2·√(2·ln2)·σ ≈ 2.3548σ   (FWHM)
#   I/A = 1/e   → 2·√2·σ       ≈ 2.8284σ
#   I/A = 1/e²  → 4σ                        (also the D4σ width of a Gaussian)
#
# Note the ordering: FWHM < FW@1/e < FW@1/e², because a lower threshold is
# crossed further out on the flanks.
GAUSSIAN_TO_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
FW_1E_FACTOR = 2.0 * math.sqrt(2.0)
D4SIGMA_FACTOR = 4.0

# Web interface
DEFAULT_DASH_PORT = 8050
DEFAULT_UPDATE_INTERVAL_MS = 50  # milliseconds

# Maximum frames in the rolling-average buffer. Caps memory at N x frame
# size, and bounds the input the GUI offers.
MAX_AVG_FRAMES = 32

# Display
MAX_DISPLAY_DIM = 1024
