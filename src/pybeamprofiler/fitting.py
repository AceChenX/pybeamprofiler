"""Numerical core of the profiler: Gaussian models, width measurements, fits.

Everything here is a plain function over numpy arrays — no camera handles, no
plotting state — so the maths can be exercised (and reused) without standing up
a :class:`~pybeamprofiler.beamprofiler.BeamProfiler`.

Two families of width measurement live side by side:

* **Model-based** — fit a Gaussian and read σ off the fit.  Accurate and
  noise-tolerant *if* the beam really is Gaussian.
* **Direct** — :func:`measure_fwhm` and :func:`measure_d4s` on a profile, and
  :func:`measure_fwhm_2d` and :func:`measure_d4s_2d` on a whole frame, read
  the width straight off the data.  They make no assumption about the beam
  shape, which is what ISO 11146 asks for, and pay for it in noise.

Both follow two rules. The background comes from the beam-free edges of the
data by a robust mean, never from its minimum (the notes above
:func:`estimate_background` say why). And data with nothing standing clear of
its own noise has no width: the direct measurements return NaN and the fits
report failure, rather than measuring the noise.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.ndimage import zoom as _ndimage_zoom
from scipy.optimize import curve_fit

from .constants import (
    FIT_2D_WINDOW_SIGMAS,
    GAUSSIAN_TO_FWHM,
    MAX_FIT_2D_DIM,
    MAX_FIT_2D_EVALS,
    MAX_FIT_ITERATIONS,
    MIN_FIT_2D_SIGMA_PX,
    MIN_FIT_2D_WINDOW_PX,
)

logger = logging.getLogger(__name__)

# curve_fit raises TypeError (not RuntimeError) when the profile has fewer
# samples than free parameters, so it belongs in the same net as the ordinary
# "did not converge" failures.
_FIT_ERRORS = (RuntimeError, ValueError, TypeError)

__all__ = [
    "gaussian",
    "gaussian_2d",
    "estimate_background",
    "measure_fwhm",
    "measure_fwhm_2d",
    "measure_d4s",
    "measure_d4s_2d",
    "fit_1d_gaussian",
    "fit_2d_gaussian",
    "downsample",
]


def gaussian(x: np.ndarray, a: float, x0: float, sigma: float, offset: float) -> np.ndarray:
    """1D Gaussian ``a·exp(-(x-x0)²/2σ²) + offset``.

    Args:
        x: Sample positions.
        a: Amplitude above the baseline.
        x0: Center position.
        sigma: Standard deviation.
        offset: Baseline offset.

    Returns:
        Gaussian values at *x*.
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


def gaussian_2d(
    xy: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
) -> np.ndarray:
    """Rotated 2D Gaussian, flattened for :func:`scipy.optimize.curve_fit`.

    The trig terms are hoisted out of the array maths and the exponent is
    built with in-place numpy ops — this runs once per Levenberg-Marquardt
    iteration over the whole grid, so the temporaries add up.

    Args:
        xy: ``(x, y)`` coordinate grids of matching shape.
        amplitude: Peak amplitude above the baseline.
        x0: Center x position.
        y0: Center y position.
        sigma_x: Standard deviation along the first principal axis.
        sigma_y: Standard deviation along the second principal axis.
        theta: Counter-clockwise rotation of the *sigma_x* axis, in
            radians, measured in the array's own (x, y) coordinates.
        offset: Baseline offset.

    Returns:
        Flattened Gaussian values.
    """
    x, y = xy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos2 = cos_t * cos_t
    sin2 = sin_t * sin_t
    sin2t = 2.0 * sin_t * cos_t
    sx2_inv = 0.5 / (sigma_x * sigma_x)
    sy2_inv = 0.5 / (sigma_y * sigma_y)

    a = cos2 * sx2_inv + sin2 * sy2_inv
    # Note the sign: the textbook form of this expression is written for an
    # image drawn with y pointing *down*, and puts the sigma_x axis at -theta
    # in array coordinates. Flipping it makes theta the plain
    # counter-clockwise angle of the sigma_x axis in (x, y), so every consumer
    # -- the ellipse overlay, the reported angle, the simulator -- can use it
    # directly instead of remembering to negate.
    b = 0.5 * sin2t * (sx2_inv - sy2_inv)
    c = sin2 * sx2_inv + cos2 * sy2_inv

    dx = x - float(x0)
    dy = y - float(y0)
    out = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
    np.negative(out, out=out)
    np.exp(out, out=out)
    out *= amplitude
    out += offset
    return out.ravel()


# ── Background and noise ────────────────────────────────────────────────
#
# Every direct measurement below starts by removing the background, and both
# obvious choices are wrong on real data:
#
# * The minimum of a noisy floor sits well *below* the true level -- five
#   noise sigmas down on a megapixel frame, or at zero if the black level
#   clips -- so subtracting it leaves a pedestal under the whole frame.
#   Second moments weight that pedestal by distance squared, and the
#   simulator's 200x180 px beam measured about 1050 px across.
# * The median fixes that, but a camera whose black level is set close to
#   zero clips the bottom of its noise, and clipped noise is asymmetric: at
#   10 +/- 10 counts 17% of background pixels read 0 and the true mean is
#   10.84 against a median of 10.00. Moments are linear in intensity, so they
#   need the *mean*: the median's 0.84 counts, summed over a megapixel, come
#   to a third of the simulator beam's entire signal.
#
# So the level is a mean, taken after rejecting only the bright tail (the part
# a stray bit of beam would contaminate); the dark side is kept, clipped or not.

# Fraction of each edge used as the "beam-free" border ring.
_BORDER_FRACTION = 0.05

# How many border-ring pixels the background estimate looks at: every n-th,
# n = ring size // this, so between this many and twice it (see
# estimate_background).
_BACKGROUND_SAMPLES = 20_000

# How far above the noise a peak must stand to count as a beam at all.
_SIGNIFICANCE = 5.0

# MAD -> sigma for Gaussian noise.
_MAD_TO_SIGMA = 1.4826


def _robust_mean(values: np.ndarray) -> tuple[float, float]:
    """Mean and noise of a mostly-background sample, ignoring bright outliers.

    Returns ``(mean, noise_sigma)``. Noise comes from the median absolute
    deviation, which a few bright pixels cannot move; the mean then excludes
    anything more than five noise sigmas above the median.
    """
    data = values[np.isfinite(values)]
    if data.size == 0:
        return 0.0, 0.0
    median = float(np.median(data))
    noise = float(np.median(np.abs(data - median)) * _MAD_TO_SIGMA)
    keep = data <= median + _SIGNIFICANCE * max(noise, 1e-12)
    return float(data[keep].mean()), noise


def estimate_background(image: np.ndarray) -> tuple[float, float]:
    """Per-pixel background level and noise, from the frame's border ring.

    A beam profiler is aligned so the beam sits away from the edges, which
    makes the outer few percent of the frame the best available sample of the
    ambient level. Bright outliers are rejected before averaging, so a beam
    that does clip one edge still leaves the estimate clean.

    Args:
        image: 2D frame.

    Returns:
        ``(level, noise_sigma)`` in counts per pixel.
    """
    data = np.asarray(image)
    h, w = data.shape
    k = max(1, int(round(_BORDER_FRACTION * min(h, w))))
    if 2 * k >= min(h, w):
        ring = data.ravel()
    else:
        ring = np.concatenate(
            [data[:k].ravel(), data[-k:].ravel(), data[k:-k, :k].ravel(), data[k:-k, -k:].ravel()]
        )
    # The ring of a megapixel frame holds some 200k pixels, and the two
    # medians over all of them were two thirds of the cost of measuring a
    # frame (4 of 6 ms). A regular subsample (22k pixels there) agrees with
    # the full ring to about a tenth of a count at 10 counts of noise, in a
    # seventh of the time.
    step = max(1, ring.size // _BACKGROUND_SAMPLES)
    return _robust_mean(ring[::step].astype(float))


def _profile_ends(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two outer stretches of a profile -- its presumed beam-free part.

    ``None`` for a profile too short to have any: with fewer than five
    samples, the "ends" would be the beam itself.
    """
    data = np.asarray(profile, dtype=float)
    k = max(2, int(round(_BORDER_FRACTION * data.size)))
    if 2 * k >= data.size:
        return None
    return data[:k], data[-k:]


def _profile_noise(profile: np.ndarray) -> float:
    """Robust noise sigma of a 1D profile, measured where the baseline is.

    Uses the median absolute first difference *within each end* of the
    profile. Differencing cancels a sloped background; staying at the ends
    keeps the beam's own slope out of it. (Taking first differences across the
    whole profile read a three-sample-wide beam's edges as noise and then
    declared the beam insignificant.) Differencing two independent samples
    doubles the variance, hence the root-two.

    A profile too short to have beam-free ends reports zero: its noise can't
    be told apart from the beam, so nothing on it is judged insignificant.
    """
    ends = _profile_ends(profile)
    if ends is None:
        return 0.0
    d = np.concatenate([np.diff(end) for end in ends])
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float(np.median(np.abs(d - np.median(d))) * _MAD_TO_SIGMA / np.sqrt(2.0))


def _profile_baseline(profile: np.ndarray) -> float:
    """Baseline of a 1D profile, from its two ends (see :func:`_robust_mean`)."""
    ends = _profile_ends(profile)
    data = np.asarray(profile, dtype=float) if ends is None else np.concatenate(ends)
    return _robust_mean(data)[0]


def _stands_clear(peak: float, noise: float) -> bool:
    """Whether a peak (already baseline-subtracted) is signal rather than noise."""
    return peak > 0 and not (noise > 0 and peak < _SIGNIFICANCE * noise)


def _projections_stand_clear(col: np.ndarray, row: np.ndarray, noise: float) -> bool:
    """Whether a frame holds a beam at all, judged on its two projections.

    *col* is the background-subtracted projection onto x (each sample sums a
    column of ``row.size`` pixels) and *row* the one onto y. The beam has
    been integrated along one axis there, and stands far clearer of the noise
    than in any single pixel; each projection sample's noise is the per-pixel
    noise times the root of the number of pixels it sums.
    """
    return _stands_clear(float(col.max()), noise * np.sqrt(row.size)) and _stands_clear(
        float(row.max()), noise * np.sqrt(col.size)
    )


def _projections(data: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Background-subtracted projections of *data* onto x and onto y.

    Summed in the frame's own dtype and only then made float: numpy widens an
    integer sum to 64 bits, so nothing overflows, and no float copy of the
    whole frame is ever made.
    """
    h, w = data.shape
    col = data.sum(axis=0).astype(float) - level * h
    row = data.sum(axis=1).astype(float) - level * w
    return col, row


def _beam_in_frame(image: np.ndarray) -> bool:
    """Whether a frame holds a beam at all: the verdict D4σ, FWHM and the 2D
    fit reach, for a measurement that only looks at part of the frame."""
    data = np.asarray(image)
    level, noise = estimate_background(data)
    return _projections_stand_clear(*_projections(data, level), noise)


def measure_fwhm(profile: np.ndarray, baseline: float | None = None) -> tuple[float, float, float]:
    """Measure the Full Width at Half Maximum directly off a profile.

    Walks out from the peak to the first sample below half maximum on each
    side, then interpolates linearly between that sample and its neighbour for
    sub-pixel resolution.  No Gaussian assumed.

    Done naively on noisy data this reads systematically *narrow*: the noisy
    maximum reads high, and noise on a shallow flank dips below half maximum
    before the true crossing -- a sigma-50 profile at a peak signal-to-noise
    of 10 came out 30% too small. So the profile is first smoothed with a
    boxcar no wider than a fifth of the beam's own FWHM, and the (sub-1%)
    broadening that adds is taken back out in quadrature. On the same
    profiles that leaves -1.4% on average.

    Args:
        profile: 1D intensity profile.
        baseline: Background level to measure half-maximum from. If omitted it
            is estimated from the ends of the profile -- never from the
            minimum, which on noisy data sits well below the true floor and
            inflates the width.

    Returns:
        ``(center, fwhm, peak_value)`` in pixel units. A profile with no peak
        standing clear of the noise has no meaningful width, and reports NaN
        for both centre and width rather than a made-up number.
    """
    data = np.asarray(profile, dtype=float)
    base = _profile_baseline(data) if baseline is None else float(baseline)
    data = data - base
    if data.size == 0 or not np.all(np.isfinite(data)):
        return math.nan, math.nan, 0.0

    peak_idx = int(np.argmax(data))
    peak_value = float(data[peak_idx])
    if not _stands_clear(peak_value, _profile_noise(data)):
        return math.nan, math.nan, max(peak_value, 0.0)

    # Size the smoothing from a crude width -- samples above half the raw
    # peak -- so it can never be a large fraction of the beam. Baseline has
    # already been subtracted, so zero padding at the ends adds no artefact.
    half = max(0, int((data > peak_value / 2.0).sum()) // 10)
    if half:
        kernel = np.full(2 * half + 1, 1.0 / (2 * half + 1))
        data = np.convolve(data, kernel, mode="same")
        peak_idx = int(np.argmax(data))
        peak_value = float(data[peak_idx])
    half_max = peak_value / 2.0

    left_idx = peak_idx
    while left_idx > 0 and data[left_idx] > half_max:
        left_idx -= 1
    if left_idx < peak_idx and data[left_idx] < half_max:
        # The loop stopped because this sample dipped below half max, so its
        # right neighbour is still above it — the denominator can't be zero.
        frac = (half_max - data[left_idx]) / (data[left_idx + 1] - data[left_idx])
        left_pos = left_idx + frac
    else:
        left_pos = float(left_idx)

    right_idx = peak_idx
    while right_idx < len(data) - 1 and data[right_idx] > half_max:
        right_idx += 1
    if right_idx > peak_idx and data[right_idx] < half_max:
        frac = (half_max - data[right_idx]) / (data[right_idx - 1] - data[right_idx])
        right_pos = right_idx - frac
    else:
        right_pos = float(right_idx)

    width = right_pos - left_pos
    if half:
        # A boxcar of width L has sigma L/sqrt(12); widths of convolved
        # Gaussian-like peaks add in quadrature.
        box_fwhm = GAUSSIAN_TO_FWHM * (2 * half + 1) / np.sqrt(12.0)
        width = float(np.sqrt(max(width * width - box_fwhm * box_fwhm, 0.0)))
    return (left_pos + right_pos) / 2.0, width, peak_value


# ISO 11146 integrates the second moments over a window three beam widths
# across, centred on the centroid, and iterates until the window settles.
_D4S_WINDOW = 1.5  # half-width of that window, in units of D4-sigma
_D4S_MAX_ITERATIONS = 25


def _window(centre: float, width: float, n: int) -> tuple[int, int]:
    """ISO 11146 integration window around *centre*, clipped to ``[0, n)``."""
    lo = max(0, int(np.floor(centre - _D4S_WINDOW * width)))
    hi = min(n, int(np.ceil(centre + _D4S_WINDOW * width)) + 1)
    return lo, hi


def _is_finite_frame(data: np.ndarray) -> bool:
    """Whether every pixel is finite -- trivially so for an integer frame."""
    return data.dtype.kind not in "fc" or bool(np.all(np.isfinite(data)))


def _kept_variance(cut: float, ndim: int) -> float:
    """Share of a Gaussian's variance left after discarding everything below
    *cut* times its peak.

    Moments of a beam's bright core underestimate its width, by a factor
    that depends only on where the cut was made, so it can be divided back
    out. In 1D the cut sits at ``a = sqrt(2 ln(1/cut))`` sigmas and the kept
    share is ``1 - 2 a phi(a) / erf(a/sqrt2)``. In 2D the kept region is an
    ellipse and the share, per axis, is ``1 - c cut / (2 (1 - cut))`` with
    ``c = 2 ln(1/cut)``. A 20% cut keeps 69% in 1D and 60% in 2D.
    """
    cut = min(max(cut, 1e-6), 0.99)
    c = 2.0 * np.log(1.0 / cut)
    if ndim == 1:
        a = np.sqrt(c)
        # phi(a) is exactly cut / sqrt(2 pi) at the cut point.
        return float(1.0 - 2.0 * a * cut / np.sqrt(2.0 * np.pi) / math.erf(a / np.sqrt(2.0)))
    return float(1.0 - c * cut / (2.0 * (1.0 - cut)))


def _core_cut(peak: float, noise: float) -> float:
    """Level separating a beam's core from the noise around it.

    20% of the peak on its own is not enough. On a small beam the noise can
    be a sizeable share of the peak, and that cut alone let 650 noise-only
    columns into the "core" of a 36 px wide beam.
    """
    return max(0.2 * peak, _SIGNIFICANCE * noise)


def _initial_moments(
    coords: np.ndarray, weights: np.ndarray, noise: float
) -> tuple[float, float] | None:
    """Centre and D4-sigma from the beam's core only, as a starting point.

    Moments of the whole profile are useless as a start: after baseline
    subtraction the noise floor is zero-mean but enormous in aggregate, and a
    second moment weights it by distance squared. Keeping only samples well
    above the noise isolates the beam, and the width lost to the cut is
    divided back out (see :func:`_kept_variance`). The window iteration that
    follows corrects whatever that misses on a non-Gaussian beam.
    """
    peak = float(weights.max())
    if peak <= 0:
        return None
    cut = _core_cut(peak, noise)
    core = weights > cut
    total = float(weights[core].sum())
    if total <= 0:
        return None
    centre = float((coords[core] * weights[core]).sum() / total)
    var = float(((coords[core] - centre) ** 2 * weights[core]).sum() / total)
    var /= _kept_variance(cut / peak, ndim=1)
    return centre, 4.0 * np.sqrt(max(var, 0.25))


def measure_d4s(profile: np.ndarray, baseline: float | None = None) -> tuple[float, float]:
    """Measure the D4σ (ISO 11146 second-moment) width directly off a profile.

    The second moment is intensity-weighted, so unlike a Gaussian fit it stays
    meaningful for flat-top, multi-lobed, or otherwise non-Gaussian beams. It
    is also brutally sensitive to background, because it weights every sample
    by its distance squared. So, following ISO 11146, the baseline is removed
    first and the moments are taken only inside a window three beam widths
    across, re-centred and re-sized until it stops moving.

    For a 2D frame prefer :func:`measure_d4s_2d`: projecting the whole frame
    first sums the noise of every beam-free row into each sample.

    Args:
        profile: 1D intensity profile.
        baseline: Background level. Estimated from the profile's ends, and then
            from outside the integration window, if not given.

    Returns:
        ``(center, d4sigma_width)`` in pixel units. A profile with no peak
        standing clear of the noise reports NaN for both.
    """
    data = np.asarray(profile, dtype=float)
    n = data.size
    if n == 0 or not np.all(np.isfinite(data)):
        return math.nan, math.nan

    fixed_baseline = baseline is not None
    base = float(baseline) if fixed_baseline else _profile_baseline(data)
    coords = np.arange(n, dtype=float)

    excess = data - base
    noise = _profile_noise(data)
    if not _stands_clear(float(excess.max()), noise):
        return math.nan, math.nan

    start = _initial_moments(coords, excess, noise)
    if start is None:
        return math.nan, math.nan
    centre, width = start

    seen: set[tuple[int, int]] = set()
    for _ in range(_D4S_MAX_ITERATIONS):
        lo, hi = _window(centre, width, n)
        if (lo, hi) in seen:
            # Settled (see _settle_window): the moments, and the baseline
            # outside the window, depend on nothing but the window.
            break
        if hi - lo < 3:
            break
        seen.add((lo, hi))
        if not fixed_baseline:
            outside = np.concatenate([data[:lo], data[hi:]])
            if outside.size >= max(8, n // 10):
                base = _robust_mean(outside)[0]
        w = data[lo:hi] - base
        xs = coords[lo:hi]
        total = float(w.sum())
        if total <= 0:
            break
        new_centre = float((xs * w).sum() / total)
        var = float(((xs - new_centre) ** 2 * w).sum() / total)
        if var <= 0:
            break
        centre, width = new_centre, 4.0 * np.sqrt(var)

    return centre, width


def measure_d4s_2d(
    image: np.ndarray, background: float | None = None
) -> tuple[float, float, float, float]:
    """ISO 11146 D4σ widths of a 2D frame, along the image axes.

    The integration area is a rectangle three beam widths across in each
    axis, centred on the centroid and iterated until it settles -- which is
    what the standard specifies, and what keeps the noise of the beam-free
    majority of the sensor out of the moments. Without it, a 6 px beam on a
    megapixel frame at 10 counts of read noise measured 48 times too wide.

    Args:
        image: 2D frame.
        background: Per-pixel background level; estimated from the frame's
            border ring if omitted.

    Returns:
        ``(center_x, center_y, d4sigma_x, d4sigma_y)`` in pixels. NaN
        throughout if no beam stands clear of the noise.
    """
    data = np.asarray(image)
    if data.size == 0 or not _is_finite_frame(data):
        return (math.nan, math.nan, math.nan, math.nan)
    level, noise = estimate_background(data)
    if background is not None:
        level = float(background)
    return _d4s_2d(data, level, noise)[:4]


def _settle_window(
    coords: np.ndarray, excess: np.ndarray, centre: float, width: float
) -> tuple[float, float, tuple[int, int] | None]:
    """Iterate the ISO window on a baseline-subtracted profile until it settles.

    Returns:
        ``(centre, d4s_width, (lo, hi))`` for the last window measured, or
        the starting values and ``None`` if no window of three samples fits.
    """
    n = excess.size
    window = None
    seen: set[tuple[int, int]] = set()
    for _ in range(_D4S_MAX_ITERATIONS):
        lo, hi = _window(centre, width, n)
        if (lo, hi) in seen:
            # The moments depend on nothing but the window, so a window seen
            # before means the iteration has settled -- or, with noise, is
            # flipping an edge between two neighbouring pixels, which it will
            # do forever. The two differ by a column six sigmas out.
            break
        if hi - lo < 3:
            break
        window = (lo, hi)
        seen.add(window)
        w = excess[lo:hi]
        xs = coords[lo:hi]
        total = float(w.sum())
        if total <= 0:
            break
        new_centre = float((xs * w).sum() / total)
        var = float(((xs - new_centre) ** 2 * w).sum() / total)
        if var <= 0:
            break
        centre, width = new_centre, 4.0 * np.sqrt(var)
    return centre, width, window


def _d4s_2d(
    data: np.ndarray, level: float, noise: float
) -> tuple[float, float, float, float, tuple[int, int, int, int] | None]:
    """The work behind :func:`measure_d4s_2d`, plus the final window.

    The moments inside a rectangular window follow from its column and row
    sums alone, so nothing else is computed: the frame is summed as it came
    off the camera, and the background is taken off the sums rather than off
    every pixel.

    The window is settled one axis at a time -- x on the profile of the
    current band of rows, then y on the profile of the band of columns that
    gives -- and the two alternate until neither moves. Each axis feels the
    other only through the band it is summed over, so this reaches the same
    fixed point as moving both at once in far fewer full-band sums.

    Returns:
        ``(cx, cy, d4s_x, d4s_y, (x0, x1, y0, y1))``, or NaNs and ``None``.
    """
    nan5 = (math.nan, math.nan, math.nan, math.nan, None)
    h, w = data.shape
    col, row = _projections(data, level)
    if not _projections_stand_clear(col, row, noise):
        return nan5

    xs = np.arange(w, dtype=float)
    ys = np.arange(h, dtype=float)
    start_x = _initial_moments(xs, col, noise * np.sqrt(h))
    start_y = _initial_moments(ys, row, noise * np.sqrt(w))
    if start_x is None or start_y is None:
        return nan5
    (cx, dx), (cy, dy) = start_x, start_y

    bounds = None
    seen: set[tuple[int, int, int, int]] = set()
    for _ in range(_D4S_MAX_ITERATIONS):
        y0, y1 = _window(cy, dy, h)
        if y1 - y0 < 3:
            break
        band_x = data[y0:y1].sum(axis=0).astype(float) - level * (y1 - y0)
        cx, dx, window_x = _settle_window(xs, band_x, cx, dx)
        if window_x is None:
            break
        x0, x1 = window_x
        band_y = data[:, x0:x1].sum(axis=1).astype(float) - level * (x1 - x0)
        cy, dy, window_y = _settle_window(ys, band_y, cy, dy)
        if window_y is None:
            break
        bounds = (x0, x1, *window_y)
        if bounds in seen:
            break  # settled, or flipping between neighbours (see _settle_window)
        seen.add(bounds)

    return cx, cy, dx, dy, bounds


def measure_fwhm_2d(
    image: np.ndarray, background: float | None = None
) -> tuple[float, float, float, float]:
    """FWHM of a 2D frame along the image axes, from band-limited profiles.

    A full-frame projection sums the noise of the whole sensor height into
    every sample, and that noise makes the half-maximum crossings wander: on
    a 6 px beam at 10 counts of read noise the width scattered by 7% from
    frame to frame, the worst of 40 frames reading 22% narrow. Summing only
    across the band the beam actually occupies (from :func:`measure_d4s_2d`'s
    integration window) cuts that noise by the square root of the band's
    share of the frame -- to 1.3% scatter on the same frames.

    Args:
        image: 2D frame.
        background: Per-pixel background level; estimated from the border
            ring if omitted.

    Returns:
        ``(center_x, center_y, fwhm_x, fwhm_y)`` in pixels, NaN throughout if
        no beam stands clear of the noise.
    """
    data = np.asarray(image)
    if data.size == 0 or not _is_finite_frame(data):
        return (math.nan, math.nan, math.nan, math.nan)
    h, w = data.shape
    level, noise = estimate_background(data)
    if background is not None:
        level = float(background)

    cx, *_, bounds = _d4s_2d(data, level, noise)
    if math.isnan(cx):
        # Same verdict as D4σ on the same frame: no beam at all.
        return (math.nan, math.nan, math.nan, math.nan)
    # A frame too small for an integration window is measured whole.
    x0, x1, y0, y1 = bounds if bounds is not None else (0, w, 0, h)

    band_x = data[y0:y1].sum(axis=0).astype(float)  # along x, over the beam's rows
    band_y = data[:, x0:x1].sum(axis=1).astype(float)  # along y, over its columns
    fx_c, fx_w, _ = measure_fwhm(band_x, baseline=level * (y1 - y0))
    fy_c, fy_w, _ = measure_fwhm(band_y, baseline=level * (x1 - x0))
    return fx_c, fy_c, fx_w, fy_w


def _cold_p0_1d(profile: np.ndarray) -> list[float]:
    """Initial 1D fit guess derived from the profile itself.

    The width comes from how many samples stand above half the peak. The
    old guess, a tenth of the profile length whatever the beam, started a
    2 px beam at 100 px, and the solver ran out of evaluations on the way.
    """
    data = np.asarray(profile, dtype=float)
    base = _profile_baseline(data)
    peak_idx = int(np.argmax(data))
    amplitude = float(data[peak_idx]) - base
    above = int((data - base > amplitude / 2.0).sum()) if amplitude > 0 else 1
    return [amplitude, float(peak_idx), max(above / GAUSSIAN_TO_FWHM, 0.5), base]


def _plausible_1d(popt: np.ndarray, n: int, noise: float) -> bool:
    """Whether 1D fit parameters describe a beam on a profile of *n* samples.

    A solver that reports success has only found *a* minimum. A warm start
    left behind by a beam that moved will happily settle on a dip (negative
    amplitude), run its centre off to 3e10 px, or lock onto a bump in the
    noise, and caching that result as the next frame's start keeps the fit
    there indefinitely.

    Args:
        popt: ``[amplitude, center, sigma, offset]``.
        n: Profile length.
        noise: Noise sigma of the profile (see :func:`_profile_noise`).
    """
    a, x0, sigma, _ = (float(v) for v in popt)
    return bool(
        np.all(np.isfinite(popt))
        and _stands_clear(a, noise)
        and 0.0 <= x0 <= n - 1
        and 0.1 <= abs(sigma) <= n
    )


def _fit_1d(
    profile: np.ndarray, last_popt: np.ndarray | list[Any] | None = None
) -> tuple[np.ndarray, bool]:
    """Fit a 1D Gaussian and say whether the result can be trusted.

    Tries the warm start first (see :func:`fit_1d_gaussian`), then a cold
    guess, and accepts the first result that passes :func:`_plausible_1d`.
    A profile with nothing standing clear of its own noise is not fitted at
    all: whatever the solver found there would be a fit to noise.

    Returns:
        ``(popt, ok)``. When *ok* is ``False``, *popt* is the cold guess and
        describes no beam.
    """
    data = np.asarray(profile, dtype=float)
    n = data.size
    if n == 0:
        return np.array([0.0, 0.0, 1.0, 0.0]), False
    if not np.all(np.isfinite(data)):
        return np.array([0.0, float(n) / 2.0, 1.0, 0.0]), False

    cold = _cold_p0_1d(data)
    noise = _profile_noise(data)
    if not _stands_clear(float(data.max()) - _profile_baseline(data), noise):
        return np.asarray(cold, dtype=float), False

    starts = [list(last_popt), cold] if last_popt is not None else [cold]
    x = np.arange(n)
    for p0 in starts:
        try:
            popt, _ = curve_fit(gaussian, x, data, p0=p0, maxfev=MAX_FIT_ITERATIONS)
        except _FIT_ERRORS as e:
            logger.debug("1D fit from %s failed: %s", p0, e)
            continue
        if _plausible_1d(popt, n, noise):
            return popt, True
        logger.debug("1D fit from %s landed on an implausible %s", p0, popt)
    return np.asarray(cold, dtype=float), False


def fit_1d_gaussian(
    profile: np.ndarray,
    last_popt: np.ndarray | list[Any] | None = None,
) -> np.ndarray | list[Any]:
    """Fit a 1D Gaussian to *profile*.

    Seeding from the previous frame's parameters (*last_popt*) is what keeps
    live streaming cheap -- the solver usually converges in a couple of
    iterations. The catch is that a beam which jumps more than a few sigma
    leaves the warm guess on a flat part of the error surface, where the fit
    either stalls or "converges" onto nonsense, and would otherwise stay stuck
    on stale parameters forever. So a warm start that fails, or lands on
    something that isn't a beam, is retried once from a fresh estimate.

    Args:
        profile: 1D intensity profile.
        last_popt: Previous fit parameters to warm-start from, or ``None``.

    Returns:
        ``[amplitude, center, sigma, offset]``. If no plausible fit exists --
        no beam in the profile, or a solver failure -- this is the initial
        guess instead.
    """
    if len(profile) == 0:
        return [0.0, 0.0, 1.0, 0.0]
    return _fit_1d(profile, last_popt)[0]


def canonical_ellipse(sigma_x: float, sigma_y: float, theta: float) -> tuple[float, float, float]:
    """Put a fitted ellipse into a single, stable parameterisation.

    The rotated 2D Gaussian is degenerate: ``(sx, sy, theta)`` and
    ``(sy, sx, theta + 90 deg)`` describe exactly the same ellipse, and the
    solver picks between them arbitrarily. Left alone, that makes consecutive
    frames of a near-circular beam flip between the two -- the reported X and
    Y widths swap, the angle jumps by 90 degrees, and the warm start for the
    next frame is half a rotation away from where the last one landed.

    Args:
        sigma_x: First principal-axis sigma (may be negative; the model is
            even in sigma).
        sigma_y: Second principal-axis sigma.
        theta: Rotation in radians.

    Returns:
        ``(major, minor, theta)`` with ``major >= minor >= 0`` and *theta*
        wrapped into ``[0, pi)``, so *theta* is always the orientation of the
        major axis.
    """
    sigma_x, sigma_y = abs(float(sigma_x)), abs(float(sigma_y))
    if sigma_y > sigma_x:
        sigma_x, sigma_y = sigma_y, sigma_x
        theta += np.pi / 2
    return sigma_x, sigma_y, float(theta % np.pi)


def image_axis_sigmas(sigma_x: float, sigma_y: float, theta: float) -> tuple[float, float]:
    """Project a tilted ellipse onto the image axes.

    Returns the sigmas an observer measures along the sensor's own x and y —
    the same quantities the 1D projection fits report, so ``fit='2d'`` and
    ``fit='1d'`` describe widths in the same terms.

    These are also invariant under the axis-swap degeneracy that
    :func:`canonical_ellipse` resolves, which is what makes them stable to
    display frame to frame.

    Args:
        sigma_x: Principal-axis sigma along *theta*.
        sigma_y: Principal-axis sigma across it.
        theta: Rotation in radians.

    Returns:
        ``(sigma_along_image_x, sigma_along_image_y)``.
    """
    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2
    vx, vy = sigma_x * sigma_x, sigma_y * sigma_y
    return float(np.sqrt(vx * cos2 + vy * sin2)), float(np.sqrt(vx * sin2 + vy * cos2))


def _fit_region(
    image: np.ndarray,
    max_dim: int,
    sigma_hint: float | tuple[float, float] | None,
    center_hint: tuple[float, float] | None,
) -> tuple[np.ndarray, int, int]:
    """Pick the part of the frame to fit, and where it sits in the original.

    Decimating a tightly focused beam below about a pixel of sigma makes the
    fit unreliable and then non-convergent. The tempting fix — fitting the
    whole frame at full resolution — is far worse: a 3 px beam on a 1 MPix
    sensor took two seconds, which would wedge the live view harder than the
    problem it solved.

    Cropping is the right lever. A small beam only occupies a small part of
    the sensor, so a window around it is both faster *and* better resolved
    than decimating everything. Large beams are left alone and decimated by
    the caller as usual.

    Args:
        image: Full-resolution frame.
        max_dim: Longest edge the fit grid may have after decimation.
        sigma_hint: Rough beam sigma in pixels along ``(x, y)``, or one value
            for a round beam, if known.
        center_hint: Rough beam centre ``(x, y)`` in pixels, if known.

    Returns:
        ``(region, x_offset, y_offset)`` — the sub-image to fit and its origin
        in the original frame, so fitted coordinates can be shifted back.
    """
    h, w = image.shape
    if sigma_hint is None or center_hint is None:
        return image, 0, 0
    sx, sy = sigma_hint if isinstance(sigma_hint, tuple) else (sigma_hint, sigma_hint)
    cx, cy = center_hint
    if not all(np.isfinite(v) for v in (sx, sy, cx, cy)) or min(sx, sy) <= 0:
        return image, 0, 0

    # Only worth cropping if decimation would smear the beam out.
    if min(sx, sy) * max_dim / max(h, w) >= MIN_FIT_2D_SIGMA_PX:
        return image, 0, 0

    # Each axis gets its own half-width. Sizing both from the smaller sigma
    # cropped the long axis of an elongated beam down to little more than
    # one sigma of it.
    half_x = max(FIT_2D_WINDOW_SIGMAS * sx, MIN_FIT_2D_WINDOW_PX / 2)
    half_y = max(FIT_2D_WINDOW_SIGMAS * sy, MIN_FIT_2D_WINDOW_PX / 2)
    x0 = int(max(0, min(w - 1, round(cx - half_x))))
    x1 = int(max(x0 + 1, min(w, round(cx + half_x))))
    y0 = int(max(0, min(h - 1, round(cy - half_y))))
    y1 = int(max(y0 + 1, min(h, round(cy + half_y))))

    if (x1 - x0) >= w and (y1 - y0) >= h:
        return image, 0, 0
    return image[y0:y1, x0:x1], x0, y0


def _moment_seed(image: np.ndarray) -> list[float] | None:
    """Initial 2D fit parameters estimated from the image's own moments.

    The obvious cold start — peak position, a tenth of the frame for each
    sigma, and theta = 0 — is a bad seed for a tilted beam: the solver can
    settle into an axis-aligned compromise that is rounder than the real
    ellipse and report convergence. A 3:1 beam at 34 degrees came back as
    14 x 11 at 0 degrees that way.

    Second moments give the centre, both widths *and* the orientation in one
    pass over the (already decimated) grid, which costs almost nothing and
    starts the solver next to the answer instead of across a ridge from it.

    They are taken over the beam's core only, above the background and well
    clear of the noise. Moments of the whole frame are ruled by its noise: a
    second moment weights every pixel by distance squared, and on the
    simulator's tilted camera that dragged the seed far enough off for 11
    cold fits in 300 to fail. The widths lost to the cut are divided back
    out (see :func:`_kept_variance`); the orientation is unaffected by it.

    Args:
        image: The grid the fit will actually run on.

    Returns:
        ``[amplitude, x0, y0, sigma_major, sigma_minor, theta, offset]``, or
        ``None`` if the image carries no usable signal.
    """
    data = image.astype(float)
    if not np.all(np.isfinite(data)):
        return None
    base, noise = estimate_background(data)
    excess = data - base
    peak = float(excess.max())
    if peak <= 0:
        return None
    cut = _core_cut(peak, noise)
    weights = np.where(excess > cut, excess, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return None

    h, w = weights.shape
    col = weights.sum(axis=0)
    row = weights.sum(axis=1)
    xs = np.arange(w, dtype=float)
    ys = np.arange(h, dtype=float)
    cx = float((col * xs).sum() / total)
    cy = float((row * ys).sum() / total)

    dx = xs - cx
    dy = ys - cy
    vxx = float((col * dx * dx).sum() / total)
    vyy = float((row * dy * dy).sum() / total)
    # The cross term is the only part that needs the full 2D array.
    vxy = float((weights * np.outer(dy, dx)).sum() / total)

    trace = vxx + vyy
    det = vxx * vyy - vxy * vxy
    disc = max(trace * trace / 4.0 - det, 0.0)
    root = np.sqrt(disc)
    kept = _kept_variance(cut / peak, ndim=2)
    var_major = (trace / 2.0 + root) / kept
    var_minor = (trace / 2.0 - root) / kept
    if var_major <= 0:
        return None

    theta = 0.5 * np.arctan2(2.0 * vxy, vxx - vyy)
    return [
        peak,
        cx,
        cy,
        float(np.sqrt(var_major)),
        float(np.sqrt(max(var_minor, 0.25))),
        float(theta),
        base,
    ]


def _peak_seed(image: np.ndarray) -> list[float]:
    """Last-resort 2D guess: the brightest pixel, a tenth of the frame per sigma."""
    fh, fw = image.shape
    pmax, pmin = float(np.max(image)), float(np.min(image))
    y0, x0 = np.unravel_index(int(np.argmax(image)), image.shape)
    return [pmax - pmin, float(x0), float(y0), fw / 10.0, fh / 10.0, 0.0, pmin]


def _stretch_ellipse(
    sigma_x: float, sigma_y: float, theta: float, kx: float, ky: float
) -> tuple[float, float, float]:
    """The ellipse ``(sigma_x, sigma_y, theta)`` after stretching x by *kx*
    and y by *ky*.

    Done on the covariance matrix, ``S C S`` with ``S = diag(kx, ky)``, then
    diagonalised again. Scaling both sigmas by one averaged factor is only
    right for an untilted ellipse or an even stretch: a tilted ellipse's axes
    lie along neither x nor y, so a stretch that differs between the two
    turns the ellipse as well as resizing it.

    Returns:
        ``(sigma_major, sigma_minor, theta_major)``.
    """
    c, s = np.cos(theta), np.sin(theta)
    vx, vy = sigma_x * sigma_x, sigma_y * sigma_y
    a = (vx * c * c + vy * s * s) * kx * kx
    b = (vx - vy) * s * c * kx * ky
    d = (vx * s * s + vy * c * c) * ky * ky
    half_trace = (a + d) / 2.0
    root = np.sqrt(max(half_trace * half_trace - (a * d - b * b), 0.0))
    return (
        float(np.sqrt(half_trace + root)),
        float(np.sqrt(max(half_trace - root, 0.0))),
        float(0.5 * np.arctan2(2.0 * b, a - d)),
    )


def _plausible_2d(popt: np.ndarray, fw: int, fh: int, noise: float) -> bool:
    """Whether 2D fit parameters describe a beam on an *fw* x *fh* grid.

    The same trap as in 1D (:func:`_plausible_1d`): a reported convergence
    only means the solver stopped. A warm start left behind by a beam that
    moved can stop on a centre outside the frame, on a sigma that has
    collapsed onto a single pixel, or -- the one that looks most like a
    result -- on a bump in the noise near where the beam used to be: an
    amplitude of 3 counts on 10 counts of noise, reported as a 3 px beam.

    The fitted beam is held to the same standard as the frame itself in
    :func:`_projections_stand_clear`: each of its projections must stand clear of
    the noise those projections would carry.

    Args:
        popt: ``[amplitude, x0, y0, sigma_x, sigma_y, theta, offset]``, in
            fit-grid coordinates.
        fw: Grid width.
        fh: Grid height.
        noise: Per-pixel noise sigma on the grid.
    """
    amplitude, x0, y0, sigma_x, sigma_y, theta = (float(v) for v in popt[:6])
    limit = max(fw, fh)
    if not (
        np.all(np.isfinite(popt))
        and 0.0 <= x0 <= fw - 1
        and 0.0 <= y0 <= fh - 1
        and 0.1 <= abs(sigma_x) <= limit
        and 0.1 <= abs(sigma_y) <= limit
    ):
        return False
    along_x, along_y = image_axis_sigmas(sigma_x, sigma_y, theta)
    # Peak of the model's projection onto x (summed over y), and onto y.
    root_2pi = np.sqrt(2.0 * np.pi)
    return _stands_clear(amplitude * root_2pi * along_y, noise * np.sqrt(fh)) and _stands_clear(
        amplitude * root_2pi * along_x, noise * np.sqrt(fw)
    )


def fit_2d_gaussian(
    image: np.ndarray,
    last_popt: np.ndarray | list[Any] | None = None,
    max_dim: int = MAX_FIT_2D_DIM,
    sigma_hint: float | tuple[float, float] | None = None,
    center_hint: tuple[float, float] | None = None,
) -> tuple[np.ndarray | list[Any], bool]:
    """Fit a rotated 2D Gaussian to *image*.

    Images larger than *max_dim* on the longest edge are shrunk before fitting
    and the resulting coordinates scaled back up, which keeps 2D fitting
    interactive on megapixel sensors.

    Up to three attempts are made, cheapest first, and the first plausible
    result wins (see :func:`_plausible_2d`):

    1. Levenberg-Marquardt from the warm start. Unbounded, and about 1.8x
       faster than the bounded solver, but free to wander off.
    2. The bounded solver from the same warm start.
    3. The bounded solver from a fresh moment seed. This is what rescues a
       beam that jumped, or a sensor geometry that changed under the cache.

    A frame with no beam standing clear of its noise is not fitted at all.

    Args:
        image: 2D intensity array.
        last_popt: Previous fit parameters (full-resolution coordinates) to
            warm-start from, or ``None`` for a cold start.
        max_dim: Longest edge the fit is allowed to see.
        sigma_hint: Rough beam sigma in pixels along ``(x, y)``, or one value
            for a round beam, if known.
        center_hint: Rough beam centre ``(x, y)`` in pixels, if known. With
            *sigma_hint* this lets a small beam be cropped out of a large
            sensor instead of decimated into illegibility.

    Returns:
        ``([amplitude, x0, y0, sigma_x, sigma_y, theta, offset], ok)``, in
        the canonical form of :func:`canonical_ellipse` so that consecutive
        frames stay comparable. When *ok* is ``False`` no plausible fit was
        found -- there was no beam, or every attempt failed -- and the
        parameters are only the initial guess: don't report them, and don't
        cache them as a warm start.
    """
    # A small beam is cropped out of the sensor; everything else is decimated.
    region, off_x, off_y = _fit_region(image, max_dim, sigma_hint, center_hint)
    h, w = region.shape

    def _canonical(p: list[Any]) -> np.ndarray:
        p = [float(v) for v in p]
        p[3], p[4], p[5] = canonical_ellipse(p[3], p[4], p[5])
        return np.asarray(p, dtype=float)

    if min(h, w) < 3:
        # Too thin to fit seven parameters against.
        seed = _peak_seed(np.asarray(region, dtype=float))
        seed[1] += off_x
        seed[2] += off_y
        return _canonical(seed), False

    ds = max_dim / max(h, w)
    if ds < 1.0:
        # Per-axis factors so neither edge can round down below three pixels.
        shape = (max(3, round(h * ds)), max(3, round(w * ds)))
        fit_img = _ndimage_zoom(region.astype(float), (shape[0] / h, shape[1] / w), order=1)
    else:
        fit_img = region.astype(float)
    fh, fw = fit_img.shape

    # ndimage.zoom lines up the *first and last pixel centres* of input and
    # output, so index j in the decimated image sits at j*(n_in-1)/(n_out-1)
    # in the original -- not at j/ds. Using the naive ratio biases the fitted
    # centre low by (1/ds - 1)/2 px: 3.4 px decimating 1024 to 128, and 7.4 px
    # from 2048, which on a 5 um pitch is 37 um of position error.
    kx = (w - 1) / (fw - 1)
    ky = (h - 1) / (fh - 1)

    def _to_image(p: Any) -> list[Any]:
        """Fit-grid parameters -> full-frame coordinates."""
        p = [float(v) for v in p]
        p[1] = p[1] * kx + off_x
        p[2] = p[2] * ky + off_y
        p[3], p[4], p[5] = _stretch_ellipse(p[3], p[4], p[5], kx, ky)
        return p

    def _to_fit(p: Any) -> list[Any]:
        """Full-frame parameters -> fit-grid coordinates."""
        p = [float(v) for v in p]
        p[1] = (p[1] - off_x) / kx
        p[2] = (p[2] - off_y) / ky
        p[3], p[4], p[5] = _stretch_ellipse(p[3], p[4], p[5], 1.0 / kx, 1.0 / ky)
        return p

    cold = _moment_seed(fit_img) or _peak_seed(fit_img)
    guess = _canonical(_to_image(cold))
    level, noise = estimate_background(fit_img)
    if not _projections_stand_clear(*_projections(fit_img, level), noise):
        return guess, False

    x, y = np.arange(fw), np.arange(fh)
    xv, yv = np.meshgrid(x, y)
    xy_flat = (xv.ravel(), yv.ravel())
    data_flat = fit_img.ravel()
    # The box is looser than the plausibility test on purpose: a beam near an
    # edge may need the solver to step outside the frame on its way in.
    limit = 2.0 * max(fw, fh)
    lower = [0.0, -0.5 * fw, -0.5 * fh, 0.1, 0.1, -np.pi, -np.inf]
    upper = [np.inf, 1.5 * fw, 1.5 * fh, limit, limit, np.pi, np.inf]

    def _lm(p0: list[Any]) -> np.ndarray:
        popt, _ = curve_fit(
            gaussian_2d, xy_flat, data_flat, p0=p0, method="lm", maxfev=MAX_FIT_ITERATIONS
        )
        return popt

    def _bounded(p0: list[Any]) -> np.ndarray:
        # curve_fit rejects a start outside the box ("x0 is infeasible"), and
        # a warm guess can drift out of it. It is only a seed, so nudge it in.
        start = np.clip(np.asarray(p0, dtype=float), lower, upper)
        # NB: the bounded solver is least_squares, which takes ``max_nfev``.
        # ``maxfev`` is an lm-only name, silently ignored once bounds are
        # given, which left scipy's own 100-per-parameter (700) default.
        popt, _ = curve_fit(
            gaussian_2d,
            xy_flat,
            data_flat,
            p0=start,
            bounds=(lower, upper),
            max_nfev=MAX_FIT_2D_EVALS,
        )
        return popt

    attempts: list[tuple[str, Any, list[Any]]] = []
    if last_popt is not None:
        warm = _to_fit(last_popt)
        attempts += [("warm LM", _lm, warm), ("warm bounded", _bounded, warm)]
    attempts.append(("cold", _bounded, cold))

    for name, solve, p0 in attempts:
        try:
            popt = solve(p0)
        except _FIT_ERRORS as e:
            logger.debug("2D %s fit failed: %s", name, e)
            continue
        if _plausible_2d(popt, fw, fh, noise):
            return _canonical(_to_image(popt)), True
        logger.debug("2D %s fit landed on an implausible %s", name, popt)

    return guess, False


def downsample(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Shrink *image* so its longest edge is at most *max_dim* pixels.

    Decimation is nearest-neighbour (fancy indexing), which is several times
    faster than interpolated zoom at this size and — more importantly for a
    profiler — preserves real sensor counts instead of inventing blended ones,
    so what the heatmap shows is what the pixel actually read.  Since there is
    no anti-aliasing either way, the visual difference is negligible.

    Args:
        image: 2D array at full resolution.
        max_dim: Maximum pixels on the longest edge.

    Returns:
        A decimated view/copy, or *image* itself when it already fits.
    """
    h, w = image.shape
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    dh = max(1, round(h * scale))
    dw = max(1, round(w * scale))
    rows = np.linspace(0, h - 1, dh).round().astype(np.intp)
    cols = np.linspace(0, w - 1, dw).round().astype(np.intp)
    return image[rows[:, None], cols]
