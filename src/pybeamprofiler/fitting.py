"""Numerical core of the profiler: Gaussian models, width measurements, fits.

Everything here is a plain function over numpy arrays — no camera handles, no
plotting state — so the maths can be exercised (and reused) without standing up
a :class:`~pybeamprofiler.beamprofiler.BeamProfiler`.

Two families of width measurement live side by side:

* **Model-based** — fit a Gaussian and read σ off the fit.  Accurate and
  noise-tolerant *if* the beam really is Gaussian.
* **Direct** — :func:`measure_fwhm` and :func:`measure_d4s` read the width
  straight off the profile.  Slower to converge on noisy data but they make no
  assumption about the beam shape, which is what ISO 11146 asks for.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.ndimage import zoom as _ndimage_zoom
from scipy.optimize import curve_fit

from .constants import MAX_FIT_2D_DIM, MAX_FIT_ITERATIONS

logger = logging.getLogger(__name__)

# curve_fit raises TypeError (not RuntimeError) when the profile has fewer
# samples than free parameters, so it belongs in the same net as the ordinary
# "did not converge" failures.
_FIT_ERRORS = (RuntimeError, ValueError, TypeError)

__all__ = [
    "gaussian",
    "gaussian_2d",
    "measure_fwhm",
    "measure_d4s",
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
        theta: Rotation of the principal axes, in radians.
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
    b = 0.5 * sin2t * (sy2_inv - sx2_inv)
    c = sin2 * sx2_inv + cos2 * sy2_inv

    dx = x - float(x0)
    dy = y - float(y0)
    out = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
    np.negative(out, out=out)
    np.exp(out, out=out)
    out *= amplitude
    out += offset
    return out.ravel()


def measure_fwhm(profile: np.ndarray) -> tuple[float, float, float]:
    """Measure the Full Width at Half Maximum directly off a profile.

    Walks out from the peak to the first sample below half maximum on each
    side, then interpolates linearly between that sample and its neighbour for
    sub-pixel resolution.  No Gaussian assumed.

    Args:
        profile: 1D intensity profile.

    Returns:
        ``(center, fwhm, peak_value)`` in pixel units.
    """
    profile = profile - np.min(profile)  # Remove baseline
    peak_idx = int(np.argmax(profile))
    peak_value = float(profile[peak_idx])
    half_max = peak_value / 2.0

    left_idx = peak_idx
    while left_idx > 0 and profile[left_idx] > half_max:
        left_idx -= 1
    if left_idx < peak_idx and profile[left_idx] < half_max:
        # The loop stopped because this sample dipped below half max, so its
        # right neighbour is still above it — the denominator can't be zero.
        frac = (half_max - profile[left_idx]) / (profile[left_idx + 1] - profile[left_idx])
        left_pos = left_idx + frac
    else:
        left_pos = float(left_idx)

    right_idx = peak_idx
    while right_idx < len(profile) - 1 and profile[right_idx] > half_max:
        right_idx += 1
    if right_idx > peak_idx and profile[right_idx] < half_max:
        frac = (half_max - profile[right_idx]) / (profile[right_idx - 1] - profile[right_idx])
        right_pos = right_idx - frac
    else:
        right_pos = float(right_idx)

    return (left_pos + right_pos) / 2.0, right_pos - left_pos, peak_value


def measure_d4s(profile: np.ndarray) -> tuple[float, float]:
    """Measure the D4σ (ISO 11146 second-moment) width directly off a profile.

    The second moment is intensity-weighted, so unlike a Gaussian fit it stays
    meaningful for flat-top, multi-lobed, or otherwise non-Gaussian beams.  It
    is, however, sensitive to background: subtract a clean baseline (or window
    the profile) before trusting it on low-contrast data.

    Args:
        profile: 1D intensity profile.

    Returns:
        ``(center, d4sigma_width)`` in pixel units.  A blank profile yields the
        array midpoint and a width of 1 px rather than a division by zero.
    """
    profile = profile - np.min(profile)  # Remove baseline
    profile = np.maximum(profile, 0)  # Guard against float round-off

    total_intensity = float(np.sum(profile))
    if total_intensity == 0:
        return len(profile) / 2.0, 1.0

    x = np.arange(len(profile), dtype=float)
    center = float(np.sum(x * profile) / total_intensity)
    variance = float(np.sum(((x - center) ** 2) * profile) / total_intensity)

    return center, 4.0 * np.sqrt(variance)


def _cold_p0_1d(profile: np.ndarray) -> list[float]:
    """Initial 1D fit guess derived from the profile itself."""
    n = len(profile)
    pmax, pmin = float(np.max(profile)), float(np.min(profile))
    return [pmax - pmin, float(np.argmax(profile)), n / 10.0, pmin]


def fit_1d_gaussian(
    profile: np.ndarray,
    last_popt: np.ndarray | list[Any] | None = None,
) -> np.ndarray | list[Any]:
    """Fit a 1D Gaussian to *profile*.

    Seeding from the previous frame's parameters (*last_popt*) is what keeps
    live streaming cheap — the solver usually converges in a couple of
    iterations.  The catch is that a beam which jumps more than a few σ leaves
    the warm guess on a flat part of the error surface, where the fit stalls
    and would otherwise stay stuck on stale parameters forever.  So a failed
    warm start is retried once from a fresh estimate.

    Args:
        profile: 1D intensity profile.
        last_popt: Previous fit parameters to warm-start from, or ``None``.

    Returns:
        ``[amplitude, center, sigma, offset]``.  Falls back to the initial
        guess if the fit does not converge.
    """
    n = len(profile)
    if n == 0:
        return [0.0, 0.0, 1.0, 0.0]

    x = np.arange(n)
    warm = last_popt is not None
    p0 = list(last_popt) if last_popt is not None else _cold_p0_1d(profile)

    for attempt in range(2):
        try:
            popt, _ = curve_fit(gaussian, x, profile, p0=p0, maxfev=MAX_FIT_ITERATIONS)
            return popt
        except _FIT_ERRORS as e:
            if attempt == 0 and warm:
                logger.debug("1D warm-start fit failed (%s); retrying from a cold guess", e)
                p0 = _cold_p0_1d(profile)
                continue
            logger.debug("1D fit failed: %s, using initial guess", e)
            return np.asarray(p0, dtype=float)

    return np.asarray(p0, dtype=float)  # pragma: no cover - loop always returns


def fit_2d_gaussian(
    image: np.ndarray,
    last_popt: np.ndarray | list[Any] | None = None,
    max_dim: int = MAX_FIT_2D_DIM,
) -> tuple[np.ndarray | list[Any], bool]:
    """Fit a rotated 2D Gaussian to *image*.

    Images larger than *max_dim* on the longest edge are shrunk before fitting
    and the resulting coordinates scaled back up, which keeps 2D fitting
    interactive on megapixel sensors.

    Args:
        image: 2D intensity array.
        last_popt: Previous fit parameters (full-resolution coordinates) to
            warm-start from, or ``None`` for a cold start.
        max_dim: Longest edge the fit is allowed to see.

    Returns:
        ``([amplitude, x0, y0, sigma_x, sigma_y, theta, offset], converged)``.
        When *converged* is ``False`` the parameters are the initial guess and
        the caller should not cache them as a warm start.
    """
    h, w = image.shape

    if max(h, w) > max_dim:
        ds = max_dim / max(h, w)
        fit_img = _ndimage_zoom(image.astype(float), ds, order=1)
        inv_ds = 1.0 / ds
    else:
        fit_img = image
        inv_ds = 1.0

    fh, fw = fit_img.shape

    def _rescale(p: list[Any], factor: float) -> None:
        """Scale centers and sigmas between fit and full resolution, in place."""
        for i in (1, 2, 3, 4):
            p[i] *= factor

    if last_popt is not None:
        p0 = list(last_popt)
        _rescale(p0, 1.0 / inv_ds)  # full-res → downsampled coordinates
    else:
        pmax, pmin = float(np.max(fit_img)), float(np.min(fit_img))
        y0, x0 = np.unravel_index(int(np.argmax(fit_img)), fit_img.shape)
        p0 = [pmax - pmin, float(x0), float(y0), fw / 10.0, fh / 10.0, 0.0, pmin]

    x, y = np.arange(fw), np.arange(fh)
    xv, yv = np.meshgrid(x, y)
    xy_flat = (xv.ravel(), yv.ravel())
    data_flat = fit_img.ravel()
    lower = [0, 0, 0, 0.1, 0.1, -np.pi, -np.inf]
    upper = [np.inf, fw, fh, fw, fh, np.pi, np.inf]
    bounds = (lower, upper)

    # curve_fit rejects an infeasible start outright ("x0 is infeasible"), so a
    # warm guess that has drifted outside the box would take the bounded
    # fallback down with it. Nudge it back in first — it is only a seed.
    p0_bounded = list(np.clip(np.asarray(p0, dtype=float), lower, upper))

    try:
        if last_popt is not None:
            # Warm: unbounded Levenberg-Marquardt is ~1.8x faster than the
            # trust-region solver bounds force us into, but it can wander off
            # to a degenerate sigma. Sanity-check the result and fall back.
            try:
                popt, _ = curve_fit(
                    gaussian_2d,
                    xy_flat,
                    data_flat,
                    p0=p0,
                    method="lm",
                    maxfev=MAX_FIT_ITERATIONS,
                )
                if abs(popt[3]) < 0.1 or abs(popt[4]) < 0.1 or popt[3] > fw or popt[4] > fh:
                    raise RuntimeError("LM diverged")
            except _FIT_ERRORS:
                popt, _ = curve_fit(
                    gaussian_2d,
                    xy_flat,
                    data_flat,
                    p0=p0_bounded,
                    bounds=bounds,
                    maxfev=MAX_FIT_ITERATIONS,
                )
        else:
            popt, _ = curve_fit(
                gaussian_2d,
                xy_flat,
                data_flat,
                p0=p0,
                bounds=bounds,
                maxfev=MAX_FIT_ITERATIONS,
            )

        if inv_ds != 1.0:
            popt = np.asarray(popt, dtype=float)
            popt[1:5] *= inv_ds
        return popt, True

    except _FIT_ERRORS as e:
        logger.debug("2D fit failed: %s, using initial guess", e)
        _rescale(p0, inv_ds)  # back to full-resolution coordinates
        return p0, False


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
