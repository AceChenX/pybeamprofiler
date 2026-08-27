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

from .constants import (
    FIT_2D_WINDOW_SIGMAS,
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
        ``(center, d4sigma_width)`` in pixel units.  A blank or non-finite
        profile yields the array midpoint and a width of 1 px rather than a
        division by zero or a silent NaN.
    """
    profile = profile - np.min(profile)  # Remove baseline
    profile = np.maximum(profile, 0)  # Guard against float round-off

    total_intensity = float(np.sum(profile))
    # A blank profile divides by zero; a non-finite one (an inf pixel in a
    # float image) makes every moment NaN, which would then propagate silently
    # into the reported width. Both mean "no usable signal".
    if total_intensity == 0 or not np.isfinite(total_intensity):
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
    sigma_hint: float | None,
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
        sigma_hint: Rough smaller beam sigma in pixels, if known.
        center_hint: Rough beam centre ``(x, y)`` in pixels, if known.

    Returns:
        ``(region, x_offset, y_offset)`` — the sub-image to fit and its origin
        in the original frame, so fitted coordinates can be shifted back.
    """
    h, w = image.shape
    if not sigma_hint or sigma_hint <= 0 or center_hint is None:
        return image, 0, 0

    # Only worth cropping if decimation would smear the beam out.
    if sigma_hint * max_dim / max(h, w) >= MIN_FIT_2D_SIGMA_PX:
        return image, 0, 0

    half = max(FIT_2D_WINDOW_SIGMAS * sigma_hint, MIN_FIT_2D_WINDOW_PX / 2)
    cx, cy = center_hint
    x0 = int(max(0, min(w - 1, round(cx - half))))
    x1 = int(max(x0 + 1, min(w, round(cx + half))))
    y0 = int(max(0, min(h - 1, round(cy - half))))
    y1 = int(max(y0 + 1, min(h, round(cy + half))))

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

    Args:
        image: The grid the fit will actually run on.

    Returns:
        ``[amplitude, x0, y0, sigma_major, sigma_minor, theta, offset]``, or
        ``None`` if the image carries no usable signal.
    """
    data = image.astype(float)
    base = float(data.min())
    weights = data - base
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
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
    var_major = trace / 2.0 + root
    var_minor = trace / 2.0 - root
    if var_major <= 0:
        return None

    theta = 0.5 * np.arctan2(2.0 * vxy, vxx - vyy)
    return [
        float(data.max() - base),
        cx,
        cy,
        float(np.sqrt(var_major)),
        float(np.sqrt(max(var_minor, 0.25))),
        float(theta),
        base,
    ]


def fit_2d_gaussian(
    image: np.ndarray,
    last_popt: np.ndarray | list[Any] | None = None,
    max_dim: int = MAX_FIT_2D_DIM,
    sigma_hint: float | None = None,
    center_hint: tuple[float, float] | None = None,
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
        sigma_hint: Rough smaller beam sigma in pixels, if known.
        center_hint: Rough beam centre ``(x, y)`` in pixels, if known. With
            *sigma_hint* this lets a small beam be cropped out of a large
            sensor instead of decimated into illegibility.

    Returns:
        ``([amplitude, x0, y0, sigma_x, sigma_y, theta, offset], converged)``,
        in the canonical form of :func:`canonical_ellipse` so that consecutive
        frames stay comparable. When *converged* is ``False`` the parameters
        are the initial guess and the caller should not cache them as a warm
        start.
    """
    # A small beam is cropped out of the sensor; everything else is decimated.
    region, off_x, off_y = _fit_region(image, max_dim, sigma_hint, center_hint)
    h, w = region.shape

    if max(h, w) > max_dim:
        ds = max_dim / max(h, w)
        fit_img = _ndimage_zoom(region.astype(float), ds, order=1)
    else:
        fit_img = region

    fh, fw = fit_img.shape

    # ndimage.zoom lines up the *first and last pixel centres* of input and
    # output, so index j in the decimated image sits at j*(n_in-1)/(n_out-1)
    # in the original -- not at j/ds. Using the naive ratio biases the fitted
    # centre low by (1/ds - 1)/2 px: 3.4 px decimating 1024 to 128, and 7.4 px
    # from 2048, which on a 5 um pitch is 37 um of position error.
    scale_x = (w - 1) / (fw - 1) if fw > 1 else 1.0
    scale_y = (h - 1) / (fh - 1) if fh > 1 else 1.0
    # The two differ only by rounding, and sigma_x/sigma_y lie along the
    # *principal* axes rather than the image axes, so a single factor is both
    # correct in spirit and accurate to well under a tenth of a percent.
    scale_sigma = (scale_x + scale_y) / 2.0

    def _to_fit_coords(p: list[Any]) -> None:
        """Full-frame parameters → the decimated region's coordinates."""
        p[1] = (p[1] - off_x) / scale_x
        p[2] = (p[2] - off_y) / scale_y
        p[3] /= scale_sigma
        p[4] /= scale_sigma

    def _to_image_coords(p: Any) -> None:
        """Decimated-region parameters → full-frame coordinates."""
        p[1] = p[1] * scale_x + off_x
        p[2] = p[2] * scale_y + off_y
        p[3] *= scale_sigma
        p[4] *= scale_sigma

    if last_popt is not None:
        p0 = list(last_popt)
        _to_fit_coords(p0)
    else:
        p0 = _moment_seed(fit_img)
        if p0 is None:
            pmax, pmin = float(np.max(fit_img)), float(np.min(fit_img))
            y0, x0 = np.unravel_index(int(np.argmax(fit_img)), fit_img.shape)
            p0 = [pmax - pmin, float(x0), float(y0), fw / 10.0, fh / 10.0, 0.0, pmin]

    x, y = np.arange(fw), np.arange(fh)
    xv, yv = np.meshgrid(x, y)
    xy_flat = (xv.ravel(), yv.ravel())
    data_flat = fit_img.ravel()
    # NB: the bounded solver is least_squares, which takes ``max_nfev``.
    # ``maxfev`` is an lm-only name, silently ignored once bounds are given,
    # which left scipy's own 100-per-parameter (700) default in place.
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
                if (
                    abs(popt[3]) < 0.1
                    or abs(popt[4]) < 0.1
                    or abs(popt[3]) > fw
                    or abs(popt[4]) > fh
                ):
                    raise RuntimeError("LM diverged")
            except _FIT_ERRORS:
                popt, _ = curve_fit(
                    gaussian_2d,
                    xy_flat,
                    data_flat,
                    p0=p0_bounded,
                    bounds=bounds,
                    max_nfev=MAX_FIT_2D_EVALS,
                )
        else:
            popt, _ = curve_fit(
                gaussian_2d,
                xy_flat,
                data_flat,
                p0=p0,
                bounds=bounds,
                max_nfev=MAX_FIT_2D_EVALS,
            )

        popt = np.asarray(popt, dtype=float)
        _to_image_coords(popt)
        popt[3], popt[4], popt[5] = canonical_ellipse(popt[3], popt[4], popt[5])
        return popt, True

    except _FIT_ERRORS as e:
        logger.debug("2D fit failed: %s, using initial guess", e)
        _to_image_coords(p0)
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
