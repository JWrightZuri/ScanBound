"""
SPM data processing steps (levelling, row alignment, scar removal, filters).

Unlike the adaptation steps these keep the data in its physical unit (heights stay in m,
currents in A ...), so profiles, colour bars and 3D views show real values.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .fourier import FOURIER_STEP
from .registry import Param, StepRegistry, StepSpec

PIPELINE_FORMAT = "scanbound_spm_pipeline"


def _plane_fit(z, mask=None):
    rows, cols = z.shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    sel = np.isfinite(z) if mask is None else (mask & np.isfinite(z))
    A = np.column_stack([np.ones(sel.sum()), xx[sel], yy[sel]])
    c, *_ = np.linalg.lstsq(A, z[sel], rcond=None)
    return c[0] + c[1] * xx + c[2] * yy


def s_plane(z, robust):
    """Subtract the least-squares plane (robust: refit without points > 2.5 sigma away)."""
    plane = _plane_fit(z)
    if robust:
        resid = z - plane
        mask = np.abs(resid - np.median(resid)) < 2.5 * np.std(resid)
        plane = _plane_fit(z, mask)
    return z - plane


def s_poly(z, order):
    """Subtract a 2D polynomial background of the given order (fitted on a coarse grid)."""
    rows, cols = z.shape
    step = max(1, int(max(rows, cols) // 128))
    yy, xx = np.mgrid[0:rows:step, 0:cols:step].astype(float)
    sub = z[::step, ::step]
    xn, yn = xx / max(cols - 1, 1) * 2 - 1, yy / max(rows - 1, 1) * 2 - 1
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.column_stack([(xn ** i * yn ** j).ravel() for i, j in terms])
    c, *_ = np.linalg.lstsq(A, sub.ravel(), rcond=None)
    YY, XX = np.mgrid[0:rows, 0:cols].astype(float)
    XN, YN = XX / max(cols - 1, 1) * 2 - 1, YY / max(rows - 1, 1) * 2 - 1
    return z - sum(ci * XN ** i * YN ** j for ci, (i, j) in zip(c, terms))


def s_align_rows(z, method):
    """Remove line-to-line offsets of the slow scan axis."""
    if method == "Median":
        return z - np.median(z, axis=1, keepdims=True)
    if method == "Mean":
        return z - z.mean(axis=1, keepdims=True)
    if method == "Linear (per line)":
        x = np.arange(z.shape[1], dtype=float)
        coef = np.polyfit(x, z.T, 1)                       # (2, rows)
        return z - (np.outer(coef[0], x) + coef[1][:, None])
    # Median of differences: shift each line by the median step to the previous line
    diff = np.median(np.diff(z, axis=0), axis=1)
    offsets = np.concatenate([[0.0], np.cumsum(diff)])
    out = z - offsets[:, None]
    return out - np.median(out)


def s_scars(z, threshold, max_length):
    """Replace scan-line scars (short pieces of a line that jump away from both neighbours)."""
    up, down = np.roll(z, 1, axis=0), np.roll(z, -1, axis=0)
    neighbours = (up + down) / 2
    resid = z - neighbours
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) or np.std(resid) or 1.0
    bad = (np.abs(resid) > threshold * sigma) & (np.sign(z - up) == np.sign(z - down))
    bad[0], bad[-1] = False, False
    if max_length > 0:                                      # keep only short horizontal runs
        labels, n = ndimage.label(bad, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        if n:
            sizes = ndimage.sum(bad, labels, range(1, n + 1))
            long_runs = np.isin(labels, np.nonzero(sizes > max_length)[0] + 1)
            bad &= ~long_runs
    out = z.copy()
    out[bad] = neighbours[bad]
    return out


def s_background(z, sigma):
    """Remove long-range variations: subtract a heavily smoothed copy (high-pass)."""
    return z - ndimage.gaussian_filter(z, sigma)


def s_gaussian(z, sigma):
    return ndimage.gaussian_filter(z, sigma)


def s_median(z, size):
    return ndimage.median_filter(z, size=int(size))


def s_sharpen(z, strength, sigma):
    """Unsharp mask: z + strength * (z - blur(z))."""
    return z + strength * (z - ndimage.gaussian_filter(z, sigma))


def s_dog(z, sigma, ratio):
    """Difference of Gaussians (band-pass): keeps features of size ~sigma."""
    return ndimage.gaussian_filter(z, sigma) - ndimage.gaussian_filter(z, sigma * ratio)


def s_clip(z, low, high):
    lo, hi = np.percentile(z, [low, 100 - high])
    return np.clip(z, lo, hi)


def s_zero(z, reference):
    ref = {"Minimum": np.min(z), "Median": np.median(z), "Mean": np.mean(z)}[reference]
    return z - ref


def s_invert(z):
    return -z


SPM_STEPS = StepRegistry("spm", [
    StepSpec("plane", "Plane level", "gray", s_plane, [
        Param("robust", "Ignore outliers", "bool", True,
              tooltip="Refit without pixels far from the first plane (particles, steps)."),
    ], "Subtract the best-fitting plane (sample tilt)."),
    StepSpec("poly", "Polynomial background", "gray", s_poly, [
        Param("order", "Order", "int", 2, 1, 6, 1, 0),
    ], "Subtract a smooth 2D polynomial (bow, scanner curvature)."),
    StepSpec("align_rows", "Align rows", "gray", s_align_rows, [
        Param("method", "Method", "choice", "Median",
              choices=("Median", "Mean", "Median of differences", "Linear (per line)")),
    ], "Remove offsets between scan lines (horizontal streaks)."),
    StepSpec("scars", "Remove scars", "gray", s_scars, [
        Param("threshold", "Threshold (σ)", "float", 4.0, 1.0, 20.0, 0.5, 1),
        Param("max_length", "Max length (px, 0 = any)", "int", 0, 0, 2000, 1, 0),
    ], "Replace short line segments that jump away from both neighbouring lines."),
    StepSpec("background", "Remove background (high-pass)", "gray", s_background, [
        Param("sigma", "Sigma (px)", "float", 20.0, 1.0, 300.0, 1.0, 1),
    ], "Subtract a smoothed copy: removes features larger than ~sigma."),
    StepSpec("gaussian", "Gaussian smoothing", "gray", s_gaussian, [
        Param("sigma", "Sigma (px)", "float", 1.0, 0.2, 20.0, 0.1, 1),
    ], "Reduce noise."),
    StepSpec("median", "Median filter", "gray", s_median, [
        Param("size", "Size (px)", "int", 3, 2, 15, 1, 0),
    ], "Remove spikes while keeping edges."),
    StepSpec("sharpen", "Sharpen", "gray", s_sharpen, [
        Param("strength", "Strength", "float", 1.0, 0.0, 5.0, 0.1, 1),
        Param("sigma", "Radius (px)", "float", 1.0, 0.3, 10.0, 0.1, 1),
    ], "Unsharp mask."),
    StepSpec("dog", "Difference of Gaussians", "gray", s_dog, [
        Param("sigma", "Sigma (px)", "float", 1.0, 0.3, 20.0, 0.1, 1),
        Param("ratio", "Ratio", "float", 1.6, 1.1, 10.0, 0.1, 1),
    ], "Band-pass filter: emphasises features of size ~sigma."),
    StepSpec("clip", "Clip outliers", "gray", s_clip, [
        Param("low", "Low (%)", "float", 0.5, 0.0, 20.0, 0.1, 1),
        Param("high", "High (%)", "float", 0.5, 0.0, 20.0, 0.1, 1),
    ], "Limit values to percentiles (removes spikes from the data)."),
    StepSpec("zero", "Shift to zero", "gray", s_zero, [
        Param("reference", "Set to zero", "choice", "Minimum", choices=("Minimum", "Median", "Mean")),
    ], "Offset the data, e.g. so heights start at 0."),
    StepSpec("invert", "Invert", "gray", s_invert, [], "Flip the sign of the data."),
    FOURIER_STEP,
])

DEFAULT_SPM_PIPELINE: list[tuple[str, bool, dict]] = [("plane", False, {}), ("align_rows", False, {"method": "Median"})]
