"""
2D Fourier filtering (like Gwyddion's 2D FFT filter), without the drawing dialog.

The filter is a set of shapes on the (centred) Fourier transform.  They either mark frequencies
to remove (stop) or the only frequencies to keep (pass).  Shapes are in frequency units
(cycles / pixel, centre = 0, Nyquist = ±0.5), so a filter applies to images of any size, and are
mirrored automatically because the transform of real data is point-symmetric.

Shapes:  ("ellipse", cx, cy, rx, ry)   ("rect", x0, y0, x1, y1)   ("poly", ((x, y), ...))
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .registry import Param, StepSpec

MODES = ("Remove selected (stop)", "Keep only selected (pass)")


def frequency_grid(shape):
    """(fx, fy) in cycles / pixel of the fftshift-ed transform of an image of `shape`."""
    rows, cols = shape
    fx = (np.arange(cols) - cols // 2) / cols
    fy = (np.arange(rows) - rows // 2) / rows
    return np.meshgrid(fx, fy)


def _shape_mask(fx, fy, shape):
    kind = shape[0]
    if kind == "ellipse":
        _, cx, cy, rx, ry = shape
        rx, ry = max(abs(rx), 1e-9), max(abs(ry), 1e-9)
        return ((fx - cx) / rx) ** 2 + ((fy - cy) / ry) ** 2 <= 1.0
    if kind == "rect":
        _, x0, y0, x1, y1 = shape
        return (fx >= min(x0, x1)) & (fx <= max(x0, x1)) & (fy >= min(y0, y1)) & (fy <= max(y0, y1))
    if kind == "poly":
        from matplotlib.path import Path
        pts = np.asarray(shape[1], float)
        if len(pts) < 3:
            return np.zeros(fx.shape, bool)
        inside = Path(pts).contains_points(np.column_stack([fx.ravel(), fy.ravel()]))
        return inside.reshape(fx.shape)
    return np.zeros(fx.shape, bool)


def build_mask(shape, shapes, mode=MODES[0], smooth=0.0):
    """Transfer function (0..1) on the fftshift-ed grid of an image of `shape`."""
    fx, fy = frequency_grid(shape)
    selected = np.zeros(shape, bool)
    for s in shapes or ():
        selected |= _shape_mask(fx, fy, s) | _shape_mask(-fx, -fy, s)      # point-symmetric
    keep = selected if mode == MODES[1] else ~selected
    mask = keep.astype(float)
    if smooth > 0:
        mask = ndimage.gaussian_filter(mask, smooth)
    return mask


def fourier_filter(z, shapes, mode=MODES[0], smooth=0.0):
    """Filtered image (same unit); the mean is always kept."""
    z = np.asarray(z, dtype=float)
    if not shapes:
        return z.copy()
    mean = float(np.mean(z))
    F = np.fft.fftshift(np.fft.fft2(z - mean))
    out = np.fft.ifft2(np.fft.ifftshift(F * build_mask(z.shape, shapes, mode, smooth))).real
    return out + mean


def log_spectrum(z, window=False):
    """log10 |F|² of the mean-free image, centred.  window=True applies a Hann window first
    (less spectral leakage from the image edges; used to detect peaks)."""
    z = np.asarray(z, dtype=float)
    z = z - np.mean(z)
    if window:
        z = z * np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(z))
    return np.log10(np.abs(F) ** 2 + 1e-30)


def find_peaks(z, threshold=5.0, max_peaks=12, exclude_radius=0.03, radius=0.01):
    """
    Ellipses around isolated bright peaks of the spectrum (periodic noise / lattice), each
    peak once (its mirror is implied).  threshold: in standard deviations above the median.
    """
    spec = log_spectrum(z, window=True)
    fx, fy = frequency_grid(spec.shape)
    centre = np.hypot(fx, fy) < exclude_radius
    centre |= (np.abs(fx) > 0.48) | (np.abs(fy) > 0.48)          # outermost border: leakage / aliasing
    # periodic noise = sharp spike above its *local* surroundings (the broad spectrum of the real
    # features is bright too, but smooth), so compare with a local median background
    work = spec - ndimage.median_filter(spec, size=15)
    work[centre] = 0.0
    local_max = work == ndimage.maximum_filter(work, size=7)
    med = np.median(work)
    sd = 1.4826 * np.median(np.abs(work - med)) or work.std() or 1.0
    cand = np.argwhere(local_max & (work > med + threshold * sd))
    peaks = sorted(cand, key=lambda rc: -work[rc[0], rc[1]])
    shapes: list[tuple] = []
    seen: list[tuple[float, float]] = []
    for r, c in peaks:
        x, y = float(fx[r, c]), float(fy[r, c])
        if any(math.hypot(x - a, y - b) < 2 * radius or math.hypot(x + a, y + b) < 2 * radius for a, b in seen):
            continue
        seen.append((x, y))
        shapes.append(("ellipse", x, y, radius, radius))
        if len(shapes) >= max_peaks:
            break
    return shapes


def as_shapes(value):
    """Shapes from JSON lists back to hashable tuples."""
    out = []
    for s in value or ():
        if not s:
            continue
        if s[0] == "poly":
            out.append(("poly", tuple(tuple(float(v) for v in p) for p in s[1])))
        else:
            out.append((s[0],) + tuple(float(v) for v in s[1:]))
    return tuple(out)


def s_fourier(z, mode, smooth, shapes=()):
    return fourier_filter(z, as_shapes(shapes), mode, smooth)


FOURIER_STEP = StepSpec("fourier", "Fourier filter", "gray", s_fourier, [
    Param("mode", "Mode", "choice", MODES[0], choices=MODES),
    Param("smooth", "Edge softness (px)", "float", 1.0, 0.0, 10.0, 0.5, 1,
          tooltip="Soften the mask edges to avoid ringing."),
], "Remove (or keep only) frequency regions drawn on the Fourier transform: periodic noise, "
   "low-pass / high-pass / band-pass.")


FOURIER_STEP = StepSpec("fourier", "Fourier filter", "gray", s_fourier, (
    Param("mode", "Mode", "choice", MODES[0], choices=MODES),
    Param("smooth", "Edge softness (px)", "float", 1.0, 0.0, 10.0, 0.5, 1,
          tooltip="Soften the mask edges to avoid ringing."),
), "Remove (or keep only) frequency regions drawn on the Fourier transform: periodic noise, "
   "low-pass / high-pass / band-pass.", extra_keys=("shapes",))
