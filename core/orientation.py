"""
orientation.py
==============

ROI orientations: estimating them from the image and the angle conventions shared by the
labelling page and the ROI analysis.  No Qt dependency.

Convention
----------
An orientation is an angle in degrees, counter-clockwise from the image x axis as seen on
screen (the image y axis points down, so a direction (dx, dy) in pixel coordinates has the
angle atan2(-dy, dx)).  0° points right, 90° points up.

Symmetry ("fold") n: an object that looks the same after a rotation of 360°/n has its angle
reduced to [0, 360/n).  n = 1 is a polar object (head / tail, 0-360°), n = 2 an axial one
(rods, 0-180°), n = 3 a three-fold molecule (0-120°), ...
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def vector_angle(x1, y1, x2, y2) -> float:
    """Screen angle (degrees, CCW) of the direction (x1, y1) -> (x2, y2) in pixel coordinates."""
    return math.degrees(math.atan2(-(y2 - y1), x2 - x1)) % 360.0


def fold_angle(angle, fold=1):
    """Angle reduced to [0, 360/fold)."""
    period = 360.0 / max(1, int(fold))
    return np.mod(angle, period)


def relative_angle(angle, reference=None, fold=1):
    """Angle relative to a reference direction (degrees, CCW), reduced by the symmetry fold."""
    a = np.asarray(angle, dtype=float)
    if reference is not None:
        a = a - float(reference)
    return fold_angle(a, fold)


# =============================================================================
# Circular statistics (with symmetry fold)
# =============================================================================

def circular_stats(angles, fold=1) -> dict:
    """
    Mean direction, alignment (mean resultant length R: 0 = random, 1 = all parallel),
    circular standard deviation and Rayleigh test p-value of angles (degrees) with n-fold
    symmetry.  Angles are multiplied by the fold so that e.g. 10° and 190° count as equal
    for axial objects.
    """
    a = np.asarray(angles, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    out = {"n": n, "mean": np.nan, "R": np.nan, "circ_std": np.nan, "rayleigh_p": np.nan}
    if n == 0:
        return out
    k = max(1, int(fold))
    t = np.radians(a * k)
    C, S = np.cos(t).mean(), np.sin(t).mean()
    R = float(math.hypot(C, S))
    out["R"] = R
    out["mean"] = float((math.degrees(math.atan2(S, C)) / k) % (360.0 / k))
    out["circ_std"] = float(math.degrees(math.sqrt(max(0.0, -2.0 * math.log(min(R, 1.0))))) / k) if R > 1e-12 else np.inf
    Rn = n * R                                                    # Rayleigh test (Zar 1999)
    out["rayleigh_p"] = float(min(1.0, math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - Rn * Rn)) - (1 + 2 * n))))
    return out


# =============================================================================
# Estimating an orientation from the image
# =============================================================================

def _region(image, points, closed, point_radius):
    """(crop, weights mask, x0, y0) of the image area an ROI covers."""
    import cv2
    h, w = image.shape
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 1:                                   # point ROI: a disc around it
        cx, cy = pts[0]
        r = point_radius
        x0, y0 = int(max(0, math.floor(cx - r))), int(max(0, math.floor(cy - r)))
        x1, y1 = int(min(w, math.ceil(cx + r))), int(min(h, math.ceil(cy + r)))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r
        return image[y0:y1, x0:x1], inside, x0, y0
    x0, y0 = int(max(0, math.floor(pts[:, 0].min()))), int(max(0, math.floor(pts[:, 1].min())))
    x1, y1 = int(min(w, math.ceil(pts[:, 0].max()))), int(min(h, math.ceil(pts[:, 1].max())))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, None, x0, y0
    inside = np.zeros((y1 - y0, x1 - x0), np.uint8)
    poly = np.round((pts - [x0 + 0.5, y0 + 0.5]) * 16).astype(np.int32)
    if closed:
        cv2.fillPoly(inside, [poly], 1, lineType=cv2.LINE_8, shift=4)
    else:
        cv2.polylines(inside, [poly], False, 1, thickness=3, shift=4)
    return image[y0:y1, x0:x1], inside.astype(bool), x0, y0


def estimate_orientation(image, points, closed=True, polar=True, bright=True,
                         point_radius=8.0) -> Optional[float]:
    """
    Orientation (degrees, screen convention) of the feature inside an ROI, from the
    intensity-weighted second moments of the image region (its principal axis).

    polar=True also picks a direction along that axis: towards the side where the
    feature extends further (positive skewness).  This is a heuristic - check the arrows
    and flip the wrong ones.  ``bright`` = features are brighter than their surroundings.
    Returns None when the region is too small or featureless.
    """
    image = np.asarray(image, dtype=float)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if not closed and len(pts) >= 2:                 # line ROI: its own direction
        a = vector_angle(pts[0, 0], pts[0, 1], pts[-1, 0], pts[-1, 1])
        return a if polar else a % 180.0
    crop, inside, x0, y0 = _region(image, pts, closed, point_radius)
    if crop is None or inside.sum() < 5:
        return None
    vals = crop[inside]
    if not bright:
        vals = -vals
    weights = np.clip(vals - np.percentile(vals, 25), 0, None)
    if weights.sum() <= 0:
        return None
    ys, xs = np.nonzero(inside)
    wsum = weights.sum()
    cx, cy = (weights * xs).sum() / wsum, (weights * ys).sum() / wsum
    dx, dy = xs - cx, ys - cy
    cxx, cyy, cxy = (weights * dx * dx).sum() / wsum, (weights * dy * dy).sum() / wsum, (weights * dx * dy).sum() / wsum
    theta = 0.5 * math.atan2(2 * cxy, cxx - cyy)       # pixel coordinates (y down)
    ux, uy = math.cos(theta), math.sin(theta)
    if polar:
        s = dx * ux + dy * uy
        if (weights * s ** 3).sum() < 0:
            ux, uy = -ux, -uy
    angle = math.degrees(math.atan2(-uy, ux)) % 360.0
    return angle if polar else angle % 180.0
