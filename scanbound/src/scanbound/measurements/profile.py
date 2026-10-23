"""
Height (or any value) profiles along a line or path, on the physical data.

    prof = line_profile(image, annotation.geometry, width_px=3)
    prof.distance, prof.values          # metres and the image's unit (NaN where there is no data)
    step = prof.step_height()           # Measurement: plateau difference with uncertainty
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .measurement import Measurement

if TYPE_CHECKING:
    from ..annotations.geometry import Geometry
    from ..model.image import Image


@dataclass(frozen=True)
class Profile:
    distance: np.ndarray        # along the path, in the image's length unit
    values: np.ndarray          # in value_unit
    length_unit: str
    value_unit: str
    width_px: float
    points: np.ndarray          # pixel coordinates of the samples (N, 2)

    def roughness(self) -> tuple[float, float]:
        """(RMS, mean absolute deviation) of the finite values."""
        v = self.values[np.isfinite(self.values)]
        if v.size == 0:
            return math.nan, math.nan
        d = v - v.mean()
        return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))

    def step_height(self, fraction: float = 0.3) -> Measurement:
        """
        Difference between the mean of the first and the last ``fraction`` of the profile.  The
        uncertainty is the standard error of both plateau means (noise only; tilt left in the data
        is not corrected here, so level the image first).
        """
        v = self.values
        n = len(v)
        k = max(2, int(n * fraction))
        a, b = v[:k][np.isfinite(v[:k])], v[-k:][np.isfinite(v[-k:])]
        if a.size < 2 or b.size < 2:
            return Measurement("step_height", math.nan, self.value_unit, None, "not enough valid samples")
        h = float(b.mean() - a.mean())
        comps = {"noise (left plateau)": float(a.std(ddof=1) / math.sqrt(a.size)),
                 "noise (right plateau)": float(b.std(ddof=1) / math.sqrt(b.size))}
        return Measurement("step_height", h, self.value_unit, math.hypot(*comps.values()),
                           f"mean of the last {fraction:.0%} minus mean of the first {fraction:.0%} of the profile",
                           ("both ends lie on flat terraces", "the image was levelled beforehand"),
                           {"samples": [int(a.size), int(b.size)]}, comps)


def sample_path(points: np.ndarray, step_px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step_px`` pixels along a polyline: (samples (N, 2), unit tangents (N, 2))."""
    p = np.asarray(points, float).reshape(-1, 2)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(2, int(math.ceil(total / step_px)) + 1)
    t = np.linspace(0.0, total, n)
    xs, ys = np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])
    idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1)
    d = np.diff(p, axis=0)[idx]
    norm = np.hypot(d[:, 0], d[:, 1])
    tang = d / np.where(norm > 0, norm, 1)[:, None]
    return np.column_stack([xs, ys]), tang


def line_profile(image: "Image", geometry: "Geometry", width_px: float = 1.0, step_px: float = 1.0) -> Profile:
    """Values along a segment / polyline, averaged over a band ``width_px`` wide (NaN-aware)."""
    pts, tang = sample_path(geometry.points, step_px)
    normals = np.column_stack([-tang[:, 1], tang[:, 0]])
    offsets = [0.0] if width_px <= 1 else np.linspace(-(width_px - 1) / 2, (width_px - 1) / 2, int(round(width_px)))
    rows = [image.sample(pts[:, 0] + o * normals[:, 0], pts[:, 1] + o * normals[:, 1]) for o in offsets]
    with np.errstate(all="ignore"):
        vals = np.nanmean(np.vstack(rows), axis=0) if len(rows) > 1 else rows[0]
    dx, dy = image.pixel_size
    phys = np.diff(pts, axis=0) * np.array([dx, dy])
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(phys[:, 0], phys[:, 1]))])
    return Profile(dist, vals, image.length_unit, image.unit, float(width_px), pts)
