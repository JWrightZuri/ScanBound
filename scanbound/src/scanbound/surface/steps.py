"""
Step edges and terraces.

    steps = detect_steps(image)                       # polylines (pixels) with height and direction
    ter = terraces(image, [s.points for s in steps])  # label map, heights, areas, widths, layer numbers
    ter.distance_to_steps(xy)                         # physical distance of points to the nearest step

Step detection
--------------
The topography is smoothed, its gradient magnitude (in height per length, using the true pixel
size) is thresholded at ``threshold`` robust standard deviations above the median, thinned to a
one-pixel skeleton and traced into polylines; short pieces are dropped.  The height of each step is
the median difference of the smoothed topography a few pixels to either side, perpendicular to the
step.  Results are proposals: atomic rows, molecules and tip changes also make steep gradients.
Drawn step-edge annotations (the Step edge tool) work everywhere detected ones do.

Terraces
--------
The image minus a band along every step is split into connected regions.  For each: area, median
height, a width estimate (area / mean length of the steps bounding it - exact for a strip between
two parallel steps), and a layer number (heights divided by the median step height, rounded, so
monatomic steps give consecutive integers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy import ndimage

from .lattice import _data, _filled, image_scale
from .objects import distance_to_polylines


@dataclass
class Step:
    points: np.ndarray                   # (N, 2) pixel polyline
    length: float                        # physical
    height: float                        # signed: z(left of travel direction) - z(right)
    angle: float                         # screen angle of the overall direction (0..180)
    score: float = 0.0                   # mean gradient in robust standard deviations

    def to_dict(self) -> dict[str, Any]:
        return {"points": np.round(self.points, 2).tolist(), "length": self.length, "height": self.height,
                "angle": self.angle, "score": self.score}


def _trace(skel: np.ndarray) -> list[np.ndarray]:
    """Longest path through each connected piece of a skeleton, as (row, col) arrays."""
    lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
    paths = []
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for k in range(1, n + 1):
        pix = np.argwhere(lab == k)
        if len(pix) < 2:
            continue
        index = {tuple(p): i for i, p in enumerate(pix)}
        nbrs = [[index[(p[0] + a, p[1] + b)] for a, b in offs if (p[0] + a, p[1] + b) in index] for p in pix]

        def bfs(start: int) -> tuple[int, list[int]]:
            prev = {start: -1}
            order = [start]
            q = [start]
            while q:
                nq = []
                for u in q:
                    for v in nbrs[u]:
                        if v not in prev:
                            prev[v] = u
                            nq.append(v)
                            order.append(v)
                q = nq
            end = order[-1]
            path = []
            while end != -1:
                path.append(end)
                end = prev[end]
            return order[-1], path
        far, _ = bfs(0)
        _, path = bfs(far)
        paths.append(pix[path])
    return paths


def detect_steps(image: Any, *, smooth: float = 2.0, threshold: float = 6.0, min_length: float = 10.0,
                 simplify: float = 1.0, mask: Optional[np.ndarray] = None) -> list[Step]:
    """
    Step edges as polylines.  ``smooth``: Gaussian sigma in pixels; ``threshold``: robust z of the
    gradient magnitude; ``min_length``: in pixels; ``simplify``: Douglas-Peucker tolerance (pixels).
    """
    from skimage.measure import approximate_polygon
    from skimage.morphology import skeletonize
    (dx, dy), _, _, _ = image_scale(image)
    z = ndimage.gaussian_filter(_filled(_data(image)), smooth)
    gy, gx = np.gradient(z)
    gx, gy = gx / dx, gy / dy
    g = np.hypot(gx, gy)
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med))) * 1.4826 or float(g.std()) or 1.0
    zs = (g - med) / mad
    edge = zs > threshold
    if mask is not None:
        edge &= np.asarray(mask, bool)
    lab, n = ndimage.label(edge, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(edge, lab, index=np.arange(1, n + 1))
        edge = np.isin(lab, 1 + np.nonzero(sizes >= max(4, int(min_length)))[0])
    skel = skeletonize(edge)
    steps = []
    for path in _trace(skel):
        if len(path) < min_length:
            continue
        pts = approximate_polygon(path.astype(float), simplify)[:, ::-1] + 0.5   # (row, col) -> (x, y) centres
        P = pts * [dx, dy]
        L = float(np.hypot(*np.diff(P, axis=0).T).sum())
        # height across: sample at +/- off pixels perpendicular to each segment
        off = 2.5 * smooth + 2
        diffs = []
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            n = np.hypot(*d)
            if n == 0:
                continue
            nx, ny = -d[1] / n, d[0] / n                     # left normal (screen, y down: points "left" of travel)
            for t in np.linspace(0.2, 0.8, 4):
                c = a + t * d
                pl = c + off * np.array([nx, ny])
                pr = c - off * np.array([nx, ny])
                zl = ndimage.map_coordinates(z, [[pl[1] - 0.5], [pl[0] - 0.5]], order=1, mode="nearest")[0]
                zr = ndimage.map_coordinates(z, [[pr[1] - 0.5], [pr[0] - 0.5]], order=1, mode="nearest")[0]
                diffs.append(zl - zr)
        h = float(np.median(diffs)) if diffs else math.nan
        v = pts[-1] - pts[0]
        ang = math.degrees(math.atan2(-v[1] * dy, v[0] * dx)) % 180.0
        rr = np.clip(path[:, 0], 0, g.shape[0] - 1)
        cc = np.clip(path[:, 1], 0, g.shape[1] - 1)
        steps.append(Step(pts, L, h, ang, float(zs[rr, cc].mean())))
    steps.sort(key=lambda s: -s.length)
    return steps


@dataclass
class TerraceResult:
    label: np.ndarray                    # int, -1 on step bands
    terraces: list[dict[str, Any]]
    step_height: float
    scale: tuple[float, float]
    steps: list[np.ndarray] = field(default_factory=list)

    def at(self, xy: np.ndarray) -> np.ndarray:
        rows, cols = self.label.shape
        xy = np.asarray(xy, float).reshape(-1, 2)
        ix = np.clip(np.floor(xy[:, 0]).astype(int), 0, cols - 1)
        iy = np.clip(np.floor(xy[:, 1]).astype(int), 0, rows - 1)
        return self.label[iy, ix]

    def distance_to_steps(self, xy: np.ndarray) -> np.ndarray:
        sc = np.asarray(self.scale)
        if not self.steps:
            return np.full(len(np.asarray(xy).reshape(-1, 2)), np.inf)
        return distance_to_polylines(np.asarray(xy, float).reshape(-1, 2) * sc, [s * sc for s in self.steps])[0]


def _rasterize(lines: Sequence[np.ndarray], shape: tuple[int, int], width: int) -> np.ndarray:
    import cv2
    out = np.zeros(shape, np.uint8)
    for L in lines:
        p = np.round((np.asarray(L, float) - 0.5) * 16).astype(np.int32)
        if len(p) >= 2:
            cv2.polylines(out, [p], False, 1, max(1, width), lineType=cv2.LINE_8, shift=4)
    return out.astype(bool)


def terraces(image: Any, steps: Sequence[np.ndarray], *, band: float = 3.0, min_area: int = 50,
             step_height: Optional[float] = None) -> TerraceResult:
    """Terraces between step polylines (pixels).  ``band``: half-width (pixels) cut out along each step."""
    (dx, dy), _, _, _ = image_scale(image)
    z = _filled(_data(image))
    rows, cols = z.shape
    steps = [np.asarray(s, float) for s in steps]
    cut = _rasterize(steps, (rows, cols), int(round(2 * band + 1)))
    lab, n = ndimage.label(~cut)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    label = np.full((rows, cols), -1, int)
    keep = [k + 1 for k in range(n) if sizes[k] >= min_area]
    for new, k in enumerate(keep):
        label[lab == k] = new
    heights = [float(np.median(z[label == t])) for t in range(len(keep))]
    # which steps bound which terrace: sample both sides of each step
    lens = [float(np.hypot(*(np.diff(s, axis=0) * [dx, dy]).T).sum()) if len(s) > 1 else 0.0 for s in steps]
    bound: dict[int, set[int]] = {t: set() for t in range(len(keep))}
    for si, s in enumerate(steps):
        for a, b in zip(s[:-1], s[1:]):
            d = b - a
            nrm = np.hypot(*d)
            if nrm == 0:
                continue
            nvec = np.array([-d[1], d[0]]) / nrm
            for t in (0.25, 0.5, 0.75):
                c = a + t * d
                for sgn in (1, -1):
                    p = c + sgn * (band + 2) * nvec
                    ix, iy = int(np.clip(p[0], 0, cols - 1)), int(np.clip(p[1], 0, rows - 1))
                    if label[iy, ix] >= 0:
                        bound[int(label[iy, ix])].add(si)
    if step_height is None:
        diffs = []
        for t1 in range(len(keep)):
            for t2 in range(t1 + 1, len(keep)):
                if bound[t1] & bound[t2]:
                    diffs.append(abs(heights[t1] - heights[t2]))
        step_height = float(np.median(diffs)) if diffs else math.nan
    hmin = min(heights) if heights else 0.0
    out = []
    for t in range(len(keep)):
        area = float((label == t).sum() * dx * dy)
        bl = [lens[s] for s in bound[t]]
        width = area / float(np.mean(bl)) if bl and np.mean(bl) > 0 else math.nan
        layer = int(round((heights[t] - hmin) / step_height)) if step_height and np.isfinite(step_height) else 0
        out.append({"terrace": t, "area": area, "height": heights[t], "layer": layer, "width": width,
                    "bounding_steps": sorted(bound[t]), "pixels": int((label == t).sum())})
    return TerraceResult(label, out, float(step_height), (dx, dy), steps)


def distance_classes(d: np.ndarray, edges: Sequence[float]) -> list[str]:
    """Bin distances: edges (physical) [1e-9, 3e-9] -> "< 1 nm", "1-3 nm", "> 3 nm" style labels."""
    from ..model.units import format_value
    out = []
    for v in np.asarray(d, float):
        if not np.isfinite(v):
            out.append("no step")
            continue
        lab = None
        for i, e in enumerate(edges):
            if v < e:
                lab = (f"< {format_value(e, 'm', 3)}" if i == 0 else
                       f"{format_value(edges[i - 1], 'm', 3)} – {format_value(e, 'm', 3)}")
                break
        out.append(lab or f"> {format_value(edges[-1], 'm', 3)}")
    return out
