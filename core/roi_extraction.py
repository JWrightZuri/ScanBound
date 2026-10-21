"""
roi_extraction.py
=================

Turns a binary mask into one ROI description per distinct object (8-connected
component).  No Qt dependency, so the exact same code is used by
``ImageLabeling.auto_segment_to_rois`` and by the live object preview in
``image_adapt_dialog.py``.

ROI types
---------
RectROI          Bounding box (optionally padded).  Unchanged behaviour.
PolygonROI       Padded, simplified outline that always fully encloses the object.
LineSegmentROI   Centre line (longest skeleton path), simplified to a few vertices.
PointROI         One point per object, at its centroid (centre of mass).

Why the new polygons / lines are small and fast
-----------------------------------------------
* Polygon: the object is first dilated by ``padding + tolerance`` pixels and the
  outline of that grown shape is then simplified with Douglas-Peucker using
  ``tolerance``.  Douglas-Peucker moves the outline by at most ``tolerance``
  pixels, and the grown shape sat ``padding + tolerance`` pixels outside the
  object, so the simplified polygon is *guaranteed* to still enclose every
  object pixel (whole pixel squares, not just centres), however aggressively it is simplified.
  If ``max_vertices`` is exceeded, the tool first switches to the convex hull
  and then raises the tolerance (growing the padding with it) until it fits.
* Line: instead of ordering every skeleton pixel with an O(n^2) Python loop,
  the skeleton is turned into a sparse graph and the longest path is found with
  two Dijkstra sweeps in C (scipy).  That path is then simplified to only the
  vertices needed to stay within ``line_tolerance`` pixels of the centre line.
* Everything is done on per-object crops (not full-image masks per object).

Coordinates
-----------
All points are in scene units, where pixel (col, row) covers
[col, col+1] x [row, row+1] - the convention RectROI already used.  Polygon and
line points therefore sit on pixel centres (index + 0.5) instead of being shifted
half a pixel up-left as raw pixel indices are.  Polygons are clipped to the image.

Dependencies: numpy, opencv-python, scipy, scikit-image (skeletonize only)
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import asdict, dataclass, fields
from typing import Optional

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

ROI_TYPES = ("RectROI", "PolygonROI", "LineSegmentROI", "PointROI")
_SQRT2 = math.sqrt(2.0)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class RoiSettings:
    roi_type: str = "PolygonROI"
    # RectROI
    rect_padding: float = 0.0       # px added on every side of the bounding box
    # PolygonROI
    padding: float = 2.0            # px of free margin kept around the object
    tolerance: float = 2.0          # px Douglas-Peucker tolerance (bigger = fewer vertices)
    max_vertices: int = 8           # 0 = unlimited; otherwise simplify harder to fit
    convex_hull: bool = False       # force convex outlines (fewest vertices)
    # LineSegmentROI
    line_tolerance: float = 1.5     # px; bigger = fewer points along the centre line

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RoiSettings":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in valid})


@dataclass
class SegObject:
    id: int                         # label id (1..N) of the connected component
    roi_type: str
    points: np.ndarray              # (N, 2) float32, columns = (x, y) in image pixels
    bbox: tuple                     # (x, y, w, h) of the raw object pixels
    area: int                       # object area in pixels
    rect: Optional[tuple] = None    # (x1, y1, dx, dy) for RectROI

    @property
    def n_vertices(self) -> int:
        return len(self.points)


# =============================================================================
# Public API
# =============================================================================

def mask_components(mask: np.ndarray):
    """(n, labels, stats, centroids) of the 8-connected components; shareable between calls."""
    return cv2.connectedComponentsWithStats(np.asarray(mask).astype(np.uint8), connectivity=8)


def extract_objects(mask: np.ndarray, settings: Optional[RoiSettings] = None,
                    cache: Optional[dict] = None, components=None) -> list[SegObject]:
    """
    One SegObject per 8-connected foreground object of a boolean mask.

    ``cache`` (a dict kept by the caller between calls) remembers the ROI geometry of every
    object; objects whose pixels did not change since the last call are not recomputed, so
    re-extracting after a small change of the mask only processes the objects that changed.
    ``components`` lets a caller pass a ``mask_components(mask)`` result it already has.
    """
    s = settings or RoiSettings()
    if s.roi_type not in ROI_TYPES:
        raise ValueError(f"Unknown ROI type: {s.roi_type!r}")
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    H, W = mask.shape

    n, labels, stats, centroids = components if components is not None else mask_components(mask)
    known, fresh = {}, {}
    if cache is not None:
        signature = (tuple(sorted(s.to_dict().items())), H, W)
        if cache.get("signature") == signature:
            known = cache.get("items", {})
        cache["signature"] = signature

    objects: list[SegObject] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        crop = labels[y:y + h, x:x + w] == i
        rect = None
        if cache is not None:
            key = (x, y, w, h, area, hashlib.blake2b(np.packbits(crop).tobytes(), digest_size=16).digest())
            hit = known.get(key)
            if hit is not None:
                fresh[key] = hit
                if hit[0] is not None:
                    objects.append(SegObject(i, s.roi_type, hit[0], (x, y, w, h), area, hit[1]))
                continue

        if s.roi_type == "RectROI":
            pad = max(float(s.rect_padding), 0.0)
            x1, y1 = max(0.0, x - pad), max(0.0, y - pad)
            x2, y2 = min(float(W), x + w + pad), min(float(H), y + h + pad)
            rect = (x1, y1, x2 - x1, y2 - y1)
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)

        elif s.roi_type == "PointROI":
            cx, cy = centroids[i]
            pts = np.array([[cx + 0.5, cy + 0.5]], np.float32)     # pixel centre in scene units

        elif s.roi_type == "PolygonROI":
            pts = _polygon(crop, (x, y, w, h), W, H, s)

        else:  # LineSegmentROI
            # 8-connected components never touch, so thinning one object alone gives the same
            # skeleton as thinning the whole mask
            from skimage.morphology import skeletonize
            skel = skeletonize(np.pad(crop, 1))[1:-1, 1:-1]
            pts = _centre_line(skel, x, y, s.line_tolerance)

        if cache is not None:
            fresh[key] = (pts, rect)
        if pts is None:
            continue
        objects.append(SegObject(i, s.roi_type, pts, (x, y, w, h), area, rect))
    if cache is not None:
        cache["items"] = fresh
    return objects


def objects_to_rows(objects: list[SegObject], filename: str, label: str) -> list[list]:
    """
    Rows for ``annotations_df`` (columns: objectName, type, x1, y1, dx, dy, file, label).
    RectROI -> one row; PolygonROI / LineSegmentROI -> one row per vertex.
    """
    rows: list[list] = []
    for obj in objects:
        name = uuid.uuid4()
        if obj.roi_type == "RectROI":
            x1, y1, dx, dy = obj.rect
            rows.append([name, "RectROI", float(x1), float(y1), float(dx), float(dy), filename, label])
        else:
            for px, py in obj.points:
                rows.append([name, obj.roi_type, float(px), float(py), 0, 0, filename, label])
    return rows


# =============================================================================
# Polygon
# =============================================================================

def _polygon_attempt(crop, bbox, pad: float, eps: float, hull: bool) -> np.ndarray:
    """
    Outline of the object grown by pad+eps and simplified by eps, in scene
    coordinates (pixel (c, r) covers [c, c+1] x [r, r+1]).  Not clipped yet.
    """
    x, y, _, _ = bbox
    r = int(math.ceil(pad + eps)) + 1            # +1 covers the half-pixel at the contour
    canvas = np.pad(crop.astype(np.uint8), r + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grown = cv2.dilate(canvas, kernel)
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    if hull:
        cnt = cv2.convexHull(cnt)
    if eps > 0:
        cnt = cv2.approxPolyDP(cnt, eps, True)
    # canvas -> image pixel index (-(r+1)+x), pixel centre -> scene coordinate (+0.5)
    return cnt.reshape(-1, 2).astype(np.float32) + np.array([x - (r + 1) + 0.5, y - (r + 1) + 0.5],
                                                            np.float32)


def _clip_polygon(pts: np.ndarray, W: int, H: int) -> np.ndarray:
    """Sutherland-Hodgman clip against [0, W] x [0, H] (keeps the true shape of the polygon)."""
    poly = [tuple(map(float, p)) for p in pts]
    for axis, bound, keep_greater in ((0, 0.0, True), (0, float(W), False),
                                      (1, 0.0, True), (1, float(H), False)):
        if not poly:
            break

        def inside(p, axis=axis, bound=bound, kg=keep_greater):
            return p[axis] >= bound if kg else p[axis] <= bound

        def cut(a, b, axis=axis, bound=bound):
            t = (bound - a[axis]) / (b[axis] - a[axis])
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

        out = []
        for k in range(len(poly)):
            cur, prev = poly[k], poly[k - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
        poly = out
    # drop consecutive duplicates
    clean = [p for k, p in enumerate(poly) if k == 0 or p != poly[k - 1]]
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
    return np.array(clean, np.float32).reshape(-1, 2)


def _polygon(crop, bbox, W, H, s: RoiSettings) -> np.ndarray:
    pad = max(float(s.padding), 0.5)           # <0.5 px could let simplification clip the object
    eps = max(float(s.tolerance), 0.0)
    cap = int(s.max_vertices)
    if 0 < cap < 3:
        cap = 3
    hull = bool(s.convex_hull)

    pts = _polygon_attempt(crop, bbox, pad, eps, hull)
    if cap:
        if len(pts) > cap and not hull:           # 1st remedy: convex outline
            hull = True
            pts = _polygon_attempt(crop, bbox, pad, eps, hull)
        grow = max(eps, 0.5)
        for _ in range(10):                       # 2nd remedy: coarser + more padding
            if len(pts) <= cap:
                break
            grow *= 1.4
            pts = _polygon_attempt(crop, bbox, pad, grow, hull)

    if len(pts) < 3:                              # degenerate: fall back to padded box
        x, y, w, h = bbox
        p = pad + 1
        pts = np.array([[x - p, y - p], [x + w + p, y - p],
                        [x + w + p, y + h + p], [x - p, y + h + p]], np.float32)
    out = _clip_polygon(pts, W, H)
    return out if len(out) >= 3 else pts


# =============================================================================
# Centre line
# =============================================================================

def _longest_skeleton_path(skel: np.ndarray) -> Optional[np.ndarray]:
    """Ordered (x, y) pixel coordinates of the longest path through a skeleton crop."""
    ys, xs = np.nonzero(skel)
    n = len(ys)
    if n < 2:
        return None
    h, w = skel.shape
    idx = np.full((h, w), -1, np.int32)
    idx[ys, xs] = np.arange(n, dtype=np.int32)

    src_l, dst_l, w_l = [], [], []
    for dy, dx, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, _SQRT2), (1, -1, _SQRT2)):
        y2, x2 = ys + dy, xs + dx
        ok = np.nonzero((y2 < h) & (x2 >= 0) & (x2 < w))[0]
        nb = idx[y2[ok], x2[ok]]
        m = nb >= 0
        src_l.append(ok[m])
        dst_l.append(nb[m])
        w_l.append(np.full(int(m.sum()), wt))
    graph = csr_matrix((np.concatenate(w_l), (np.concatenate(src_l), np.concatenate(dst_l))),
                       shape=(n, n))

    # If the skeleton fell apart into pieces, work on the biggest one.
    n_comp, comp = connected_components(graph, directed=False)
    start = 0
    if n_comp > 1:
        start = int(np.nonzero(comp == np.argmax(np.bincount(comp)))[0][0])

    # Double sweep: farthest pixel from anywhere is an end of the longest path.
    d0 = dijkstra(graph, directed=False, indices=start)
    d0[np.isinf(d0)] = -1
    u = int(np.argmax(d0))
    dist, pred = dijkstra(graph, directed=False, indices=u, return_predecessors=True)
    dist[np.isinf(dist)] = -1
    v = int(np.argmax(dist))

    path = [v]
    while path[-1] != u:
        p = int(pred[path[-1]])
        if p < 0:
            break
        path.append(p)
    path.reverse()
    return np.stack([xs[path], ys[path]], axis=1).astype(np.float32)


def _centre_line(skel_crop, ox, oy, tolerance) -> Optional[np.ndarray]:
    path = _longest_skeleton_path(skel_crop)
    if path is None:
        return None
    if path[0, 0] > path[-1, 0] or (path[0, 0] == path[-1, 0] and path[0, 1] > path[-1, 1]):
        path = path[::-1]                          # consistent left -> right direction
    if tolerance > 0 and len(path) > 2:
        path = cv2.approxPolyDP(path.reshape(-1, 1, 2), float(tolerance), False).reshape(-1, 2)
    return path + np.array([ox + 0.5, oy + 0.5], np.float32)   # pixel centre in scene units
