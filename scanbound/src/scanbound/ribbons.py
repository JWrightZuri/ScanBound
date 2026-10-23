"""
Nanoribbon analysis: centre lines, lengths and widths of elongated objects.

The mask is thinned to a 1 px skeleton.  With ``split_at_junctions`` the skeleton is cut at its
branch points (pixels with 3 or more skeleton neighbours) and every remaining piece is one
ribbon segment, as in the original ``core/GNR_segment.py``; otherwise each object gives one
centre line, the longest path through its skeleton (the LineSegmentROI definition).

Replaces the pixel-by-pixel Python graph walk of ``run_GNR_segment`` with connected-component
labelling and a sparse-graph longest path per segment, so a full image takes milliseconds.

    ribbons = find_ribbons(mask, px_scale=0.1)
    table = ribbons_table(ribbons)           # DataFrame: length, width, angle per segment
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from .roi import longest_skeleton_path

_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)


@dataclass
class Ribbon:
    id: int                         # 1..N in the order found
    object_id: int                  # SegObject.id of the 8-connected object it belongs to
    points: np.ndarray              # (N, 2) float32 (x, y) in scene units (pixel centres = index + 0.5)
    length_px: float                # path length along the centre line
    width_px: float                 # mean object width along the centre line (2 * distance - 1)
    px_scale: Optional[float] = None

    @property
    def length_nm(self) -> Optional[float]:
        return self.length_px * self.px_scale if self.px_scale else None

    @property
    def width_nm(self) -> Optional[float]:
        return self.width_px * self.px_scale if self.px_scale else None

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        a, b = self.points[0], self.points[-1]
        return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))

    @property
    def angle(self) -> float:
        """End-to-end direction, 0-180° counter-clockwise from the x axis (image y points down)."""
        (x1, y1), (x2, y2) = self.endpoints
        return math.degrees(math.atan2(-(y2 - y1), x2 - x1)) % 180.0


def skeleton_nodes(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(end points, branch points) of a skeleton as bool images."""
    skel = np.asarray(skeleton, bool)
    count = cv2.filter2D(skel.astype(np.float32), -1, _NEIGHBOURS, borderType=cv2.BORDER_CONSTANT)
    return skel & (count == 1), skel & (count >= 3)


def _path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.hypot(*np.diff(path.astype(float), axis=0).T).sum())


def find_ribbons(mask: np.ndarray, px_scale: Optional[float] = None, min_length_px: float = 3.0,
                 split_at_junctions: bool = True, skeleton: Optional[np.ndarray] = None,
                 components: Optional[tuple] = None) -> list[Ribbon]:
    """
    Ribbon segments of a bool mask.  ``skeleton`` / ``components`` (a
    :func:`scanbound.roi.mask_components` result) can be passed in to avoid recomputing them.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if skeleton is None:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask)
    skel = np.asarray(skeleton, bool) & mask
    if components is None:
        _, objects, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    else:
        objects = components[1]
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]

    pieces = skel
    if split_at_junctions:
        _, junctions = skeleton_nodes(skel)
        pieces = skel & ~junctions
    labels, n = ndimage.label(pieces, structure=np.ones((3, 3), int))

    ribbons: list[Ribbon] = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        crop = labels[sl] == i
        path = longest_skeleton_path(crop)
        if path is None:
            continue
        oy, ox = sl[0].start, sl[1].start
        xs = path[:, 0].astype(int) + ox
        ys = path[:, 1].astype(int) + oy
        length = _path_length(path)
        if length < min_length_px:
            continue
        if path[0, 0] > path[-1, 0] or (path[0, 0] == path[-1, 0] and path[0, 1] > path[-1, 1]):
            path = path[::-1]
        width = max(1.0, 2.0 * float(dist[ys, xs].mean()) - 1.0)
        pts = (path + np.array([ox + 0.5, oy + 0.5], np.float32)).astype(np.float32)
        ribbons.append(Ribbon(len(ribbons) + 1, int(objects[ys[0], xs[0]]), pts, length, width, px_scale))
    return ribbons


def ribbons_table(ribbons: list[Ribbon]) -> pd.DataFrame:
    """One row per ribbon segment (index = ribbon id); nm columns when a pixel scale is known."""
    rows = []
    for r in ribbons:
        (x1, y1), (x2, y2) = r.endpoints
        row = {"id": r.id, "object_id": r.object_id, "length_px": r.length_px, "width_px": r.width_px,
               "angle": r.angle, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "n_points": len(r.points)}
        if r.px_scale:
            row["length_nm"] = r.length_px * r.px_scale
            row["width_nm"] = r.width_px * r.px_scale
        rows.append(row)
    cols = ["id", "object_id", "length_px", "width_px", "angle", "x1", "y1", "x2", "y2", "n_points"]
    if any(r.px_scale for r in ribbons):
        cols += ["length_nm", "width_nm"]
    return pd.DataFrame(rows, columns=cols).set_index("id")
