"""
scanbound.measure
=================

Shape measurements of segmented objects (one row per object, on the object's own pixels).

    feats = measure_objects(mask, objects, px_scale=0.2, intensity=image)   # DataFrame, index = object id
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .roi import SegObject, longest_skeleton_path

# key -> (title, unit kind): "len" / "area" scale with the pixel size, None = dimensionless
FEATURES = OrderedDict([
    ("area", ("Area", "area")),
    ("eq_diameter", ("Equivalent diameter", "len")),
    ("length", ("Length (long side)", "len")),
    ("width", ("Width (short side)", "len")),
    ("aspect_ratio", ("Aspect ratio", None)),
    ("perimeter", ("Perimeter", "len")),
    ("circularity", ("Circularity", None)),
    ("solidity", ("Solidity", None)),
    ("extent", ("Extent (fill of bounding box)", None)),
    ("orientation", ("Orientation (°)", "deg")),
    ("centre_line", ("Centre-line length", "len")),
    ("mean_intensity", ("Mean intensity", None)),
])
# Features that span orders of magnitude (log scaling makes clusters / classes sensible)
SCALE_FEATURES = {"area", "eq_diameter", "length", "width", "perimeter", "centre_line", "aspect_ratio"}

FEATURE_HELP = {
    "area": "Number of object pixels (× pixel area).",
    "eq_diameter": "Diameter of a circle with the same area.",
    "length": "Long side of the minimum-area rotated rectangle around the object.",
    "width": "Short side of the minimum-area rotated rectangle around the object.",
    "aspect_ratio": "Length / width: 1 for round or square objects, large for rods.",
    "perimeter": "Length of the outline.",
    "circularity": "4π·area / perimeter²: 1 for a disc, small for elongated or ragged objects.",
    "solidity": "Area / convex-hull area: 1 for convex objects, small for branched or curved ones.",
    "extent": "Area / bounding-box area.",
    "orientation": "Angle of the long axis, 0–180° counter-clockwise from the x axis.",
    "centre_line": "Length of the object's centre line (skeleton path): the contour length of curved rods.",
    "mean_intensity": "Mean of the (normalised 0–1) image over the object.",
}


def feature_unit(key: str, px_scale: Optional[float]) -> str:
    kind = FEATURES[key][1]
    if kind == "deg":
        return "°"
    if kind is None:
        return ""
    base = "nm" if px_scale else "px"
    return f"{base}²" if kind == "area" else base


def feature_title(key: str, px_scale: Optional[float] = None) -> str:
    unit = feature_unit(key, px_scale)
    title = FEATURES[key][0]
    return f"{title} ({unit})" if unit and unit != "°" else title


# =============================================================================
# Measurements
# =============================================================================

def _centre_line_px(obj: SegObject) -> Optional[float]:
    """Path length of a LineSegmentROI's points (pixels), else None."""
    if obj.roi_type != "LineSegmentROI" or obj.points is None or len(obj.points) < 2:
        return None
    return float(np.hypot(*np.diff(np.asarray(obj.points, float), axis=0).T).sum())


def measure_objects(mask: np.ndarray, objects: Sequence[SegObject], px_scale: Optional[float] = None,
                    intensity: Optional[np.ndarray] = None, components: Optional[tuple] = None) -> pd.DataFrame:
    """
    One row per object (index = SegObject.id), measured on the object's own pixels (not on the
    padded ROI).  Lengths are in nm and areas in nm² when ``px_scale`` (nm per pixel) is given.
    ``components`` lets a caller pass a ``mask_components(mask)`` result it already has.
    """
    mask = np.asarray(mask).astype(np.uint8)
    n, labels, stats, _ = components if components is not None else \
        cv2.connectedComponentsWithStats(mask, connectivity=8)
    s = float(px_scale) if px_scale else 1.0
    skeleton = None
    rows = []
    for obj in objects:
        i = obj.id
        if not 0 < i < n:
            continue
        x, y, w, h, area = (int(v) for v in stats[i])
        crop = (labels[y:y + h, x:x + w] == i)
        crop_u8 = crop.astype(np.uint8)
        contours, _ = cv2.findContours(np.pad(crop_u8, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea) if contours else None

        # outline around pixel squares: perimeter of the pixel-centre contour + one pixel per turn
        if cnt is not None and len(cnt) >= 3:
            # smooth the pixel staircase (it overestimates curved outlines by ~8 %), then add the
            # half pixel all round between pixel-centre contour and pixel edges
            smooth = cv2.approxPolyDP(cnt, 0.75, True)
            perimeter = cv2.arcLength(smooth if len(smooth) >= 3 else cnt, True) + math.pi
            hull_area = cv2.contourArea(cv2.convexHull(cnt)) + cv2.arcLength(cv2.convexHull(cnt), True) / 2 + 1
            solidity = min(1.0, area / hull_area) if hull_area > 0 else 1.0
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt.astype(np.float32))
            length, width = max(rw, rh) + 1.0, min(rw, rh) + 1.0
        else:
            perimeter, solidity = 4.0 * max(area, 1) ** 0.5, 1.0
            length, width = float(max(w, h)), float(min(w, h))

        ys, xs = np.nonzero(crop)
        if len(xs) >= 2:
            cov = np.cov(np.stack([xs, ys]).astype(float))
            vals, vecs = np.linalg.eigh(cov)
            vx, vy = vecs[:, int(np.argmax(vals))]
            orientation = math.degrees(math.atan2(-vy, vx)) % 180.0     # image y points down
        else:
            orientation = 0.0

        centre = _centre_line_px(obj)
        if centre is None:
            if skeleton is None:
                from skimage.morphology import skeletonize
                skeleton = skeletonize(mask.astype(bool))
            # same definition as the LineSegmentROI centre line: longest geodesic skeleton path
            path = longest_skeleton_path(skeleton[y:y + h, x:x + w] & crop)
            centre = 1.0
            if path is not None and len(path) > 1:
                centre = max(1.0, float(np.hypot(*np.diff(path.astype(float), axis=0).T).sum()))

        mean_int = float(np.asarray(intensity)[y:y + h, x:x + w][crop].mean()) if intensity is not None else np.nan
        rows.append({
            "id": i, "area": area * s * s, "eq_diameter": 2 * math.sqrt(area / math.pi) * s,
            "length": length * s, "width": width * s, "aspect_ratio": length / max(width, 1e-9),
            "perimeter": perimeter * s, "circularity": min(1.0, 4 * math.pi * area / perimeter ** 2),
            "solidity": solidity, "extent": area / float(w * h), "orientation": orientation,
            "centre_line": centre * s, "mean_intensity": mean_int,
        })
    df = pd.DataFrame(rows, columns=["id"] + list(FEATURES))
    return df.set_index("id")
