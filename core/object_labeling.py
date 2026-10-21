"""
object_labeling.py
==================

Shape measurements of segmented objects and rules that turn them into labels.  No Qt
dependency; the dialog side is core/label_assignment_panel.py.

    feats = measure_objects(mask, objects, px_scale=0.2, intensity=image)   # DataFrame, index = object id
    labels = assign_labels(feats, config)                                     # {object id: label or None}

Assignment modes (``config["mode"]``)
------------------------------------
single    every object gets ``single_label``.
rules     ordered rules ``{"label", "conditions": [{"feature", "min", "max"}]}``; the first rule
          whose conditions all hold wins, other objects get ``fallback`` (None = no ROI).
bins      one feature split by ascending ``thresholds`` into classes, ``labels[i]`` per class.
clusters  k-means on the chosen ``features`` (standardised, optionally log-scaled); groups are
          numbered by increasing value of the first feature, ``labels[i]`` per group.

A label of None means "skip": no ROI is created for the object.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

try:
    from core.roi_extraction import _longest_skeleton_path
except ImportError:                       # used as a sibling module
    from roi_extraction import _longest_skeleton_path

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


def feature_unit(key, px_scale):
    kind = FEATURES[key][1]
    if kind == "deg":
        return "°"
    if kind is None:
        return ""
    base = "nm" if px_scale else "px"
    return f"{base}²" if kind == "area" else base


def feature_title(key, px_scale=None):
    unit = feature_unit(key, px_scale)
    title = FEATURES[key][0]
    return f"{title} ({unit})" if unit and unit != "°" else title


# =============================================================================
# Measurements
# =============================================================================

def _centre_line_px(obj):
    """Path length of a LineSegmentROI's points (pixels), else None."""
    if obj.roi_type != "LineSegmentROI" or obj.points is None or len(obj.points) < 2:
        return None
    return float(np.hypot(*np.diff(np.asarray(obj.points, float), axis=0).T).sum())


def measure_objects(mask, objects, px_scale=None, intensity=None) -> pd.DataFrame:
    """
    One row per object (index = SegObject.id), measured on the object's own pixels (not on the
    padded ROI).  Lengths are in nm and areas in nm² when ``px_scale`` (nm per pixel) is given.
    """
    mask = np.asarray(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
            path = _longest_skeleton_path(skeleton[y:y + h, x:x + w] & crop)
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


# =============================================================================
# Assignment
# =============================================================================

def default_config(label=None):
    return {"mode": "single", "single_label": label,
            "rules": [], "fallback": None,
            "bins": {"feature": "area", "thresholds": [], "labels": [label], "log": True},
            "clusters": {"features": ["area", "aspect_ratio"], "k": 2, "labels": [label, label], "log": True}}


def _passes(row, cond):
    v = row.get(cond.get("feature"))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    lo, hi = cond.get("min"), cond.get("max")
    return (lo is None or v >= lo) and (hi is None or v <= hi)


def bin_index(values, thresholds):
    """Class index per value: 0 below thresholds[0], ..., len(thresholds) above the last."""
    return np.searchsorted(np.sort(np.asarray(thresholds, float)), np.asarray(values, float), side="right")


def suggest_thresholds(values, n_classes, method="quantiles", log=False):
    """n_classes - 1 ascending thresholds: equal-count quantiles or natural breaks (1D k-means)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n_classes = int(max(2, n_classes))
    if v.size < n_classes:
        return []
    t = np.log(np.clip(v, 1e-12, None)) if log else v
    if method == "natural breaks":
        centres = cluster_values(t[:, None], n_classes)[1].ravel()
        centres = np.sort(centres)
        cuts = (centres[:-1] + centres[1:]) / 2
    else:
        cuts = np.quantile(t, np.linspace(0, 1, n_classes + 1)[1:-1])
    cuts = np.exp(cuts) if log else cuts
    return [float(c) for c in np.unique(cuts)]


def cluster_values(x, k, seed=0):
    """k-means: (labels, centres).  x is (n, d), already scaled."""
    from sklearn.cluster import KMeans
    k = int(min(max(1, k), len(x)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    return km.labels_, km.cluster_centers_


def cluster_groups(features: pd.DataFrame, feature_keys, k, log=True):
    """
    Group index per object (Series aligned with ``features``), numbered by increasing mean of the
    first feature so the numbering is stable between runs.
    """
    keys = [f for f in feature_keys if f in features.columns]
    if not keys or features.empty:
        return pd.Series(np.zeros(len(features), int), index=features.index)
    x = features[keys].to_numpy(float)
    ok = np.isfinite(x).all(axis=1)
    groups = np.full(len(features), -1, int)
    if ok.sum() == 0:
        return pd.Series(groups, index=features.index)
    xs = x[ok].copy()
    for j, key in enumerate(keys):
        if log and key in SCALE_FEATURES:
            xs[:, j] = np.log(np.clip(xs[:, j], 1e-12, None))
    sd = xs.std(axis=0)
    xs = (xs - xs.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    raw, _ = cluster_values(xs, k)
    order = np.argsort([x[ok][raw == g, 0].mean() for g in range(raw.max() + 1)])
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    groups[ok] = rank[raw]
    return pd.Series(groups, index=features.index)


def assign_groups(features: pd.DataFrame, config):
    """(group per object, list of labels per group) for the current mode."""
    mode = config.get("mode", "single")
    idx = features.index
    if mode == "rules":
        rules = config.get("rules") or []
        groups = pd.Series(len(rules), index=idx)          # len(rules) = fallback group
        for oid, row in features.iterrows():
            r = row.to_dict()
            for g, rule in enumerate(rules):
                conds = rule.get("conditions") or []
                if all(_passes(r, c) for c in conds):
                    groups[oid] = g
                    break
        return groups, [r.get("label") for r in rules] + [config.get("fallback")]
    if mode == "bins":
        b = config.get("bins") or {}
        thresholds = sorted(b.get("thresholds") or [])
        labels = list(b.get("labels") or [])
        labels += [labels[-1] if labels else None] * (len(thresholds) + 1 - len(labels))
        col = b.get("feature", "area")
        values = features[col] if col in features else pd.Series(np.nan, index=idx)
        groups = pd.Series(bin_index(values.fillna(-np.inf), thresholds), index=idx)
        return groups, labels[:len(thresholds) + 1]
    if mode == "clusters":
        c = config.get("clusters") or {}
        k = int(c.get("k", 2))
        groups = cluster_groups(features, c.get("features") or [], k, bool(c.get("log", True)))
        labels = list(c.get("labels") or [])
        labels += [labels[-1] if labels else None] * (k - len(labels))
        return groups, labels[:k]
    return pd.Series(0, index=idx), [config.get("single_label")]


def assign_labels(features: pd.DataFrame, config) -> dict:
    """{object id: label or None}."""
    groups, labels = assign_groups(features, config)
    out = {}
    for oid, g in groups.items():
        out[oid] = labels[g] if 0 <= g < len(labels) else None
    return out
