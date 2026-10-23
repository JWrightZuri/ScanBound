"""
roi_analysis.py
===============

Statistics and charts for ROIs, per label, for one image or a set of images.
No Qt dependency: the same code runs inside ScanBound (core/roi_analysis_dialog.py)
or from a script / the command line on a saved annotation CSV.

Input
-----
* an annotation table in the ScanBound format
  (objectName, type, x1, y1, dx, dy, file, label; one row per vertex for polygons / lines)
* ``ImageInfo`` (width, height, pixel scale in nm/px) for every image in scope.  Images
  without ROIs still count (e.g. for coverage and density averages).

Every ROI is measured once (``AnalysisContext.objects``): area, perimeter, length / width
(minimum-area bounding rectangle, path length for lines), aspect ratio and equivalent
diameter, in nm when every image in scope has a pixel scale and in px otherwise.

Built-in analyses
-----------------
composition   How many ROIs of each label (overall, per image, per ROI type) and their density.
coverage      Fraction of the image area covered by each label (overlaps counted once).
size          Sizes and measurements: statistics per label, histograms, box plots and a table
              with one row per ROI.

Adding an analysis
------------------
Write a function that takes the context (plus keyword parameters) and returns an
``AnalysisResult``; register it with the decorator.  The dialog lists it automatically,
builds widgets for its parameters and shows / exports every table and figure it returns::

    @register_analysis(
        "orientation", "Orientation",
        "Angle distribution of the long axis per label.",
        params=[Param("bins", "Histogram bins", "int", 18, 2, 180)],
    )
    def orientation(ctx: AnalysisContext, bins: int) -> AnalysisResult:
        res = AnalysisResult()
        res.tables["Angles"] = ctx.objects[["file", "label", "length"]]
        fig, ax = ctx.figure()
        ...
        res.figures["Angle histogram"] = fig
        return res

Useful context members: ``objects`` (DataFrame, one row per ROI), ``shapes`` (geometry of
each ROI, aligned with ``objects``), ``labels``, ``files``, ``unit`` / ``area_unit``,
``image_area(file)``, ``coverage_mask(file, labels, line_width)``, ``color(label)``,
``figure()``.

Command line
------------
    python -m scanbound.analysis.roi annotations.csv --images IMAGE_DIR [--metadata metadata.csv]
                                [--out OUT_DIR] [--analyses composition coverage size]
"""

from __future__ import annotations

import math
import os
import re
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from skimage.draw import polygon as draw_polygon

EMPTY_LABEL = "Empty Label"
SYMMETRY_LABEL = "Symmetry Vector"
AREA_TYPES = ("RectROI", "EllipseROI", "PolygonROI", "MaskROI")
LINE_TYPES = ("LineSegmentROI", "VectorROI")
VERTEX_TYPES = ("LineSegmentROI", "PolygonROI", "MaskROI")

_FALLBACK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# =============================================================================
# Inputs, results and the analysis registry
# =============================================================================

@dataclass
class ImageInfo:
    file: str                           # basename, as in the annotation 'file' column
    width: int
    height: int
    px_scale: Optional[float] = None    # nm per pixel (None = unknown)
    ref_angle: Optional[float] = None   # reference direction for orientations, degrees (None = x axis)
    image: object = None                # 2D array or a callable returning it (orientation map background)
    marks: dict = field(default_factory=dict)   # kind -> [((x0, y0), (x1, y1)), ...] e.g. herringbone axes


@dataclass
class AnalysisData:
    """Everything an analysis run needs: annotation rows, image info and label colours."""
    annotations: pd.DataFrame
    images: dict                        # file -> ImageInfo
    colors: dict = field(default_factory=dict)   # label -> colour (any matplotlib colour)
    masks: dict = field(default_factory=dict)    # (file, objectName) -> (x0, y0, bits) of each MaskROI


@dataclass
class AnalysisResult:
    tables: dict = field(default_factory=dict)    # name -> DataFrame
    figures: dict = field(default_factory=dict)   # name -> matplotlib Figure
    notes: list = field(default_factory=list)     # short remarks shown above the results


@dataclass
class Param:
    name: str                       # keyword argument of the analysis function
    label: str                      # text shown in the dialog
    kind: str                       # "int", "float", "bool" or "choice"
    default: object
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: tuple = ()
    tooltip: str = ""


@dataclass
class Analysis:
    key: str
    title: str
    description: str
    func: Callable
    params: list


ANALYSES: dict = {}


def register_analysis(key, title, description="", params=()):
    """Decorator that adds an analysis function to the registry (see module docstring)."""
    def deco(func):
        ANALYSES[key] = Analysis(key, title, description, func, list(params))
        return func
    return deco


# =============================================================================
# Geometry and measurements
# =============================================================================

def _clean_label(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return EMPTY_LABEL
    text = str(value)
    return text if text and text.lower() != "nan" else EMPTY_LABEL


def _clean_name(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return text if text and text.lower() not in ("nan", "none") else None


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def _valid_scale(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def roi_shapes(annotations: pd.DataFrame, masks: Optional[dict] = None) -> list:
    """
    One dict per ROI: file, objectName, type, label, pts ((N, 2) float array, scene/pixel
    coordinates).  Vertex ROIs are grouped by (file, objectName) like the viewer does.  A MaskROI
    found in ``masks`` ((file, objectName) -> (x0, y0, bits)) also gets ``mask`` (its bool pixels)
    and ``offset`` ((x0, y0) of the mask's top-left pixel); its pts are the outer boundary.
    """
    shapes = []
    groups = {}
    if annotations is None or annotations.empty:
        return shapes
    has_name = "objectName" in annotations.columns
    has_label = "label" in annotations.columns
    has_angle = "angle" in annotations.columns
    for i, rec in enumerate(annotations.itertuples(index=False)):
        roi_type = str(rec.type).strip()
        if roi_type == "LineROI":                 # legacy name
            roi_type = "VectorROI"
        file = str(rec.file)
        label = _clean_label(rec.label) if has_label else EMPTY_LABEL
        name = _clean_name(rec.objectName) if has_name else None
        x1, y1, dx, dy = _num(rec.x1), _num(rec.y1), _num(rec.dx), _num(rec.dy)
        angle = float("nan")
        if has_angle:
            try:
                angle = float(rec.angle)
            except (TypeError, ValueError):
                angle = float("nan")
        if roi_type in VERTEX_TYPES:
            group = groups.setdefault((file, roi_type, name), {
                "file": file, "objectName": name, "type": roi_type, "label": label, "pts": [], "angle": angle})
            group["pts"].append((x1, y1))
            continue
        if roi_type == "RectROI":
            xa, xb = sorted((x1, x1 + dx))
            ya, yb = sorted((y1, y1 + dy))
            pts = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
        elif roi_type == "EllipseROI":
            t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            cx, cy = x1 + dx / 2, y1 + dy / 2
            pts = np.stack([cx + abs(dx) / 2 * np.cos(t), cy + abs(dy) / 2 * np.sin(t)], axis=1)
        elif roi_type == "PointROI":
            pts = [(x1, y1)]
        elif roi_type == "VectorROI":
            pts = [(x1, y1), (dx, dy)]            # dx, dy hold the end point
            if not math.isfinite(angle):          # a vector's orientation is its own direction
                angle = math.degrees(math.atan2(-(dy - y1), dx - x1)) % 360.0
        else:
            continue
        shapes.append({"file": file, "objectName": name or f"row{i}", "type": roi_type, "label": label,
                       "pts": np.asarray(pts, dtype=float).reshape(-1, 2), "angle": angle})
    for (file, roi_type, name), group in groups.items():
        pts = np.asarray(group["pts"], dtype=float).reshape(-1, 2)
        if len(pts) >= (2 if roi_type == "LineSegmentROI" else 3):
            group["pts"] = pts
            group["objectName"] = name or f"{roi_type}@{file}"
            if roi_type == "MaskROI" and masks and (file, name) in masks:
                x0, y0, bits = masks[(file, name)]
                group["mask"], group["offset"] = np.asarray(bits, bool), (int(x0), int(y0))
            shapes.append(group)
    return shapes


def shape_axis_angle(roi_type: str, pts: np.ndarray) -> float:
    """
    Long axis of a shape, degrees in [0, 180) (screen convention, see core/orientation.py):
    principal axis of the outline for areas, end-to-end direction for lines, NaN for points.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if roi_type in LINE_TYPES and len(pts) >= 2:
        d = pts[-1] - pts[0]
        return float(np.degrees(np.arctan2(-d[1], d[0])) % 180.0)
    if roi_type not in AREA_TYPES or len(pts) < 3:
        return float("nan")
    # sample the outline uniformly so the vertex density does not bias the axis
    closed = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    if seg.sum() <= 0:
        return float("nan")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], 200, endpoint=False)
    xy = np.stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])], axis=1)
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    if abs(vals[1] - vals[0]) < 1e-9 * max(vals[1], 1e-12):
        return float("nan")                    # round / square: no defined axis
    vx, vy = vecs[:, int(np.argmax(vals))]
    return float(np.degrees(np.arctan2(-vy, vx)) % 180.0)


def measure_shape(roi_type: str, pts: np.ndarray) -> dict:
    """Pixel measurements of one ROI (NaN where a measure does not apply)."""
    nan = float("nan")
    out = {"n_vertices": len(pts), "cx_px": nan, "cy_px": nan, "area_px": nan, "perimeter_px": nan,
           "length_px": nan, "width_px": nan}
    if len(pts) == 0:
        return out
    out["cx_px"], out["cy_px"] = (float(v) for v in pts.mean(axis=0))
    if roi_type in AREA_TYPES and len(pts) >= 3:
        x, y = pts[:, 0], pts[:, 1]
        out["area_px"] = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        closed = np.vstack([pts, pts[:1]])
        out["perimeter_px"] = float(np.hypot(*np.diff(closed, axis=0).T).sum())
        (_, _), (w, h), _ = cv2.minAreaRect(pts.astype(np.float32))
        out["length_px"], out["width_px"] = float(max(w, h)), float(min(w, h))
    elif roi_type in LINE_TYPES and len(pts) >= 2:
        out["length_px"] = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    return out


# Display name, column, unit kind ("len" / "area" / None) of the size metrics
SIZE_METRICS = [
    ("Area", "area", "area"),
    ("Equivalent diameter", "eq_diameter", "len"),
    ("Length", "length", "len"),
    ("Width", "width", "len"),
    ("Aspect ratio", "aspect_ratio", None),
    ("Perimeter", "perimeter", "len"),
    ("Skeleton length", "skeleton_length", "len"),
    ("Skeleton width", "skeleton_width", "len"),
]


# =============================================================================
# Context
# =============================================================================

class AnalysisContext:
    """Measured ROIs of the images in scope plus helpers shared by all analyses."""

    def __init__(self, data: AnalysisData, labels=None, include_unlabeled=False, exclude_border=False):
        self.images = {f: replace(info, width=int(info.width), height=int(info.height),
                                  px_scale=_valid_scale(info.px_scale))
                       for f, info in data.images.items()}
        self.files = list(self.images)
        self._colors = dict(data.colors or {})
        self.notes = []

        scaled = [f for f in self.files if self.images[f].px_scale]
        self.unit = "nm" if self.files and len(scaled) == len(self.files) else "px"
        self.area_unit = f"{self.unit}²"
        if self.unit == "px" and scaled:
            missing = [f for f in self.files if not self.images[f].px_scale]
            self.notes.append("Results are in pixels because no pixel scale is set for: "
                              + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))

        annotations = data.annotations
        if annotations is not None and not annotations.empty and self.files:
            annotations = annotations[annotations["file"].astype(str).isin(self.files)]
        shapes = roi_shapes(annotations, getattr(data, "masks", None))
        if labels is not None:
            wanted = set(labels)
            shapes = [s for s in shapes if s["label"] in wanted]
        elif not include_unlabeled:
            shapes = [s for s in shapes if s["label"] != EMPTY_LABEL]
        if exclude_border:
            kept = [s for s in shapes if not self.touches_border(s)]
            if len(kept) < len(shapes):
                self.notes.append(f"{len(shapes) - len(kept)} ROI(s) touching the image border left out.")
            shapes = kept
        self.shapes = shapes
        self.objects = self._measure(shapes)

        counts = self.objects["label"].value_counts() if len(self.objects) else pd.Series(dtype=int)
        self.labels = sorted(counts.index, key=lambda l: (-counts[l], str(l)))
        self._fallback = {}
        self._masks = {}

    def touches_border(self, shape, tolerance=1.0) -> bool:
        """True when a vertex of the ROI lies within ``tolerance`` pixels of its image's edge
        (such objects are usually cut off, so their size and shape are not their own)."""
        info = self.images.get(shape["file"])
        pts = shape["pts"]
        if info is None or len(pts) == 0:
            return False
        x, y = pts[:, 0], pts[:, 1]
        return bool(x.min() <= tolerance or y.min() <= tolerance
                    or x.max() >= info.width - tolerance or y.max() >= info.height - tolerance)

    # ------------------------------------------------------------------ measurements

    def scale(self, file) -> float:
        """Length of one pixel in the display unit."""
        if self.unit == "nm":
            return float(self.images[file].px_scale)
        return 1.0

    def image_area(self, file) -> float:
        info = self.images[file]
        return info.width * info.height * self.scale(file) ** 2

    def total_area(self) -> float:
        return float(sum(self.image_area(f) for f in self.files))

    def _measure(self, shapes):
        rows = []
        for s in shapes:
            m = measure_shape(s["type"], s["pts"])
            if s.get("mask") is not None:                    # a mask's pixels: exact area, perimeter, centre
                from ..annotations.geometry import Mask
                g = Mask(*s["offset"], s["mask"])
                m["area_px"], m["perimeter_px"] = g.area(), g.length()
                m["cx_px"], m["cy_px"] = g.centroid()
            rows.append({"file": s["file"], "objectName": s["objectName"], "type": s["type"],
                         "label": s["label"], **m, "orientation": s.get("angle", np.nan),
                         "axis_angle": shape_axis_angle(s["type"], s["pts"])})
        cols = ["file", "objectName", "type", "label", "n_vertices", "cx_px", "cy_px", "area_px",
                "perimeter_px", "length_px", "width_px", "orientation", "axis_angle"]
        df = pd.DataFrame(rows, columns=cols)
        scale = df["file"].map(lambda f: self.scale(f) if f in self.images else 1.0).astype(float)
        df["area"] = df["area_px"] * scale ** 2
        df["perimeter"] = df["perimeter_px"] * scale
        df["length"] = df["length_px"] * scale
        df["width"] = df["width_px"] * scale
        df["eq_diameter"] = 2.0 * np.sqrt(df["area"] / np.pi)
        df["aspect_ratio"] = df["length"] / df["width"].where(df["width"] > 0)
        return df

    def metric_unit(self, kind):
        return {"len": self.unit, "area": self.area_unit}.get(kind, "")

    # ------------------------------------------------------------------ coverage

    def coverage_mask(self, file, labels=None, line_width=0.0):
        """
        Boolean mask (supersampled image grid) of the area covered by the ROIs of `labels`
        (None = all labels in scope) in `file`, and the supersampling factor.  Area ROIs are
        filled; lines are drawn `line_width` wide (display units) when line_width > 0.
        """
        mask = None
        factor = 1
        for label in (self.labels if labels is None else labels):
            label_mask, factor = self._label_mask(file, label, float(line_width))
            mask = label_mask.copy() if mask is None else (mask | label_mask)
        if mask is None:
            info = self.images[file]
            mask = np.zeros((info.height, info.width), bool)
        return mask, factor

    def _label_mask(self, file, label, line_width):
        key = (file, label, line_width)
        if key in self._masks:
            return self._masks[key]
        info = self.images[file]
        factor = 2 if max(info.width, info.height) <= 2048 else 1     # 2x supersampling: <0.1 % area error
        mask = np.zeros((info.height * factor, info.width * factor), np.uint8)
        lines = []
        for s in self.shapes:
            if s["file"] != file or s["label"] != label:
                continue
            # Scene x covers pixel [x, x+1]: a mask pixel is covered when its centre is inside.
            pts = s["pts"] * factor - 0.5
            if s.get("mask") is not None:                     # the pixels themselves
                x0, y0 = s["offset"]
                bits = np.kron(s["mask"], np.ones((factor, factor), bool)) if factor > 1 else s["mask"]
                h, w = bits.shape
                r0, c0 = y0 * factor, x0 * factor
                r1, c1 = min(r0 + h, mask.shape[0]), min(c0 + w, mask.shape[1])
                rs, cs = max(r0, 0), max(c0, 0)
                if r1 > rs and c1 > cs:
                    mask[rs:r1, cs:c1] |= bits[rs - r0:r1 - r0, cs - c0:c1 - c0].astype(np.uint8)
            elif s["type"] in AREA_TYPES and len(pts) >= 3:
                rr, cc = draw_polygon(pts[:, 1], pts[:, 0], mask.shape)
                mask[rr, cc] = 1
            elif s["type"] in LINE_TYPES and line_width > 0 and len(pts) >= 2:
                lines.append(np.round(pts * 16).astype(np.int32))     # 4 fractional bits
        if lines:
            thickness = max(1, int(round(line_width / self.scale(file) * factor)))
            cv2.polylines(mask, lines, False, 1, thickness=thickness, lineType=cv2.LINE_8, shift=4)
        result = (mask.astype(bool), factor)
        self._masks[key] = result
        return result

    def image_array(self, file):
        """Grayscale image of `file` for map backgrounds, or None (loaded once, on demand)."""
        cache = self.__dict__.setdefault("_image_cache", {})
        if file not in cache:
            img = self.images[file].image if file in self.images else None
            try:
                img = img() if callable(img) else img
                img = None if img is None else np.asarray(img, dtype=float)
                if img is not None and img.ndim == 3:
                    img = img[..., :3].mean(axis=2)
            except Exception:
                img = None
            cache[file] = img
        return cache[file]

    # ------------------------------------------------------------------ plotting helpers

    def color(self, label):
        if label in self._colors:
            return self._colors[label]
        if label not in self._fallback:
            self._fallback[label] = _FALLBACK_COLORS[len(self._fallback) % len(_FALLBACK_COLORS)]
        return self._fallback[label]

    @staticmethod
    def figure(nrows=1, ncols=1, width=7.0, height=4.5, **kwargs):
        """A new (pyplot-free) matplotlib Figure and its axes."""
        fig = Figure(figsize=(width, height), layout="constrained")
        axes = fig.subplots(nrows, ncols, **kwargs)
        return fig, axes


# =============================================================================
# Built-in analyses
# =============================================================================

def _density_unit(ctx):
    return "µm²" if ctx.unit == "nm" else "10⁶ px²"


@register_analysis(
    "composition", "Label composition",
    "Number of ROIs per label (overall, per image and per ROI type), their share and their density.",
)
def label_composition(ctx: AnalysisContext) -> AnalysisResult:
    res = AnalysisResult()
    obj = ctx.objects
    if obj.empty:
        res.notes.append("No ROIs in scope.")
        return res

    total_area = ctx.total_area()
    counts = obj["label"].value_counts().reindex(ctx.labels)
    summary = pd.DataFrame({
        "label": ctx.labels,
        "count": counts.values,
        "fraction (%)": counts.values / counts.sum() * 100.0,
        f"density (per {_density_unit(ctx)})": counts.values / total_area * 1e6 if total_area else np.nan,
    })
    if len(ctx.files) > 1:
        per_image = obj.groupby(["file", "label"]).size().unstack(fill_value=0)
        per_image = per_image.reindex(index=ctx.files, columns=ctx.labels, fill_value=0)
        summary["mean per image"] = per_image.mean().values
        summary["std per image"] = per_image.std(ddof=1).values
    total_row = {"label": "All labels", "count": int(counts.sum()), "fraction (%)": 100.0,
                 f"density (per {_density_unit(ctx)})": counts.sum() / total_area * 1e6 if total_area else np.nan}
    if len(ctx.files) > 1:
        totals = per_image.sum(axis=1)
        total_row.update({"mean per image": totals.mean(), "std per image": totals.std(ddof=1)})
    res.tables["Summary"] = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    by_type = obj.groupby(["label", "type"]).size().unstack(fill_value=0).reindex(ctx.labels)
    res.tables["By ROI type"] = by_type.reset_index()

    if len(ctx.files) > 1:
        res.tables["Per image"] = per_image.assign(total=per_image.sum(axis=1)).reset_index()

    # charts
    fig, ax = ctx.figure(width=7, height=max(2.5, 0.45 * len(ctx.labels) + 1.2))
    y = np.arange(len(ctx.labels))
    ax.barh(y, counts.values, color=[ctx.color(l) for l in ctx.labels], edgecolor="black", linewidth=0.5)
    ax.set_yticks(y, [str(l) for l in ctx.labels])
    ax.invert_yaxis()
    for yi, n in zip(y, counts.values):
        ax.annotate(f" {n} ({n / counts.sum() * 100:.1f}%)", (n, yi), va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_title(f"Label composition ({int(counts.sum())} ROIs, {len(ctx.files)} image(s))")
    res.figures["Counts"] = fig

    if len(ctx.files) > 1:
        frac = per_image.div(per_image.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        fig, ax = ctx.figure(width=max(6, 0.5 * len(ctx.files) + 3), height=4.5)
        bottom = np.zeros(len(ctx.files))
        x = np.arange(len(ctx.files))
        for label in ctx.labels:
            ax.bar(x, frac[label].values, bottom=bottom, color=ctx.color(label), label=str(label),
                   edgecolor="black", linewidth=0.3)
            bottom += frac[label].values
        ax.set_xticks(x, [os.path.splitext(f)[0] for f in ctx.files], rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Share of ROIs (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Composition per image")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        res.figures["Composition per image"] = fig
    return res


@register_analysis(
    "coverage", "Coverage",
    "Fraction of the image area covered by each label (overlapping ROIs counted once). "
    "Rectangles, ellipses and polygons count; lines only with a line width > 0.",
    params=[
        Param("line_width", "Line width (nm or px)", "float", 0.0, 0.0, 1000.0,
              tooltip="Width given to line segments / vectors for coverage; 0 ignores them."),
    ],
)
def coverage(ctx: AnalysisContext, line_width: float = 0.0) -> AnalysisResult:
    res = AnalysisResult()
    obj = ctx.objects
    if not ctx.files:
        res.notes.append("No images in scope.")
        return res
    counted_types = set(AREA_TYPES) | (set(LINE_TYPES) if line_width > 0 else set())
    ignored = obj[~obj["type"].isin(counted_types)]
    if len(ignored):
        res.notes.append(f"{len(ignored)} ROI(s) without area are not counted "
                         f"({', '.join(sorted(ignored['type'].unique()))}).")
    labels = [l for l in ctx.labels if obj.loc[obj["label"] == l, "type"].isin(counted_types).any()]
    if not labels:
        res.notes.append("No ROIs with an area in scope.")
        return res

    rows = []
    for file in ctx.files:
        area = ctx.image_area(file)
        for label in labels + [None]:
            mask, _ = ctx.coverage_mask(file, None if label is None else [label], line_width)
            sel = obj[(obj["file"] == file) & obj["type"].isin(counted_types)]
            if label is not None:
                sel = sel[sel["label"] == label]
            covered = float(mask.mean()) * area
            summed = float(sel["area"].fillna(0).sum())
            rows.append({"file": file, "label": "All labels (union)" if label is None else label,
                         "n_rois": len(sel), f"covered area ({ctx.area_unit})": covered,
                         f"image area ({ctx.area_unit})": area, "coverage (%)": covered / area * 100 if area else np.nan,
                         "overlap (%)": (summed - covered) / summed * 100 if summed > 0 and label is not None
                         and line_width == 0 else np.nan})
    per_image = pd.DataFrame(rows)
    covered_col, area_col = f"covered area ({ctx.area_unit})", f"image area ({ctx.area_unit})"

    order = labels + ["All labels (union)"]
    grouped = per_image.groupby("label", sort=False)
    summary = pd.DataFrame({
        "label": order,
        "n_rois": [int(grouped.get_group(l)["n_rois"].sum()) for l in order],
        covered_col: [grouped.get_group(l)[covered_col].sum() for l in order],
        "coverage (%)": [grouped.get_group(l)[covered_col].sum() / grouped.get_group(l)[area_col].sum() * 100
                         for l in order],
    })
    if len(ctx.files) > 1:
        summary["mean per image (%)"] = [grouped.get_group(l)["coverage (%)"].mean() for l in order]
        summary["std per image (%)"] = [grouped.get_group(l)["coverage (%)"].std(ddof=1) for l in order]
    res.tables["Summary"] = summary
    if len(ctx.files) > 1:
        res.tables["Per image"] = per_image

    fig, ax = ctx.figure(width=max(5.5, 0.7 * len(order) + 2.5), height=4.5)
    x = np.arange(len(order))
    colors = [ctx.color(l) for l in labels] + ["#BBBBBB"]
    err = summary["std per image (%)"].values if len(ctx.files) > 1 else None
    ax.bar(x, summary["coverage (%)"].values, yerr=err, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    if len(ctx.files) > 1:
        for xi, label in zip(x, order):
            vals = grouped.get_group(label)["coverage (%)"].values
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.3
            ax.scatter(xi + jitter, vals, s=12, color="black", zorder=3, alpha=0.6)
    for xi, v in zip(x, summary["coverage (%)"].values):
        ax.annotate(f"{v:.1f}%", (xi, v), ha="center", va="bottom", fontsize=9,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x, [str(l) for l in order], rotation=30 if len(order) > 4 else 0, ha="right" if len(order) > 4 else "center")
    ax.set_ylabel("Coverage (%)")
    title = "Coverage per label"
    if len(ctx.files) > 1:
        title += " (bars: pooled; points: per image; error: std)"
    ax.set_title(title, fontsize=10)
    res.figures["Coverage"] = fig
    return res


@register_analysis(
    "size", "Sizes and measurements",
    "Summary statistics of the ROI sizes per label with histograms and box plots, and a table with one "
    "row per ROI: position, area, perimeter, length, width, centre-line length, aspect ratio.",
    params=[
        Param("metric", "Plotted metric", "choice", "auto",
              choices=("auto", "area", "eq_diameter", "length", "width", "aspect_ratio", "perimeter",
                       "skeleton_length", "skeleton_width", "all"),
              tooltip="auto: area for area ROIs and length for line ROIs.\n"
                      "length / width: minimum bounding rectangle. skeleton_length: length along the centre "
                      "line of the shape (all branches of a branched or fused shape added up)."),
        Param("bins", "Histogram bins", "int", 20, 2, 200),
        Param("density", "Normalise histograms", "bool", False,
              tooltip="Show probability density instead of counts (compares labels of different sizes)."),
        Param("log_scale", "Logarithmic size axis", "bool", False,
              tooltip="Log-spaced bins and axes, for labels whose sizes differ by orders of magnitude."),
    ],
)
def size_distribution(ctx: AnalysisContext, metric: str = "auto", bins: int = 20,
                      density: bool = False, log_scale: bool = False) -> AnalysisResult:
    res = AnalysisResult()
    obj = ctx.objects
    if obj.empty:
        res.notes.append("No ROIs in scope.")
        return res
    if "skeleton_length" not in obj.columns:
        from .gnr import skeleton_metrics            # centre-line length of every outline (cached)
        obj = ctx.objects = ctx.objects.join(skeleton_metrics(ctx))
        line = obj["type"].isin(LINE_TYPES)
        obj.loc[line, "skeleton_length"] = obj.loc[line, "length"]

    rows = []
    for name, col, kind in SIZE_METRICS:
        for label in ctx.labels + [None]:
            vals = obj[col] if label is None else obj.loc[obj["label"] == label, col]
            vals = vals.dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            rows.append({"metric": name, "unit": ctx.metric_unit(kind),
                         "label": "All labels" if label is None else label, "n": len(vals),
                         "mean": vals.mean(), "std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                         "median": np.median(vals), "min": vals.min(), "q25": np.percentile(vals, 25),
                         "q75": np.percentile(vals, 75), "max": vals.max()})
    res.tables["Statistics"] = pd.DataFrame(rows)
    res.tables["Objects"] = object_table(ctx, obj)
    if obj["type"].eq("PointROI").all():
        res.notes.append("Point ROIs have no size.")
        return res

    if metric == "auto":
        plot = [m for m in ("area", "length") if obj[m].notna().any()]
        if obj["area"].notna().all():
            plot = ["area"]          # only area ROIs: their length is a bounding-box size
    elif metric == "all":
        plot = [m for _, m, _ in SIZE_METRICS if obj[m].notna().any()]
    else:
        plot = [metric]

    for col in plot:
        name, kind = next((n, k) for n, c, k in SIZE_METRICS if c == col)
        data = [(l, obj.loc[obj["label"] == l, col].dropna().to_numpy(float)) for l in ctx.labels]
        data = [(l, v) for l, v in data if len(v)]
        if not data:
            res.notes.append(f"No ROI has a {name.lower()}.")
            continue
        unit = ctx.metric_unit(kind)
        axis_label = f"{name} ({unit})" if unit else name
        fig, (ax_h, ax_b) = ctx.figure(1, 2, width=11, height=4.5, gridspec_kw={"width_ratios": [3, 2]})
        allv = np.concatenate([v for _, v in data])
        use_log = log_scale and allv.min() > 0 and allv.max() > allv.min()
        if use_log:
            edges = np.geomspace(allv.min(), allv.max(), int(bins) + 1)
            ax_h.set_xscale("log")
            ax_b.set_yscale("log")
        else:
            edges = np.histogram_bin_edges(allv, bins=int(bins))
        for label, v in data:
            c = ctx.color(label)
            ax_h.hist(v, bins=edges, density=density, histtype="stepfilled", alpha=0.3, color=c)
            ax_h.hist(v, bins=edges, density=density, histtype="step", linewidth=1.5, color=c,
                      label=f"{label} (n={len(v)}, {np.mean(v):.3g} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2g})")
        ax_h.set_xlabel(axis_label)
        ax_h.set_ylabel("Density" if density else "Count")
        ax_h.legend(fontsize=8)
        ax_h.set_title(f"{name} distribution (mean ± std)")
        bp = ax_b.boxplot([v for _, v in data], patch_artist=True, showfliers=True)
        for patch, (label, _) in zip(bp["boxes"], data):
            patch.set_facecolor(ctx.color(label))
            patch.set_alpha(0.6)
        ax_b.set_xticks(range(1, len(data) + 1), [str(l) for l, _ in data],
                        rotation=30 if len(data) > 3 else 0, ha="right" if len(data) > 3 else "center")
        ax_b.set_ylabel(axis_label)
        ax_b.set_title(f"{name} per label")
        res.figures[name] = fig
    return res


def object_table(ctx: AnalysisContext, obj: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per ROI: position, area, perimeter, length, width, aspect ratio (with units)."""
    obj = ctx.objects if obj is None else obj
    cols = ["file", "objectName", "label", "type", "n_vertices", "cx_px", "cy_px", "area", "eq_diameter",
            "perimeter", "length", "width", "skeleton_length", "skeleton_width", "aspect_ratio",
            "orientation", "axis_angle"]
    table = obj[[c for c in cols if c in obj.columns]].copy()
    u = ctx.unit
    return table.rename(columns={
        "orientation": "orientation (°)", "axis_angle": "shape axis (°)",
        "area": f"area ({ctx.area_unit})", "eq_diameter": f"eq_diameter ({u})", "perimeter": f"perimeter ({u})",
        "length": f"length ({u})", "width": f"width ({u})", "skeleton_length": f"skeleton length ({u})",
        "skeleton_width": f"skeleton width ({u})"})


_FOLDS = {"1 (polar, 0–360°)": 1, "2 (axial, 0–180°)": 2, "3-fold (0–120°)": 3, "4-fold (0–90°)": 4,
          "6-fold (0–60°)": 6}
_SOURCES = ("ROI orientation", "ROI orientation, else shape axis", "Shape axis")
_REFERENCES = ("Image x axis", "Reference axis (symmetry vector)")


def _orientations(ctx, source, fold, reference):
    """Per-object relative, folded orientation (Series, NaN = none) and notes."""
    from ..measurements.orientation import relative_angle
    obj = ctx.objects
    own, axis = obj["orientation"], obj["axis_angle"]
    if source == "ROI orientation":
        raw = own
    elif source == "Shape axis":
        raw = axis
    else:
        raw = own.where(own.notna(), axis)
    notes = []
    from_axis = int((own.isna() & axis.notna()).sum()) if source != "ROI orientation" else 0
    if source == "Shape axis":
        from_axis = int(axis.notna().sum())
    if from_axis and fold == 1:
        notes.append(f"{from_axis} orientation(s) come from the shape axis, which has no head / tail: "
                     "with polar symmetry these only cover 0–180°.")
    ref = pd.Series(0.0, index=obj.index)
    if reference == _REFERENCES[1]:
        refs = {f: ctx.images[f].ref_angle for f in ctx.files}
        missing = [f for f, r in refs.items() if r is None]
        if missing:
            notes.append("No reference axis for " + ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
                         + ": the image x axis is used there.")
        ref = obj["file"].map(lambda f: refs.get(f) or 0.0).astype(float)
    rel = pd.Series(relative_angle(raw.to_numpy(float) - ref.to_numpy(float), None, fold), index=obj.index)
    return rel.where(raw.notna()), notes


def _cyclic_cmap():
    from matplotlib import colormaps
    return colormaps["hsv"]


def _angle_colors(values, period, groups=0):
    """RGBA per angle: continuous cyclic colours, or `groups` discrete colours."""
    cmap = _cyclic_cmap()
    t = np.asarray(values, float) / period
    if groups and groups > 1:
        t = (np.floor(t * groups) + 0.5) / groups
    return cmap(np.mod(t, 1.0))


def _color_wheel(ax, period, groups, reference_name):
    n = 360
    theta = np.linspace(0, np.radians(period), n, endpoint=False)
    width = np.radians(period) / n
    ax.bar(theta, np.ones(n), width=width * 1.05, bottom=0.55,
           color=_angle_colors(np.degrees(theta), period, groups), edgecolor="none")
    ax.set_ylim(0, 1.55)
    ax.set_yticks([])
    if period < 360:
        ax.set_thetamin(0)
        ax.set_thetamax(period)
    ax.tick_params(labelsize=7)
    ax.set_title(f"0° = {reference_name}", fontsize=8)
    ax.grid(False)


@register_analysis(
    "orientation", "Orientation",
    "Orientation of the ROIs per label: circular statistics, rose diagrams and colour maps in which ROIs "
    "with similar orientation share a colour. Uses the orientation set on each ROI (orientation mode / "
    "auto-orient) and, if chosen, the long axis of the shape.",
    params=[
        Param("source", "Orientation from", "choice", _SOURCES[1], choices=_SOURCES,
              tooltip="ROI orientation: set in the orientation mode or by auto-orient (has a head / tail).\n"
                      "Shape axis: long axis of the ROI outline (rods, elongated shapes; axial only)."),
        Param("symmetry", "Symmetry", "choice", "1 (polar, 0–360°)", choices=tuple(_FOLDS),
              tooltip="Rotation that maps the object onto itself: angles are reduced to 0–360°/n."),
        Param("reference", "Angles relative to", "choice", _REFERENCES[0], choices=_REFERENCES,
              tooltip="Reference axis = a1 of the image's first lattice (Lattice vector tool), or an old "
                      "reference axis / symmetry vector."),
        Param("bins", "Histogram bins", "int", 24, 4, 180),
        Param("map_coloring", "Map colours", "choice", "Continuous", choices=("Continuous", "Groups")),
        Param("groups", "Number of colour groups", "int", 6, 2, 24,
              tooltip="Used when 'Map colours' = Groups: orientations are split into this many equal sectors."),
        Param("max_maps", "Maps (images)", "int", 4, 0, 30, tooltip="Number of images drawn as orientation maps."),
    ],
)
def orientation(ctx: AnalysisContext, source=_SOURCES[1], symmetry="1 (polar, 0–360°)", reference=_REFERENCES[0],
                bins=24, map_coloring="Continuous", groups=6, max_maps=4) -> AnalysisResult:
    from ..measurements.orientation import circular_stats
    res = AnalysisResult()
    obj = ctx.objects
    if obj.empty:
        res.notes.append("No ROIs in scope.")
        return res
    fold = _FOLDS.get(symmetry, 1)
    period = 360.0 / fold
    rel, notes = _orientations(ctx, source, fold, reference)
    res.notes.extend(notes)
    oriented = rel.notna()
    if not oriented.any():
        res.notes.append("No ROI has an orientation. Set them in the labelling page (orientation mode, "
                         "auto-orient or 'convert vectors'), or choose 'Shape axis'.")
        return res
    ref_name = "x axis" if reference == _REFERENCES[0] else "reference axis"
    n_groups = int(groups) if map_coloring == "Groups" else 0

    # ---- tables
    rows = []
    edges = np.linspace(0, period, int(bins) + 1)
    for label in ctx.labels + [None]:
        sel = rel[oriented] if label is None else rel[oriented & (obj["label"] == label)]
        total = len(obj) if label is None else int((obj["label"] == label).sum())
        if label is not None and sel.empty and total == 0:
            continue
        st = circular_stats(sel.to_numpy(), fold)
        peak = np.nan
        if len(sel):
            counts, _ = np.histogram(sel, bins=edges)
            peak = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2
        rows.append({"label": "All labels" if label is None else label, "n oriented": st["n"],
                     "n without": total - st["n"], "mean (°)": st["mean"], "alignment R": st["R"],
                     "circular std (°)": st["circ_std"], "Rayleigh p": st["rayleigh_p"], "most common (°)": peak})
    res.tables["Summary"] = pd.DataFrame(rows)
    if len(ctx.files) > 1:
        per = []
        for (file, label), sel in rel[oriented].groupby([obj.loc[oriented, "file"], obj.loc[oriented, "label"]]):
            st = circular_stats(sel.to_numpy(), fold)
            per.append({"file": file, "label": label, "n": st["n"], "mean (°)": st["mean"], "alignment R": st["R"]})
        res.tables["Per image"] = pd.DataFrame(per)
    table = obj[["file", "objectName", "label", "type"]].copy()
    table["orientation (°)"] = obj["orientation"]
    table["shape axis (°)"] = obj["axis_angle"]
    table[f"relative, folded (°)"] = rel
    if n_groups:
        table["colour group"] = np.where(oriented, np.floor(rel.fillna(0) / period * n_groups).astype(int) + 1, 0)
    res.tables["Objects"] = table

    # ---- rose diagrams
    labels = [l for l in ctx.labels if (oriented & (obj["label"] == l)).any()][:9]
    ncols = min(3, len(labels))
    nrows = int(math.ceil(len(labels) / ncols))
    fig = Figure(figsize=(3.4 * ncols, 3.3 * nrows), layout="constrained")
    for k, label in enumerate(labels):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="polar")
        sel = rel[oriented & (obj["label"] == label)]
        counts, _ = np.histogram(sel, bins=edges)
        width = np.radians(period) / int(bins)
        ax.bar(np.radians(edges[:-1]), counts, width=width, align="edge",
               color=ctx.color(label), edgecolor="white", linewidth=0.5, alpha=0.9)
        st = circular_stats(sel.to_numpy(), fold)
        if np.isfinite(st["mean"]) and counts.max() > 0:
            ax.plot([0, np.radians(st["mean"])], [0, counts.max() * st["R"]], color="black", lw=2)
        if period < 360:
            ax.set_thetamin(0)
            ax.set_thetamax(period)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{label} (n={st['n']}, R={st['R']:.2f})", fontsize=9)
    fig.suptitle(f"Orientation relative to the {ref_name} (black line: mean direction × alignment R)", fontsize=9)
    res.figures["Rose diagrams"] = fig

    # ---- linear histogram
    fig, ax = ctx.figure(width=7, height=3.8)
    data = [rel[oriented & (obj["label"] == l)].to_numpy() for l in labels]
    ax.hist(data, bins=edges, stacked=True, color=[ctx.color(l) for l in labels], label=[str(l) for l in labels],
            edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, period)
    ax.set_xlabel(f"Orientation relative to the {ref_name} (°)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    res.figures["Angle histogram"] = fig

    # ---- orientation maps
    from matplotlib.collections import LineCollection, PolyCollection
    shapes = {(s["file"], s["objectName"]): s for s in ctx.shapes}
    mapped = [f for f in ctx.files if (oriented & (obj["file"] == f)).any()][:int(max_maps)]
    for file in mapped:
        info = ctx.images[file]
        fig = Figure(figsize=(8.6, 6.4), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[5, 1.3])
        ax = fig.add_subplot(gs[0, 0])
        img = ctx.image_array(file)
        if img is not None:
            ax.imshow(img, cmap="gray", extent=(0, img.shape[1], img.shape[0], 0), interpolation="nearest")
        else:
            ax.set_facecolor("#202020")
        sub = obj[obj["file"] == file]
        polys, poly_colors, lines, line_colors, points, point_colors = [], [], [], [], [], []
        grey_polys, grey_lines = [], []
        arrows = []
        for idx, row in sub.iterrows():
            s = shapes.get((file, row["objectName"]))
            if s is None:
                continue
            a = rel.loc[idx]
            color = _angle_colors([a], period, n_groups)[0] if np.isfinite(a) else None
            pts = s["pts"]
            if s["type"] in AREA_TYPES:
                (polys if color is not None else grey_polys).append(pts)
                if color is not None:
                    poly_colors.append(color)
            elif s["type"] in LINE_TYPES:
                (lines if color is not None else grey_lines).append(pts)
                if color is not None:
                    line_colors.append(color)
            elif s["type"] == "PointROI" and color is not None:
                points.append(pts[0])
                point_colors.append(color)
            own = row["orientation"]
            if np.isfinite(own) and source != "Shape axis":
                cx, cy = pts.mean(axis=0)
                size = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 6.0)
                arrows.append((cx, cy, np.cos(np.radians(own)) * size * 0.4, -np.sin(np.radians(own)) * size * 0.4))
        if grey_polys:
            ax.add_collection(PolyCollection(grey_polys, facecolors="none", edgecolors="#9e9e9e", linewidths=0.8))
        if polys:
            ax.add_collection(PolyCollection(polys, facecolors=[(*c[:3], 0.55) for c in poly_colors],
                                             edgecolors=poly_colors, linewidths=1.2))
        if grey_lines:
            ax.add_collection(LineCollection(grey_lines, colors="#9e9e9e", linewidths=1))
        if lines:
            ax.add_collection(LineCollection(lines, colors=line_colors, linewidths=2.5))
        if points:
            p = np.asarray(points)
            ax.scatter(p[:, 0], p[:, 1], c=point_colors, s=40, edgecolors="black", linewidths=0.5)
        if arrows:
            a = np.asarray(arrows)
            ax.quiver(a[:, 0], a[:, 1], a[:, 2], a[:, 3], angles="xy", scale_units="xy", scale=1,
                      color="white", width=0.004, headwidth=4, pivot="middle")
        ax.set_xlim(0, info.width)
        ax.set_ylim(info.height, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{file}: {int(oriented[obj['file'] == file].sum())} oriented ROIs", fontsize=9)
        wheel = fig.add_subplot(gs[0, 1], projection="polar")
        _color_wheel(wheel, period, n_groups, ref_name)
        res.figures[f"Map {os.path.splitext(file)[0]}"] = fig
    if len([f for f in ctx.files if (oriented & (obj["file"] == f)).any()]) > len(mapped):
        res.notes.append(f"Maps drawn for the first {len(mapped)} images (raise 'Maps (images)' for more).")
    return res


# =============================================================================
# Running and exporting
# =============================================================================

def default_params(key) -> dict:
    return {p.name: p.default for p in ANALYSES[key].params}


def run_analyses(ctx: AnalysisContext, keys=None, params=None) -> dict:
    """
    Run analyses (default: all registered) and return {key: AnalysisResult}.  A failing
    analysis returns a result with the error as a note instead of stopping the others.
    """
    params = params or {}
    results = {}
    for key in keys or list(ANALYSES):
        kwargs = {**default_params(key), **params.get(key, {})}
        try:
            result = ANALYSES[key].func(ctx, **kwargs)
        except Exception as e:      # keep the other analyses running
            result = AnalysisResult(notes=[f"Analysis failed: {e}", traceback.format_exc(limit=3)])
        result.notes = list(ctx.notes) + list(result.notes)
        results[key] = result
    return results


def _safe(text):
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("_") or "x"


def export_results(results: dict, out_dir: str, dpi=200) -> list:
    """Write every table (CSV), figure (PNG) and the notes (notes.txt) to out_dir. Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    notes = []
    for key, result in results.items():
        title = ANALYSES[key].title if key in ANALYSES else key
        for name, table in result.tables.items():
            path = os.path.join(out_dir, f"{_safe(key)}_{_safe(name)}.csv")
            table.to_csv(path, index=False)
            written.append(path)
        for name, fig in result.figures.items():
            path = os.path.join(out_dir, f"{_safe(key)}_{_safe(name)}.png")
            fig.savefig(path, dpi=dpi)
            written.append(path)
        if result.notes:
            notes.append(f"[{title}]\n" + "\n".join(result.notes))
    if notes:
        path = os.path.join(out_dir, "notes.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(notes))
        written.append(path)
    return written


# =============================================================================
# Command line
# =============================================================================

def _images_from_dir(files, image_dir, metadata_csv=None):
    scales = {}
    if metadata_csv and os.path.isfile(metadata_csv):
        meta = pd.read_csv(metadata_csv)
        if {"Filename", "Pixel Scale X (nm)"} <= set(meta.columns):
            scales = dict(zip(meta["Filename"].astype(str), meta["Pixel Scale X (nm)"]))
    images = {}
    for file in files:
        img = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"warning: cannot read {file}, skipped")
            continue
        scale = scales.get(file)
        scale = float(scale) if scale is not None and not pd.isna(scale) else None
        images[file] = ImageInfo(file, img.shape[1], img.shape[0], scale)
    return images


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="ROI statistics and charts from a ScanBound annotation CSV.")
    parser.add_argument("annotations")
    parser.add_argument("--images", required=True, help="directory containing the annotated images")
    parser.add_argument("--metadata", help="metadata CSV with pixel scales (default: IMAGES/metadata.csv)")
    parser.add_argument("--out", help="output directory (default: IMAGES/roi_analysis_<timestamp>)")
    parser.add_argument("--analyses", nargs="*", choices=list(ANALYSES), default=None)
    parser.add_argument("--include-unlabeled", action="store_true")
    args = parser.parse_args(argv)

    annotations = pd.read_csv(args.annotations)
    files = list(dict.fromkeys(annotations["file"].astype(str)))
    metadata = args.metadata or os.path.join(args.images, "metadata.csv")
    images = _images_from_dir(files, args.images, metadata)
    ctx = AnalysisContext(AnalysisData(annotations, images), include_unlabeled=args.include_unlabeled)
    results = run_analyses(ctx, args.analyses)
    out = args.out or os.path.join(args.images, f"roi_analysis_{datetime.now():%Y%m%d_%H%M%S}")
    for path in export_results(results, out):
        print(path)


if __name__ == "__main__":
    main()
