"""
ScanBound's old ``annotations.csv`` format, in and out.

Columns: ``objectName, type, x1, y1, dx, dy, file, label[, angle]`` with one row per vertex for
polygons and line segments.  ``type`` is the old viewer's ROI class:

=================  ===========================  ==========================================
old type           new kind / geometry          notes
=================  ===========================  ==========================================
RectROI            feature / rect               x1, y1, dx, dy = left, top, width, height
EllipseROI         feature / ellipse            same
PointROI           feature / point              x1, y1
PolygonROI         feature / polygon            one row per vertex
MaskROI            feature / polygon            written for masks: one row per vertex of the outer
                                                boundary (the CSV cannot hold the pixels)
LineSegmentROI     centre_line / polyline       one row per vertex
VectorROI, LineROI orientation_vector / segment x1, y1 -> dx, dy are start and END points;
                                                attached to the smallest feature containing it
VectorROI labelled "Symmetry Vector"  reference_axis / segment
VectorROI labelled "Herringbone Axis" herringbone_axis / segment
=================  ===========================  ==========================================

``"Empty Label"`` becomes no label.  The ``file`` column names the image the old app showed
(often a JPG exported from a scan: ``<scan stem>_<channel>.jpg``); a *resolver* maps it to a
project image id and, when the JPG's pixel grid differs from the raw scan (incomplete lines
cropped, anisotropic pixels resampled), to the 3x3 matrix that converts JPG pixels to raw pixels.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .annotation import Annotation, Origin
from .geometry import Ellipse, Point, Polygon, Polyline, Rect, Segment

EMPTY_LABEL = "Empty Label"
SYMMETRY_LABEL = "Symmetry Vector"
HERRINGBONE_LABEL = "Herringbone Axis"
COLUMNS = ["objectName", "type", "x1", "y1", "dx", "dy", "file", "label"]
ANGLE = "angle"

# resolver(file name) -> (image id, optional 3x3 matrix legacy px -> image px); None = skip
Resolver = Callable[[str], Optional[tuple[str, Optional[np.ndarray]]]]


def _label(v: object) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return None if s in ("", EMPTY_LABEL, "nan", "None") else s


def _angle(v: object) -> Optional[float]:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"type", "x1", "y1", "dx", "dy", "file"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: not a ScanBound annotation file (missing columns {', '.join(sorted(missing))})")
    return df


def from_frame(df: pd.DataFrame, resolver: Optional[Resolver] = None, origin: Optional[Origin] = None,
               status: str = "manual") -> tuple[list[Annotation], list[str]]:
    """Annotations from an old-format table, and a list of problems (rows skipped and why)."""
    origin = origin or Origin("import", "annotations.csv")
    resolver = resolver or (lambda name: (name, None))
    out: list[Annotation] = []
    problems: list[str] = []
    if "objectName" not in df.columns:
        df = df.assign(objectName=[f"row{i}" for i in range(len(df))])
    df = df.copy()
    df["objectName"] = df["objectName"].astype(str)
    # rows of one vertex-object can only be grouped by name; old files wrote the Python repr of
    # the ROI object there, which is unique per ROI as long as it was alive, so grouping still works
    missing: dict[str, int] = {}
    resolved: dict[str, Optional[tuple[str, Optional[np.ndarray]]]] = {}
    for (fname, name, rtype), rows in df.groupby(["file", "objectName", "type"], sort=False):
        fname = str(fname)
        if fname not in resolved:
            resolved[fname] = resolver(fname)
        target = resolved[fname]
        if target is None:
            missing[fname] = missing.get(fname, 0) + 1
            continue
        image_id, matrix = target
        label = _label(rows["label"].iloc[0]) if "label" in rows else None
        angle = _angle(rows[ANGLE].iloc[0]) if ANGLE in rows else None
        r0 = rows.iloc[0]
        try:
            x1, y1, dx, dy = (float(r0[c]) for c in ("x1", "y1", "dx", "dy"))
            kind = "feature"
            if rtype in ("RectROI", "EllipseROI"):
                geom = (Rect if rtype == "RectROI" else Ellipse)(x1, y1, dx, dy)
            elif rtype == "PointROI":
                geom = Point(x1, y1)
            elif rtype in ("PolygonROI", "MaskROI"):
                geom = Polygon(rows[["x1", "y1"]].to_numpy(float))
            elif rtype == "LineSegmentROI":
                pts = rows[["x1", "y1"]].to_numpy(float)
                geom = Polyline(pts) if len(pts) > 2 else Segment(pts)
                kind = "centre_line"
            elif rtype in ("VectorROI", "LineROI"):
                geom = Segment.from_coords(x1, y1, dx, dy)
                kind = {SYMMETRY_LABEL: "reference_axis", HERRINGBONE_LABEL: "herringbone_axis"}.get(
                    label or "", "orientation_vector")
                if kind != "orientation_vector":
                    label = None
            else:
                problems.append(f"{fname}: unknown ROI type {rtype!r}, skipped")
                continue
            if matrix is not None:
                geom = geom.transformed(matrix)
            if kind == "centre_line" and isinstance(geom, Polyline) and len(geom.points) == 2:
                geom = Segment(geom.points)
        except (ValueError, TypeError) as exc:
            problems.append(f"{fname}: {rtype} {name}: {exc}")
            continue
        out.append(Annotation(image_id, kind, geom, label=label, status=status, origin=origin, orientation=angle,
                              attrs={"legacy_name": name} if not name.startswith("<") else {}))
    for fname, n in missing.items():
        problems.append(f"{fname}: no matching image in the project, {n} annotation(s) skipped")
    _attach_vectors(out)
    return out, problems


def _attach_vectors(items: list[Annotation]) -> None:
    """Give each orientation vector the smallest feature that contains both of its ends as parent."""
    features: dict[str, list[Annotation]] = {}
    for a in items:
        if a.kind == "feature" and a.geometry.closed:
            features.setdefault(a.image_id, []).append(a)
    for i, a in enumerate(items):
        if a.kind != "orientation_vector":
            continue
        inside = [f for f in features.get(a.image_id, [])
                  if all(f.geometry.contains(x, y) for x, y in a.geometry.points)]
        if inside:
            parent = min(inside, key=lambda f: f.geometry.area())
            items[i] = Annotation(a.image_id, a.kind, a.geometry, a.label, a.status, a.origin, a.orientation,
                                  parent.id, a.attrs, id=a.id, created=a.created)


def load_csv(path: str, resolver: Optional[Resolver] = None, status: str = "manual"
             ) -> tuple[list[Annotation], list[str]]:
    import os
    return from_frame(read_csv(path), resolver, Origin("import", os.path.basename(path)), status)


_TYPE_OUT = {"rect": "RectROI", "ellipse": "EllipseROI", "point": "PointROI", "polygon": "PolygonROI",
             "mask": "MaskROI"}


def to_frame(annotations: Iterable[Annotation], file_name: Callable[[str], str] = lambda i: i,
             statuses: tuple[str, ...] = ("manual", "accepted")) -> pd.DataFrame:
    """Old-format table (with the ``angle`` column).  Measurement lines and lattice vectors are
    written as VectorROI; rejected / unreviewed proposals are left out unless ``statuses`` says so.
    An image without a reference axis gets the a1 vector of its first lattice as its Symmetry Vector."""
    from collections import defaultdict
    from ..measurements.report import reference_axis
    annotations = list(annotations)
    per_image: dict[str, list[Annotation]] = defaultdict(list)
    for a in annotations:
        per_image[a.image_id].append(a)
    as_symmetry = set()
    for anns in per_image.values():
        ref = reference_axis(anns)
        if ref is not None and ref.kind == "lattice_vector":
            as_symmetry.add(ref.id)
    rows = []
    for a in annotations:
        if a.status not in statuses:
            continue
        f = file_name(a.image_id)
        label = a.label or EMPTY_LABEL
        ang = float("nan") if a.orientation is None else a.orientation
        g = a.geometry
        name = f"{a.id}"
        if a.kind == "feature":
            t = _TYPE_OUT[g.type]
            if g.type in ("rect", "ellipse"):
                rows.append([name, t, g.x, g.y, g.w, g.h, f, label, ang])  # type: ignore[attr-defined]
            elif g.type == "point":
                rows.append([name, t, g.x, g.y, 0, 0, f, label, ang])  # type: ignore[attr-defined]
            elif g.type == "mask":
                rows.extend([name, t, x, y, 0, 0, f, label, ang] for x, y in g.outline())
            else:
                rows.extend([name, t, x, y, 0, 0, f, label, ang] for x, y in g.points)
        elif a.kind == "centre_line":
            rows.extend([name, "LineSegmentROI", x, y, 0, 0, f, label, ang] for x, y in g.points)
        elif g.type == "segment":
            (x0, y0), (x1, y1) = g.points
            lab = {"reference_axis": SYMMETRY_LABEL, "herringbone_axis": HERRINGBONE_LABEL}.get(a.kind, label)
            if a.id in as_symmetry:
                lab = SYMMETRY_LABEL
            rows.append([name, "VectorROI", x0, y0, x1, y1, f, lab, ang])
        else:
            rows.extend([name, "LineSegmentROI", x, y, 0, 0, f, label, ang] for x, y in g.points)
    return pd.DataFrame(rows, columns=COLUMNS + [ANGLE])


def legacy_display_transform(raw_shape: tuple[int, int], valid_rows: np.ndarray, pixel_size: tuple[float, float],
                             legacy_size: tuple[int, int]) -> Optional[np.ndarray]:
    """
    Matrix legacy-JPG px -> raw px for a JPG written by the old scan browser, which dropped scan
    lines containing NaN and resampled non-square pixels to the finer pixel size.  ``legacy_size``
    is the JPG's (width, height).  None if the JPG cannot have come from this scan.
    """
    rows, cols = raw_shape
    kept = np.flatnonzero(valid_rows)
    if kept.size == 0:
        return None
    n_rows = kept.size
    dx, dy = pixel_size
    if abs(dx / dy - 1) < 1e-3:
        exp_w, exp_h = cols, n_rows
    elif dx > dy:
        exp_w, exp_h = int(round(cols * dx / dy)), n_rows
    else:
        exp_w, exp_h = cols, int(round(n_rows * dy / dx))
    w, h = legacy_size
    if abs(w - exp_w) > 1 or abs(h - exp_h) > 1:
        # an integer enlargement (colorbar exports) keeps the aspect ratio
        k = w / exp_w if exp_w else 0
        if k < 1 or abs(k - round(k)) > 1e-6 or abs(h - exp_h * round(k)) > 1:
            return None
    sx, sy = cols / w, n_rows / h
    first = float(kept[0])
    if kept.size != kept[-1] - kept[0] + 1:          # NaN lines in the middle: no affine map
        return None
    return np.array([[sx, 0, 0], [0, sy, first], [0, 0, 1]], float)
