"""Conversions between annotation shapes (all in pixel coordinates; widths in physical units)."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .annotation import Annotation
from .geometry import Polygon


def thicken_polyline(points: np.ndarray, half_width: float, end_buffer: float = 0.0) -> np.ndarray:
    """Outline (M, 2) of a polyline widened by ``half_width`` on each side and extended by
    ``end_buffer`` at both ends (same units as the points)."""
    p = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    if end_buffer > 0:
        for end, nxt in ((0, 1), (-1, -2)):
            d = p[end] - p[nxt]
            n = np.hypot(*d)
            if n > 0:
                p[end] = p[end] + d / n * end_buffer
    tang = np.empty_like(p)
    tang[0] = p[1] - p[0]
    tang[-1] = p[-1] - p[-2]
    tang[1:-1] = p[2:] - p[:-2]
    length = np.hypot(tang[:, 0], tang[:, 1])
    safe = np.where(length > 0, length, 1.0)
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1) / safe[:, None]
    normals[length == 0] = 0.0
    return np.vstack([p + normals * half_width, (p - normals * half_width)[::-1]])


def lines_to_outlines(annotations: Iterable[Annotation], width: float, end_buffer: float = 0.0,
                      pixel_size: tuple[float, float] = (1.0, 1.0)) -> list[tuple[Annotation, Annotation]]:
    """[(centre line, new polygon feature)].  ``width`` and ``end_buffer`` are in the unit of
    ``pixel_size`` (e.g. metres), so the outline has the right width along x and y even when the
    pixels are not square.  The new feature keeps the label and records where it came from."""
    sx, sy = pixel_size
    out = []
    for a in annotations:
        if a.kind != "centre_line" or len(a.geometry.points) < 2:
            continue
        phys = a.geometry.points * np.array([sx, sy])
        poly = thicken_polyline(phys, width / 2.0, end_buffer) / np.array([sx, sy])
        new = Annotation(a.image_id, "feature", Polygon(poly), label=a.label, status=a.status, origin=a.origin,
                         attrs={**a.attrs, "from_centre_line": a.id, "width": width})
        out.append((a, new))
    return out


def outlines_to_centre_lines(annotations: Iterable[Annotation], params=None,
                             pixel_size: tuple[float, float] = (1.0, 1.0), run_id: str = ""
                             ) -> list[tuple[Annotation, list[Annotation]]]:
    """
    [(shape, its skeleton segments as centre lines)]: every closed feature (outline or painted mask)
    is skeletonised and split into separate segments at junctions and sharp bends
    (:mod:`scanbound.ribbon_graph`).  The centre lines belong to the shape (``parent``), with its
    label; correct or delete them and the skeleton analyses use them instead of the shape's
    skeleton.  Non-square pixels are handled.
    """
    import math

    from ..model.provenance import software_version
    from ..ribbon_graph import (RibbonParams, mask_outlines, polygon_ribbons, skeleton_graph,
                                split_ribbons)
    from .annotation import Origin
    from .geometry import Polyline, Segment

    params = params or RibbonParams()
    sx, sy = pixel_size
    s = math.sqrt(sx * sy) or 1.0
    f = np.array([sx / s, sy / s])                       # raw px -> square px
    origin = Origin("detector", "ribbon split", software_version(), run_id, None, {"params": params.to_dict()})
    out = []
    for a in annotations:
        if a.kind != "feature" or not a.geometry.closed or a.geometry.type == "point":
            continue
        if a.geometry.type == "mask":                    # painted: its pixel boundary, holes included
            mg = a.geometry
            outer, holes = mask_outlines(getattr(mg, "bits"), (getattr(mg, "x0"), getattr(mg, "y0")))
            if outer is None:
                continue
            graph = skeleton_graph(outer * f, params, [h * f for h in holes])
            ribbons = split_ribbons(graph, params) if graph is not None else []
        else:
            pts = np.asarray(a.geometry.outline(), float) * f
            _g, ribbons = polygon_ribbons(pts, params)
        lines = []
        for k, r in enumerate(ribbons, start=1):
            p = _simplify(r.points, 0.35 * max(r.width, 1.0)) / f
            g = Segment(p) if len(p) == 2 else Polyline(p)
            if g.is_degenerate():
                continue
            lines.append(Annotation(a.image_id, "centre_line", g, label=a.label, status="proposed", origin=origin,
                                    parent=a.id, attrs={"ribbon": k, "width_px": float(r.width),
                                                        "end_extension_px": [float(v) for v in r.end_extension]}))
        out.append((a, lines))
    return out


def _simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: a centre line with few, editable vertices."""
    import cv2
    p = np.asarray(points, np.float32).reshape(-1, 1, 2)
    q = cv2.approxPolyDP(p, float(tolerance), False).reshape(-1, 2).astype(float)
    return q if len(q) >= 2 else np.asarray(points, float)[[0, -1]]
