"""
Measurements of single annotations, in physical units, with uncertainty budgets.

    ms = measure_annotation(ann, image)              # {"length": Measurement, "angle": ...}
    print(ms["length"].explain())
    # length = 12.43 ± 0.27 nm
    # Method: path length of the centre line on the calibrated pixel grid (dx = 49.3 pm, dy = 49.3 pm)
    # Uncertainty (1σ, combined in quadrature): calibration 249 pm, placement 35 pm
    # Assumes: end points placed to ±0.5 px (1σ) by hand
    # Assumes: scale x: from the file header (...), ±2 % (1σ)

Uncertainty model
-----------------
*calibration*: the relative uncertainty of each axis' :class:`~scanbound.model.Calibration`,
propagated per axis (x and y independent).  *placement*: each vertex is placed with a standard
deviation of ``manual_px`` (hand drawn) or ``detector_px`` (computed) pixels; for lengths only the
two end points count (interior vertices of a centre line move it sideways, which barely changes
its length), for areas the outline moves outward or inward by that much.  Both are assumptions
and are listed in every measurement they enter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..annotations.annotation import Annotation
from ..annotations.geometry import Segment, angle_deg
from ..model.units import format_value
from .measurement import Measurement, combine

if TYPE_CHECKING:
    from ..model.image import Image


@dataclass(frozen=True)
class UncertaintyModel:
    manual_px: float = 0.5
    detector_px: float = 1.0

    def placement(self, ann: Annotation) -> tuple[float, str]:
        if ann.origin.method in ("detector", "model") and not ann.edited:
            return self.detector_px, f"outline / points placed to ±{self.detector_px:g} px (1σ) by {ann.origin.name or 'a detector'}"
        return self.manual_px, f"end points placed to ±{self.manual_px:g} px (1σ) by hand"


DEFAULT_MODEL = UncertaintyModel()


def _scale(image: Optional["Image"]) -> tuple[tuple[float, float], str, tuple[float, float], list[str]]:
    """((dx, dy), unit, (rel_x, rel_y), assumptions)."""
    if image is None or not image.calibrated:
        return (1.0, 1.0), "px", (0.0, 0.0), ["no calibration: lengths are in pixels"]
    dx, dy = image.pixel_size
    rx, ry = image.x.calibration.rel_uncertainty, image.y.calibration.rel_uncertainty
    notes = [f"scale x: {image.x.calibration.describe()}"]
    if image.y.calibration != image.x.calibration:
        notes.append(f"scale y: {image.y.calibration.describe()}")
    return (dx, dy), image.x.unit, (rx, ry), notes


def _inputs(ann: Annotation, image: Optional["Image"]) -> dict:
    d = {"annotation": ann.id, "image": ann.image_id}
    if image is not None and image.calibrated:
        d["pixel_size"] = list(image.pixel_size)
    return d


def path_length(points: np.ndarray, scale: tuple[float, float], rel: tuple[float, float], sigma_px: float,
                closed: bool = False) -> tuple[float, dict[str, float]]:
    """Length and its uncertainty components (calibration, placement)."""
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[:1]])
    d = np.diff(p, axis=0) * np.asarray(scale)
    seg = np.hypot(d[:, 0], d[:, 1])
    L = float(seg.sum())
    if L == 0:
        return 0.0, {}
    safe = np.where(seg > 0, seg, 1)
    lx = float((d[:, 0] ** 2 / safe).sum())        # dL/d(ln sx)
    ly = float((d[:, 1] ** 2 / safe).sum())
    comps = {"calibration": math.hypot(lx * rel[0], ly * rel[1])}
    if sigma_px > 0 and not closed:
        # along-line pixel size at each end
        def along(v: np.ndarray) -> float:
            n = np.hypot(*v)
            return float(np.hypot(v[0] / n * scale[0], v[1] / n * scale[1])) if n > 0 else math.sqrt(scale[0] * scale[1])
        raw = np.diff(np.asarray(points, float), axis=0)
        comps["placement"] = sigma_px * math.hypot(along(raw[0]), along(raw[-1]))
    return L, comps


def _angle(seg_delta: tuple[float, float], scale: tuple[float, float], length_phys: float, sigma_px: float,
           ) -> tuple[float, float]:
    ang = angle_deg(seg_delta[0], seg_delta[1], scale)
    if length_phys <= 0:
        return ang, math.inf
    perp = math.sqrt(scale[0] * scale[1])
    return ang, math.degrees(math.sqrt(2) * sigma_px * perp / length_phys)


def measure_annotation(ann: Annotation, image: Optional["Image"] = None, *,
                       model: UncertaintyModel = DEFAULT_MODEL,
                       reference: Optional[Annotation] = None, fold: int = 1) -> dict[str, Measurement]:
    """All geometric measurements that make sense for the annotation's kind and geometry."""
    scale, unit, rel, cal_notes = _scale(image)
    sigma, place_note = model.placement(ann)
    g = ann.geometry
    inp = _inputs(ann, image)
    out: dict[str, Measurement] = {}
    grid = (f"on the calibrated pixel grid (dx = {format_value(scale[0], unit)}, dy = {format_value(scale[1], unit)})"
            if unit != "px" else "in pixels")
    assumptions = (place_note, *cal_notes)

    if g.type in ("segment", "polyline"):
        L, comps = path_length(g.points, scale, rel, sigma)
        what = "centre line" if ann.kind == "centre_line" else "line"
        out["length"] = Measurement("length", L, unit, combine(comps), f"path length of the {what} {grid}",
                                    assumptions, inp, comps)
        delta = tuple(g.points[-1] - g.points[0])
        e2e, ecomps = path_length(np.vstack([g.points[0], g.points[-1]]), scale, rel, sigma)
        if g.type == "polyline":
            out["end_to_end"] = Measurement("end_to_end", e2e, unit, combine(ecomps),
                                            f"straight distance between the end points {grid}", assumptions, inp, ecomps)
            if L > 0:
                out["straightness"] = Measurement("straightness", min(1.0, e2e / L), "",
                                                  None, "end-to-end distance / path length (1 = straight)", (), inp)
        ang, s_ang = _angle(delta, scale, e2e, sigma)  # type: ignore[arg-type]
        directed = ann.kind in ("orientation_vector", "reference_axis", "lattice_vector")
        if not directed:
            ang = ang % 180.0
        out["angle"] = Measurement("angle", ang, "°", s_ang,
                                   ("direction start → end" if directed else "axis direction (0-180°)") +
                                   ", counter-clockwise from the image x axis, physical (pixel aspect corrected)",
                                   (place_note,), inp, {"placement": s_ang})
        if reference is not None and isinstance(reference.geometry, Segment):
            ref_len, _ = path_length(reference.geometry.points, scale, rel, 0)
            ref_ang, ref_s = _angle(reference.geometry.delta(), scale, ref_len, model.placement(reference)[0])
            rel_ang = ((ang - ref_ang) % 360.0) % (360.0 / max(1, fold))
            comps2 = {"placement": s_ang, "reference axis": ref_s}
            out["relative_angle"] = Measurement("relative_angle", rel_ang, "°", combine(comps2),
                                                f"angle relative to the reference axis {reference.id}"
                                                + (f", reduced by {fold}-fold symmetry" if fold > 1 else ""),
                                                (place_note,), {**inp, "reference": reference.id}, comps2)

    if g.closed:
        a_px = g.area()
        A = a_px * scale[0] * scale[1]
        P, pcomps = path_length(g.outline(), scale, rel, 0.0, closed=True)
        comps = {"calibration": A * math.hypot(rel[0], rel[1]),
                 "placement": P * sigma * math.sqrt(scale[0] * scale[1])}
        out["area"] = Measurement("area", A, f"{unit}²", combine(comps), f"area of the {g.type} {grid}",
                                  (place_note.replace("end points", "outline"), *cal_notes), inp, comps)
        out["perimeter"] = Measurement("perimeter", P, unit, combine(pcomps), f"perimeter of the {g.type} {grid}",
                                       tuple(cal_notes), inp, pcomps)
        x0, y0, x1, y1 = g.bbox()
        for name, extent, s in (("width", x1 - x0, scale[0]), ("height", y1 - y0, scale[1])):
            r = rel[0] if name == "width" else rel[1]
            c = {"calibration": extent * s * r, "placement": math.sqrt(2) * sigma * s}
            out[name] = Measurement(name, extent * s, unit, combine(c), f"bounding-box {name} {grid}",
                                    assumptions, inp, c)
        out["eq_diameter"] = Measurement("eq_diameter", 2 * math.sqrt(A / math.pi), unit,
                                         combine(comps) / max(1e-300, math.sqrt(math.pi * A)) if A > 0 else None,
                                         "diameter of a circle with the same area", assumptions, inp)

    cx, cy = g.centroid()
    c = {"placement": sigma * scale[0]}
    out["x"] = Measurement("x", cx * scale[0], unit, sigma * scale[0], f"centroid x from the image's left edge {grid}",
                           (place_note,), inp, c)
    out["y"] = Measurement("y", cy * scale[1], unit, sigma * scale[1], f"centroid y from the image's top edge {grid}",
                           (place_note,), inp, {"placement": sigma * scale[1]})

    if ann.orientation is not None:
        out["orientation"] = Measurement("orientation", ann.orientation, "°", None,
                                         "orientation attribute of the feature (set by hand or estimated "
                                         "from intensity moments)", (), inp)
    return out


def lattice(a1: Annotation, a2: Annotation, image: Optional["Image"] = None, *,
            model: UncertaintyModel = DEFAULT_MODEL) -> dict[str, Measurement]:
    """Lattice constants |a1|, |a2| and the angle between two lattice vectors."""
    scale, unit, rel, notes = _scale(image)
    s1, n1 = model.placement(a1)
    s2, _ = model.placement(a2)
    L1, c1 = path_length(a1.geometry.points, scale, rel, s1)
    L2, c2 = path_length(a2.geometry.points, scale, rel, s2)
    t1, e1 = _angle(tuple(a1.geometry.points[1] - a1.geometry.points[0]), scale, L1, s1)  # type: ignore[arg-type]
    t2, e2 = _angle(tuple(a2.geometry.points[1] - a2.geometry.points[0]), scale, L2, s2)  # type: ignore[arg-type]
    gamma = abs((t2 - t1 + 180.0) % 360.0 - 180.0)
    inp = {"annotations": [a1.id, a2.id], "image": a1.image_id}
    method = "length of the lattice vector on the calibrated grid"
    return {
        "a1": Measurement("a1", L1, unit, combine(c1), method, (n1, *notes), inp, c1),
        "a2": Measurement("a2", L2, unit, combine(c2), method, (n1, *notes), inp, c2),
        "gamma": Measurement("gamma", gamma, "°", math.hypot(e1, e2), "angle between the two lattice vectors",
                             (n1,), inp, {"a1 direction": e1, "a2 direction": e2}),
    }
