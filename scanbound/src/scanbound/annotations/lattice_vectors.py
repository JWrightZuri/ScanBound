"""
Lattice vectors typed in as numbers, and lattice-vector annotations turned into numbers.

A lattice of an image is a pair of ``lattice_vector`` annotations with the same ``attrs["lattice"]``
and roles ``a1`` / ``a2``, drawn from a common origin (pixel coordinates).  Values are physical:
lengths in the image's unit (pixels when uncalibrated) and angles in degrees counter-clockwise from
the image x axis (0° right, 90° up, the convention of all measured angles), computed from the
physical vector so that non-square pixels give the true angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .annotation import Annotation
from .geometry import Segment

Scale = tuple[float, float]


@dataclass(frozen=True)
class LatticeValues:
    a1_length: float
    a1_angle: float              # degrees CCW from +x
    a2_length: float
    a2_angle: float
    origin: tuple[float, float]  # pixel coordinates of the common start
    unit: str = "px"

    @property
    def gamma(self) -> float:
        """Angle from a1 to a2, degrees in (-180, 180] (positive = a2 counter-clockwise of a1)."""
        g = (self.a2_angle - self.a1_angle) % 360.0
        return g - 360.0 if g > 180.0 else g

    def with_gamma(self, gamma: float) -> "LatticeValues":
        return LatticeValues(self.a1_length, self.a1_angle, self.a2_length, (self.a1_angle + gamma) % 360.0,
                             self.origin, self.unit)

    def swapped(self) -> "LatticeValues":
        return LatticeValues(self.a2_length, self.a2_angle, self.a1_length, self.a1_angle, self.origin, self.unit)

    def physical(self) -> tuple[np.ndarray, np.ndarray]:
        """(a1, a2) as physical vectors with y pointing down (the lattice-fit convention)."""
        return _phys(self.a1_length, self.a1_angle), _phys(self.a2_length, self.a2_angle)

    def problems(self) -> list[str]:
        out = []
        for n, v in (("|a1|", self.a1_length), ("|a2|", self.a2_length)):
            if not (math.isfinite(v) and v > 0):
                out.append(f"{n} must be a positive number.")
        g = abs(self.gamma)
        if min(g, 180.0 - g) < 5.0:
            out.append(f"a1 and a2 are (nearly) parallel (γ = {self.gamma:.1f}°): they must point along different "
                       "lattice directions.")
        return out


def _phys(length: float, angle_deg: float) -> np.ndarray:
    t = math.radians(angle_deg)
    return np.array([length * math.cos(t), -length * math.sin(t)])


def _angle(phys: np.ndarray) -> float:
    return math.degrees(math.atan2(-float(phys[1]), float(phys[0]))) % 360.0


def segment_physical(seg: Any, scale: Scale) -> np.ndarray:
    (x0, y0), (x1, y1) = seg.points[0], seg.points[-1]
    return np.array([(x1 - x0) * scale[0], (y1 - y0) * scale[1]])


def values_from_pair(vecs: Sequence[Annotation], scale: Scale, unit: str = "px") -> Optional[LatticeValues]:
    """Values of a drawn pair [a1, a2] (None unless both are there)."""
    if len(vecs) < 2:
        return None
    p1, p2 = (segment_physical(v.geometry, scale) for v in vecs[:2])
    o = vecs[0].geometry.points[0]
    return LatticeValues(float(np.hypot(*p1)), _angle(p1), float(np.hypot(*p2)), _angle(p2),
                         (float(o[0]), float(o[1])), unit)


def values_from_fit(fit: Any) -> LatticeValues:
    """Values of a lattice fit (its a1, a2 and origin)."""
    o = fit.to_pixel_from_phys(fit.origin)[0]
    a1, a2 = np.asarray(fit.a1, float), np.asarray(fit.a2, float)
    return LatticeValues(float(np.hypot(*a1)), _angle(a1), float(np.hypot(*a2)), _angle(a2),
                         (float(o[0]), float(o[1])), fit.unit)


def segments(values: LatticeValues, scale: Scale) -> tuple[Segment, Segment]:
    """The two segments (pixel coordinates) the values describe."""
    ox, oy = values.origin
    out = []
    for p in values.physical():
        out.append(Segment([(ox, oy), (ox + p[0] / scale[0], oy + p[1] / scale[1])]))
    return out[0], out[1]


def annotations_for(values: LatticeValues, image_id: str, scale: Scale, lattice: str,
                    existing: Sequence[Annotation] = ()) -> tuple[list[Annotation], list[Annotation]]:
    """(new, updated) annotations that make ``lattice`` of the image equal to ``values``: existing
    vectors keep their ids, labels and status; missing ones are created as manual annotations."""
    s1, s2 = segments(values, scale)
    by_role = {a.attrs.get("role"): a for a in existing}
    if len(existing) >= 1 and "a1" not in by_role:
        by_role["a1"] = existing[0]
    if len(existing) >= 2 and "a2" not in by_role:
        by_role["a2"] = existing[1]
    new, updated = [], []
    for role, seg in (("a1", s1), ("a2", s2)):
        old = by_role.get(role)
        attrs = {"lattice": lattice, "role": role, "typed": True}
        if old is None:
            new.append(Annotation(image_id, "lattice_vector", seg, status="manual", attrs=attrs))
        else:
            updated.append(old.evolve(geometry=seg, attrs={**old.attrs, **attrs}))
    return new, updated


def physical_pair(annotations: Sequence[Annotation], scale: Scale, lattice: Optional[str] = None
                  ) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
    """(a1, a2, lattice name) as physical vectors (y down) from the image's lattice-vector
    annotations: the named lattice, else the first complete one.  None when there is no complete pair."""
    from ..measurements.report import lattice_pairs
    pairs = lattice_pairs([a for a in annotations if a.status != "rejected"],
                          statuses=("manual", "accepted", "proposed"))
    for name, vecs in pairs.items():
        if (lattice is None or name == lattice) and len(vecs) >= 2:
            return segment_physical(vecs[0].geometry, scale), segment_physical(vecs[1].geometry, scale), name
    return None
