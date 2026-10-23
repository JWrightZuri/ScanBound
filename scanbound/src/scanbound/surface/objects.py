"""
The objects the surface analyses work on: molecules, adatoms, defects ... taken from annotations.

    objs = objects_from_annotations(store.for_image(image_id), image, labels={"molecule"})
    objs[0].xy, objs[0].sigma_px, objs[0].orientation, objs[0].handedness

An object is a feature annotation reduced to what the statistics need: a centre (pixels) with a
placement uncertainty, an orientation with its uncertainty when one is known, a handedness when
one is known, and a link back to the annotation (``id``) so every number can be traced and the
annotation corrected.

Handedness
----------
Chirality is kept separate from orientation: an arrow does not tell left from right.  It comes
from the annotation's ``attrs["handedness"]`` ("L" / "R", set in the app from the context menu)
or, when that is missing, from a label -> handedness map (``chirality={"mol-L": "L", "mol-R": "R"}``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..annotations import Annotation, GROUND_TRUTH
from ..annotations.geometry import Segment
from ..measurements.geometric import DEFAULT_MODEL, UncertaintyModel

HANDS = ("L", "R")


@dataclass
class SurfaceObject:
    id: str
    xy: np.ndarray                       # (2,) pixel centre
    sigma_px: float                      # placement uncertainty of the centre (1σ, pixels)
    label: Optional[str] = None
    status: str = "manual"
    orientation: Optional[float] = None  # screen degrees
    orientation_sigma: Optional[float] = None
    handedness: Optional[str] = None     # "L" / "R" / None
    area_px: float = 0.0
    confidence: Optional[float] = None   # detector confidence when proposed
    attrs: dict = field(default_factory=dict)

    @property
    def x(self) -> float:
        return float(self.xy[0])

    @property
    def y(self) -> float:
        return float(self.xy[1])


def handedness_of(ann: Annotation, chirality: Optional[dict[str, str]] = None) -> Optional[str]:
    h = ann.attrs.get("handedness")
    if isinstance(h, str) and h.upper() in HANDS:
        return h.upper()
    if chirality and ann.label in chirality:
        v = str(chirality[ann.label]).upper()
        return v if v in HANDS else None
    return None


def objects_from_annotations(annotations: Iterable[Annotation], image: Any = None, *,
                             labels: Optional[Iterable[str]] = None, exclude_labels: Iterable[str] = (),
                             statuses: Sequence[str] = GROUND_TRUTH, kinds: Sequence[str] = ("feature",),
                             chirality: Optional[dict[str, str]] = None,
                             model: UncertaintyModel = DEFAULT_MODEL) -> list[SurfaceObject]:
    """
    Feature annotations as surface objects.  ``labels``: only these (None = all labels);
    ``statuses``: ground truth by default (manual and accepted), so unreviewed proposals do not
    enter statistics silently; pass ``("manual", "accepted", "proposed")`` to include them.
    """
    anns = list(annotations)
    keep_labels = None if labels is None else set(labels)
    excl = set(exclude_labels)
    vectors = {a.parent: a for a in anns if a.kind == "orientation_vector" and a.parent}
    scale = image.pixel_size if image is not None and getattr(image, "calibrated", False) else (1.0, 1.0)
    out = []
    for a in anns:
        if a.kind not in kinds or a.status not in statuses:
            continue
        if keep_labels is not None and a.label not in keep_labels:
            continue
        if a.label in excl:
            continue
        sigma, _ = model.placement(a)
        cx, cy = a.geometry.centroid()
        osig = None
        if a.orientation is not None:
            v = vectors.get(a.id)
            if v is not None and isinstance(v.geometry, Segment):
                dx, dy = v.geometry.delta()
                L = math.hypot(dx * scale[0], dy * scale[1])
                s_v, _ = model.placement(v)
                if L > 0:
                    osig = math.degrees(math.sqrt(2) * s_v * math.sqrt(scale[0] * scale[1]) / L)
        out.append(SurfaceObject(a.id, np.array([cx, cy], float), float(sigma), a.label, a.status, a.orientation, osig,
                                 handedness_of(a, chirality), float(a.geometry.area()), a.origin.confidence,
                                 dict(a.attrs)))
    return out


def phys(points_px: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    return np.asarray(points_px, float).reshape(-1, 2) * np.asarray(scale, float)


def distance_to_polylines(P: np.ndarray, lines: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(distance, index of the nearest line) from each point to a set of polylines (same units)."""
    P = np.asarray(P, float).reshape(-1, 2)
    best = np.full(len(P), np.inf)
    which = np.full(len(P), -1)
    for k, L in enumerate(lines):
        L = np.asarray(L, float).reshape(-1, 2)
        if len(L) == 1:
            d = np.hypot(*(P - L[0]).T)
        else:
            a, b = L[:-1], L[1:]
            ab = b - a
            t = ((P[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-300)[None]
            t = np.clip(t, 0, 1)
            proj = a[None] + t[..., None] * ab[None]
            d = np.hypot(*(P[:, None, :] - proj).transpose(2, 0, 1)).min(axis=1)
        closer = d < best
        best[closer] = d[closer]
        which[closer] = k
    return best, which


def nearest(P: np.ndarray, Q: np.ndarray, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Distance and index of the nearest point of Q for every point of P."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, float).reshape(-1, 2)
    Q = np.asarray(Q, float).reshape(-1, 2)
    if len(Q) == 0 or (exclude_self and len(Q) < 2):
        return np.full(len(P), np.inf), np.full(len(P), -1)
    t = cKDTree(Q)
    if exclude_self:
        d, j = t.query(P, 2)
        return d[:, 1], j[:, 1]
    d, j = t.query(P)
    return d, j
