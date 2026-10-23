"""
What an annotation *means*.

All annotations share one geometry framework (:mod:`.geometry`); the kind decides which shapes
are allowed, how the annotation is validated and which measurements it gets.  A line segment can
be a measurement line, a molecule's orientation vector, an atomic lattice vector or the image's
reference axis - same geometry, different meaning and rules.

Adding a kind::

    register_kind(AnnotationKind(
        "step_edge", "Step edge", ("polyline", "segment"),
        "A terrace step; measured by its length and the height difference across it.",
        validators=(min_length(2.0),)))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:
    from .annotation import Annotation
    from .store import AnnotationStore


@dataclass(frozen=True)
class Issue:
    level: str          # "error" (cannot be used) / "warning" (usable, but check it)
    code: str
    message: str
    annotation: Optional[str] = None


# validator(annotation, context) -> issues; context gives the image shape and the store
Validator = Callable[["Annotation", "ValidationContext"], Sequence[Issue]]


@dataclass
class ValidationContext:
    image_shape: Optional[tuple[int, int]] = None
    store: Optional["AnnotationStore"] = None


@dataclass(frozen=True)
class AnnotationKind:
    key: str
    title: str
    geometries: tuple[str, ...]
    description: str = ""
    validators: tuple[Validator, ...] = ()
    directed: bool = False             # the geometry has a start and an end that matter
    feature: bool = False              # a physical object on the sample (counted, detected, trained on)
    unique_per_image: bool = False
    measurements: tuple[str, ...] = ()  # names of measurements offered (see scanbound.measurements)
    color: str = ""                    # default drawing colour when the label has none

    def allows(self, geometry_type: str) -> bool:
        return geometry_type in self.geometries


# ---- validators --------------------------------------------------------------------------

def _non_degenerate(ann: "Annotation", ctx: ValidationContext) -> list[Issue]:
    if ann.geometry.is_degenerate():
        what = "zero area" if ann.geometry.closed else "zero length"
        return [Issue("error", "degenerate", f"{ann.geometry.type} has {what}", ann.id)]
    return []


def _inside_image(ann: "Annotation", ctx: ValidationContext) -> list[Issue]:
    if ctx.image_shape is None:
        return []
    rows, cols = ctx.image_shape
    x0, y0, x1, y1 = ann.geometry.bbox()
    tol = 0.5
    if x1 < 0 or y1 < 0 or x0 > cols or y0 > rows:
        return [Issue("error", "outside", "lies completely outside the image", ann.id)]
    if x0 < -tol or y0 < -tol or x1 > cols + tol or y1 > rows + tol:
        return [Issue("warning", "partly_outside", "extends beyond the image border", ann.id)]
    return []


def min_length(px: float) -> Validator:
    def check(ann: "Annotation", ctx: ValidationContext) -> list[Issue]:
        L = ann.geometry.length()
        if 0 < L < px:
            return [Issue("warning", "short", f"only {L:.2g} px long: its direction / length is poorly defined",
                          ann.id)]
        return []
    return check


def _unique(ann: "Annotation", ctx: ValidationContext) -> list[Issue]:
    if ctx.store is None:
        return []
    others = [a for a in ctx.store.for_image(ann.image_id, kinds=(ann.kind,)) if a.id != ann.id
              and a.status != "rejected"]
    if others:
        return [Issue("error", "not_unique", f"an image has one {KINDS[ann.kind].title.lower()}; "
                                             f"{len(others)} other(s) exist", ann.id)]
    return []


def _parent_contains(ann: "Annotation", ctx: ValidationContext) -> list[Issue]:
    if ann.parent is None or ctx.store is None:
        return []
    parent = ctx.store.get(ann.parent)
    if parent is None:
        return [Issue("warning", "orphan", "belongs to an annotation that no longer exists", ann.id)]
    if parent.geometry.closed and not all(parent.geometry.contains(x, y) for x, y in ann.geometry.points):
        return [Issue("warning", "outside_parent", "is not inside the feature it belongs to", ann.id)]
    return []


def _lattice_pair(ann: "Annotation", ctx: ValidationContext) -> list[Issue]:
    if ctx.store is None:
        return []
    group = ann.attrs.get("lattice")
    if not group:
        return [Issue("warning", "no_group", "not assigned to a lattice: pair two lattice vectors to measure", ann.id)]
    mates = [a for a in ctx.store.for_image(ann.image_id, kinds=("lattice_vector",))
             if a.attrs.get("lattice") == group and a.status != "rejected"]
    if len(mates) > 2:
        return [Issue("error", "too_many", f"lattice {group!r} has {len(mates)} vectors (2 expected)", ann.id)]
    if len(mates) == 1:
        return [Issue("warning", "unpaired", f"lattice {group!r} needs a second vector to be measured", ann.id)]
    if len(mates) == 2:
        a, b = mates
        import math
        ang = abs((a.geometry.angle() - b.geometry.angle() + 180) % 360 - 180)   # type: ignore[attr-defined]
        if ang < 5 or ang > 175:
            return [Issue("error", "collinear", "the two lattice vectors are (nearly) parallel", ann.id)]
        if math.dist(a.geometry.start, b.geometry.start) > 0.1 * min(a.geometry.length(), b.geometry.length()):  # type: ignore[attr-defined]
            return [Issue("warning", "origin", "the two lattice vectors do not start at the same point", ann.id)]
    return []


KINDS: dict[str, AnnotationKind] = {}


def register_kind(kind: AnnotationKind, replace: bool = False) -> AnnotationKind:
    if kind.key in KINDS and not replace:
        raise ValueError(f"annotation kind {kind.key!r} already registered")
    KINDS[kind.key] = kind
    return kind


def kind(key: str) -> AnnotationKind:
    try:
        return KINDS[key]
    except KeyError:
        raise KeyError(f"unknown annotation kind {key!r}; known: {', '.join(KINDS)}") from None


_BASE = (_non_degenerate, _inside_image)

register_kind(AnnotationKind(
    "feature", "Feature", ("rect", "polygon", "ellipse", "point", "mask"),
    "An object on the surface (molecule, island, vacancy ...): counted, measured, detected and trained on.",
    _BASE, feature=True, measurements=("area", "perimeter", "size", "centroid", "orientation")))
register_kind(AnnotationKind(
    "centre_line", "Centre line", ("polyline", "segment"),
    "The centre line of an elongated feature such as a nanoribbon: measured by its contour length.",
    _BASE + (min_length(2.0),), feature=True, measurements=("length", "end_to_end", "straightness", "angle")))
register_kind(AnnotationKind(
    "measurement_line", "Measurement line", ("segment", "polyline"),
    "A ruler: distance between two points and the height profile along it.",
    _BASE + (min_length(1.0),), measurements=("length", "angle", "profile")))
register_kind(AnnotationKind(
    "orientation_vector", "Orientation vector", ("segment",),
    "The direction of a feature (tail -> head); usually belongs to a feature annotation.",
    _BASE + (min_length(2.0), _parent_contains), directed=True, measurements=("angle", "relative_angle")))
register_kind(AnnotationKind(
    "reference_axis", "Reference axis", ("segment",),
    "A high-symmetry direction of the substrate; orientations are measured against it. Kept for older "
    "projects: new reference directions are drawn as lattice vector a1.",
    _BASE + (min_length(5.0), _unique), directed=True, unique_per_image=True, measurements=("angle",),
    color="#F5E400"))
register_kind(AnnotationKind(
    "lattice_vector", "Lattice vector", ("segment",),
    "One of two primitive vectors a1, a2 of a surface lattice, paired by the 'lattice' attribute ('role' = "
    "a1 / a2). a1 of the first lattice is also the image's reference direction for orientations.",
    _BASE + (min_length(2.0), _lattice_pair), directed=True, measurements=("length", "angle", "lattice"),
    color="#F5E400"))

register_kind(AnnotationKind(
    "herringbone_axis", "Herringbone axis", ("segment",),
    "The direction of the Au(111) herringbone ridges in one domain; ribbon alignment is measured against the "
    "nearest one (draw one per domain).",
    _BASE + (min_length(3.0),), measurements=("angle",), color="#FF6E40"))

register_kind(AnnotationKind(
    "step_edge", "Step edge", ("polyline", "segment"),
    "A terrace step edge: bounds terraces; objects are measured by their distance to it.",
    _BASE + (min_length(2.0),), measurements=("length", "angle"), color="#FF7F0E"))
register_kind(AnnotationKind(
    "region", "Analysis region", ("polygon", "rect", "ellipse", "mask"),
    "An area for the surface analyses: fit a lattice or count objects inside it (its label names it, "
    "e.g. 'domain A'; the label 'exclude' keeps an area out of the analyses).",
    _BASE, measurements=("area", "perimeter"), color="#B07AFF"))


def validate(ann: "Annotation", ctx: Optional[ValidationContext] = None) -> list[Issue]:
    """Every issue of an annotation: geometry allowed for its kind, then the kind's own rules."""
    ctx = ctx or ValidationContext()
    k = kind(ann.kind)
    issues: list[Issue] = []
    if not k.allows(ann.geometry.type):
        issues.append(Issue("error", "geometry", f"a {k.title.lower()} cannot be a {ann.geometry.type} "
                                                 f"(allowed: {', '.join(k.geometries)})", ann.id))
        return issues
    for v in k.validators:
        issues.extend(v(ann, ctx))
    return issues


__all__ = ["AnnotationKind", "Issue", "KINDS", "ValidationContext", "kind", "min_length", "register_kind", "validate",
           "field"]
