"""
The annotation record.

An :class:`Annotation` ties a geometry to an image, a kind (its meaning), a label (its class) and
a review status.  Automated results are annotations too: a detector *proposes*, a person
*accepts*, *rejects* or edits.  The :class:`Origin` records who or what made it, so a training set
can say exactly which annotations are human ground truth.

Statuses
--------
``manual``    drawn by a person (ground truth)
``proposed``  made by a detector / model, not reviewed yet
``accepted``  proposed, then confirmed (possibly after editing; ``edited`` says so)
``rejected``  proposed, then judged wrong (kept, so evaluations and audits can see it)
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from ..model.provenance import now_iso
from .geometry import Geometry, geometry_from_dict

STATUSES = ("manual", "proposed", "accepted", "rejected")
GROUND_TRUTH = ("manual", "accepted")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Origin:
    """Who or what created an annotation."""

    method: str = "manual"            # "manual" / "detector" / "model" / "import"
    name: str = ""                    # "adapt pipeline", "yolo:best.pt", "annotations.csv" ...
    version: str = ""
    run: str = ""                     # id of the run that produced it (links to the project's run log)
    confidence: Optional[float] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"method": self.method}
        for k in ("name", "version", "run"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.details:
            d["details"] = self.details
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Origin":
        d = d or {}
        conf = d.get("confidence")
        return cls(str(d.get("method", "manual")), str(d.get("name", "")), str(d.get("version", "")),
                   str(d.get("run", "")), None if conf is None else float(conf), dict(d.get("details") or {}))


MANUAL = Origin()


@dataclass(frozen=True)
class Annotation:
    image_id: str
    kind: str
    geometry: Geometry
    label: Optional[str] = None
    status: str = "manual"
    origin: Origin = MANUAL
    orientation: Optional[float] = None    # degrees, 0..360 counter-clockwise from +x (features)
    parent: Optional[str] = None           # id of the annotation this one belongs to
    attrs: dict = field(default_factory=dict)
    edited: bool = False                   # geometry or label changed after it was proposed
    note: str = ""
    id: str = field(default_factory=new_id)
    created: str = field(default_factory=now_iso)
    modified: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {STATUSES}")
        if self.orientation is not None:
            o = float(self.orientation)
            object.__setattr__(self, "orientation", None if not math.isfinite(o) else o % 360.0)
        if self.label is not None and not str(self.label).strip():
            object.__setattr__(self, "label", None)

    @property
    def is_ground_truth(self) -> bool:
        return self.status in GROUND_TRUTH

    @property
    def confidence(self) -> Optional[float]:
        return self.origin.confidence

    def evolve(self, **changes: Any) -> "Annotation":
        """A modified copy (``modified`` updated; proposals that are edited get ``edited=True``)."""
        if self.status == "proposed" and ({"geometry", "label"} & changes.keys()) and "edited" not in changes:
            changes["edited"] = True
        changes.setdefault("modified", now_iso())
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "image": self.image_id, "kind": self.kind,
                             "geometry": self.geometry.to_dict(), "status": self.status,
                             "origin": self.origin.to_dict(), "created": self.created}
        for k in ("label", "orientation", "parent", "note", "modified"):
            v = getattr(self, k)
            if v not in (None, ""):
                d[k] = v
        if self.attrs:
            d["attrs"] = self.attrs
        if self.edited:
            d["edited"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Annotation":
        return cls(image_id=str(d["image"]), kind=str(d["kind"]), geometry=geometry_from_dict(d["geometry"]),
                   label=d.get("label"), status=str(d.get("status", "manual")),
                   origin=Origin.from_dict(d.get("origin")), orientation=d.get("orientation"),
                   parent=d.get("parent"), attrs=dict(d.get("attrs") or {}), edited=bool(d.get("edited", False)),
                   note=str(d.get("note", "")), id=str(d.get("id") or new_id()), created=str(d.get("created", "")),
                   modified=str(d.get("modified", "")))
