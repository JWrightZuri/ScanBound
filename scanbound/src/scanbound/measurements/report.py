"""
Measurement tables for whole images and projects (the same numbers the app shows).

    rows = measure_image(project.annotations.for_image(img_id), project.processed_image(img_id))
    df = measure_project(project)                      # one row per measurement, every image

Only ground truth is measured by default.  Orientations are also given relative to the image's
reference direction: its reference axis (projects made before the two tools were merged) or else
the a1 vector of its first lattice.  Lattice vectors are paired by their ``lattice`` attribute and
told apart by ``role`` ("a1" / "a2"; drawing order for older annotations).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from ..annotations import GROUND_TRUTH, Annotation
from .geometric import DEFAULT_MODEL, UncertaintyModel, lattice, measure_annotation
from .measurement import Measurement

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from ..model import Image
    from ..project import Project


def lattice_pairs(annotations: Iterable[Annotation], statuses: Sequence[str] = GROUND_TRUTH
                  ) -> dict[str, list[Annotation]]:
    """{lattice name: [a1, a2]} (or [a1] while the second vector is missing), in name order."""
    groups: dict[str, list[Annotation]] = defaultdict(list)
    for a in annotations:
        if a.kind == "lattice_vector" and a.status in statuses:
            groups[str(a.attrs.get("lattice", ""))].append(a)
    order = {"a1": 0, "a2": 1}
    out = {}
    for name in sorted(groups):
        vecs = groups[name]
        vecs = sorted(vecs, key=lambda a: (order.get(a.attrs.get("role"), 2), a.created))
        out[name] = vecs[:2]
    return out


def reference_axis(annotations: Iterable[Annotation]) -> Optional[Annotation]:
    """The annotation that gives the image's reference direction: a reference axis if the image has
    one, else a1 of its first lattice."""
    anns = list(annotations)
    for a in anns:
        if a.kind == "reference_axis" and a.status in GROUND_TRUTH:
            return a
    for vecs in lattice_pairs(anns).values():
        if vecs:
            return vecs[0]
    return None


def measure_all(annotations: Sequence[Annotation], image: Optional["Image"], *, fold: int = 1,
                model: UncertaintyModel = DEFAULT_MODEL,
                statuses: Sequence[str] = GROUND_TRUTH) -> list[tuple[Annotation, dict[str, Measurement]]]:
    """[(annotation, {name: Measurement})] for the annotations of one image (lattice pairs included
    as an extra entry for the first vector of each pair)."""
    ref = reference_axis(annotations)
    out = []
    for a in annotations:
        if a.status not in statuses:
            continue
        own = ref is not None and (a.id == ref.id or a.kind == "reference_axis")
        ms = measure_annotation(a, image, model=model, reference=None if own else ref, fold=fold)
        out.append((a, ms))
    for name, vecs in lattice_pairs(annotations, statuses).items():
        if len(vecs) == 2:
            out.append((vecs[0], {f"{name} {k}": m for k, m in lattice(vecs[0], vecs[1], image, model=model).items()}))
    return out


def measure_image(annotations: Sequence[Annotation], image: Optional["Image"], **kw) -> list[dict]:
    rows = []
    for a, ms in measure_all(annotations, image, **kw):
        for name, m in ms.items():
            rows.append({"image": a.image_id, "annotation": a.id, "kind": a.kind, "label": a.label, "status": a.status,
                         "measurement": name, "value": m.value, "uncertainty": m.uncertainty, "unit": m.unit,
                         "method": m.method, "assumptions": "; ".join(m.assumptions)})
    return rows


def measure_project(project: "Project", image_ids: Optional[Iterable[str]] = None, **kw) -> "pd.DataFrame":
    import pandas as pd
    ids = list(image_ids) if image_ids is not None else project.annotations.image_ids()
    rows: list[dict] = []
    for i in ids:
        anns = project.annotations.for_image(i)
        if not anns:
            continue
        try:
            img = project.processed_image(i)
        except Exception:  # noqa: BLE001 - measured in pixels when the file cannot be read
            img = None
        rows += measure_image(anns, img, **kw)
    return pd.DataFrame(rows, columns=["image", "annotation", "kind", "label", "status", "measurement", "value",
                                       "uncertainty", "unit", "method", "assumptions"])
