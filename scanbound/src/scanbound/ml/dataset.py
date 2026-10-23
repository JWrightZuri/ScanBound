"""
Export ground-truth annotations to a YOLO dataset.

    report = export_yolo(project, "datasets/gnr", task="segment", val_ratio=0.2, seed=42)
    report.yaml_path          # pass to train()

Only ground truth is exported: manual annotations and accepted proposals of kind "feature" with a
label.  Proposals still waiting for review and rejected ones are never training data.

The split is by *file*, so two channels of one scan never end up on both sides.  Every exported
picture is listed in ``manifest.json`` with the source file's checksum, processing chain, display
settings and annotation ids - enough to trace a training sample back to the raw data and make it
again.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np

from ..annotations import GROUND_TRUTH, Annotation, Ellipse, Point, Rect
from ..model.provenance import software_version
from .render import RenderSpec, render_for_model

if TYPE_CHECKING:  # pragma: no cover
    from ..project import Project

TASKS = ("detect", "segment")
MANIFEST = "manifest.json"


@dataclass
class DatasetReport:
    root: str
    yaml_path: str
    classes: list[str]
    counts: dict[str, int] = field(default_factory=dict)      # split -> images
    instances: dict[str, int] = field(default_factory=dict)   # class -> instances
    skipped: list[str] = field(default_factory=list)          # human-readable reasons

    def summary(self) -> str:
        parts = [f"{n} {s}" for s, n in self.counts.items()]
        inst = ", ".join(f"{k}: {v}" for k, v in self.instances.items())
        return f"{' / '.join(parts)} images; {inst}" + (f"; {len(self.skipped)} skipped" if self.skipped else "")


def _yolo_line(ann: Annotation, cls: int, w: int, h: int, task: str) -> Optional[str]:
    g = ann.geometry
    if isinstance(g, Point) or not g.closed:
        return None
    x0, y0, x1, y1 = g.bbox()
    x0, x1 = max(0.0, x0), min(float(w), x1)
    y0, y1 = max(0.0, y0), min(float(h), y1)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    if task == "detect" or (type(g) is Rect):
        if task == "segment":                              # a rect is its own 4-point polygon
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        else:
            return (f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                    f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    else:
        pts = g.outline(64) if isinstance(g, Ellipse) else g.outline()
    pts = np.column_stack([np.clip(pts[:, 0] / w, 0, 1), np.clip(pts[:, 1] / h, 0, 1)])
    return f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)


def ground_truth(project: "Project", image_id: str, labels: Optional[Sequence[str]] = None) -> list[Annotation]:
    anns = project.annotations.for_image(image_id, kinds=("feature",), statuses=GROUND_TRUTH)
    return [a for a in anns if a.label and (labels is None or a.label in labels)]


def export_yolo(project: "Project", out_dir: str, *, image_ids: Optional[Iterable[str]] = None,
                labels: Optional[Sequence[str]] = None, task: str = "detect", val_ratio: float = 0.2,
                seed: int = 42, render: RenderSpec = RenderSpec(), include_empty: bool = False) -> DatasetReport:
    """Write ``images/{train,val}``, ``labels/{train,val}``, ``dataset.yaml`` and ``manifest.json``.

    ``include_empty`` also exports images marked as reviewed that have no ground truth (negative
    examples: "there is nothing here"), which reduces false positives.
    """
    from PIL import Image as PILImage

    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    ids = list(image_ids) if image_ids is not None else project.image_ids()
    report_skipped: list[str] = []
    selected: dict[str, list[Annotation]] = {}
    for i in ids:
        gt = ground_truth(project, i, labels)
        reviewed = project.images.get(i) is not None and project.images[i].reviewed
        if gt or (include_empty and reviewed):
            selected[i] = gt
    classes = sorted({a.label for v in selected.values() for a in v if a.label}) if labels is None else list(labels)
    if not selected or not classes:
        raise ValueError("no ground-truth annotations to export (manual or accepted features with a label)")
    cls_of = {c: k for k, c in enumerate(classes)}

    files = sorted({project.split_image_id(i)[0] for i in selected})
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(round(len(files) * val_ratio)) if len(files) > 1 else 0
    if val_ratio > 0 and len(files) > 1:
        n_val = max(1, min(n_val, len(files) - 1))
    split_of = {f: ("val" if k < n_val else "train") for k, f in enumerate(files)}

    for s in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", s), exist_ok=True)

    counts = {"train": 0, "val": 0}
    instances = {c: 0 for c in classes}
    entries = []
    used_names: set[str] = set()
    for image_id, anns in selected.items():
        try:
            img = project.processed_image(image_id)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not stop the export
            report_skipped.append(f"{image_id}: cannot load ({exc})")
            continue
        entry = project.entry(image_id)
        h, w = img.shape
        rgb = render_for_model(img, entry.display, render)
        split = split_of[project.split_image_id(image_id)[0]]
        stem = _unique(_stem(image_id), used_names)
        lines, kept = [], []
        for a in anns:
            line = _yolo_line(a, cls_of[a.label], w, h, task)
            if line is None:
                report_skipped.append(f"{image_id}: annotation {a.id} ({a.geometry.type}) has no area inside the image")
                continue
            lines.append(line)
            kept.append(a.id)
            instances[a.label] += 1
        ext = "jpg" if render.image_format == "jpg" else "png"
        pil = PILImage.fromarray(rgb)
        pil.save(os.path.join(out_dir, "images", split, f"{stem}.{ext}"), **({"quality": 95} if ext == "jpg" else {}))
        with open(os.path.join(out_dir, "labels", split, f"{stem}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        counts[split] += 1
        entries.append({
            "image": f"images/{split}/{stem}.{ext}", "image_id": image_id, "split": split, "size": [w, h],
            "pixel_size": list(img.pixel_size), "pixel_unit": img.x.unit,
            "source": img.source.to_dict() if img.source else None, "sha256": project.files[entry.file].sha256 or project.hash_file(entry.file),
            "chain": entry.chain.to_dict(), "display": entry.display.to_dict(), "annotations": kept,
        })

    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\n")
        fh.write("names:\n" + "".join(f"  {k}: {json.dumps(c)}\n" for k, c in enumerate(classes)))
    manifest = {
        "format": "scanbound_dataset", "version": 1, "software": software_version(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "task": task,
        "classes": classes, "render": render.to_dict(), "seed": seed, "val_ratio": val_ratio,
        "statuses": list(GROUND_TRUTH), "project": project.path, "images": entries, "skipped": report_skipped,
    }
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return DatasetReport(out_dir, yaml_path, classes, counts, instances, report_skipped)


def _stem(image_id: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in image_id.replace("::", "__"))
    return keep.strip("._") or "image"


def _unique(stem: str, used: set[str]) -> str:
    out, k = stem, 1
    while out.lower() in used:
        k += 1
        out = f"{stem}_{k}"
    used.add(out.lower())
    return out
