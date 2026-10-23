"""
How good is a detector?  Compare its proposals with ground truth.

    ev = evaluate(gt_annotations, predicted_annotations, iou_threshold=0.5, iou="box")
    print(ev.table())                 # precision / recall / F1 / AP per class and overall

Matching is greedy by confidence (the usual detection protocol): each prediction, most confident
first, takes the unmatched ground truth of the same class (or any class with ``class_aware=False``)
it overlaps most, if the IoU reaches the threshold.  Predictions without a confidence (the adapt
pipeline) count as equally confident; their AP is then the precision at full recall of the set.

IoU is computed on bounding boxes (``"box"``) or on the filled shapes (``"mask"``), image by image.

``review_stats`` measures the detector from the review instead: how many proposals a person
accepted unchanged, accepted after correcting them, or rejected.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from ..annotations import Annotation, AnnotationStore, Geometry, bbox_iou

IOU_MODES = ("box", "mask")


def geometry_iou(a: Geometry, b: Geometry, mode: str = "box") -> float:
    if mode == "box":
        return bbox_iou(a.bbox(), b.bbox())
    if mode != "mask":
        raise ValueError(f"iou must be one of {IOU_MODES}")
    if not (a.closed and b.closed):
        return bbox_iou(a.bbox(), b.bbox())
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
        return 0.0
    x0, y0 = math.floor(min(ax0, bx0)), math.floor(min(ay0, by0))
    x1, y1 = math.ceil(max(ax1, bx1)), math.ceil(max(ay1, by1))
    scale = max(1.0, 64.0 / max(1, min(x1 - x0, y1 - y0)))      # rasterise small shapes finer
    shape = (int(math.ceil((y1 - y0) * scale)) + 1, int(math.ceil((x1 - x0) * scale)) + 1)
    m = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    ma, mb = a.transformed(m).mask(shape), b.transformed(m).mask(shape)
    union = np.count_nonzero(ma | mb)
    return float(np.count_nonzero(ma & mb) / union) if union else 0.0


@dataclass
class Match:
    pred: Annotation
    gt: Optional[Annotation]
    iou: float

    @property
    def tp(self) -> bool:
        return self.gt is not None


def match(gt: Sequence[Annotation], pred: Sequence[Annotation], iou_threshold: float = 0.5, iou: str = "box",
          class_aware: bool = True) -> tuple[list[Match], list[Annotation]]:
    """(one Match per prediction, unmatched ground truth) for annotations of *one image*."""
    order = sorted(pred, key=lambda p: -(p.origin.confidence if p.origin.confidence is not None else 1.0))
    free = list(gt)
    out = []
    for p in order:
        best, best_iou = None, iou_threshold
        for g in free:
            if class_aware and g.label != p.label:
                continue
            v = geometry_iou(p.geometry, g.geometry, iou)
            if v >= best_iou:
                best, best_iou = g, v
        if best is not None:
            free.remove(best)
        out.append(Match(p, best, best_iou if best is not None else 0.0))
    return out, free


def average_precision(scores: Sequence[float], tp: Sequence[bool], n_gt: int) -> float:
    """Area under the interpolated precision-recall curve (all points, as in VOC 2010+ / COCO)."""
    if n_gt == 0:
        return float("nan")
    if not len(scores):
        return 0.0
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    hits = np.asarray(tp, bool)[order]
    ctp, cfp = np.cumsum(hits), np.cumsum(~hits)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    r = np.concatenate([[0.0], recall, [recall[-1]]])
    p = np.concatenate([[1.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


@dataclass
class ClassMetrics:
    label: Optional[str]
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0
    ap: float = float("nan")
    mean_iou: float = float("nan")

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp

    @property
    def fn(self) -> int:
        return self.n_gt - self.tp

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else float("nan")

    @property
    def f1(self) -> float:
        d = self.n_pred + self.n_gt
        return 2 * self.tp / d if d else float("nan")

    def to_dict(self) -> dict:
        return {"label": self.label, "n_gt": self.n_gt, "n_pred": self.n_pred, "tp": self.tp, "fp": self.fp,
                "fn": self.fn, "precision": self.precision, "recall": self.recall, "f1": self.f1, "ap": self.ap,
                "mean_iou": self.mean_iou}


@dataclass
class Evaluation:
    per_class: dict[Optional[str], ClassMetrics]
    overall: ClassMetrics
    iou_threshold: float
    iou: str
    matches: list[Match] = field(default_factory=list, repr=False)
    missed: list[Annotation] = field(default_factory=list, repr=False)

    @property
    def map(self) -> float:
        aps = [m.ap for m in self.per_class.values() if not math.isnan(m.ap)]
        return float(np.mean(aps)) if aps else float("nan")

    def false_positives(self) -> list[Annotation]:
        return [m.pred for m in self.matches if not m.tp]

    def to_dict(self) -> dict:
        return {"iou_threshold": self.iou_threshold, "iou": self.iou, "map": self.map,
                "overall": self.overall.to_dict(), "per_class": [m.to_dict() for m in self.per_class.values()]}

    def table(self) -> str:
        def f(v: float) -> str:
            return "  -  " if math.isnan(v) else f"{v:5.3f}"
        rows = [f"{'class':<20} {'GT':>5} {'pred':>5} {'TP':>5} {'FP':>5} {'FN':>5}  prec  recall  F1    AP@{self.iou_threshold:g}"]
        for m in list(self.per_class.values()) + [self.overall]:
            name = "all" if m is self.overall else str(m.label)
            rows.append(f"{name[:20]:<20} {m.n_gt:>5} {m.n_pred:>5} {m.tp:>5} {m.fp:>5} {m.fn:>5}  "
                        f"{f(m.precision)} {f(m.recall)}  {f(m.f1)} {f(m.ap)}")
        return "\n".join(rows) + f"\nmAP@{self.iou_threshold:g} ({self.iou} IoU): {f(self.map)}"


def evaluate(gt: Iterable[Annotation], pred: Iterable[Annotation], iou_threshold: float = 0.5,
             iou: str = "box", class_aware: bool = True) -> Evaluation:
    """Per-class and overall metrics; annotations are grouped by ``image_id`` before matching."""
    gt_by, pr_by = defaultdict(list), defaultdict(list)
    for a in gt:
        gt_by[a.image_id].append(a)
    for a in pred:
        pr_by[a.image_id].append(a)
    matches: list[Match] = []
    missed: list[Annotation] = []
    for image_id in set(gt_by) | set(pr_by):
        m, free = match(gt_by[image_id], pr_by[image_id], iou_threshold, iou, class_aware)
        matches += m
        missed += free
    labels = sorted({a.label for v in gt_by.values() for a in v} | {m.pred.label for m in matches},
                    key=lambda s: (s is None, s or ""))
    gts = [a for v in gt_by.values() for a in v]

    def metrics(label: Optional[str], ms: list[Match], n_gt: int) -> ClassMetrics:
        scores = [m.pred.origin.confidence if m.pred.origin.confidence is not None else 1.0 for m in ms]
        ious = [m.iou for m in ms if m.tp]
        return ClassMetrics(label, n_gt, len(ms), sum(m.tp for m in ms), average_precision(scores, [m.tp for m in ms], n_gt),
                            float(np.mean(ious)) if ious else float("nan"))

    per_class = {}
    if class_aware:
        for lab in labels:
            per_class[lab] = metrics(lab, [m for m in matches if m.pred.label == lab],
                                     sum(1 for a in gts if a.label == lab))
    overall = metrics(None, matches, len(gts))
    if not class_aware:
        per_class[None] = overall
    return Evaluation(per_class, overall, iou_threshold, iou, matches, missed)


def review_stats(store: AnnotationStore, run: Optional[str] = None, name: Optional[str] = None) -> dict:
    """How the proposals of one run (or of one detector name) fared in review."""
    anns = store.query(lambda a: a.origin.method != "manual" and (run is None or a.origin.run == run)
                       and (name is None or a.origin.name == name))
    out = {"total": len(anns), "pending": 0, "accepted": 0, "accepted_edited": 0, "rejected": 0}
    for a in anns:
        if a.status == "proposed":
            out["pending"] += 1
        elif a.status == "accepted":
            out["accepted_edited" if a.edited else "accepted"] += 1
        elif a.status == "rejected":
            out["rejected"] += 1
    reviewed = out["accepted"] + out["accepted_edited"] + out["rejected"]
    out["acceptance_rate"] = (out["accepted"] + out["accepted_edited"]) / reviewed if reviewed else float("nan")
    out["unchanged_rate"] = out["accepted"] / reviewed if reviewed else float("nan")
    return out
