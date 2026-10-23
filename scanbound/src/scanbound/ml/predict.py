"""
Apply a trained YOLO model; every detection becomes a *proposed* annotation with its confidence.

    anns = predict_annotations("runs/gnr/weights/best.pt", image, display=entry.display,
                               image_id=img_id, run_id=run.id)
    project.annotations.add(anns)

The image is rendered with the model's own ``RenderSpec`` (read from ``scanbound_train.json`` next
to the weights when the model was trained here), so it sees pictures made like its training data.
Segmentation models give polygons, detection models rectangles.  Loaded models are cached.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Optional, Sequence

import numpy as np

from ..annotations import Annotation, Origin, Polygon, Rect
from ..display import DisplaySettings
from ..model import Image
from ..model.provenance import file_digest, software_version
from .render import RenderSpec, render_for_model

TRAIN_INFO = "scanbound_train.json"


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> Any:
    from ultralytics import YOLO
    return YOLO(path)


def load_model(path: str) -> Any:
    path = os.path.abspath(path)
    return _load(path, os.path.getmtime(path))


def train_info(weights: str) -> dict:
    """What ``train`` recorded about a model (empty for models trained elsewhere)."""
    d = os.path.dirname(os.path.abspath(weights))
    for p in (os.path.join(d, TRAIN_INFO), os.path.join(os.path.dirname(d), TRAIN_INFO)):
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def predict_annotations(weights: str, image: Image, *, display: Optional[DisplaySettings] = None,
                        render: Optional[RenderSpec] = None, image_id: Optional[str] = None, conf: float = 0.25,
                        iou: float = 0.5, classes: Optional[Sequence[int]] = None, imgsz: Optional[int] = None,
                        run_id: str = "", device: Optional[str] = None) -> list[Annotation]:
    model = load_model(weights)
    info = train_info(weights)
    spec = render or RenderSpec.from_dict(info.get("render"))
    rgb = render_for_model(image, display, spec)
    kwargs: dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
    if classes is not None:
        kwargs["classes"] = list(classes)
    if imgsz:
        kwargs["imgsz"] = int(imgsz)
    elif info.get("imgsz"):
        kwargs["imgsz"] = int(info["imgsz"])
    if device:
        kwargs["device"] = device
    result = model.predict(source=rgb[..., ::-1].copy(), **kwargs)[0]     # ultralytics expects BGR arrays
    names = result.names or {}
    origin_base = {"weights": os.path.basename(weights), "weights_sha256": _digest(weights),
                   "render": spec.to_dict(), "conf": conf, "iou": iou}
    valid = image.valid
    out: list[Annotation] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    polys = result.masks.xy if getattr(result, "masks", None) is not None else None
    h, w = image.shape
    for k, (box, score, c) in enumerate(zip(boxes, scores, cls)):
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        geom: Any = Rect.from_corners(x0, y0, x1, y1)
        if polys is not None and k < len(polys) and len(polys[k]) >= 3:
            geom = Polygon(np.asarray(polys[k], float))
        attrs: dict[str, Any] = {"class_id": int(c)}
        inside = valid[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        if inside.size and not inside.all():
            attrs["note"] = f"{100 * (1 - inside.mean()):.0f}% of the box has no data"
        origin = Origin("model", f"yolo:{os.path.basename(weights)}", software_version(), run_id, float(score),
                        origin_base)
        out.append(Annotation(image_id or image.id, "feature", geom, label=str(names.get(int(c), c)),
                              status="proposed", origin=origin, attrs=attrs))
    return out


@lru_cache(maxsize=16)
def _digest_cached(path: str, mtime: float) -> str:
    return file_digest(path)


def _digest(path: str) -> str:
    try:
        return _digest_cached(os.path.abspath(path), os.path.getmtime(path))
    except OSError:
        return ""
