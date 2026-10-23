"""
Train a YOLO model on a dataset made by ``export_yolo`` (or any YOLO dataset).

    save_dir = train("datasets/gnr/dataset.yaml", weights="yolo26n.pt", epochs=100, device="cpu")

Next to the trained weights it writes ``scanbound_train.json``: the training settings and the
dataset manifest's render settings, classes and image list, so predictions use the same rendering
and every model can be traced to the annotations it learned from.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..model.provenance import software_version
from .dataset import MANIFEST
from .predict import TRAIN_INFO


def train(data_yaml: str, *, weights: str = "yolo26n.pt", epochs: int = 100, imgsz: int = 640, batch: int = 16,
          device: str = "cpu", project: Optional[str] = None, name: str = "detector", patience: int = 50,
          workers: int = 4, on_epoch: Optional[Callable[[int, int, dict], None]] = None, **extra: Any) -> str:
    """Run training; returns the run's save directory (containing ``weights/best.pt``).

    ``on_epoch(epoch, epochs, metrics)`` is called after every epoch (for progress bars); ``extra`` is
    passed to ``ultralytics.YOLO.train`` unchanged (augmentation settings etc.).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    if on_epoch is not None:
        def _cb(trainer: Any) -> None:
            on_epoch(int(trainer.epoch) + 1, int(trainer.epochs), dict(getattr(trainer, "metrics", {}) or {}))
        model.add_callback("on_fit_epoch_end", _cb)
    settings = {"data": os.path.abspath(data_yaml), "epochs": epochs, "imgsz": imgsz, "batch": batch,
                "device": device, "name": name, "patience": patience, "workers": workers, **extra}
    if project:
        settings["project"] = os.path.abspath(project)
    results = model.train(pretrained=True, plots=True, save=True, exist_ok=False, **settings)
    save_dir = str(getattr(results, "save_dir", None) or model.trainer.save_dir)
    write_train_info(save_dir, data_yaml, weights, settings)
    return save_dir


def write_train_info(save_dir: str, data_yaml: str, weights: str, settings: dict) -> str:
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), MANIFEST)
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    info = {
        "format": "scanbound_train", "version": 1, "software": software_version(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_weights": weights, "settings": settings, "imgsz": settings.get("imgsz"),
        "classes": manifest.get("classes"), "task": manifest.get("task"), "render": manifest.get("render"),
        "dataset": {"manifest": manifest_path if manifest else None, "seed": manifest.get("seed"),
                    "images": [{"image_id": e["image_id"], "split": e["split"], "sha256": e.get("sha256")}
                               for e in manifest.get("images", [])]},
    }
    path = os.path.join(save_dir, TRAIN_INFO)
    os.makedirs(save_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    return path
