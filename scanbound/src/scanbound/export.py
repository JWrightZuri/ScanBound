"""Writing detection results: annotation CSV (the labelling page's format), feature tables, overlays."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Union

import pandas as pd

from .detect import Detection

ANNOTATION_COLUMNS = ["objectName", "type", "x1", "y1", "dx", "dy", "file", "label"]
PathLike = Union[str, "os.PathLike[str]"]


def annotations_frame(detections: Iterable[Detection]) -> pd.DataFrame:
    """All labelled ROIs of several images in the ``annotations_df`` layout."""
    rows: list[list] = []
    for d in detections:
        rows.extend(d.annotation_rows())
    df = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    df["objectName"] = df["objectName"].astype(str)
    return df


def features_frame(detections: Iterable[Detection]) -> pd.DataFrame:
    """Feature tables of several images stacked, with ``file`` and ``label`` columns."""
    parts = []
    for d in detections:
        if d.features is None or d.features.empty:
            continue
        f = d.features.copy()
        f.insert(0, "label", [d.labels.get(int(i)) for i in f.index])
        f.insert(0, "file", os.path.basename(d.path) if d.path else "")
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def ribbons_frame(detections: Iterable[Detection]) -> pd.DataFrame:
    parts = []
    for d in detections:
        if d.ribbons:
            t = d.ribbons_table().reset_index()
            t.insert(0, "file", os.path.basename(d.path) if d.path else "")
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def save_overlay(detection: Detection, path: PathLike) -> None:
    import cv2
    rgb = detection.overlay()
    if not cv2.imwrite(os.fspath(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {os.fspath(path)}")


def write_results(detections: list[Detection], out_dir: PathLike, overlays: bool = False) -> list[str]:
    """annotations.csv, features.csv, ribbons.csv (if analysed), summary.json and optional overlays."""
    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)
    written = []

    def target(name: str) -> str:
        p = os.path.join(out, name)
        written.append(p)
        return p

    annotations_frame(detections).to_csv(target("annotations.csv"), index=False)
    feats = features_frame(detections)
    if not feats.empty:
        feats.to_csv(target("features.csv"), index=False)
    ribbons = ribbons_frame(detections)
    if not ribbons.empty:
        ribbons.to_csv(target("ribbons.csv"), index=False)
    with open(target("summary.json"), "w", encoding="utf-8") as fh:
        json.dump([d.summary() for d in detections], fh, indent=2)
    if overlays:
        for d in detections:
            stem = os.path.splitext(os.path.basename(d.path or "image"))[0]
            save_overlay(d, target(f"{stem}_overlay.png"))
    return written
