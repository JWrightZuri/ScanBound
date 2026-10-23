"""
Automated results as *proposals*: annotations a person reviews, corrects, accepts or rejects.

    from scanbound.proposals import propose
    anns, det = propose(Detector(cfg), project.processed_image(img_id), display=entry.display)
    project.annotations.add(anns)          # status "proposed", origin = detector + run id

The detector sees the image as it is displayed (levels, gamma) by default - the same picture the
person tuned the pipeline on in the old app, without the JPG's 8-bit rounding and compression -
or the data min-max normalised (``input_mode="data"``).  Pixels without data are never part of an
object.  Each proposal stores the object's shape features, so rules can be re-applied later.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .annotations import Annotation, Mask, Origin, Point, Polygon, Polyline, Rect, Segment
from .detect import Detection, Detector
from .display import DisplaySettings, ImageStats as DisplayStats, normalized
from .model import Image
from .model.provenance import params_digest, software_version
from .roi import SegObject

INPUT_MODES = ("display", "data")


def detector_input(image: Image, display: Optional[DisplaySettings] = None,
                   mode: str = "display") -> tuple[np.ndarray, np.ndarray]:
    """(float32 image in [0, 1] with invalid samples filled, valid mask)."""
    valid = image.valid
    if mode == "display":
        s = display or DisplaySettings()
        t = normalized(image.data, DisplayStats(image.data), s)
    elif mode == "data":
        d = np.asarray(image.data, float)
        st = image.stats
        span = (st.maximum - st.minimum) or 1.0
        t = ((d - st.minimum) / span).astype(np.float32)
    else:
        raise ValueError(f"unknown input mode {mode!r}; expected one of {INPUT_MODES}")
    if not valid.all():
        fill = float(np.nanmedian(t)) if valid.any() else 0.0
        t = np.where(valid, t, fill).astype(np.float32)
    return np.ascontiguousarray(t, dtype=np.float32), valid


def object_geometry(obj: SegObject) -> tuple[str, object]:
    """(annotation kind, geometry) for a segmented object."""
    if obj.roi_type == "RectROI" and obj.rect is not None:
        x, y, w, h = obj.rect
        return "feature", Rect(float(x), float(y), float(w), float(h))
    if obj.roi_type == "MaskROI" and obj.pixels is not None:
        x0, y0, _, _ = obj.bbox
        return "feature", Mask(int(x0), int(y0), obj.pixels)
    pts = np.asarray(obj.points, float)
    if obj.roi_type == "PointROI":
        return "feature", Point(float(pts[0, 0]), float(pts[0, 1]))
    if obj.roi_type == "LineSegmentROI":
        if len(pts) < 2:
            return "feature", Point(float(pts[0, 0]), float(pts[0, 1]))
        return "centre_line", (Segment(pts) if len(pts) == 2 else Polyline(pts))
    if len(pts) < 3:
        x0, y0, w, h = obj.bbox
        return "feature", Rect(float(x0), float(y0), float(w), float(h))
    return "feature", Polygon(pts)


def propose(detector: Detector, image: Image, *, image_id: Optional[str] = None,
            display: Optional[DisplaySettings] = None, input_mode: str = "display", run_id: str = "",
            name: str = "adapt pipeline", default_label: Optional[str] = None) -> tuple[list[Annotation], Detection]:
    """Run ``detector`` on ``image`` and turn every object into a proposed annotation.  Objects whose
    labelling rule says "skip" are left out; with plain single-label labelling and no label set,
    objects get ``default_label``."""
    gray, valid = detector_input(image, display, input_mode)
    dx, dy = image.pixel_size
    px_nm = math.sqrt(dx * dy) * 1e9 if image.calibrated and image.x.unit == "m" else None
    det = detector.detect(gray, px_nm, path=image.source.path or None, channel=image.source.channel or None,
                          valid=valid)
    config = detector.config.to_dict()
    origin = Origin("detector", name, software_version(), run_id, None,
                    {"config": params_digest(config), "input": input_mode})
    feats = det.features
    out = []
    lab_cfg = detector.config.labeling
    unlabeled_single = lab_cfg.mode == "single" and lab_cfg.single_label is None
    pairs = [(o, default_label) for o in det.objects] if unlabeled_single else det.labeled_objects()
    for obj, label in pairs:
        kind, geom = object_geometry(obj)
        attrs: dict = {"object": int(obj.id), "area_px": int(obj.area)}
        if feats is not None and obj.id in feats.index:
            attrs["features"] = {k: (None if not np.isfinite(v) else float(v))
                                 for k, v in feats.loc[obj.id].items() if isinstance(v, (int, float, np.number))}
        if px_nm is not None and not image.isotropic:
            attrs["note"] = "shape features use the geometric-mean pixel size (pixels are not square)"
        out.append(Annotation(image_id or image.id, kind, geom, label=label or None, status="proposed",  # type: ignore[arg-type]
                              origin=origin, attrs=attrs))
    return out, det


def detection_summary(det: Detection) -> dict:
    s = det.summary()
    s.pop("file", None)
    return s
