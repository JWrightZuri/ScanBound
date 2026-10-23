"""
Project -> input of the ROI analyses.

* Only ground truth (manual + accepted) is analysed; proposals waiting for review never are.
* The analyses assume square pixels.  For images with non-square pixels the geometry is mapped to
  a square grid of the geometric-mean pixel size first, so lengths, areas and angles come out in
  correct physical units (the background image is resampled the same way, on demand).
* An image's reference direction for orientations: its reference axis annotation, or else the a1
  vector of its first lattice.
* Herringbone axes are passed on as marks of their image (the alignment analysis uses them).
* Mask features keep their pixels (``AnalysisData.masks``), so the analyses see the exact shape.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np

from ..annotations import GROUND_TRUTH, Annotation, Mask
from ..annotations.legacy import to_frame
from ..measurements.report import reference_axis
from .roi import AnalysisContext, AnalysisData, ImageInfo

MARK_KINDS = ("herringbone_axis",)

if TYPE_CHECKING:  # pragma: no cover
    from ..project import Project


def _square(image_shape: tuple[int, int], dx: float, dy: float) -> tuple[float, float, int, int]:
    """(fx, fy, width, height) mapping raw pixels to square pixels of size sqrt(dx*dy)."""
    h, w = image_shape
    s = math.sqrt(dx * dy)
    fx, fy = dx / s, dy / s
    return fx, fy, max(1, int(round(w * fx))), max(1, int(round(h * fy)))


def analysis_data(project: "Project", image_ids: Optional[Iterable[str]] = None,
                  statuses: Sequence[str] = GROUND_TRUTH) -> AnalysisData:
    ids = list(image_ids) if image_ids is not None else project.image_ids()
    images: dict[str, ImageInfo] = {}
    anns: list[Annotation] = []
    for image_id in ids:
        img = project.raw_image(image_id)
        h, w = img.shape
        dx, dy = img.pixel_size
        physical = img.calibrated and img.x.unit == "m" and dx > 0 and dy > 0
        fx = fy = 1.0
        ww, hh = w, h
        if physical and not img.isotropic:
            fx, fy, ww, hh = _square((h, w), dx, dy)
        m = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1.0]])
        ref = None
        ref_ann = reference_axis(project.annotations.for_image(image_id, statuses=statuses))
        if ref_ann is not None:                     # reference axis, or a1 of the first lattice
            g = ref_ann.geometry if (fx == 1.0 and fy == 1.0) else ref_ann.geometry.transformed(m)
            (x0, y0), (x1, y1) = g.points[0], g.points[-1]
            ref = math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 360.0
        marks: dict[str, list] = {}
        for a in project.annotations.for_image(image_id, statuses=statuses):
            g = a.geometry if (fx == 1.0 and fy == 1.0) else a.geometry.transformed(m)
            a = a.evolve(geometry=g) if g is not a.geometry else a
            if a.kind in ("measurement_line", "lattice_vector"):     # rulers, not objects
                continue
            if a.kind == "reference_axis":
                continue
            if a.kind in MARK_KINDS:                                 # substrate directions, not objects
                (x0, y0), (x1, y1) = a.geometry.points[0], a.geometry.points[-1]
                marks.setdefault(a.kind, []).append(((x0, y0), (x1, y1)))
                continue
            anns.append(a)
        px_nm = math.sqrt(dx * dy) * 1e9 if physical else None
        images[image_id] = ImageInfo(image_id, ww, hh, px_nm, ref, _background(project, image_id, (fx, fy, ww, hh)),
                                     marks)
    frame = to_frame(anns, statuses=tuple(statuses))
    colors = {name: project.labels.color(name) for name in project.labels.names}
    masks = {(a.image_id, a.id): (a.geometry.x0, a.geometry.y0, a.geometry.bits)
             for a in anns if a.kind == "feature" and isinstance(a.geometry, Mask)}
    return AnalysisData(frame, images, colors, masks)


def _background(project: "Project", image_id: str, resample: tuple[float, float, int, int]):
    def load() -> np.ndarray:
        data = np.asarray(project.processed_image(image_id).filled("nearest"), float)
        fx, fy, ww, hh = resample
        if (fx, fy) != (1.0, 1.0):
            import cv2
            data = cv2.resize(data, (ww, hh), interpolation=cv2.INTER_LINEAR)
        return data
    return load


def analysis_context(project: "Project", image_ids: Optional[Iterable[str]] = None,
                     labels: Optional[Sequence[str]] = None, include_unlabeled: bool = False) -> AnalysisContext:
    return AnalysisContext(analysis_data(project, image_ids), labels=labels, include_unlabeled=include_unlabeled)
