"""
Small crops of annotated features, sorted into folders per label (and orientation bin) - for
quick visual checks, classifier training or figures.

    n = export_clips(project, "clips", context=1.6, per_group=20, angle_bin=10)
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

from ..annotations import GROUND_TRUTH
from ..measurements.report import reference_axis
from .render import RenderSpec, render_for_model

if TYPE_CHECKING:  # pragma: no cover
    from ..project import Project


def _safe(text: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text)).strip("_") or "unknown"


def export_clips(project: "Project", out_dir: str, *, image_ids: Optional[Iterable[str]] = None, context: float = 1.6,
                 per_group: int = 20, angle_bin: Optional[float] = None, seed: int = 0,
                 render: RenderSpec = RenderSpec()) -> int:
    """Crops of ground-truth features (box enlarged by ``context``), at most ``per_group`` per label
    (or per label and orientation bin relative to the image's reference axis).  Returns the count."""
    from PIL import Image as PILImage

    ids = list(image_ids) if image_ids is not None else project.annotations.image_ids()
    groups: dict[tuple, list] = defaultdict(list)
    for i in ids:
        anns = project.annotations.for_image(i)
        ref = reference_axis(anns)
        ref_angle = None
        if ref is not None:
            (x0, y0), (x1, y1) = ref.geometry.points[0], ref.geometry.points[-1]
            ref_angle = float(np.degrees(np.arctan2(-(y1 - y0), x1 - x0)))
        for a in anns:
            if a.kind != "feature" or a.status not in GROUND_TRUTH:
                continue
            key: tuple = (a.label or "unlabeled",)
            if angle_bin and a.orientation is not None:
                rel = (a.orientation - (ref_angle or 0.0)) % 360.0
                b = int(rel // angle_bin) * angle_bin
                key += (f"{b:03.0f}-{b + angle_bin:03.0f}deg",)
            groups[key].append(a)
    rng = random.Random(seed)
    n = 0
    rendered: dict[str, np.ndarray] = {}
    for key, anns in groups.items():
        rng.shuffle(anns)
        folder = os.path.join(out_dir, *(_safe(k) for k in key))
        os.makedirs(folder, exist_ok=True)
        for a in anns[:per_group]:
            if a.image_id not in rendered:
                rendered[a.image_id] = render_for_model(project.processed_image(a.image_id),
                                                        project.entry(a.image_id).display, render)
            rgb = rendered[a.image_id]
            h, w = rgb.shape[:2]
            x0, y0, x1, y1 = a.geometry.bbox()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half = max(x1 - x0, y1 - y0, 4.0) * context / 2
            c0, r0 = int(max(0, np.floor(cx - half))), int(max(0, np.floor(cy - half)))
            c1, r1 = int(min(w, np.ceil(cx + half))), int(min(h, np.ceil(cy + half)))
            if c1 - c0 < 2 or r1 - r0 < 2:
                continue
            stem = _safe(os.path.splitext(os.path.basename(a.image_id.rpartition("::")[0]))[0])
            PILImage.fromarray(np.ascontiguousarray(rgb[r0:r1, c0:c1])).save(os.path.join(folder, f"{stem}_{a.id}.png"))
            n += 1
    return n
