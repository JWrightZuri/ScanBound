"""
Overlay rendering without Qt: each detected object in its own colour (or its label's colour)
with the generated ROI drawn on top.  Returns an RGB uint8 array that can be saved with
OpenCV / Pillow or shown in any viewer.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ._image import to_u8


def object_palette(n: int) -> np.ndarray:
    """(n, 3) uint8 RGB colours, row 0 = background black, neighbours well separated."""
    hue = ((np.arange(n) * 69) % 180).astype(np.uint8)         # ~137.5 deg golden-angle steps
    sat = np.where(np.arange(n) % 2 == 0, 210, 150).astype(np.uint8)
    hsv = np.stack([hue, sat, np.full(n, 255, np.uint8)], axis=-1).reshape(n, 1, 3)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(n, 3)
    rgb[0] = 0
    return rgb


def _hex_rgb(color: str) -> tuple[int, int, int]:
    """'#rgb' / '#rrggbb' -> (r, g, b)."""
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"not a hex colour: {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def render_objects(source: np.ndarray, mask: np.ndarray,
                   objects: Optional[list], object_colors: Optional[dict] = None,
                   components=None, base: Optional[np.ndarray] = None) -> np.ndarray:
    """
    RGB overlay: dimmed original, each connected object filled with its own colour,
    and (if given) the generated ROI outline, vertices and object IDs on top.
    IDs match SegObject.id (both use 8-connected labelling of the same mask).
    ``object_colors`` {id: '#rrggbb' or None} colours objects by their assigned label;
    None marks a skipped object (dark gray, no ROI outline).
    ``components`` / ``base`` (dimmed RGB original) can be passed in to avoid recomputing them.
    """
    h, w = mask.shape
    if components is not None:
        n, labels = components[0], components[1]
    else:
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if base is None:
        base = np.repeat((to_u8(source) * 0.35).astype(np.uint8)[..., None], 3, axis=2)
    rgb = base.copy()
    fg = labels > 0
    palette = object_palette(n)
    skipped = set()
    if object_colors:
        for oid, color in object_colors.items():
            if 0 < oid < n:
                if color is None:
                    palette[oid] = (70, 70, 70)
                    skipped.add(oid)
                else:
                    palette[oid] = _hex_rgb(color)
    rgb[fg] = palette[labels[fg]]

    if objects:
        t = max(1, int(round(max(h, w) / 600)))                 # keep lines visible when scaled down
        S = 4                                                   # sub-pixel bits for drawing
        for o in objects:
            if o.id in skipped:
                continue
            # scene units (pixel covers [c, c+1]) -> pixel-centre drawing coordinates
            pts = np.round((o.points - 0.5) * (1 << S)).astype(np.int32).reshape(-1, 1, 2)
            if o.roi_type == "PointROI":
                cx, cy = (int(v) for v in pts[0, 0])
                r = (3 * t + 2) << S
                cv2.circle(rgb, (cx, cy), r, (255, 255, 255), t, cv2.LINE_AA, shift=S)
                cv2.circle(rgb, (cx, cy), (t + 1) << S, (255, 220, 0), -1, cv2.LINE_AA, shift=S)
                continue
            closed = o.roi_type != "LineSegmentROI"
            cv2.polylines(rgb, [pts], closed, (255, 255, 255), t, cv2.LINE_AA, shift=S)
            if o.roi_type not in ("RectROI", "MaskROI"):
                for (px, py) in pts[:, 0]:
                    cv2.circle(rgb, (int(px), int(py)), (t + 1) << S, (255, 220, 0), -1,
                               cv2.LINE_AA, shift=S)
        if len(objects) <= 80:                                  # IDs only while they stay readable
            scale = max(0.4, max(h, w) / 1100)
            for o in objects:
                x, y, _, _ = o.bbox
                org = (int(x), int(max(y - 3, 12 * scale + 2)))
                txt = str(o.id)
                cv2.putText(rgb, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                            t + 2, cv2.LINE_AA)
                cv2.putText(rgb, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                            t, cv2.LINE_AA)
    return rgb


def render_ribbons(source: np.ndarray, mask: np.ndarray, ribbons: list) -> np.ndarray:
    """RGB overlay of ribbon centre lines (one colour per segment) on the dimmed image."""
    base = np.repeat((to_u8(source) * 0.35).astype(np.uint8)[..., None], 3, axis=2)
    base[np.asarray(mask, bool)] //= 2
    base[np.asarray(mask, bool)] += 60
    palette = object_palette(len(ribbons) + 1)
    t = max(1, int(round(max(mask.shape) / 600)))
    for r, color in zip(ribbons, palette[1:]):
        pts = np.round((r.points - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(base, [pts], False, tuple(int(c) for c in color), t, cv2.LINE_AA, shift=4)
    return base
