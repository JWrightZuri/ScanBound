"""
The picture a model sees.  Export and prediction both call ``render_for_model`` with the same
``RenderSpec``, which is stored in the dataset manifest and next to trained weights, so a model is
always applied to images made the way its training images were.

Modes
-----
display_rgb   the image as displayed (levels, gamma, colormap) - what the old app exported as JPG
display_gray  levels and gamma, gray (colormap-independent; better across differently styled images)
data_gray     the data min-max normalised, ignoring display settings

Pixels without data become black (0).  Geometry stays in raw pixel coordinates: the rendered
picture has exactly the shape of the data, also for non-square pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..display import DisplaySettings, ImageStats as DisplayStats, render
from ..model import Image
from ..proposals import detector_input

RENDER_MODES = ("display_rgb", "display_gray", "data_gray")


@dataclass(frozen=True)
class RenderSpec:
    mode: str = "display_rgb"
    image_format: str = "png"          # png is lossless; "jpg" reproduces the old exports

    def __post_init__(self) -> None:
        if self.mode not in RENDER_MODES:
            raise ValueError(f"unknown render mode {self.mode!r}; expected one of {RENDER_MODES}")
        if self.image_format not in ("png", "jpg"):
            raise ValueError("image_format must be 'png' or 'jpg'")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RenderSpec":
        d = d or {}
        return cls(d.get("mode", "display_rgb"), d.get("image_format", "png"))


def render_for_model(image: Image, display: Optional[DisplaySettings] = None,
                     spec: RenderSpec = RenderSpec()) -> np.ndarray:
    """(H, W, 3) uint8 RGB."""
    valid = image.valid
    if spec.mode == "display_rgb":
        s = display or DisplaySettings()
        out = render(image.data, DisplayStats(image.data), s)
        if out.ndim == 2:
            out = np.repeat(out[..., None], 3, axis=2)
    else:
        t, _ = detector_input(image, display, "display" if spec.mode == "display_gray" else "data")
        g = np.clip(t * 255 + 0.5, 0, 255).astype(np.uint8)
        out = np.repeat(g[..., None], 3, axis=2)
    if not valid.all():
        out = out.copy()
        out[~valid] = 0
    return np.ascontiguousarray(out)
