"""
Export images as pictures: displayed colours, true aspect, scale bar, optional colorbar.

    path = export_image(image, display, "out/scan_Z.png", colorbar=True)

Besides the picture it writes

* a row in ``metadata.csv`` in the same folder (the old app's format: pixel scale in nm, so older
  tools and the legacy annotation importer keep working);
* ``<picture>.json`` with the image header, processing history and display settings - enough to
  regenerate the picture from the raw file.

Non-square pixels are resampled to square ones for the picture (it then looks like the surface);
the metadata records the picture's own pixel size.  Pixels without data are drawn in ``nan_color``.
No Qt dependency.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Optional

import numpy as np

from .display import DisplaySettings, ImageStats as DisplayStats, colorbar_image, render_rgba
from .model import Image
from .model.units import format_value

METADATA_FIELDS = ["Filename", "Channel", "Modifications", "Dimensions", "Pixel Scale X (nm)", "Pixel Scale Y (nm)"]


def _nice(v: float) -> float:
    if v <= 0 or not math.isfinite(v):
        return 1.0
    e = math.floor(math.log10(v))
    for m in (5, 2, 1):
        if m * 10 ** e <= v:
            return m * 10 ** e
    return 10 ** e


def figure_array(image: Image, display: Optional[DisplaySettings] = None, *, scale_bar: bool = True,
                 colorbar: bool = False, nan_color: tuple = (0, 0, 0), min_size: int = 512) -> tuple[np.ndarray, tuple[float, float]]:
    """(RGB uint8 picture, its pixel size (x, y) in the image's length unit or (nan, nan))."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    s = display or DisplaySettings()
    stats = DisplayStats(image.data)
    rgba = render_rgba(image.data, stats, s, invalid=(*nan_color, 255))
    pic = PILImage.fromarray(rgba[..., :3].copy())
    h, w = image.shape
    dx, dy = image.pixel_size
    calibrated = image.calibrated and dx > 0 and dy > 0
    out_w, out_h = w, h
    if calibrated and not image.isotropic:                  # square pixels of the smaller size
        p = min(dx, dy)
        out_w, out_h = max(1, round(w * dx / p)), max(1, round(h * dy / p))
    k = max(1, math.ceil(min_size / max(1, min(out_w, out_h)))) if (scale_bar or colorbar) else 1
    out_w, out_h = out_w * k, out_h * k
    if (out_w, out_h) != (w, h):
        pic = pic.resize((out_w, out_h), PILImage.Resampling.NEAREST)
    px = (dx * w / out_w, dy * h / out_h) if calibrated else (math.nan, math.nan)

    if scale_bar and calibrated:
        draw = ImageDraw.Draw(pic, "RGBA")
        length = _nice(0.2 * out_w * px[0])
        bar = length / px[0]
        font_px = max(10, int(out_h * 0.035))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_px)
        except OSError:
            font = ImageFont.load_default()
        text = format_value(length, image.x.unit, 3)
        margin = int(out_w * 0.03)
        x1, y1 = out_w - margin, out_h - margin
        x0 = x1 - bar
        th = font_px + 4
        draw.rectangle([x0 - 8, y1 - th - 14, x1 + 8, y1 + 6], fill=(0, 0, 0, 130))
        draw.rectangle([x0, y1 - 4, x1, y1], fill=(255, 255, 255, 255))
        tw = draw.textlength(text, font=font)
        draw.text(((x0 + x1) / 2 - tw / 2, y1 - th - 10), text, fill=(255, 255, 255, 255), font=font)

    arr = np.asarray(pic)
    if colorbar:
        bar_rgba = colorbar_image(stats, s, out_h, label=image.quantity, unit=image.unit)
        cb = PILImage.fromarray(bar_rgba).convert("RGB")
        if cb.height != out_h:
            cb = cb.resize((max(1, round(cb.width * out_h / cb.height)), out_h), PILImage.Resampling.LANCZOS)
        canvas = PILImage.new("RGB", (out_w + cb.width, out_h), "white")
        canvas.paste(PILImage.fromarray(arr), (0, 0))
        canvas.paste(cb, (out_w, 0))
        arr = np.asarray(canvas)
    return np.ascontiguousarray(arr), px


def export_image(image: Image, display: Optional[DisplaySettings], path: str, *, scale_bar: bool = True,
                 colorbar: bool = False, metadata: bool = True, quality: int = 95) -> str:
    from PIL import Image as PILImage

    arr, px = figure_array(image, display, scale_bar=scale_bar, colorbar=colorbar)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pil = PILImage.fromarray(arr)
    if path.lower().endswith((".jpg", ".jpeg")):
        pil.save(path, quality=quality)
    else:
        pil.save(path)
    if metadata:
        to_nm = 1e9 if image.calibrated and image.x.unit == "m" else None
        name = os.path.basename(path)
        steps = "; ".join(r.op.split(".", 1)[-1] for r in image.history if r.effect != "none")
        img_w = arr.shape[1] if not colorbar else int(round(image.extent[0] / px[0])) if math.isfinite(px[0]) else arr.shape[1]
        row = {"Filename": name, "Channel": image.source.channel or image.quantity, "Modifications": steps,
               "Dimensions": f"{img_w} x {arr.shape[0]}",
               "Pixel Scale X (nm)": "" if to_nm is None else px[0] * to_nm,
               "Pixel Scale Y (nm)": "" if to_nm is None else px[1] * to_nm}
        write_metadata_row(os.path.join(os.path.dirname(os.path.abspath(path)), "metadata.csv"), row)
        side = {"format": "scanbound_figure", "version": 1, "picture": name, "picture_pixel_size": list(px),
                "image": image.header(), "display": (display or DisplaySettings()).to_dict(),
                "history": [r.to_dict() for r in image.history], "scale_bar": scale_bar, "colorbar": colorbar}
        with open(path + ".json", "w", encoding="utf-8") as fh:
            json.dump(side, fh, indent=1, default=str)
    return path


def write_metadata_row(metadata_file: str, row: dict) -> None:
    """Insert or replace the row for row['Filename']."""
    fields = list(METADATA_FIELDS)
    rows = []
    if os.path.exists(metadata_file):
        with open(metadata_file, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fields += [c for c in (reader.fieldnames or []) if c not in fields]
            rows = [r for r in reader if r.get("Filename") != row["Filename"]]
    rows.append(row)
    with open(metadata_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
