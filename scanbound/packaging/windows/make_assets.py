"""
Make the Windows icon and installer wizard images from the ScanBound logo.

    python make_assets.py <logo.png> <output folder>

Writes scanbound.ico (16-256 px), wizard.bmp / wizard@2x.bmp (the installer's side banner) and
wizard_small.bmp / wizard_small@2x.bmp (its top-right corner).
"""

from __future__ import annotations

import os
import sys

from PIL import Image

ICON_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
BANNER = (164, 314)          # Inno Setup's wizard image at 100 % scaling
SMALL = (55, 55)
BACKGROUND = (24, 24, 32)    # the logo's dark tone, so the banner reads as one piece


def _square(logo: Image.Image) -> Image.Image:
    side = max(logo.size)
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.paste(logo, ((side - logo.width) // 2, (side - logo.height) // 2), logo)
    return out


def _on_background(logo: Image.Image, size: tuple[int, int], fraction: float, bg) -> Image.Image:
    w, h = size
    side = int(min(w, h) * fraction)
    mark = _square(logo).resize((side, side), Image.LANCZOS)
    out = Image.new("RGB", size, bg)
    out.paste(mark, ((w - side) // 2, (h - side) // 2), mark)
    return out


def main(logo_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    logo = Image.open(logo_path).convert("RGBA")
    square = _square(logo)
    square.resize((256, 256), Image.LANCZOS).save(os.path.join(out_dir, "scanbound.ico"),
                                                  sizes=[(s, s) for s in ICON_SIZES])
    for scale, suffix in ((1, ""), (2, "@2x")):
        banner = (BANNER[0] * scale, BANNER[1] * scale)
        _on_background(logo, banner, 0.85, BACKGROUND).save(os.path.join(out_dir, f"wizard{suffix}.bmp"))
        small = (SMALL[0] * scale, SMALL[1] * scale)
        _on_background(logo, small, 0.95, (255, 255, 255)).save(os.path.join(out_dir, f"wizard_small{suffix}.bmp"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
