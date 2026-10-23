"""The practice image of the getting-started tour: a real STM scan of 9AP molecules on PdGa(111), shipped in
``scanbound/tutorial`` and copied into its own folder (~/.scanbound/tutorial) so the tour's project file is
written there.  Without that file the tour falls back to a made-up image (bright islands and a few short rods
on a tilted, noisy background, written as a 16-bit PNG)."""

from __future__ import annotations

import os

import cv2
import numpy as np

FOLDER = os.path.join(os.path.expanduser("~"), ".scanbound", "tutorial")
FILE = "practice_islands.png"
EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "tutorial", "example_9AP_pdga111.sxm")


def practice_image(size: int = 512, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = 0.55 * xx / size + 0.35 * yy / size                       # tilt: what Plane level removes
    layer = np.zeros((size, size), np.float32)
    centres = []
    while len(centres) < 22:                                         # round islands, not touching
        c = rng.uniform(40, size - 40, 2)
        if all(np.hypot(*(c - d)) > 60 for d in centres):
            centres.append(c)
    for cx, cy in centres[:16]:
        r = rng.uniform(9, 17)
        cv2.circle(layer, (int(cx), int(cy)), int(r), 1.0, -1, lineType=cv2.LINE_AA)
    for cx, cy in centres[16:]:                                      # a few rods
        a = rng.uniform(0, np.pi)
        d = rng.uniform(18, 30) * np.array([np.cos(a), np.sin(a)])
        cv2.line(layer, (int(cx - d[0]), int(cy - d[1])), (int(cx + d[0]), int(cy + d[1])), 1.0, 7, cv2.LINE_AA)
    img += 0.45 * cv2.GaussianBlur(layer, (0, 0), 1.5)
    img += rng.normal(0, 0.025, img.shape).astype(np.float32)
    img -= img.min()
    return (img / img.max() * 65535).astype(np.uint16)


def practice_folder(fresh: bool = True) -> str:
    """Write the practice image (once) and return its folder; ``fresh`` drops the project saved there by an
    earlier tour, so the tour starts on a clean image."""
    os.makedirs(FOLDER, exist_ok=True)
    made = os.path.join(FOLDER, FILE)
    if os.path.isfile(EXAMPLE):
        import shutil
        path = os.path.join(FOLDER, os.path.basename(EXAMPLE))
        if not os.path.isfile(path) or os.path.getsize(path) != os.path.getsize(EXAMPLE):
            shutil.copyfile(EXAMPLE, path)
        if os.path.isfile(made):                  # the made-up image of earlier versions: one image only
            os.remove(made)
    elif not os.path.isfile(made):
        cv2.imwrite(made, practice_image())
    if fresh:
        from ...project import DEFAULT_NAME
        old = os.path.join(FOLDER, DEFAULT_NAME)
        if os.path.isfile(old):
            os.remove(old)
    return FOLDER
