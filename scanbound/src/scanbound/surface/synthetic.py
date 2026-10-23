"""
Synthetic surfaces with a known answer, for tests, demonstrations and checking settings.

    img, truth = lattice_image(a=0.28e-9, gamma=120, angle=12, size=(256, 256), pixel=0.03e-9,
                               vacancies=0.02, adatoms=5, noise=0.05, seed=1)
    truth["sites"]        # pixel positions of every atom drawn
    truth["vacancies"]    # pixel positions of the lattice sites left empty
    truth["adatoms"]      # pixel positions of the extra atoms
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..model import Axis, Image


def _basis(a: float, b: float, gamma: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Physical vectors (y down) of a lattice with a1 at ``angle`` (screen, CCW)."""
    t, g = math.radians(angle), math.radians(gamma)
    a1 = np.array([math.cos(t), -math.sin(t)]) * a
    a2 = np.array([math.cos(t + g), -math.sin(t + g)]) * b
    return a1, a2


def _blobs(shape: tuple[int, int], pts_px: np.ndarray, sigma_px: tuple[float, float], amp: np.ndarray) -> np.ndarray:
    rows, cols = shape
    out = np.zeros(shape)
    rx, ry = int(4 * sigma_px[0]) + 1, int(4 * sigma_px[1]) + 1
    for (x, y), A in zip(pts_px, amp):
        x0, x1 = max(0, int(x) - rx), min(cols, int(x) + rx + 1)
        y0, y1 = max(0, int(y) - ry), min(rows, int(y) + ry + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xx = np.arange(x0, x1) + 0.5
        yy = np.arange(y0, y1) + 0.5
        g = np.exp(-0.5 * (((xx[None, :] - x) / sigma_px[0]) ** 2 + ((yy[:, None] - y) / sigma_px[1]) ** 2))
        out[y0:y1, x0:x1] += A * g
    return out


def lattice_image(a: float = 0.28e-9, b: Optional[float] = None, gamma: float = 120.0, angle: float = 10.0,
                  size: tuple[int, int] = (200, 200), pixel: float = 0.03e-9, pixel_y: Optional[float] = None,
                  vacancies: float = 0.0, adatoms: int = 0, noise: float = 0.05, seed: int = 0,
                  displacement: Optional[Any] = None, molecules: Optional[list[dict]] = None,
                  origin: Optional[tuple[float, float]] = None, width: float = 0.25) -> tuple[Image, dict]:
    """
    An image of Gaussian atoms on a lattice (rows x cols = ``size``) with known defects.

    ``displacement``: function (x, y) -> (dx, dy) in metres applied to each atom (strain fields).
    ``molecules``: list of {"x", "y" (pixels), "angle" (deg), "length" (m), "height"}: elongated
    blobs drawn on top (positions are kept in the truth).
    """
    rng = np.random.default_rng(seed)
    b = b or a
    pixel_y = pixel_y or pixel
    rows, cols = size
    a1, a2 = _basis(a, b, gamma, angle)
    A = np.column_stack([a1, a2])
    o = np.array(origin if origin is not None else (cols * pixel / 2 + 0.13 * a, rows * pixel_y / 2 + 0.07 * a))
    corners = np.array([[0, 0], [cols * pixel, 0], [0, rows * pixel_y], [cols * pixel, rows * pixel_y]]) - o
    uv = np.linalg.solve(A, corners.T).T
    lo, hi = np.floor(uv.min(0)) - 1, np.ceil(uv.max(0)) + 1
    m, n = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    idx = np.column_stack([m.ravel(), n.ravel()])
    P = o + idx @ A.T
    if displacement is not None:
        P = P + np.array([displacement(x, y) for x, y in P])
    px = P / [pixel, pixel_y]
    inside = (px[:, 0] > -2) & (px[:, 0] < cols + 2) & (px[:, 1] > -2) & (px[:, 1] < rows + 2)
    px, idx = px[inside], idx[inside]
    interior = (px[:, 0] > 3) & (px[:, 0] < cols - 3) & (px[:, 1] > 3) & (px[:, 1] < rows - 3)
    vac = np.zeros(len(px), bool)
    if vacancies > 0:
        cand = np.nonzero(interior)[0]
        k = int(round(vacancies * len(cand)))
        vac[rng.choice(cand, size=k, replace=False)] = True
    atoms = px[~vac]
    sig = (width * a / pixel / 2.0, width * a / pixel_y / 2.0)
    z = _blobs(size, atoms, sig, np.ones(len(atoms)))
    ad = np.zeros((0, 2))
    if adatoms:
        # in hollow-like positions (cell centres), away from the border
        cells = idx[interior & ~vac]
        pick = cells[rng.choice(len(cells), size=min(adatoms, len(cells)), replace=False)]
        hollow = [2 / 3, 1 / 3] if abs(gamma - 120) < 1 else [0.5, 0.5]
        Pad = o + (pick + hollow) @ A.T
        ad = Pad / [pixel, pixel_y]
        z += _blobs(size, ad, sig, np.full(len(ad), 1.6))
    mols = []
    for mol in molecules or []:
        L = mol.get("length", 1.0e-9) / pixel
        t = math.radians(mol.get("angle", 0.0))
        ux, uy = math.cos(t), -math.sin(t)
        pts = np.array([[mol["x"] + s * ux * L / 2, mol["y"] + s * uy * L / 2] for s in np.linspace(-1, 1, 7)])
        z += _blobs(size, pts, (sig[0] * 1.5, sig[1] * 1.5), np.full(len(pts), mol.get("height", 2.0)))
        mols.append(dict(mol))
    z += noise * rng.standard_normal(size)
    img = Image(z * 1e-10, x=Axis.spatial("x", cols, pixel), y=Axis.spatial("y", rows, pixel_y), quantity="Z",
                unit="m", id="synthetic")
    truth = {"a1": a1, "a2": a2, "origin": o, "sites": atoms, "indices": idx[~vac], "vacancies": px[vac],
             "vacancy_indices": idx[vac], "adatoms": ad, "molecules": mols, "pixel": (pixel, pixel_y)}
    return img, truth
