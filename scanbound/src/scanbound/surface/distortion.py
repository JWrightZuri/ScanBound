"""
Local lattice distortion: how the observed sites depart from a fitted reference lattice.

    d = local_distortion(fit)                    # one row per indexed site
    d.displacement                               # observed - ideal (physical, x right / y up)
    d.strain                                     # (N, 2, 2) small-strain tensor ε = ½(F + Fᵀ) - I
    d.rotation                                   # local rotation, degrees (counter-clockwise)
    d.dilatation                                 # relative change of the local cell area
    d.confidence                                 # 0..1: neighbours used and how well an affine map fits them
    grid = d.map("dilatation", fit.shape)        # interpolated image (NaN outside the sites' hull)

Method
------
For every indexed site the neighbours within ``radius`` lattice constants are collected.  Their
ideal positions relative to the site (from the indices and the reference lattice) and observed
positions relative to the site are related by a 2 x 2 deformation gradient F, found by least
squares.  ε = ½(F + Fᵀ) - I, the rotation is ½(F₂₁ - F₁₂) (small angles) and the dilatation
det(F) - 1.  All tensors use x to the right and y *up* (the screen convention of the angles).

The reference is the image's own mean lattice unless another one is given, so a uniform distortion
of the whole image (drift, calibration, or a real homogeneous strain) does not show up here; the
mean fit itself compared with a known lattice is what reveals it (see ``calibrate_from_lattice``).
What remains is real local relaxation *and* any non-uniform scanner distortion, creep, or
site-detection error: it is an observation, not yet a strain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .lattice import LATTICE, LatticeFit

_FLIP = np.array([1.0, -1.0])            # y down -> y up


@dataclass
class DistortionMap:
    fit: LatticeFit
    site: np.ndarray                     # indices (into fit.sites) of the sites described
    xy: np.ndarray                       # (N, 2) pixel positions
    displacement: np.ndarray             # (N, 2) physical, y up
    F: np.ndarray                        # (N, 2, 2) deformation gradient (y up)
    n_neighbors: np.ndarray              # (N,)
    misfit: np.ndarray                   # (N,) rms residual of the local affine fit / lattice constant
    radius: float

    @property
    def strain(self) -> np.ndarray:
        I = np.eye(2)
        return 0.5 * (self.F + np.transpose(self.F, (0, 2, 1))) - I

    @property
    def rotation(self) -> np.ndarray:
        return np.degrees(0.5 * (self.F[:, 1, 0] - self.F[:, 0, 1]))

    @property
    def dilatation(self) -> np.ndarray:
        F = self.F
        return F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0] - 1.0     # NaN where F is unknown, no warning

    @property
    def shear(self) -> np.ndarray:
        """Maximum shear strain √(((εxx - εyy)/2)² + εxy²)."""
        e = self.strain
        return np.sqrt(((e[:, 0, 0] - e[:, 1, 1]) / 2) ** 2 + e[:, 0, 1] ** 2)

    @property
    def confidence(self) -> np.ndarray:
        """1 with a full shell of neighbours and a perfect affine fit, dropping with missing neighbours
        and with local misfit (0.1 lattice constants rms -> 0.5)."""
        full = np.clip(self.n_neighbors / max(1.0, np.nanmax(self.n_neighbors) if len(self.n_neighbors) else 1.0), 0, 1)
        return full * (1.0 / (1.0 + (self.misfit / 0.1) ** 2))

    def values(self, quantity: str) -> np.ndarray:
        e = self.strain
        table = {
            "displacement": np.hypot(self.displacement[:, 0], self.displacement[:, 1]),
            "displacement_x": self.displacement[:, 0], "displacement_y": self.displacement[:, 1],
            "exx": e[:, 0, 0], "eyy": e[:, 1, 1], "exy": e[:, 0, 1],
            "dilatation": self.dilatation, "rotation": self.rotation, "shear": self.shear,
            "confidence": self.confidence, "misfit": self.misfit,
        }
        if quantity not in table:
            raise KeyError(f"unknown quantity {quantity!r}; known: {', '.join(table)}")
        return table[quantity]

    QUANTITIES = ("displacement", "displacement_x", "displacement_y", "exx", "eyy", "exy", "dilatation", "rotation",
                  "shear", "confidence", "misfit")

    def map(self, quantity: str, shape: Optional[tuple[int, int]] = None, step: int = 1,
            min_confidence: float = 0.0) -> np.ndarray:
        """``quantity`` interpolated onto the pixel grid (linear, NaN outside the sites' convex hull)."""
        from scipy.interpolate import griddata
        rows, cols = shape or self.fit.shape
        v = self.values(quantity)
        ok = np.isfinite(v) & (self.confidence >= min_confidence)
        out = np.full((rows, cols), np.nan)
        if ok.sum() < 3:
            return out
        gy, gx = np.mgrid[0:rows:step, 0:cols:step]
        g = griddata(self.xy[ok], v[ok], (gx + 0.5, gy + 0.5), method="linear")
        if step == 1:
            return g
        from scipy import ndimage
        return ndimage.zoom(np.nan_to_num(g, nan=np.nan), step, order=0)[:rows, :cols]

    def summary(self) -> dict[str, float]:
        good = self.confidence >= 0.5
        e = self.strain[good]
        return {"sites": int(len(self.site)), "reliable": int(good.sum()),
                "displacement_rms": float(np.sqrt(np.mean(np.sum(self.displacement[good] ** 2, axis=1)))) if good.any() else math.nan,
                "dilatation_std": float(np.std(self.dilatation[good])) if good.any() else math.nan,
                "rotation_std_deg": float(np.std(self.rotation[good])) if good.any() else math.nan,
                "exx_std": float(np.std(e[:, 0, 0])) if good.any() else math.nan,
                "eyy_std": float(np.std(e[:, 1, 1])) if good.any() else math.nan}

    def rows(self) -> list[dict]:
        """One dict per site (for tables / CSV)."""
        e = self.strain
        conf = self.confidence
        out = []
        for i, s in enumerate(self.site):
            out.append({"site": int(s), "x_px": float(self.xy[i, 0]), "y_px": float(self.xy[i, 1]),
                        "m": int(self.fit.indices[s, 0]), "n": int(self.fit.indices[s, 1]),
                        "ux": float(self.displacement[i, 0]), "uy": float(self.displacement[i, 1]),
                        "exx": float(e[i, 0, 0]), "eyy": float(e[i, 1, 1]), "exy": float(e[i, 0, 1]),
                        "rotation_deg": float(self.rotation[i]), "dilatation": float(self.dilatation[i]),
                        "neighbors": int(self.n_neighbors[i]), "misfit": float(self.misfit[i]),
                        "confidence": float(conf[i])})
        return out


def local_distortion(fit: LatticeFit, radius: float = 1.6, min_neighbors: int = 3,
                     reference: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> DistortionMap:
    """
    Displacement, deformation gradient, strain and rotation at every indexed site.

    ``radius``: neighbourhood in units of the longer lattice vector (1.6 = first shell of a
    hexagonal or square lattice plus the square's diagonals).  ``reference``: (origin, a1, a2) of
    another lattice (physical, y down) to measure against instead of this image's mean fit.
    """
    on = np.nonzero(fit.status == LATTICE)[0]
    origin, a1, a2 = (fit.origin, fit.a1, fit.a2) if reference is None else reference
    A = np.column_stack([a1, a2])
    P = fit.sites.xy[on] * np.asarray(fit.scale)
    idx = fit.indices[on].astype(float)
    ideal = origin + idx @ A.T
    disp = (P - ideal) * _FLIP
    amax = max(np.linalg.norm(a1), np.linalg.norm(a2))
    amin = min(np.linalg.norm(a1), np.linalg.norm(a2))
    # neighbours by index distance (robust against large displacements)
    tree = cKDTree(ideal)
    F = np.tile(np.eye(2), (len(on), 1, 1))
    nn = np.zeros(len(on), int)
    misfit = np.full(len(on), np.nan)
    for i in range(len(on)):
        js = [j for j in tree.query_ball_point(ideal[i], radius * amax + 1e-12) if j != i]
        nn[i] = len(js)
        if len(js) < min_neighbors:
            continue
        X = (ideal[js] - ideal[i]) * _FLIP
        Y = (P[js] - P[i]) * _FLIP
        if np.linalg.matrix_rank(X, tol=1e-6 * amin) < 2:
            continue
        Ft, *_ = np.linalg.lstsq(X, Y, rcond=None)       # Y ≈ X Fᵀ
        F[i] = Ft.T
        r = Y - X @ Ft
        misfit[i] = float(np.sqrt((r ** 2).sum(axis=1).mean())) / amin
    bad = ~np.isfinite(misfit)
    F[bad] = np.nan
    return DistortionMap(fit, on, fit.sites.xy[on], disp, F, nn, np.where(bad, np.inf, misfit), radius)
