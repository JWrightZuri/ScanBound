"""
Registering images of the same area (other channels, other biases, ncAFM vs STM, before / after,
repeated frames) and linking the objects seen in them.

    reg = register_translation(img_a, img_b)          # phase correlation, sub-pixel
    reg = register_rigid(img_a, img_b)                # + rotation and scale (log-polar Fourier)
    reg = register_landmarks(pts_a, pts_b, "affine")  # from clicked / matched points
    reg = register_lattice(fit_a, fit_b, coarse=reg)  # from two lattice fits (lattice-periodic ambiguity resolved by ``coarse``)
    reg.apply(xy_a)                                   # pixel in A -> pixel in B
    reg.error_px                                      # registration error estimate (1σ, pixels of B)
    links = link_objects(objs_a, objs_b, reg)          # matched, disappeared, appeared

A :class:`Registration` maps pixel coordinates of image A to image B with a 3 x 3 matrix and carries
an error estimate made the way that suits its method: the residual of landmarks (with
leave-one-out), the width of the correlation peak, or the lattice fits' uncertainties.  Channels of
one scan file share their pixel grid and need no registration (forward / backward channels can be
offset by piezo hysteresis: register them if it matters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .lattice import LatticeFit, _data, _filled, _plane_removed


@dataclass
class Registration:
    matrix: np.ndarray                   # 3 x 3, pixel A -> pixel B
    method: str
    error_px: float = math.nan           # 1σ estimate of the mapped-position error, pixels of B
    n_points: int = 0
    details: dict = field(default_factory=dict)

    @classmethod
    def identity(cls, method: str = "same pixel grid") -> "Registration":
        return cls(np.eye(3), method, 0.0)

    def apply(self, xy: np.ndarray) -> np.ndarray:
        p = np.asarray(xy, float).reshape(-1, 2)
        q = np.column_stack([p, np.ones(len(p))]) @ self.matrix.T
        return q[:, :2] / q[:, 2:3]

    def inverse(self) -> "Registration":
        return Registration(np.linalg.inv(self.matrix), self.method + " (inverse)", self.error_px, self.n_points,
                            dict(self.details))

    def then(self, other: "Registration") -> "Registration":
        """This mapping followed by ``other`` (A -> B -> C)."""
        e = math.hypot(self.error_px if np.isfinite(self.error_px) else 0.0,
                       other.error_px if np.isfinite(other.error_px) else 0.0)
        return Registration(other.matrix @ self.matrix, f"{self.method} + {other.method}", e)

    @property
    def translation(self) -> tuple[float, float]:
        return float(self.matrix[0, 2]), float(self.matrix[1, 2])

    @property
    def rotation(self) -> float:
        """Screen rotation, degrees (counter-clockwise)."""
        return math.degrees(math.atan2(-self.matrix[1, 0], self.matrix[0, 0]))

    @property
    def scale(self) -> float:
        return float(math.sqrt(abs(np.linalg.det(self.matrix[:2, :2]))))

    def describe(self) -> str:
        tx, ty = self.translation
        txt = f"{self.method}: shift ({tx:+.2f}, {ty:+.2f}) px"
        if abs(self.rotation) > 0.01 or abs(self.scale - 1) > 1e-4:
            txt += f", rotation {self.rotation:+.2f}°, scale {self.scale:.4f}"
        if np.isfinite(self.error_px):
            txt += f", error ≈ {self.error_px:.2f} px (1σ)"
        return txt

    def to_dict(self) -> dict[str, Any]:
        return {"matrix": self.matrix.tolist(), "method": self.method, "error_px": self.error_px,
                "n_points": self.n_points, "details": self.details}

    @classmethod
    def from_dict(cls, d: dict) -> "Registration":
        return cls(np.asarray(d["matrix"], float), str(d.get("method", "")), float(d.get("error_px", math.nan)),
                   int(d.get("n_points", 0)), dict(d.get("details") or {}))


def _prep(image: Any) -> np.ndarray:
    a = _plane_removed(_filled(_data(image)))
    s = a.std()
    return (a - a.mean()) / (s if s > 0 else 1.0)


def _same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, c = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return a[:r, :c], b[:r, :c]


def register_translation(a: Any, b: Any, upsample: int = 20) -> Registration:
    """Shift of B relative to A by phase correlation (same pixel size assumed)."""
    from skimage.registration import phase_cross_correlation
    A, B = _same_shape(_prep(a), _prep(b))
    shift, err, _ = phase_cross_correlation(A, B, upsample_factor=upsample, normalization=None)
    dy, dx = -shift                                       # skimage: shift that registers B onto A
    M = np.eye(3)
    M[0, 2], M[1, 2] = dx, dy
    # error: width of the correlation peak is ~ the feature size; use 1/upsample plus the
    # skimage error metric (0 = perfect) as a rough proxy
    e = max(1.0 / upsample, float(err) * 2.0)
    return Registration(M, "phase correlation (translation)", e, 0, {"phase_error": float(err)})


def register_rigid(a: Any, b: Any, upsample: int = 20) -> Registration:
    """Rotation, scale and shift of B relative to A (log-polar Fourier magnitude, then phase correlation)."""
    from skimage.registration import phase_cross_correlation
    from skimage.transform import AffineTransform, warp, warp_polar
    A, B = _same_shape(_prep(a), _prep(b))
    win = np.outer(np.hanning(A.shape[0]), np.hanning(A.shape[1]))
    FA = np.abs(np.fft.fftshift(np.fft.fft2(A * win)))
    FB = np.abs(np.fft.fftshift(np.fft.fft2(B * win)))
    radius = min(A.shape) // 2
    pa = warp_polar(np.log1p(FA), radius=radius, scaling="log")
    pb = warp_polar(np.log1p(FB), radius=radius, scaling="log")
    pa, pb = pa[:, : radius // 2], pb[:, : radius // 2]   # the low frequencies carry the structure
    sh, _, _ = phase_cross_correlation(pa, pb, upsample_factor=upsample, normalization=None)
    rot = sh[0] * 360.0 / pa.shape[0]
    klog = radius / math.log(radius)
    sc = math.exp(sh[1] / klog)
    # B is A rotated by -rot (screen) and scaled by 1/sc about the centre: undo on B, then find the shift
    c = np.array([A.shape[1] / 2, A.shape[0] / 2])
    best = None
    # the Fourier magnitude cannot tell 180° apart, and the polar-axis sign convention is checked
    # rather than assumed: try each reading and keep the one that correlates best
    for r in (rot, -rot, rot + 180.0, -rot + 180.0):
        for k in {sc, 1.0 / sc, 1.0}:
            t = math.radians(r)
            R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) * k
            M = np.eye(3)
            M[:2, :2] = R
            M[:2, 2] = c - R @ c                          # A -> B about the centre
            Bw = warp(B, AffineTransform(matrix=M), preserve_range=True)   # B sampled at M(A coords)
            s2, err, _ = phase_cross_correlation(A, Bw, upsample_factor=upsample, normalization=None)
            if best is None or err < best[1]:
                best = (M, err, s2, r, k)
    M, err, s2, r, sc = best  # type: ignore[misc]
    T = np.eye(3)
    T[0, 2], T[1, 2] = -s2[1], -s2[0]
    full = M @ T
    return Registration(full, "log-polar Fourier + phase correlation (rigid + scale)", max(1.0 / upsample, 2 * float(err)),
                        0, {"rotation": r, "scale": sc, "phase_error": float(err)})


def register_landmarks(src: np.ndarray, dst: np.ndarray, kind: str = "similarity") -> Registration:
    """
    From matched landmarks (pixels in A, pixels in B): ``"translation"``, ``"similarity"`` (rotation +
    uniform scale), ``"affine"`` (also shear / anisotropic scale - only when the images really
    differ that way, e.g. different drift).  The error estimate is the leave-one-out prediction
    error (falls back to the residual when there are too few points).
    """
    from skimage.transform import estimate_transform
    P = np.asarray(src, float).reshape(-1, 2)
    Q = np.asarray(dst, float).reshape(-1, 2)
    need = {"translation": 1, "similarity": 2, "affine": 3}[kind]
    if len(P) < need or len(P) != len(Q):
        raise ValueError(f"{kind} registration needs at least {need} matched point pairs")

    def fit(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        if kind == "translation":
            M = np.eye(3)
            M[:2, 2] = (q - p).mean(axis=0)
            return M
        return estimate_transform(kind, p, q).params
    M = fit(P, Q)
    res = Registration(M, f"landmarks ({kind})", n_points=len(P)).apply(P) - Q
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    loo = []
    if len(P) > need + 1:
        for i in range(len(P)):
            keep = np.arange(len(P)) != i
            Mi = fit(P[keep], Q[keep])
            loo.append(float(np.hypot(*(Registration(Mi, "").apply(P[i:i + 1])[0] - Q[i]))))
    err = float(np.sqrt(np.mean(np.square(loo)))) if loo else rms
    return Registration(M, f"landmarks ({kind})", err, len(P), {"rms_residual": rms, "leave_one_out": bool(loo)})


def register_lattice(fit_a: LatticeFit, fit_b: LatticeFit, coarse: Optional[Registration] = None) -> Registration:
    """
    The map that carries lattice A onto lattice B.  The linear part comes from the two lattice
    bases (choosing, among the symmetry-equivalent bases of B, the one closest to the coarse
    registration, or to no rotation); the shift is fixed only up to a lattice vector, so the
    coarse registration (e.g. phase correlation, or one landmark) picks the right one.
    """
    sa, sb = np.asarray(fit_a.scale), np.asarray(fit_b.scale)
    Ap = np.column_stack([fit_a.a1, fit_a.a2]) / sa[:, None]       # pixel-space bases
    Bp = np.column_stack([fit_b.a1, fit_b.a2]) / sb[:, None]
    oa, ob = fit_a.origin / sa, fit_b.origin / sb
    guess = coarse.matrix[:2, :2] if coarse is not None else np.eye(2)
    best = None
    for t in [(1, 0, 0, 1), (0, -1, 1, -1), (-1, 1, -1, 0), (-1, 0, 0, -1), (0, 1, -1, 1), (1, -1, 1, 0),
              (0, -1, 1, 0), (0, 1, -1, 0)]:
        U = np.array([[t[0], t[1]], [t[2], t[3]]], float)
        L = (Bp @ U) @ np.linalg.inv(Ap)
        d = np.linalg.norm(L - guess)
        if best is None or d < best[0]:
            best = (d, L, U)
    _, L, U = best  # type: ignore[misc]
    # x_b = L (x_a - oa) + ob + Bp k  for an unknown integer k
    base = ob - L @ oa
    if coarse is not None:
        c = np.array([fit_a.shape[1] / 2, fit_a.shape[0] / 2]) if fit_a.shape[0] else np.zeros(2)
        target = coarse.apply(c[None])[0]
        k = np.round(np.linalg.solve(Bp, target - (L @ c + base)))
    else:
        k = np.zeros(2)
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = base + Bp @ k
    # error: origin uncertainties of both fits, in pixels of B
    ea = math.sqrt(max(0.0, fit_a.cov[0, 0] / sa[0] ** 2 + fit_a.cov[1, 1] / sa[1] ** 2) / 2)
    eb = math.sqrt(max(0.0, fit_b.cov[0, 0] / sb[0] ** 2 + fit_b.cov[1, 1] / sb[1] ** 2) / 2)
    return Registration(M, "lattice fits", math.hypot(ea, eb), int(fit_a.n_sites + fit_b.n_sites),
                        {"basis_change": U.tolist(), "lattice_shift": k.tolist(),
                         "note": "exact modulo one lattice vector: the shift comes from the coarse registration"
                         if coarse is not None else "the shift is only known modulo a lattice vector"})


def link_objects(a: Sequence[Any], b: Sequence[Any], reg: Optional[Registration] = None,
                 max_distance: float = 5.0) -> dict[str, Any]:
    """
    Pair objects of A with objects of B (Hungarian assignment on registered positions, gated at
    ``max_distance`` pixels of B).  Returns ``{"pairs": [(i, j, distance)], "only_a": [...],
    "only_b": [...]}`` (``only_a``: disappeared / not seen in B; ``only_b``: appeared).
    """
    from scipy.optimize import linear_sum_assignment
    reg = reg or Registration.identity()
    PA = np.array([o.xy for o in a], float).reshape(-1, 2)
    PB = np.array([o.xy for o in b], float).reshape(-1, 2)
    if len(PA) == 0 or len(PB) == 0:
        return {"pairs": [], "only_a": list(range(len(PA))), "only_b": list(range(len(PB)))}
    QA = reg.apply(PA)
    C = np.hypot(QA[:, None, 0] - PB[None, :, 0], QA[:, None, 1] - PB[None, :, 1])
    big = 1e6
    Cg = np.where(C <= max_distance, C, big)
    ri, ci = linear_sum_assignment(Cg)
    pairs = [(int(i), int(j), float(C[i, j])) for i, j in zip(ri, ci) if C[i, j] <= max_distance]
    ma = {p[0] for p in pairs}
    mb = {p[1] for p in pairs}
    return {"pairs": pairs, "only_a": [i for i in range(len(PA)) if i not in ma],
            "only_b": [j for j in range(len(PB)) if j not in mb]}


def resample(image: Any, reg: Registration, shape: tuple[int, int]) -> np.ndarray:
    """Image A resampled onto B's pixel grid (bilinear, NaN outside A)."""
    from scipy import ndimage
    a = np.asarray(getattr(image, "data", image), float)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    inv = reg.inverse()
    p = inv.apply(np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5]))
    out = ndimage.map_coordinates(np.nan_to_num(a), [p[:, 1] - 0.5, p[:, 0] - 0.5], order=1, cval=np.nan)
    inside = (p[:, 0] >= 0) & (p[:, 0] <= a.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] <= a.shape[0])
    return np.where(inside, out, np.nan).reshape(rows, cols)
