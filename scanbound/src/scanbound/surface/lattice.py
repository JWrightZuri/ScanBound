"""
Surface lattices: find the atoms (or repeating motifs), fit a lattice to them, index every site.

    sites = detect_sites(image)                          # local maxima, sub-pixel, with a score
    fit = fit_lattice(sites, image)                      # robust fit: origin, a1, a2 + covariance
    fit.measurements()["a1"].explain()                   # |a1| = 0.2771 ± 0.0058 nm ...
    fit.indices                                          # (m, n) of every site, -1 when not indexed
    fit.to_fractional(xy)                                # pixel -> (u, v) lattice coordinates
    fit.predicted(image.shape)                           # where the lattice puts sites in the image
    bravais_candidates(fit)                              # oblique / rectangular / square / hexagonal ...

Coordinates
-----------
Inputs and outputs are *pixel* coordinates (x right, y down, the annotations' convention).
Everything is fitted in *physical* coordinates ``(x * dx, y * dy)`` so non-square pixels give the
right lattice.  Angles follow the rest of ScanBound: degrees counter-clockwise from the image x axis
as seen on screen.

Robustness
----------
Indices are assigned by growing outward from one site through its neighbours, one lattice step at
a time, so slow drift or a curved scan does not break the indexing the way a global rounding would.
Sites that cannot be reached by a consistent step, or that land on an index already taken by a
closer site, are kept as *extra* (candidate adatoms, noise, a different domain); they do not enter
the fit.  The fit itself is an iteratively reweighted least-squares fit (Huber weights), so a few
badly placed sites cannot pull the lattice.

What the uncertainty covers: the scatter of the sites around the fitted lattice (statistical) and
the pixel-size calibration.  It does *not* cover thermal drift, piezo creep or non-linearity, which
distort the whole image the same way the lattice is distorted: compare up/down or forward/backward
scans, or calibrate against a known lattice (:func:`calibrate_from_lattice`), before reading small
differences from a reference value as physics.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..measurements.measurement import Measurement, combine
from ..model.units import format_value

Scale = tuple[float, float]

EXTRA, LATTICE = 0, 1                    # site status codes


# ======================================================================== helpers
def image_scale(image: Any) -> tuple[Scale, str, tuple[float, float], list[str]]:
    """((dx, dy), unit, (rel_x, rel_y), calibration notes) of an Image (pixels when uncalibrated)."""
    if image is None or not getattr(image, "calibrated", False):
        return (1.0, 1.0), "px", (0.0, 0.0), ["no calibration: lengths are in pixels"]
    dx, dy = image.pixel_size
    notes = [f"scale x: {image.x.calibration.describe()}"]
    if image.y.calibration != image.x.calibration:
        notes.append(f"scale y: {image.y.calibration.describe()}")
    return (float(dx), float(dy)), image.x.unit, (image.x.calibration.rel_uncertainty,
                                                  image.y.calibration.rel_uncertainty), notes


def _data(image: Any) -> np.ndarray:
    return np.asarray(getattr(image, "data", image), dtype=float)


def _filled(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a.astype(float, copy=True)
    if bad.all():
        return np.zeros_like(a, dtype=float)
    _, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
    return a[iy, ix].astype(float)


def _plane_removed(a: np.ndarray) -> np.ndarray:
    rows, cols = a.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    X = np.column_stack([np.ones(a.size), xx.ravel(), yy.ravel()])
    coef, *_ = np.linalg.lstsq(X, a.ravel(), rcond=None)
    return a - (X @ coef).reshape(a.shape)


def screen_angle(v: Sequence[float]) -> float:
    """Screen angle (degrees CCW, y up) of a vector given in y-down coordinates."""
    return math.degrees(math.atan2(-float(v[1]), float(v[0]))) % 360.0


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def gamma_deg(a1: np.ndarray, a2: np.ndarray) -> float:
    c = float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def reduce_basis(a1: Sequence[float], a2: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    The reduced (shortest) basis of the lattice spanned by a1, a2 (Lagrange-Gauss reduction), in a
    fixed convention: |a1| <= |a2|, a1 -> a2 counter-clockwise on screen, angle between them >= 90°
    (120° for hexagonal lattices), a1 the one closest to the image +x direction among equivalent
    choices.
    """
    u, v = np.asarray(a1, float), np.asarray(a2, float)
    for _ in range(100):
        if np.dot(v, v) < np.dot(u, u):
            u, v = v, u
        mu = round(float(np.dot(u, v) / np.dot(u, u)))
        if mu == 0:
            break
        v = v - mu * u
    best, best_key = (u, v), None
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    tol = 1e-3 * max(lu, lv)
    cands = []
    for p, q in ((u, v), (v, u)):
        for s1 in (1, -1):
            for s2 in (1, -1):
                cands.append((s1 * p, s2 * q))
    for p, q in cands:
        # also the equivalent pair (p, q - p) / (p, q + p) when |q -/+ p| == |q| (hexagonal, centred)
        for qq in (q, q - p, q + p):
            if abs(np.linalg.norm(qq) - np.linalg.norm(q)) > tol and qq is not q:
                continue
            if np.linalg.norm(p) > np.linalg.norm(qq) + tol:
                continue
            ccw = _cross(p, qq) < 0                    # y down: negative cross product = CCW on screen
            g = gamma_deg(p, qq)
            ang = screen_angle(p)
            key = (not ccw, g < 89.0, round(min(ang, 360.0 - ang), 6))
            if best_key is None or key < best_key:
                best, best_key = (p, qq), key
    return np.asarray(best[0], float), np.asarray(best[1], float)


# ======================================================================== sites
@dataclass
class SiteSet:
    """Candidate atomic sites (or motif centres) of one image, in pixel coordinates."""

    xy: np.ndarray                         # (N, 2) pixel coordinates
    score: np.ndarray                      # (N,) prominence in robust standard deviations
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.xy)

    def subset(self, keep: np.ndarray) -> "SiteSet":
        return SiteSet(self.xy[keep], self.score[keep], dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        return {"xy": np.round(self.xy, 3).tolist(), "score": np.round(self.score, 3).tolist(), "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "SiteSet":
        xy = np.asarray(d.get("xy", []), float).reshape(-1, 2)
        sc = np.asarray(d.get("score", np.ones(len(xy))), float)
        return cls(xy, sc, dict(d.get("params") or {}))


def fft_peaks(image: Any, scale: Optional[Scale] = None, n: int = 40, snr_min: float = 4.0,
              min_period: Optional[float] = None, max_period: Optional[float] = None) -> list[dict]:
    """
    Peaks of the power spectrum, strongest first: ``{"k": (kx, ky) in cycles per unit length,
    "period", "angle" (screen, of the real-space wave vector), "power", "snr"}``.  Only one of each
    ±k pair is returned.  ``snr`` is the peak power over the median power at the same |k|.
    """
    if scale is None:
        scale = image_scale(image)[0]
    a = _plane_removed(_filled(_data(image)))
    rows, cols = a.shape
    win = np.outer(np.hanning(rows), np.hanning(cols))
    P = np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(rows, scale[1]))
    fx = np.fft.fftshift(np.fft.fftfreq(cols, scale[0]))
    KX, KY = np.meshgrid(fx, fy)
    K = np.hypot(KX, KY)
    Lmin = min(cols * scale[0], rows * scale[1])
    kmin = 1.0 / max_period if max_period else 3.0 / Lmin
    kmax = 1.0 / min_period if min_period else np.inf
    ok = (K >= kmin) & (K <= kmax)
    # radial background: median power in rings of |k|
    kstep = max(abs(fx[1] - fx[0]) if cols > 1 else 1.0, abs(fy[1] - fy[0]) if rows > 1 else 1.0)
    ring = np.round(K / kstep).astype(int)
    bg = ndimage.median(P, labels=ring, index=np.arange(ring.max() + 1))
    bg = np.maximum(np.asarray(bg, float), 1e-300)
    snr = P / bg[ring]
    logp = np.log(P + 1e-300)
    peaks = (logp == ndimage.maximum_filter(logp, size=5)) & ok & (snr >= snr_min)
    iy, ix = np.nonzero(peaks)
    out = []
    for y, x in zip(iy, ix):
        kx, ky = fx[x], fy[y]
        if ky < 0 or (ky == 0 and kx <= 0):
            continue
        # sub-pixel: parabola through the log power on each axis
        def refine(c0, cm, cp):
            d = cm - 2 * c0 + cp
            return 0.5 * (cm - cp) / d if d < 0 else 0.0
        dx = refine(logp[y, x], logp[y, x - 1], logp[y, x + 1]) if 0 < x < cols - 1 else 0.0
        dy = refine(logp[y, x], logp[y - 1, x], logp[y + 1, x]) if 0 < y < rows - 1 else 0.0
        kx += dx * (fx[1] - fx[0] if cols > 1 else 0.0)
        ky += dy * (fy[1] - fy[0] if rows > 1 else 0.0)
        kk = math.hypot(kx, ky)
        out.append({"k": (float(kx), float(ky)), "period": 1.0 / kk, "angle": screen_angle((kx, ky)) % 180.0,
                    "power": float(P[y, x]), "snr": float(snr[y, x]), "pixel": (int(x), int(y))})
    out.sort(key=lambda p: -p["snr"])
    return out[:n]


def lattice_from_fft(image: Any, scale: Optional[Scale] = None, **kw: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Real-space primitive vectors (physical, y down) from the two strongest non-collinear peaks."""
    if scale is None:
        scale = image_scale(image)[0]
    peaks = fft_peaks(image, scale, **kw)
    if len(peaks) < 2:
        return None
    k1 = np.array(peaks[0]["k"])
    strong = [p for p in peaks if p["snr"] >= 0.25 * peaks[0]["snr"]]
    k2 = None
    for p in strong[1:]:
        k = np.array(p["k"])
        ang = abs(math.degrees(math.atan2(_cross(k1, k), float(np.dot(k1, k)))))
        if 20 < ang < 160:
            k2 = k
            break
    if k2 is None:
        return None
    A = np.linalg.inv(np.array([k1, k2]))           # rows of K: k_i . a_j = delta_ij  ->  A = K^-1 (columns)
    return reduce_basis(A[:, 0], A[:, 1])


def estimate_spacing(image: Any, scale: Optional[Scale] = None) -> Optional[float]:
    """Typical nearest-neighbour period (physical units) from the strongest Fourier peak."""
    vecs = lattice_from_fft(image, scale)
    if vecs is not None:
        return float(min(np.linalg.norm(vecs[0]), np.linalg.norm(vecs[1])))
    peaks = fft_peaks(image, scale, n=8)
    return float(peaks[0]["period"]) if peaks else None


def detect_sites(image: Any, scale: Optional[Scale] = None, *, spacing: Optional[float] = None,
                 bright: bool = True, threshold: float = 1.5, smooth: float = 0.15, min_separation: float = 0.55,
                 mask: Optional[np.ndarray] = None, max_sites: int = 50000) -> SiteSet:
    """
    Atomic sites (or motif centres) as local extrema, with sub-pixel positions.

    ``spacing``: expected nearest-neighbour distance (physical units; estimated from the Fourier
    transform when None).  ``bright``: sites are maxima (False: minima, e.g. ncAFM Δf).
    ``threshold``: minimum prominence in robust standard deviations of the high-passed image.
    ``smooth``: Gaussian smoothing as a fraction of the spacing.  ``min_separation``: two sites
    closer than this fraction of the spacing are merged (the stronger one is kept).
    ``mask``: bool array, True where sites may be found.
    """
    if scale is None:
        scale = image_scale(image)[0]
    a = _filled(_data(image))
    valid = np.isfinite(_data(image))
    if not bright:
        a = -a
    if spacing is None:
        spacing = estimate_spacing(image, scale)
        if spacing is None:
            spacing = 8.0 * math.sqrt(scale[0] * scale[1])
    sp = (spacing / scale[0], spacing / scale[1])                       # spacing in pixels along x, y
    sig = (max(0.5, smooth * sp[1]), max(0.5, smooth * sp[0]))          # (rows, cols)
    sm = ndimage.gaussian_filter(a, sig)
    bgsig = (max(1.0, 1.5 * sp[1]), max(1.0, 1.5 * sp[0]))
    hp = sm - ndimage.gaussian_filter(a, bgsig)
    med = float(np.median(hp))
    mad = float(np.median(np.abs(hp - med))) * 1.4826 or float(hp.std()) or 1.0
    z = (hp - med) / mad
    fr = max(1, int(round(min_separation * sp[1]))), max(1, int(round(min_separation * sp[0])))
    yy, xx = np.ogrid[-fr[0]:fr[0] + 1, -fr[1]:fr[1] + 1]
    foot = (yy / max(fr[0], 1)) ** 2 + (xx / max(fr[1], 1)) ** 2 <= 1.0
    peak = (hp == ndimage.maximum_filter(hp, footprint=foot)) & (z >= threshold) & valid
    if mask is not None:
        peak &= np.asarray(mask, bool)
    iy, ix = np.nonzero(peak)
    rows, cols = a.shape
    xs, ys = ix.astype(float) + 0.5, iy.astype(float) + 0.5            # pixel centres
    inner_x = (ix > 0) & (ix < cols - 1)
    inner_y = (iy > 0) & (iy < rows - 1)
    c = hp[iy, ix]
    l_ = np.where(inner_x, hp[iy, np.clip(ix - 1, 0, cols - 1)], c)
    r_ = np.where(inner_x, hp[iy, np.clip(ix + 1, 0, cols - 1)], c)
    d = l_ - 2 * c + r_
    xs += np.where(d < 0, 0.5 * (l_ - r_) / np.where(d < 0, d, -1), 0.0).clip(-0.5, 0.5)
    u_ = np.where(inner_y, hp[np.clip(iy - 1, 0, rows - 1), ix], c)
    b_ = np.where(inner_y, hp[np.clip(iy + 1, 0, rows - 1), ix], c)
    d = u_ - 2 * c + b_
    ys += np.where(d < 0, 0.5 * (u_ - b_) / np.where(d < 0, d, -1), 0.0).clip(-0.5, 0.5)
    score = z[iy, ix]
    order = np.argsort(-score)[:max_sites]
    params = {"spacing": float(spacing), "bright": bright, "threshold": threshold, "smooth": smooth,
              "min_separation": min_separation, "method": "local extrema of the band-passed image"}
    return SiteSet(np.column_stack([xs, ys])[order], score[order], params)


def lattice_from_sites(xy_phys: np.ndarray, k: int = 6) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Primitive vectors (physical) from the most common nearest-neighbour vectors of the sites."""
    P = np.asarray(xy_phys, float)
    if len(P) < 4:
        return None
    tree = cKDTree(P)
    kk = min(k + 1, len(P))
    d, j = tree.query(P, kk)
    vecs = (P[j[:, 1:]] - P[:, None, :]).reshape(-1, 2)
    dist = d[:, 1:].ravel()
    ok = np.isfinite(dist) & (dist > 0)
    vecs, dist = vecs[ok], dist[ok]
    if len(vecs) < 4:
        return None
    nn = float(np.median(d[:, 1]))
    vecs = vecs[dist < 2.5 * nn]
    # fold +v / -v together
    flip = (vecs[:, 1] < 0) | ((vecs[:, 1] == 0) & (vecs[:, 0] < 0))
    vecs[flip] *= -1
    vt = cKDTree(vecs)
    counts = np.array([len(vt.query_ball_point(v, 0.18 * np.linalg.norm(v))) for v in vecs])

    def centre(v):
        members = vecs[vt.query_ball_point(v, 0.18 * np.linalg.norm(v))]
        return members.mean(axis=0)

    order = np.lexsort((np.linalg.norm(vecs, axis=1), -counts))
    best = counts[order[0]]
    cand = [i for i in order if counts[i] >= 0.4 * best]
    cand.sort(key=lambda i: np.linalg.norm(vecs[i]))
    v1 = centre(vecs[cand[0]])
    v2 = None
    for i in cand[1:]:
        v = vecs[i]
        ang = abs(math.degrees(math.atan2(_cross(v1, v), float(np.dot(v1, v)))))
        if 20 < ang < 160:
            v2 = centre(v)
            break
    if v2 is None:
        return None
    return reduce_basis(v1, v2)


# ======================================================================== fit
@dataclass
class LatticeFit:
    """
    A fitted lattice r = origin + m a1 + n a2 (physical coordinates, y down) and its sites.

    ``params`` are (ox, oy, a1x, a1y, a2x, a2y) with covariance ``cov`` (6 x 6, statistical).
    ``status`` per site: 1 = on the lattice (indexed), 0 = extra (not indexed).
    """

    origin: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    cov: np.ndarray
    scale: Scale
    unit: str
    sites: SiteSet
    indices: np.ndarray                    # (N, 2) int; (-1, -1)... meaningless where status == 0
    status: np.ndarray                     # (N,) int
    residual: np.ndarray                   # (N, 2) observed - predicted, physical
    weights: np.ndarray                    # (N,) final IRLS weights (0 = not used)
    rel_calibration: tuple[float, float] = (0.0, 0.0)
    calibration_notes: tuple[str, ...] = ()
    image_id: str = ""
    shape: tuple[int, int] = (0, 0)
    region: Optional[np.ndarray] = None    # polygon (pixel) the fit was restricted to
    name: str = "lattice"
    params_used: dict = field(default_factory=dict)

    # ---- basics -----------------------------------------------------------------
    @property
    def A(self) -> np.ndarray:
        return np.column_stack([self.a1, self.a2])

    @property
    def on_lattice(self) -> np.ndarray:
        return self.status == LATTICE

    @property
    def n_sites(self) -> int:
        return int(self.on_lattice.sum())

    @property
    def rms(self) -> float:
        r = self.residual[self.on_lattice]
        return float(np.sqrt((r ** 2).sum(axis=1).mean())) if len(r) else math.nan

    @property
    def displacement(self) -> np.ndarray:
        """|observed - predicted| per site (physical); NaN for extra sites."""
        d = np.hypot(self.residual[:, 0], self.residual[:, 1])
        return np.where(self.on_lattice, d, np.nan)

    def to_phys(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float).reshape(-1, 2) * np.asarray(self.scale)

    def to_pixel_from_phys(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, float).reshape(-1, 2) / np.asarray(self.scale)

    def to_fractional(self, xy: np.ndarray) -> np.ndarray:
        """Pixel coordinates -> lattice coordinates (u, v): r = origin + u a1 + v a2."""
        P = self.to_phys(xy) - self.origin
        return np.linalg.solve(self.A, P.T).T

    def to_pixel(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        return self.to_pixel_from_phys(self.origin + uv @ self.A.T)

    def direction(self, h: float, k: float) -> np.ndarray:
        """The physical vector h a1 + k a2."""
        return h * self.a1 + k * self.a2

    def direction_angle(self, h: float = 1, k: float = 0) -> float:
        """Screen angle of the lattice direction [h k] (degrees)."""
        return screen_angle(self.direction(h, k))

    def predicted(self, shape: Optional[tuple[int, int]] = None, margin: float = 0.0,
                  region: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """(indices (M, 2), pixel positions (M, 2)) of every lattice site inside the image."""
        rows, cols = shape or self.shape
        corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]], float)
        uv = self.to_fractional(corners)
        lo, hi = np.floor(uv.min(axis=0)) - 1, np.ceil(uv.max(axis=0)) + 1
        if np.prod(hi - lo) > 4e6:
            raise ValueError("the lattice is too fine for this image (more than 4 million sites)")
        m, n = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        idx = np.column_stack([m.ravel(), n.ravel()]).astype(int)
        xy = self.to_pixel(idx)
        inside = (xy[:, 0] >= margin) & (xy[:, 0] <= cols - margin) & (xy[:, 1] >= margin) & (xy[:, 1] <= rows - margin)
        poly = region if region is not None else self.region
        if poly is not None and len(poly) >= 3:
            from matplotlib.path import Path
            inside &= Path(np.asarray(poly, float)).contains_points(xy)
        return idx[inside], xy[inside]

    # ---- derived quantities with uncertainty --------------------------------------
    def _derived(self, f) -> tuple[float, float]:  # noqa: ANN001
        p0 = np.concatenate([self.origin, self.a1, self.a2])
        v0 = f(p0)
        J = np.zeros(6)
        for i in range(6):
            h = 1e-6 * max(1e-12, np.abs(p0[2:]).max())
            p = p0.copy()
            p[i] += h
            J[i] = (f(p) - v0) / h
        var = float(J @ self.cov @ J)
        return float(v0), math.sqrt(max(var, 0.0))

    def measurements(self) -> dict[str, Measurement]:
        """|a1|, |a2|, γ, the orientation of a1, the cell area and the fit scatter, with uncertainties."""
        unit = self.unit
        notes = tuple(self.calibration_notes)
        inp = {"image": self.image_id, "lattice": self.name, "sites": self.n_sites}
        method = (f"robust least-squares lattice fit to {self.n_sites} indexed sites "
                  f"(Huber-weighted, indices grown site by site)")
        drift = "thermal drift, creep and scanner non-linearity are not included (they distort the lattice itself)"
        rel = self.rel_calibration
        out: dict[str, Measurement] = {}
        for name, vec_slice in (("a1", slice(2, 4)), ("a2", slice(4, 6))):
            L, sL = self._derived(lambda p, s=vec_slice: float(np.hypot(*p[s])))
            v = (self.a1 if name == "a1" else self.a2) / max(L, 1e-300)
            cal = L * math.hypot(v[0] * rel[0], v[1] * rel[1])
            comps = {"fit": sL, "calibration": cal}
            out[name] = Measurement(name, L, unit, combine(comps), method, (drift, *notes), inp, comps)
        g, sg = self._derived(lambda p: gamma_deg(p[2:4], p[4:6]))
        out["gamma"] = Measurement("gamma", g, "°", sg, "angle between a1 and a2 from the lattice fit", (drift,), inp,
                                   {"fit": sg})
        t, st = self._derived(lambda p: screen_angle(p[2:4]))
        out["a1_angle"] = Measurement("a1_angle", t, "°", st, "direction of a1, counter-clockwise from the image x axis",
                                      (drift,), inp, {"fit": st})
        ar, sa = self._derived(lambda p: abs(_cross(p[2:4], p[4:6])))
        out["cell_area"] = Measurement("cell_area", ar, f"{unit}²", math.hypot(sa, ar * math.hypot(*rel)),
                                       "area of the primitive cell |a1 × a2|", (drift, *notes), inp,
                                       {"fit": sa, "calibration": ar * math.hypot(*rel)})
        out["rms"] = Measurement("rms", self.rms, unit, None,
                                 "root-mean-square distance of indexed sites from their lattice position "
                                 "(site precision plus real distortions)", (), inp)
        out["sites"] = Measurement("sites", float(self.n_sites), "", None,
                                   f"sites on the lattice ({int((~self.on_lattice).sum())} extra)", (), inp)
        return out

    def describe(self) -> str:
        m = self.measurements()
        return (f"{self.name}: a1 = {m['a1'].format()}, a2 = {m['a2'].format()}, γ = {m['gamma'].format()}, "
                f"a1 at {m['a1_angle'].format()}, {self.n_sites} sites, rms {format_value(self.rms, self.unit)}")

    # ---- persistence ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "image": self.image_id, "origin": self.origin.tolist(), "a1": self.a1.tolist(),
                "a2": self.a2.tolist(), "cov": self.cov.tolist(), "scale": list(self.scale), "unit": self.unit,
                "sites": self.sites.to_dict(), "indices": self.indices.tolist(), "status": self.status.tolist(),
                "weights": np.round(self.weights, 4).tolist(), "rel_calibration": list(self.rel_calibration),
                "calibration_notes": list(self.calibration_notes), "shape": list(self.shape),
                "region": None if self.region is None else np.asarray(self.region).tolist(),
                "params_used": self.params_used}

    @classmethod
    def from_dict(cls, d: dict) -> "LatticeFit":
        sites = SiteSet.from_dict(d["sites"])
        origin, a1, a2 = (np.asarray(d[k], float) for k in ("origin", "a1", "a2"))
        idx = np.asarray(d.get("indices", []), int).reshape(-1, 2)
        status = np.asarray(d.get("status", []), int)
        scale = tuple(d.get("scale", (1.0, 1.0)))
        A = np.column_stack([a1, a2])
        res = sites.xy * np.asarray(scale) - (origin + idx @ A.T)
        return cls(origin, a1, a2, np.asarray(d.get("cov", np.zeros((6, 6))), float), scale,  # type: ignore[arg-type]
                   str(d.get("unit", "px")), sites, idx, status, res, np.asarray(d.get("weights", status), float),
                   tuple(d.get("rel_calibration", (0.0, 0.0))), tuple(d.get("calibration_notes", ())),  # type: ignore[arg-type]
                   str(d.get("image", "")), tuple(d.get("shape", (0, 0))),  # type: ignore[arg-type]
                   None if d.get("region") is None else np.asarray(d["region"], float), str(d.get("name", "lattice")),
                   dict(d.get("params_used") or {}))


def _index_sites(P: np.ndarray, A: np.ndarray, start: int, step_tol: float, reach: float) -> np.ndarray:
    """Grow integer indices outward from ``start`` through neighbours; NaN where unreachable."""
    Ainv = np.linalg.inv(A)
    amin = min(np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1]))
    tree = cKDTree(P)
    idx = np.full((len(P), 2), np.nan)
    remaining = set(range(len(P)))

    def grow(s: int, seed_idx: np.ndarray) -> None:
        idx[s] = seed_idx
        remaining.discard(s)
        q = deque([s])
        while q:
            i = q.popleft()
            for j in tree.query_ball_point(P[i], reach):
                if j not in remaining:
                    continue
                f = Ainv @ (P[j] - P[i])
                st = np.round(f)
                if np.linalg.norm(A @ (f - st)) < step_tol * amin:
                    idx[j] = idx[i] + st
                    remaining.discard(j)
                    q.append(j)

    grow(start, np.zeros(2))
    # other patches (separated by a gap wider than ``reach``): seed them from the global lattice
    origin = P[start]
    for _ in range(50):
        if len(remaining) < 3:
            break
        rem = np.array(sorted(remaining))
        f = (Ainv @ (P[rem] - origin).T).T
        resid = np.linalg.norm((f - np.round(f)) @ A.T, axis=1)
        s = int(rem[np.argmin(resid)])
        if resid.min() > step_tol * amin:
            break
        before = len(remaining)
        grow(s, np.round(Ainv @ (P[s] - origin)))
        if before - len(remaining) < 3:
            break
    return idx


def _weighted_fit(P: np.ndarray, idx: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted LSQ of P = o + m a1 + n a2.  Returns (params (6), cov (6x6), residual)."""
    X = np.column_stack([np.ones(len(idx)), idx[:, 0], idx[:, 1]])
    sw = np.sqrt(w)
    XtWX = (X * w[:, None]).T @ X
    bx = np.linalg.lstsq(X * sw[:, None], P[:, 0] * sw, rcond=None)[0]
    by = np.linalg.lstsq(X * sw[:, None], P[:, 1] * sw, rcond=None)[0]
    res = P - np.column_stack([X @ bx, X @ by])
    n_eff = float(w.sum())
    dof = max(1.0, 2 * n_eff - 6)
    s2 = float((w * (res ** 2).sum(axis=1)).sum() / dof)
    try:
        C = s2 * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        C = np.full((3, 3), np.inf)
    cov = np.zeros((6, 6))
    # parameter order (ox, oy, a1x, a1y, a2x, a2y); x block = (ox, a1x, a2x), y block = (oy, a1y, a2y)
    xi, yi = [0, 2, 4], [1, 3, 5]
    for a in range(3):
        for b in range(3):
            cov[xi[a], xi[b]] = C[a, b]
            cov[yi[a], yi[b]] = C[a, b]
    params = np.array([bx[0], by[0], bx[1], by[1], bx[2], by[2]])
    return params, cov, res


def fit_lattice(sites: SiteSet, image: Any = None, *, scale: Optional[Scale] = None,
                a1: Optional[Sequence[float]] = None, a2: Optional[Sequence[float]] = None,
                origin_xy: Optional[Sequence[float]] = None, step_tol: float = 0.3, min_score: Optional[float] = None,
                region: Optional[np.ndarray] = None, name: str = "lattice", huber: float = 1.345,
                vectors_in_pixels: bool = False) -> LatticeFit:
    """
    Fit a lattice to ``sites``.

    ``a1``, ``a2``: starting vectors (physical units, y down; pixels when ``vectors_in_pixels``) -
    e.g. from two lattice-vector annotations.  Found from the sites, or from the Fourier transform
    when the sites are too irregular, when not given.  ``origin_xy``: the site (pixel) to start
    indexing from (index (0, 0)); the strongest site near the centre otherwise.  ``step_tol``: a
    neighbour is accepted as one lattice step away when it lies within this fraction of the shorter
    lattice vector of the predicted position.  ``region``: polygon (pixels) to restrict the fit to.
    """
    if scale is None:
        scale = image_scale(image)[0]
    _, unit, rel, notes = image_scale(image) if image is not None else (scale, "px", (0.0, 0.0), [])
    if image is None and scale != (1.0, 1.0):
        unit = "m"
    use = np.ones(len(sites), bool)
    if min_score is not None:
        use &= sites.score >= min_score
    if region is not None and len(region) >= 3:
        from matplotlib.path import Path
        use &= Path(np.asarray(region, float)).contains_points(sites.xy)
    S = sites.subset(use)
    if len(S) < 4:
        raise ValueError(f"only {len(S)} site(s) to fit: at least 4 are needed (lower the threshold or check the "
                         "region)")
    P = S.xy * np.asarray(scale)
    if a1 is not None and a2 is not None:
        v1, v2 = np.asarray(a1, float), np.asarray(a2, float)
        if vectors_in_pixels:
            v1, v2 = v1 * np.asarray(scale), v2 * np.asarray(scale)
        if abs(_cross(v1, v2)) < 1e-6 * np.linalg.norm(v1) * np.linalg.norm(v2):
            raise ValueError("the two starting vectors are parallel")
    else:
        guess = lattice_from_sites(P)
        if guess is None and image is not None:
            guess = lattice_from_fft(image, scale)
        if guess is None:
            raise ValueError("could not find two lattice vectors: draw two lattice vectors as a starting guess")
        v1, v2 = guess
    A = np.column_stack([v1, v2])
    amax = max(np.linalg.norm(v1), np.linalg.norm(v2))
    if origin_xy is not None:
        starts = [int(np.argmin(np.hypot(*(S.xy - np.asarray(origin_xy, float)).T)))]
    else:
        # a few typical sites near the centre (not the brightest: that may be an adatom); keep the
        # start that indexes the most sites
        c = P.mean(axis=0)
        dc = np.hypot(*(P - c).T)
        cand = np.argsort(dc)[:max(5, len(P) // 20)]
        med = np.median(S.score)
        starts = [int(i) for i in cand[np.argsort(np.abs(S.score[cand] - med))][:5]]
    best_idx, best_n = None, -1
    for start in starts:
        idx = _index_sites(P, A, start, step_tol, 2.2 * amax)
        n_ok = int(np.isfinite(idx[:, 0]).sum())
        if n_ok > best_n:
            best_idx, best_n = idx, n_ok
        if n_ok > 0.9 * len(P):
            break
    idx = best_idx
    assigned = np.isfinite(idx[:, 0])
    if assigned.sum() < 4:
        raise ValueError("fewer than 4 sites could be indexed: check the starting vectors or the site detection")
    idx_i = np.where(assigned[:, None], idx, 0).astype(int)
    status = assigned.astype(int)
    # duplicates: one site per index (the one closest to the lattice wins, after a first fit)
    w = status.astype(float)
    params = None
    cov = np.zeros((6, 6))
    res = np.zeros_like(P)
    for it in range(12):
        params, cov, res = _weighted_fit(P, idx_i, np.where(status == LATTICE, np.maximum(w, 1e-9), 0.0))
        d = np.hypot(res[:, 0], res[:, 1])
        if it == 0 or it == 3:
            keys: dict[tuple[int, int], int] = {}
            for i in np.nonzero(status == LATTICE)[0]:
                k = (int(idx_i[i, 0]), int(idx_i[i, 1]))
                j = keys.get(k)
                if j is None or d[i] < d[j]:
                    if j is not None:
                        status[j] = EXTRA
                    keys[k] = i
                else:
                    status[i] = EXTRA
        on = status == LATTICE
        s = 1.4826 * float(np.median(d[on])) if on.any() else 1.0
        s = max(s, 1e-12 * amax)
        cth = huber * s
        w_new = np.where(d <= cth, 1.0, cth / np.maximum(d, 1e-300)) * on
        if np.allclose(w_new, w, atol=1e-4):
            w = w_new
            break
        w = w_new
    assert params is not None
    origin, va1, va2 = params[0:2], params[2:4], params[4:6]
    # re-express in the reduced, conventional basis (indices transform with it)
    r1, r2 = reduce_basis(va1, va2)
    T = np.linalg.solve(np.column_stack([va1, va2]), np.column_stack([r1, r2]))   # [r1 r2] = [a1 a2] T
    Tr = np.round(T)
    if np.allclose(T, Tr, atol=1e-6) and abs(round(np.linalg.det(Tr))) == 1:
        Ti = np.round(np.linalg.inv(Tr)).astype(int)
        idx_i = (Ti @ idx_i.T).T
        G = np.zeros((6, 6))                       # params' = G params (origin unchanged)
        G[0, 0] = G[1, 1] = 1
        for col in range(2):                       # r_col = sum_j T[j, col] a_j
            for comp in range(2):
                for j in range(2):
                    G[2 + 2 * col + comp, 2 + 2 * j + comp] = Tr[j, col]
        cov = G @ cov @ G.T
        va1, va2 = r1, r2
    # extra sites: index and residual of the nearest lattice position (for defect analysis)
    Afin = np.column_stack([va1, va2])
    ex = np.nonzero(status != LATTICE)[0]
    if len(ex):
        f = np.linalg.solve(Afin, (P[ex] - origin).T).T
        idx_i[ex] = np.round(f).astype(int)
        res[ex] = P[ex] - (origin + np.round(f) @ Afin.T)
    full_idx = np.full((len(sites), 2), -1, int)
    full_status = np.zeros(len(sites), int)
    full_res = np.zeros((len(sites), 2))
    full_w = np.zeros(len(sites))
    where = np.nonzero(use)[0]
    full_idx[where] = idx_i
    full_status[where] = status
    full_res[where] = res
    full_w[where] = w
    # residuals of the sites left out of the fit (score / region), against the fitted lattice
    Afit = np.column_stack([va1, va2])
    out_of_fit = np.nonzero(~use)[0]
    if len(out_of_fit):
        Po = sites.xy[out_of_fit] * np.asarray(scale)
        f = np.linalg.solve(Afit, (Po - origin).T).T
        full_res[out_of_fit] = Po - (origin + np.round(f) @ Afit.T)
    shape = tuple(_data(image).shape) if image is not None else (0, 0)
    return LatticeFit(np.asarray(origin, float), np.asarray(va1, float), np.asarray(va2, float), cov,
                      (float(scale[0]), float(scale[1])), unit, sites, full_idx, full_status, full_res, full_w, rel,
                      tuple(notes), str(getattr(image, "id", "")), shape,  # type: ignore[arg-type]
                      None if region is None else np.asarray(region, float), name,
                      {"step_tol": step_tol, "min_score": min_score, "huber": huber,
                       "start": "given" if a1 is not None else "automatic"})


def fit_image(image: Any, *, spacing: Optional[float] = None, bright: bool = True, threshold: float = 1.5,
              a1: Optional[Sequence[float]] = None, a2: Optional[Sequence[float]] = None,
              region: Optional[np.ndarray] = None, step_tol: float = 0.3, name: str = "lattice",
              mask: Optional[np.ndarray] = None) -> LatticeFit:
    """Detect sites and fit a lattice in one go.  When starting vectors are given (physical units)
    the expected spacing comes from them."""
    scale = image_scale(image)[0]
    if spacing is None and a1 is not None and a2 is not None:
        spacing = float(min(np.linalg.norm(a1), np.linalg.norm(a2)))
    sites = detect_sites(image, scale, spacing=spacing, bright=bright, threshold=threshold, mask=mask)
    return fit_lattice(sites, image, a1=a1, a2=a2, region=region, step_tol=step_tol, name=name)


# ======================================================================== symmetry
@dataclass(frozen=True)
class BravaisCandidate:
    name: str
    consistent: bool
    length_dev: float          # relative deviation of the constrained lengths (0 if none)
    angle_dev: float           # deviation of the constrained angle, degrees (0 if none)
    chi2: float                # deviation over its tolerance, squared and summed
    note: str


BRAVAIS_ORDER = ("hexagonal", "square", "centred rectangular", "rectangular", "oblique")


def bravais_candidates(fit: LatticeFit, length_tol: float = 0.02, angle_tol: float = 1.5,
                       n_sigma: float = 3.0) -> list[BravaisCandidate]:
    """
    Which 2D Bravais lattices are consistent with the fit, most symmetric first.

    A constraint (equal lengths, a right angle, 120°) is *consistent* when the fitted value meets
    it within ``n_sigma`` statistical uncertainties *or* within the tolerance ``length_tol``
    (relative) / ``angle_tol`` (degrees), whichever is larger.  The tolerance stands for drift and
    calibration errors, which the statistical uncertainty does not include: set it from what your
    instrument does on a reference lattice.
    """
    m = fit.measurements()
    L1, L2, g = m["a1"].value, m["a2"].value, m["gamma"].value
    sL = math.hypot(m["a1"].components.get("fit", 0.0), m["a2"].components.get("fit", 0.0)) / max(L1, 1e-300)
    sg = m["gamma"].uncertainty or 0.0
    tl = max(length_tol, n_sigma * sL)
    ta = max(angle_tol, n_sigma * sg)
    dL = abs(L1 - L2) / ((L1 + L2) / 2)
    # centred rectangular (rhombic): equal lengths; or a2 projected on a1 = |a1| / 2
    proj = abs(np.dot(fit.a1, fit.a2)) / (np.linalg.norm(fit.a1) ** 2)
    dproj = abs(proj - 0.5)
    tests = {
        "hexagonal": ([dL], [abs(g - 120.0)], "|a1| = |a2|, γ = 120°"),
        "square": ([dL], [abs(g - 90.0)], "|a1| = |a2|, γ = 90°"),
        "centred rectangular": ([min(dL, dproj)], [], "|a1| = |a2| (rhombic cell) or a1·a2 = |a1|²/2"),
        "rectangular": ([], [abs(g - 90.0)], "γ = 90°"),
        "oblique": ([], [], "no constraint"),
    }
    out = []
    for name in BRAVAIS_ORDER:
        ls, angs, note = tests[name]
        chi2 = sum((x / tl) ** 2 for x in ls) + sum((x / ta) ** 2 for x in angs)
        ok = all(x <= tl for x in ls) and all(x <= ta for x in angs)
        out.append(BravaisCandidate(name, ok, max(ls) if ls else 0.0, max(angs) if angs else 0.0, chi2, note))
    out.sort(key=lambda c: (not c.consistent, BRAVAIS_ORDER.index(c.name) if c.consistent else c.chi2))
    return out


def symmetry_fold(fit: LatticeFit, **kw: Any) -> int:
    """Rotational symmetry of the best consistent lattice (6, 4 or 2)."""
    best = next((c for c in bravais_candidates(fit, **kw) if c.consistent), None)
    return {"hexagonal": 6, "square": 4}.get(best.name if best else "", 2)


# ======================================================================== calibration
def calibrate_from_lattice(fit: LatticeFit, a_known: float, b_known: Optional[float] = None,
                           gamma_known: Optional[float] = None, rel_uncertainty_known: float = 0.0) -> dict[str, Any]:
    """
    Pixel size that makes the fitted lattice match a known one (e.g. the substrate's surface lattice
    constant from the literature).  Returns ``{"dx", "dy", "rel_uncertainty", "scale_x", "scale_y",
    "shear_deg", "note"}``: ``scale_x`` / ``scale_y`` are the factors the current calibration is off
    by along x and y; ``shear_deg`` the remaining non-orthogonality (drift / skew) that a pixel size
    cannot correct.  Only meaningful for an atomically resolved, single-domain area.
    """
    b_known = b_known or a_known
    gamma_known = gamma_known if gamma_known is not None else gamma_deg(fit.a1, fit.a2)
    # reference basis with the same orientation as the fit
    g = math.radians(gamma_known)
    th = math.atan2(-fit.a1[1], fit.a1[0])                       # screen angle of a1 (y up)
    r1 = np.array([math.cos(th), -math.sin(th)]) * a_known
    r2 = np.array([math.cos(th + g), -math.sin(th + g)]) * b_known
    # observed pixel vectors
    p1, p2 = fit.a1 / np.asarray(fit.scale), fit.a2 / np.asarray(fit.scale)
    # find diagonal D (dx, dy) minimising | D p - r |  (least squares, per axis)
    Px = np.array([p1[0], p2[0]])
    Py = np.array([p1[1], p2[1]])
    Rx = np.array([r1[0], r2[0]])
    Ry = np.array([r1[1], r2[1]])
    dx = float(abs(Px @ Rx / max(Px @ Px, 1e-300)))
    dy = float(abs(Py @ Ry / max(Py @ Py, 1e-300)))
    # how well does a diagonal scale explain it? residual shear
    q1, q2 = p1 * [dx, dy], p2 * [dx, dy]
    shear = gamma_deg(q1, q2) - gamma_known
    m = fit.measurements()
    stat = (m["a1"].components.get("fit", 0.0) / max(m["a1"].value, 1e-300))
    rel = math.hypot(stat, rel_uncertainty_known)
    return {"dx": dx, "dy": dy, "rel_uncertainty": rel, "scale_x": dx / fit.scale[0] if fit.scale[0] else math.nan,
            "scale_y": dy / fit.scale[1] if fit.scale[1] else math.nan, "shear_deg": shear,
            "note": ("pixel size from the known lattice; a remaining shear of "
                     f"{shear:.2f}° is drift or skew that a pixel size cannot remove")}
