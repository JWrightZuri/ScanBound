"""
Periodicities beyond the primitive lattice: reconstructions, superstructures, moiré patterns.

    rep = analyse_periodicities(image, fit)
    rep.peaks              # every Fourier peak with its indices (h, k) in the substrate's reciprocal lattice
    rep.superstructure     # None, or the superstructure cell: matrix, Wood notation, commensurability
    rep.candidates         # user / standard candidate cells ranked by how well they predict the peaks

How it works
------------
Each Fourier peak k is expressed in the reciprocal basis of the fitted real-space lattice:
h = k·a1, k = k·a2 (so the substrate's own peaks have integer indices).  Peaks with clearly
non-integer indices belong to a longer period.  The two strongest non-collinear ones define a
superstructure reciprocal basis, which gives the superstructure cell B; written in the substrate
basis, B = A M.  When M is (close to) an integer matrix the superstructure is commensurate; when it
is close to a fraction with a small denominator it is a coincidence (higher-order commensurate)
lattice; otherwise it is incommensurate - a candidate moiré or an overlayer with its own lattice.

All of this is a *candidate* description: a periodic contrast modulation can come from the
structure, from electronic effects, from the tip, or from feedback / scan artefacts.  The peak list
keeps the signal-to-noise of each peak, so weak "superstructure" peaks can be judged for what they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

import numpy as np

from .lattice import LatticeFit, fft_peaks, gamma_deg, image_scale, reduce_basis, screen_angle


@dataclass
class Periodicity:
    k: tuple[float, float]               # cycles per unit length (y down)
    period: float
    angle: float                          # screen angle of the wave vector, 0..180
    snr: float
    hk: tuple[float, float]               # indices in the substrate's reciprocal basis
    integer: bool                         # |hk - round(hk)| small: a substrate peak (or its harmonic)
    fraction: str                         # e.g. "(1/2, 0)" rational approximation


@dataclass
class Superstructure:
    b1: np.ndarray                        # physical, y down
    b2: np.ndarray
    matrix: np.ndarray                    # B = A M  (float)
    matrix_int: np.ndarray                # rounded
    deviation: float                      # max |M - round(M)|
    kind: str                             # "commensurate" / "coincidence" / "incommensurate"
    wood: str                             # Wood notation when it applies, else ""
    rotation: float                       # angle between b1 and a1, degrees
    ratio: tuple[float, float]            # |b1| / |a1|, |b2| / |a2|
    note: str = ""

    def matrix_text(self) -> str:
        m = self.matrix_int if self.kind == "commensurate" else self.matrix
        fmt = (lambda x: f"{int(round(x))}") if self.kind == "commensurate" else (lambda x: f"{x:.2f}")
        return f"({fmt(m[0, 0])} {fmt(m[1, 0])} / {fmt(m[0, 1])} {fmt(m[1, 1])})"


@dataclass
class CandidateMatch:
    name: str
    matrix: np.ndarray
    predicted: int                        # predicted non-substrate peaks within the analysed range
    matched: int                          # observed non-integer peaks it explains
    explained_fraction: float             # of the observed non-integer peaks
    mean_error: float                     # in reciprocal-lattice units
    score: float


@dataclass
class PeriodicityReport:
    peaks: list[Periodicity]
    superstructure: Optional[Superstructure]
    candidates: list[CandidateMatch] = field(default_factory=list)
    fft_vs_fit: dict[str, float] = field(default_factory=dict)

    def rows(self) -> list[dict[str, Any]]:
        return [{"period": p.period, "angle_deg": p.angle, "snr": p.snr, "h": p.hk[0], "k": p.hk[1],
                 "substrate": p.integer, "fraction": p.fraction} for p in self.peaks]


def _frac(x: float, max_den: int = 8) -> str:
    f = Fraction(x).limit_denominator(max_den)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def wood_notation(fit: LatticeFit, b1: np.ndarray, b2: np.ndarray, tol: float = 0.03) -> str:
    """'(p × q)Rθ' (or c(p × q)) when b1, b2 enclose the same angle as a1, a2 and are rotated alike."""
    a1, a2 = fit.a1, fit.a2
    if abs(gamma_deg(b1, b2) - gamma_deg(a1, a2)) > 2.0:
        # centred cell?  try (b1, b1 + 2 b2)-style conventional cells: report the primitive matrix only
        return ""
    p = np.linalg.norm(b1) / np.linalg.norm(a1)
    q = np.linalg.norm(b2) / np.linalg.norm(a2)
    rot = (screen_angle(b1) - screen_angle(a1) + 180) % 360 - 180
    rot2 = (screen_angle(b2) - screen_angle(a2) + 180) % 360 - 180
    if abs(rot - rot2) > 2.0:
        return ""

    def num(x: float) -> str:
        for root in (1, 2, 3, 5, 7, 13, 19, 21):
            r = x / math.sqrt(root)
            if abs(r - round(r)) < tol * max(1.0, r) and round(r) >= 1:
                n = int(round(r))
                return (f"{n}" if root == 1 else (f"√{root}" if n == 1 else f"{n}√{root}"))
        return f"{x:.2f}"
    txt = f"({num(p)} × {num(q)})"
    if abs(rot) > 0.5:
        txt += f"R{abs(rot):.0f}°"
    return txt


STANDARD_CANDIDATES: dict[str, list[list[int]]] = {
    "(1 × 1)": [[1, 0], [0, 1]],
    "(2 × 1)": [[2, 0], [0, 1]],
    "(1 × 2)": [[1, 0], [0, 2]],
    "(2 × 2)": [[2, 0], [0, 2]],
    "(3 × 3)": [[3, 0], [0, 3]],
    "(√3 × √3)R30° (hex)": [[2, 1], [-1, 1]],
    "(2√3 × 2√3)R30° (hex)": [[4, 2], [-2, 2]],
    "c(2 × 2) (square)": [[1, 1], [-1, 1]],
    "(√7 × √7)R19.1° (hex)": [[3, 1], [-1, 2]],
}


def match_candidates(fit: LatticeFit, peaks: list[Periodicity], candidates: Optional[dict[str, Any]] = None,
                     tol: float = 0.06) -> list[CandidateMatch]:
    """
    Rank candidate superstructure cells (name -> 2 x 2 integer matrix M, columns = b in units of a)
    by how well their reciprocal lattices explain the observed non-substrate peaks, with a penalty
    for predicted peaks that are not observed (weak peaks may simply be invisible, so the penalty is
    mild).
    """
    cands = candidates or STANDARD_CANDIDATES
    obs = np.array([p.hk for p in peaks if not p.integer]).reshape(-1, 2)
    hk_all = np.array([p.hk for p in peaks]).reshape(-1, 2)
    hmax = float(np.abs(hk_all).max()) + 0.05 if len(hk_all) else 1.5
    out = []
    for name, M in cands.items():
        M = np.asarray(M, float)
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        # reciprocal of the superstructure in substrate reciprocal units: rows of inv(M)
        Minv = np.linalg.inv(M)
        rng = range(-8, 9)
        pred = []
        for i in rng:
            for j in rng:
                hk = np.array([i, j]) @ Minv            # (i, j) superstructure index -> substrate (h, k)
                if np.all(np.abs(hk) <= hmax) and not np.allclose(hk, np.round(hk), atol=1e-6):
                    pred.append(hk)
        pred_arr = np.array(pred).reshape(-1, 2)
        if len(obs) == 0:
            out.append(CandidateMatch(name, M, len(pred_arr), 0, 1.0 if len(pred_arr) == 0 else 0.0, 0.0,
                                      1.0 if len(pred_arr) == 0 else 0.0))
            continue
        matched, errs = 0, []
        for o in obs:
            d = np.min(np.minimum(np.linalg.norm(pred_arr - o, axis=1), np.linalg.norm(pred_arr + o, axis=1))) \
                if len(pred_arr) else np.inf
            if d < tol:
                matched += 1
                errs.append(d)
        frac = matched / len(obs)
        n_pred = max(1, len(pred_arr) // 2)
        score = frac - 0.2 * max(0.0, 1.0 - matched / n_pred)     # mild penalty: weak peaks may be invisible
        out.append(CandidateMatch(name, M, len(pred_arr) // 2, matched, frac, float(np.mean(errs)) if errs else math.nan,
                                  score))
    out.sort(key=lambda c: -c.score)
    return out


def analyse_periodicities(image: Any, fit: LatticeFit, *, n_peaks: int = 30, snr_min: Optional[float] = None,
                          integer_tol: float = 0.08, max_index: float = 3.0,
                          candidates: Optional[dict[str, Any]] = None) -> PeriodicityReport:
    """Classify Fourier peaks against the fitted lattice and look for a superstructure.

    ``snr_min`` defaults to the level that pure noise exceeds by chance in about one image in twenty
    (spectral power of noise is exponentially distributed, so the level grows with the pixel count).
    Peaks beyond ``max_index`` substrate reciprocal units are ignored (aliasing and noise live there)."""
    scale = image_scale(image)[0]
    A = np.column_stack([fit.a1, fit.a2])
    if snr_min is None:
        n_pix = float(np.prod(np.shape(getattr(image, "data", image))))
        snr_min = max(6.0, math.log(max(n_pix, 2.0) / 0.05) / math.log(2.0))
    raw = fft_peaks(image, scale, n=4 * n_peaks, snr_min=snr_min)
    peaks: list[Periodicity] = []
    for p in raw:
        k = np.array(p["k"])
        hk = k @ A                                         # h = k·a1, k = k·a2
        if np.abs(hk).max() > max_index:
            continue
        if len(peaks) >= n_peaks:
            break
        if hk[0] < -1e-9 or (abs(hk[0]) < 1e-9 and hk[1] < 0):
            hk = -hk
        integer = bool(np.all(np.abs(hk - np.round(hk)) < integer_tol))
        peaks.append(Periodicity(p["k"], p["period"], p["angle"], p["snr"], (float(hk[0]), float(hk[1])), integer,
                                 f"({_frac(hk[0])}, {_frac(hk[1])})"))
    # substrate peaks: how well do Fourier and real-space lattices agree?
    sub = [p for p in peaks if p.integer and any(abs(round(x)) > 0 for x in p.hk)]
    agree = {}
    if sub:
        dev = [float(np.linalg.norm(np.array(p.hk) - np.round(p.hk))) for p in sub]
        agree = {"substrate_peaks": float(len(sub)), "mean_index_deviation": float(np.mean(dev)),
                 "max_index_deviation": float(np.max(dev))}
    ss = None
    extra = [p for p in peaks if not p.integer]
    if len(extra) >= 2:
        k1 = np.array(extra[0].k)
        k2 = None
        for p in extra[1:]:
            k = np.array(p.k)
            ang = abs(math.degrees(math.atan2(k1[0] * k[1] - k1[1] * k[0], float(k1 @ k))))
            if 15 < ang < 165:
                k2 = k
                break
        if k2 is not None:
            # the superstructure reciprocal lattice contains the substrate's: reduce with substrate
            # reciprocal vectors added, so a harmonic (e.g. 2/3) still gives the right cell
            B = np.linalg.inv(np.array([k1, k2]))
            b1, b2 = reduce_basis(B[:, 0], B[:, 1])
            # among the symmetry-equivalent choices, the cell whose b1 is closest to a1
            from .lattice import symmetry_fold
            fold = symmetry_fold(fit)
            best = None
            for r in range(fold):
                t = math.radians(360.0 * r / fold)
                R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
                c1, c2 = R @ b1, R @ b2
                d = abs((screen_angle(c1) - screen_angle(fit.a1) + 180) % 360 - 180)
                if best is None or d < best[0] - 1e-6:
                    best = (d, c1, c2)
            b1, b2 = best[1], best[2]
            M = np.linalg.solve(A, np.column_stack([b1, b2]))
            Mi = np.round(M)
            dev = float(np.abs(M - Mi).max())
            if dev < 0.08 and abs(np.linalg.det(Mi)) >= 1:
                kind = "commensurate"
            else:
                # coincidence: some small multiple of M is integer
                kind = "incommensurate"
                for n in range(2, 6):
                    if np.abs(n * M - np.round(n * M)).max() < 0.08:
                        kind = "coincidence"
                        break
            rot = (screen_angle(b1) - screen_angle(fit.a1) + 180) % 360 - 180
            note = {"commensurate": "commensurate with the substrate lattice: a reconstruction or ordered overlayer "
                                    "candidate",
                    "coincidence": "a higher-order commensurate (coincidence) cell: a long-period overlayer candidate",
                    "incommensurate": "incommensurate with the fitted lattice: a moiré or independent overlayer "
                                      "candidate (or a non-structural modulation)"}[kind]
            ss = Superstructure(b1, b2, M, Mi, dev, kind, wood_notation(fit, b1, b2) if kind == "commensurate" else "",
                                rot, (float(np.linalg.norm(b1) / np.linalg.norm(fit.a1)),
                                      float(np.linalg.norm(b2) / np.linalg.norm(fit.a2))), note)
    cands = match_candidates(fit, peaks, candidates) if extra or candidates else []
    return PeriodicityReport(peaks, ss, cands, agree)


def moire_period(a: float, b: float, theta_deg: float) -> tuple[float, float]:
    """
    Period and rotation of the moiré of two hexagonal (or square) lattices with constants a, b
    rotated by θ: λ = a b / √(a² + b² - 2 a b cos θ), and its rotation φ relative to lattice ``a``
    (degrees): the moiré's reciprocal vector is G_a - G_b with |G| = 1 / lattice constant.
    """
    t = math.radians(theta_deg)
    den = math.sqrt(a * a + b * b - 2 * a * b * math.cos(t))
    lam = math.inf if den == 0 else a * b / den
    phi = math.degrees(math.atan2(-a * math.sin(t), b - a * math.cos(t)))
    return lam, phi


def moire_overlayer(a: float, period: float, phi_deg: float) -> tuple[float, float]:
    """
    Inverse of :func:`moire_period`: the overlayer constant b and twist θ that give a moiré of
    ``period`` rotated by ``phi_deg`` (relative to the substrate) on a substrate of constant a.
    In reciprocal space the moiré vector is the difference of the two lattices' vectors:
    G_b = G_a - G_m.
    """
    Ga = 1.0 / a
    Gm = 1.0 / period
    p = math.radians(phi_deg)
    gx, gy = Ga - Gm * math.cos(p), -Gm * math.sin(p)
    Gb = math.hypot(gx, gy)
    return 1.0 / Gb, math.degrees(math.atan2(gy, gx))
