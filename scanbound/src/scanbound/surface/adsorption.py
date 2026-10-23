"""
Adsorption geometry: every molecule (or adatom) in the coordinate system of the substrate lattice.

    res = adsorption_registry(objects, fit, alignment=al, context=ctx)
    res.rows                         # one dict per object: lattice coordinates, candidate site, offsets, angles ...
    res.occupancy()                  # counts per site class (plain and confidence-weighted)
    res.heatmap(bins=48)             # density of positions inside the unit cell
    res.clusters                     # data-driven preferred positions in the cell

Lattice coordinates
-------------------
r = r0 + u a1 + v a2.  (u, v) say where the object sits in lattice units; (u mod 1, v mod 1) where
it sits *inside* its cell, which makes objects in different cells comparable.  Their uncertainty
combines the object's placement uncertainty and the lattice fit's (origin and vectors, growing
with the distance from the origin).

Site assignment
---------------
Each candidate site (from a surface model, or by default the image maxima, the bridges between
them and the hollows) is scored by a Gaussian likelihood of the distance between the object centre
and the site, with σ² = (position uncertainty)² + (site tolerance)².  The site tolerance (default
0.12 lattice constants) stands for everything that is not in the statistical uncertainty: model
registration, a molecular centre that is not the anchoring atom, tip convolution.  A "none of these"
alternative with the likelihood of a 2.5σ miss catches objects between candidate sites.  The
normalised scores are reported as *assignment confidence* with the alternatives: they rank
candidates under these assumptions; they are not proof of a bonding configuration.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .lattice import LatticeFit, screen_angle
from .objects import SurfaceObject, distance_to_polylines, nearest
from .statistics import compare_groups, relative_orientation, symmetry_period
from .structure import Alignment, Atom, SiteDef, SurfaceModel


def default_alignment(fit: LatticeFit) -> Alignment:
    """Treat the image's lattice maxima as the atoms: top = on a maximum, bridge, hollow."""
    model = SurfaceModel("image maxima", float(np.linalg.norm(fit.a1)), float(np.linalg.norm(fit.a2)),
                         float(math.degrees(math.acos(np.clip(np.dot(fit.a1, fit.a2) /
                                                              (np.linalg.norm(fit.a1) * np.linalg.norm(fit.a2)), -1, 1)))),
                         [Atom("max", 0.0, 0.0, 0)],
                         notes="sites relative to the image maxima; whether a maximum is an atom depends on the "
                               "surface, bias and tip")
    renamed = []
    for s in model.sites:
        name = {"top": "on maximum"}.get(s.name, s.name)
        renamed.append(SiteDef(name, s.u, s.v, s.kind))
    model.sites = renamed
    return Alignment(model, fit, np.eye(2), np.zeros(2))


@dataclass
class SurfaceContext:
    """Other things on the surface an object's environment is measured against (pixels)."""

    steps: list[np.ndarray] = field(default_factory=list)          # step-edge polylines
    defects: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    defect_labels: list[str] = field(default_factory=list)
    domain_map: Optional[np.ndarray] = None                        # int label image (-1 none)
    terrace_map: Optional[np.ndarray] = None                       # int label image (-1 none)

    def region(self, which: str, xy: np.ndarray) -> np.ndarray:
        m = self.domain_map if which == "domain" else self.terrace_map
        if m is None:
            return np.full(len(xy), -1)
        rows, cols = m.shape
        ix = np.clip(np.floor(xy[:, 0]).astype(int), 0, cols - 1)
        iy = np.clip(np.floor(xy[:, 1]).astype(int), 0, rows - 1)
        return m[iy, ix]


@dataclass
class AdsorptionResult:
    rows: list[dict[str, Any]]
    sites: list[tuple[SiteDef, np.ndarray]]
    fit: LatticeFit
    alignment: Alignment
    uv_cell: np.ndarray                 # (N, 2)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def occupancy(self, by: Optional[str] = None, min_confidence: float = 0.0) -> dict[str, Any]:
        """Counts per assigned site (``by``: split by a row field such as "domain", "terrace", "label")."""
        def tally(rows: list[dict]) -> dict[str, Any]:
            plain = Counter(r["site"] for r in rows if r["site_p"] >= min_confidence)
            weighted: Counter = Counter()
            for r in rows:
                for name, p in r["site_probabilities"].items():
                    weighted[name] += p
            return {"counts": dict(plain), "weighted": dict(weighted), "n": len(rows),
                    "uncertain": sum(1 for r in rows if r["site_p"] < 0.6)}
        if by is None:
            return tally(self.rows)
        groups: dict[str, list[dict]] = {}
        for r in self.rows:
            groups.setdefault(str(r.get(by)), []).append(r)
        return {g: tally(rs) for g, rs in groups.items()}

    def compare(self, by: str = "domain") -> Any:
        occ = self.occupancy(by)
        return compare_groups({g: Counter(v["counts"]) for g, v in occ.items()})

    def heatmap(self, bins: int = 40, bandwidth: float = 0.04) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u edges, v edges, density) of the positions in the cell, smoothed with a periodic Gaussian of
        ``bandwidth`` cell units; density is relative to a uniform distribution (1 = random)."""
        from scipy import ndimage
        H, eu, ev = np.histogram2d(self.uv_cell[:, 0], self.uv_cell[:, 1], bins=bins, range=[[0, 1], [0, 1]])
        if bandwidth > 0:
            H = ndimage.gaussian_filter(H, bandwidth * bins, mode="wrap")
        n = max(1, len(self.uv_cell))
        return eu, ev, H / (n / bins ** 2)

    def uniformity_p(self, bins: int = 0) -> float:
        """Chi-square test of the in-cell positions against a uniform distribution."""
        from scipy.stats import chisquare
        n = len(self.uv_cell)
        if n < 10:
            return math.nan
        b = bins or max(2, min(8, int(math.sqrt(n / 5))))
        H, _, _ = np.histogram2d(self.uv_cell[:, 0], self.uv_cell[:, 1], bins=b, range=[[0, 1], [0, 1]])
        return float(chisquare(H.ravel()).pvalue)


def _min_image(du: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Shortest physical vector among the periodic images of fractional differences du (N, 2)."""
    du = (du + 0.5) % 1.0 - 0.5
    best = du @ A.T
    bestn = np.hypot(best[:, 0], best[:, 1])
    for s in ((1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)):
        v = (du + s) @ A.T
        n = np.hypot(v[:, 0], v[:, 1])
        better = n < bestn
        best[better], bestn[better] = v[better], n[better]
    return best


def _clusters(uv: np.ndarray, A: np.ndarray, sites: list[tuple[SiteDef, np.ndarray]], bandwidth: float = 0.05,
              min_count: int = 3) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Peaks of the periodic density of in-cell positions (mean shift on the torus)."""
    n = len(uv)
    if n < min_count:
        return [], np.full(n, -1)
    pts = uv.copy()
    for _ in range(30):
        new = pts.copy()
        for i in range(n):
            d = (uv - pts[i] + 0.5) % 1.0 - 0.5
            w = np.exp(-0.5 * (np.hypot(*(d @ A.T).T) / (bandwidth * np.linalg.norm(A[:, 0]))) ** 2)
            new[i] = (pts[i] + (w[:, None] * d).sum(0) / w.sum()) % 1.0
        if np.abs((new - pts + 0.5) % 1 - 0.5).max() < 1e-5:
            pts = new
            break
        pts = new
    centres: list[np.ndarray] = []
    lab = np.full(n, -1)
    tol = 2 * bandwidth
    for i in range(n):
        for k, c in enumerate(centres):
            if np.hypot(*(((pts[i] - c + 0.5) % 1 - 0.5) @ A.T)) < tol * np.linalg.norm(A[:, 0]):
                lab[i] = k
                break
        else:
            centres.append(pts[i])
            lab[i] = len(centres) - 1
    out, remap = [], {}
    for k, c in enumerate(centres):
        mem = np.nonzero(lab == k)[0]
        if len(mem) < min_count:
            continue
        d = _min_image(uv[mem] - c, A)
        spread = float(np.sqrt((d ** 2).sum(1).mean()))
        near, nd = None, math.inf
        for s, p in sites:
            dd = float(np.hypot(*_min_image((c - p)[None], A)[0]))
            if dd < nd:
                near, nd = s.name, dd
        remap[k] = len(out)
        out.append({"cluster": len(out), "u": float(c[0]), "v": float(c[1]), "count": int(len(mem)),
                    "fraction": len(mem) / n, "spread": spread, "nearest_site": near, "site_distance": nd})
    lab = np.array([remap.get(int(k), -1) for k in lab])
    return out, lab


def adsorption_registry(objects: Sequence[SurfaceObject], fit: LatticeFit, alignment: Optional[Alignment] = None,
                        context: Optional[SurfaceContext] = None, *, site_tolerance: float = 0.12,
                        direction: tuple[float, float] = (1, 0), mol_fold: int = 1, sub_fold: int = 1,
                        mirror: bool = False, cluster_bandwidth: float = 0.05) -> AdsorptionResult:
    """
    Register objects to the lattice: lattice coordinates, candidate adsorption sites, offsets,
    orientation relative to the crystal direction ``direction`` = [h k] (in the alignment's model
    cell when a model is aligned, else in the fitted cell), and distances to steps, defects and
    the nearest other object.
    """
    al = alignment or default_alignment(fit)
    ctx = context or SurfaceContext()
    A = fit.A
    amin = float(min(np.linalg.norm(fit.a1), np.linalg.norm(fit.a2)))
    sc = np.asarray(fit.scale, float)
    sites = al.site_fractional()
    xy = np.array([o.xy for o in objects], float).reshape(-1, 2)
    uv = fit.to_fractional(xy) if len(xy) else np.zeros((0, 2))
    cell = np.floor(uv)
    uvc = uv - cell
    dir_vec = al.direction(*direction) if alignment is not None else fit.direction(*direction)
    dir_angle = screen_angle(dir_vec)
    # direction uncertainty from the fit (numerical, through the parameter covariance)
    p0 = np.concatenate([fit.origin, fit.a1, fit.a2])
    T = al.T if alignment is not None else np.eye(2)
    hk = T @ np.array(direction, float)

    def ang(p: np.ndarray) -> float:
        return screen_angle(hk[0] * p[2:4] + hk[1] * p[4:6])
    J = np.zeros(6)
    for i in range(6):
        h = 1e-7 * max(1e-12, np.abs(p0[2:]).max())
        q = p0.copy()
        q[i] += h
        J[i] = ((ang(q) - ang(p0) + 180) % 360 - 180) / h
    dir_sigma = math.sqrt(max(0.0, float(J @ fit.cov @ J)))
    Ainv = np.linalg.inv(A)
    P = xy * sc
    d_step, step_i = distance_to_polylines(P, [np.asarray(s) * sc for s in ctx.steps]) if ctx.steps else (
        np.full(len(xy), np.inf), np.full(len(xy), -1))
    d_def, def_i = nearest(P, ctx.defects * sc) if len(ctx.defects) else (np.full(len(xy), np.inf),
                                                                            np.full(len(xy), -1))
    d_nb, nb_i = nearest(P, P, exclude_self=True)
    dom = ctx.region("domain", xy)
    ter = ctx.region("terrace", xy)
    clusters, clab = _clusters(uvc, A, sites, cluster_bandwidth) if len(uvc) else ([], np.zeros(0, int))
    period = symmetry_period(mol_fold, sub_fold)
    rows = []
    for i, o in enumerate(objects):
        # position covariance (physical) -> lattice coordinates
        S_pos = np.diag((o.sigma_px * sc) ** 2)
        Jp = np.hstack([np.eye(2), uv[i, 0] * np.eye(2), uv[i, 1] * np.eye(2)])
        S_lat = Jp @ fit.cov @ Jp.T
        S = S_pos + S_lat
        S_uv = Ainv @ S @ Ainv.T
        sig_r = math.sqrt(max(np.trace(S) / 2, 0.0))
        sig_tot = math.hypot(sig_r, site_tolerance * amin)
        # candidate sites
        cand = []
        for s, p in sites:
            v = _min_image((uvc[i] - p)[None], A)[0]
            cand.append((s, float(np.hypot(*v)), v))
        like = np.array([math.exp(-0.5 * (d / sig_tot) ** 2) for _, d, _ in cand] + [math.exp(-0.5 * 2.5 ** 2)])
        post = like / like.sum()
        order = np.argsort(-post)
        probs: dict[str, float] = {}
        for k, (s, d, v) in enumerate(cand):
            probs[s.name] = probs.get(s.name, 0.0) + float(post[k])
        probs["none"] = float(post[-1])
        best_k = int(order[0])
        if best_k == len(cand):
            site_name, kind, d_site, off = "none", "", math.nan, np.array([math.nan, math.nan])
        else:
            s, d_site, off = cand[best_k]
            site_name, kind = s.name, s.kind
        alts = sorted(((n, p) for n, p in probs.items() if n != site_name), key=lambda t: -t[1])[:3]
        row: dict[str, Any] = {
            "id": o.id, "label": o.label, "status": o.status, "x_px": o.x, "y_px": o.y,
            "x": float(P[i, 0]), "y": float(P[i, 1]), "u": float(uv[i, 0]), "v": float(uv[i, 1]),
            "cell_m": int(cell[i, 0]), "cell_n": int(cell[i, 1]), "u_cell": float(uvc[i, 0]), "v_cell": float(uvc[i, 1]),
            "sigma_u": float(math.sqrt(max(S_uv[0, 0], 0))), "sigma_v": float(math.sqrt(max(S_uv[1, 1], 0))),
            "sigma_position": sig_r, "site": site_name, "site_kind": kind, "site_distance": d_site,
            "site_offset_x": float(off[0]), "site_offset_y": float(-off[1]),        # y up
            "site_p": float(probs.get(site_name, 0.0)), "site_probabilities": probs,
            "alternatives": "; ".join(f"{n} {p:.2f}" for n, p in alts),
            "position_cluster": int(clab[i]) if len(clab) else -1,
            "domain": int(dom[i]), "terrace": int(ter[i]),
            "d_step": float(d_step[i]), "d_defect": float(d_def[i]),
            "nearest_defect_label": (ctx.defect_labels[int(def_i[i])] if 0 <= int(def_i[i]) < len(ctx.defect_labels)
                                     else ""),
            "d_neighbor": float(d_nb[i]), "neighbor": objects[int(nb_i[i])].id if nb_i[i] >= 0 else "",
            "handedness": o.handedness or "", "confidence": o.confidence,
        }
        if o.orientation is not None:
            rel = float(relative_orientation(o.orientation, dir_angle, mol_fold, sub_fold, mirror))
            row["orientation_abs"] = o.orientation % 360.0
            row["orientation_rel"] = rel
            row["orientation_rel_sigma"] = (math.hypot(o.orientation_sigma, dir_sigma) if o.orientation_sigma is not None
                                            else None)
        rows.append(row)
    return AdsorptionResult(rows, sites, fit, al, uvc, clusters,
                            {"site_tolerance": site_tolerance, "direction": list(direction), "direction_angle": dir_angle,
                             "direction_sigma": dir_sigma, "mol_fold": mol_fold, "sub_fold": sub_fold, "mirror": mirror,
                             "period": period, "model": al.model.name})
