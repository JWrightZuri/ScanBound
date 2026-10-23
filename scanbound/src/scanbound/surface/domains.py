"""
Domains: regions with a different lattice orientation, spacing or registry (translation).

    doms = find_domains(sites, image)              # rotation domains, then translational (antiphase) ones
    doms.domains[0].fit.describe()                 # each domain has its own lattice fit
    doms.label_map                                 # (rows, cols) int: domain id per pixel, -1 = none
    doms.boundaries                                # [(id_a, id_b, length)]

Method
------
1.  Every site gets a local orientation from its neighbour bonds, ψₙ = Σ exp(i n θ) with n the
    lattice's rotational symmetry (6 hexagonal, 4 square, 2 otherwise), and a local spacing (the
    median neighbour distance).
2.  Neighbouring sites whose orientations differ by less than ``angle_tol`` and spacings by less
    than ``spacing_tol`` are joined; connected groups of at least ``min_sites`` sites are rotation
    domains, and each gets its own lattice fit.
3.  Inside a rotation domain, sites that do not sit on the domain's lattice but form a connected,
    consistent group of their own (same fractional offset) are a translational domain: the same
    lattice shifted by a fraction of a cell (antiphase / registry domains).

Every pixel is then given the domain of its nearest site (within one lattice spacing), which gives
the label map, areas and boundary lengths.  Boundary lengths are measured on the pixel staircase and
multiplied by π/4, the average correction for boundaries of random direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.spatial import cKDTree

from .lattice import LatticeFit, SiteSet, fit_lattice, image_scale, reduce_basis


@dataclass
class Domain:
    id: int
    kind: str                            # "rotation" / "translation"
    sites: np.ndarray                    # indices into the SiteSet
    orientation: float                   # mean local bond orientation, degrees (mod 360/fold)
    spacing: float                       # median neighbour distance (physical)
    fit: Optional[LatticeFit] = None
    parent: Optional[int] = None         # translation domains: the rotation domain they sit in
    offset: Optional[tuple[float, float]] = None   # translation domains: fractional shift vs the parent lattice
    area: float = 0.0                    # physical
    pixels: int = 0

    def describe(self, unit: str = "") -> str:
        if self.kind == "translation":
            return (f"domain {self.id}: translational, {len(self.sites)} sites, shifted by "
                    f"({self.offset[0]:+.2f}, {self.offset[1]:+.2f}) cells from domain {self.parent}")  # type: ignore[index]
        txt = f"domain {self.id}: {len(self.sites)} sites, bonds at {self.orientation:.1f}°"
        if self.fit is not None:
            txt += " — " + self.fit.describe()
        return txt


@dataclass
class DomainResult:
    domains: list[Domain]
    label_map: np.ndarray
    boundaries: list[tuple[int, int, float]]
    orientation: np.ndarray              # per site, degrees
    order: np.ndarray                    # per site |ψₙ| (1 = perfect local symmetry)
    spacing: np.ndarray                  # per site
    fold: int
    unit: str
    params: dict = field(default_factory=dict)

    def site_domain(self) -> np.ndarray:
        out = np.full(len(self.orientation), -1)
        for d in self.domains:
            out[d.sites] = d.id
        return out

    def summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for d in self.domains:
            r: dict[str, Any] = {"domain": d.id, "kind": d.kind, "sites": len(d.sites), "orientation_deg": d.orientation,
                                 "spacing": d.spacing, "area": d.area, "unit": self.unit}
            if d.fit is not None:
                m = d.fit.measurements()
                for k in ("a1", "a2", "gamma", "a1_angle"):
                    r[k] = m[k].value
                    r[k + "_unc"] = m[k].uncertainty
            if d.offset is not None:
                r["offset_u"], r["offset_v"] = d.offset
                r["parent"] = d.parent
            rows.append(r)
        return rows


def local_order(P: np.ndarray, fold: int, shell: float = 1.35) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """(orientation deg, |ψ|, spacing, neighbour lists) of every point; orientation in [0, 360/fold)."""
    tree = cKDTree(P)
    k = min(len(P), 9)
    d, j = tree.query(P, k)
    nn = d[:, 1] if k > 1 else np.full(len(P), np.nan)
    orient = np.full(len(P), np.nan)
    order = np.zeros(len(P))
    spacing = np.full(len(P), np.nan)
    nbrs: list[list[int]] = []
    for i in range(len(P)):
        if not np.isfinite(nn[i]):
            nbrs.append([])
            continue
        sel = [int(jj) for dd, jj in zip(d[i, 1:], j[i, 1:]) if np.isfinite(dd) and dd <= shell * nn[i]]
        nbrs.append(sel)
        if not sel:
            continue
        v = P[sel] - P[i]
        th = np.arctan2(-v[:, 1], v[:, 0])
        psi = np.exp(1j * fold * th).mean()
        orient[i] = (math.degrees(np.angle(psi)) / fold) % (360.0 / fold)
        order[i] = abs(psi)
        spacing[i] = float(np.median(np.hypot(v[:, 0], v[:, 1])))
    return orient, order, spacing, nbrs


def guess_fold(P: np.ndarray) -> int:
    """6 or 4 when most sites have that local bond symmetry (median |ψₙ| > 0.6), else 2."""
    best, fold = 0.6, 2
    for n in (6, 4):
        _, order, _, _ = local_order(P, n)
        med = float(np.median(order)) if len(order) else 0.0
        if med > best:
            best, fold = med, n
    return fold


def _components(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def _angdiff(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:
    return np.abs((a - b + period / 2) % period - period / 2)


def find_domains(sites: SiteSet, image: Any = None, *, fold: Optional[int] = None, angle_tol: float = 4.0,
                 spacing_tol: float = 0.06, min_sites: int = 25, translational: bool = True,
                 scale: Optional[tuple[float, float]] = None, shape: Optional[tuple[int, int]] = None,
                 fit_domains: bool = True) -> DomainResult:
    """Rotation and translational domains of a site set.  ``fold``: lattice symmetry (found from the
    sites when None)."""
    if scale is None:
        scale = image_scale(image)[0]
    unit = image_scale(image)[1] if image is not None else "px"
    if shape is None:
        shape = tuple(np.asarray(getattr(image, "data", image)).shape) if image is not None else (
            int(sites.xy[:, 1].max()) + 1, int(sites.xy[:, 0].max()) + 1)
    P = sites.xy * np.asarray(scale)
    if fold is None:
        fold = guess_fold(P)
    period = 360.0 / fold
    orient, order, spacing, nbrs = local_order(P, fold)
    edges = []
    for i, js in enumerate(nbrs):
        for j in js:
            if j <= i:
                continue
            if not (np.isfinite(orient[i]) and np.isfinite(orient[j])):
                continue
            if _angdiff(np.array(orient[i]), np.array(orient[j]), period) > angle_tol:
                continue
            if abs(spacing[i] - spacing[j]) > spacing_tol * 0.5 * (spacing[i] + spacing[j]):
                continue
            edges.append((i, j))
    comp = _components(len(P), edges)
    groups = [np.nonzero(comp == c)[0] for c in np.unique(comp)]
    groups = sorted([g for g in groups if len(g) >= min_sites], key=len, reverse=True)
    domains: list[Domain] = []
    for g in groups:
        # circular mean of the orientations in the group
        z = np.exp(1j * np.radians(orient[g] * fold)).mean()
        dom = Domain(len(domains), "rotation", g, (math.degrees(np.angle(z)) / fold) % period,
                     float(np.nanmedian(spacing[g])))
        if fit_domains:
            try:
                dom.fit = fit_lattice(sites.subset(np.isin(np.arange(len(P)), g)), image, scale=scale,
                                      name=f"domain {dom.id}")
                # the fit's site arrays are for the subset: keep the mapping
                dom.fit.params_used["site_index"] = g.tolist()
            except ValueError:
                dom.fit = None
        domains.append(dom)
    # translational domains inside each rotation domain: group the sites by their fractional offset
    # from the domain's lattice (global rounding, not the step-by-step indexing, which can creep
    # across a boundary through intermediate sites)
    if translational:
        for parent in list(domains):
            f = parent.fit
            if f is None:
                continue
            members = np.asarray(parent.sites)
            uv = f.to_fractional(sites.xy[members])
            ph = uv - np.round(uv)
            Af = np.column_stack([f.a1, f.a2])
            amin = min(np.linalg.norm(f.a1), np.linalg.norm(f.a2))
            pos = {int(e): k for k, e in enumerate(members)}
            e2 = []
            for k, i in enumerate(members):
                for j in nbrs[i]:
                    kk = pos.get(j)
                    if kk is None or kk <= k:
                        continue
                    dph = ph[k] - ph[kk]
                    dph -= np.round(dph)
                    if np.linalg.norm(Af @ dph) < 0.15 * amin:
                        e2.append((k, kk))
            cc = _components(len(members), e2)
            parts = sorted((np.nonzero(cc == c)[0] for c in np.unique(cc)), key=len, reverse=True)
            parts = [p for p in parts if len(p) >= min_sites]
            if len(parts) < 2:
                continue
            ref = parts[0]
            z0 = np.exp(2j * np.pi * ph[ref]).mean(axis=0)
            for sel in parts[1:]:
                z = np.exp(2j * np.pi * ph[sel]).mean(axis=0)
                off = np.angle(z / z0) / (2 * np.pi)
                g = members[sel]
                parent.sites = np.setdiff1d(parent.sites, g)
                domains.append(Domain(len(domains), "translation", g, parent.orientation, parent.spacing,
                                      parent=parent.id, offset=(float(off[0]), float(off[1]))))
            if fit_domains:
                try:
                    keep = np.isin(np.arange(len(P)), parent.sites)
                    parent.fit = fit_lattice(sites.subset(keep), image, scale=scale, name=f"domain {parent.id}")
                    parent.fit.params_used["site_index"] = np.nonzero(keep)[0].tolist()
                except ValueError:
                    pass
            # offsets against the parent's own (re-fitted) lattice
            if parent.fit is not None:
                zp = np.exp(2j * np.pi * parent.fit.to_fractional(sites.xy[parent.sites])).mean(axis=0)
                for d in domains:
                    if d.parent == parent.id:
                        zd = np.exp(2j * np.pi * parent.fit.to_fractional(sites.xy[d.sites])).mean(axis=0)
                        off = np.angle(zd / zp) / (2 * np.pi)
                        d.offset = (float(off[0]), float(off[1]))
    # label map: nearest site within one spacing
    rows, cols = shape
    label = np.full((rows, cols), -1, int)
    site_dom = np.full(len(P), -1)
    for d in domains:
        site_dom[d.sites] = d.id
    if len(P):
        tree = cKDTree(P)
        gy, gx = np.mgrid[0:rows, 0:cols]
        G = np.column_stack([(gx.ravel() + 0.5) * scale[0], (gy.ravel() + 0.5) * scale[1]])
        dist, near = tree.query(G)
        sp = np.nanmedian(spacing) if np.isfinite(spacing).any() else np.inf
        lab = np.where(dist <= sp, site_dom[near], -1)
        label = lab.reshape(rows, cols)
    pix_area = scale[0] * scale[1]
    for d in domains:
        d.pixels = int((label == d.id).sum())
        d.area = d.pixels * pix_area
    # boundaries between labelled domains: Cauchy-Crofton estimate from transitions along rows,
    # columns and both diagonals (unbiased for any boundary direction, ~5 % low for straight ones)
    bounds: dict[tuple[int, int], float] = {}
    dx, dy = scale
    dd = dx * dy / math.hypot(dx, dy)
    for a, b, spacing_ in ((label[:, :-1], label[:, 1:], dy), (label[:-1, :], label[1:, :], dx),
                           (label[:-1, :-1], label[1:, 1:], dd), (label[:-1, 1:], label[1:, :-1], dd)):
        diff = (a != b) & (a >= 0) & (b >= 0)
        for x, y in zip(a[diff], b[diff]):
            key = (int(min(x, y)), int(max(x, y)))
            bounds[key] = bounds.get(key, 0.0) + spacing_
    boundaries = [(k[0], k[1], v * math.pi / 8) for k, v in sorted(bounds.items())]
    return DomainResult(domains, label, boundaries, orient, order, spacing, fold, unit,
                        {"angle_tol": angle_tol, "spacing_tol": spacing_tol, "min_sites": min_sites, "fold": fold})


def rotate_basis(a1: np.ndarray, a2: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Both vectors (y down) turned counter-clockwise on screen."""
    t = math.radians(degrees)
    R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    return reduce_basis(R @ a1, R @ a2)
