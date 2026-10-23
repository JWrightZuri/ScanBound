"""
Point defects measured against a fitted lattice: vacancies, adatoms, extra or unusual sites,
defect clusters and rows, and their local environments.

    res = find_defects(fit, image)
    res.defects                      # list of Defect: kind, position, lattice index, confidence, environment ...
    res.complexes()                  # recurring local environments (same code = same arrangement up to symmetry)
    res.density(area)                # per unit area
    anns = res.to_annotations(image_id)   # proposed point features to review in the app

What each kind means
--------------------
``vacancy``        a lattice position inside the analysed area with no detected site; its
                   confidence is how dark it is compared with the occupied sites (1 = as dark as a
                   hollow, 0 = as bright as an atom).
``adatom``         a detected site that is not on the lattice and is brighter than the typical site.
``extra site``     a detected site not on the lattice and not brighter (noise, a dim adsorbate,
                   a site of another domain); confidence from its detection prominence.
``unusual site``   a lattice site whose brightness or displacement is far outside the population
                   (robust z-score above ``unusual_z``): an substitutional atom, a subsurface defect, a
                   tip artefact ... - a candidate to look at, nothing more.

Every defect is a proposal: nothing here knows chemistry.  Vacancies near the image edge or next to
large adsorbates are especially likely to be artefacts of site detection; the margin and the
confidence are there to keep them out of statistics.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..annotations import Annotation, Origin, Point
from .lattice import LATTICE, LatticeFit, _data, _filled, symmetry_fold


@dataclass
class Defect:
    kind: str
    xy: np.ndarray                       # pixel
    index: tuple[int, int]               # nearest lattice index (m, n)
    confidence: float
    score: float                         # site prominence (detected) or darkness (vacancies)
    environment: str = ""                # canonical occupancy code of the first neighbour shell
    occupied_neighbors: int = 0
    shell_size: int = 0
    cluster: int = -1
    cluster_size: int = 1
    site: int = -1                       # index into fit.sites (-1 for vacancies)
    note: str = ""


@dataclass
class DefectResult:
    defects: list[Defect]
    fit: LatticeFit
    shell: list[tuple[int, int]]
    lines: list[dict[str, Any]] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    analysed_sites: int = 0               # lattice positions inside the analysed area

    def counts(self) -> Counter:
        return Counter(d.kind for d in self.defects)

    def complexes(self, kinds: tuple[str, ...] = ("vacancy",)) -> list[dict[str, Any]]:
        """Clusters of defects grouped by their composition and shape (lattice-index pattern up to the
        lattice's rotations)."""
        groups: dict[int, list[Defect]] = {}
        for d in self.defects:
            if d.kind in kinds and d.cluster >= 0:
                groups.setdefault(d.cluster, []).append(d)
        sym = _rotations(self.fit)
        out: Counter = Counter()
        examples: dict[str, int] = {}
        for cid, ds in groups.items():
            idx = np.array([d.index for d in ds], float)
            code = _shape_code(idx, sym)
            key = f"{len(ds)} × {'/'.join(sorted({d.kind for d in ds}))}: {code}"
            out[key] += 1
            examples.setdefault(key, cid)
        return [{"complex": k, "count": v, "example_cluster": examples[k]} for k, v in out.most_common()]

    def environments(self) -> list[dict[str, Any]]:
        out: Counter = Counter((d.kind, d.environment) for d in self.defects if d.environment)
        return [{"kind": k, "environment": e, "count": c} for (k, e), c in out.most_common()]

    def density(self, area: float, kind: Optional[str] = None) -> float:
        n = sum(1 for d in self.defects if kind is None or d.kind == kind)
        return n / area if area > 0 else math.nan

    def concentration(self, kind: str = "vacancy") -> float:
        """Defects per analysed lattice site."""
        n = sum(1 for d in self.defects if d.kind == kind)
        return n / self.analysed_sites if self.analysed_sites else math.nan

    def nearest_neighbor_distances(self, kind: Optional[str] = None) -> np.ndarray:
        xy = np.array([d.xy for d in self.defects if kind is None or d.kind == kind]).reshape(-1, 2)
        if len(xy) < 2:
            return np.zeros(0)
        P = xy * np.asarray(self.fit.scale)
        d, _ = cKDTree(P).query(P, 2)
        return d[:, 1]

    def rows(self) -> list[dict[str, Any]]:
        sc = np.asarray(self.fit.scale)
        return [{"kind": d.kind, "x_px": float(d.xy[0]), "y_px": float(d.xy[1]), "x": float(d.xy[0] * sc[0]),
                 "y": float(d.xy[1] * sc[1]), "m": d.index[0], "n": d.index[1], "confidence": d.confidence,
                 "score": d.score, "environment": d.environment, "occupied_neighbors": d.occupied_neighbors,
                 "shell": d.shell_size, "cluster": d.cluster, "cluster_size": d.cluster_size, "note": d.note}
                for d in self.defects]

    def to_annotations(self, image_id: str, run: str = "", min_confidence: float = 0.0,
                       status: str = "proposed") -> list[Annotation]:
        out = []
        for d in self.defects:
            if d.confidence < min_confidence:
                continue
            attrs = {"surface": "defect", "lattice_index": list(d.index), "environment": d.environment,
                     "cluster": d.cluster, "cluster_size": d.cluster_size}
            out.append(Annotation(image_id, "feature", Point(float(d.xy[0]), float(d.xy[1])), label=d.kind,
                                  status=status, origin=Origin("detector", "lattice defect finder", run=run,
                                                               confidence=float(round(d.confidence, 3)),
                                                               details={"score": round(float(d.score), 3)}),
                                  attrs=attrs, note=d.note))
        return out


def _rotations(fit: LatticeFit) -> list[np.ndarray]:
    """Integer matrices of the lattice's point-group rotations, in the fitted basis."""
    fold = symmetry_fold(fit)
    A = fit.A
    out = []
    for k in range(fold):
        t = math.radians(360.0 * k / fold)
        R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
        M = np.linalg.solve(A, R @ A)
        out.append(np.round(M))
    return out


def _shape_code(idx: np.ndarray, rots: list[np.ndarray]) -> str:
    best = None
    for R in rots:
        p = idx @ R.T
        p = p - p.min(axis=0)
        key = tuple(sorted(map(tuple, p.astype(int).tolist())))
        if best is None or key < best:
            best = key
    return " ".join(f"({a},{b})" for a, b in best or ())


def first_shell(fit: LatticeFit) -> list[tuple[int, int]]:
    """Index offsets of the nearest-neighbour shell (4 or 6 sites, or 2 + 2 for oblique lattices)."""
    cands = [(m, n) for m in range(-2, 3) for n in range(-2, 3) if (m, n) != (0, 0)]
    L = {c: float(np.linalg.norm(fit.direction(*c))) for c in cands}
    d0 = min(L.values())
    shell = [c for c in cands if L[c] < 1.08 * d0]
    if len(shell) < 4:
        d1 = sorted(set(round(v / d0, 3) for v in L.values()))[1] * d0
        shell = [c for c in cands if L[c] < 1.02 * d1]
    ang = {c: math.atan2(-fit.direction(*c)[1], fit.direction(*c)[0]) % (2 * math.pi) for c in shell}
    return sorted(shell, key=lambda c: ang[c])


def _env_code(bits: list[str]) -> str:
    """Rotation-canonical code of a ring of neighbour states ("1" occupied, "0" empty, "?" unknown)."""
    n = len(bits)
    return "".join(min(tuple(bits[(i + k) % n] for i in range(n)) for k in range(n))) if n else ""


def find_defects(fit: LatticeFit, image: Any = None, *, margin: float = 1.0, vacancy_tol: float = 0.35,
                 min_confidence: float = 0.25, cluster_distance: float = 1.6, unusual_z: float = 5.0,
                 region: Optional[np.ndarray] = None, exclude: Optional[np.ndarray] = None) -> DefectResult:
    """
    Compare the fitted lattice with the detected sites.

    ``margin``: lattice positions closer than this many lattice constants to the image edge (or the
    fit region's edge) are not analysed.  ``vacancy_tol``: a lattice position counts as occupied
    when any detected site lies within this fraction of the lattice constant.  ``min_confidence``:
    vacancies darker than this (0..1, see module doc) are reported.  ``cluster_distance``: defects
    closer than this many lattice constants form a cluster.  ``exclude``: bool mask (True = do not
    analyse, e.g. under molecules).
    """
    amin = float(min(np.linalg.norm(fit.a1), np.linalg.norm(fit.a2)))
    sc = np.asarray(fit.scale, float)
    shape = fit.shape if image is None else tuple(_data(image).shape)
    rows, cols = shape
    margin_px = margin * amin / min(sc)
    idx_pred, xy_pred = fit.predicted(shape, margin=margin_px, region=region)
    if exclude is not None:
        ix = np.clip(xy_pred[:, 0].astype(int), 0, cols - 1)
        iy = np.clip(xy_pred[:, 1].astype(int), 0, rows - 1)
        keep = ~np.asarray(exclude, bool)[iy, ix]
        idx_pred, xy_pred = idx_pred[keep], xy_pred[keep]
    sites_xy = fit.sites.xy
    on = fit.status == LATTICE
    occupied = {(int(m), int(n)) for (m, n), s in zip(fit.indices, on) if s}
    tree = cKDTree(sites_xy * sc) if len(sites_xy) else None
    # intensities for vacancy darkness
    z = None
    if image is not None:
        a = _filled(_data(image))
        z = ndimage.gaussian_filter(a, max(0.5, 0.12 * amin / min(sc)))

    def sample(xy: np.ndarray) -> np.ndarray:
        if z is None:
            return np.full(len(xy), np.nan)
        return ndimage.map_coordinates(z, [xy[:, 1] - 0.5, xy[:, 0] - 0.5], order=1, mode="nearest")

    occ_xy = sites_xy[on]
    I_occ = sample(occ_xy) if len(occ_xy) else np.zeros(0)
    # hollow-ish reference: cell centres
    I_hol = sample(fit.to_pixel(idx_pred + 0.5)) if len(idx_pred) else np.zeros(0)
    m_occ = float(np.nanmedian(I_occ)) if len(I_occ) else math.nan
    m_hol = float(np.nanmedian(I_hol)) if len(I_hol) else math.nan
    contrast = m_occ - m_hol if np.isfinite(m_occ) and np.isfinite(m_hol) else math.nan
    defects: list[Defect] = []
    for k, (ij, xy) in enumerate(zip(idx_pred, xy_pred)):
        key = (int(ij[0]), int(ij[1]))
        if key in occupied:
            continue
        if tree is not None and tree.query_ball_point(xy * sc, vacancy_tol * amin):
            continue
        I = float(sample(xy[None])[0])
        conf = float(np.clip((m_occ - I) / contrast, 0, 1)) if np.isfinite(contrast) and contrast > 0 else 0.5
        if conf < min_confidence:
            continue
        defects.append(Defect("vacancy", xy.copy(), key, conf, conf))
    # extra sites
    med_score = float(np.median(fit.sites.score[on])) if on.any() else 0.0
    inside = np.ones(len(sites_xy), bool)
    inside &= (sites_xy[:, 0] >= margin_px) & (sites_xy[:, 0] <= cols - margin_px)
    inside &= (sites_xy[:, 1] >= margin_px) & (sites_xy[:, 1] <= rows - margin_px)
    poly = region if region is not None else fit.region
    if poly is not None and len(poly) >= 3:
        from matplotlib.path import Path
        inside &= Path(np.asarray(poly, float)).contains_points(sites_xy)
    if exclude is not None:
        ix = np.clip(sites_xy[:, 0].astype(int), 0, cols - 1)
        iy = np.clip(sites_xy[:, 1].astype(int), 0, rows - 1)
        inside &= ~np.asarray(exclude, bool)[iy, ix]
    for i in np.nonzero((~on) & inside)[0]:
        score = float(fit.sites.score[i])
        bright = score > 1.3 * med_score
        kind = "adatom" if bright else "extra site"
        conf = float(np.clip(score / (2.0 * max(med_score, 1e-9)), 0, 1)) if bright else \
            float(np.clip(score / max(med_score, 1e-9), 0, 1))
        res = float(np.hypot(*fit.residual[i])) / amin
        note = f"{res:.2f} lattice constants from the nearest lattice position"
        defects.append(Defect(kind, sites_xy[i].copy(), (int(fit.indices[i, 0]), int(fit.indices[i, 1])), conf, score,
                              site=int(i), note=note))
    # unusual lattice sites: robust z of brightness and displacement
    if on.any():
        disp = np.hypot(*fit.residual[on].T) / amin
        feats = [fit.sites.score[on]]
        if len(I_occ):
            feats.append(I_occ)
        feats.append(disp)
        Z = []
        for f in feats:
            med = np.nanmedian(f)
            mad = 1.4826 * np.nanmedian(np.abs(f - med)) or (np.nanstd(f) or 1.0)
            Z.append(np.abs(f - med) / mad)
        zmax = np.nanmax(np.vstack(Z), axis=0)
        on_idx = np.nonzero(on)[0]
        for k in np.nonzero(zmax > unusual_z)[0]:
            i = on_idx[k]
            if not inside[i]:
                continue
            which = ["prominence", "brightness", "displacement"][int(np.nanargmax([Zf[k] for Zf in Z]))] if len(Z) == 3 \
                else "prominence"
            defects.append(Defect("unusual site", sites_xy[i].copy(), (int(fit.indices[i, 0]), int(fit.indices[i, 1])),
                                  float(min(1.0, zmax[k] / (2 * unusual_z))), float(zmax[k]), site=int(i),
                                  note=f"{which} {zmax[k]:.1f} robust standard deviations from the typical site"))
    # neighbour shell occupancy
    shell = first_shell(fit)
    analysed = {(int(a), int(b)) for a, b in idx_pred}
    vac = {d.index for d in defects if d.kind == "vacancy"}
    for d in defects:
        bits = []
        for dm, dn in shell:
            nb = (d.index[0] + dm, d.index[1] + dn)
            if nb in occupied and nb not in vac:
                bits.append("1")
            elif nb in vac:
                bits.append("0")
            else:
                bits.append("?")                  # outside the analysed area, or not resolved (e.g. obscured)
        d.environment = _env_code(bits)
        d.occupied_neighbors = bits.count("1")
        d.shell_size = len(shell)
    # clusters
    if defects:
        P = np.array([d.xy for d in defects]) * sc
        pairs = cKDTree(P).query_pairs(cluster_distance * amin)
        parent = list(range(len(defects)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a_, b_ in pairs:
            parent[find(a_)] = find(b_)
        roots = [find(i) for i in range(len(defects))]
        cnt = Counter(roots)
        ids: dict[int, int] = {}
        for i, d in enumerate(defects):
            if cnt[roots[i]] > 1:
                d.cluster = ids.setdefault(roots[i], len(ids))
                d.cluster_size = cnt[roots[i]]
    lines = _line_defects(defects, fit)
    return DefectResult(defects, fit, shell, lines,
                        {"margin": margin, "vacancy_tol": vacancy_tol, "min_confidence": min_confidence,
                         "cluster_distance": cluster_distance, "unusual_z": unusual_z,
                         "atom_contrast": contrast}, len(analysed))


def _line_defects(defects: list[Defect], fit: LatticeFit, min_members: int = 3) -> list[dict[str, Any]]:
    """Clusters whose members lie on one lattice row: rows of vacancies, adatom chains ..."""
    by: dict[int, list[Defect]] = {}
    for d in defects:
        if d.cluster >= 0:
            by.setdefault(d.cluster, []).append(d)
    out = []
    sc = np.asarray(fit.scale)
    for cid, ds in by.items():
        if len(ds) < min_members:
            continue
        P = np.array([d.xy for d in ds]) * sc
        c = P - P.mean(0)
        w, v = np.linalg.eigh(c.T @ c / len(P))
        if w[1] <= 0 or w[0] / w[1] > 0.05:
            continue
        direction = v[:, 1]
        # nearest low-index lattice direction
        best, bd = (1, 0), 180.0
        for h in range(-2, 3):
            for k in range(-2, 3):
                if (h, k) == (0, 0):
                    continue
                u = fit.direction(h, k)
                ang = math.degrees(math.acos(min(1.0, abs(float(u @ direction)) / np.linalg.norm(u))))
                if ang < bd - 1e-6:
                    best, bd = (h, k), ang
        length = float(np.ptp(c @ direction))
        out.append({"cluster": cid, "members": len(ds), "kinds": "/".join(sorted({d.kind for d in ds})),
                    "direction": f"[{best[0]} {best[1]}]", "angle_to_direction": bd, "length": length})
    return out


def crop(image: Any, xy: np.ndarray, half: int) -> np.ndarray:
    """A square crop (2 half + 1 pixels) around a pixel position, NaN-padded at the border."""
    a = np.asarray(getattr(image, "data", image), float)
    rows, cols = a.shape
    cx, cy = int(math.floor(xy[0])), int(math.floor(xy[1]))
    out = np.full((2 * half + 1, 2 * half + 1), np.nan)
    y0, y1 = max(0, cy - half), min(rows, cy + half + 1)
    x0, x1 = max(0, cx - half), min(cols, cx + half + 1)
    out[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)] = a[y0:y1, x0:x1]
    return out


def export_crops(image: Any, defects: list[Defect], folder: str, half: int = 16, prefix: str = "defect") -> list[str]:
    """One PNG per defect (contrast stretched per crop) plus ``index.csv``; returns the file names."""
    import csv
    import os
    from PIL import Image as PILImage
    os.makedirs(folder, exist_ok=True)
    names = []
    with open(os.path.join(folder, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "kind", "x_px", "y_px", "m", "n", "confidence", "environment", "cluster"])
        for i, d in enumerate(defects):
            c = crop(image, d.xy, half)
            lo, hi = np.nanpercentile(c, [1, 99]) if np.isfinite(c).any() else (0, 1)
            g = np.clip((np.nan_to_num(c, nan=lo) - lo) / max(hi - lo, 1e-30), 0, 1)
            name = f"{prefix}_{i:04d}_{d.kind.replace(' ', '_')}.png"
            PILImage.fromarray((g * 255).astype(np.uint8)).resize((4 * c.shape[1], 4 * c.shape[0]),
                                                                 PILImage.NEAREST).save(os.path.join(folder, name))
            w.writerow([name, d.kind, f"{d.xy[0]:.2f}", f"{d.xy[1]:.2f}", d.index[0], d.index[1],
                        f"{d.confidence:.3f}", d.environment, d.cluster])
            names.append(name)
    return names
