"""
Molecular assembly: neighbour graphs, pair statistics, order parameters, chains, motifs.

    g = neighbor_graph(objects, scale, cutoff=2.0e-9)       # edges between neighbouring molecules
    g.pairs                                                  # distance, relative angle, handedness, direction ...
    g.clusters()                                             # connected groups (cluster sizes)
    pcf = pair_correlation(objects, shape, scale, r_max=5e-9)       # g(r) with a random-arrangement envelope
    ce = clark_evans(objects, shape, scale)                  # clustered / random / regular, with p
    chains = find_chains(g)                                   # rows and chains of aligned neighbours
    hits = find_motif(template_ids, objects, scale)           # where else the same arrangement occurs

Normalisation and limits
------------------------
g(r) uses the translation edge correction of a rectangular window (the image), so pairs near the
border are not under-counted.  Its reference is complete spatial randomness at the same mean
density over the whole image: if molecules can only sit on part of it (one terrace, between
islands), g(r) is inflated at all r by the inverse of the occupied fraction - restrict the analysis
to that region or read relative peak heights.  A peak at r means pairs at that separation are more
frequent than in a random arrangement; it does not by itself establish a bond.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .objects import SurfaceObject


def _xy(objects: Sequence[SurfaceObject], scale: tuple[float, float]) -> np.ndarray:
    return np.array([o.xy for o in objects], float).reshape(-1, 2) * np.asarray(scale, float)


def _segments_cross(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    def orient(a, b, c):  # noqa: ANN001
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, p2, q1) != orient(p1, p2, q2)) and (orient(q1, q2, p1) != orient(q1, q2, p2))


@dataclass
class NeighborGraph:
    objects: list[SurfaceObject]
    edges: np.ndarray                    # (E, 2) object indices
    pairs: list[dict[str, Any]]
    scale: tuple[float, float]
    cutoff: float

    def neighbors(self, i: int) -> list[int]:
        out = [int(b) for a, b in self.edges if a == i] + [int(a) for a, b in self.edges if b == i]
        return sorted(set(out))

    def degree(self) -> np.ndarray:
        d = np.zeros(len(self.objects), int)
        for a, b in self.edges:
            d[a] += 1
            d[b] += 1
        return d

    def clusters(self) -> np.ndarray:
        """Connected-component id of every object."""
        parent = list(range(len(self.objects)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in self.edges:
            parent[find(int(a))] = find(int(b))
        roots = [find(i) for i in range(len(self.objects))]
        ids: dict[int, int] = {}
        return np.array([ids.setdefault(r, len(ids)) for r in roots])

    def cluster_sizes(self) -> Counter:
        c = Counter(self.clusters())
        return Counter(c.values())


def neighbor_graph(objects: Sequence[SurfaceObject], scale: tuple[float, float], cutoff: Optional[float] = None,
                   method: str = "delaunay", mol_fold: int = 1, lattice: Any = None,
                   steps: Sequence[np.ndarray] = (), defects: Optional[np.ndarray] = None,
                   defect_radius: Optional[float] = None) -> NeighborGraph:
    """
    Neighbour pairs: Delaunay edges no longer than ``cutoff`` (``method="delaunay"``), or all pairs
    within ``cutoff`` (``"cutoff"``).  Default cutoff: 1.5 × the median nearest-neighbour distance.
    For each pair: distance, direction of the connecting vector (screen angle, and relative to the
    lattice's a1 when ``lattice`` is given, with the displacement in lattice units), relative
    orientation (reduced by ``mol_fold``), relative handedness, whether a step edge crosses the
    connecting line, whether a defect lies within ``defect_radius`` of it.
    """
    objs = list(objects)
    P = _xy(objs, scale)
    n = len(P)
    if n < 2:
        return NeighborGraph(objs, np.zeros((0, 2), int), [], scale, cutoff or 0.0)
    tree = cKDTree(P)
    nn = tree.query(P, 2)[0][:, 1]
    if cutoff is None:
        cutoff = 1.5 * float(np.median(nn))
    if method == "delaunay" and n >= 3:
        try:
            tri = Delaunay(P)
            E = set()
            for s in tri.simplices:
                for k in range(3):
                    a, b = int(s[k]), int(s[(k + 1) % 3])
                    E.add((min(a, b), max(a, b)))
            edges = np.array([e for e in sorted(E) if np.hypot(*(P[e[0]] - P[e[1]])) <= cutoff], int).reshape(-1, 2)
        except Exception:  # noqa: BLE001 - degenerate (collinear) sets fall back to the cutoff
            edges = tree.query_pairs(cutoff, output_type="ndarray")
    else:
        edges = tree.query_pairs(cutoff, output_type="ndarray")
    sc = np.asarray(scale, float)
    steps_p = [np.asarray(s, float) * sc for s in steps]
    D = None if defects is None or len(defects) == 0 else np.asarray(defects, float).reshape(-1, 2) * sc
    rad = defect_radius if defect_radius is not None else 0.25 * cutoff
    period = 360.0 / max(1, mol_fold)
    pairs = []
    for a, b in edges:
        oa, ob = objs[a], objs[b]
        v = P[b] - P[a]
        d = float(np.hypot(*v))
        direction = math.degrees(math.atan2(-v[1], v[0])) % 180.0
        row: dict[str, Any] = {"a": oa.id, "b": ob.id, "i": int(a), "j": int(b), "distance": d,
                               "direction": direction}
        if oa.orientation is not None and ob.orientation is not None:
            dd = (ob.orientation - oa.orientation) % period
            row["relative_angle"] = min(dd, period - dd)
        if oa.handedness and ob.handedness:
            row["handedness"] = "same" if oa.handedness == ob.handedness else "opposite"
        if lattice is not None:
            duv = np.linalg.solve(lattice.A, v)
            row["du"], row["dv"] = float(duv[0]), float(duv[1])
            row["direction_rel"] = (direction - lattice.direction_angle(1, 0)) % 180.0
        crosses = False
        for L in steps_p:
            for q1, q2 in zip(L[:-1], L[1:]):
                if _segments_cross(P[a], P[b], q1, q2):
                    crosses = True
                    break
            if crosses:
                break
        row["step_between"] = crosses
        if D is not None:
            t = np.clip(((D - P[a]) @ v) / max(d * d, 1e-300), 0, 1)
            proj = P[a] + t[:, None] * v
            dist = np.hypot(*(D - proj).T)
            row["defect_between"] = bool(np.any((dist <= rad) & (t > 0) & (t < 1)))
        pairs.append(row)
    g = NeighborGraph(objs, edges, pairs, scale, float(cutoff))
    cl = g.clusters()
    sizes = Counter(cl)
    for row in pairs:
        row["cluster_size"] = sizes[cl[row["i"]]]
    return g


# ======================================================================== spatial statistics
@dataclass
class PairCorrelation:
    r: np.ndarray                        # bin centres (physical)
    g: np.ndarray
    lo: np.ndarray                       # random-arrangement envelope (2.5 %, 97.5 %)
    hi: np.ndarray
    n: int
    density: float
    peaks: list[float] = field(default_factory=list)

    def rows(self) -> list[dict[str, float]]:
        return [{"r": float(r), "g": float(g), "random_lo": float(a), "random_hi": float(b)}
                for r, g, a, b in zip(self.r, self.g, self.lo, self.hi)]


def _gr(P: np.ndarray, W: float, H: float, edges: np.ndarray) -> np.ndarray:
    n = len(P)
    if n < 2:
        return np.full(len(edges) - 1, np.nan)
    tree = cKDTree(P)
    pr = tree.query_pairs(edges[-1], output_type="ndarray")
    if len(pr) == 0:
        return np.zeros(len(edges) - 1)
    v = P[pr[:, 1]] - P[pr[:, 0]]
    d = np.hypot(v[:, 0], v[:, 1])
    w = 1.0 / np.maximum((W - np.abs(v[:, 0])) * (H - np.abs(v[:, 1])), 1e-300)
    hist, _ = np.histogram(d, bins=edges, weights=2 * w)            # ordered pairs
    A = W * H
    shell = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return hist * A * A / (n * (n - 1)) / shell


def pair_correlation(objects: Sequence[SurfaceObject] | np.ndarray, shape: tuple[int, int],
                     scale: tuple[float, float], r_max: Optional[float] = None, bins: int = 50, n_sim: int = 99,
                     seed: int = 0) -> PairCorrelation:
    """g(r) (translation-corrected) with the 2.5-97.5 % envelope of ``n_sim`` random arrangements."""
    P = np.asarray(objects, float).reshape(-1, 2) * np.asarray(scale) if isinstance(objects, np.ndarray) else \
        _xy(objects, scale)
    rows, cols = shape
    W, H = cols * scale[0], rows * scale[1]
    r_max = r_max or 0.25 * min(W, H)
    edges = np.linspace(0, r_max, bins + 1)
    g = _gr(P, W, H, edges)
    rng = np.random.default_rng(seed)
    sims = np.array([_gr(np.column_stack([rng.uniform(0, W, len(P)), rng.uniform(0, H, len(P))]), W, H, edges)
                     for _ in range(n_sim)]) if len(P) > 1 else np.zeros((1, bins))
    lo, hi = np.nanpercentile(sims, [2.5, 97.5], axis=0)
    rc = 0.5 * (edges[1:] + edges[:-1])
    peaks = [float(rc[i]) for i in range(1, bins - 1)
             if g[i] > hi[i] and g[i] >= g[i - 1] and g[i] >= g[i + 1] and g[i] > 1]
    return PairCorrelation(rc, g, lo, hi, len(P), len(P) / (W * H), peaks)


def clark_evans(objects: Sequence[SurfaceObject], shape: tuple[int, int], scale: tuple[float, float]) -> dict[str, float]:
    """
    Clark-Evans aggregation index R = mean nearest-neighbour distance / its expectation under
    complete spatial randomness, with Donnelly's edge correction for a rectangle.  R < 1 clustered,
    ≈ 1 random, > 1 regular (spread out).  ``p_clustered`` / ``p_regular`` are one-sided normal
    approximations (reliable for about 20 points or more).
    """
    from scipy.stats import norm
    P = _xy(objects, scale)
    n = len(P)
    rows, cols = shape
    W, H = cols * scale[0], rows * scale[1]
    A, per = W * H, 2 * (W + H)
    if n < 3:
        return {"n": n, "R": math.nan}
    d = cKDTree(P).query(P, 2)[0][:, 1]
    mean = float(d.mean())
    E = 0.5 * math.sqrt(A / n) + (0.0514 + 0.041 / math.sqrt(n)) * per / n
    var = 0.0703 * A / n ** 2 + 0.037 * per * math.sqrt(A / n ** 5)
    z = (mean - E) / math.sqrt(var)
    return {"n": n, "mean_nn": mean, "expected_nn": E, "R": mean / E, "z": z, "p_clustered": float(norm.cdf(z)),
            "p_regular": float(norm.sf(z))}


def bond_order(objects: Sequence[SurfaceObject], scale: tuple[float, float], fold: int = 6,
               graph: Optional[NeighborGraph] = None) -> tuple[np.ndarray, float]:
    """Local bond-orientational order |ψₙ| per object (1 = perfect n-fold neighbourhood) and the
    global |⟨ψₙ⟩| (1 = one orientation across the whole image)."""
    g = graph or neighbor_graph(objects, scale)
    P = _xy(objects, scale)
    psi = np.zeros(len(P), complex)
    cnt = np.zeros(len(P))
    for a, b in g.edges:
        v = P[b] - P[a]
        t = math.atan2(-v[1], v[0])
        psi[a] += np.exp(1j * fold * t)
        psi[b] += np.exp(1j * fold * (t + math.pi))
        cnt[a] += 1
        cnt[b] += 1
    loc = np.where(cnt > 0, psi / np.maximum(cnt, 1), 0)
    has = cnt > 0
    glob = float(abs(loc[has].mean())) if has.any() else math.nan
    return np.abs(loc), glob


def orientation_correlation(objects: Sequence[SurfaceObject], scale: tuple[float, float], mol_fold: int = 1,
                            r_max: Optional[float] = None, bins: int = 20) -> dict[str, np.ndarray]:
    """C(r) = ⟨cos(n (θᵢ - θⱼ))⟩ over pairs at separation r (n = ``mol_fold``): 1 parallel, 0 uncorrelated,
    -1 perpendicular (n = 2) / antiparallel (n = 1)."""
    objs = [o for o in objects if o.orientation is not None]
    P = _xy(objs, scale)
    th = np.radians([o.orientation for o in objs])
    if len(P) < 2:
        return {"r": np.zeros(0), "C": np.zeros(0), "n": np.zeros(0)}
    r_max = r_max or float(np.ptp(P, axis=0).min()) / 2
    pr = cKDTree(P).query_pairs(r_max, output_type="ndarray")
    d = np.hypot(*(P[pr[:, 0]] - P[pr[:, 1]]).T)
    c = np.cos(mol_fold * (th[pr[:, 0]] - th[pr[:, 1]]))
    edges = np.linspace(0, r_max, bins + 1)
    idx = np.digitize(d, edges) - 1
    C = np.array([c[idx == k].mean() if np.any(idx == k) else np.nan for k in range(bins)])
    N = np.array([np.sum(idx == k) for k in range(bins)])
    return {"r": 0.5 * (edges[1:] + edges[:-1]), "C": C, "n": N}


def coverage(objects: Sequence[SurfaceObject], shape: tuple[int, int], scale: tuple[float, float],
             cell_area: Optional[float] = None, area: Optional[float] = None) -> dict[str, float]:
    """Number density, area fraction (from the annotated outlines) and, with the substrate cell area,
    molecules per surface cell."""
    rows, cols = shape
    A = area or rows * cols * scale[0] * scale[1]
    n = len(objects)
    occupied = sum(o.area_px for o in objects) * scale[0] * scale[1]
    out = {"n": n, "density": n / A, "area_fraction": occupied / A}
    if cell_area:
        out["per_cell"] = n * cell_area / A
    return out


# ======================================================================== chains and motifs
def find_chains(graph: NeighborGraph, angle_tol: float = 15.0, min_length: int = 3) -> list[dict[str, Any]]:
    """Rows / chains: sequences of neighbours whose connecting vectors keep the same direction
    (within ``angle_tol``) through each molecule."""
    P = _xy(graph.objects, graph.scale)
    E = [tuple(e) for e in graph.edges.tolist()]
    nbr: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(E):
        nbr.setdefault(a, []).append(k)
        nbr.setdefault(b, []).append(k)
    parent = list(range(len(E)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def axis(k: int) -> float:
        a, b = E[k]
        v = P[b] - P[a]
        return math.degrees(math.atan2(-v[1], v[0])) % 180.0
    for node, ks in nbr.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                d = abs(axis(ks[i]) - axis(ks[j]))
                d = min(d, 180 - d)
                if d <= angle_tol:
                    parent[find(ks[i])] = find(ks[j])
    groups: dict[int, list[int]] = {}
    for k in range(len(E)):
        groups.setdefault(find(k), []).append(k)
    out = []
    for ks in groups.values():
        nodes = sorted({n for k in ks for n in E[k]})
        if len(nodes) < min_length:
            continue
        ang = np.radians([axis(k) * 2 for k in ks])
        mean_axis = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180.0
        Q = P[nodes]
        u = np.array([math.cos(math.radians(mean_axis)), -math.sin(math.radians(mean_axis))])
        length = float(np.ptp(Q @ u))
        spacing = float(np.median([np.hypot(*(P[E[k][1]] - P[E[k][0]])) for k in ks]))
        out.append({"members": [graph.objects[i].id for i in nodes], "size": len(nodes), "direction": mean_axis,
                    "length": length, "spacing": spacing})
    out.sort(key=lambda c: -c["size"])
    return out


@dataclass
class Motif:
    """An arrangement of objects relative to an anchor: positions (physical, y up, anchor at 0),
    orientations relative to the anchor's, labels and handedness."""

    offsets: np.ndarray
    rel_orient: list[Optional[float]]
    labels: list[Optional[str]]
    hands: list[Optional[str]]
    anchor_label: Optional[str]
    anchor_hand: Optional[str]
    size: float

    @classmethod
    def from_objects(cls, members: Sequence[SurfaceObject], scale: tuple[float, float]) -> "Motif":
        P = _xy(members, scale) * [1, -1]
        c = P.mean(axis=0)
        k = int(np.argmin(np.hypot(*(P - c).T)))               # anchor: the member nearest the centre
        anchor = members[k]
        others = [i for i in range(len(members)) if i != k]
        off = P[others] - P[k]
        ro = [None if (members[i].orientation is None or anchor.orientation is None)
              else (members[i].orientation - anchor.orientation) % 360 for i in others]
        return cls(off, ro, [members[i].label for i in others], [members[i].handedness for i in others],
                   anchor.label, anchor.handedness, float(np.hypot(*off.T).max()) if len(off) else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"offsets": self.offsets.tolist(), "rel_orient": self.rel_orient, "labels": self.labels,
                "hands": self.hands, "anchor_label": self.anchor_label, "anchor_hand": self.anchor_hand}


def find_motif(motif: Motif, objects: Sequence[SurfaceObject], scale: tuple[float, float], tol: float = 0.2,
               angle_tol: float = 20.0, allow_mirror: bool = True, use_labels: bool = True,
               use_orientation: bool = True) -> list[dict[str, Any]]:
    """
    Every place the motif occurs: for each candidate anchor, the motif is rotated (and mirrored when
    allowed) to put its first member on each neighbour of the anchor, the remaining members are
    matched to the nearest neighbours (Hungarian assignment), and the match is accepted when every
    member lies within ``tol`` × the motif size of its target (and orientation / label agree).  This
    is geometric graph matching, so it does not care about image rotation or the molecules'
    apparent shape.  Mirrored matches are reported separately (``mirrored``) - for chiral
    arrangements that is the enantiomeric motif.
    """
    from scipy.optimize import linear_sum_assignment
    objs = list(objects)
    P = _xy(objs, scale) * [1, -1]
    m = len(motif.offsets)
    if m == 0 or len(objs) <= m:
        return []
    tree = cKDTree(P)
    reach = motif.size * (1 + tol) + 1e-30
    out: list[dict[str, Any]] = []
    seen: set = set()
    for i, o in enumerate(objs):
        if use_labels and motif.anchor_label is not None and o.label != motif.anchor_label:
            continue
        cand = [j for j in tree.query_ball_point(P[i], reach) if j != i]
        if len(cand) < m:
            continue
        Q = P[cand] - P[i]
        for mirror in ((False, True) if allow_mirror else (False,)):
            M = motif.offsets * ([1, -1] if mirror else [1, 1])
            far = M[int(np.argmax(np.hypot(*M.T)))]             # the member that sets the rotation
            ref_ang = math.atan2(far[1], far[0])
            for q in Q:
                rot = math.atan2(q[1], q[0]) - ref_ang
                R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
                T = M @ R.T
                cost = np.hypot(*(T[:, None, :] - Q[None, :, :]).transpose(2, 0, 1))
                ri, ci = linear_sum_assignment(cost)
                err = cost[ri, ci]
                if np.any(err > tol * max(motif.size, 1e-30)):
                    continue
                ok = True
                for a_, b_ in zip(ri, ci):
                    target = objs[cand[b_]]
                    if use_labels and motif.labels[a_] is not None and target.label != motif.labels[a_]:
                        ok = False
                        break
                    if use_orientation and motif.rel_orient[a_] is not None and target.orientation is not None \
                            and o.orientation is not None:
                        want = (motif.rel_orient[a_] * (-1 if mirror else 1) + o.orientation) % 360
                        d = abs((target.orientation - want + 180) % 360 - 180)
                        if d > angle_tol:
                            ok = False
                            break
                if not ok:
                    continue
                members = tuple(sorted([o.id] + [objs[cand[b_]].id for b_ in ci]))
                key = (members, mirror)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"anchor": o.id, "members": list(members), "rms": float(np.sqrt((err ** 2).mean())),
                            "rotation": math.degrees(rot) % 360, "mirrored": mirror})
                break
    # a match found both plain and mirrored (symmetric motif) counts once, as plain
    plain = {tuple(h["members"]) for h in out if not h["mirrored"]}
    return [h for h in out if not (h["mirrored"] and tuple(h["members"]) in plain)]
