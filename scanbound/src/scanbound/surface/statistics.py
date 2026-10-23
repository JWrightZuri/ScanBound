"""
Population statistics: orientations relative to the crystal, orientation families, chirality,
group comparisons and randomisation tests.

    rel = relative_orientation(angles, fit.direction_angle(1, 0), mol_fold=2, sub_fold=6)
    fams = orientation_families(rel, period=symmetry_period(2, 6))
    chirality_stats(["L", "R", "L", None])           # counts, fraction L with a 95 % interval, binomial p
    compare_groups({"terrace 1": counts1, "terrace 2": counts2})   # chi-square / Fisher

Angle conventions
-----------------
Absolute angles are screen angles (degrees counter-clockwise from the image x axis).  A *relative*
angle is measured from a crystal direction and reduced by the symmetry the question allows:
the molecule's own rotational symmetry (``mol_fold``: 1 polar, 2 rod-like, 3 three-fold ...), the
substrate's (``sub_fold``: 6 for a hexagonal surface, where directions 60° apart are equivalent),
and optionally mirror symmetry (``mirror=True`` folds θ and -θ together; only use it when handedness
is tracked separately or does not matter).  The combined period is 360° / lcm(mol_fold, sub_fold).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np


# ======================================================================== orientations
def symmetry_period(mol_fold: int = 1, sub_fold: int = 1) -> float:
    return 360.0 / math.lcm(max(1, int(mol_fold)), max(1, int(sub_fold)))


def relative_orientation(angles: Any, reference: float = 0.0, mol_fold: int = 1, sub_fold: int = 1,
                         mirror: bool = False) -> np.ndarray:
    """Angles relative to ``reference`` reduced to [0, period) (or [0, period/2] with ``mirror``)."""
    p = symmetry_period(mol_fold, sub_fold)
    a = np.mod(np.asarray(angles, float) - float(reference), p)
    if mirror:
        a = np.minimum(a, p - a)
    return a


def circular_kde(angles: Sequence[float], period: float, bandwidth: float, n: int = 720,
                 weights: Optional[Sequence[float]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Density on [0, period) (integrates to 1) from a wrapped Gaussian kernel of ``bandwidth`` degrees."""
    a = np.asarray(angles, float)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(a) & np.isfinite(w)
    a, w = a[ok], w[ok]
    x = np.linspace(0, period, n, endpoint=False)
    if len(a) == 0 or w.sum() <= 0:
        return x, np.zeros(n)
    d = (x[:, None] - a[None, :] + period / 2) % period - period / 2
    k = sum(np.exp(-0.5 * ((d + s * period) / bandwidth) ** 2) for s in (-1, 0, 1))
    dens = (k * w[None, :]).sum(axis=1) / (w.sum() * bandwidth * math.sqrt(2 * math.pi))
    return x, dens


@dataclass
class Family:
    centre: float
    count: int
    weight: float
    spread: float                      # circular standard deviation of the members, degrees
    members: np.ndarray                # indices

    def describe(self, total: int) -> str:
        return f"{self.centre:.1f}° ± {self.spread:.1f}°: {self.count} ({100 * self.count / max(1, total):.0f} %)"


@dataclass
class FamilyResult:
    families: list[Family]
    assignment: np.ndarray             # family index per angle, -1 = none
    period: float
    bandwidth: float
    rayleigh_p: float                  # Rayleigh test: one preferred direction vs uniform
    uniform_p: float = math.nan        # chi-square test of any departure from uniform (multi-modal too)
    kde: tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.zeros(0), np.zeros(0)))

    def rows(self) -> list[dict[str, Any]]:
        n = len(self.assignment)
        return [{"family": i, "centre_deg": f.centre, "count": f.count, "fraction": f.count / max(1, n),
                 "weighted": f.weight, "spread_deg": f.spread} for i, f in enumerate(self.families)]


def _circ_mean_std(a: np.ndarray, period: float, w: Optional[np.ndarray] = None) -> tuple[float, float, float]:
    t = np.radians(a * 360.0 / period)
    w = np.ones(len(a)) if w is None else w
    C, S = (w * np.cos(t)).sum() / w.sum(), (w * np.sin(t)).sum() / w.sum()
    R = math.hypot(C, S)
    mean = (math.degrees(math.atan2(S, C)) * period / 360.0) % period
    std = math.degrees(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-12))))) * period / 360.0
    return mean, std, R


def orientation_families(angles: Sequence[float], period: float = 360.0, bandwidth: Optional[float] = None,
                         min_fraction: float = 0.05, weights: Optional[Sequence[float]] = None,
                         max_distance: Optional[float] = None) -> FamilyResult:
    """
    Discrete preferred orientations: peaks of a circular kernel density holding at least
    ``min_fraction`` of the population and clearly more than a uniform distribution would put in
    the same window (3 Poisson standard deviations).  Each angle joins the nearest peak within ``max_distance`` (default 2.5 ×
    bandwidth); the rest stay unassigned.  ``rayleigh_p`` tests the whole set against a random
    (uniform) orientation distribution on the period.
    """
    a = np.asarray(angles, float)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(a)
    n = int(ok.sum())
    if bandwidth is None:
        bandwidth = period / 24.0
    x, dens = circular_kde(a[ok], period, bandwidth, weights=w[ok])
    peaks = []
    m = len(dens)
    for i in range(m):
        if dens[i] >= dens[i - 1] and dens[i] > dens[(i + 1) % m]:
            peaks.append(x[i])
    maxd = max_distance or 2.5 * bandwidth
    assign = np.full(len(a), -1)
    if peaks and n:
        pk = np.array(peaks)
        d = np.abs((a[:, None] - pk[None, :] + period / 2) % period - period / 2)
        j = np.where(ok, np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1), -1)
        dist = np.where(ok, d[np.arange(len(a)), np.maximum(j, 0)], np.inf)
        assign = np.where(dist <= maxd, j, -1)
    fams: list[Family] = []
    remap = {}
    for k, p in enumerate(peaks):
        mem = np.nonzero(assign == k)[0]
        if len(mem) < max(2, min_fraction * n):
            continue
        # more members than a uniform distribution puts in the same window (3 Poisson σ)?
        expected = n * min(1.0, 2 * maxd / period)
        if len(mem) < expected + 3 * math.sqrt(max(expected, 1e-9)):
            continue
        mean, std, _ = _circ_mean_std(a[mem], period, w[mem])
        remap[k] = len(fams)
        fams.append(Family(mean, len(mem), float(w[mem].sum()), std, mem))
    assign = np.array([remap.get(int(k), -1) if k >= 0 else -1 for k in assign], int)
    # Rayleigh test on the folded angles
    p_r = math.nan
    if n:
        _, _, R = _circ_mean_std(a[ok], period)
        Rn = n * R
        p_r = float(min(1.0, math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - Rn * Rn)) - (1 + 2 * n))))
    p_u = math.nan
    if n >= 10:
        from scipy.stats import chisquare
        nb = int(max(3, min(36, n // 5)))
        h, _ = np.histogram(np.mod(a[ok], period), bins=nb, range=(0, period))
        p_u = float(chisquare(h).pvalue)
    return FamilyResult(fams, assign, period, bandwidth, p_r, p_u, (x, dens))


# ======================================================================== proportions
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def chirality_stats(hands: Sequence[Optional[str]], weights: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """
    Counts of L, R and unknown; the fraction L among known with its 95 % Wilson interval; a
    two-sided binomial test of a racemic (50:50) population; and, with ``weights`` (e.g. detector
    confidences), the confidence-weighted fraction.
    """
    from scipy.stats import binomtest
    hs = [h if h in ("L", "R") else None for h in hands]
    nL, nR = hs.count("L"), hs.count("R")
    n = nL + nR
    out: dict[str, Any] = {"L": nL, "R": nR, "unknown": hs.count(None), "n_known": n}
    if n:
        out["fraction_L"] = nL / n
        out["fraction_L_ci95"] = wilson_interval(nL, n)
        out["racemic_p"] = float(binomtest(nL, n, 0.5).pvalue)
        out["ee"] = (nL - nR) / n
    if weights is not None:
        w = np.asarray(weights, float)
        wl = float(sum(wi for h, wi in zip(hs, w) if h == "L"))
        wr = float(sum(wi for h, wi in zip(hs, w) if h == "R"))
        if wl + wr > 0:
            out["weighted_fraction_L"] = wl / (wl + wr)
    return out


@dataclass
class GroupComparison:
    groups: list[str]
    categories: list[str]
    table: np.ndarray
    test: str
    statistic: float
    p: float
    warning: str = ""

    def describe(self) -> str:
        txt = f"{self.test}: p = {self.p:.3g}"
        return txt + (f" ({self.warning})" if self.warning else "")


def compare_groups(counts: dict[str, Counter | dict], categories: Optional[Sequence[str]] = None) -> GroupComparison:
    """
    Do the category proportions differ between groups (terraces, domains, labels ...)?  Chi-square
    test of independence; Fisher's exact test for 2 x 2 tables with small counts.  A small p
    says the proportions differ more than sampling alone would explain *if* the objects are
    independent - neighbouring molecules often are not, so read it as an indication.
    """
    from scipy.stats import chi2_contingency, fisher_exact
    groups = [g for g in counts if sum(dict(counts[g]).values()) > 0]
    cats = list(categories) if categories else sorted({c for g in groups for c in dict(counts[g])}, key=str)
    T = np.array([[dict(counts[g]).get(c, 0) for c in cats] for g in groups], float)
    keep = T.sum(axis=0) > 0
    T, cats = T[:, keep], [c for c, k in zip(cats, keep) if k]
    if T.shape[0] < 2 or T.shape[1] < 2:
        return GroupComparison(groups, cats, T, "not enough groups or categories", math.nan, math.nan)
    chi2, p, dof, expected = chi2_contingency(T)
    warn = ""
    if (expected < 5).mean() > 0.2:
        warn = "more than 20 % of expected counts below 5: the chi-square p-value is approximate"
    if T.shape == (2, 2) and (expected < 5).any():
        _, p = fisher_exact(T)
        return GroupComparison(groups, cats, T, "Fisher exact test", math.nan, float(p))
    return GroupComparison(groups, cats, T, f"chi-square test ({dof} dof)", float(chi2), float(p), warn)


# ======================================================================== randomisation
def random_points(n: int, shape: tuple[int, int], mask: Optional[np.ndarray] = None,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """``n`` uniform random pixel positions in the image (or in the True pixels of ``mask``)."""
    rng = rng or np.random.default_rng()
    rows, cols = shape
    if mask is None:
        return np.column_stack([rng.uniform(0, cols, n), rng.uniform(0, rows, n)])
    iy, ix = np.nonzero(mask)
    if len(iy) == 0:
        return np.zeros((0, 2))
    k = rng.integers(0, len(iy), n)
    return np.column_stack([ix[k] + rng.uniform(0, 1, n), iy[k] + rng.uniform(0, 1, n)])


@dataclass
class ProximityTest:
    observed_mean: float               # mean distance from each object to the nearest target (physical)
    observed_within: float             # fraction of objects within ``radius`` of a target
    random_mean: float
    random_within: float
    enrichment: float                  # observed_within / random_within
    p_closer: float                    # one-sided: objects are closer to targets than random positions
    p_farther: float
    radius: float
    n_objects: int
    n_targets: int
    n_sim: int
    null: str
    sim_means: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def describe(self, unit: str = "") -> str:
        from ..model.units import format_value
        return (f"{self.n_objects} objects, {self.n_targets} targets: {100 * self.observed_within:.0f} % within "
                f"{format_value(self.radius, unit)} (random: {100 * self.random_within:.0f} %, enrichment "
                f"{self.enrichment:.2f}); mean nearest distance {format_value(self.observed_mean, unit)} vs "
                f"{format_value(self.random_mean, unit)} random; p(closer) = {self.p_closer:.3g}, "
                f"p(farther) = {self.p_farther:.3g} ({self.null})")


def proximity_test(objects_px: np.ndarray, targets_px: np.ndarray, shape: tuple[int, int],
                   scale: tuple[float, float] = (1.0, 1.0), radius: Optional[float] = None, n_sim: int = 999,
                   mask: Optional[np.ndarray] = None, seed: int = 0, distance_fn: Any = None) -> ProximityTest:
    """
    Are objects closer to (or farther from) the targets than random positions would be?

    The objects are compared with ``n_sim`` sets of the same number of points placed uniformly at
    random over the image (or over ``mask``: e.g. the terrace area the objects could occupy, which
    is the fairer null hypothesis).  Targets stay fixed.  ``distance_fn(points_px) -> distances``
    replaces nearest-target distance (e.g. distance to step-edge polylines).
    """
    from .objects import nearest
    rng = np.random.default_rng(seed)
    sc = np.asarray(scale, float)
    O = np.asarray(objects_px, float).reshape(-1, 2)
    Tg = np.asarray(targets_px, float).reshape(-1, 2)

    def dist(P: np.ndarray) -> np.ndarray:
        if distance_fn is not None:
            return np.asarray(distance_fn(P), float)
        return nearest(P * sc, Tg * sc)[0]
    d_obs = dist(O)
    if radius is None:
        radius = float(np.median(d_obs)) if len(d_obs) else 0.0
    sims_mean, sims_within = [], []
    for _ in range(n_sim):
        R = random_points(len(O), shape, mask, rng)
        d = dist(R)
        sims_mean.append(float(np.mean(d)) if len(d) else math.nan)
        sims_within.append(float(np.mean(d <= radius)) if len(d) else math.nan)
    sm, sw = np.array(sims_mean), np.array(sims_within)
    om = float(np.mean(d_obs)) if len(d_obs) else math.nan
    ow = float(np.mean(d_obs <= radius)) if len(d_obs) else math.nan
    p_closer = float((1 + np.sum(sm <= om)) / (n_sim + 1))
    p_farther = float((1 + np.sum(sm >= om)) / (n_sim + 1))
    rw = float(np.nanmean(sw))
    return ProximityTest(om, ow, float(np.nanmean(sm)), rw, ow / rw if rw > 0 else math.inf, p_closer, p_farther,
                         float(radius), len(O), len(Tg), n_sim,
                         "uniform random positions over " + ("the allowed area" if mask is not None else "the image"),
                         sm)


def label_permutation_test(xy: np.ndarray, labels: Sequence[Any], a: Any, b: Any, scale: tuple[float, float],
                           radius: float, n_sim: int = 999, seed: int = 0) -> dict[str, float]:
    """
    Are objects of type ``a`` found next to objects of type ``b`` more often than if the labels were
    shuffled among the same positions?  (Tests association between types, keeping the spatial
    arrangement of all objects fixed - e.g. adatoms vs vacancies, or L vs R molecules.)
    Statistic: mean number of ``b`` neighbours within ``radius`` per ``a`` object.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    P = np.asarray(xy, float).reshape(-1, 2) * np.asarray(scale, float)
    lab = np.asarray(list(labels), dtype=object)
    pairs = cKDTree(P).query_pairs(radius, output_type="ndarray") if len(P) > 1 else np.zeros((0, 2), int)

    def stat(L: np.ndarray) -> float:
        ia, ib = L[pairs[:, 0]], L[pairs[:, 1]]
        if a == b:
            cnt = 2 * np.sum((ia == a) & (ib == a))
        else:
            cnt = np.sum((ia == a) & (ib == b)) + np.sum((ia == b) & (ib == a))
        na = np.sum(L == a)
        return float(cnt / na) if na else math.nan
    obs = stat(lab)
    sims = np.array([stat(rng.permutation(lab)) for _ in range(n_sim)])
    return {"observed": obs, "expected": float(np.nanmean(sims)),
            "p_more": float((1 + np.sum(sims >= obs)) / (n_sim + 1)),
            "p_less": float((1 + np.sum(sims <= obs)) / (n_sim + 1)), "radius": radius, "n_sim": n_sim}


def count_by(values: Iterable[Any]) -> Counter:
    return Counter(values)
