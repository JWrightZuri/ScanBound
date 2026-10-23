"""
Per-object descriptors: shape and landmarks (conformations), and values in every registered channel
(STM at several biases, current, ncAFM Δf ...), with clustering and outlier flags.

    desc = shape_descriptors(objects, image)                 # lobes, asymmetry, size, apparent height ...
    tab = channel_features(objects, {"Z 1V": (img1, None), "df": (img2, reg)})   # values per channel
    cl = cluster(tab, keys=["Z 1V:contrast", "df:contrast"], k=2)                  # populations
    odd = outliers(tab, keys)                                                    # unusual combinations

Every feature name says where it comes from (``<channel>:<feature>`` for channel values, plain
names for shape), so a classification can be checked for what it relies on: geometry, electronic
contrast, force contrast, or a combination.  Channel values are sampled on the registered position of
each object; the registration error is carried in the table (``<channel>:reg_error_px``) so values
from badly registered channels can be discounted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy import ndimage

from .lattice import _data, _filled
from .objects import SurfaceObject
from .registration import Registration


def _disc(shape: tuple[int, int], cx: float, cy: float, r: float, scale: tuple[float, float]) -> np.ndarray:
    rows, cols = shape
    yy, xx = np.ogrid[0:rows, 0:cols]
    return ((xx + 0.5 - cx) * scale[0]) ** 2 + ((yy + 0.5 - cy) * scale[1]) ** 2 <= r * r


def _radius(o: SurfaceObject, scale: tuple[float, float], default: float) -> float:
    if o.area_px > 0:
        return math.sqrt(o.area_px * scale[0] * scale[1] / math.pi)
    return default


def shape_descriptors(objects: Sequence[SurfaceObject], image: Any, *, radius: Optional[float] = None,
                      lobe_sigma: float = 0.6, lobe_threshold: float = 0.35) -> list[dict[str, Any]]:
    """
    For each object: apparent height (max over the local background ring), size (equivalent
    radius), elongation and orientation from intensity moments, intensity asymmetry (offset of the
    intensity centroid from the geometric centre, in units of the radius), and *lobes*: local maxima
    inside the object above ``lobe_threshold`` of its height, as (r, angle) relative to the centre
    and to the object's orientation (when known).  Lobe angles relative to the orientation are what
    makes conformers comparable whatever the molecule's rotation.
    """
    a = _filled(_data(image))
    sc = tuple(getattr(image, "pixel_size", (1.0, 1.0))) if getattr(image, "calibrated", False) else (1.0, 1.0)
    rows, cols = a.shape
    if radius is None:
        rs = [_radius(o, sc, 0.0) for o in objects if o.area_px > 0]
        radius = float(np.median(rs)) if rs else 6.0 * math.sqrt(sc[0] * sc[1])
    sm = ndimage.gaussian_filter(a, lobe_sigma)
    out = []
    for o in objects:
        r = _radius(o, sc, radius)
        rpx = int(math.ceil(2.2 * r / min(sc)))
        x0, x1 = max(0, int(o.x) - rpx), min(cols, int(o.x) + rpx + 1)
        y0, y1 = max(0, int(o.y) - rpx), min(rows, int(o.y) + rpx + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            out.append({"id": o.id})
            continue
        crop = sm[y0:y1, x0:x1]
        inside = _disc(crop.shape, o.x - x0, o.y - y0, r, sc)
        ring = _disc(crop.shape, o.x - x0, o.y - y0, 2.0 * r, sc) & ~_disc(crop.shape, o.x - x0, o.y - y0, 1.4 * r, sc)
        if inside.sum() < 3:
            out.append({"id": o.id})
            continue
        bg = float(np.median(crop[ring])) if ring.any() else float(np.median(crop))
        v = crop - bg
        height = float(v[inside].max())
        w = np.clip(v, 0, None) * inside
        yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
        X, Y = (xx + 0.5 + x0 - o.x) * sc[0], (y0 + yy + 0.5 - o.y) * sc[1]
        W = w.sum()
        d: dict[str, Any] = {"id": o.id, "label": o.label, "apparent_height": height, "radius": r,
                             "background": bg}
        if W > 0:
            mx, my = (w * X).sum() / W, (w * Y).sum() / W
            cxx = (w * (X - mx) ** 2).sum() / W
            cyy = (w * (Y - my) ** 2).sum() / W
            cxy = (w * (X - mx) * (Y - my)).sum() / W
            ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
            d["elongation"] = float(math.sqrt(ev[1] / ev[0])) if ev[0] > 0 else math.inf
            d["asymmetry"] = float(math.hypot(mx, my) / r) if r > 0 else math.nan
            d["moment_angle"] = math.degrees(-0.5 * math.atan2(2 * cxy, cxx - cyy)) % 180.0
        # lobes
        peaks = (crop == ndimage.maximum_filter(crop, size=3)) & inside & (v > lobe_threshold * height)
        py, px = np.nonzero(peaks)
        lobes = []
        for yy_, xx_ in zip(py, px):
            dx_, dy_ = (x0 + xx_ + 0.5 - o.x) * sc[0], (y0 + yy_ + 0.5 - o.y) * sc[1]
            ang = math.degrees(math.atan2(-dy_, dx_)) % 360
            rel = (ang - o.orientation) % 360 if o.orientation is not None else None
            lobes.append({"r": math.hypot(dx_, dy_), "angle": ang, "angle_rel": rel,
                          "height": float(v[yy_, xx_] / height) if height > 0 else math.nan})
        lobes.sort(key=lambda L: -L["height"])
        d["lobes"] = lobes
        d["n_lobes"] = len(lobes)
        if lobes:
            d["lobe_r_mean"] = float(np.mean([L["r"] for L in lobes]))
            d["lobe_height_ratio"] = float(lobes[-1]["height"] / lobes[0]["height"]) if len(lobes) > 1 else 1.0
        out.append(d)
    return out


def channel_features(objects: Sequence[SurfaceObject], channels: dict[str, tuple[Any, Optional[Registration]]], *,
                     radius: Optional[float] = None, reference_scale: tuple[float, float] = (1.0, 1.0)) -> list[dict[str, Any]]:
    """
    Values of every channel at every object: ``mean`` inside the object's disc, ``max``, ``min``,
    ``contrast`` (mean inside minus median of the surrounding ring) and ``std``.  ``channels``:
    name -> (image, registration from the objects' image to that channel's image; None = same grid).
    ``radius``: physical radius of the disc (default: from each object's area, else 3 pixels).
    """
    rows_out = [{"id": o.id, "label": o.label} for o in objects]
    for name, (img, reg) in channels.items():
        a = _filled(_data(img))
        sc = tuple(img.pixel_size) if getattr(img, "calibrated", False) else (1.0, 1.0)
        reg = reg or Registration.identity()
        P = reg.apply(np.array([o.xy for o in objects], float).reshape(-1, 2))
        for row, o, p in zip(rows_out, objects, P):
            r = radius or (_radius(o, reference_scale, 0.0) or 3.0 * min(sc))
            rp = int(math.ceil(2.2 * r / min(sc))) + 1
            x0, x1 = max(0, int(p[0]) - rp), min(a.shape[1], int(p[0]) + rp + 1)
            y0, y1 = max(0, int(p[1]) - rp), min(a.shape[0], int(p[1]) + rp + 1)
            row[f"{name}:reg_error_px"] = reg.error_px
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            crop = a[y0:y1, x0:x1]
            ins = _disc(crop.shape, p[0] - x0, p[1] - y0, r, sc)
            ring = _disc(crop.shape, p[0] - x0, p[1] - y0, 2.0 * r, sc) & ~_disc(crop.shape, p[0] - x0, p[1] - y0, 1.4 * r, sc)
            if not ins.any():
                continue
            vals = crop[ins]
            row[f"{name}:mean"] = float(vals.mean())
            row[f"{name}:max"] = float(vals.max())
            row[f"{name}:min"] = float(vals.min())
            row[f"{name}:std"] = float(vals.std())
            row[f"{name}:contrast"] = float(vals.mean() - (np.median(crop[ring]) if ring.any() else np.median(crop)))
    return rows_out


def _matrix(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r.get(k, np.nan)) if r.get(k) is not None else np.nan for k in keys] for r in rows], float)
    ok = np.all(np.isfinite(X), axis=1)
    return X, ok


def standardise(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(X - med), axis=0)
    sd = np.where(mad > 0, mad, np.nanstd(X, axis=0))
    return (X - med) / np.where(sd > 0, sd, 1.0)


def kmeans(X: np.ndarray, k: int, n_init: int = 10, seed: int = 0, iters: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """(labels, centres, inertia) - k-means++ with restarts, numpy only."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
        C = np.array(C)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (lab, C, inertia)
    return best  # type: ignore[return-value]


def silhouette(X: np.ndarray, lab: np.ndarray) -> float:
    if len(set(lab.tolist())) < 2 or len(X) < 3:
        return math.nan
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        same = lab == lab[i]
        a = D[i, same].sum() / max(1, same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in set(lab.tolist()) if j != lab[i])
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))


@dataclass
class ClusterResult:
    keys: list[str]
    labels: np.ndarray                  # per row, -1 = missing values
    k: int
    silhouette: float
    centres: np.ndarray                 # in standardised units
    sizes: list[int] = field(default_factory=list)

    def describe(self) -> str:
        return (f"{self.k} groups from {', '.join(self.keys)} (silhouette {self.silhouette:.2f}; sizes "
                f"{', '.join(map(str, self.sizes))})")


def cluster(rows: Sequence[dict[str, Any]], keys: Sequence[str], k: Optional[int] = None, k_max: int = 6,
            seed: int = 0) -> ClusterResult:
    """
    Group objects by the listed features (robustly standardised).  With ``k`` None the number of
    groups (2..k_max) with the best silhouette is chosen; a silhouette below ~0.25 means there is no
    convincing grouping.
    """
    X, ok = _matrix(rows, keys)
    Z = standardise(X[ok]) if ok.any() else np.zeros((0, len(keys)))
    labels = np.full(len(rows), -1)
    if len(Z) < 3:
        return ClusterResult(list(keys), labels, 0, math.nan, np.zeros((0, len(keys))))
    best = None
    for kk in ([k] if k else range(2, min(k_max, len(Z) - 1) + 1)):
        lab, C, _ = kmeans(Z, kk, seed=seed)
        s = silhouette(Z, lab) if len(Z) <= 3000 else math.nan
        if best is None or (np.isfinite(s) and s > best[2]):
            best = (lab, C, s, kk)
    lab, C, s, kk = best  # type: ignore[misc]
    labels[ok] = lab
    return ClusterResult(list(keys), labels, kk, s, C, [int((lab == j).sum()) for j in range(kk)])


def outliers(rows: Sequence[dict[str, Any]], keys: Sequence[str], z: float = 3.5) -> list[dict[str, Any]]:
    """Objects whose feature combination is far from the population (robust Mahalanobis distance, as
    an equivalent z); the largest single-feature deviation is named so the reason is visible."""
    X, ok = _matrix(rows, keys)
    if ok.sum() < len(keys) + 3:
        return []
    Z = standardise(X[ok])
    C = np.cov(Z.T) if Z.shape[1] > 1 else np.array([[np.var(Z)]])
    Ci = np.linalg.pinv(np.atleast_2d(C))
    m = np.sqrt(np.einsum("ij,jk,ik->i", Z, Ci, Z))
    from scipy.stats import chi2, norm
    p = chi2.sf(m ** 2, df=len(keys))
    zeq = norm.isf(np.clip(p / 2, 1e-300, 0.5))
    idx = np.nonzero(ok)[0]
    out = []
    for k, i in enumerate(idx):
        if zeq[k] >= z:
            worst = int(np.argmax(np.abs(Z[k])))
            out.append({"id": rows[i]["id"], "label": rows[i].get("label"), "z": float(zeq[k]),
                        "main_feature": keys[worst], "feature_z": float(Z[k, worst])})
    out.sort(key=lambda r: -r["z"])
    return out


def class_templates(image: Any, objects: Sequence[SurfaceObject], half: int = 12) -> dict[str, np.ndarray]:
    """Mean image of each label's objects, each crop rotated to its orientation (when known) so the
    template is in the molecule's own frame."""
    from scipy.ndimage import rotate
    from .defects import crop
    groups: dict[str, list[np.ndarray]] = {}
    for o in objects:
        c = crop(image, o.xy, half)
        if not np.isfinite(c).all():
            continue
        c = c - np.median(c)
        if o.orientation is not None:
            c = rotate(c, -o.orientation, reshape=False, order=1, mode="nearest")
        groups.setdefault(str(o.label), []).append(c)
    return {k: np.mean(v, axis=0) for k, v in groups.items() if v}


def template_scores(image: Any, objects: Sequence[SurfaceObject], templates: dict[str, np.ndarray],
                    half: int = 12) -> list[dict[str, Any]]:
    """Normalised cross-correlation of each object (in its own frame) with every class template."""
    from scipy.ndimage import rotate
    from .defects import crop
    out = []
    for o in objects:
        c = crop(image, o.xy, half)
        row: dict[str, Any] = {"id": o.id, "label": o.label}
        if np.isfinite(c).all():
            c = c - np.median(c)
            if o.orientation is not None:
                c = rotate(c, -o.orientation, reshape=False, order=1, mode="nearest")
            for name, t in templates.items():
                a, b = c - c.mean(), t - t.mean()
                den = math.sqrt(float((a * a).sum() * (b * b).sum()))
                row[f"ncc:{name}"] = float((a * b).sum() / den) if den > 0 else math.nan
        out.append(row)
    return out
