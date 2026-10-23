"""
Image quality and tip-state checks, kept as a separate layer so low-quality regions do not silently
enter statistics.

    rep = assess(image, fit=fit)                         # QualityReport: issues + metrics
    rep.issues                                          # level, code, message, rows (when local)
    rep.bad_rows                                        # scan lines to keep out of statistics
    keep = rep.filter_objects(objects)                  # objects not on flagged lines
    cmp = compare_scans(img_a, img_b)                   # tip change / drift between repeated scans

Checks
------
``line_jump``     a scan line offset from its neighbours (feedback glitch, tip jump)
``streak``        a single line unlike both neighbours (noise spike, tip touching)
``tip_change``    the image character (roughness of the high-pass image per line) changes abruptly
                  at one line and stays changed: a tip change during the scan is likely
``noise``         high-frequency noise relative to the signal
``double_tip``    a strong repeat of the whole image content at an offset that is not a lattice
                  vector (a "ghost" image from a second tip apex)
``lattice``       atomic-resolution quality from the lattice fit (scatter, indexed fraction)
``drift``         shift between repeated scans or forward / backward channels

Each check is a flag with the evidence behind it; none of them is a verdict on the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy import ndimage

from .lattice import LatticeFit, _data, _filled, _plane_removed


@dataclass
class Flag:
    level: str                           # "info" / "warning" / "error"
    code: str
    message: str
    rows: list[int] = field(default_factory=list)
    value: Optional[float] = None


@dataclass
class QualityReport:
    issues: list[Flag]
    metrics: dict[str, float]
    shape: tuple[int, int]

    @property
    def bad_rows(self) -> list[int]:
        rows: set[int] = set()
        for f in self.issues:
            if f.level != "info" and f.code in ("line_jump", "streak"):
                rows.update(f.rows)
        return sorted(rows)

    def row_mask(self, pad: int = 1) -> np.ndarray:
        """Bool image, True on flagged lines (± ``pad``)."""
        m = np.zeros(self.shape, bool)
        for r in self.bad_rows:
            m[max(0, r - pad):r + pad + 1] = True
        return m

    def filter_objects(self, objects: Sequence[Any], pad: int = 1) -> list[Any]:
        bad = set()
        for r in self.bad_rows:
            bad.update(range(r - pad, r + pad + 1))
        return [o for o in objects if int(math.floor(o.xy[1])) not in bad]

    @property
    def score(self) -> float:
        """1 = no warnings; each warning costs 0.15, each error 0.4 (a rough triage number only)."""
        s = 1.0
        for f in self.issues:
            s -= {"warning": 0.15, "error": 0.4}.get(f.level, 0.0)
        return max(0.0, s)

    def rows(self) -> list[dict[str, Any]]:
        return [{"level": f.level, "code": f.code, "message": f.message,
                 "rows": ",".join(map(str, f.rows[:50])) + (" …" if len(f.rows) > 50 else ""), "value": f.value}
                for f in self.issues]

    def text(self) -> str:
        lines = [f"Quality score {self.score:.2f} (triage only)"]
        lines += [f"[{f.level}] {f.message}" for f in self.issues] or ["No problems found by the automatic checks."]
        lines += [f"{k}: {v:.4g}" for k, v in self.metrics.items()]
        return "\n".join(lines)


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    if not mad or not np.isfinite(mad):
        mad = np.nanstd(x) or 1.0
    return (x - med) / mad


def _change_point(x: np.ndarray, min_seg: int = 8) -> tuple[int, float]:
    """Best single split of a series by mean shift: (index, t statistic)."""
    n = len(x)
    best = (-1, 0.0)
    if n < 2 * min_seg:
        return best
    c = np.cumsum(x)
    c2 = np.cumsum(x * x)
    for k in range(min_seg, n - min_seg):
        n1, n2 = k, n - k
        m1, m2 = c[k - 1] / n1, (c[-1] - c[k - 1]) / n2
        v1 = c2[k - 1] / n1 - m1 * m1
        v2 = (c2[-1] - c2[k - 1]) / n2 - m2 * m2
        s = math.sqrt(max(v1, 0) / n1 + max(v2, 0) / n2) or 1e-30
        t = abs(m1 - m2) / s
        if t > best[1]:
            best = (k, t)
    return best


def assess(image: Any, fit: Optional[LatticeFit] = None, *, jump_z: float = 8.0, streak_z: float = 8.0,
           tip_t: float = 12.0, ghost_threshold: float = 0.35) -> QualityReport:
    raw = _data(image)
    a = _plane_removed(_filled(raw))
    rows, cols = a.shape
    issues: list[Flag] = []
    metrics: dict[str, float] = {}
    nan_frac = float((~np.isfinite(raw)).mean())
    if nan_frac > 0:
        issues.append(Flag("warning" if nan_frac > 0.02 else "info", "missing",
                           f"{100 * nan_frac:.1f} % of the samples are missing (incomplete scan lines)", value=nan_frac))
    # line offsets: median of row-to-row differences
    if rows > 4:
        d = np.median(np.diff(a, axis=0), axis=1)          # d[k]: typical change from line k to k + 1
        zd = _robust_z(np.abs(d))
        flagged = set(int(k) for k in np.nonzero(zd > min(jump_z, streak_z))[0])
        streak_rows, jump_rows = [], []
        for k in sorted(flagged):
            if k + 1 in flagged and np.sign(d[k]) != np.sign(d[k + 1]):
                streak_rows.append(k + 1)                  # up then down again: one odd line
            elif k - 1 in flagged and np.sign(d[k - 1]) != np.sign(d[k]):
                continue                                   # the second half of a streak
            elif zd[k] > jump_z:
                jump_rows.append(k + 1)
        if streak_rows:
            issues.append(Flag("warning", "streak", f"{len(streak_rows)} scan line(s) differ from both neighbours "
                                                    f"(noise spikes or the tip touching)", streak_rows))
        if jump_rows:
            issues.append(Flag("warning", "line_jump", f"{len(jump_rows)} abrupt offset(s) between scan lines "
                                                       f"(feedback glitch or tip jump)", jump_rows))
        # tip change: per-line roughness of the band-passed image
        bp = ndimage.gaussian_filter(a, 1.0) - ndimage.gaussian_filter(a, 4.0)
        rough = np.std(bp, axis=1)
        k, t = _change_point(np.log(np.maximum(rough, 1e-300)))
        metrics["tip_change_t"] = t
        if k > 0 and t > tip_t:
            ratio = float(np.median(rough[k:]) / max(np.median(rough[:k]), 1e-300))
            issues.append(Flag("warning", "tip_change", f"the image character changes at line {k} (feature contrast "
                                                        f"× {ratio:.2f} after it): a tip change during the scan is "
                                                        f"likely - compare the two parts before pooling them", [k], ratio))
    # noise: robust std of the Laplacian vs the signal
    lap = ndimage.laplace(a)
    noise = 1.4826 * float(np.median(np.abs(lap - np.median(lap)))) / math.sqrt(20)
    signal = float(np.percentile(a, 99) - np.percentile(a, 1))
    snr = signal / noise if noise > 0 else math.inf
    metrics["noise"] = noise
    metrics["signal_to_noise"] = snr
    if snr < 10:
        issues.append(Flag("warning", "noise", f"low signal-to-noise ({snr:.1f})", value=snr))
    # double tip: autocorrelation secondary peak away from the origin and lattice vectors
    hp = a - ndimage.gaussian_filter(a, max(2.0, min(rows, cols) / 32))
    F = np.fft.fft2(hp * np.outer(np.hanning(rows), np.hanning(cols)))
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2)))
    ac /= ac.max() if ac.max() > 0 else 1.0
    cy, cx = rows // 2, cols // 2
    yy, xx = np.mgrid[0:rows, 0:cols]
    rr = np.hypot(yy - cy, xx - cx)
    core = rr < max(4.0, 0.03 * min(rows, cols))
    pk = (ac == ndimage.maximum_filter(ac, size=7)) & ~core & (rr < 0.4 * min(rows, cols))
    if fit is not None:
        # discount lattice-periodic peaks
        P = np.column_stack([xx[pk] - cx, yy[pk] - cy]) * np.asarray(fit.scale)
        f = np.linalg.solve(fit.A, P.T).T
        near_lattice = np.hypot(*((f - np.round(f)) @ fit.A.T).T) < 0.2 * min(np.linalg.norm(fit.a1), np.linalg.norm(fit.a2))
        ys, xs = np.nonzero(pk)
        pk[ys[near_lattice], xs[near_lattice]] = False
    if pk.any():
        ys, xs = np.nonzero(pk)
        j = int(np.argmax(ac[ys, xs]))
        h = float(ac[ys[j], xs[j]])
        metrics["ghost_peak"] = h
        if h > ghost_threshold and fit is not None:
            issues.append(Flag("warning", "double_tip", f"the image repeats itself at an offset of ({xs[j] - cx}, "
                                                        f"{ys[j] - cy}) px (or that plus lattice vectors) with {100 * h:.0f} % correlation, not a "
                                                        f"lattice vector: possible double tip", value=h))
    # lattice quality
    if fit is not None:
        amin = min(np.linalg.norm(fit.a1), np.linalg.norm(fit.a2))
        frac = fit.n_sites / max(1, len(fit.sites))
        rel = fit.rms / amin if amin > 0 else math.nan
        metrics["lattice_rms_rel"] = rel
        metrics["lattice_indexed_fraction"] = frac
        if rel > 0.15:
            issues.append(Flag("warning", "lattice", f"sites scatter by {100 * rel:.0f} % of the lattice constant: "
                                                     f"atomic resolution is marginal or the lattice is distorted",
                               value=rel))
        if frac < 0.7:
            issues.append(Flag("warning", "lattice", f"only {100 * frac:.0f} % of the detected sites fit the lattice",
                               value=frac))
    return QualityReport(issues, metrics, (rows, cols))


def compare_scans(a: Any, b: Any, fit_a: Optional[LatticeFit] = None, fit_b: Optional[LatticeFit] = None,
                  objects_a: Sequence[Any] = (), objects_b: Sequence[Any] = ()) -> dict[str, Any]:
    """
    Two scans of the same area (repeated frames, or forward / backward): the shift between them
    (drift or hysteresis), whether the feature contrast changed (tip change), and - with lattice
    fits - whether the lattice changed (drift distortion).
    """
    from .registration import register_translation
    reg = register_translation(a, b)
    A, B = _plane_removed(_filled(_data(a))), _plane_removed(_filled(_data(b)))
    bpa = ndimage.gaussian_filter(A, 1.0) - ndimage.gaussian_filter(A, 4.0)
    bpb = ndimage.gaussian_filter(B, 1.0) - ndimage.gaussian_filter(B, 4.0)
    ratio = float(bpb.std() / max(bpa.std(), 1e-300))
    out: dict[str, Any] = {"shift_px": reg.translation, "contrast_ratio": ratio, "registration": reg.describe(),
                           "flags": []}
    if abs(math.log(max(ratio, 1e-12))) > math.log(1.4):
        out["flags"].append(f"feature contrast changed by × {ratio:.2f}: possible tip change between the scans")
    if fit_a is not None and fit_b is not None:
        ma, mb = fit_a.measurements(), fit_b.measurements()
        for k in ("a1", "a2", "gamma"):
            d = mb[k].value - ma[k].value
            s = math.hypot(ma[k].uncertainty or 0, mb[k].uncertainty or 0)
            out[f"delta_{k}"] = d
            if s > 0 and abs(d) > 5 * s:
                out["flags"].append(f"{k} differs by {d:.3g} ({abs(d) / s:.0f} σ): drift or creep distorts one scan")
    if objects_a and objects_b:
        sa = np.median([o.area_px for o in objects_a if o.area_px > 0] or [np.nan])
        sb = np.median([o.area_px for o in objects_b if o.area_px > 0] or [np.nan])
        if np.isfinite(sa) and np.isfinite(sb) and sa > 0:
            out["size_ratio"] = float(sb / sa)
            if abs(math.log(sb / sa)) > math.log(1.3):
                out["flags"].append(f"apparent feature size changed by × {sb / sa:.2f}: tip or setpoint change?")
    return out
