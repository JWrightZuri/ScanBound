"""
The Au(111) herringbone reconstruction: where its ridges run, measured from the image.

The 22 x √3 reconstruction of Au(111) shows as pairs of bright ridges (soliton walls) about
6.3 nm apart that run along the <11-2> directions and zig-zag by ±120° every ~25 nm.  Molecules
and nanoribbons often align with the ridges, so the ridge direction next to each ribbon is the
natural reference for its orientation.

The ridges are found with a structure tensor on a band-passed copy of the image:

1. adsorbates (the ROI outlines and bright outliers) and a margin around them are masked out and
   filled from their surroundings, so ribbons and their dark rims do not count; every scan line is
   levelled on its bare pixels, so line-to-line offsets do not read as stripes along the fast scan;
2. a band-pass keeps corrugations with periods between ~0.5x and ~2x the herringbone period;
3. the structure tensor, averaged over ``window_nm``, gives the local ridge direction (across the
   intensity gradient) and its coherence (1 = perfect stripes, 0 = no preferred direction).

Where the bare surface is hidden (dense ribbon layers) nothing reliable can be measured; mark the
direction by hand instead (a herringbone axis annotation) or use one from another image of the
same area.

Angles follow the app convention: degrees counter-clockwise from +x on screen (image y points
down), axial (0-180°).

    field = herringbone_field(image, px_nm=0.04, exclude=ribbon_mask)
    field.peaks                 # dominant ridge directions in the image
    field.direction_near(points, radius_px)   # local direction next to a ribbon
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

__all__ = ["HerringboneField", "HerringboneDomains", "herringbone_field", "herringbone_domains", "herringbone_axes",
           "axial_difference", "axial_mean", "AU111_PERIOD_NM"]

AU111_PERIOD_NM = 6.3          # distance between neighbouring ridge pairs (22 x 0.288 nm)


@dataclass
class HerringboneField:
    angle: np.ndarray             # (H, W) local ridge direction, degrees [0, 180)
    coherence: np.ndarray         # (H, W) 0..1
    valid: np.ndarray             # (H, W) bool: bare surface with a clear stripe pattern
    px_nm: float
    period_nm: Optional[float]    # strongest corrugation period in the band (None: no clear peak)
    peaks: list = field(default_factory=list)     # [(angle, share of the coherent area)], strongest first
    params: dict = field(default_factory=dict)
    coverage: float = 0.0         # fraction of the image with a clear stripe pattern

    @property
    def detected(self) -> bool:
        return bool(self.peaks)

    def direction_near(self, points: np.ndarray, radius_px: float, min_coherence: float = 0.25
                       ) -> tuple[float, float]:
        """
        (ridge direction, strength) around a set of points (a ribbon's centre line): the
        coherence-weighted axial mean of the valid pixels within ``radius_px``.  NaN if none.
        """
        H, W = self.angle.shape
        pts = np.asarray(points, float).reshape(-1, 2)
        if len(pts) == 0:
            return float("nan"), 0.0
        r = max(float(radius_px), 1.0)
        x0 = int(max(math.floor(pts[:, 0].min() - r), 0))
        x1 = int(min(math.ceil(pts[:, 0].max() + r) + 1, W))
        y0 = int(max(math.floor(pts[:, 1].min() - r), 0))
        y1 = int(min(math.ceil(pts[:, 1].max() + r) + 1, H))
        if x1 <= x0 or y1 <= y0:
            return float("nan"), 0.0
        near = np.zeros((y1 - y0, x1 - x0), np.uint8)
        import cv2
        line = np.round((pts - [x0 + 0.5, y0 + 0.5]) * 16).astype(np.int32)
        cv2.polylines(near, [line], False, 1, thickness=max(1, int(round(2 * r))), shift=4)
        if len(pts) == 1:
            cv2.circle(near, tuple(int(v) for v in line[0] // 16), int(r), 1, -1)
        sel = near.astype(bool) & self.valid[y0:y1, x0:x1] & (self.coherence[y0:y1, x0:x1] >= min_coherence)
        if not sel.any():
            return float("nan"), 0.0
        a = self.angle[y0:y1, x0:x1][sel]
        w = self.coherence[y0:y1, x0:x1][sel]
        mean, R = axial_mean(a, w)
        return mean, float(R * w.mean())


def axial_mean(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> tuple[float, float]:
    """Mean of axial angles (degrees, period 180) and its resultant length R (0..1)."""
    a = np.radians(2.0 * np.asarray(angles, float))
    w: np.ndarray = np.ones_like(a) if weights is None else np.asarray(weights, float)
    if a.size == 0 or w.sum() <= 0:
        return float("nan"), 0.0
    total = float(w.sum())
    c, s = float((w * np.cos(a)).sum()) / total, float((w * np.sin(a)).sum()) / total
    return float(np.degrees(0.5 * math.atan2(s, c)) % 180.0), float(math.hypot(c, s))


def axial_difference(a, b, fold: int = 1):
    """
    Angle between two axial directions, 0-90° (fold=1).  ``fold`` > 1 treats the reference as
    ``fold`` equivalent directions 180°/fold apart (3 for the rotational domains of Au(111)):
    the result is the angle to the nearest of them, 0-90°/fold.
    """
    period = 180.0 / max(int(fold), 1)
    d = np.mod(np.asarray(a, float) - np.asarray(b, float), period)
    return np.minimum(d, period - d)


def _fill(img: np.ndarray, keep: np.ndarray, sigma: float) -> np.ndarray:
    """Replace the pixels outside ``keep`` by a normalised-convolution average of their surroundings."""
    k = keep.astype(float)
    num = ndimage.gaussian_filter(np.where(keep, img, 0.0), sigma)
    den = ndimage.gaussian_filter(k, sigma)
    filled = np.where(den > 1e-3, num / np.maximum(den, 1e-3), float(np.median(img[keep])) if keep.any() else 0.0)
    return np.where(keep, img, filled)


def herringbone_field(image: np.ndarray, px_nm: float, period_nm: float = AU111_PERIOD_NM,
                      exclude: Optional[np.ndarray] = None, window_nm: Optional[float] = None,
                      margin_nm: float = 1.0, min_coherence: float = 0.25, max_peaks: int = 3
                      ) -> HerringboneField:
    """
    Local herringbone ridge directions of a topography image (see the module docstring).
    ``exclude``: bool mask of adsorbates (ROI outlines); bright outliers are excluded as well.
    """
    if not px_nm or px_nm <= 0:
        raise ValueError("the herringbone needs a calibrated pixel size")
    img = np.asarray(image, float)
    if img.ndim != 2:
        raise ValueError("image must be 2D")
    period_px = period_nm / px_nm
    window_nm = window_nm or 1.5 * period_nm
    if min(img.shape) < 2.0 * period_px:          # smaller than two periods: no stripes to see (and slow)
        z = np.zeros(img.shape)
        return HerringboneField(z, z, z.astype(bool), px_nm, None, [],
                                {"period_nm": period_nm, "window_nm": window_nm, "margin_nm": margin_nm,
                                 "min_coherence": min_coherence})
    factor = period_px / 16.0                     # ~16 px per period is plenty: work on a smaller copy
    if factor > 1.25 and min(img.shape) / factor >= 32:
        small = ndimage.zoom(np.where(np.isfinite(img), img, np.nanmedian(img)), 1.0 / factor, order=1)
        ex_small = None
        if exclude is not None:
            ex_small = ndimage.zoom(np.asarray(exclude, np.uint8), 1.0 / factor, order=0).astype(bool)
        bad = ndimage.zoom((~np.isfinite(img)).astype(np.uint8), 1.0 / factor, order=0).astype(bool)
        small = np.where(bad, np.nan, small)
        f = herringbone_field(small, px_nm * img.shape[1] / small.shape[1], period_nm, ex_small, window_nm,
                              margin_nm, min_coherence, max_peaks)
        zy, zx = img.shape[0] / small.shape[0], img.shape[1] / small.shape[1]
        up = lambda a, order: ndimage.zoom(a, (zy, zx), order=order)[:img.shape[0], :img.shape[1]]  # noqa: E731
        ang = up(f.angle, 0)
        coh = up(f.coherence, 1)
        val = up(f.valid.astype(np.uint8), 0).astype(bool)
        if ang.shape != img.shape:                # rounding: pad with the edge values
            pad = ((0, img.shape[0] - ang.shape[0]), (0, img.shape[1] - ang.shape[1]))
            ang, coh, val = (np.pad(a, pad, mode="edge") for a in (ang, coh, val))
        return HerringboneField(ang, coh, val, px_nm, f.period_nm, f.peaks, f.params, f.coverage)

    params = {"period_nm": period_nm, "window_nm": window_nm, "margin_nm": margin_nm,
              "min_coherence": min_coherence}
    H, W = img.shape
    finite = np.isfinite(img)
    if not finite.any():
        z = np.zeros_like(img)
        return HerringboneField(z, z, z.astype(bool), px_nm, None, [], params)
    img = np.where(finite, img, np.nanmedian(img))
    img = img - np.median(img, axis=1, keepdims=True)
    hp = img - ndimage.gaussian_filter(img, 2.0 * period_px)
    med = np.median(hp)
    mad = np.median(np.abs(hp - med)) * 1.4826 or float(hp.std()) or 1.0
    excl = hp > med + 4.0 * mad                          # bright outliers: adsorbates without an outline
    if exclude is not None:
        excl |= np.asarray(exclude, bool)
    excl |= ~finite
    margin = max(1, int(round(margin_nm / px_nm)))
    if excl.any():
        excl = ndimage.binary_dilation(excl, iterations=margin)
    keep = ~excl
    if keep.mean() < 0.05:
        z = np.zeros_like(img)
        return HerringboneField(z, z, keep & False, px_nm, None, [], params)   # (almost) no bare surface
    # level on the bare surface only: plane, then every scan line's offset (else the fast-scan
    # direction reads as stripes), then everything slower than ~2 periods (terraces, drift)
    yy, xx = np.mgrid[0:H, 0:W]
    A = np.column_stack([xx[keep], yy[keep], np.ones(int(keep.sum()))])
    coef, *_ = np.linalg.lstsq(A, img[keep], rcond=None)
    img = img - (coef[0] * xx + coef[1] * yy + coef[2])
    fill_sigma = max(period_px / 2.0, 1.0)
    for _ in range(2):
        filled = _fill(img, keep, fill_sigma)
        resid = np.where(keep, filled - ndimage.gaussian_filter(filled, 2.0 * period_px), np.nan)
        offs = np.zeros(H)
        rows = keep.any(axis=1)                    # lines covered by adsorbates keep offset 0
        offs[rows] = np.nanmedian(resid[rows], axis=1)
        # ridges along the fast scan also shift whole lines: keep the offsets' variation on the
        # herringbone scale (signal), remove the rest (line jumps)
        band = ndimage.gaussian_filter1d(offs, max(period_px / 8.0, 0.5)) - ndimage.gaussian_filter1d(offs, period_px / 2.0)
        img = img - (offs - band)[:, None]
    filled = _fill(img, keep, fill_sigma)
    hp = filled - ndimage.gaussian_filter(filled, 2.0 * period_px)
    # band-pass: periods ~0.5x .. ~2x the herringbone period
    bp = ndimage.gaussian_filter(hp, max(period_px / 8.0, 0.5)) - ndimage.gaussian_filter(hp, period_px / 2.0)
    gy, gx = np.gradient(bp)
    w = keep.astype(float)
    s = max(window_nm / px_nm / 2.0, 1.0)
    norm = ndimage.gaussian_filter(w, s) + 1e-9
    jxx = ndimage.gaussian_filter(w * gx * gx, s) / norm
    jyy = ndimage.gaussian_filter(w * gy * gy, s) / norm
    jxy = ndimage.gaussian_filter(w * gx * gy, s) / norm
    trace = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(trace, 1e-30)
    phi = 0.5 * np.arctan2(2 * jxy, jxx - jyy)             # gradient direction, image coordinates
    # ridges run across the gradient; screen angle = -image angle (y points down)
    angle = (np.degrees(-phi) + 90.0) % 180.0
    energy = trace / (np.median(trace[keep]) + 1e-30)
    valid = keep & (norm > 0.5) & (energy > 0.2)
    period = _dominant_period(hp, keep, px_nm, period_nm)
    coverage = float((valid & (coh >= min_coherence)).mean())
    peaks = _peaks(angle, coh * valid, min_coherence, max_peaks) if coverage >= 0.15 else []
    return HerringboneField(angle, np.clip(coh, 0, 1), valid, px_nm, period, peaks, params, coverage)


def _dominant_period(bp: np.ndarray, keep: np.ndarray, px_nm: float, period_nm: float) -> Optional[float]:
    """Strongest corrugation period (nm) between 0.5x and 2x the expected one, from the power spectrum."""
    H, W = bp.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    f = np.abs(np.fft.fftshift(np.fft.fft2(np.where(keep, bp, 0) * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=px_nm))
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=px_nm))
    k = np.hypot(*np.meshgrid(fx, fy))
    lo, hi = 1.0 / (2.0 * period_nm), 2.0 / period_nm
    band = (k >= lo) & (k <= hi)
    if band.sum() < 4:
        return None
    nb = 48
    edges = np.linspace(lo, hi, nb + 1)
    idx = np.clip(np.digitize(k[band], edges) - 1, 0, nb - 1)
    power = np.bincount(idx, weights=f[band], minlength=nb) / np.maximum(np.bincount(idx, minlength=nb), 1)
    j = int(np.argmax(power))
    if power[j] < 1.5 * np.median(power):
        return None
    return float(1.0 / ((edges[j] + edges[j + 1]) / 2.0))


def _peaks(angle: np.ndarray, weight: np.ndarray, min_coherence: float, max_peaks: int) -> list:
    """Dominant directions: peaks of the coherence-weighted angle histogram (5° bins, circular)."""
    w = np.where(weight >= min_coherence, weight, 0.0)
    total = w.sum()
    if total <= 0:
        return []
    hist, edges = np.histogram(angle, bins=36, range=(0, 180), weights=w)
    smooth = ndimage.convolve1d(hist, np.array([1, 2, 3, 2, 1], float) / 9.0, mode="wrap")
    out: list[tuple[float, float]] = []
    taken = np.zeros(36, bool)
    for _ in range(max_peaks):
        cand = np.where(taken, -1, smooth)
        j = int(np.argmax(cand))
        if cand[j] <= 0:
            break
        share = float(hist[[(j + d) % 36 for d in range(-2, 3)]].sum() / total)
        if out and share < 0.1:
            break
        centre = edges[j] + 2.5
        sel = np.abs(((angle - centre) + 90) % 180 - 90) <= 12.5
        mean, _ = axial_mean(angle[sel & (w > 0)], w[sel & (w > 0)])
        out.append((mean if math.isfinite(mean) else centre, share))
        for d in range(-4, 5):                               # suppress ±20°
            taken[(j + d) % 36] = True
    if out and out[0][1] < 0.15:
        return []                                           # no preferred direction at all
    return out


@dataclass
class HerringboneDomains:
    """
    The image split into herringbone domains: connected areas whose ridges run one way.  The
    zig-zag arms of the reconstruction and its rotational domains are separate domains, so the
    pattern is followed across the whole image, not just where it is clearest.
    """
    measured: np.ndarray          # (H, W) int: domain id where the ridges are measured, 0 elsewhere
    labels: np.ndarray            # (H, W) int: every pixel given its nearest domain (0 only without any)
    angles: list = field(default_factory=list)    # per domain id - 1: mean ridge direction (deg)
    areas: list = field(default_factory=list)     # per domain id - 1: measured share of the image

    def __len__(self) -> int:
        return len(self.angles)

    def domain_of(self, points: np.ndarray) -> int:
        """Domain most of ``points`` (x, y in pixels) lie in; 0 if none."""
        H, W = self.labels.shape
        pts = np.asarray(points, float).reshape(-1, 2)
        if not len(self) or not len(pts):
            return 0
        x = np.clip(pts[:, 0].astype(int), 0, W - 1)
        y = np.clip(pts[:, 1].astype(int), 0, H - 1)
        ids = self.labels[y, x]
        ids = ids[ids > 0]
        return int(np.argmax(np.bincount(ids))) if len(ids) else 0


def herringbone_domains(f: HerringboneField, min_coherence: float = 0.25, tolerance: float = 20.0,
                        min_size_periods: float = 1.5) -> HerringboneDomains:
    """
    Domains of the detected field: every clearly striped pixel joins the detected direction
    (``f.peaks``) it is within ``tolerance`` of; each connected area at least ``min_size_periods``
    periods across is a domain.  Pixels outside every domain (adsorbates, unclear surface) are
    given the nearest domain in ``labels``.
    """
    H, W = f.angle.shape
    measured = np.zeros((H, W), np.int32)
    if not f.peaks:
        return HerringboneDomains(measured, measured.copy())
    period_px = (f.params.get("period_nm") or AU111_PERIOD_NM) / f.px_nm
    ok = f.valid & (f.coherence >= min_coherence)
    centres = np.array([p for p, _ in f.peaks], float)
    d = np.abs((f.angle[..., None] - centres + 90.0) % 180.0 - 90.0)
    nearest = d.argmin(axis=-1)
    close = d.min(axis=-1) <= tolerance
    r = max(int(round(period_px / 4.0)), 1)
    st = ndimage.generate_binary_structure(2, 1)
    min_area = (min_size_periods * period_px) ** 2
    angles, areas = [], []
    for j in range(len(centres)):
        sel = ok & close & (nearest == j)
        sel = ndimage.binary_opening(ndimage.binary_closing(sel, st, iterations=r), st, iterations=r)
        lab, n = ndimage.label(sel)
        if not n:
            continue
        sizes = ndimage.sum(sel, lab, range(1, n + 1))
        for i in np.flatnonzero(sizes >= min_area):
            region = (lab == i + 1) & ~(measured > 0)
            w = ok & region
            mean, _ = axial_mean(f.angle[w], f.coherence[w]) if w.any() else (float(centres[j]), 0.0)
            angles.append(float(mean if math.isfinite(mean) else centres[j]))
            areas.append(float(region.mean()))
            measured[region] = len(angles)
    labels = measured.copy()
    if angles and (measured == 0).any():
        _, (iy, ix) = ndimage.distance_transform_edt(measured == 0, return_indices=True)
        labels = measured[iy, ix]
    return HerringboneDomains(measured, labels, angles, areas)


def herringbone_axes(f: HerringboneField, length_periods: float = 3.0, min_coherence: float = 0.25,
                     spacing_periods: float = 5.0, max_axes: int = 200,
                     domains: Optional[HerringboneDomains] = None
                     ) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """
    Herringbone axes to propose as annotations, covering the whole image: every domain (see
    :func:`herringbone_domains`) gets one, and a large domain one about every ``spacing_periods``
    periods, each deep inside it and along its local mean direction.
    [((x0, y0), (x1, y1), share of the image of its domain)] in scene coordinates, the largest
    domains first.
    """
    dom = domains if domains is not None else herringbone_domains(f, min_coherence)
    period_px = (f.params.get("period_nm") or AU111_PERIOD_NM) / f.px_nm
    ok = f.valid & (f.coherence >= min_coherence)
    reach = spacing_periods * period_px
    out: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    boxes = ndimage.find_objects(dom.measured)
    for i in np.argsort(dom.areas)[::-1]:
        box = boxes[i]
        if box is None:
            continue
        oy, ox = box[0].start, box[1].start
        region = dom.measured[box] == i + 1
        okr, ang, coh = ok[box], f.angle[box], f.coherence[box]
        yy, xx = np.ogrid[0:region.shape[0], 0:region.shape[1]]
        free = region.copy()
        placed = 0
        while len(out) < max_axes and free.any():
            depth = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]
            cy, cx = np.unravel_index(int(np.argmax(depth)), depth.shape)
            if placed and depth[cy, cx] < max(period_px / 2.0, 1.0):
                break                                  # what is left are slivers along the edges
            near = (yy - cy) ** 2 + (xx - cx) ** 2 <= reach ** 2
            sel = near & region & okr
            mean, _ = axial_mean(ang[sel], coh[sel]) if sel.any() else (dom.angles[i], 0.0)
            half = 0.5 * min(length_periods * period_px, 2.0 * max(depth[cy, cx], period_px))
            t = math.radians(mean if math.isfinite(mean) else dom.angles[i])
            dx, dy = half * math.cos(t), -half * math.sin(t)          # screen angle -> image direction
            x, y = ox + cx + 0.5, oy + cy + 0.5
            out.append(((x - dx, y - dy), (x + dx, y + dy), float(dom.areas[i])))
            free &= ~near
            placed += 1
    return out
