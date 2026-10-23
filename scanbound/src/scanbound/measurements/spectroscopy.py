"""
Spectroscopy processing: smoothing, numerical derivatives and normalised conductance.

Each function returns a new :class:`~scanbound.model.Spectrum` with the computed signal added and
a :class:`~scanbound.model.ProcessingRecord` appended, so the chain from raw ``I(V)`` to a plotted
``dI/dV`` can be shown and repeated.
"""

from __future__ import annotations

import numpy as np

from ..model import ProcessingRecord, Signal, Spectrum


def _get(sp: Spectrum, name: str) -> Signal:
    s = sp.signal(name)
    if s is None:
        raise KeyError(f"no signal {name!r}; available: {', '.join(sp.names())}")
    return s


def smooth(sp: Spectrum, signal: str, window: int = 7, order: int = 2) -> Spectrum:
    """Savitzky-Golay smoothing (window in samples, odd)."""
    from scipy.signal import savgol_filter
    s = _get(sp, signal)
    window = max(order + 2 | 1, int(window) | 1)
    window = min(window, (len(s.data) // 2) * 2 - 1) if len(s.data) > order + 2 else len(s.data)
    data = np.asarray(s.data, float)
    good = np.isfinite(data)
    out = np.full_like(data, np.nan)
    if good.sum() > window:
        out[good] = savgol_filter(data[good], window, min(order, window - 1))
    tag = (s.tag + " [smooth]").strip()
    return sp.with_signal(Signal(s.name, s.unit, out, tag),
                          ProcessingRecord("spectrum.smooth", {"signal": s.key, "window": window, "order": order},
                                           "derived"))


def derivative(sp: Spectrum, signal: str = "Current", window: int = 9, order: int = 2) -> Spectrum:
    """Numerical d(signal)/d(sweep) with a Savitzky-Golay derivative filter (e.g. dI/dV from I(V))."""
    from scipy.signal import savgol_filter
    s = _get(sp, signal)
    x = sp.x
    data = np.asarray(s.data, float)
    good = np.isfinite(data) & np.isfinite(x)
    out = np.full_like(data, np.nan)
    n = int(good.sum())
    if n >= 5:
        w = max(5, min(int(window) | 1, (n // 2) * 2 - 1))
        step = float(np.median(np.diff(x[good])))
        out[good] = savgol_filter(data[good], w, min(order, w - 1), deriv=1, delta=step)
    else:
        w = 0
    unit = f"{s.unit}/{sp.axis.unit}" if s.unit and sp.axis.unit else ""
    name = f"d{s.name}/d{(sp.axis.label or sp.axis.name).replace(' calc', '')}"
    return sp.with_signal(Signal(name, unit, out, s.tag),
                          ProcessingRecord("spectrum.derivative", {"signal": s.key, "window": w, "order": order},
                                           "derived"))


def normalized_conductance(sp: Spectrum, didv: str, current: str = "Current", epsilon: float = 0.0) -> Spectrum:
    """(dI/dV) / (I/V), regularised near V = 0 by ``epsilon`` (in the unit of I/V; 0 = automatic)."""
    g = _get(sp, didv)
    i = _get(sp, current)
    v = sp.x
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = np.asarray(i.data, float) / v
        gap = ~np.isfinite(iv) & np.isfinite(np.asarray(i.data, float)) & np.isfinite(v)   # V = 0: 0/0
        if gap.any() and (~gap & np.isfinite(iv)).sum() >= 2:
            ok = np.isfinite(iv)
            order = np.argsort(v[ok])
            iv[gap] = np.interp(v[gap], v[ok][order], iv[ok][order])           # the limit, from the neighbours
        if epsilon <= 0:
            epsilon = float(np.nanmedian(np.abs(iv))) * 0.05 if np.isfinite(iv).any() else 1e-30
        norm = np.asarray(g.data, float) / np.sqrt(iv ** 2 + epsilon ** 2)
    return sp.with_signal(Signal(f"({g.name})/(I/V)", "", norm, g.tag),
                          ProcessingRecord("spectrum.normalize", {"didv": g.key, "current": i.key,
                                                                  "epsilon": epsilon}, "derived"))
