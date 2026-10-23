"""
How a 2D data array becomes the displayed image: display levels (contrast / brightness),
gamma, optional histogram equalisation and the colormap.  No Qt dependency.

Levels are stored *relative* to each image's robust range (its 0.5 - 99.5 percentile
range): f_lo = 0, f_hi = 1 is "auto contrast", f_lo = -0.2 shows a little more of the dark
end, and so on.  The same settings therefore give a comparable look on every image of a
folder, even though their absolute values differ.

    stats = ImageStats(data)
    s = DisplaySettings(cmap="afmhot", gamma=0.8)
    rgb = render(data, stats, s)              # (H, W, 3) uint8, or (H, W) uint8 for gray
    rgba = colorbar_image(stats, s, height_px=512, label="Z", unit="m")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import FuncNorm, LinearSegmentedColormap

# =============================================================================
# Colormaps
# =============================================================================

_CUSTOM = {
    # dark brown through gold to white, the usual look of STM topography
    "SPM gold": ["#000000", "#2d1500", "#6e3a05", "#b8740f", "#e8b84a", "#fff1c4", "#ffffff"],
    "Blue ice": ["#000000", "#0b1a3c", "#1d4f8f", "#3f8fd0", "#9fd3f2", "#ffffff"],
    "Green phosphor": ["#000000", "#06290f", "#0f6b2a", "#3bb35a", "#b6f5b0", "#ffffff"],
}
for _name, _colors in _CUSTOM.items():
    if _name not in colormaps:
        colormaps.register(LinearSegmentedColormap.from_list(_name, _colors, N=256))

COLORMAP_GROUPS = [
    ("SPM / microscopy", ["gray", "afmhot", "SPM gold", "copper", "gist_heat", "bone", "Blue ice",
                          "Green phosphor", "hot", "pink"]),
    ("Perceptually uniform", ["viridis", "inferno", "magma", "plasma", "cividis", "turbo", "cubehelix"]),
    ("Diverging (use 'Symmetric about 0')", ["RdBu_r", "coolwarm", "seismic", "bwr", "PuOr_r", "BrBG",
                                              "PiYG", "berlin", "managua", "vanimo"]),
    ("Sequential", ["Greys", "Blues", "Greens", "Oranges", "Purples", "Reds", "YlOrBr", "YlOrRd", "YlGnBu",
                    "GnBu", "PuBu", "BuPu", "OrRd", "Wistia", "gist_earth", "terrain", "ocean", "gnuplot2",
                    "CMRmap"]),
]
# Excluded on purpose: qualitative (tab10, Set1, Paired, ...), cyclic (hsv, twilight) and
# banded maps (flag, prism) - they invent edges that are not in the data.
DIVERGING = set(COLORMAP_GROUPS[2][1])


def available_colormaps():
    return [(g, [n for n in names if n in colormaps]) for g, names in COLORMAP_GROUPS]


def lut(cmap_name: str, reverse=False) -> np.ndarray:
    """(256, 3) uint8 lookup table."""
    cmap = colormaps[cmap_name] if cmap_name in colormaps else colormaps["gray"]
    table = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255 + 0.5).astype(np.uint8)
    return table[::-1].copy() if reverse else table


# =============================================================================
# Settings and statistics
# =============================================================================

@dataclass
class DisplaySettings:
    cmap: str = "gray"
    reverse: bool = False
    f_lo: float = 0.0           # low level, relative to the robust range (0 = 0.5th percentile)
    f_hi: float = 1.0           # high level, relative to the robust range (1 = 99.5th percentile)
    gamma: float = 1.0          # < 1 brightens dark features, > 1 brightens bright features
    equalize: bool = False      # histogram equalisation between the levels
    symmetric: bool = False     # levels symmetric about 0 (for diverging colormaps)
    robust_percent: float = 0.5

    def copy(self, **changes):
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d) -> "DisplaySettings":
        d = d or {}
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class ImageStats:
    """Range information of one image, computed once per image."""

    def __init__(self, data, robust_percent=0.5, max_samples=262_144):
        values = np.asarray(data, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            values = np.zeros(1)
        if values.size > max_samples:
            values = values[np.random.default_rng(0).choice(values.size, max_samples, replace=False)]
        self.sample = np.sort(values)
        self.dmin, self.dmax = float(self.sample[0]), float(self.sample[-1])
        self.r_lo, self.r_hi = (float(v) for v in np.percentile(self.sample, [robust_percent,
                                                                               100 - robust_percent]))
        if self.r_hi <= self.r_lo:
            self.r_lo, self.r_hi = self.dmin, self.dmax
        if self.r_hi <= self.r_lo:
            self.r_lo, self.r_hi = self.dmin - 1.0, self.dmax + 1.0

    @property
    def span(self):
        return self.r_hi - self.r_lo

    def to_value(self, f):
        return self.r_lo + f * self.span

    def to_fraction(self, value):
        return (value - self.r_lo) / self.span

    def histogram(self, bins=256, lo=None, hi=None):
        lo = self.dmin if lo is None else lo
        hi = self.dmax if hi is None or hi <= lo else hi
        counts, edges = np.histogram(self.sample, bins=bins, range=(lo, hi))
        return counts, edges


def levels(stats: ImageStats, s: DisplaySettings):
    """(low, high) display levels in data units."""
    lo, hi = stats.to_value(s.f_lo), stats.to_value(s.f_hi)
    if s.symmetric:
        m = max(abs(lo), abs(hi)) or 1.0
        lo, hi = -m, m
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-6, 1e-12)
    return lo, hi


def full_range_fractions(stats: ImageStats):
    return stats.to_fraction(stats.dmin), stats.to_fraction(stats.dmax)


# =============================================================================
# Transfer function
# =============================================================================

def transfer_functions(stats: ImageStats, s: DisplaySettings):
    """
    forward(values) -> t in [0, 1] (position in the colormap) and its inverse, the same
    mapping used for the image and for the colorbar ticks.
    """
    lo, hi = levels(stats, s)
    gamma = max(float(s.gamma), 1e-3)
    if s.equalize:
        inside = stats.sample[(stats.sample >= lo) & (stats.sample <= hi)]
        if inside.size > 1:
            counts, edges = np.histogram(inside, bins=1024, range=(lo, hi))
            cdf = np.concatenate([[0.0], np.cumsum(counts)]) / counts.sum()
            # strictly increasing so the inverse exists; keeps a small linear share
            cdf = 0.97 * cdf + 0.03 * np.linspace(0, 1, cdf.size)

            def forward(v):
                return np.interp(np.asarray(v, dtype=float), edges, cdf) ** gamma

            def inverse(t):
                return np.interp(np.clip(np.asarray(t, dtype=float), 0, 1) ** (1 / gamma), cdf, edges)

            return forward, inverse

    def forward(v):
        return np.clip((np.asarray(v, dtype=float) - lo) / (hi - lo), 0.0, 1.0) ** gamma

    def inverse(t):
        return lo + np.clip(np.asarray(t, dtype=float), 0, 1) ** (1 / gamma) * (hi - lo)

    return forward, inverse


def render(data, stats: ImageStats, s: DisplaySettings) -> np.ndarray:
    """Displayed image: (H, W) uint8 for plain gray, (H, W, 3) uint8 otherwise. NaN -> 0."""
    forward, _ = transfer_functions(stats, s)
    data = np.asarray(data, dtype=float)
    t = forward(np.nan_to_num(data, nan=stats.dmin))
    idx = np.clip((t * 255 + 0.5).astype(np.int32), 0, 255).astype(np.uint8)
    if s.cmap in ("gray", "Greys_r", "grey") and not s.reverse:
        return np.ascontiguousarray(idx)
    return np.ascontiguousarray(lut(s.cmap, s.reverse)[idx])


def render_rgba(data, stats: ImageStats, s: DisplaySettings, invalid=(0, 0, 0, 0)) -> np.ndarray:
    """(H, W, 4) uint8; samples without data (NaN) get the ``invalid`` colour (transparent by default)."""
    forward, _ = transfer_functions(stats, s)
    data = np.asarray(data, dtype=float)
    bad = ~np.isfinite(data)
    t = forward(np.where(bad, stats.dmin, data))
    idx = np.clip((t * 255 + 0.5).astype(np.int32), 0, 255).astype(np.uint8)
    rgba = np.empty(data.shape + (4,), np.uint8)
    rgba[..., :3] = lut(s.cmap, s.reverse)[idx]
    rgba[..., 3] = 255
    if bad.any():
        rgba[bad] = invalid
    return rgba


def normalized(data, stats: ImageStats, s: DisplaySettings) -> np.ndarray:
    """Float32 image in [0, 1] exactly as displayed (levels, gamma, equalisation; no colormap).
    NaN stays NaN.  This is what detectors see when they work "on the display"."""
    forward, _ = transfer_functions(stats, s)
    data = np.asarray(data, dtype=float)
    bad = ~np.isfinite(data)
    t = forward(np.where(bad, stats.dmin, data)).astype(np.float32)
    t[bad] = np.nan
    return t


# =============================================================================
# Units and colorbar
# =============================================================================

_LENGTH_TO_M = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "nm": 1e-9, "pm": 1e-12,
                "Å": 1e-10, "A_len": 1e-10}
_SI_UNITS = {"m", "A", "V", "Hz", "N", "Pa", "s", "S", "F", "W", "N/m"}
_PREFIXES = [(1e-15, "f"), (1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, ""), (1e3, "k"),
             (1e6, "M"), (1e9, "G")]


def display_unit(unit: str, span: float):
    """(factor, label): multiply data by factor to show it in `label` (e.g. m -> nm)."""
    unit = unit or ""
    base, to_base = unit, 1.0
    if unit in _LENGTH_TO_M and unit != "m":
        base, to_base = "m", _LENGTH_TO_M[unit]
    if base not in _SI_UNITS or not math.isfinite(span) or span <= 0:
        return 1.0, unit
    span_base = span * to_base
    scale, prefix = _PREFIXES[0]
    for sc, p in _PREFIXES:
        if span_base >= sc * 0.999:
            scale, prefix = sc, p
    return to_base / scale, prefix + base


def colorbar_image(stats: ImageStats, s: DisplaySettings, height_px: int, label="", unit="",
                   background="white", oversample: float = 1.0) -> np.ndarray:
    """
    RGBA image of a vertical colorbar matching `render` exactly.  The layout is that of a bar
    `height_px` tall; `oversample` renders it that many times larger (same layout, sharper
    text and gradient) for display scaled down by the same factor.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure

    lo, hi = levels(stats, s)
    factor, unit_label = display_unit(unit, max(abs(lo), abs(hi)))
    forward, inverse = transfer_functions(stats, s)
    span = hi - lo

    # Same mapping as the image inside [lo, hi], continued linearly outside it so that tick
    # values beyond the levels fall off the bar instead of piling up at its ends.
    def bar_forward(v):
        v = np.asarray(v, dtype=float) / factor
        t = forward(np.clip(v, lo, hi))
        return np.where(v < lo, (v - lo) / span, np.where(v > hi, 1 + (v - hi) / span, t))

    def bar_inverse(t):
        t = np.asarray(t, dtype=float)
        v = inverse(np.clip(t, 0, 1))
        return np.where(t < 0, lo + t * span, np.where(t > 1, hi + (t - 1) * span, v)) * factor

    norm = FuncNorm((bar_forward, bar_inverse), vmin=lo * factor, vmax=hi * factor)
    cmap = colormaps[s.cmap] if s.cmap in colormaps else colormaps["gray"]
    if s.reverse:
        cmap = cmap.reversed()

    height_px = max(int(height_px), 120)
    dpi = 100
    font = max(6.0, height_px * 0.032 * 72 / dpi)
    width_px = int(max(80, font * dpi / 72 * 7.5))
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi * max(1.0, float(oversample)),
                 facecolor=background)
    canvas = FigureCanvasAgg(fig)
    margin_v = 0.5 * font * dpi / 72 / height_px
    cax = fig.add_axes((0.08, margin_v, 0.22, 1 - 2 * margin_v))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.tick_params(labelsize=font, length=font * 0.4, width=max(0.5, font / 12))
    formatter = cb.ax.yaxis.get_major_formatter()
    if hasattr(formatter, "set_useOffset"):
        formatter.set_useOffset(False)          # show -107.4 nm, not "-0.4 + -107"
    cb.outline.set_linewidth(max(0.5, font / 12))
    text = f"{label} ({unit_label})" if unit_label else label
    if text:
        cb.set_label(text, fontsize=font)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()
