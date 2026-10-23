"""
Publication figures: an image at a journal column width with a scale bar, colour bar, panel letter and
(optionally) its annotations, as PNG / TIFF at print resolution or as vector PDF / SVG.

    style = FigureStyle(width_mm=85, colorbar="right", panel_label="a")
    fig = render_figure(image, display, style, annotations=anns, label_colors={"molecule": "#e6194b"})
    save_figure(fig, "fig2a.pdf", style)              # format from the extension
    png = figure_png(fig, style)                       # bytes, e.g. for the clipboard

The picture shows the displayed colours (levels, gamma, colour map) at the true physical aspect, so
non-square pixels look like the surface; the colour bar is rendered through the same transfer as the
image, so it stays correct with gamma or histogram equalisation.  No Qt dependency.
"""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Iterable, Mapping, Optional

import numpy as np

from .display import DisplaySettings, ImageStats as DisplayStats, display_unit, levels, render_rgba
from .model import Image
from .model.units import format_value

POSITIONS = ("lower right", "lower left", "upper right", "upper left")
COLORBARS = ("none", "right", "bottom")
FORMATS = {".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF", ".pdf": "PDF", ".svg": "SVG", ".jpg": "JPEG"}
WIDTHS_MM = {"Single column (85 mm)": 85.0, "1.5 column (120 mm)": 120.0, "Double column (175 mm)": 175.0}


@dataclass
class FigureStyle:
    width_mm: float = 85.0                   # total figure width
    dpi: int = 600                           # raster outputs (PNG / TIFF / clipboard)
    font: str = "Arial"
    font_size: float = 8.0                   # pt
    scale_bar: bool = True
    scale_bar_position: str = "lower right"
    scale_bar_length: float = 0.0            # in the image's length unit (m); 0 = a round length chosen
    scale_bar_color: str = "white"
    scale_bar_box: bool = False              # a translucent box behind the bar
    colorbar: str = "right"
    colorbar_label: str = ""                 # "" = quantity (unit)
    annotations: bool = False
    annotation_width: float = 0.8            # pt
    panel_label: str = ""                    # e.g. "a"
    axes: bool = False                       # physical axes with ticks instead of a bare image
    nan_color: str = "#000000"
    crop: Optional[tuple[float, float, float, float]] = None   # x0, y0, x1, y1 in pixels

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crop"] = list(self.crop) if self.crop else None
        return d

    @classmethod
    def from_dict(cls, d: Optional[Mapping]) -> "FigureStyle":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        out = cls(**{k: v for k, v in d.items() if k in known})
        if out.crop is not None:
            out.crop = tuple(float(v) for v in out.crop)  # type: ignore[assignment]
        if out.scale_bar_position not in POSITIONS:
            out.scale_bar_position = "lower right"
        if out.colorbar not in COLORBARS:
            out.colorbar = "right"
        return out


def _nice(v: float) -> float:
    if v <= 0 or not math.isfinite(v):
        return 1.0
    e = math.floor(math.log10(v))
    for m in (5, 2, 1):
        if m * 10 ** e <= v * 1.0000001:
            return m * 10 ** e
    return 10 ** e


def _rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return 0, 0, 0


def _crop_box(image: Image, crop) -> tuple[int, int, int, int]:  # noqa: ANN001
    h, w = image.shape
    if not crop:
        return 0, 0, w, h
    x0, y0, x1, y1 = crop
    x0, x1 = sorted((int(math.floor(max(0, min(w, x0)))), int(math.ceil(max(0, min(w, x1))))))
    y0, y1 = sorted((int(math.floor(max(0, min(h, y0)))), int(math.ceil(max(0, min(h, y1))))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0, 0, w, h
    return x0, y0, x1, y1


def render_figure(image: Image, display: Optional[DisplaySettings] = None, style: Optional[FigureStyle] = None, *,
                  annotations: Iterable = (), label_colors: Optional[Mapping[str, str]] = None):
    """A matplotlib Figure of ``image`` as displayed, laid out for print."""
    import matplotlib as mpl
    from matplotlib.figure import Figure

    st = style or FigureStyle()
    s = display or DisplaySettings()
    stats = DisplayStats(image.data)
    x0, y0, x1, y1 = _crop_box(image, st.crop)
    rgba = render_rgba(image.data, stats, s, invalid=(*_rgb(st.nan_color), 255))[y0:y1, x0:x1, :3]
    h, w = rgba.shape[:2]
    dx, dy = image.pixel_size
    calibrated = image.calibrated and dx > 0 and dy > 0
    if calibrated:
        lf, lunit = display_unit(image.x.unit, max(w * dx, h * dy))
        ext_w, ext_h = w * dx * lf, h * dy * lf
    else:
        lf, lunit = 1.0, "px"
        ext_w, ext_h = float(w), float(h)

    rc = {"font.family": "sans-serif", "font.sans-serif": [st.font, "Arial", "Helvetica", "DejaVu Sans"],
          "font.size": st.font_size, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
          "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "svg.fonttype": "none",
          "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
          "axes.edgecolor": "black", "figure.facecolor": "white", "axes.facecolor": "white"}
    with mpl.rc_context(rc):
        pt = 1 / 72                                           # inch
        W = st.width_mm / 25.4
        fs = st.font_size * pt
        cb_right = st.colorbar == "right"
        cb_bottom = st.colorbar == "bottom"
        cb_thick = max(0.10, 0.035 * W)
        # first guesses (inch); the text is then measured and the margins corrected so that axis titles,
        # tick labels and the colour bar label always fit inside the figure
        m = {"left": 4.2 * fs if st.axes else 0.0, "bottom": 0.0, "top": 0.6 * fs if st.axes else 0.0,
             "right": (0.6 * fs + cb_thick + 6.5 * fs) if cb_right else (1.8 * fs if st.axes else 0.0),
             "gap": 3.8 * fs if st.axes else 0.8 * fs}          # image bottom to colour bar top (bottom bar)
        if cb_bottom:
            m["bottom"] = 3.2 * fs
        elif st.axes:
            m["bottom"] = 3.0 * fs

        def draw(m):  # noqa: ANN001, ANN202
            img_w = max(0.15 * W, W - m["left"] - m["right"])
            img_h = img_w * ext_h / ext_w
            cb_y = m["bottom"]
            img_y = cb_y + cb_thick + m["gap"] if cb_bottom else m["bottom"]
            H = img_y + img_h + m["top"]
            fig = Figure(figsize=(W, H), dpi=100)
            ax = fig.add_axes((m["left"] / W, img_y / H, img_w / W, img_h / H))
            ax.imshow(rgba, extent=(0, ext_w, ext_h, 0), interpolation="nearest", aspect="auto")
            ax.set_xlim(0, ext_w)
            ax.set_ylim(ext_h, 0)
            if st.axes:
                ax.set_xlabel(f"x ({lunit})")
                ax.set_ylabel(f"y ({lunit})")
            else:
                ax.set_axis_off()
            if st.annotations:
                _draw_annotations(ax, annotations, label_colors or {}, image, (x0, y0), lf if calibrated else None,
                                  st.annotation_width)
            if st.scale_bar and calibrated:
                _scale_bar(ax, image, st, lf, ext_w, ext_h)
            if st.panel_label:
                color = st.scale_bar_color if not st.axes else "black"
                ax.text(0.03 * ext_w, 0.03 * ext_h, st.panel_label, color=color, fontsize=st.font_size + 2,
                        fontweight="bold", ha="left", va="top")
            cax = None
            if cb_right or cb_bottom:
                cax = _colorbar(fig, image, stats, s, st, W, H, m["left"], img_y, img_w, img_h, cb_thick, fs,
                                cb_y=cb_y)
            return fig, ax, cax

        fig, ax, cax = draw(m)
        for i in range(7):
            d = _overflow(fig, ax, cax, cb_bottom, 0.6 * fs)
            if i == 6 or all(-0.03 < v < 0.002 for v in d.values()):     # fits, without much spare room
                break
            if i >= 2:                                           # ticks may move as the image resizes: only grow
                d = {k: max(0.0, v) for k, v in d.items()}
            m = {k: max(0.0, m[k] + d.get(k, 0.0)) for k in m}
            fig, ax, cax = draw(m)
    fig._scanbound_style = st                              # noqa: SLF001 - read by save_figure
    fig._scanbound_rc = rc                                 # noqa: SLF001 - fonts and font embedding at save time
    return fig


def _scale_bar(ax, image: Image, st: FigureStyle, lf: float, ext_w: float, ext_h: float) -> None:  # noqa: ANN001
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    length = st.scale_bar_length if st.scale_bar_length > 0 else _nice(0.22 * ext_w / lf)
    size = length * lf
    label = format_value(length, image.x.unit, 3)
    bar = AnchoredSizeBar(ax.transData, size, label, st.scale_bar_position, pad=0.35, borderpad=0.5, sep=2.5,
                          color=st.scale_bar_color, frameon=st.scale_bar_box, size_vertical=0.022 * min(ext_w, ext_h),
                          label_top=True, fontproperties=FontProperties(size=st.font_size))
    if st.scale_bar_box:
        bar.patch.set_facecolor("black" if st.scale_bar_color.lower() in ("white", "#ffffff", "#fff") else "white")
        bar.patch.set_alpha(0.45)
        bar.patch.set_edgecolor("none")
    ax.add_artist(bar)


def _colorbar(fig, image: Image, stats, s: DisplaySettings, st: FigureStyle, W, H, left, bottom, img_w, img_h,  # noqa: ANN001
              thick, fs, cb_y: float = 0.0) -> None:
    """A colour bar right of the image, or below it at height ``cb_y`` (under the axis labels)."""
    lo, hi = levels(stats, s)
    f, unit = display_unit(image.unit, max(abs(lo), abs(hi), hi - lo))
    ramp = np.linspace(lo, hi, 256)
    colors = render_rgba(ramp[:, None], stats, s)[..., :3]          # through the image's own transfer
    label = st.colorbar_label or (f"{image.quantity} ({unit})" if unit else image.quantity)
    if st.colorbar == "right":
        cax = fig.add_axes(((left + img_w + 0.6 * fs) / W, bottom / H, thick / W, img_h / H))
        cax.imshow(colors[::-1], extent=(0, 1, lo * f, hi * f), aspect="auto", interpolation="bilinear")
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(label)
    else:
        cax = fig.add_axes((left / W, cb_y / H, img_w / W, thick / H))
        cax.imshow(colors[:, 0][None, :], extent=(lo * f, hi * f, 0, 1), aspect="auto", interpolation="bilinear")
        cax.set_yticks([])
        cax.set_xlabel(label)
    for sp in cax.spines.values():
        sp.set_linewidth(0.5)
    return cax


def _wrap(text, limit_px: float, renderer, horizontal: bool) -> None:  # noqa: ANN001
    words = text.get_text().replace("\n", " ").split()
    if not words:
        return

    def length() -> float:
        e = text.get_window_extent(renderer)
        return e.width if horizontal else e.height

    lines: list[str] = []
    for w in words:
        text.set_text(" ".join([*lines[-1:], w]) if lines else w)
        if lines and length() > limit_px:
            lines.append(w)
        elif lines:
            lines[-1] = f"{lines[-1]} {w}"
        else:
            lines.append(w)
    text.set_text("\n".join(lines))


def _overflow(fig, ax, cax, cb_bottom: bool, gap: float) -> dict:  # noqa: ANN001
    """How much each margin (inch) must grow (+) or may shrink (-) so that all text fits with a small pad."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    to_in = fig.dpi_scale_trans.inverted()
    if cax is not None:                                        # a long colour bar label wraps to the bar's length
        bar = cax.get_window_extent(r)
        label = cax.xaxis.label if cb_bottom else cax.yaxis.label
        _wrap(label, bar.width if cb_bottom else bar.height, r, cb_bottom)
    box = fig.get_tightbbox(r)                                 # inch already
    img = ax.get_window_extent(r).transformed(to_in)
    eps = 1e-3

    def pad(outer: float, edge: float, sign: int) -> float:    # a bare image edge stays flush with the figure
        return 0.04 if sign * (outer - edge) > eps else 0.0

    d = {"left": pad(box.x0, img.x0, -1) - box.x0, "right": box.x1 - (W - pad(box.x1, img.x1, 1)),
         "bottom": pad(box.y0, img.y0, -1) - box.y0, "top": box.y1 - (H - pad(box.y1, img.y1, 1))}
    if cb_bottom and cax is not None:
        a = ax.get_tightbbox(r).transformed(to_in)              # the image's tick labels and x title
        c = cax.get_window_extent(r).transformed(to_in)
        d["gap"] = (c.y1 + gap) - a.y0
    return d


def _draw_annotations(ax, annotations, colors: Mapping[str, str], image: Image, origin, lf: Optional[float],  # noqa: ANN001
                      width: float) -> None:
    from matplotlib.patches import Polygon as MplPolygon

    dx, dy = image.pixel_size if lf is not None else (1.0, 1.0)
    sx, sy = (dx * lf, dy * lf) if lf is not None else (1.0, 1.0)
    ox, oy = origin
    for a in annotations:
        if a.kind not in ("feature", "centre_line", "measurement_line", "orientation_vector"):
            continue
        g = a.geometry
        color = colors.get(a.label or "", "#ffd400") if a.kind == "feature" or a.kind == "centre_line" else "#00e5ff"
        pts = np.asarray(g.points if g.type == "point" else g.outline(), float)
        xy = np.column_stack(((pts[:, 0] - ox) * sx, (pts[:, 1] - oy) * sy))
        if g.type == "point":
            ax.plot(xy[:, 0], xy[:, 1], "o", ms=2.5, mfc="none", mec=color, mew=width)
        elif g.closed:
            ax.add_patch(MplPolygon(xy, closed=True, fill=False, ec=color, lw=width))
        else:
            ax.plot(xy[:, 0], xy[:, 1], "-", color=color, lw=width)
            if a.kind == "orientation_vector" and len(xy) >= 2:
                ax.annotate("", xy=xy[-1], xytext=xy[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=width))


def _savefig(fig, *args, **kw) -> None:  # noqa: ANN001
    """savefig under the figure's own rc settings: tick labels are made at draw time and PDF / SVG font
    embedding is read at save time, so both would otherwise follow the app's rcParams."""
    import matplotlib as mpl

    with mpl.rc_context(getattr(fig, "_scanbound_rc", {})):
        fig.savefig(*args, **kw)


def figure_png(fig, style: Optional[FigureStyle] = None, dpi: Optional[int] = None) -> bytes:  # noqa: ANN001
    st = style or getattr(fig, "_scanbound_style", FigureStyle())
    buf = io.BytesIO()
    _savefig(fig, buf, format="png", dpi=dpi or st.dpi, facecolor="white")
    return buf.getvalue()


def figure_svg(fig) -> bytes:  # noqa: ANN001
    buf = io.BytesIO()
    _savefig(fig, buf, format="svg", facecolor="white")
    return buf.getvalue()


def save_figure(fig, path: str, style: Optional[FigureStyle] = None, *, image: Optional[Image] = None,  # noqa: ANN001
                display: Optional[DisplaySettings] = None) -> str:
    """Write the figure (format from the extension) and ``<path>.json`` with how it was made."""
    st = style or getattr(fig, "_scanbound_style", FigureStyle())
    ext = os.path.splitext(path)[1].lower()
    if ext not in FORMATS:
        raise ValueError(f"unsupported figure format {ext!r}; use one of {', '.join(sorted(FORMATS))}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    kw: dict = {"facecolor": "white"}
    if FORMATS[ext] in ("PNG", "TIFF", "JPEG"):
        kw["dpi"] = st.dpi
    if FORMATS[ext] == "TIFF":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    _savefig(fig, path, **kw)
    side = {"format": "scanbound_publication_figure", "version": 1, "file": os.path.basename(path),
            "style": st.to_dict()}
    if image is not None:
        side["image"] = image.header()
        side["history"] = [r.to_dict() for r in image.history]
    if display is not None:
        side["display"] = display.to_dict()
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=1, default=str)
    return path
