"""Light / dark appearance of the whole application (Qt widgets and matplotlib plots).

    apply_theme("dark")            # or "light", or "system" (follow the operating system)
    mark(label, "muted")           # secondary text; "warn" for warnings, "error", "ok"

Widgets never hard-code text colours: ``mark`` gives them a tone, coloured for the current theme and
recoloured when it changes.  The platform's own style (e.g. Windows 11) is kept in both themes; only a
style that cannot draw dark is swapped for Fusion with a dark palette.  No application style sheet is
set, so native widgets keep their look.
"""

from __future__ import annotations

import weakref
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory, QWidget

THEMES = ("system", "light", "dark")

_TONES = {                      # light, dark
    "muted": ("#5f6368", "#a0a4a8"),
    "warn": ("#8a4b00", "#f0b45a"),
    "error": ("#b00020", "#ff6b6b"),
    "ok": ("#2e7d32", "#7fd17f"),
}

_DARK = {
    QPalette.ColorRole.Window: "#2b2d30",
    QPalette.ColorRole.WindowText: "#e3e3e3",
    QPalette.ColorRole.Base: "#1f2023",
    QPalette.ColorRole.AlternateBase: "#27292c",
    QPalette.ColorRole.ToolTipBase: "#3a3d41",
    QPalette.ColorRole.ToolTipText: "#e3e3e3",
    QPalette.ColorRole.PlaceholderText: "#8a8d91",
    QPalette.ColorRole.Text: "#e3e3e3",
    QPalette.ColorRole.Button: "#33363a",
    QPalette.ColorRole.ButtonText: "#e3e3e3",
    QPalette.ColorRole.BrightText: "#ff5252",
    QPalette.ColorRole.Light: "#45484d",
    QPalette.ColorRole.Midlight: "#3a3d41",
    QPalette.ColorRole.Mid: "#2f3235",
    QPalette.ColorRole.Dark: "#1a1b1d",
    QPalette.ColorRole.Shadow: "#0d0d0e",
    QPalette.ColorRole.Highlight: "#2f6fb7",
    QPalette.ColorRole.HighlightedText: "#ffffff",
    QPalette.ColorRole.Link: "#6cb4ff",
    QPalette.ColorRole.LinkVisited: "#b48cff",
}
_DARK_DISABLED = {
    QPalette.ColorRole.WindowText: "#7a7d81",
    QPalette.ColorRole.Text: "#7a7d81",
    QPalette.ColorRole.ButtonText: "#7a7d81",
    QPalette.ColorRole.HighlightedText: "#b0b0b0",
}

_current = "light"
_native: Optional[str] = None             # the platform style the app started with (e.g. windows11)
DARK_CAPABLE = ("windows11", "macos", "macintosh")   # native styles that draw a dark theme themselves


def is_dark() -> bool:
    return _current == "dark"


def resolve(mode: str) -> str:
    """'light' or 'dark' for a mode ('system' asks the operating system)."""
    if mode in ("light", "dark"):
        return mode
    hints = QGuiApplication.styleHints()
    try:
        return "dark" if hints.colorScheme() == Qt.ColorScheme.Dark else "light"
    except AttributeError:                     # Qt < 6.5
        return "light"


def tone(name: str) -> str:
    """The colour of a text tone in the current theme."""
    light, dark = _TONES[name]
    return dark if is_dark() else light


_marked: "weakref.WeakSet[QWidget]" = weakref.WeakSet()


def mark(widget: QWidget, name: Optional[str]) -> QWidget:
    """Give ``widget`` a text tone (None removes it)."""
    widget.setProperty("tone", name or "")
    _marked.add(widget)
    _color(widget)
    return widget


def _color(widget: QWidget) -> None:
    name = widget.property("tone") or ""
    try:
        if name in _TONES:
            pal = QPalette()                   # only these roles are set; the rest follows the theme
            c = QColor(tone(name))
            for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text):
                pal.setColor(role, c)
            widget.setPalette(pal)
        else:
            widget.setPalette(QPalette())
    except RuntimeError:                       # the widget is gone
        pass


def _palette(dark: bool) -> QPalette:
    style = QStyleFactory.create("Fusion")
    pal = QPalette(style.standardPalette()) if style is not None else QPalette()
    if dark:
        for role, c in _DARK.items():
            pal.setColor(role, QColor(c))
        for role, c in _DARK_DISABLED.items():
            pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(c))
    return pal


def apply_theme(mode: str = "system") -> str:
    """Apply a theme to the running application; returns the one in effect ('light' / 'dark')."""
    global _current, _native
    app = QApplication.instance()
    if app is None:
        return _current
    if _native is None:                        # keep the platform's own look (buttons, panels, tabs)
        _native = app.style().name().lower()
    try:
        scheme = {"dark": Qt.ColorScheme.Dark, "light": Qt.ColorScheme.Light}.get(mode, Qt.ColorScheme.Unknown)
        QGuiApplication.styleHints().setColorScheme(scheme)
    except AttributeError:                     # Qt < 6.8
        pass
    _current = resolve(mode)
    dark = _current == "dark"
    if dark and _native not in DARK_CAPABLE:
        # the native style cannot draw dark (e.g. windowsvista): Fusion with a dark palette instead
        app.setStyle("Fusion")
        app.setPalette(_palette(True))
    else:
        if app.style().name().lower() != _native:
            app.setStyle(_native)
        app.setPalette(QPalette())             # an empty palette: back to the platform's own (light or dark)
    for w in list(_marked):
        _color(w)
    _style_matplotlib(dark)
    for w in app.allWidgets():
        restyle = getattr(w, "theme_changed", None)
        if callable(restyle):
            restyle()
    _restyle_figures(app)
    return _current


# ----------------------------------------------------------------------------- matplotlib
def _mpl_colors(dark: bool) -> dict:
    if dark:
        return {"face": "#1f2023", "axes": "#1f2023", "fg": "#e3e3e3", "grid": "#45484d"}
    return {"face": "#ffffff", "axes": "#ffffff", "fg": "#222222", "grid": "#d0d0d0"}


def _style_matplotlib(dark: bool) -> None:
    try:
        import matplotlib as mpl
    except ImportError:
        return
    c = _mpl_colors(dark)
    mpl.rcParams.update({
        "figure.facecolor": c["face"], "axes.facecolor": c["axes"], "savefig.facecolor": "white",
        "axes.edgecolor": c["fg"], "axes.labelcolor": c["fg"], "text.color": c["fg"],
        "xtick.color": c["fg"], "ytick.color": c["fg"], "grid.color": c["grid"],
        "legend.facecolor": c["axes"], "legend.edgecolor": c["grid"],
    })


def style_figure(fig) -> None:  # noqa: ANN001 - a matplotlib Figure
    """Recolour an existing figure for the current theme (new artists follow rcParams by themselves)."""
    c = _mpl_colors(is_dark())
    fig.set_facecolor(c["face"])
    for ax in fig.axes:
        ax.set_facecolor(c["axes"])
        for spine in ax.spines.values():
            spine.set_edgecolor(c["fg"])
        ax.tick_params(colors=c["fg"], which="both")
        ax.xaxis.label.set_color(c["fg"])
        ax.yaxis.label.set_color(c["fg"])
        ax.title.set_color(c["fg"])
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(c["axes"])
            for t in leg.get_texts():
                t.set_color(c["fg"])


def _restyle_figures(app: QApplication) -> None:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except ImportError:
        return
    for w in app.allWidgets():
        if isinstance(w, FigureCanvasQTAgg):
            style_figure(w.figure)
            w.draw_idle()
