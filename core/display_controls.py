"""
display_controls.py
===================

Qt controls for core/image_display.py:

* ``LevelsHistogram``  histogram of the image with the colormap underneath and two
  draggable handles for the low / high display levels (drag between them to shift both,
  double-click for auto contrast).
* ``DisplayControls``  histogram + brightness / contrast / gamma sliders + options.
  Emits ``changed(final)``: final=False while the user drags (cheap preview), True when
  the adjustment is finished.
* ``fill_colormap_combo``  grouped colormap list with gradient previews.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap, QStandardItem
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QSlider, QVBoxLayout, QWidget)

from core import image_display as idp


def _lut_image(cmap, reverse=False, width=256, height=1):
    table = idp.lut(cmap, reverse)
    row = np.ascontiguousarray(np.repeat(table[None, :, :], height, axis=0))
    if width != 256:
        row = np.ascontiguousarray(row[:, np.linspace(0, 255, width).astype(int)])
    h, w, _ = row.shape
    return QImage(row.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def fill_colormap_combo(combo: QComboBox, current="gray"):
    """Grouped, previewed colormap entries; the colormap name is the item's userData."""
    combo.blockSignals(True)
    combo.clear()
    combo.setIconSize(QSize(64, 12))
    model = combo.model()
    bold = QFont()
    bold.setBold(True)
    for group, names in idp.available_colormaps():
        header = QStandardItem(group)
        header.setFlags(Qt.ItemFlag.NoItemFlags)
        header.setFont(bold)
        model.appendRow(header)
        for name in names:
            combo.addItem(QIcon(QPixmap.fromImage(_lut_image(name, width=64, height=12))), f"  {name}", name)
    index = combo.findData(current)
    combo.setCurrentIndex(index if index >= 0 else 1)
    combo.blockSignals(False)


class LevelsHistogram(QWidget):
    levelsDragged = pyqtSignal(float, float)    # new (lo, hi) in data units while dragging
    dragFinished = pyqtSignal()
    autoRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._stats = None
        self._lo = self._hi = 0.0
        self._x0 = self._x1 = 0.0          # value range shown on the x axis
        self._counts = None
        self._lut = None
        self._drag = None

    def sizeHint(self):
        return QSize(260, 86)

    def set_stats(self, stats, lo, hi, cmap, reverse):
        self._stats = stats
        # show the data range plus the levels, so handles outside the data stay visible
        self._x0 = min(stats.dmin, lo)
        self._x1 = max(stats.dmax, hi)
        if self._x1 <= self._x0:
            self._x1 = self._x0 + 1.0
        # cap the axis so a few far outliers do not squeeze the histogram into one bin
        pad = 0.6 * (stats.r_hi - stats.r_lo)
        self._x0 = max(self._x0, min(stats.r_lo - pad, lo))
        self._x1 = min(self._x1, max(stats.r_hi + pad, hi))
        counts, _ = stats.histogram(bins=160, lo=self._x0, hi=self._x1)
        self._counts = np.sqrt(counts.astype(float))
        self.set_levels(lo, hi, cmap, reverse)

    def set_levels(self, lo, hi, cmap=None, reverse=None):
        self._lo, self._hi = lo, hi
        if cmap is not None:
            self._lut = _lut_image(cmap, bool(reverse))
        self.update()

    # geometry ----------------------------------------------------------------
    def _plot_rect(self):
        return QRectF(6, 4, self.width() - 12, self.height() - 22)

    def _x_of(self, value):
        r = self._plot_rect()
        return r.left() + (value - self._x0) / (self._x1 - self._x0) * r.width()

    def _value_of(self, x):
        r = self._plot_rect()
        return self._x0 + (x - r.left()) / r.width() * (self._x1 - self._x0)

    # painting ----------------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        r = self._plot_rect()
        if self._counts is None:
            p.setPen(self.palette().color(self.foregroundRole()))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "No image")
            return
        top = self._counts.max() or 1.0
        n = len(self._counts)
        bw = r.width() / n
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(120, 120, 120))
        for i, c in enumerate(self._counts):
            h = c / top * r.height()
            p.drawRect(QRectF(r.left() + i * bw, r.bottom() - h, bw + 0.5, h))
        # colormap strip mapped between the levels
        strip = QRectF(r.left(), r.bottom() + 4, r.width(), 10)
        xl, xh = self._x_of(self._lo), self._x_of(self._hi)
        if self._lut is not None:
            p.fillRect(QRectF(strip.left(), strip.top(), max(0.0, xl - strip.left()), strip.height()),
                       QColor(*self._lut.pixelColor(0, 0).getRgb()[:3]))
            p.fillRect(QRectF(xh, strip.top(), max(0.0, strip.right() - xh), strip.height()),
                       QColor(*self._lut.pixelColor(255, 0).getRgb()[:3]))
            p.drawImage(QRectF(xl, strip.top(), max(1.0, xh - xl), strip.height()), self._lut)
        # shaded outside, handles
        shade = QColor(255, 255, 255, 120)
        p.fillRect(QRectF(r.left(), r.top(), max(0.0, xl - r.left()), r.height()), shade)
        p.fillRect(QRectF(xh, r.top(), max(0.0, r.right() - xh), r.height()), shade)
        for x, color in ((xl, QColor("#1565c0")), (xh, QColor("#c62828"))):
            p.setPen(QPen(color, 2))
            p.drawLine(QPointF(x, r.top()), QPointF(x, strip.bottom()))
            p.setBrush(QBrush(color))
            p.drawPolygon([QPointF(x - 5, r.top()), QPointF(x + 5, r.top()), QPointF(x, r.top() + 7)])

    # interaction ---------------------------------------------------------------
    def mousePressEvent(self, event):
        if self._stats is None or event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        xl, xh = self._x_of(self._lo), self._x_of(self._hi)
        if abs(x - xl) <= 6 and abs(x - xl) <= abs(x - xh):
            self._drag = ("lo", x, self._lo, self._hi)
        elif abs(x - xh) <= 6:
            self._drag = ("hi", x, self._lo, self._hi)
        elif xl < x < xh:
            self._drag = ("both", x, self._lo, self._hi)
        else:
            self._drag = ("lo" if x < xl else "hi", x, self._lo, self._hi)
            self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        x = event.position().x()
        if self._drag is None:
            near = self._counts is not None and min(abs(x - self._x_of(self._lo)), abs(x - self._x_of(self._hi))) <= 6
            self.setCursor(Qt.CursorShape.SizeHorCursor if near else Qt.CursorShape.ArrowCursor)
            return
        kind, x0, lo0, hi0 = self._drag
        v = self._value_of(x)
        dv = v - self._value_of(x0)
        gap = (self._x1 - self._x0) * 1e-3
        if kind == "lo":
            lo, hi = min(v, hi0 - gap), hi0
        elif kind == "hi":
            lo, hi = lo0, max(v, lo0 + gap)
        else:
            lo, hi = lo0 + dv, hi0 + dv
        self.levelsDragged.emit(lo, hi)

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self._drag = None
            self.dragFinished.emit()

    def mouseDoubleClickEvent(self, event):
        self.autoRequested.emit()


class DisplayControls(QWidget):
    changed = pyqtSignal(bool)      # final

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = idp.DisplaySettings()
        self.stats = None
        self.unit = ""

        self.histogram = LevelsHistogram(self)
        self.histogram.levelsDragged.connect(self._on_dragged)
        self.histogram.dragFinished.connect(lambda: self.changed.emit(True))
        self.histogram.autoRequested.connect(self.auto_levels)
        self.range_label = QLabel("")
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range_label.setStyleSheet("color: #555;")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.brightness = self._slider(-100, 100, 0, "Shift the display window (brighter to the right)")
        self.contrast = self._slider(-100, 100, 0, "Narrow (right) or widen (left) the display window")
        self.gamma = self._slider(-100, 100, 0, "Gamma: left brightens dark features, right bright features")
        self._value_labels = {}
        for row, (name, slider) in enumerate((("Brightness", self.brightness), ("Contrast", self.contrast),
                                              ("Gamma", self.gamma))):
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(slider, row, 1)
            label = QLabel("")
            label.setMinimumWidth(34)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(label, row, 2)
            self._value_labels[slider] = label
        self.brightness.valueChanged.connect(self._on_brightness_contrast)
        self.contrast.valueChanged.connect(self._on_brightness_contrast)
        self.gamma.valueChanged.connect(self._on_gamma)

        self.cb_reverse = QCheckBox("Reverse colormap")
        self.cb_symmetric = QCheckBox("Symmetric about 0")
        self.cb_symmetric.setToolTip("Centre the levels on zero (for diverging colormaps and signed signals)")
        self.cb_equalize = QCheckBox("Equalize histogram")
        self.cb_equalize.setToolTip("Spread the colours evenly over the data: reveals faint features "
                                    "in images with a few very high or low regions")
        for cb in (self.cb_reverse, self.cb_symmetric, self.cb_equalize):
            cb.toggled.connect(self._on_options)

        self.btn_auto = QPushButton("Auto")
        self.btn_auto.setToolTip("Levels at the 0.5 % and 99.5 % percentiles (ignores outliers)")
        self.btn_full = QPushButton("Full range")
        self.btn_full.setToolTip("Levels at the minimum and maximum of the data")
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Auto levels, gamma 1, no equalisation")
        self.btn_auto.clicked.connect(self.auto_levels)
        self.btn_full.clicked.connect(self.full_range)
        self.btn_reset.clicked.connect(self.reset)

        options = QGridLayout()
        options.setContentsMargins(0, 0, 0, 0)
        options.addWidget(self.cb_reverse, 0, 0)
        options.addWidget(self.cb_symmetric, 0, 1)
        options.addWidget(self.cb_equalize, 1, 0, 1, 2)
        buttons = QHBoxLayout()
        for b in (self.btn_auto, self.btn_full, self.btn_reset):
            buttons.addWidget(b)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.histogram)
        lay.addWidget(self.range_label)
        lay.addLayout(grid)
        lay.addLayout(options)
        lay.addLayout(buttons)
        lay.addStretch(1)
        self._sync_widgets()

    @staticmethod
    def _slider(lo, hi, value, tip):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        s.setToolTip(tip)
        s.sliderReleased.connect(lambda: None)
        return s

    # ------------------------------------------------------------------ public
    def set_image(self, stats: idp.ImageStats, unit=""):
        """New image (keeps the relative settings)."""
        self.stats = stats
        self.unit = unit or ""
        self._sync_widgets()

    def set_cmap(self, name):
        self.settings.cmap = name
        self._sync_widgets()

    def levels(self):
        return idp.levels(self.stats, self.settings) if self.stats is not None else (0.0, 1.0)

    def auto_levels(self):
        self.settings.f_lo, self.settings.f_hi = 0.0, 1.0
        self._sync_widgets()
        self.changed.emit(True)

    def full_range(self):
        if self.stats is None:
            return
        self.settings.f_lo, self.settings.f_hi = idp.full_range_fractions(self.stats)
        self.settings.symmetric = False
        self._sync_widgets()
        self.changed.emit(True)

    def reset(self):
        self.settings = idp.DisplaySettings(cmap=self.settings.cmap, reverse=self.settings.reverse)
        self._sync_widgets()
        self.changed.emit(True)

    # ------------------------------------------------------------------ internal
    def _on_dragged(self, lo, hi):
        if self.stats is None:
            return
        self.settings.symmetric = False
        self.settings.f_lo, self.settings.f_hi = self.stats.to_fraction(lo), self.stats.to_fraction(hi)
        self._sync_widgets()
        self.changed.emit(False)

    def _on_brightness_contrast(self, _value=None):
        width = 2.0 ** (-self.contrast.value() / 33.0)
        centre = 0.5 - self.brightness.value() / 100.0
        self.settings.f_lo, self.settings.f_hi = centre - width / 2, centre + width / 2
        self._sync_widgets(sliders=False)
        self._emit_slider_change()

    def _on_gamma(self, _value=None):
        self.settings.gamma = round(5.0 ** (self.gamma.value() / 100.0), 3)
        self._sync_widgets(sliders=False)
        self._emit_slider_change()

    def _emit_slider_change(self):
        dragging = any(s.isSliderDown() for s in (self.brightness, self.contrast, self.gamma))
        self.changed.emit(not dragging)

    def _on_options(self, _checked=None):
        self.settings.reverse = self.cb_reverse.isChecked()
        self.settings.symmetric = self.cb_symmetric.isChecked()
        self.settings.equalize = self.cb_equalize.isChecked()
        self._sync_widgets()
        self.changed.emit(True)

    def _sync_widgets(self, sliders=True):
        s = self.settings
        if sliders:
            width = max(s.f_hi - s.f_lo, 1e-6)
            centre = (s.f_hi + s.f_lo) / 2
            for slider, value in ((self.brightness, round((0.5 - centre) * 100)),
                                  (self.contrast, round(-33.0 * math.log2(width))),
                                  (self.gamma, round(100 * math.log(max(s.gamma, 1e-3)) / math.log(5)))):
                slider.blockSignals(True)
                slider.setValue(int(max(slider.minimum(), min(slider.maximum(), value))))
                slider.blockSignals(False)
            for cb, value in ((self.cb_reverse, s.reverse), (self.cb_symmetric, s.symmetric),
                              (self.cb_equalize, s.equalize)):
                cb.blockSignals(True)
                cb.setChecked(value)
                cb.blockSignals(False)
        self._value_labels[self.brightness].setText(f"{self.brightness.value():+d}")
        self._value_labels[self.contrast].setText(f"{self.contrast.value():+d}")
        self._value_labels[self.gamma].setText(f"{s.gamma:.2f}")
        if self.stats is None:
            self.range_label.setText("")
            return
        lo, hi = self.levels()
        factor, unit = idp.display_unit(self.unit, max(abs(lo), abs(hi)))
        self.range_label.setText(f"Levels: {lo * factor:.4g} … {hi * factor:.4g} {unit}".strip())
        if self.histogram._stats is self.stats:
            self.histogram.set_levels(lo, hi, s.cmap, s.reverse)
        else:
            self.histogram.set_stats(self.stats, lo, hi, s.cmap, s.reverse)
