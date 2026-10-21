"""
line_profile.py
===============

Height profiles along lines drawn on an image, and the window that plots them.

Lines are given in scene coordinates of the image viewer (pixel (c, r) covers [c, c+1] x
[r, r+1]); ``sample_profile`` interpolates the data along the line (bilinear), optionally
averaging over a band of ``width`` pixels perpendicular to it.
"""

from __future__ import annotations

import csv
import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from scipy import ndimage

from core import image_display as idp


def sample_profile(data, x1, y1, x2, y2, width=1, step=1.0):
    """
    (distance_px, values) along the line (x1, y1) -> (x2, y2) in scene coordinates.
    One sample per ``step`` pixel; ``width`` > 1 averages over a perpendicular band.
    """
    data = np.asarray(data, dtype=float)
    length = math.hypot(x2 - x1, y2 - y1)
    n = max(2, int(math.ceil(length / step)) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = x1 + t * (x2 - x1) - 0.5                      # scene -> array (pixel centre) coordinates
    ys = y1 + t * (y2 - y1) - 0.5
    offsets = [0.0]
    if width > 1 and length > 0:
        offsets = np.linspace(-(width - 1) / 2, (width - 1) / 2, int(width))
    nx, ny = -(y2 - y1) / (length or 1), (x2 - x1) / (length or 1)
    samples = [ndimage.map_coordinates(data, [ys + o * ny, xs + o * nx], order=1, mode="nearest")
               for o in offsets]
    return t * length, np.mean(samples, axis=0)


def roughness(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    d = v - v.mean()
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


class ProfileWindow(QDialog):
    """
    Plot of all profiles of the current image.  ``provider()`` returns a dict:
        data      2D array (displayed, processed image)
        px        nm per pixel or None
        unit      data unit (e.g. "m")
        name      channel / file title
        lines     [(name, colour, x1, y1, x2, y2)]
    """
    clearRequested = pyqtSignal()

    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Line profiles")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(760, 560)
        self._provider = provider
        self._last = []                 # [(name, colour, distance, values)] in display units

        self.band = QSpinBox()
        self.band.setRange(1, 99)
        self.band.setValue(1)
        self.band.setSuffix(" px")
        self.band.setToolTip("Average the profile over a band this wide (reduces noise)")
        self.band.valueChanged.connect(self.refresh)
        self.relative = QCheckBox("Relative to start")
        self.relative.setToolTip("Subtract the first value of each profile")
        self.relative.toggled.connect(self.refresh)
        self.level = QCheckBox("Level (end points)")
        self.level.setToolTip("Subtract the straight line between the first and last point of each profile")
        self.level.toggled.connect(self.refresh)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Line width:"))
        controls.addWidget(self.band)
        controls.addWidget(self.relative)
        controls.addWidget(self.level)
        controls.addStretch(1)

        self.figure = Figure(figsize=(6, 3.4), layout="constrained")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot = QWidget()
        pl = QVBoxLayout(plot)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(NavigationToolbar2QT(self.canvas, plot))
        pl.addWidget(self.canvas, 1)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Profile", "Length", "Δ height (end − start)", "Min", "Max",
                                              "Peak-to-valley", "Rq (RMS)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(plot)
        split.addWidget(self.table)
        split.setSizes([400, 130])

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #555;")
        buttons = QHBoxLayout()
        for text, slot in (("Copy data", self.copy_data), ("Export CSV…", self.export_csv),
                           ("Clear profiles", self.clearRequested.emit)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)

        lay = QVBoxLayout(self)
        lay.addLayout(controls)
        lay.addWidget(split, 1)
        lay.addWidget(self.hint)
        lay.addLayout(buttons)

    def refresh(self, *_):
        info = self._provider() or {}
        data, lines = info.get("data"), info.get("lines") or []
        px, unit = info.get("px"), info.get("unit", "")
        self.ax.clear()
        self.table.setRowCount(0)
        self._last = []
        if data is None or not lines:
            self.ax.text(0.5, 0.5, "Draw a line on the image (left-drag)", ha="center", va="center",
                         transform=self.ax.transAxes, color="#777")
            self.hint.setText("Profile mode: left-drag on the image to draw a profile, drag its ends or body to "
                              "adjust, right click to delete it.")
            self.canvas.draw_idle()
            return
        finite = np.asarray(data)[np.isfinite(data)]
        factor, zunit = idp.display_unit(unit, float(np.max(np.abs(finite))) if finite.size else 1.0)
        dist_unit = "nm" if px else "px"
        for name, colour, x1, y1, x2, y2 in lines:
            d, v = sample_profile(data, x1, y1, x2, y2, self.band.value())
            v = v * factor
            if self.level.isChecked() and len(v) > 1:
                v = v - np.linspace(v[0], v[-1], len(v))
            elif self.relative.isChecked():
                v = v - v[0]
            d = d * px if px else d
            self._last.append((name, colour, d, v))
            self.ax.plot(d, v, color=colour, lw=1.5, label=name)
            rq, _ = roughness(v)
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, text in enumerate([name, f"{d[-1]:.4g} {dist_unit}", f"{v[-1] - v[0]:.4g} {zunit}",
                                      f"{np.nanmin(v):.4g} {zunit}", f"{np.nanmax(v):.4g} {zunit}",
                                      f"{np.nanmax(v) - np.nanmin(v):.4g} {zunit}", f"{rq:.4g} {zunit}"]):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setForeground(Qt.GlobalColor.black)
                    item.setBackground(_qcolor(colour, 60))
                self.table.setItem(r, c, item)
        self.ax.set_xlabel(f"Distance ({dist_unit})")
        label = info.get("name") or "Value"
        self.ax.set_ylabel(f"{label} ({zunit})" if zunit else label)
        self.ax.grid(alpha=0.3)
        if len(lines) > 1:
            self.ax.legend(fontsize=8)
        self.hint.setText(f"{len(lines)} profile(s). Drag the ends of a line to adjust it, right click a line to "
                          "delete it." + ("" if px else " No pixel scale: distances in pixels."))
        self.canvas.draw_idle()

    def _rows(self):
        rows = []
        for name, _, d, v in self._last:
            rows += [(name, float(a), float(b)) for a, b in zip(d, v)]
        return rows

    def copy_data(self):
        text = "profile\tdistance\tvalue\n" + "\n".join(f"{n}\t{a:.6g}\t{b:.6g}" for n, a, b in self._rows())
        QGuiApplication.clipboard().setText(text)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export profiles", "profiles.csv", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["profile", "distance", "value"])
            writer.writerows(self._rows())


def _qcolor(color, alpha=255):
    from PyQt6.QtGui import QColor
    c = QColor(color)
    c.setAlpha(alpha)
    return c
