"""
Point spectroscopy (.dat): plot, derivative, normalisation, averaging; tip positions on the scan.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ...measurements.spectroscopy import derivative, normalized_conductance, smooth
from ...model.spectrum import Spectrum, average
from ...model.units import best_prefix
from ..session import Session
from ..theme import mark
from ..qtutil import save_file_name


SPECTRUM_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
                   "#bcbd22", "#7f7f7f")


class SpectraDock(QWidget):
    positionsChanged = pyqtSignal(object)        # [(rel, x_px, y_px, colour or None)] on the current image

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.files = QListWidget()
        self.files.setToolTip("Tick spectra to plot them together (and average them)")
        self.files.itemChanged.connect(lambda _it: self.plot())
        self.files.currentItemChanged.connect(lambda *_: self._on_current())
        self.signal = QComboBox()
        self.signal.currentIndexChanged.connect(lambda _i: self.plot())
        self.didv = QCheckBox("d/dV (numerical)")
        self.didv.toggled.connect(lambda _v: self.plot())
        self.norm = QCheckBox("Normalise (dI/dV)/(I/V)")
        self.norm.toggled.connect(lambda _v: self.plot())
        self.window = QSpinBox()
        self.window.setRange(0, 101)
        self.window.setSingleStep(2)
        self.window.setValue(0)
        self.window.setToolTip("Savitzky-Golay smoothing window in samples (0 = none)")
        self.window.valueChanged.connect(lambda _v: self.plot())
        self.backward = QCheckBox("Backward sweeps")
        self.backward.toggled.connect(lambda _v: self.plot())
        self.avg = QCheckBox("Average ticked")
        self.avg.toggled.connect(lambda _v: self.plot())
        self.on_scan = QCheckBox("All positions on scan")
        self.on_scan.setToolTip("Also mark where the other spectra were taken (grey rings). Ticked spectra are always "
                                "shown as dots in their plot colour.")
        self.on_scan.toggled.connect(lambda _v: self.update_positions(report=True))
        export = QPushButton("Export CSV…")
        export.clicked.connect(self.export)

        self.figure = Figure(figsize=(4, 3), layout="constrained")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.info = QLabel()
        self.info.setWordWrap(True)
        mark(self.info, "muted")

        opts = QGridLayout()
        opts.addWidget(QLabel("Signal"), 0, 0)
        opts.addWidget(self.signal, 0, 1, 1, 2)
        opts.addWidget(self.didv, 1, 0)
        opts.addWidget(self.norm, 1, 1)
        opts.addWidget(QLabel("Smooth"), 2, 0)
        opts.addWidget(self.window, 2, 1)
        opts.addWidget(self.backward, 3, 0)
        opts.addWidget(self.avg, 3, 1)
        opts.addWidget(self.on_scan, 4, 0)
        opts.addWidget(export, 4, 1)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.files, 1)
        ll.addLayout(opts)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(NavigationToolbar2QT(self.canvas, right))
        rl.addWidget(self.canvas, 1)
        rl.addWidget(self.info)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([220, 480])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(split)
        self._cache: dict[str, Spectrum] = {}
        self._last: list[tuple[str, np.ndarray, np.ndarray]] = []
        self._off_scan: list[str] = []
        session.projectChanged.connect(self.rebuild)
        session.imageChanged.connect(lambda _i: self.update_positions())

    # ------------------------------------------------------------------ files
    def rebuild(self) -> None:
        self._cache.clear()
        self.files.blockSignals(True)
        self.files.clear()
        for rel in sorted(self.session.project.spectra_files(), key=str.lower):
            it = QListWidgetItem(rel)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, rel)
            self.files.addItem(it)
        self.files.blockSignals(False)
        self.update_positions()

    def spectrum(self, rel: str) -> Optional[Spectrum]:
        if rel not in self._cache:
            try:
                self._cache[rel] = self.session.project.data_file(rel).spectrum()
            except Exception as exc:  # noqa: BLE001
                self.session.warn(f"{rel}: {exc}")
                return None
        return self._cache[rel]

    def show_file(self, rel: str) -> None:
        for i in range(self.files.count()):
            it = self.files.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == rel:
                self.files.setCurrentItem(it)
                if it.checkState() != Qt.CheckState.Checked:
                    it.setCheckState(Qt.CheckState.Checked)
                return

    def _on_current(self) -> None:
        it = self.files.currentItem()
        sp = self.spectrum(it.data(Qt.ItemDataRole.UserRole)) if it else None
        if sp is None:
            return
        names = list(dict.fromkeys(s.name for s in sp.signals))
        cur = self.signal.currentText()
        self.signal.blockSignals(True)
        self.signal.clear()
        self.signal.addItems(names)
        pick = cur if cur in names else next((n for n in ("LI Demod 1 X", "Current") if n in names), names[0] if names else "")
        self.signal.setCurrentText(pick)
        self.signal.blockSignals(False)
        self.plot()

    def selected(self) -> list[str]:
        out = []
        for i in range(self.files.count()):
            it = self.files.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        if not out and self.files.currentItem() is not None:
            out = [self.files.currentItem().data(Qt.ItemDataRole.UserRole)]
        return out

    # ------------------------------------------------------------------ plotting
    def _processed(self, sp: Spectrum, name: str) -> tuple[Spectrum, str]:
        tag = "[bwd]" if self.backward.isChecked() else ""
        s = sp.signal(name, tag) or sp.signal(name)
        if s is None:
            return sp, ""
        key = s.key
        if self.window.value() >= 3:
            sp = smooth(sp, key, window=self.window.value() | 1)
            key = sp.signals[-1].key
        if self.didv.isChecked() or self.norm.isChecked():
            sp = derivative(sp, key)
            key = sp.signals[-1].key
            if self.norm.isChecked():
                cur = sp.signal("Current", tag) or sp.signal("Current")
                if cur is not None:
                    sp = normalized_conductance(sp, key, cur.key)
                    key = sp.signals[-1].key
        return sp, key

    def plot(self) -> None:
        self.ax.clear()
        self._last = []
        name = self.signal.currentText()
        rels = self.selected()
        curves = []
        for rel in rels:
            sp = self.spectrum(rel)
            if sp is None or not name:
                continue
            try:
                p, key = self._processed(sp, name)
            except Exception as exc:  # noqa: BLE001
                self.session.warn(f"{rel}: {exc}")
                continue
            s = p.signal(key) if key else None
            if s is not None:
                curves.append((rel, p, s))
        self.update_positions()
        if not curves:
            self.ax.text(0.5, 0.5, "Pick a spectrum (.dat)", ha="center", va="center", transform=self.ax.transAxes,
                         color="#777")
            self.canvas.draw_idle()
            return
        xunit = curves[0][1].axis.unit
        xs, xu = best_prefix(float(np.nanmax(np.abs(curves[0][1].x))) or 1.0, xunit)
        yunit = curves[0][2].unit
        ys, yu = best_prefix(float(np.nanmax(np.abs(curves[0][2].data))) or 1.0, yunit)
        if self.avg.isChecked() and len(curves) > 1:
            x, mean, sem = average([c[1] for c in curves], curves[0][2].key)
            self.ax.plot(x / xs, mean / ys, color="k", lw=1.8, label=f"mean of {len(curves)}")
            self.ax.fill_between(x / xs, (mean - sem) / ys, (mean + sem) / ys, color="k", alpha=0.2, lw=0)
            self._last.append(("mean", x, mean))
        else:
            for rel, p, s in curves:
                self.ax.plot(p.x / xs, s.data / ys, lw=1.2, label=os.path.basename(rel), color=self.color(rel))
                self._last.append((rel, p.x, s.data))
        self.ax.set_xlabel(f"{curves[0][1].axis.label or curves[0][1].axis.name} ({xu})")
        self.ax.set_ylabel(f"{curves[0][2].key.split(' (')[0]}" + (f" ({yu})" if yu else ""))
        self.ax.grid(alpha=0.3)
        if len(curves) > 1:
            self.ax.legend(fontsize=7)
        self.canvas.draw_idle()
        sp0 = curves[0][1]
        steps = "; ".join(r.op.split(".")[-1] for r in sp0.history) or "raw"
        where = ""
        if self._off_scan:
            where = (" · not on this scan: " + ", ".join(os.path.basename(r) for r in self._off_scan[:3])
                     + (" …" if len(self._off_scan) > 3 else ""))
        self.info.setText(f"{sp0.kind or 'spectrum'} · {len(sp0.x)} points · processing: {steps}{where}")

    def export(self) -> None:
        if not self._last:
            return
        path = save_file_name(self, "Export spectra", "spectra.csv", "CSV files (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["spectrum", "x", "value"])
            for rel, x, y in self._last:
                w.writerows([rel, float(a), float(b)] for a, b in zip(x, y))

    # ------------------------------------------------------------------ positions
    def color(self, rel: str) -> str:
        """The plot colour of a ticked spectrum (also its dot on the scan)."""
        ticked = [r for r in self.selected()]
        k = ticked.index(rel) if rel in ticked else 0
        return SPECTRUM_COLORS[k % len(SPECTRUM_COLORS)]

    def update_positions(self, report: bool = False) -> None:
        """Tip positions on the current scan: ticked spectra as dots in their plot colour; with
        'All positions' the others as grey rings."""
        self._off_scan = []
        try:
            img = self.session.image()
        except Exception:  # noqa: BLE001
            img = None
        out = []
        ticked = self.selected()
        if img is not None and img.frame is not None:
            h, w = img.shape
            for i in range(self.files.count()):
                rel = self.files.item(i).data(Qt.ItemDataRole.UserRole)
                on = rel in ticked
                if not on and not self.on_scan.isChecked():
                    continue
                sp = self.spectrum(rel)
                if sp is None or sp.position is None:
                    continue
                px = img.frame.world_to_pixel(np.array([[sp.position[0], sp.position[1]]]), img.shape)[0]
                if 0 <= px[0] <= w and 0 <= px[1] <= h:
                    out.append((rel, float(px[0]), float(px[1]), self.color(rel) if on else None))
                elif on:
                    self._off_scan.append(rel)
        self.positionsChanged.emit(out)
        if report and self.on_scan.isChecked():
            self.session.info(f"{len(out)} spectrum position(s) on this scan")
