"""
spm_processing.py
=================

Processing pipeline of the SPM image page: an ordered list of steps that can be added,
enabled / disabled, reordered and tuned, like the image adaptation dialog (it reuses that
dialog's step cards and cached pipeline runner).

Unlike the adaptation steps, these keep the data in its physical unit (heights stay in m,
currents in A ...), so line profiles, the colorbar and the 3D view show real values.

Adding a step: write ``func(data, **params) -> data`` and add a ``StepSpec`` to SPM_STEPS.
"""

from __future__ import annotations

import json

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QScrollArea,
                             QToolButton, QVBoxLayout, QWidget)
from scipy import ndimage

from core.fourier_filter import FOURIER_STEP, FourierCard
from core.image_adapt_dialog import Param, StepCard, StepSpec, run_pipeline

PIPELINE_FORMAT = "scanbound_spm_pipeline"


# =============================================================================
# Steps (data in physical units, shape preserved)
# =============================================================================

def _plane_fit(z, mask=None):
    rows, cols = z.shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    sel = np.isfinite(z) if mask is None else (mask & np.isfinite(z))
    A = np.column_stack([np.ones(sel.sum()), xx[sel], yy[sel]])
    c, *_ = np.linalg.lstsq(A, z[sel], rcond=None)
    return c[0] + c[1] * xx + c[2] * yy


def s_plane(z, robust):
    """Subtract the least-squares plane (robust: refit without points > 2.5 sigma away)."""
    plane = _plane_fit(z)
    if robust:
        resid = z - plane
        mask = np.abs(resid - np.median(resid)) < 2.5 * np.std(resid)
        plane = _plane_fit(z, mask)
    return z - plane


def s_poly(z, order):
    """Subtract a 2D polynomial background of the given order (fitted on a coarse grid)."""
    rows, cols = z.shape
    step = max(1, int(max(rows, cols) // 128))
    yy, xx = np.mgrid[0:rows:step, 0:cols:step].astype(float)
    sub = z[::step, ::step]
    xn, yn = xx / max(cols - 1, 1) * 2 - 1, yy / max(rows - 1, 1) * 2 - 1
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.column_stack([(xn ** i * yn ** j).ravel() for i, j in terms])
    c, *_ = np.linalg.lstsq(A, sub.ravel(), rcond=None)
    YY, XX = np.mgrid[0:rows, 0:cols].astype(float)
    XN, YN = XX / max(cols - 1, 1) * 2 - 1, YY / max(rows - 1, 1) * 2 - 1
    return z - sum(ci * XN ** i * YN ** j for ci, (i, j) in zip(c, terms))


def s_align_rows(z, method):
    """Remove line-to-line offsets of the slow scan axis."""
    if method == "Median":
        return z - np.median(z, axis=1, keepdims=True)
    if method == "Mean":
        return z - z.mean(axis=1, keepdims=True)
    if method == "Linear (per line)":
        x = np.arange(z.shape[1], dtype=float)
        coef = np.polyfit(x, z.T, 1)                       # (2, rows)
        return z - (np.outer(coef[0], x) + coef[1][:, None])
    # Median of differences: shift each line by the median step to the previous line
    diff = np.median(np.diff(z, axis=0), axis=1)
    offsets = np.concatenate([[0.0], np.cumsum(diff)])
    out = z - offsets[:, None]
    return out - np.median(out)


def s_scars(z, threshold, max_length):
    """Replace scan-line scars (short pieces of a line that jump away from both neighbours)."""
    up, down = np.roll(z, 1, axis=0), np.roll(z, -1, axis=0)
    neighbours = (up + down) / 2
    resid = z - neighbours
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) or np.std(resid) or 1.0
    bad = (np.abs(resid) > threshold * sigma) & (np.sign(z - up) == np.sign(z - down))
    bad[0], bad[-1] = False, False
    if max_length > 0:                                      # keep only short horizontal runs
        labels, n = ndimage.label(bad, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        if n:
            sizes = ndimage.sum(bad, labels, range(1, n + 1))
            long_runs = np.isin(labels, np.nonzero(sizes > max_length)[0] + 1)
            bad &= ~long_runs
    out = z.copy()
    out[bad] = neighbours[bad]
    return out


def s_background(z, sigma):
    """Remove long-range variations: subtract a heavily smoothed copy (high-pass)."""
    return z - ndimage.gaussian_filter(z, sigma)


def s_gaussian(z, sigma):
    return ndimage.gaussian_filter(z, sigma)


def s_median(z, size):
    return ndimage.median_filter(z, size=int(size))


def s_sharpen(z, strength, sigma):
    """Unsharp mask: z + strength * (z - blur(z))."""
    return z + strength * (z - ndimage.gaussian_filter(z, sigma))


def s_dog(z, sigma, ratio):
    """Difference of Gaussians (band-pass): keeps features of size ~sigma."""
    return ndimage.gaussian_filter(z, sigma) - ndimage.gaussian_filter(z, sigma * ratio)


def s_clip(z, low, high):
    lo, hi = np.percentile(z, [low, 100 - high])
    return np.clip(z, lo, hi)


def s_zero(z, reference):
    ref = {"Minimum": np.min, "Median": np.median, "Mean": np.mean}[reference](z)
    return z - ref


def s_invert(z):
    return -z


SPM_STEPS = {s.key: s for s in [
    StepSpec("plane", "Plane level", "gray", s_plane, [
        Param("robust", "Ignore outliers", "bool", True,
              tooltip="Refit without pixels far from the first plane (particles, steps)."),
    ], "Subtract the best-fitting plane (sample tilt)."),
    StepSpec("poly", "Polynomial background", "gray", s_poly, [
        Param("order", "Order", "int", 2, 1, 6, 1, 0),
    ], "Subtract a smooth 2D polynomial (bow, scanner curvature)."),
    StepSpec("align_rows", "Align rows", "gray", s_align_rows, [
        Param("method", "Method", "choice", "Median",
              choices=("Median", "Mean", "Median of differences", "Linear (per line)")),
    ], "Remove offsets between scan lines (horizontal streaks)."),
    StepSpec("scars", "Remove scars", "gray", s_scars, [
        Param("threshold", "Threshold (σ)", "float", 4.0, 1.0, 20.0, 0.5, 1),
        Param("max_length", "Max length (px, 0 = any)", "int", 0, 0, 2000, 1, 0),
    ], "Replace short line segments that jump away from both neighbouring lines."),
    StepSpec("background", "Remove background (high-pass)", "gray", s_background, [
        Param("sigma", "Sigma (px)", "float", 20.0, 1.0, 300.0, 1.0, 1),
    ], "Subtract a smoothed copy: removes features larger than ~sigma."),
    StepSpec("gaussian", "Gaussian smoothing", "gray", s_gaussian, [
        Param("sigma", "Sigma (px)", "float", 1.0, 0.2, 20.0, 0.1, 1),
    ], "Reduce noise."),
    StepSpec("median", "Median filter", "gray", s_median, [
        Param("size", "Size (px)", "int", 3, 2, 15, 1, 0),
    ], "Remove spikes while keeping edges."),
    StepSpec("sharpen", "Sharpen", "gray", s_sharpen, [
        Param("strength", "Strength", "float", 1.0, 0.0, 5.0, 0.1, 1),
        Param("sigma", "Radius (px)", "float", 1.0, 0.3, 10.0, 0.1, 1),
    ], "Unsharp mask."),
    StepSpec("dog", "Difference of Gaussians", "gray", s_dog, [
        Param("sigma", "Sigma (px)", "float", 1.0, 0.3, 20.0, 0.1, 1),
        Param("ratio", "Ratio", "float", 1.6, 1.1, 10.0, 0.1, 1),
    ], "Band-pass filter: emphasises features of size ~sigma."),
    StepSpec("clip", "Clip outliers", "gray", s_clip, [
        Param("low", "Low (%)", "float", 0.5, 0.0, 20.0, 0.1, 1),
        Param("high", "High (%)", "float", 0.5, 0.0, 20.0, 0.1, 1),
    ], "Limit values to percentiles (removes spikes from the data)."),
    StepSpec("zero", "Shift to zero", "gray", s_zero, [
        Param("reference", "Set to zero", "choice", "Minimum", choices=("Minimum", "Median", "Mean")),
    ], "Offset the data, e.g. so heights start at 0."),
    StepSpec("invert", "Invert", "gray", s_invert, [], "Flip the sign of the data."),
    FOURIER_STEP,
]}

DEFAULT_PIPELINE = [("plane", False, {}), ("align_rows", False, {"method": "Median"})]


# =============================================================================
# Panel
# =============================================================================

class ProcessingPanel(QWidget):
    """Step cards with add / remove / reorder / enable, save / load and 'apply to all'."""
    changed = pyqtSignal()                  # debounced
    applyAllRequested = pyqtSignal()
    fourierEditRequested = pyqtSignal(object)   # FourierCard whose filter should be edited
    closeRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cards: list[StepCard] = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self.changed)

        title = QLabel("<b>Image processing</b>")
        close = QToolButton()
        close.setText("✕")
        close.setToolTip("Close the processing panel (the steps stay active)")
        close.clicked.connect(self.closeRequested)
        head = QHBoxLayout()
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(close)
        info = QLabel("Steps run top to bottom. Tick a step to apply it, ▲▼ to reorder.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")

        container = QWidget()
        self.cards_layout = QVBoxLayout(container)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

        add = QToolButton()
        add.setText("＋ Add step")
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(add)
        for key, spec in SPM_STEPS.items():
            act = menu.addAction(spec.title)
            act.setToolTip(spec.description)
            act.triggered.connect(lambda _=False, k=key: self.add_step(k))
        menu.setToolTipsVisible(True)
        add.setMenu(menu)
        row1 = QHBoxLayout()
        row1.addWidget(add)
        row1.addStretch(1)
        for text, tip, slot in (("Load…", "Load a processing pipeline", self._load_dialog),
                                ("Save…", "Save the processing pipeline", self._save_dialog),
                                ("Reset", "Back to the default steps (all off)", self.reset)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row1.addWidget(b)
        apply_all = QPushButton("Apply to all images")
        apply_all.setToolTip("Process every loaded image with these steps")
        apply_all.clicked.connect(self.applyAllRequested)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(head)
        lay.addWidget(info)
        lay.addWidget(scroll, 1)
        lay.addLayout(row1)
        lay.addWidget(apply_all)
        self.reset(emit=False)

    # ------------------------------------------------------------------ editing
    def add_step(self, key, params=None, enabled=True, emit=True):
        if key == "fourier":
            card = FourierCard(SPM_STEPS[key], params, enabled)
            card.edit_requested.connect(self.fourierEditRequested)
        else:
            card = StepCard(SPM_STEPS[key], params, enabled)
        card.changed.connect(self._schedule)
        card.remove_requested.connect(self._remove)
        card.move_requested.connect(self._move)
        self.cards_layout.insertWidget(len(self.cards), card)
        self.cards.append(card)
        if emit:
            self._schedule()
            if key == "fourier":            # a new Fourier step is edited right away
                self.fourierEditRequested.emit(card)
        return card

    def _remove(self, card):
        self.cards.remove(card)
        self.cards_layout.removeWidget(card)
        card.setParent(None)
        card.deleteLater()
        self._schedule()

    def _move(self, card, delta):
        i = self.cards.index(card)
        j = i + delta
        if not 0 <= j < len(self.cards):
            return
        self.cards.insert(j, self.cards.pop(i))
        self.cards_layout.removeWidget(card)
        self.cards_layout.insertWidget(j, card)
        self._schedule()

    def clear(self):
        for card in self.cards:
            self.cards_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self.cards = []

    def reset(self, emit=True):
        self.clear()
        for key, enabled, params in DEFAULT_PIPELINE:
            self.add_step(key, params, enabled, emit=False)
        if emit:
            self._schedule()

    def _schedule(self):
        self._timer.start()

    # ------------------------------------------------------------------ state
    def steps(self):
        """[(StepSpec, params, enabled)] for run_pipeline."""
        return [(c.spec, c.values(), c.is_enabled()) for c in self.cards]

    def active(self):
        return [c for c in self.cards if c.is_enabled()]

    def summary(self):
        """Short text of the enabled steps, e.g. for metadata."""
        parts = []
        for c in self.active():
            vals = ", ".join(f"{k}={len(v)} regions" if k == "shapes" else f"{k}={v}" for k, v in c.values().items())
            parts.append(f"{c.spec.title}" + (f" ({vals})" if vals else ""))
        return "; ".join(parts)

    def run(self, data, cache=None):
        """Process `data`; step errors are shown on their cards."""
        result, errors = run_pipeline(np.asarray(data, dtype=float), self.steps(), cache)
        for i, card in enumerate(self.cards):
            card.set_error(errors.get(i))
        return result

    def index_of(self, card):
        return self.cards.index(card)

    def to_dict(self):
        steps = []
        for c in self.cards:
            d = c.to_dict()
            if "shapes" in d["params"]:              # tuples -> JSON lists
                d["params"] = dict(d["params"], shapes=[list(s) if s[0] != "poly" else ["poly", [list(p) for p in s[1]]]
                                                        for s in d["params"]["shapes"]])
            steps.append(d)
        return {"format": PIPELINE_FORMAT, "version": 1, "steps": steps}

    def set_dict(self, data):
        steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(steps, list):
            raise ValueError("Not a processing pipeline file.")
        unknown = [s.get("type") for s in steps if s.get("type") not in SPM_STEPS]
        self.clear()
        for s in steps:
            if s.get("type") in SPM_STEPS:
                self.add_step(s["type"], s.get("params") or {}, bool(s.get("enabled", True)), emit=False)
        self._schedule()
        return unknown

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save processing", "spm_processing.json", "JSON files (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2)

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load processing", "", "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                unknown = self.set_dict(json.load(fh))
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        if unknown:
            QMessageBox.warning(self, "Loaded with warnings", "Skipped unknown steps: " + ", ".join(map(str, unknown)))
