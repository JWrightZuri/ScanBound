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

from .fourier_filter import FOURIER_STEP, FourierCard  # noqa: F401
from .image_adapt_dialog import Param, StepCard, StepSpec, run_pipeline

PIPELINE_FORMAT = "scanbound_spm_pipeline"


# =============================================================================
# Steps (data in physical units, shape preserved): defined in scanbound.steps.spm
# =============================================================================

from scanbound.steps.spm import (  # noqa: E402,F401
    SPM_STEPS, _plane_fit, s_align_rows, s_background, s_clip, s_dog, s_gaussian, s_invert, s_median, s_plane,
    s_poly, s_scars, s_sharpen, s_zero,
)
from ..theme import mark  # noqa: E402
from ..qtutil import save_file_name

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
        mark(info, "muted")

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
        def key(s):  # noqa: ANN001,ANN202 - pipeline files say "type", saved chains say "key"
            return s.get("type", s.get("key"))
        unknown = [key(s) for s in steps if key(s) not in SPM_STEPS]
        self.clear()
        for s in steps:
            if key(s) in SPM_STEPS:
                self.add_step(key(s), s.get("params") or {}, bool(s.get("enabled", True)), emit=False)
        self._schedule()
        return unknown

    def _save_dialog(self):
        path = save_file_name(self, "Save processing", "spm_processing.json", "JSON files (*.json)")
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
