"""
label_assignment_panel.py
=========================

"Labels" tab of the image adaptation dialog: decides which label each segmented object gets
(see core/object_labeling.py for the modes).  The panel owns the configuration, computes the
assignment from the object measurements and emits ``assignmentChanged`` so the dialog can
recolour its preview.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                             QGridLayout, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
                             QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QStackedWidget,
                             QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from . import _ol as ol
from ..theme import mark
from ..qtutil import save_file_name

SKIP_TEXT = "Skip (no ROI)"
NO_LABEL_TEXT = "(no label)"
RULES_FILE_FORMAT = "scanbound_label_rules"
ANY = -1.0                                   # spin box value meaning "no bound"
MODES = [("single", "One label for all objects"),
         ("rules", "Rules: thresholds on shape measures"),
         ("bins", "Classes: split one measure into ranges"),
         ("clusters", "Automatic groups (clustering)")]
_FALLBACK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


class LabelCombo(QComboBox):
    """Editable label chooser: the label list, typed new names, or 'Skip (no ROI)' (``none_text``: what None
    means here, e.g. '(no label)' where None keeps the objects without a label)."""

    def __init__(self, labels, value=None, parent=None, none_text=SKIP_TEXT):
        super().__init__(parent)
        self.none_text = none_text
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.addItems(list(labels) + [none_text])
        self.setMinimumContentsLength(10)
        self.set_value(value)

    def value(self):
        text = self.currentText().strip()
        return None if not text or text == self.none_text else text

    def set_value(self, value):
        self.blockSignals(True)
        self.setCurrentText(self.none_text if value is None else str(value))
        self.blockSignals(False)


def _bound_spin(value=None, tip=""):
    s = QDoubleSpinBox()
    s.setRange(ANY, 1e12)
    s.setDecimals(3)
    s.setSpecialValueText("any")
    s.setKeyboardTracking(False)
    s.setValue(ANY if value is None else float(value))
    s.setToolTip(tip)
    s.setMinimumWidth(80)
    return s


def _spin_value(s):
    return None if s.value() <= ANY else float(s.value())


class FeatureCombo(QComboBox):
    def __init__(self, px_scale, value="area", parent=None):
        super().__init__(parent)
        for key in ol.FEATURES:
            self.addItem(ol.feature_title(key, px_scale), key)
            self.setItemData(self.count() - 1, ol.FEATURE_HELP.get(key, ""), Qt.ItemDataRole.ToolTipRole)
        self.set_value(value)

    def value(self):
        return self.currentData()

    def set_value(self, key):
        i = self.findData(key)
        self.setCurrentIndex(i if i >= 0 else 0)


# =============================================================================
# Rules page
# =============================================================================

class ConditionRow(QWidget):
    changed = pyqtSignal()
    removed = pyqtSignal(object)

    def __init__(self, px_scale, cond=None, parent=None):
        super().__init__(parent)
        cond = cond or {}
        self.feature = FeatureCombo(px_scale, cond.get("feature", "area"))
        self.lo = _bound_spin(cond.get("min"), "Lower bound (inclusive); 'any' = no bound")
        self.hi = _bound_spin(cond.get("max"), "Upper bound (inclusive); 'any' = no bound")
        rm = QToolButton()
        rm.setText("✕")
        rm.setToolTip("Remove condition")
        rm.clicked.connect(lambda: self.removed.emit(self))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 0, 0)
        lay.addWidget(self.feature, 2)
        lay.addWidget(QLabel("from"))
        lay.addWidget(self.lo, 1)
        lay.addWidget(QLabel("to"))
        lay.addWidget(self.hi, 1)
        lay.addWidget(rm)
        self.feature.currentIndexChanged.connect(self.changed)
        self.lo.valueChanged.connect(self.changed)
        self.hi.valueChanged.connect(self.changed)

    def to_dict(self):
        return {"feature": self.feature.value(), "min": _spin_value(self.lo), "max": _spin_value(self.hi)}


class RuleCard(QFrame):
    changed = pyqtSignal()
    removed = pyqtSignal(object)
    moved = pyqtSignal(object, int)

    def __init__(self, labels, px_scale, rule=None, features=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._px = px_scale
        self._features = features
        rule = rule or {}
        self.title = QLabel()
        self.label = LabelCombo(labels, rule.get("label"))
        self.label.currentTextChanged.connect(self.changed)
        add = QPushButton("+ condition")
        add.setAutoDefault(False)
        add.clicked.connect(lambda: self.add_condition())
        head = QHBoxLayout()
        head.addWidget(self.title)
        head.addWidget(self.label, 1)
        head.addWidget(add)
        for text, tip, slot in (("▲", "Move up (earlier rules win)", lambda: self.moved.emit(self, -1)),
                                ("▼", "Move down", lambda: self.moved.emit(self, 1)),
                                ("✕", "Remove rule", lambda: self.removed.emit(self))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            head.addWidget(b)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(6, 4, 6, 4)
        self.body.setSpacing(2)
        self.body.addLayout(head)
        self.rows = []
        for cond in rule.get("conditions") or []:
            self.add_condition(cond, emit=False)

    def set_number(self, n):
        self.title.setText(f"<b>Rule {n}</b> → label:")

    def add_condition(self, cond=None, emit=True):
        if cond is None:
            # start from the median of the measure so the condition does something visible
            feature = "area"
            med = None
            if self._features is not None and not self._features.empty:
                med = float(np.nanmedian(self._features[feature]))
            cond = {"feature": feature, "min": med, "max": None}
        row = ConditionRow(self._px, cond)
        row.changed.connect(self.changed)
        row.removed.connect(self._remove_row)
        self.rows.append(row)
        self.body.addWidget(row)
        if emit:
            self.changed.emit()

    def _remove_row(self, row):
        self.rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self.changed.emit()

    def to_dict(self):
        return {"label": self.label.value(), "conditions": [r.to_dict() for r in self.rows]}


class RulesPage(QWidget):
    changed = pyqtSignal()

    def __init__(self, labels, px_scale, parent=None):
        super().__init__(parent)
        self._labels, self._px = labels, px_scale
        self.features = None
        self.cards = []
        info = QLabel("Rules are checked from top to bottom; an object gets the label of the first rule whose "
                      "conditions all hold. A rule without conditions matches everything.")
        info.setWordWrap(True)
        mark(info, "muted")
        container = QWidget()
        self.cards_layout = QVBoxLayout(container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        add = QPushButton("+ Add rule")
        add.setAutoDefault(False)
        add.clicked.connect(lambda: self.add_rule())
        self.fallback = LabelCombo(labels, None)
        self.fallback.currentTextChanged.connect(self.changed)
        bottom = QHBoxLayout()
        bottom.addWidget(add)
        bottom.addStretch(1)
        bottom.addWidget(QLabel("Objects matching no rule:"))
        bottom.addWidget(self.fallback)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(info)
        lay.addWidget(scroll, 1)
        lay.addLayout(bottom)

    def add_rule(self, rule=None, emit=True):
        if rule is None:
            label = self._labels[len(self.cards) % len(self._labels)] if self._labels else None
            rule = {"label": label, "conditions": []}
            new = True
        else:
            new = False
        card = RuleCard(self._labels, self._px, rule, self.features)
        if new:
            card.add_condition(emit=False)
        card.changed.connect(self.changed)
        card.removed.connect(self._remove)
        card.moved.connect(self._move)
        self.cards.append(card)
        self.cards_layout.insertWidget(len(self.cards) - 1, card)
        self._renumber()
        if emit:
            self.changed.emit()

    def _remove(self, card):
        self.cards.remove(card)
        card.setParent(None)
        card.deleteLater()
        self._renumber()
        self.changed.emit()

    def _move(self, card, delta):
        i = self.cards.index(card)
        j = i + delta
        if not 0 <= j < len(self.cards):
            return
        self.cards.insert(j, self.cards.pop(i))
        self.cards_layout.removeWidget(card)
        self.cards_layout.insertWidget(j, card)
        self._renumber()
        self.changed.emit()

    def _renumber(self):
        for n, card in enumerate(self.cards, 1):
            card.set_number(n)

    def set_rules(self, rules, fallback):
        for card in list(self.cards):
            card.setParent(None)
            card.deleteLater()
        self.cards = []
        for rule in rules or []:
            self.add_rule(rule, emit=False)
        self.fallback.set_value(fallback)

    def rules(self):
        return [c.to_dict() for c in self.cards]


# =============================================================================
# Classes (bins) and clusters pages
# =============================================================================

class BinsPage(QWidget):
    changed = pyqtSignal()

    def __init__(self, labels, px_scale, parent=None):
        super().__init__(parent)
        self._labels, self._px = labels, px_scale
        self.features = None
        self.feature = FeatureCombo(px_scale, "area")
        self.n = QSpinBox()
        self.n.setRange(2, 10)
        self.n.setValue(2)
        self.log = QCheckBox("log scale")
        self.log.setChecked(True)
        self.log.setToolTip("Suggest thresholds on a logarithmic scale (sizes spanning orders of magnitude)")
        self.method = QComboBox()
        self.method.addItems(["natural breaks", "quantiles"])
        self.method.setToolTip("natural breaks: thresholds between the natural groups of values (1D k-means)\n"
                               "quantiles: classes with equal numbers of objects")
        suggest = QPushButton("Suggest thresholds")
        suggest.setAutoDefault(False)
        suggest.clicked.connect(self.suggest)
        top = QGridLayout()
        top.addWidget(QLabel("Measure:"), 0, 0)
        top.addWidget(self.feature, 0, 1, 1, 3)
        top.addWidget(QLabel("Classes:"), 1, 0)
        top.addWidget(self.n, 1, 1)
        top.addWidget(self.method, 1, 2)
        top.addWidget(self.log, 1, 3)
        top.addWidget(suggest, 2, 0, 1, 4)
        self.grid = QGridLayout()
        self.grid.setColumnStretch(3, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addLayout(self.grid)
        lay.addStretch(1)
        self.thresholds, self.label_combos, self.count_labels, self.range_labels = [], [], [], []
        self._build(2, [], [])
        self.feature.currentIndexChanged.connect(lambda: self.suggest())
        self.n.valueChanged.connect(self._on_n)

    def _clear(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _build(self, n, thresholds, labels):
        self._clear()
        thresholds = sorted(thresholds)[:n - 1]
        unit = ol.feature_unit(self.feature.value(), self._px)
        self.thresholds, self.label_combos, self.count_labels, self.range_labels = [], [], [], []
        for i in range(n):
            lab = labels[i] if i < len(labels) else (self._labels[i % len(self._labels)] if self._labels else None)
            rng = QLabel()
            combo = LabelCombo(self._labels, lab)
            combo.currentTextChanged.connect(self.changed)
            count = QLabel("")
            mark(count, "muted")
            self.grid.addWidget(QLabel(f"Class {i + 1}"), i, 0)
            self.grid.addWidget(rng, i, 1)
            if i < n - 1:
                spin = QDoubleSpinBox()
                spin.setRange(-1e12, 1e12)
                spin.setDecimals(3)
                spin.setKeyboardTracking(False)
                spin.setSuffix(f" {unit}" if unit else "")
                spin.setToolTip("Upper limit of this class = lower limit of the next")
                spin.setValue(thresholds[i] if i < len(thresholds) else 0.0)
                spin.valueChanged.connect(self._on_threshold)
                self.thresholds.append(spin)
                self.grid.addWidget(spin, i, 2)
            else:
                self.grid.addWidget(QLabel("and above"), i, 2)
            self.grid.addWidget(combo, i, 3)
            self.grid.addWidget(count, i, 4)
            self.label_combos.append(combo)
            self.count_labels.append(count)
            self.range_labels.append(rng)
        self._update_ranges()

    def _update_ranges(self):
        values = [s.value() for s in self.thresholds]
        for i, lab in enumerate(self.range_labels):
            lab.setText("below" if i == 0 else f"from {values[i - 1]:.4g} to")
            if i == len(self.range_labels) - 1 and i > 0:
                lab.setText(f"from {values[i - 1]:.4g}")

    def _on_threshold(self, _v=None):
        self._update_ranges()
        self.changed.emit()

    def _on_n(self, n):
        labels = [c.value() for c in self.label_combos]
        self._build(n, [s.value() for s in self.thresholds], labels)
        self.suggest()

    def suggest(self):
        if self.features is None or self.features.empty:
            self._build(self.n.value(), [s.value() for s in self.thresholds], [c.value() for c in self.label_combos])
            self.changed.emit()
            return
        feature = self.feature.value()
        cuts = ol.suggest_thresholds(self.features[feature], self.n.value(), self.method.currentText(),
                                     self.log.isChecked() and feature in ol.SCALE_FEATURES)
        labels = [c.value() for c in self.label_combos]
        self._build(self.n.value(), cuts, labels)
        self.changed.emit()

    def set_counts(self, counts):
        for i, lab in enumerate(self.count_labels):
            lab.setText(f"{counts.get(i, 0)} objects")

    def to_dict(self):
        return {"feature": self.feature.value(), "thresholds": [s.value() for s in self.thresholds],
                "labels": [c.value() for c in self.label_combos], "log": self.log.isChecked(),
                "method": self.method.currentText()}

    def set_dict(self, d):
        for w in (self.feature, self.n, self.log, self.method):
            w.blockSignals(True)
        self.feature.set_value(d.get("feature", "area"))
        n = len(d.get("thresholds") or []) + 1
        self.n.setValue(max(2, n))
        self.log.setChecked(bool(d.get("log", True)))
        if d.get("method"):
            self.method.setCurrentText(d["method"])
        for w in (self.feature, self.n, self.log, self.method):
            w.blockSignals(False)
        self._build(max(2, n), d.get("thresholds") or [], d.get("labels") or [])


class ClustersPage(QWidget):
    changed = pyqtSignal()

    def __init__(self, labels, px_scale, parent=None):
        super().__init__(parent)
        self._labels, self._px = labels, px_scale
        info = QLabel("Objects are grouped by similarity of the ticked measures (k-means on standardised "
                      "values). Groups are numbered by increasing value of the first ticked measure.")
        info.setWordWrap(True)
        mark(info, "muted")
        self.feature_list = QListWidget()
        self.feature_list.setMaximumHeight(120)
        for key in ol.FEATURES:
            item = QListWidgetItem(ol.feature_title(key, px_scale))
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(ol.FEATURE_HELP.get(key, ""))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if key in ("area", "aspect_ratio") else Qt.CheckState.Unchecked)
            self.feature_list.addItem(item)
        self.feature_list.itemChanged.connect(lambda _: self.changed.emit())
        self.k = QSpinBox()
        self.k.setRange(2, 10)
        self.k.setValue(2)
        self.k.valueChanged.connect(self._on_k)
        self.log = QCheckBox("log-scale size measures")
        self.log.setChecked(True)
        self.log.toggled.connect(self.changed)
        row = QHBoxLayout()
        row.addWidget(QLabel("Groups:"))
        row.addWidget(self.k)
        row.addWidget(self.log)
        row.addStretch(1)
        self.grid = QGridLayout()
        self.grid.setColumnStretch(2, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(info)
        lay.addWidget(self.feature_list)
        lay.addLayout(row)
        lay.addLayout(self.grid)
        lay.addStretch(1)
        self.label_combos, self.summaries = [], []
        self._build(2, [])

    def _on_k(self, k):
        self._build(k, [c.value() for c in self.label_combos])
        self.changed.emit()

    def _build(self, k, labels):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.label_combos, self.summaries = [], []
        for i in range(k):
            lab = labels[i] if i < len(labels) else (self._labels[i % len(self._labels)] if self._labels else None)
            combo = LabelCombo(self._labels, lab)
            combo.currentTextChanged.connect(self.changed)
            summary = QLabel("")
            mark(summary, "muted")
            summary.setWordWrap(True)
            self.grid.addWidget(QLabel(f"Group {i + 1}"), i, 0)
            self.grid.addWidget(combo, i, 1)
            self.grid.addWidget(summary, i, 2)
            self.label_combos.append(combo)
            self.summaries.append(summary)

    def selected_features(self):
        return [self.feature_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.feature_list.count())
                if self.feature_list.item(i).checkState() == Qt.CheckState.Checked]

    def set_summaries(self, texts):
        for i, lab in enumerate(self.summaries):
            lab.setText(texts.get(i, "no objects"))

    def to_dict(self):
        return {"features": self.selected_features(), "k": self.k.value(),
                "labels": [c.value() for c in self.label_combos], "log": self.log.isChecked()}

    def set_dict(self, d):
        self.feature_list.blockSignals(True)
        wanted = set(d.get("features") or ["area", "aspect_ratio"])
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in wanted
                               else Qt.CheckState.Unchecked)
        self.feature_list.blockSignals(False)
        self.k.blockSignals(True)
        self.k.setValue(int(d.get("k", 2)))
        self.k.blockSignals(False)
        self.log.blockSignals(True)
        self.log.setChecked(bool(d.get("log", True)))
        self.log.blockSignals(False)
        self._build(int(d.get("k", 2)), d.get("labels") or [])


# =============================================================================
# Panel
# =============================================================================

class LabelAssignmentPanel(QWidget):
    assignmentChanged = pyqtSignal()
    featuresNeeded = pyqtSignal()      # the mode needs measurements the panel does not have yet

    def __init__(self, labels=None, colors=None, px_scale=None, default_label=None, parent=None):
        super().__init__(parent)
        self.labels = [str(l) for l in (labels or [])]
        if default_label and default_label not in self.labels:
            self.labels.insert(0, default_label)
        self.colors = dict(colors or {})
        self.px_scale = px_scale
        self.features = pd.DataFrame()
        self.assignment = {}           # object id -> label or None
        self.groups = pd.Series(dtype=int)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.recompute)

        self.mode = QComboBox()
        for key, text in MODES:
            self.mode.addItem(text, key)
        self.mode.currentIndexChanged.connect(self._on_mode)
        # one label for all: the label chosen in the main window, else none (never a label of earlier work)
        self.single = LabelCombo(self.labels, default_label, none_text=NO_LABEL_TEXT)
        self.default_label = default_label
        self.single.currentTextChanged.connect(self._schedule)
        single_page = QWidget()
        sl = QHBoxLayout(single_page)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("Label:"))
        sl.addWidget(self.single, 1)
        sl.addStretch(1)
        self.rules = RulesPage(self.labels, px_scale)
        self.bins = BinsPage(self.labels, px_scale)
        self.clusters = ClustersPage(self.labels, px_scale)
        for page in (self.rules, self.bins, self.clusters):
            page.changed.connect(self._schedule)
        self.pages = QStackedWidget()
        for page in (single_page, self.rules, self.bins, self.clusters):
            self.pages.addWidget(page)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)

        # histogram + objects table
        self.plot_feature = FeatureCombo(px_scale, "area")
        self.plot_feature.currentIndexChanged.connect(self._draw_histogram)
        self.plot_log = QCheckBox("log x")
        self.plot_log.setChecked(True)
        self.plot_log.toggled.connect(self._draw_histogram)
        # fixed margins + one persistent axes: constrained layout re-measures every tick label on
        # each redraw and was the slowest part of a parameter change
        self.figure = Figure(figsize=(4, 1.6))
        self.figure.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.96)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(110)
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.setInterval(150)
        self._hist_timer.timeout.connect(self._draw_histogram_now)
        hist_tab = QWidget()
        hl = QVBoxLayout(hist_tab)
        hl.setContentsMargins(0, 0, 0, 0)
        hrow = QHBoxLayout()
        hrow.addWidget(QLabel("Measure:"))
        hrow.addWidget(self.plot_feature, 1)
        hrow.addWidget(self.plot_log)
        hl.addLayout(hrow)
        hl.addWidget(self.canvas, 1)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(hist_tab, "Histogram")
        self.view_tabs.addTab(self.table, "Objects (IDs as in preview)")
        self.view_tabs.currentChanged.connect(lambda _: self._refresh_views())

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Assign labels by:"))
        mrow.addWidget(self.mode, 1)
        for text, tip, slot in (("Save rules…", "Save the label assignment (all modes) to a JSON file",
                                 self._save_dialog),
                                ("Load rules…", "Load a label assignment JSON (or the labels of a saved pipeline)",
                                 self._load_dialog)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            mrow.addWidget(b)
        lay.addLayout(mrow)
        lay.addWidget(self.pages, 3)
        lay.addWidget(self.summary)
        lay.addWidget(self.view_tabs, 2)
        if px_scale is None:
            note = QLabel("No pixel scale for this image: sizes are in pixels.")
            mark(note, "warn")
            lay.addWidget(note)

    # ------------------------------------------------------------------ config
    def mode_key(self):
        return self.mode.currentData()

    def needs_features(self):
        return self.mode_key() != "single" or self.isVisible()

    def config(self):
        return {"mode": self.mode_key(), "single_label": self.single.value(),
                "rules": self.rules.rules(), "fallback": self.rules.fallback.value(),
                "bins": self.bins.to_dict(), "clusters": self.clusters.to_dict()}

    def set_config(self, cfg):
        if not isinstance(cfg, dict):
            return
        single = cfg.get("single_label", self.single.value())
        if single is not None and single not in self.labels:     # saved with a label this project no longer has
            single = self.default_label
        self.single.set_value(single)
        self.rules.set_rules(cfg.get("rules") or [], cfg.get("fallback"))
        if isinstance(cfg.get("bins"), dict):
            self.bins.set_dict(cfg["bins"])
        if isinstance(cfg.get("clusters"), dict):
            self.clusters.set_dict(cfg["clusters"])
        i = self.mode.findData(cfg.get("mode", "single"))
        self.mode.setCurrentIndex(max(i, 0))
        self._on_mode()

    # ------------------------------------------------------------------ files
    def save_rules(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"format": RULES_FILE_FORMAT, "version": 1, "labeling": self.config()}, fh, indent=2)

    def load_rules(self, path):
        """Load a rules file, or the "labeling" part of a saved image adaptation pipeline."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = data.get("labeling") if isinstance(data, dict) else None
        if not isinstance(cfg, dict) or "mode" not in cfg:
            raise ValueError("No label rules found in this file.")
        self.set_config(cfg)
        missing = sorted({l for l in self._config_labels(cfg) if l not in self.labels})
        return missing

    @staticmethod
    def _config_labels(cfg):
        labels = [cfg.get("single_label"), cfg.get("fallback")]
        labels += [r.get("label") for r in cfg.get("rules") or []]
        labels += list((cfg.get("bins") or {}).get("labels") or [])
        labels += list((cfg.get("clusters") or {}).get("labels") or [])
        return [l for l in labels if l]

    def _save_dialog(self):
        path = save_file_name(self, "Save label rules", "label_rules.json", "JSON files (*.json)")
        if not path:
            return
        try:
            self.save_rules(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load label rules", "", "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            missing = self.load_rules(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        if missing:
            QMessageBox.information(self, "New labels",
                                    "These labels are not in the current label list and will be added when the "
                                    "ROIs are applied:\n" + "\n".join(missing))

    # ------------------------------------------------------------------ data
    def set_features(self, features: pd.DataFrame):
        self.features = features if features is not None else pd.DataFrame()
        self.rules.features = self.features
        first = self.bins.features is None or self.bins.features.empty
        self.bins.features = self.features
        if first and not self.features.empty and not any(s.value() for s in self.bins.thresholds):
            self.bins.suggest()          # first data: sensible starting thresholds
        self.recompute()

    def _on_mode(self, *_):
        self.pages.setCurrentIndex(max(self.mode.currentIndex(), 0))
        if self.mode_key() == "rules" and not self.rules.cards:
            self.rules.add_rule(emit=False)
        if self.mode_key() == "bins":
            self.plot_feature.set_value(self.bins.feature.value())
        self._schedule()

    def _schedule(self, *_):
        self._timer.start()

    def recompute(self):
        self._timer.stop()
        cfg = self.config()
        if (cfg["mode"] != "single" and self.features is not None and len(self.features)
                and "area" not in self.features.columns):
            self.featuresNeeded.emit()
            return
        if self.features is None or self.features.empty:
            self.assignment, self.groups = {}, pd.Series(dtype=int)
        else:
            try:
                groups, labels = ol.assign_groups(self.features, cfg)
            except Exception as exc:                 # e.g. too few objects for k groups
                self.summary.setText(f"<span style='color:#a00'>Assignment failed: {exc}</span>")
                return
            self.groups = groups
            self.assignment = {oid: (labels[g] if 0 <= g < len(labels) else None) for oid, g in groups.items()}
            self._update_page_info(groups, labels)
        self._update_summary()
        self._refresh_views()
        self.assignmentChanged.emit()

    def _update_page_info(self, groups, labels):
        counts = groups.value_counts().to_dict()
        mode = self.mode_key()
        if mode == "bins":
            self.bins.set_counts(counts)
            if self.plot_feature.value() != self.bins.feature.value():
                self.plot_feature.set_value(self.bins.feature.value())
        elif mode == "clusters":
            feats = self.clusters.selected_features()
            texts = {}
            for g in range(len(labels)):
                sel = self.features[groups == g]
                if sel.empty:
                    continue
                parts = [f"n={len(sel)}"]
                for f in feats[:3]:
                    unit = ol.feature_unit(f, self.px_scale)
                    parts.append(f"{ol.FEATURES[f][0].split(' (')[0].lower()} {sel[f].mean():.3g}{(' ' + unit) if unit and unit != '°' else ''}")
                texts[g] = ", ".join(parts)
            self.clusters.set_summaries(texts)

    def color(self, label):
        if label in self.colors:
            return self.colors[label]
        extra = sorted({l for l in self.assignment.values() if l and l not in self.colors})
        return _FALLBACK_COLORS[extra.index(label) % len(_FALLBACK_COLORS)] if label in extra else "#888888"

    def counts(self):
        out = {}
        for lab in self.assignment.values():
            out[lab] = out.get(lab, 0) + 1
        return out

    def _update_summary(self):
        counts = self.counts()
        if not counts:
            self.summary.setText("<i>No objects yet (the pipeline needs a threshold step).</i>")
            return
        parts = []
        for lab, n in sorted(counts.items(), key=lambda kv: (kv[0] is None, -kv[1])):
            if lab is None:
                parts.append(f"<span style='color:#777'>skipped: {n}</span>")
            else:
                new = "" if lab in self.labels else " <i>(new label)</i>"
                parts.append(f"<span style='color:{self.color(lab)}'>■</span> <b>{lab}</b>: {n}{new}")
        self.summary.setText(" &nbsp; ".join(parts))

    # ------------------------------------------------------------------ views
    def _refresh_views(self):
        if not self.isVisible():
            self._views_stale = True        # redrawn when the tab is shown
            return
        self._views_stale = False
        if self.view_tabs.currentIndex() == 0:
            self._draw_histogram()
        else:
            self._fill_table()

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_views_stale", True):
            self._refresh_views()

    def _draw_histogram(self, *_):
        """Debounced: the plot follows after a short pause in the parameter changes."""
        self._hist_timer.start()

    def _draw_histogram_now(self):
        ax = self.ax
        ax.clear()
        ax.set_axis_on()
        feature = self.plot_feature.value()
        if self.features is None or self.features.empty or feature not in self.features:
            ax.text(0.5, 0.5, "no objects", ha="center", va="center", transform=ax.transAxes, color="#777")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        values = self.features[feature]
        log = self.plot_log.isChecked() and feature in ol.SCALE_FEATURES and (values > 0).all()
        finite = values[np.isfinite(values)]
        if finite.empty:
            self.canvas.draw_idle()
            return
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1
        edges = np.geomspace(lo, hi, 31) if log else np.linspace(lo, hi, 31)
        groups = {}
        for oid, v in finite.items():
            groups.setdefault(self.assignment.get(oid), []).append(v)
        order = sorted(groups, key=lambda l: (l is None, str(l)))
        ax.hist([groups[l] for l in order], bins=edges, stacked=True,
                color=[self.color(l) if l is not None else "#bbbbbb" for l in order],
                label=[l if l is not None else "skipped" for l in order])
        if log:
            ax.set_xscale("log")
        if self.mode_key() == "bins" and feature == self.bins.feature.value():
            for t in self.bins.to_dict()["thresholds"]:
                ax.axvline(t, color="black", lw=1, ls="--")
        ax.set_xlabel(ol.feature_title(feature, self.px_scale), fontsize=8)
        ax.set_ylabel("objects", fontsize=8)
        ax.tick_params(labelsize=7)
        if len(order) > 1:
            ax.legend(fontsize=7, frameon=False)
        self.canvas.draw_idle()

    def _fill_table(self):
        df = self.features
        t = self.table
        t.setSortingEnabled(False)
        if df is None or df.empty:
            t.setRowCount(0)
            return
        cols = ["ID", "Label"] + [ol.feature_title(k, self.px_scale) for k in ol.FEATURES]
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setRowCount(len(df))
        for r, (oid, row) in enumerate(df.iterrows()):
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole, int(oid))
            t.setItem(r, 0, item)
            lab = self.assignment.get(oid)
            li = QTableWidgetItem("skipped" if lab is None else lab)
            if lab is not None:
                li.setForeground(QColor(self.color(lab)))
            t.setItem(r, 1, li)
            for c, key in enumerate(ol.FEATURES, 2):
                v = row[key]
                cell = QTableWidgetItem()
                cell.setData(Qt.ItemDataRole.DisplayRole, float(f"{v:.4g}") if np.isfinite(v) else None)
                t.setItem(r, c, cell)
        t.setSortingEnabled(True)
