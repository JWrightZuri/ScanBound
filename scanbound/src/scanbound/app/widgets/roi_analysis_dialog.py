"""
roi_analysis_dialog.py
======================

Non-modal window that runs the analyses registered in core/roi_analysis.py on the ROIs of
the current image or of all loaded images, shows every chart and table in tabs and exports
them (CSV + PNG).  New analyses registered with ``@register_analysis`` appear automatically,
with widgets generated from their ``Param`` list.

The window asks a provider for fresh data on every run, so it can stay open while labelling::

    def provider(scope):            # scope: "current" or "all"
        return AnalysisData(annotations_df, {file: ImageInfo(...)}, colors={label: "#hex"})

    dlg = RoiAnalysisDialog(provider, parent=self)
    dlg.show()
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QRadioButton,
    QSpinBox, QSplitter, QTableView, QTabWidget, QVBoxLayout, QWidget,
)

from scanbound.analysis.roi import (
    ANALYSES, EMPTY_LABEL, SYMMETRY_LABEL, AnalysisContext, export_results, run_analyses,
)
from ..theme import mark

DEFAULT_ANALYSES = ("composition", "coverage", "size")


class DataFrameModel(QAbstractTableModel):
    """Read-only table model for a pandas DataFrame (numbers shown with 4 significant digits)."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df.reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return _format(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float, np.number)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        self.layoutAboutToBeChanged.emit()
        self._df = self._df.sort_values(self._df.columns[column],
                                        ascending=order == Qt.SortOrder.AscendingOrder,
                                        kind="mergesort").reset_index(drop=True)
        self.layoutChanged.emit()

    def dataframe(self):
        return self._df


def _format(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{value:.4g}"
    return str(value)


class RoiAnalysisDialog(QDialog):
    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI analysis")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(1300, 820)
        self._provider = provider
        self._results = {}
        self._param_values = {key: {p.name: p.default for p in a.params} for key, a in ANALYSES.items()}
        self._label_checks = {}          # label -> checked, kept across refreshes

        # ---------------------------------------------------------------- controls
        scope_box = QGroupBox("Images")
        self.rb_current = QRadioButton("Current image")
        self.rb_all = QRadioButton("All loaded images")
        self.rb_current.setChecked(True)
        self.cb_border = QCheckBox("Leave out objects touching the image border")
        self.cb_border.setToolTip("ROIs with a point within 1 pixel of the image edge are usually cut off, "
                                  "so their size and shape would bias the statistics.")
        lay = QVBoxLayout(scope_box)
        lay.addWidget(self.rb_current)
        lay.addWidget(self.rb_all)
        lay.addWidget(self.cb_border)
        self.rb_current.toggled.connect(self.refresh_labels)

        label_box = QGroupBox("Labels")
        self.label_list = QListWidget()
        self.label_list.itemChanged.connect(self._on_label_toggled)
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(lambda: self._check_all_labels(True))
        btn_none.clicked.connect(lambda: self._check_all_labels(False))
        row = QHBoxLayout()
        row.addWidget(btn_all)
        row.addWidget(btn_none)
        lay = QVBoxLayout(label_box)
        lay.addWidget(self.label_list)
        lay.addLayout(row)

        analysis_box = QGroupBox("Analyses")
        self.analysis_list = QListWidget()
        for key, analysis in ANALYSES.items():
            item = QListWidgetItem(analysis.title)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(analysis.description)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if key in DEFAULT_ANALYSES else Qt.CheckState.Unchecked)
            self.analysis_list.addItem(item)
        self.analysis_list.currentItemChanged.connect(self._show_params)
        self.description = QLabel()
        self.description.setWordWrap(True)
        mark(self.description, "muted")
        self.param_widget = QWidget()
        self.param_form = QFormLayout(self.param_widget)
        self.param_form.setContentsMargins(0, 0, 0, 0)
        lay = QVBoxLayout(analysis_box)
        lay.addWidget(self.analysis_list)
        lay.addWidget(self.description)
        lay.addWidget(self.param_widget)

        self.btn_run = QPushButton("Run analysis")
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self.run)
        self.btn_export = QPushButton("Export results...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export)
        self.status = QLabel()
        self.status.setWordWrap(True)

        controls = QWidget()
        lay = QVBoxLayout(controls)
        lay.addWidget(scope_box)
        lay.addWidget(label_box, 2)
        lay.addWidget(analysis_box, 3)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.btn_export)
        lay.addWidget(self.status)
        controls.setMinimumWidth(280)
        controls.setMaximumWidth(380)

        # ---------------------------------------------------------------- results
        self.tabs = QTabWidget()
        placeholder = QLabel("Choose the images, labels and analyses, then press \"Run analysis\".")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(placeholder, "Results")

        splitter = QSplitter()
        splitter.addWidget(controls)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(splitter)

        self.analysis_list.setCurrentRow(0)
        self.refresh_labels()

    # ------------------------------------------------------------------ data / labels

    def scope(self):
        return "current" if self.rb_current.isChecked() else "all"

    def _data(self):
        return self._provider(self.scope())

    def refresh_labels(self, *_):
        data = self._data()
        labels = []
        if data is not None and data.annotations is not None and not data.annotations.empty \
                and "label" in data.annotations.columns:
            labels = [str(l) for l in data.annotations["label"].fillna(EMPTY_LABEL).unique()]
        labels.sort(key=lambda l: (l in (EMPTY_LABEL, SYMMETRY_LABEL), l.lower()))
        self.label_list.blockSignals(True)
        self.label_list.clear()
        for label in labels:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = self._label_checks.get(label, label not in (EMPTY_LABEL, SYMMETRY_LABEL))
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.label_list.addItem(item)
        self.label_list.blockSignals(False)
        n_images = len(data.images) if data is not None else 0
        self.status.setText(f"{n_images} image(s), {len(labels)} label(s) in scope.")

    def _on_label_toggled(self, item):
        self._label_checks[item.text()] = item.checkState() == Qt.CheckState.Checked

    def _check_all_labels(self, checked):
        for i in range(self.label_list.count()):
            self.label_list.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def selected_labels(self):
        return [self.label_list.item(i).text() for i in range(self.label_list.count())
                if self.label_list.item(i).checkState() == Qt.CheckState.Checked]

    def selected_analyses(self):
        return [self.analysis_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.analysis_list.count())
                if self.analysis_list.item(i).checkState() == Qt.CheckState.Checked]

    # ------------------------------------------------------------------ parameters

    def _show_params(self, item, _previous=None):
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        if item is None:
            self.description.clear()
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        analysis = ANALYSES[key]
        self.description.setText(analysis.description)
        values = self._param_values[key]
        for p in analysis.params:
            widget = self._param_editor(key, p, values[p.name])
            widget.setToolTip(p.tooltip)
            self.param_form.addRow(p.label, widget)

    def _param_editor(self, key, p, value):
        def store(v, name=p.name):
            self._param_values[key][name] = v

        if p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.minimum if p.minimum is not None else -10**9),
                       int(p.maximum if p.maximum is not None else 10**9))
            w.setValue(int(value))
            w.valueChanged.connect(store)
        elif p.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(p.minimum if p.minimum is not None else -1e9, p.maximum if p.maximum is not None else 1e9)
            w.setValue(float(value))
            w.valueChanged.connect(store)
        elif p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(store)
        elif p.kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in p.choices])
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(store)
        else:
            raise ValueError(f"Unknown parameter kind {p.kind!r}")
        return w

    # ------------------------------------------------------------------ run / show / export

    def run(self):
        keys = self.selected_analyses()
        if not keys:
            self.status.setText("Select at least one analysis.")
            return
        data = self._data()
        if data is None or not data.images:
            self.status.setText("No image loaded.")
            return
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ctx = AnalysisContext(data, labels=self.selected_labels(), exclude_border=self.cb_border.isChecked())
            self._results = run_analyses(ctx, keys, self._param_values)
        finally:
            QGuiApplication.restoreOverrideCursor()
        self._show_results()
        n = len(ctx.objects)
        self.status.setText(f"Analysed {n} ROI(s) with {len(ctx.labels)} label(s) on {len(ctx.files)} image(s) "
                            f"(unit: {ctx.unit}).")
        self.btn_export.setEnabled(True)

    def _show_results(self):
        current = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else None
        self.tabs.clear()
        for key, result in self._results.items():
            page = QWidget()
            lay = QVBoxLayout(page)
            if result.notes:
                notes = QLabel("\n".join(n for n in result.notes if not n.startswith("Traceback")))
                notes.setWordWrap(True)
                mark(notes, "warn")
                lay.addWidget(notes)
            inner = QTabWidget()
            for name, fig in result.figures.items():
                inner.addTab(self._figure_widget(fig), name)
            for name, table in result.tables.items():
                inner.addTab(self._table_widget(table), f"Table: {name}")
            if inner.count():
                lay.addWidget(inner, 1)
            else:
                lay.addStretch(1)
            title = ANALYSES[key].title if key in ANALYSES else key
            self.tabs.addTab(page, title)
            if title == current:
                self.tabs.setCurrentWidget(page)

    @staticmethod
    def _figure_widget(fig):
        w = QWidget()
        lay = QVBoxLayout(w)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumSize(360, 260)         # a canvas squeezed to nothing collapses the figure's layout
        lay.addWidget(NavigationToolbar2QT(canvas, w))
        lay.addWidget(canvas, 1)
        return w

    @staticmethod
    def _table_widget(table):
        w = QWidget()
        lay = QVBoxLayout(w)
        view = QTableView()
        model = DataFrameModel(table, view)
        view.setModel(model)
        view.setSortingEnabled(True)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        view.resizeColumnsToContents()
        copy = QPushButton("Copy table")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(
            model.dataframe().to_csv(sep="\t", index=False)))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(copy)
        lay.addWidget(view, 1)
        lay.addLayout(row)
        return w

    def export(self):
        if not self._results:
            return
        base = QFileDialog.getExistingDirectory(self, "Export ROI analysis to directory")
        if not base:
            return
        out = os.path.join(base, f"roi_analysis_{datetime.now():%Y%m%d_%H%M%S}")
        try:
            written = export_results(self._results, out)
        except Exception as e:
            self.status.setText(f"Export failed: {e}")
            return
        self.status.setText(f"Exported {len(written)} file(s) to {out}")
