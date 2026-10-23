"""Window showing a surface-analysis Report: summary, tables, plots, export, and proposals to review."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..session import Session
from ..theme import mark

MAX_ROWS = 5000


def _fmt(v: Any) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return ""
        return f"{v:.4g}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return ", ".join(_fmt(x) for x in list(v)[:12]) + (" …" if len(v) > 12 else "")
    return "" if v is None else str(v)


class _NumItem(QTableWidgetItem):
    """Sorts by number when both cells are numbers."""

    def __lt__(self, other: "QTableWidgetItem") -> bool:
        a, b = self.data(Qt.ItemDataRole.UserRole), other.data(Qt.ItemDataRole.UserRole)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return super().__lt__(other)


def table_widget(rows: list[dict[str, Any]]) -> QTableWidget:
    keys: list[str] = []
    for r in rows[:MAX_ROWS]:
        for k in r:
            if k not in keys:
                keys.append(k)
    t = QTableWidget(min(len(rows), MAX_ROWS), len(keys))
    t.setHorizontalHeaderLabels(keys)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    for i, r in enumerate(rows[:MAX_ROWS]):
        for j, k in enumerate(keys):
            v = r.get(k)
            it = _NumItem(_fmt(v))
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                fv = float(v)
                if math.isfinite(fv):
                    it.setData(Qt.ItemDataRole.UserRole, fv)
            t.setItem(i, j, it)
    t.setSortingEnabled(True)
    t.resizeColumnsToContents()
    for j in range(len(keys)):
        t.setColumnWidth(j, min(t.columnWidth(j), 260))
    return t


def plot_widget(spec: dict[str, Any]) -> QWidget:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5, 3.6), tight_layout=True)
    kind = spec.get("kind")
    if kind == "polar":
        ax = fig.add_subplot(111, projection="polar")
        period = float(spec.get("period", 360.0))
        a = np.radians(np.asarray(spec["data"], float) * 360.0 / period)
        ax.hist(a, bins=max(12, int(min(72, len(a) // 2 + 6))), color="#4C8BF5", alpha=0.85)
        ax.set_xticks(np.radians(np.arange(0, 360, 45)))
        ax.set_xticklabels([f"{x * period / 360:.0f}°" for x in np.arange(0, 360, 45)])
    else:
        ax = fig.add_subplot(111)
        if kind == "hist":
            ax.hist(np.asarray(spec["data"], float), bins=30, color="#4C8BF5", alpha=0.85)
            ax.set_ylabel("count")
        elif kind == "heatmap":
            im = ax.imshow(np.asarray(spec["data"], float), origin="lower", extent=spec.get("extent"), cmap="magma",
                           aspect="equal")
            fig.colorbar(im, ax=ax)
            pts = np.asarray(spec.get("points", []), float).reshape(-1, 2)
            if len(pts):
                ax.plot(pts[:, 0], pts[:, 1], ".", color="#00E5FF", ms=3)
        elif kind == "band":
            x = np.asarray(spec["x"], float)
            ax.fill_between(x, spec["lo"], spec["hi"], color="#999999", alpha=0.35, label="random (95 %)")
            ax.plot(x, spec["y"], color="#4C8BF5", label="observed")
            ax.axhline(1.0, color="#777777", lw=0.8)
            ax.legend(fontsize=8)
        elif kind == "scatter":
            g = np.asarray(spec.get("groups", [0] * len(spec["x"])))
            ax.scatter(spec["x"], spec["y"], c=g, cmap="tab10", s=14)
        else:
            ax.plot(spec["x"], spec["y"], "-o", ms=3, color="#4C8BF5")
    ax.set_title(spec.get("title", ""), fontsize=10)
    if spec.get("x_label") and kind != "polar":
        ax.set_xlabel(spec["x_label"])
    if spec.get("y_label") and kind != "polar":
        ax.set_ylabel(spec["y_label"])
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    canvas = FigureCanvasQTAgg(fig)
    lay.addWidget(NavigationToolbar2QT(canvas, w))
    lay.addWidget(canvas, 1)
    return w


class ReportWindow(QDialog):
    """Non-modal: several reports can stay open next to the image."""

    def __init__(self, session: Session, report: Any, parent: Optional[QWidget] = None,
                 on_proposals: Optional[Callable[[list], None]] = None):
        super().__init__(parent)
        self.session = session
        self.report = report
        self.setWindowTitle(report.title)
        self.setModal(False)
        self.resize(900, 640)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        tabs = QTabWidget()
        text = QTextBrowser()
        text.setPlainText("\n".join(report.summary))
        tabs.addTab(text, "Summary")
        for name, rows in report.tables.items():
            if not rows:
                continue
            t = table_widget(rows)
            t.cellDoubleClicked.connect(lambda r, _c, tw=t: self._goto(tw, r))
            tabs.addTab(t, name.capitalize())
        for spec in report.plots:
            try:
                tabs.addTab(plot_widget(spec), spec.get("title", "Plot")[:28])
            except Exception as exc:  # noqa: BLE001 - one bad plot should not lose the report
                tabs.addTab(QLabel(f"Plot failed: {exc}"), "Plot")
        hint = QLabel("Double-click a row with an id to select that annotation. Results are candidates with "
                      "their evidence; check them against the image.")
        hint.setWordWrap(True)
        mark(hint, "muted")
        export = QPushButton("Export…")
        export.setToolTip("One CSV per table, report.json and summary.txt with provenance (files, hashes, parameters)")
        export.clicked.connect(self.export)
        copy = QPushButton("Copy summary")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(report.text()))
        row = QHBoxLayout()
        row.addWidget(hint, 1)
        if report.proposals and on_proposals is not None:
            from ..features import REVIEW
            add = QPushButton(f"Add {len(report.proposals)} for review" if REVIEW
                              else f"Add {len(report.proposals)} as annotations")
            add.setToolTip("Add the results as proposed annotations: accept or reject them in the Review panel" if REVIEW
                           else "Add the results to the image as annotations (edit or delete them as usual)")
            add.clicked.connect(lambda: (on_proposals(report.proposals), add.setEnabled(False)))
            row.addWidget(add)
        row.addWidget(copy)
        row.addWidget(export)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs, 1)
        lay.addLayout(row)

    def _goto(self, table: QTableWidget, row: int) -> None:
        cols = [table.horizontalHeaderItem(j).text() for j in range(table.columnCount())]
        for key in ("id", "object", "after_object", "before_object"):
            if key in cols:
                it = table.item(row, cols.index(key))
                ann_id = it.text() if it else ""
                a = self.session.store.get(ann_id)
                if a is not None:
                    if a.image_id != self.session.current_id:
                        self.session.set_current(a.image_id)
                    self.session.select([ann_id])
                    parent = self.parent()
                    if parent is not None and hasattr(parent, "center_on"):
                        parent.center_on(ann_id)
                return

    def export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export report to folder")
        if not folder:
            return
        import os
        sub = os.path.join(folder, "".join(c if c.isalnum() else "_" for c in self.report.title).strip("_"))
        files = self.report.save(sub)
        self.session.info(f"Exported {len(files)} file(s) to {sub}")
