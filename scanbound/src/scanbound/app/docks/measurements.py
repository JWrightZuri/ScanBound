"""Measurements of the selected annotations, each with its uncertainty and how it was obtained."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QPushButton, QSpinBox, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ...annotations import GROUND_TRUTH, KINDS
from ...measurements import measure_all, measure_project
from ..session import Session
from ..theme import mark
from ..qtutil import save_file_name


class MeasurementsDock(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Quantity", "Value", "Method"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.fold = QSpinBox()
        self.fold.setRange(1, 12)
        self.fold.setToolTip("Rotational symmetry of the features: angles are reduced to 0…360°/n "
                             "(1 = polar, 2 = rods, 3 = three-fold molecules …)")
        self.fold.valueChanged.connect(self.refresh)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        mark(self.hint, "muted")
        copy = QPushButton("Copy")
        copy.clicked.connect(self.copy)
        export = QPushButton("Export all…")
        export.setToolTip("Every measurement of every annotated image (ground truth only) as CSV")
        export.clicked.connect(self.export)
        row = QHBoxLayout()
        row.addWidget(QLabel("Symmetry"))
        row.addWidget(self.fold)
        row.addStretch(1)
        row.addWidget(copy)
        row.addWidget(export)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.hint)
        lay.addLayout(row)
        session.selectionChanged.connect(self.refresh)
        session.annotationsChanged.connect(lambda _e: self.refresh())
        session.imageDataChanged.connect(lambda _i: self.refresh())

    def refresh(self) -> None:
        self.tree.clear()
        s = self.session
        sel = s.selected()
        if not sel:
            self.hint.setText("Select annotations to see their measurements. Hover a value for how it was "
                              "calculated and what it assumes.")
            return
        try:
            img = s.image()
        except Exception:  # noqa: BLE001
            img = None
        anns = s.annotations()
        from ...measurements import reference_axis
        ref = reference_axis(anns)
        ref_ok = ref is not None
        chosen = {a.id for a in sel}
        results = measure_all(anns, img, fold=self.fold.value(), statuses=tuple({*GROUND_TRUTH, "proposed", "rejected"}))
        for a, ms in results:
            if a.id not in chosen:
                continue
            kind = KINDS.get(a.kind)
            title = f"{kind.title if kind else a.kind}" + (f" — {a.label}" if a.label else "") + f"  [{a.status}]"
            top = QTreeWidgetItem([title, "", a.origin.name or a.origin.method])
            for m in ms.values():
                it = QTreeWidgetItem([m.name, m.format(), m.method])
                it.setToolTip(0, m.explain())
                it.setToolTip(1, m.explain())
                it.setToolTip(2, m.explain())
                it.setData(1, Qt.ItemDataRole.UserRole, m.value)
                top.addChild(it)
            self.tree.addTopLevelItem(top)
            top.setExpanded(True)
        notes = []
        if img is not None and not img.calibrated:
            notes.append("No pixel size: values are in pixels (set it in the Info panel).")
        if not ref_ok:
            notes.append("No reference direction on this image: angles are from the image x axis (draw a lattice "
                         "vector a1 with the Lattice vector tool to measure against a substrate direction).")
        elif ref.kind == "lattice_vector":
            notes.append(f"Angles are also given relative to a1 of {ref.attrs.get('lattice', 'the lattice')}.")
        self.hint.setText(" ".join(notes))

    def copy(self) -> None:
        lines = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            lines.append(top.text(0))
            for j in range(top.childCount()):
                c = top.child(j)
                lines.append(f"{c.text(0)}\t{c.text(1)}\t{c.text(2)}")
        QGuiApplication.clipboard().setText("\n".join(lines))

    def export(self) -> None:
        path = save_file_name(self, "Export measurements", "measurements.csv", "CSV files (*.csv)")
        if not path:
            return
        df = measure_project(self.session.project, fold=self.fold.value())
        df.to_csv(path, index=False)
        self.session.info(f"Exported {len(df)} measurement(s) of {df['image'].nunique() if len(df) else 0} image(s) to {path}")
