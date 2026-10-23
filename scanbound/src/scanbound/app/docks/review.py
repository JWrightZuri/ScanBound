"""
Review automated results: every proposal can be inspected, corrected, accepted or rejected.

The table lists proposals of the current image (or of all images).  Selecting a row shows it on
the canvas; corrections are made with the normal editing tools.  Keys: A accept, R reject, N next.
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QTableView, QVBoxLayout, QWidget)

from ...annotations import Annotation
from ...ml.evaluate import review_stats
from ..qtutil import qcolor
from ..session import Session
from ..theme import mark

COLUMNS = ["Image", "Label", "Kind", "Confidence", "Status", "Source"]
FILTERS = [("To review", ("proposed",)), ("Rejected", ("rejected",)), ("Accepted", ("accepted",)),
           ("All automated", ("proposed", "accepted", "rejected"))]


class ProposalModel(QAbstractTableModel):
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.ids: list[str] = []

    def set_ids(self, ids: list[str]) -> None:
        self.beginResetModel()
        self.ids = ids
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(self.ids)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: B008, N802
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return None

    def annotation(self, row: int) -> Optional[Annotation]:
        return self.session.store.get(self.ids[row]) if 0 <= row < len(self.ids) else None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        a = self.annotation(index.row())
        if a is None:
            return None
        c = index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.UserRole):
            if c == 0:
                return a.image_id.replace("::", " : ")
            if c == 1:
                return a.label or "(none)"
            if c == 2:
                return a.kind.replace("_", " ") + (f" ({a.geometry.type})" if a.kind == "feature" else "")
            if c == 3:
                conf = a.origin.confidence
                if role == Qt.ItemDataRole.UserRole:
                    return -1.0 if conf is None else conf
                return "" if conf is None else f"{conf:.2f}"
            if c == 4:
                return a.status + (" (edited)" if a.edited else "")
            if c == 5:
                return a.origin.name or a.origin.method
        if role == Qt.ItemDataRole.ForegroundRole and c == 1:
            return qcolor(self.session.label_color(a.label))
        if role == Qt.ItemDataRole.ToolTipRole:
            notes = [f"{k}: {v}" for k, v in (a.attrs.get("features") or {}).items() if v is not None][:12]
            return "\n".join([f"run {a.origin.run}" if a.origin.run else "", a.attrs.get("note", "")] + notes).strip()
        return None


class ReviewDock(QWidget):
    centerRequested = pyqtSignal(str)            # annotation id

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.model = ProposalModel(session)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(3, Qt.SortOrder.DescendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self._on_rows)

        self.filter = QComboBox()
        for name, _ in FILTERS:
            self.filter.addItem(name)
        self.filter.currentIndexChanged.connect(self.refresh)
        self.all_images = QCheckBox("All images")
        self.all_images.toggled.connect(self.refresh)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        mark(self.summary, "muted")

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        self.threshold.setToolTip("Confidence threshold for the bulk actions")
        b_accept = QPushButton("Accept (A)")
        b_accept.clicked.connect(self.accept)
        b_reject = QPushButton("Reject (R)")
        b_reject.clicked.connect(self.reject)
        b_next = QPushButton("Next (N)")
        b_next.clicked.connect(lambda: self.next_pending(1))
        b_hi = QPushButton("Accept all ≥")
        b_hi.setToolTip("Accept every listed proposal with at least this confidence (proposals without a "
                        "confidence count as 1)")
        b_hi.clicked.connect(lambda: self.bulk(True))
        b_lo = QPushButton("Reject all <")
        b_lo.clicked.connect(lambda: self.bulk(False))
        b_done = QPushButton("Mark image reviewed")
        b_done.setToolTip("Every object on this image is annotated: unannotated areas are true negatives "
                          "(the image can be exported as a negative example)")
        b_done.clicked.connect(lambda: session.current_id and session.set_reviewed(session.current_id, True))

        top = QHBoxLayout()
        top.addWidget(self.filter)
        top.addWidget(self.all_images)
        top.addStretch(1)
        grid = QGridLayout()
        grid.addWidget(b_accept, 0, 0)
        grid.addWidget(b_reject, 0, 1)
        grid.addWidget(b_next, 0, 2)
        grid.addWidget(b_hi, 1, 0)
        grid.addWidget(self.threshold, 1, 1)
        grid.addWidget(b_lo, 1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        lay.addLayout(grid)
        lay.addWidget(b_done)
        lay.addWidget(self.summary)
        self._syncing = False
        session.annotationsChanged.connect(lambda _e: self.refresh())
        session.imageChanged.connect(lambda _i: self.refresh())
        session.projectChanged.connect(self.refresh)
        session.selectionChanged.connect(self._sync_selection)

    def statuses(self) -> tuple:
        return FILTERS[self.filter.currentIndex()][1]

    def refresh(self) -> None:
        s = self.session
        st = self.statuses()
        if self.all_images.isChecked():
            anns = s.store.query(lambda a: a.status in st and a.origin.method != "manual")
        else:
            anns = [a for a in s.annotations() if a.status in st and a.origin.method != "manual"]
        self.model.set_ids([a.id for a in anns])
        stats = review_stats(s.store)
        if stats["total"]:
            rate = stats["acceptance_rate"]
            self.summary.setText(f"{stats['pending']} to review, {stats['accepted']} accepted unchanged, "
                                 f"{stats['accepted_edited']} accepted after correction, {stats['rejected']} rejected"
                                 + ("" if math.isnan(rate) else f" — acceptance {100 * rate:.0f} %"))
        else:
            self.summary.setText("No automated results yet: run a detector (Detect menu).")
        self._sync_selection()

    def _rows_ids(self) -> list[str]:
        rows = {self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()}
        return [self.model.ids[r] for r in sorted(rows) if r < len(self.model.ids)]

    def _on_rows(self, *_):
        if self._syncing:
            return
        ids = self._rows_ids()
        if not ids:
            return
        a = self.session.store.get(ids[0])
        if a is None:
            return
        if a.image_id != self.session.current_id:
            self.session.set_current(a.image_id)
        self.session.select([i for i in ids if (x := self.session.store.get(i)) and x.image_id == a.image_id])
        self.centerRequested.emit(a.id)

    def _sync_selection(self) -> None:
        sel = set(self.session.selection)
        self._syncing = True
        try:
            sm = self.table.selectionModel()
            sm.clearSelection()
            for r, i in enumerate(self.model.ids):
                if i in sel:
                    idx = self.proxy.mapFromSource(self.model.index(r, 0))
                    sm.select(idx, sm.SelectionFlag.Select | sm.SelectionFlag.Rows)
        finally:
            self._syncing = False

    def _targets(self) -> list[str]:
        sel = [i for i in self.session.selection if (a := self.session.store.get(i)) and a.origin.method != "manual"]
        return sel or self._rows_ids()

    def accept(self) -> None:
        ids = self._targets()
        if ids:
            self.session.store.accept(ids)
            self.next_pending(1)

    def reject(self) -> None:
        ids = self._targets()
        if ids:
            self.session.store.reject(ids)
            self.next_pending(1)

    def bulk(self, accept: bool) -> None:
        t = self.threshold.value()
        anns = [self.session.store.get(i) for i in self.model.ids]
        anns = [a for a in anns if a is not None and a.status == "proposed"]
        conf = (lambda a: 1.0 if a.origin.confidence is None else a.origin.confidence)
        ids = [a.id for a in anns if (conf(a) >= t) == accept]
        if ids:
            (self.session.store.accept if accept else self.session.store.reject)(ids)
            self.session.info(f"{'Accepted' if accept else 'Rejected'} {len(ids)} proposal(s)")

    def next_pending(self, delta: int = 1) -> None:
        """Select the next proposal still to review (this image first, then the others)."""
        s = self.session
        pending = [a for a in s.annotations() if a.status == "proposed"]
        if not pending:
            others = s.store.query(lambda a: a.status == "proposed")
            if not others:
                s.info("Nothing left to review.")
                s.select([])
                return
            s.set_current(others[0].image_id)
            pending = [a for a in s.annotations() if a.status == "proposed"]
            if not pending:
                return
        pending.sort(key=lambda a: (a.geometry.bbox()[1] // 32, a.geometry.bbox()[0]))   # reading order
        cur = s.selection[0] if s.selection else None
        ids = [a.id for a in pending]
        k = ids.index(cur) if cur in ids else -1
        nxt = ids[(k + delta) % len(ids)] if k >= 0 else ids[0]
        s.select([nxt])
        self.centerRequested.emit(nxt)
