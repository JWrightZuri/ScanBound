"""Labels: the project's label set with colours, number keys and counts; the active label."""

from __future__ import annotations

from collections import Counter

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QHBoxLayout, QInputDialog, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                             QToolButton, QVBoxLayout, QWidget)

from ...annotations import GROUND_TRUTH
from ..qtutil import save_file_name, swatch
from ..session import Session


class LabelsDock(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.list = QListWidget()
        self.list.setToolTip("Click: label for new annotations. Double-click: give the selected annotations this label.\n"
                             "Keys 1-9 do the same for the first nine labels; 0 removes the label.")
        self.list.currentItemChanged.connect(self._on_current)
        self.list.itemDoubleClicked.connect(lambda it: self.assign(it.data(Qt.ItemDataRole.UserRole)))
        buttons = QHBoxLayout()
        for text, tip, slot in (("＋", "Add a label", self.add), ("Rename", "Rename the label (merges with an existing one)", self.rename),
                                ("−", "Remove the label", self.remove), ("▲", "Move up", lambda: self.move(-1)),
                                ("▼", "Move down", lambda: self.move(1))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        for text, tip, slot in (("Load…", "Add labels from a text / CSV file (one per line)", self.load_list),
                                ("Save…", "Save the label list (one per line)", self.save_list)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        assign = QPushButton("Assign to selection")
        assign.clicked.connect(lambda: self.assign(self.session.current_label))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.list, 1)
        lay.addLayout(buttons)
        lay.addWidget(assign)
        session.labelsChanged.connect(self.refresh)
        session.projectChanged.connect(self.refresh)
        session.annotationsChanged.connect(lambda _ev: self.refresh())
        session.currentLabelChanged.connect(lambda _l: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        s = self.session
        labels = s.project.labels
        counts = Counter(a.label for a in s.store if a.status in GROUND_TRUTH and a.kind in ("feature", "centre_line"))
        here = Counter(a.label for a in s.annotations() if a.status in GROUND_TRUTH)
        self.list.blockSignals(True)
        self.list.clear()
        none = QListWidgetItem(swatch(s.label_color(None)), "(no label)")
        none.setData(Qt.ItemDataRole.UserRole, None)
        self.list.addItem(none)
        current = none
        for name in labels.names:
            k = labels.key_for(name)
            text = f"{k}  {name}" if k is not None else f"    {name}"
            text += f"    {here.get(name, 0)} here / {counts.get(name, 0)} total"
            it = QListWidgetItem(swatch(labels.color(name)), text)
            it.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(it)
            if name == s.current_label:
                current = it
        self.list.setCurrentItem(current)
        self.list.blockSignals(False)

    def _on_current(self, it, _prev) -> None:  # noqa: ANN001
        if it is not None:
            self.session.set_current_label(it.data(Qt.ItemDataRole.UserRole))

    def _selected_name(self):
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def add(self) -> None:
        name, ok = QInputDialog.getText(self, "Add label", "Label name:")
        name = name.strip()
        if ok and name:
            self.session.project.labels.add(name)
            self.session.labels_edited()
            self.session.set_current_label(name)

    def rename(self) -> None:
        old = self._selected_name()
        if not old:
            return
        new, ok = QInputDialog.getText(self, "Rename label", f"New name for {old!r}:", text=old)
        new = new.strip()
        if not ok or not new or new == old:
            return
        s = self.session
        with s.store.transaction(f"Rename label {old!r}"):
            s.store.rename_label(old, new)
        if new in s.project.labels:              # merge into the existing label
            s.project.labels.remove(old)
        else:
            s.project.labels.rename(old, new)
        s.labels_edited()
        if s.current_label == old:
            s.set_current_label(new)

    def remove(self) -> None:
        name = self._selected_name()
        if not name:
            return
        s = self.session
        used = [a.id for a in s.store if a.label == name]
        if used:
            r = QMessageBox.question(self, "Remove label",
                                     f"{len(used)} annotation(s) have the label {name!r}. Remove the label from them "
                                     "(they stay, unlabelled)?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            if r != QMessageBox.StandardButton.Yes:
                return
            s.store.set_label(used, None)
        s.project.labels.remove(name)
        s.labels_edited()
        if s.current_label == name:
            s.set_current_label(None)

    def move(self, delta: int) -> None:
        name = self._selected_name()
        if not name:
            return
        labels = self.session.project.labels
        labels.move(name, max(0, labels.names.index(name) + delta))
        self.session.labels_edited()

    def load_list(self) -> None:
        import csv
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Load labels", self.session.project.root, "Label lists (*.txt *.csv)")
        if not path:
            return
        with open(path, newline="", encoding="utf-8-sig") as fh:
            names = [r[0].strip() for r in csv.reader(fh) if r and r[0].strip() and not r[0].startswith("#")]
        added = self.session.project.labels.merge(names)
        self.session.labels_edited()
        self.session.info(f"Loaded {len(names)} label(s) ({len(added)} new)")

    def save_list(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path = save_file_name(self, "Save labels", self.session.project.root, "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.session.project.labels.names) + "\n")

    def assign(self, label) -> None:  # noqa: ANN001
        s = self.session
        if s.selection:
            s.store.set_label(list(s.selection), label)
