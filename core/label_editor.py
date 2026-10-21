"""
label_editor.py
===============

The label list of the labelling page: one model (``LabelManager``) that owns the list and an
editor panel (``LabelEditor``) that changes it.  Every change takes effect immediately and is
announced through signals, so the labelling page never has to copy lists around:

    manager.changed            the list (or its order) changed  -> rebuild picker, recolour ROIs
    manager.renamed(old, new)  a label was renamed / merged     -> relabel the ROIs that used it
    manager.removed(name, how) a label was removed              -> ROIs become unlabeled or are deleted

The order of the list sets each label's colour and its number-key shortcut.
"""

from __future__ import annotations

import csv
import os

from PyQt6.QtCore import QObject, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
                             QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

EMPTY_LABEL = "Empty Label"
RESERVED = {EMPTY_LABEL.lower(), "symmetry vector"}

# "Add as" choices: how one typed name becomes label(s)
VARIANTS = [
    ("Single label", lambda n: [n]),
    ("Chiral set: (None), (S), (R)", lambda n: [f"{n} (None)", f"{n} (S)", f"{n} (R)"]),
    ("GNR: name (GNR)", lambda n: [f"{n} (GNR)"]),
]


class LabelManager(QObject):
    changed = pyqtSignal()
    renamed = pyqtSignal(str, str)          # old, new (new may be an existing label: merge)
    removed = pyqtSignal(str, str)          # name, "unlabel" | "delete"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: list[str] = []

    # ------------------------------------------------------------------ queries
    def labels(self) -> list:
        return list(self._labels)

    def __contains__(self, name):
        return name in self._labels

    def __len__(self):
        return len(self._labels)

    def validate(self, name, allow_existing=False):
        """Error message for an unusable label name, or None."""
        name = (name or "").strip()
        if not name:
            return "The label name is empty."
        if name.lower() in RESERVED:
            return f"'{name}' is reserved."
        if not allow_existing and name in self._labels:
            return f"'{name}' is already in the list."
        if "," in name:
            return "Label names cannot contain commas."
        return None

    # ------------------------------------------------------------------ changes
    def set_labels(self, labels):
        clean = []
        for name in labels or []:
            name = str(name).strip()
            if name and name.lower() not in RESERVED and name not in clean:
                clean.append(name)
        if clean != self._labels:
            self._labels = clean
            self.changed.emit()

    def add(self, names) -> list:
        """Append new names (invalid / existing ones are skipped). Returns the names added."""
        if isinstance(names, str):
            names = [names]
        added = []
        for name in names:
            name = str(name).strip()
            if self.validate(name) is None and name not in added:
                added.append(name)
        if added:
            self._labels.extend(added)
            self.changed.emit()
        return added

    def merge(self, names) -> list:
        """Keep the current labels (and their colours), append the new ones."""
        return self.add([n for n in names if n not in self._labels])

    def rename(self, old, new):
        """Rename a label; renaming onto an existing label merges the two."""
        new = new.strip()
        if old not in self._labels or old == new:
            return False
        if new in self._labels:
            self._labels.remove(old)
        else:
            self._labels[self._labels.index(old)] = new
        self.renamed.emit(old, new)
        self.changed.emit()
        return True

    def remove(self, name, how="unlabel"):
        if name not in self._labels:
            return False
        self._labels.remove(name)
        self.removed.emit(name, how)
        self.changed.emit()
        return True

    def move(self, name, delta):
        if name not in self._labels:
            return False
        i = self._labels.index(name)
        j = max(0, min(len(self._labels) - 1, i + delta))
        if i == j:
            return False
        self._labels.insert(j, self._labels.pop(i))
        self.changed.emit()
        return True

    # ------------------------------------------------------------------ files
    @staticmethod
    def read_file(path) -> list:
        """One label per line (first column of a CSV); blank lines and '#' comments ignored."""
        with open(path, newline="", encoding="utf-8-sig") as fh:
            return [row[0].strip() for row in csv.reader(fh) if row and row[0].strip() and not row[0].startswith("#")]

    def write_file(self, path):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for name in self._labels:
                writer.writerow([name])


def _swatch(color, size=14):
    pix = QPixmap(size, size)
    pix.fill(QColor(color))
    return QIcon(pix)


class LabelEditor(QWidget):
    """
    Panel to add, rename, reorder and remove labels.  ``color_for(name)`` gives the colour of
    a label and ``count_provider()`` the number of ROIs per label (all images).
    """
    labelActivated = pyqtSignal(str)        # user picked a label in the editor
    saveExamplesRequested = pyqtSignal()
    closeRequested = pyqtSignal()

    def __init__(self, manager: LabelManager, color_for=None, count_provider=None, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.color_for = color_for or (lambda name: "#888888")
        self.count_provider = count_provider or (lambda: {})
        self._counts = {}
        self._filling = False

        title = QLabel("<b>Labels</b>")
        close = QToolButton()
        close.setText("✕")
        close.setToolTip("Close the label panel")
        close.clicked.connect(self.closeRequested)
        head = QHBoxLayout()
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(close)

        # ---- add row
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("New label name (several: separate with ';')")
        self.name_edit.returnPressed.connect(self.add_from_input)
        self.name_edit.textChanged.connect(lambda _: self._message(""))
        self.variant = QComboBox()
        for text, _ in VARIANTS:
            self.variant.addItem(text)
        self.variant.setToolTip("Chiral set adds three labels: name (None), name (S), name (R)")
        add = QPushButton("Add")
        add.setAutoDefault(False)
        add.clicked.connect(self.add_from_input)
        add_row = QHBoxLayout()
        add_row.addWidget(self.name_edit, 1)
        add_row.addWidget(add)
        self.message = QLabel("")
        self.message.setWordWrap(True)

        # ---- table: colour, key, name, ROI count
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Key", "Label", "ROIs"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                   QAbstractItemView.EditTrigger.EditKeyPressed)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setIconSize(QSize(14, 14))
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.setToolTip("Double-click a name to rename it (renaming onto an existing label merges them).\n"
                              "The order sets the colour and the number key of each label.")

        def tool(text, tip, slot):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            return b

        edit_row = QHBoxLayout()
        edit_row.addWidget(tool("▲", "Move up", lambda: self._move(-1)))
        edit_row.addWidget(tool("▼", "Move down", lambda: self._move(1)))
        edit_row.addWidget(tool("Rename", "Rename the selected label (F2)", self._rename_selected))
        edit_row.addWidget(tool("Remove", "Remove the selected label (Delete)", self._remove_selected))
        edit_row.addStretch(1)

        file_row = QHBoxLayout()
        for text, tip, slot in (("Load…", "Load a label list (one label per line)", self.load_dialog),
                                ("Save…", "Save the label list (one label per line)", self.save_dialog),
                                ("Add used labels", "Add the labels of existing ROIs that are not in the list",
                                 self.add_used_labels)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            file_row.addWidget(b)

        # ---- label previews (filled by the labelling page)
        self.previews = QGroupBox("Label previews (first box of each label)")
        self.previews.setCheckable(True)
        self.previews.setChecked(False)
        pl = QVBoxLayout(self.previews)
        self.preview_area = QScrollArea()
        self.preview_area.setWidgetResizable(True)
        self.preview_area.setMinimumHeight(140)
        inner = QWidget()
        self.preview_grid = QGridLayout(inner)
        self.preview_area.setWidget(inner)
        save_examples = QPushButton("Save feature parameters…")
        save_examples.setAutoDefault(False)
        save_examples.clicked.connect(self.saveExamplesRequested)
        pl.addWidget(self.preview_area)
        pl.addWidget(save_examples)
        self.previews.toggled.connect(lambda on: (self.preview_area.setVisible(on), save_examples.setVisible(on)))
        self.preview_area.setVisible(False)
        save_examples.setVisible(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(head)
        lay.addLayout(add_row)
        lay.addWidget(self.variant)
        lay.addWidget(self.message)
        lay.addWidget(self.table, 1)
        lay.addLayout(edit_row)
        lay.addLayout(file_row)
        lay.addWidget(self.previews)

        manager.changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ view
    def _message(self, text, error=False):
        self.message.setText(text)
        self.message.setStyleSheet("color: #b00020;" if error else "color: #2e7d32;")

    def selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 1)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def select(self, name):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 1).data(Qt.ItemDataRole.UserRole) == name:
                self.table.blockSignals(True)
                self.table.selectRow(r)
                self.table.blockSignals(False)
                return

    def refresh(self):
        keep = self.selected()
        self._counts = self.count_provider() or {}
        self._filling = True
        self.table.setRowCount(0)
        for i, name in enumerate(self.manager.labels()):
            self.table.insertRow(i)
            key = QTableWidgetItem(str(i + 1) if i < 9 else "")
            key.setFlags(key.flags() & ~Qt.ItemFlag.ItemIsEditable)
            key.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            key.setToolTip("Number key that selects this label" if i < 9 else "")
            item = QTableWidgetItem(_swatch(self.color_for(name)), name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            count = QTableWidgetItem(str(self._counts.get(name, 0)))
            count.setFlags(count.flags() & ~Qt.ItemFlag.ItemIsEditable)
            count.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(i, 0, key)
            self.table.setItem(i, 1, item)
            self.table.setItem(i, 2, count)
        self._filling = False
        if keep:
            self.select(keep)

    def update_counts(self):
        self._counts = self.count_provider() or {}
        self._filling = True
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 1).data(Qt.ItemDataRole.UserRole)
            self.table.item(r, 2).setText(str(self._counts.get(name, 0)))
        self._filling = False

    # ------------------------------------------------------------------ actions
    def add_from_input(self):
        text = self.name_edit.text()
        expand = VARIANTS[self.variant.currentIndex()][1]
        names = [n for part in text.split(";") for n in expand(part.strip()) if part.strip()]
        if not names:
            self._message("Type a label name first.", error=True)
            return
        errors = [e for e in (self.manager.validate(n) for n in names) if e]
        added = self.manager.add(names)
        if added:
            self.name_edit.clear()
            self.select(added[-1])
            self.labelActivated.emit(added[-1])
            self._message(f"Added: {', '.join(added)}" + (f"  ({'; '.join(errors)})" if errors else ""))
        else:
            self._message("; ".join(errors) or "Nothing added.", error=True)

    def _move(self, delta):
        name = self.selected()
        if name and self.manager.move(name, delta):
            self.select(name)

    def _rename_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.table.editItem(self.table.item(rows[0].row(), 1))

    def _on_item_changed(self, item):
        if self._filling or item.column() != 1:
            return
        old = item.data(Qt.ItemDataRole.UserRole)
        new = item.text().strip()
        if new == old:
            return
        error = self.manager.validate(new, allow_existing=True)
        if error is None and new in self.manager:
            n = self._counts.get(old, 0)
            answer = QMessageBox.question(self, "Merge labels",
                                          f"'{new}' already exists. Merge '{old}' into '{new}'?"
                                          + (f"\n{n} ROI(s) will be relabelled." if n else ""))
            if answer != QMessageBox.StandardButton.Yes:
                error = "Rename cancelled."
        if error:
            self._message(error, error=True)
            self.refresh()
            return
        self.manager.rename(old, new)
        self.select(new)
        self._message(f"Renamed '{old}' to '{new}'.")

    def _remove_selected(self):
        name = self.selected()
        if not name:
            return
        n = self._counts.get(name, 0)
        how = "unlabel"
        if n:
            box = QMessageBox(self)
            box.setWindowTitle("Remove label")
            box.setText(f"{n} ROI(s) use '{name}'. What should happen to them?")
            keep = box.addButton("Keep them as 'Empty Label'", QMessageBox.ButtonRole.AcceptRole)
            delete = box.addButton("Delete them", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is delete:
                how = "delete"
            elif clicked is not keep:
                return
        self.manager.remove(name, how)
        self._message(f"Removed '{name}'" + ("" if not n else (" and its ROIs." if how == "delete" else
                                                                 "; its ROIs are now unlabelled.")))

    def _on_selection(self):
        name = self.selected()
        if name:
            self.labelActivated.emit(name)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.table.hasFocus():
            self._remove_selected()
            return
        if event.key() == Qt.Key.Key_F2:
            self._rename_selected()
            return
        super().keyPressEvent(event)

    def add_used_labels(self):
        used = [n for n in self.count_provider() if n and n.lower() not in RESERVED]
        added = self.manager.merge(used)
        self._message(f"Added {len(added)} label(s) used by ROIs: {', '.join(added)}" if added
                      else "Every label used by ROIs is already in the list.")

    def load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load label list", "", "Label lists (*.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            names = LabelManager.read_file(path)
        except (OSError, UnicodeDecodeError) as exc:
            self._message(f"Could not read the file: {exc}", error=True)
            return
        if len(self.manager):
            box = QMessageBox(self)
            box.setWindowTitle("Load label list")
            box.setText(f"{len(names)} label(s) in {os.path.basename(path)}. Replace the current list or add them?")
            add = box.addButton("Add to the list", QMessageBox.ButtonRole.AcceptRole)
            replace = box.addButton("Replace the list", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is replace:
                self.manager.set_labels(names)
                self._message(f"Loaded {len(self.manager)} label(s) from {os.path.basename(path)}.")
            elif box.clickedButton() is add:
                added = self.manager.merge(names)
                self._message(f"Added {len(added)} label(s) from {os.path.basename(path)}.")
            return
        self.manager.set_labels(names)
        self._message(f"Loaded {len(self.manager)} label(s) from {os.path.basename(path)}.")

    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save label list", "labels.csv", "Label lists (*.csv)")
        if not path:
            return
        try:
            self.manager.write_file(path)
            self._message(f"Saved {len(self.manager)} label(s) to {os.path.basename(path)}.")
        except OSError as exc:
            self._message(f"Save failed: {exc}", error=True)
