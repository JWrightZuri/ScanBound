"""Help → Tutorials… and the welcome shown on the first start."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                             QVBoxLayout, QWidget)

from ..theme import mark
from .content import TUTORIALS
from .engine import Tutorial, done_tutorials

WELCOMED_KEY = "tutorials/welcomed"


class TutorialChooser(QDialog):
    """Lists the tutorials (✓ = finished before); Start runs the chosen one."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Tutorials")
        self.resize(620, 420)
        self.chosen: Optional[Tutorial] = None
        done = done_tutorials()
        self.list = QListWidget()
        level = None
        for t in TUTORIALS:
            if t.level != level:
                level = t.level
                head = QListWidgetItem(level)
                head.setFlags(Qt.ItemFlag.NoItemFlags)
                f = head.font()
                f.setBold(True)
                head.setFont(f)
                self.list.addItem(head)
            it = QListWidgetItem(f"{'✓' if t.key in done else '   '}  {t.title}   ·  {t.minutes} min")
            it.setData(Qt.ItemDataRole.UserRole, t.key)
            it.setToolTip(t.summary)
            self.list.addItem(it)
        self.list.currentItemChanged.connect(self._show)
        self.list.itemDoubleClicked.connect(lambda _it: self._start())
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary.setMinimumWidth(220)
        note = QLabel("Each tutorial runs in the program itself: it highlights what to use and ticks each step "
                      "off when you have done it. Steps you would rather skip have a button that does them for you.")
        note.setWordWrap(True)
        mark(note, "muted")
        right = QVBoxLayout()
        right.addWidget(self.summary, 1)
        right.addWidget(note)
        body = QHBoxLayout()
        body.addWidget(self.list, 3)
        body.addLayout(right, 2)
        self.start = QPushButton("Start")
        self.start.setDefault(True)
        self.start.clicked.connect(self._start)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        row.addWidget(self.start)
        lay = QVBoxLayout(self)
        lay.addLayout(body, 1)
        lay.addLayout(row)
        first = next(k for k in range(self.list.count()) if self.list.item(k).data(Qt.ItemDataRole.UserRole))
        nxt = next((k for k in range(self.list.count())
                    if (key := self.list.item(k).data(Qt.ItemDataRole.UserRole)) and key not in done), first)
        self.list.setCurrentRow(nxt)

    def _tutorial(self) -> Optional[Tutorial]:
        it = self.list.currentItem()
        key = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        return next((t for t in TUTORIALS if t.key == key), None)

    def _show(self, *_) -> None:
        t = self._tutorial()
        self.start.setEnabled(t is not None)
        if t is not None:
            self.summary.setText(f"<h3>{t.title}</h3><p>{t.summary}</p><p><i>{len(t.steps)} steps, about "
                                 f"{t.minutes} minutes.</i></p>")

    def _start(self) -> None:
        self.chosen = self._tutorial()
        if self.chosen is not None:
            self.accept()


class Welcome(QDialog):
    """First start: offer the getting-started tour."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to ScanBound")
        self.choice = ""
        text = QLabel("<h2>Welcome to ScanBound</h2>"
                      "<p>New here? A short guided tour shows the whole workflow in the program itself: open a scan, "
                      "process it, make a label, draw and remove a shape, find features with the adapt pipeline "
                      "and correct the results.</p>"
                      "<p>It takes about 8 minutes and can use a practice image, so you need no data of your own. "
                      "You can start it, and the advanced tutorials, at any time from <b>Help → Tutorials…</b> "
                      "(F1).</p>")
        text.setWordWrap(True)
        text.setMinimumWidth(440)
        self.again = QCheckBox("Show this when ScanBound starts")
        tour = QPushButton("Take the tour")
        tour.setDefault(True)
        tour.clicked.connect(lambda: self._pick("tour"))
        browse = QPushButton("All tutorials…")
        browse.clicked.connect(lambda: self._pick("browse"))
        later = QPushButton("Not now")
        later.clicked.connect(lambda: self._pick(""))
        row = QHBoxLayout()
        row.addWidget(self.again)
        row.addStretch(1)
        row.addWidget(later)
        row.addWidget(browse)
        row.addWidget(tour)
        lay = QVBoxLayout(self)
        lay.addWidget(text)
        lay.addLayout(row)

    def _pick(self, choice: str) -> None:
        self.choice = choice
        QSettings("ScanBound", "ScanBound").setValue(WELCOMED_KEY, not self.again.isChecked())
        self.accept()


def should_welcome() -> bool:
    return not QSettings("ScanBound", "ScanBound").value(WELCOMED_KEY, False, type=bool)
