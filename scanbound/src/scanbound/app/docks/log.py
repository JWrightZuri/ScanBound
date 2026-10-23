"""Activity log: messages of the session, with time stamps."""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ..session import Session


class LogDock(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(5000)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.text.clear)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(clear)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.text, 1)
        lay.addLayout(row)
        session.message.connect(self.append)

    def append(self, text: str, level: str = "info") -> None:
        prefix = {"warning": "⚠ ", "error": "✖ "}.get(level, "")
        self.text.appendPlainText(f"{datetime.now():%H:%M:%S}  {prefix}{text}")
        self.text.moveCursor(QTextCursor.MoveOperation.End)
