"""
What a tutorial draws: the spotlight over a window and the step card beside it.

The spotlight is a child of the target's window that lets every mouse click through: the user works
in the real program while it dims everything but the target.  The card is a small frameless tool
window owned by the window the user can use right now (the main window, or a modal dialog while one
is open), so its buttons always work.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPalette, QPen
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QToolButton,
                             QVBoxLayout, QWidget)

from ..theme import mark

DIM = QColor(0, 0, 0, 80)
CARD_WIDTH = 380


class Spotlight(QWidget):
    """Dims its window except ``hole`` (window coordinates) and rings the hole in the accent colour."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hole: Optional[QRect] = None
        self.dim = True

    def set_hole(self, hole: Optional[QRect], dim: bool = True) -> None:
        if hole == self.hole and dim == self.dim:
            return
        self.hole, self.dim = hole, dim
        self.update()

    def follow_parent(self) -> None:
        p = self.parentWidget()
        if p is not None and self.geometry() != p.rect():
            self.setGeometry(p.rect())
        self.raise_()

    def paintEvent(self, _event) -> None:  # noqa: ANN001, N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        whole = QPainterPath()
        whole.addRect(QRectF(self.rect()))
        if self.hole is None:
            if self.dim:
                p.fillPath(whole, DIM)
            return
        r = QRectF(self.hole.adjusted(-5, -5, 5, 5))
        hole = QPainterPath()
        hole.addRoundedRect(r, 6, 6)
        if self.dim:
            p.fillPath(whole.subtracted(hole), DIM)
        accent = self.palette().color(QPalette.ColorRole.Highlight)
        glow = QColor(accent)
        glow.setAlpha(70)
        p.setPen(QPen(glow, 7))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(QPen(accent, 2.5))
        p.drawRoundedRect(r, 6, 6)


class StepCard(QFrame):
    """Title, text, a 'your turn' line and Back / Do it for me / Next."""
    nextClicked = pyqtSignal()
    backClicked = pyqtSignal()
    showClicked = pyqtSignal()
    closeClicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("tutorialCard")
        # never take the keyboard from the main window: the tools' keys (B, G, Delete …) must keep working
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAutoFillBackground(True)
        self.setFixedWidth(CARD_WIDTH)
        self._drag: Optional[QPoint] = None
        self.moved_by_user = False

        self.where = QLabel()
        mark(self.where, "muted")
        close = QToolButton()
        close.setText("✕")
        close.setAutoRaise(True)
        close.setToolTip("Close the tutorial (Help → Tutorials… opens it again)")
        close.clicked.connect(self.closeClicked)
        head = QHBoxLayout()
        head.addWidget(self.where, 1)
        head.addWidget(close)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.title = QLabel()
        self.title.setWordWrap(True)
        f = self.title.font()
        f.setPointSizeF(f.pointSizeF() * 1.25)
        f.setBold(True)
        self.title.setFont(f)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setOpenExternalLinks(True)
        self.body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.back = QPushButton("Back")
        self.back.clicked.connect(self.backClicked)
        self.show_me = QPushButton("Do it for me")
        self.show_me.setToolTip("Let ScanBound do this step, so you can see the result")
        self.show_me.clicked.connect(self.showClicked)
        self.next = QPushButton("Next")
        self.next.setDefault(True)
        self.next.clicked.connect(self.nextClicked)
        for b in (self.back, self.show_me, self.next):
            b.setAutoDefault(False)
        row = QHBoxLayout()
        row.addWidget(self.back)
        row.addStretch(1)
        row.addWidget(self.show_me)
        row.addWidget(self.next)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)
        lay.addLayout(head)
        lay.addWidget(self.progress)
        lay.addWidget(self.title)
        lay.addWidget(self.body)
        lay.addWidget(self.status)
        lay.addLayout(row)

    def set_step(self, tutorial: str, index: int, count: int, title: str, text: str) -> None:
        self.where.setText(f"{tutorial}  ·  step {index + 1} of {count}")
        self.progress.setRange(0, count)
        self.progress.setValue(index + 1)
        self.title.setText(title)
        self.body.setText(text)
        self.back.setEnabled(index > 0)
        self.next.setText("Finish" if index == count - 1 else "Next")
        self.moved_by_user = False
        self.adjustSize()

    def set_status(self, text: str, tone: Optional[str]) -> None:
        self.status.setText(text)
        self.status.setVisible(bool(text))
        mark(self.status, tone)
        self.adjustSize()

    def paintEvent(self, event) -> None:  # noqa: ANN001, N802
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), 2))
        p.drawRect(QRectF(self.rect()).adjusted(1, 1, -1, -1))

    # the card can be dragged anywhere by its background
    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)
            self.moved_by_user = True
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._drag = None
        super().mouseReleaseEvent(e)


def place_card(card: QRect, target: Optional[QRect], area: QRect, window: QRect, gap: int = 18) -> QPoint:
    """Top left of the card (all global coordinates): beside the target (right, left, below, above)
    where it fits on the screen ``area`` without covering it; else in a corner of the target or, with
    no target, the bottom right of ``window``."""
    w, h = card.width(), card.height()

    def clamp(x: int, y: int) -> QPoint:
        x = max(area.left() + 4, min(x, area.right() - w - 4))
        y = max(area.top() + 4, min(y, area.bottom() - h - 4))
        return QPoint(x, y)

    if target is None or target.isEmpty():
        return clamp(window.right() - w - 28, window.bottom() - h - 40)
    right, left = QPoint(target.right() + gap, target.top()), QPoint(target.left() - gap - w, target.top())
    below, above = QPoint(target.left(), target.bottom() + gap), QPoint(target.left(), target.top() - gap - h)
    # a menu title or a button: below it, so the card does not hide the menu bar or the buttons beside it
    candidates = (below, right, left, above) if target.height() < 60 else (right, left, below, above)
    for pt in candidates:
        c = clamp(pt.x(), pt.y())
        if not QRect(c, card.size()).intersects(target.adjusted(-6, -6, 6, 6)):
            return c
    # a big target (the image, a whole panel): inside it, bottom right
    return clamp(target.right() - w - 24, target.bottom() - h - 24)
