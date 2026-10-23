"""
The tutorial runner and the vocabulary tutorials are written in.

A Step's callables all take one argument, the Ctx (``c.win`` the main window, ``c.session``,
``c.memo`` a dict the steps of one run share).  Any of them may fail (a panel was closed, a dialog is
not open yet): the runner treats an error as "nothing to point at" / "not done yet" and carries on.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from PyQt6 import sip
from PyQt6.QtCore import QObject, QPoint, QRect, QSettings, QTimer, pyqtSignal
from PyQt6.QtWidgets import QAbstractButton, QApplication, QDockWidget, QFileDialog, QGroupBox, QScrollArea, QWidget

from .overlay import Spotlight, StepCard, place_card

TargetResult = Union[QWidget, tuple, None]       # a widget, (widget, QRect in its coordinates) or nothing


@dataclass
class Ctx:
    win: Any
    memo: dict = field(default_factory=dict)

    @property
    def session(self):  # noqa: ANN201
        return self.win.session


@dataclass
class Step:
    title: str
    text: str                                                   # rich text
    target: Optional[Callable[[Ctx], TargetResult]] = None      # what to point at
    enter: Optional[Callable[[Ctx], None]] = None               # on showing the step: raise a panel, note a count
    done: Optional[Callable[[Ctx], bool]] = None                # the task of the step is done
    show: Optional[Callable[[Ctx], None]] = None                # "Do it for me"
    show_text: str = "Do it for me"
    task: str = ""                                              # "Your turn: …" while not done
    advance: bool = False                                       # go on by itself once done (else the user clicks Next)
    dim: bool = True                                            # dim the window around the target


@dataclass
class Tutorial:
    key: str
    title: str
    summary: str
    steps: list[Step]
    level: str = "Basics"
    minutes: int = 5
    needs_image: bool = True               # an image must be open: the runner starts with the open step


# ---------------------------------------------------------------------------------------------------- targets
def _norm(text: str) -> str:
    return text.replace("&", "").replace("…", "").replace("...", "").strip().lower()


def menu(title: str) -> Callable[[Ctx], TargetResult]:
    """A menu of the menu bar, e.g. menu("Detect")."""
    def find(c: Ctx) -> TargetResult:
        bar = c.win.menuBar()
        for a in bar.actions():
            if _norm(a.text()) == _norm(title):
                return bar, bar.actionGeometry(a)
        return None
    return find


def dock(name: str) -> Callable[[Ctx], TargetResult]:
    """A panel of the main window, e.g. dock("Files")."""
    return lambda c: c.win.docks[name]


def tool(key: str) -> Callable[[Ctx], TargetResult]:
    """A tool button of the left tool bar, e.g. tool("polygon")."""
    return lambda c: c.win.tool_toolbar.widgetForAction(c.win.tool_actions[key])


def find_button(root: Optional[QWidget], text: str) -> Optional[QWidget]:
    """The first visible button below ``root`` whose text starts with ``text`` (ignoring &, … and case)."""
    if root is None:
        return None
    want = _norm(text)
    hits = [b for b in root.findChildren(QAbstractButton) if _norm(b.text()).startswith(want)]
    visible = [b for b in hits if b.isVisible()]
    return (visible or hits or [None])[0]


def find_group(root: Optional[QWidget], title: str) -> Optional[QWidget]:
    if root is None:
        return None
    return next((g for g in root.findChildren(QGroupBox) if _norm(g.title()) == _norm(title)), None)


def window(class_name: str) -> Optional[QWidget]:
    """The visible top-level window of that class (dialogs the program opened), if any."""
    for w in QApplication.topLevelWidgets():
        if type(w).__name__ == class_name and w.isVisible():
            return w
    return None


def button(text: str, within: Callable[[Ctx], Optional[QWidget]]) -> Callable[[Ctx], TargetResult]:
    return lambda c: find_button(within(c), text)


def group(title: str, within: Callable[[Ctx], Optional[QWidget]]) -> Callable[[Ctx], TargetResult]:
    return lambda c: find_group(within(c), title)


def in_window(class_name: str, part: Optional[Callable[[QWidget], TargetResult]] = None) -> Callable[[Ctx], TargetResult]:
    """A dialog the program opened (or a part of it)."""
    def find(_c: Ctx) -> TargetResult:
        w = window(class_name)
        return w if w is None or part is None else part(w)
    return find


def raise_dock(c: Ctx, name: str) -> None:
    d = c.win.docks[name]
    d.show()
    d.raise_()


def later(fn: Callable[[], Any]) -> None:
    """Run after the current event: for actions that open a modal dialog (exec blocks)."""
    QTimer.singleShot(0, fn)


# ---------------------------------------------------------------------------------------------------- runner
class Guide(QObject):
    """Runs one tutorial at a time over a main window."""
    finished = pyqtSignal(str, bool)              # tutorial key, completed

    TICK_MS = 200
    DONE_KEY = "tutorials/done"

    def __init__(self, win: QWidget):
        super().__init__(win)
        self.win = win
        self.tutorial: Optional[Tutorial] = None
        self.steps: list[Step] = []
        self.index = 0
        self.ctx = Ctx(win)
        self._done = False
        self._advance_pending = False
        self._scrolled: Optional[int] = None          # id of the target already scrolled into view
        self._errors: set = set()
        self._make_widgets()
        self.timer = QTimer(self)
        self.timer.setInterval(self.TICK_MS)
        self.timer.timeout.connect(self.tick)

    def _make_widgets(self) -> None:
        self.card = StepCard(self.win)
        self.card.nextClicked.connect(self.next)
        self.card.backClicked.connect(self.back)
        self.card.showClicked.connect(self._show_me)
        self.card.closeClicked.connect(lambda: self.stop(False))
        self.card.hide()
        self.spot = Spotlight(self.win)
        self.spot.hide()

    def _ensure_widgets(self) -> None:
        """The card and spotlight sit on the window the user can use, often a dialog; when that dialog is
        deleted they go with it, so make them again (on the main window) and show the current step."""
        if not (sip.isdeleted(self.card) or sip.isdeleted(self.spot)):
            return
        for w in (self.card, self.spot):
            if not sip.isdeleted(w):
                w.deleteLater()
        self._make_widgets()
        if self.running:
            step = self.steps[self.index]
            self.card.set_step(self.tutorial.title, self.index, len(self.steps), step.title, step.text)
            self.card.show_me.setVisible(step.show is not None)
            self.card.show_me.setText(step.show_text)
            self._update_status(step)

    # ------------------------------------------------------------------ control
    @property
    def running(self) -> bool:
        return self.tutorial is not None

    def start(self, tutorial: Tutorial) -> None:
        from .content import OPEN_IMAGE
        if self.running:
            self.stop(False)
        self.tutorial = tutorial
        self.ctx = Ctx(self.win)
        steps = list(tutorial.steps)
        if tutorial.needs_image and not self.win.session.current_id and OPEN_IMAGE not in steps:
            steps.insert(1 if steps and steps[0].target is None else 0, OPEN_IMAGE)
        self.steps = steps
        self.timer.start()
        self.go(0)

    def stop(self, completed: bool) -> None:
        if not self.running:
            return
        key = self.tutorial.key
        self.timer.stop()
        self._ensure_widgets()
        self.card.hide()
        self.spot.hide()
        self.tutorial = None
        if completed:
            mark_done(key)
        self.finished.emit(key, completed)

    def go(self, index: int) -> None:
        if not self.running:
            return
        self.index = max(0, min(index, len(self.steps) - 1))
        self._ensure_widgets()
        step = self.steps[self.index]
        self._done = False
        self._advance_pending = False
        self._scrolled = None
        self._call(step.enter)
        self.card.set_step(self.tutorial.title, self.index, len(self.steps), step.title, step.text)
        self.card.show_me.setVisible(step.show is not None)
        self.card.show_me.setText(step.show_text)
        self._update_status(step)
        self.tick()

    def next(self) -> None:
        if not self.running:
            return
        if self.index >= len(self.steps) - 1:
            self.stop(True)
        else:
            self.go(self.index + 1)

    def back(self) -> None:
        self.go(self.index - 1)

    def _show_me(self) -> None:
        step = self.steps[self.index] if self.running else None
        if step is not None and step.show is not None:
            self._call(step.show)
            self.tick()

    # ------------------------------------------------------------------ each tick
    def _call(self, fn, default=None):  # noqa: ANN001, ANN202
        if fn is None:
            return default
        try:
            return fn(self.ctx)
        except Exception:  # noqa: BLE001 - a tutorial must never break the program
            text = traceback.format_exc()
            if text not in self._errors:             # report each problem once
                self._errors.add(text)
                sys.stderr.write("tutorial step: " + text)
            return default

    def _update_status(self, step: Step) -> None:
        if step.done is None:
            self.card.set_status("", None)
        elif self._done:
            last = self.index >= len(self.steps) - 1
            self.card.set_status("✓  Done" + ("" if last else "  — moving on…" if step.advance else "  — click Next"), "ok")
        else:
            self.card.set_status("Your turn: " + (step.task or "follow the steps above."), "warn")

    def _host(self, target_window: Optional[QWidget]) -> QWidget:
        """The window the card belongs to: one the user can click in now."""
        modal = QApplication.activeModalWidget()
        if modal is not None and modal.isVisible() and not isinstance(modal, QFileDialog):
            return modal
        return target_window or self.win

    def _resolve(self, step: Step) -> tuple[Optional[QWidget], Optional[QRect]]:
        res = self._call(step.target)
        if isinstance(res, tuple):
            w, r = res
        else:
            w, r = res, None
        if not isinstance(w, QWidget) or not w.isVisible():
            return None, None
        if r is None:
            r = w.rect()
        if self._scrolled != id(w):                   # a target inside a scroll area: bring it into view once
            self._scrolled = id(w)
            p = w.parentWidget()
            while p is not None:
                if isinstance(p, QScrollArea):
                    p.ensureWidgetVisible(w, 20, 20)
                    break
                if isinstance(p, QDockWidget) or p.isWindow():
                    break
                p = p.parentWidget()
        return w, r

    def tick(self) -> None:
        if not self.running:
            return
        self._ensure_widgets()
        step = self.steps[self.index]
        w, r = self._resolve(step)
        tw = w.window() if w is not None else None

        # spotlight on the target's window
        if tw is not None:
            if self.spot.parentWidget() is not tw:
                self.spot.setParent(tw)
            self.spot.follow_parent()
            top_left = w.mapTo(tw, r.topLeft()) if w is not tw else r.topLeft()
            hole = QRect(top_left, r.size()).intersected(tw.rect())
            self.spot.set_hole(hole, step.dim)
            self.spot.show()
        else:
            self.spot.hide()

        # the card on the window the user can use now, beside the target
        host = self._host(tw)
        if self.card.parentWidget() is not host:
            pos = self.card.pos()
            self.card.setParent(host, self.card.windowFlags())
            self.card.move(pos)
        if not self.card.moved_by_user:
            screen = (host.screen() or QApplication.primaryScreen()).availableGeometry()
            target = None
            if w is not None:
                target = QRect(w.mapToGlobal(r.topLeft()), r.size())
            frame = QRect(host.mapToGlobal(QPoint(0, 0)), host.size())
            self.card.move(place_card(self.card.frameGeometry(), target, screen, frame))
        if not self.card.isVisible():
            self.card.show()
            self.card.raise_()

        # done?
        if step.done is not None and not self._done and self._call(step.done, False):
            self._done = True
            self._update_status(step)
            if step.advance and self.index < len(self.steps) - 1 and not self._advance_pending:
                self._advance_pending = True
                k = self.index
                QTimer.singleShot(1100, lambda: self.running and self.index == k and self.next())


# ---------------------------------------------------------------------------------------------------- progress
def _settings() -> QSettings:
    return QSettings("ScanBound", "ScanBound")


def done_tutorials() -> set[str]:
    v = _settings().value(Guide.DONE_KEY, [])
    return set(v if isinstance(v, list) else [v] if v else [])


def mark_done(key: str) -> None:
    _settings().setValue(Guide.DONE_KEY, sorted(done_tutorials() | {key}))
