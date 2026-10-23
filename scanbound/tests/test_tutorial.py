"""Tutorials: every step of the getting-started tour can be done with its "Do it for me" button (including
the modal adapt dialog), the advanced tours find their targets, and Help opens them."""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from scanbound.app.mainwindow import MainWindow
    from scanbound.app.tutorial import engine, sample
    ini = str(tmp_path / "tutorials.ini")                 # tutorial progress stays out of the user's settings
    monkeypatch.setattr(engine, "_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    monkeypatch.setattr(sample, "FOLDER", str(tmp_path / "practice"))
    w = MainWindow()
    w.show()
    yield w
    w.session.project.dirty = False
    if getattr(w, "_guide", None) is not None:
        w._guide.stop(False)
    w.close()


def _wait(qapp, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_practice_image_has_islands():
    import numpy as np
    from scanbound.app.tutorial.sample import practice_image
    img = practice_image()
    assert img.dtype == np.uint16 and img.shape == (512, 512)
    assert img.max() > 60000


def test_getting_started_runs_end_to_end_with_do_it_for_me(qapp, win):
    """A driver on a timer (it keeps running inside the dialogs' own event loops) presses "Do it for me" on
    each step that has one, waits until the step is done and goes on."""
    from PyQt6.QtCore import QTimer
    from scanbound.app.tutorial.content import GETTING_STARTED
    g = win.guide
    seen, pressed = [], set()
    targets_missing = []

    def drive() -> None:
        if not g.running:
            return
        step = g.steps[g.index]
        if not seen or seen[-1] != step.title:
            seen.append(step.title)
        if step.target is not None and g._resolve(step)[0] is None and step.title not in targets_missing:
            targets_missing.append(step.title)
        if step.done is not None and not g._done:
            if step.show is not None and (g.index, "show") not in pressed:
                pressed.add((g.index, "show"))
                g._show_me()
            return
        if step.done is not None and step.advance and g.index < len(g.steps) - 1:
            return                                    # it moves on by itself
        g.next()

    timer = QTimer()
    timer.setInterval(60)
    timer.timeout.connect(drive)
    timer.start()
    g.start(GETTING_STARTED)
    ok = _wait(qapp, lambda: not g.running, 90)
    timer.stop()
    assert ok, f"stuck at step {g.index}: {g.steps[g.index].title} (seen {seen})"
    s = win.session
    assert [x.title for x in GETTING_STARTED.steps] == [t for t in seen if t != "Open an image"] or \
        set(x.title for x in GETTING_STARTED.steps) <= set(seen)
    assert s.entry.chain.active                                       # processed
    assert "molecule" in s.project.labels.names                         # label made
    assert not [a for a in s.annotations() if a.status == "proposed"]
    assert [a for a in s.annotations() if a.status == "accepted"]      # the pipeline found the molecules
    from scanbound.app.tutorial.engine import done_tutorials
    assert "getting_started" in done_tutorials()
    # the dialog steps are only checked while the dialog is open; the rest must find their widget
    assert not [t for t in targets_missing if t not in ("Open an image", "Save the pipeline", "The pipeline steps",
                                                         "Check the preview", "Objects and labels")], targets_missing


def test_advanced_tutorials_find_their_targets(qapp, win):
    from scanbound.app.tutorial.content import TUTORIALS
    from scanbound.app.tutorial.sample import practice_folder
    win._open_path(practice_folder())
    assert _wait(qapp, lambda: bool(win.session.current_id))
    g = win.guide
    # targets inside windows that a step opens are only there after its button
    needs_window = {"Choose what to analyse", "Run and export", "Length, width, branching", "Alignment to the herringbone",
                    "Run them", "Substrate and defaults", "Adjust single molecules", "Export", "Size, fonts, bars",
                    "Save or copy"}
    for t in TUTORIALS[1:]:
        g.start(t)
        for k, step in enumerate(g.steps):
            g.go(k)
            qapp.processEvents()
            if step.target is None or step.title in needs_window:
                continue
            w, _r = g._resolve(step)
            assert w is not None, f"{t.title}: {step.title}"
            assert g.card.isVisible()
        g.stop(False)
    assert not g.card.isVisible() and not g.spot.isVisible()


def test_help_menu_lists_tutorials(win):
    help_menu = next(a.menu() for a in win.menuBar().actions() if a.text().replace("&", "") == "Help")
    texts = [a.text() for a in help_menu.actions()]
    assert "Tutorials…" in texts and "Getting started tour" in texts


def test_chooser_lists_every_tutorial(qapp, win):
    from scanbound.app.tutorial import TUTORIALS
    from scanbound.app.tutorial.chooser import TutorialChooser
    dlg = TutorialChooser(win)
    keys = [dlg.list.item(k).data(0x0100) for k in range(dlg.list.count())]
    assert [k for k in keys if k] == [t.key for t in TUTORIALS]
    assert dlg._tutorial() is TUTORIALS[0]                          # nothing done yet: the tour first


def test_card_placement_avoids_the_target():
    from PyQt6.QtCore import QRect
    from scanbound.app.tutorial.overlay import place_card
    screen = QRect(0, 0, 1600, 1000)
    card = QRect(0, 0, 380, 300)
    target = QRect(100, 100, 200, 40)
    p = place_card(card, target, screen, screen)
    assert not QRect(p, card.size()).intersects(target)
    assert p.y() > target.bottom()                                      # a small target: the card goes below
    panel = QRect(1300, 50, 290, 900)                                   # a tall panel on the right: left of it
    p = place_card(card, panel, screen, screen)
    assert QRect(p, card.size()).right() < panel.left()
    big = QRect(0, 0, 1600, 1000)
    p = place_card(card, big, screen, screen)
    assert screen.contains(QRect(p, card.size()))


def test_card_survives_a_deleted_dialog(qapp, win):
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtWidgets import QDialog
    from scanbound.app.tutorial.content import GETTING_STARTED
    g = win.guide
    g.start(GETTING_STARTED)
    dlg = QDialog(win)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.open()                                            # window-modal: the card moves onto it
    qapp.processEvents()
    g.tick()
    assert g.card.parentWidget() is dlg
    dlg.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    g.tick()                                              # used to raise "StepCard has been deleted"
    g.next()
    assert g.card.isVisible() and g.card.parentWidget() is win
    g.stop(False)
