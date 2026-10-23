"""YOLO is optional: its commands are disabled until it is installed, and the installer runs pip."""

from __future__ import annotations

import os
import shutil
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def session(qapp, tmp_path):
    from scanbound.app.session import Session
    shutil.copy(os.path.join(DATA, "aniso_0020.sxm"), tmp_path / "scan.sxm")
    s = Session()
    s.open_folder(str(tmp_path))
    return s


@pytest.mark.parametrize("installed", [False, True])
def test_yolo_commands_follow_the_install(monkeypatch, session, installed):
    from scanbound.app import yolo_install
    from scanbound.app.dialogs.ml import TrainDialog
    from scanbound.app.mainwindow import MainWindow
    monkeypatch.setattr(yolo_install, "yolo_available", lambda: installed)
    monkeypatch.setattr("scanbound.app.dialogs.ml.yolo_available", lambda: installed)

    w = MainWindow(session)
    assert all(a.isEnabled() == installed for a in w.yolo_actions)
    assert all(a.isVisible() for a in w.yolo_actions)              # shown either way
    assert w.install_yolo_action.isVisible() == (not installed)

    dlg = TrainDialog(session, w)
    assert dlg.training.isEnabled() == installed
    assert dlg.b_train.isEnabled() == installed
    assert dlg.b_export.isEnabled()                                 # dataset export never needs YOLO
    assert dlg.yolo_note.isHidden() == installed
    dlg._set_yolo(True)                                             # what a finished install does
    assert dlg.training.isEnabled() and dlg.b_train.isEnabled() and dlg.yolo_note.isHidden()
    w.close()


def test_pip_arguments():
    from scanbound.app.yolo_install import CPU_INDEX, REQUIREMENT, pip_arguments
    assert pip_arguments()[:3] == ["-m", "pip", "install"] and REQUIREMENT in pip_arguments()
    assert CPU_INDEX not in pip_arguments() and CPU_INDEX in pip_arguments(cpu_only=True)


def test_installer_reports_success_and_failure(qapp, monkeypatch):
    from scanbound.app import yolo_install
    calls = []
    # stand-in for pip: a Python one-liner that prints and exits with the given code
    for code, found in ((3, False), (0, True)):
        monkeypatch.setattr(yolo_install, "pip_arguments", lambda cpu_only=False, c=code: ["-c", f"print('pip'); raise SystemExit({c})"])
        monkeypatch.setattr(yolo_install, "yolo_available", lambda f=found: f)
        dlg = yolo_install.YoloInstallDialog()
        dlg.installed.connect(lambda: calls.append(code))
        dlg.start()
        assert dlg.proc is not None and dlg.proc.waitForFinished(20000)
        qapp.processEvents()
        log = dlg.log.toPlainText()
        assert "pip" in log and sys.executable in log
        assert ("YOLO is installed" in log) == found and ("exit code 3" in log) == (not found)
        assert dlg.b_install.isHidden() == found
    assert calls == [0]
