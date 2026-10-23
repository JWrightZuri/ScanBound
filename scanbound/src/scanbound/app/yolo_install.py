"""
Optional YOLO support: is it installed, and a dialog that installs it into the running Python.

The standard install leaves out Ultralytics YOLO and PyTorch (the ``ml`` extra, about 0.7 GB), so
the commands that need them stay in the menus but are disabled until "Machine learning → Install
YOLO…" has run pip.  Dataset export, evaluation and feature clips work without it.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Optional

from PyQt6.QtCore import QProcess, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                             QVBoxLayout, QWidget)

REQUIREMENT = "ultralytics>=8.3"            # keep in step with the ml extra in pyproject.toml
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
INSTALL_HINT = "Needs YOLO, which is not installed: Machine learning → Install YOLO…"


def yolo_available() -> bool:
    """True when ``ultralytics`` can be imported (checked without importing it, which is slow)."""
    importlib.invalidate_caches()           # pick up a package pip installed while the app runs
    return importlib.util.find_spec("ultralytics") is not None


def python_executable() -> str:
    """The Python to run pip with: python.exe rather than the windowless pythonw.exe the installed app uses."""
    exe = sys.executable
    head, name = os.path.split(exe)
    if name.lower() == "pythonw.exe" and os.path.exists(os.path.join(head, "python.exe")):
        return os.path.join(head, "python.exe")
    return exe


def pip_arguments(cpu_only: bool = False) -> list[str]:
    """Arguments for ``sys.executable`` that install YOLO into this environment."""
    args = ["-m", "pip", "install", "--disable-pip-version-check", REQUIREMENT]
    if cpu_only:
        # PyPI's Linux torch wheels bundle CUDA (several GB); the CPU index has the small builds
        args += ["--extra-index-url", CPU_INDEX]
    return args


class YoloInstallDialog(QDialog):
    """Runs pip in a separate process and shows its output; emits ``installed`` when YOLO imports."""

    installed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Install YOLO")
        self.resize(720, 460)
        self.proc: Optional[QProcess] = None
        about = QLabel("Detecting and training with YOLO models needs Ultralytics YOLO and PyTorch. They are not "
                       "part of the standard install because they take about 0.7 GB.\n\n"
                       f"This installs {REQUIREMENT} and its dependencies into the Python ScanBound runs in:\n"
                       f"{python_executable()}")
        about.setWordWrap(True)
        self.cpu_only = QCheckBox("CPU-only PyTorch (much smaller download; no NVIDIA GPU training)")
        self.cpu_only.setChecked(True)
        self.cpu_only.setVisible(sys.platform.startswith("linux"))     # Windows / macOS wheels are CPU-only already
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.b_install = QPushButton("Install")
        self.b_install.setDefault(True)
        self.b_install.clicked.connect(self.start)
        self.b_close = QPushButton("Close")
        self.b_close.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.b_install)
        buttons.addWidget(self.b_close)
        lay = QVBoxLayout(self)
        lay.addWidget(about)
        lay.addWidget(self.cpu_only)
        lay.addWidget(self.log, 1)
        lay.addLayout(buttons)

    def start(self) -> None:
        args = pip_arguments(self.cpu_only.isVisible() and self.cpu_only.isChecked())
        self.log.appendPlainText(" ".join([python_executable(), *args]) + "\n")
        self.b_install.setEnabled(False)
        self.cpu_only.setEnabled(False)
        self.b_close.setText("Cancel")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._error)
        self.proc.start(python_executable(), args)

    def _read(self) -> None:
        if self.proc is not None:
            text = bytes(self.proc.readAllStandardOutput().data()).decode(errors="replace")
            self.log.insertPlainText(text)
            self.log.ensureCursorVisible()

    def _error(self, err: QProcess.ProcessError) -> None:
        if err == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(f"Could not start {python_executable()}.")
            self._done()

    def _finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self._read()
        self._done()
        if code == 0 and yolo_available():
            self.log.appendPlainText("\nYOLO is installed. The YOLO commands are now enabled.")
            self.b_install.setVisible(False)
            self.installed.emit()
        else:
            self.log.appendPlainText(f"\npip stopped with exit code {code}; see the messages above.")

    def _done(self) -> None:
        self.proc = None
        self.b_install.setEnabled(True)
        self.cpu_only.setEnabled(True)
        self.b_close.setText("Close")

    def reject(self) -> None:
        if self.proc is not None:
            r = QMessageBox.question(self, "Install YOLO", "Stop the installation?")
            if r != QMessageBox.StandardButton.Yes:
                return
            self.proc.finished.disconnect(self._finished)
            self.proc.kill()
            self.proc.waitForFinished(3000)
            self.log.appendPlainText("\nInstallation cancelled.")
            self._done()
            return
        super().reject()
