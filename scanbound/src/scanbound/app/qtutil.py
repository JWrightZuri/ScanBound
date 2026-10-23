"""Small Qt helpers: numpy -> QImage, colours, background jobs."""

from __future__ import annotations

import os
import re
import traceback
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap


def to_qimage(arr: np.ndarray) -> QImage:
    """(H, W) uint8 gray, (H, W, 3) RGB or (H, W, 4) RGBA uint8 -> QImage (owns a copy)."""
    a = np.ascontiguousarray(arr)
    if a.dtype != np.uint8:
        raise TypeError("expected uint8")
    h, w = a.shape[:2]
    if a.ndim == 2:
        img = QImage(a.data, w, h, a.strides[0], QImage.Format.Format_Grayscale8)
    elif a.shape[2] == 3:
        img = QImage(a.data, w, h, a.strides[0], QImage.Format.Format_RGB888)
    else:
        img = QImage(a.data, w, h, a.strides[0], QImage.Format.Format_RGBA8888)
    return img.copy()


def save_file_name(parent: Any, caption: str, start: str, filters: str) -> str:
    """QFileDialog.getSaveFileName that behaves the same on every platform.

    A bare file name as ``start`` opens in the last used folder (or the home folder) rather than the
    working directory, which is "/" for an app started from the macOS Finder.  When the user types a
    name without an extension, the selected filter's extension is added: the native macOS dialog does
    that itself, the Linux (GTK / KDE / Qt) dialogs do not.
    """
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QFileDialog

    if not os.path.dirname(start):
        folder = str(QSettings("ScanBound", "ScanBound").value("last_dir", "") or "")
        start = os.path.join(folder if os.path.isdir(folder) else os.path.expanduser("~"), start)
    path, chosen = QFileDialog.getSaveFileName(parent, caption, start, filters)
    if path and not os.path.splitext(os.path.basename(path))[1]:
        m = re.search(r"\*(\.[\w.]+)", chosen or filters)
        if m:
            path += m.group(1)
    return path


def qcolor(color: str, alpha: int = 255) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def swatch(color: str, size: int = 14) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    return QIcon(pm)


class _Signals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)


class Job(QRunnable):
    """``fn(*args, progress=callback)`` (if it accepts ``progress``) on the thread pool."""

    def __init__(self, fn: Callable[..., Any], *args: Any, with_progress: bool = False, **kwargs: Any):
        super().__init__()
        self.fn, self.args, self.kwargs, self.with_progress = fn, args, kwargs, with_progress
        self.signals = _Signals()
        self.cancelled = False

    def run(self) -> None:
        try:
            if self.with_progress:
                self.kwargs["progress"] = self.signals.progress.emit
            out = self.fn(*self.args, **self.kwargs)
        except Exception:  # noqa: BLE001 - reported to the GUI
            if not self.cancelled:
                self.signals.failed.emit(traceback.format_exc())
            return
        if not self.cancelled:
            self.signals.done.emit(out)


_JOBS: set[Job] = set()


def run_job(fn: Callable[..., Any], *args: Any, done: Optional[Callable[[Any], None]] = None,
            failed: Optional[Callable[[str], None]] = None, progress: Optional[Callable[[Any], None]] = None,
            pool: Optional[QThreadPool] = None, **kwargs: Any) -> Job:
    job = Job(fn, *args, with_progress=progress is not None, **kwargs)
    job.setAutoDelete(False)
    _JOBS.add(job)

    def finish(cb: Optional[Callable[[Any], None]]) -> Callable[[Any], None]:
        def f(v: Any) -> None:
            # the worker thread may still be returning from run(): let go of the job a little later,
            # else Python deletes the QRunnable under it
            QTimer.singleShot(2000, lambda: _JOBS.discard(job))
            if cb is not None:
                cb(v)
        return f
    job.signals.done.connect(finish(done))
    job.signals.failed.connect(finish(failed))
    if progress is not None:
        job.signals.progress.connect(progress)
    (pool or QThreadPool.globalInstance()).start(job)
    return job
