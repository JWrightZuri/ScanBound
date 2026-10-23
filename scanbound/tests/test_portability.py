"""Behaviour that differs between Windows, macOS and Linux: paths, file names, dialogs, start-up."""

from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from scanbound.project import Project


def test_paths_written_on_windows_open_elsewhere(tmp_path):
    (tmp_path / "sub").mkdir()
    p = Project(str(tmp_path))
    assert p.abs("sub\\scan.sxm") == os.path.join(str(tmp_path), "sub", "scan.sxm")
    assert p.rel(os.path.join(str(tmp_path), "sub", "scan.sxm")) == "sub/scan.sxm"


def test_old_annotation_names_resolve_regardless_of_folder_and_case(tmp_path):
    cv2.imwrite(str(tmp_path / "Image_01.jpg"), np.full((8, 8), 128, np.uint8))
    p = Project.open_folder(str(tmp_path))
    resolve = p.legacy_resolver()
    want = resolve("Image_01.jpg")
    assert want is not None
    assert resolve("C:\\Users\\someone\\ScanBound\\saved_output\\Image_01.jpg") == want
    assert resolve("image_01.JPG") == want
    assert resolve("other.jpg") is None


def test_module_entry_points_do_nothing_on_import():
    """multiprocessing's spawn start method (Windows, macOS, Detector.detect_many) imports __main__
    again in every worker; that must not start the CLI or the app."""
    code = "import scanbound.__main__, scanbound.app.__main__; print('imported')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.returncode == 0 and "imported" in out.stdout, out.stderr


def test_opengl_mode_from_environment(monkeypatch):
    pytest.importorskip("PyQt6.QtGui")
    from PyQt6.QtGui import QSurfaceFormat
    from scanbound.app.widgets import glview

    before = QSurfaceFormat.defaultFormat()
    try:
        monkeypatch.setenv(glview.GL_ENV, "core")
        assert glview.configure_opengl() == "core"
        fmt = QSurfaceFormat.defaultFormat()
        assert fmt.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile and fmt.version() == (4, 1)
        monkeypatch.setenv(glview.GL_ENV, "")
        assert glview.configure_opengl() == "default"
    finally:
        QSurfaceFormat.setDefaultFormat(before)


@pytest.mark.parametrize("typed, chosen, expected", [
    ("results", "CSV files (*.csv)", "results.csv"),
    ("results.txt", "CSV files (*.csv)", "results.txt"),
    ("view", "PNG (*.png);;JPEG (*.jpg)", "view.png"),
    ("view", "JPEG (*.jpg)", "view.jpg"),
])
def test_save_dialog_adds_the_filter_extension(monkeypatch, tmp_path, typed, chosen, expected):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from scanbound.app import qtutil

    seen = {}

    def fake(parent, caption, start, filters):  # noqa: ANN001, ANN202
        seen["start"] = start
        return str(tmp_path / typed), chosen

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", fake)
    assert qtutil.save_file_name(None, "Save", "x.csv", "CSV files (*.csv)") == str(tmp_path / expected)
    assert os.path.isabs(seen["start"])          # never relative to the working directory
