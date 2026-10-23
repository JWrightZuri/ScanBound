"""
The ScanBound logo as the application icon (title bar, taskbar, dialogs such as Help → About).
"""

from __future__ import annotations

import os
import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

LOGO = os.path.join(os.path.dirname(__file__), "resources", "sb_logo.png")
APP_USER_MODEL_ID = "ScanBound.ScanBound"     # same id as the installer's shortcuts, so they group in the taskbar


def app_icon() -> QIcon:
    return QIcon(LOGO) if os.path.exists(LOGO) else QIcon()


def apply_branding(app: QApplication) -> None:
    """Set the window icon for every window and, on Windows, make the taskbar show it instead of Python's."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except (AttributeError, OSError):
            pass
    app.setWindowIcon(app_icon())
