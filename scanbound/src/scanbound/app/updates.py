"""
The update prompt: at start-up (in the background, silent when offline) and Help → Check for updates….

Shows the new version's changes, then Install and restart / Skip this version / Not now.  Installing
asks to save the open project, downloads the update with a progress bar, hands over to the installer
or the update helper, and closes ScanBound; the new version starts by itself.  The logic is in the
Qt-free :mod:`scanbound.updater`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog,
                             QPushButton, QTextBrowser, QVBoxLayout, QWidget)

from .. import __version__, updater
from ..updater import Install, Release
from .qtutil import run_job

if TYPE_CHECKING:
    from .mainwindow import MainWindow

APP_NAME = "ScanBound"
STARTUP_DELAY_MS = 2500                  # after the window is up and a project given on the command line has opened


class UpdateDialog(QDialog):
    """What is new in ``release``; ``choice`` is "install", "skip" or "later" afterwards."""

    def __init__(self, release: Release, install: Install, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.release, self.install, self.choice = release, install, "later"
        self.setWindowTitle("Update available")
        self.resize(560, 440)
        head = QLabel(f"<b>ScanBound {html.escape(release.version)} is available.</b> You have {__version__}.")
        notes = QTextBrowser()
        notes.setOpenExternalLinks(True)
        items = release.change_list()
        body = "".join(f"<li>{html.escape(t)}</li>" for t in items) or "<li>No release notes.</li>"
        link = f'<p><a href="{html.escape(release.page)}">Full release notes</a></p>' if release.page else ""
        notes.setHtml(f"<p>What's new:</p><ul>{body}</ul>{link}")
        safe = QLabel("Updating replaces only the program. Your projects, data, settings and annotations are "
                      "kept, and projects from older versions open as before.")
        safe.setWordWrap(True)
        self.startup = QCheckBox("Check for updates when ScanBound starts")
        self.startup.setChecked(bool(updater.load_state().get("check_on_startup", True)))

        self.b_install = QPushButton("Install and restart" if install.can_update else "Open release page")
        self.b_install.setDefault(True)
        self.b_install.clicked.connect(self._install)
        b_skip = QPushButton("Skip this version")
        b_skip.clicked.connect(lambda: self._done("skip"))
        b_later = QPushButton("Not now")
        b_later.clicked.connect(lambda: self._done("later"))
        buttons = QHBoxLayout()
        buttons.addWidget(b_skip)
        buttons.addStretch(1)
        buttons.addWidget(b_later)
        buttons.addWidget(self.b_install)

        lay = QVBoxLayout(self)
        lay.addWidget(head)
        lay.addWidget(notes, 1)
        if not install.can_update and install.explain():
            why = QLabel(install.explain())
            why.setWordWrap(True)
            lay.addWidget(why)
        lay.addWidget(safe)
        lay.addWidget(self.startup)
        lay.addLayout(buttons)

    def _install(self) -> None:
        if not self.install.can_update:
            QDesktopServices.openUrl(QUrl(self.release.page))
            self._done("later")
            return
        self._done("install")

    def _done(self, choice: str) -> None:
        self.choice = choice
        updater.save_state(check_on_startup=self.startup.isChecked(),
                           **({"skipped_version": self.release.version} if choice == "skip" else {}))
        self.accept()

    def reject(self) -> None:          # Esc / the close button: same as Not now
        self._done("later")


class Updates:
    """Owns the checks and the install for one main window."""

    def __init__(self, window: "MainWindow"):
        self.w = window
        self.busy = False

    # ------------------------------------------------------------------ checking
    def check_at_startup(self) -> None:
        self._report_last_update()
        if updater.check_on_startup():
            QTimer.singleShot(STARTUP_DELAY_MS, lambda: self.check(manual=False))

    def check(self, manual: bool = True) -> None:
        if self.busy:
            return
        self.busy = True

        def done(release: Optional[Release]) -> None:
            self.busy = False
            if release is None:
                if manual:
                    QMessageBox.information(self.w, APP_NAME, f"ScanBound {__version__} is the latest version.")
                return
            if manual or updater.should_prompt(release):
                self.offer(release)

        def failed(tb: str) -> None:
            self.busy = False
            if manual:
                QMessageBox.warning(self.w, APP_NAME, "Could not check for updates.\n\n" + tb.strip().splitlines()[-1])

        run_job(updater.check_for_update, done=done, failed=failed)

    def offer(self, release: Release) -> None:
        install = updater.detect_install()
        dlg = UpdateDialog(release, install, self.w)
        dlg.exec()
        if dlg.choice == "install":
            self.install(release, install)

    # ------------------------------------------------------------------ installing
    def install(self, release: Release, install: Install) -> None:
        if not self.w._confirm_discard():                # save (or drop) the open project first
            return
        asset = release.installer
        size = f" ({asset.size / 1e6:.0f} MB)" if install.kind == "installer" and asset and asset.size else ""
        dlg = QProgressDialog(f"Downloading ScanBound {release.version}{size} …", "Cancel", 0, 0, self.w)
        dlg.setWindowTitle("Updating ScanBound")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.show()
        state = {"cancel": False}
        dlg.canceled.connect(lambda: state.update(cancel=True))

        def progress(p: tuple[int, int]) -> None:
            done, total = p
            if total:
                dlg.setMaximum(1000)
                dlg.setValue(int(1000 * done / total))

        def fetch(progress=None):  # noqa: ANN001, ANN202
            cb = (lambda d, t: progress((d, t))) if progress else None
            if install.kind == "installer":
                return updater.download_installer(release, cb, lambda: state["cancel"])
            return updater.stage_source_update(release, cb, lambda: state["cancel"])

        def ready(path: str) -> None:
            dlg.close()
            try:
                if install.kind == "installer":
                    updater.launch_installer(path, release)
                else:
                    updater.launch_source_update(install, path, release)
            except Exception as exc:  # noqa: BLE001 - nothing has changed yet: report and keep running
                QMessageBox.warning(self.w, APP_NAME, f"Could not start the update: {exc}")
                return
            self.w.session.info(f"Installing ScanBound {release.version}; ScanBound restarts when it is done.")
            self.w.close()
            QApplication.quit()

        def failed(tb: str) -> None:
            dlg.close()
            if not state["cancel"]:
                QMessageBox.warning(self.w, APP_NAME, "The update could not be downloaded, nothing was changed.\n\n"
                                    + tb.strip().splitlines()[-1])

        run_job(fetch, done=ready, failed=failed, progress=progress)

    # ------------------------------------------------------------------ after an update
    def _report_last_update(self) -> None:
        result = updater.finish_pending()
        if "updated" in result:
            self.w.session.info(f"ScanBound was updated to {result['updated']}.")
        if "failed" in result:
            QMessageBox.warning(self.w, APP_NAME, "The last update did not finish, so the previous version was "
                                f"kept and nothing else changed.\n\n{result['failed']}\n\n"
                                f"The details are in {updater.UPDATES_DIR}.")
        if result.get("reinstall_yolo"):
            QTimer.singleShot(500, self._offer_yolo)

    def _offer_yolo(self) -> None:
        r = QMessageBox.question(self.w, APP_NAME, "YOLO was installed before the update, and the update replaced "
                                 "the Python it was installed in. Install YOLO again now?")
        updater.save_state(reinstall_yolo=None)
        if r == QMessageBox.StandardButton.Yes:
            self.w.install_yolo()
