"""
The main window: canvas in the middle, tools on the left, panels around, menus on top.

Workflow: File → Open folder (STM data folder) → browse scans → draw or detect annotations →
correct them → measure → train / evaluate a detector.  (The proposal review workflow is kept in
the code but switched off: ``features.REVIEW``.)  Everything is saved in one project
file next to the data (``scanbound_project.json``); the data files are never modified.
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Optional

from PyQt6.QtCore import QPointF, QSettings, QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                             QMainWindow, QMenu, QMessageBox, QProgressDialog, QTabWidget, QToolBar, QToolButton,
                             QVBoxLayout, QWidget)

from ..annotations import GROUND_TRUTH, KINDS
from ..model.provenance import software_version
from ..project import DEFAULT_NAME
from .canvas import ImageCanvas
from .docks.browser import BrowserDock
from .docks.info import InfoDock
from .docks.labels import LabelsDock
from .docks.log import LogDock
from .docks.measurements import MeasurementsDock
from .docks.processing import ProcessingDock
from .docks.review import ReviewDock
from .dialogs.detect import untouched_result as _untouched_result
from .features import REVIEW
from .docks.spectra import SpectraDock
from .docks.surface import SurfaceDock
from .qtutil import run_job, save_file_name
from .session import Session
from .tools import TOOLS, Tool
from .icons import tool_icon
from .theme import apply_theme, mark
from .widgets.glview import configure_opengl

APP_NAME = "ScanBound"
LAYOUT_VERSION = 4                       # bump when the default panel layout changes
MAX_FILES = 3000                         # ask before opening more data files than this
TOOL_ORDER = ["select", "pan", "rect", "ellipse", "polygon", "paint", "point", "centre_line", "measure", "orientation",
              "lattice", "herringbone", "step_edge", "region"]


class MainWindow(QMainWindow):
    def __init__(self, session: Optional[Session] = None):
        super().__init__()
        self.session = session or Session()
        self.settings = QSettings("ScanBound", "ScanBound")
        self.session.hidden_statuses = {"rejected"}
        self.resize(1500, 950)

        # ---------------------------------------------------------------- centre
        self.canvas = ImageCanvas(self.session)
        self.channel = QComboBox()
        self.channel.setMinimumWidth(160)
        self.channel.setToolTip("Channel of the current file")
        self.channel.currentTextChanged.connect(self._on_channel)
        self.zoom_label = QLabel("")
        self.image_label = QLabel("")
        self.image_label.setStyleSheet("font-weight: bold;")
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        for text, tip, slot in (("◀", "Previous file (Page Up)", lambda: self.browser.step(-1)),
                                ("▶", "Next file (Page Down)", lambda: self.browser.step(1))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addWidget(self.image_label)
        bar.addStretch(1)
        bar.addWidget(QLabel("Channel"))
        bar.addWidget(self.channel)
        fit = QToolButton()
        fit.setText("Fit")
        fit.setToolTip("Fit the image to the window (F)")
        fit.clicked.connect(self.canvas.fit)
        bar.addWidget(fit)
        bar.addWidget(self.zoom_label)
        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addLayout(bar)
        cl.addWidget(self.canvas, 1)
        self.setCentralWidget(centre)

        # ---------------------------------------------------------------- docks
        self.browser = BrowserDock(self.session)
        self.labels = LabelsDock(self.session)
        self.review = ReviewDock(self.session) if REVIEW else None
        self.measurements = MeasurementsDock(self.session)
        self.processing = ProcessingDock(self.session)
        self.info = InfoDock(self.session)
        self.spectra = SpectraDock(self.session)
        self.log = LogDock(self.session)
        from .surface_overlay import SurfaceOverlay
        self.surface_overlay = SurfaceOverlay()
        self.canvas.scene_.addItem(self.surface_overlay)
        self.canvas.scene_.addItem(self.surface_overlay.map_item)
        self.surface = SurfaceDock(self.session, self.surface_overlay)
        self.surface.reportReady.connect(self.show_surface_report)
        self.docks: dict[str, QDockWidget] = {}
        self._dock("Files", self.browser, Qt.DockWidgetArea.LeftDockWidgetArea)
        self._dock("Labels", self.labels, Qt.DockWidgetArea.RightDockWidgetArea)
        if REVIEW:
            self._dock("Review", self.review, Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock("Measure", self.measurements, Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock("Display", self.processing, Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock("Info", self.info, Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock("Surface", self.surface, Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock("Spectroscopy", self.spectra, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._dock("Log", self.log, Qt.DockWidgetArea.BottomDockWidgetArea)
        # one panel on each side; the right-hand panels share one set of tabs at the top
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks)
        right = [self.docks[k] for k in ("Display", "Labels", "Review", "Measure", "Info", "Surface") if k in self.docks]
        for a, b in zip(right, right[1:]):
            self.tabifyDockWidget(a, b)
        self.docks["Display"].raise_()
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.docks["Spectroscopy"].hide()                # shown when a spectrum is opened
        self.docks["Log"].hide()                         # View > Panels > Log
        QTimer.singleShot(0, self._initial_sizes)

        self.cursor_label = QLabel("")
        self.statusBar().addWidget(self.cursor_label, 1)
        self.tool_hint = QLabel("")
        mark(self.tool_hint, "muted")
        self.statusBar().addPermanentWidget(self.tool_hint)

        self._build_tools()
        self._build_menus()

        # ---------------------------------------------------------------- wiring
        s = self.session
        s.imageChanged.connect(self._on_image)
        s.dirtyChanged.connect(lambda _d: self._update_title())
        s.projectChanged.connect(self._update_title)
        s.message.connect(self._on_message)
        s.annotationsChanged.connect(lambda _e: self._update_undo())
        self.canvas.cursorMoved.connect(self._on_cursor)
        self.canvas.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f} %"))
        self.canvas.contextRequested.connect(self._context_menu)
        self.canvas.toolChanged.connect(self._on_tool_changed)
        if REVIEW:
            self.review.centerRequested.connect(self.center_on)
        self.browser.spectrumActivated.connect(self._show_spectrum)
        self.spectra.positionsChanged.connect(self._show_positions)
        self.processing.processing.fourierEditRequested.connect(self.edit_fourier)
        self._profile_window = None
        self._surface_window = None
        self._analysis_window = None
        self._profile_timer = QTimer(self)
        self._profile_timer.setSingleShot(True)
        self._profile_timer.setInterval(120)
        self._profile_timer.timeout.connect(self._refresh_profiles)
        s.annotationsChanged.connect(lambda _e: self._profile_window is not None and self._profile_timer.start())
        s.imageDataChanged.connect(lambda _i: self._profile_window is not None and self._profile_timer.start())
        self._marker_items: list = []
        self._last_positions: list = []
        self.canvas.zoomChanged.connect(lambda _z: self._last_positions and self._show_positions(self._last_positions))

        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self._fit_to_screen()
        state = self.settings.value("state")
        if state is not None:
            self.restoreState(state, LAYOUT_VERSION)    # a layout saved by an older version is ignored
            self._fit_to_screen(panels=True)            # panels side by side can need more than the screen
        QTimer.singleShot(0, self.docks["Display"].raise_)   # the program always opens on the Display panel
        self._update_title()
        self._update_undo()
        self.canvas.set_tool("select")
        if s.project is not None:            # a session opened before the window: bring every panel up to date
            s.projectChanged.emit()
            if s.current_id:
                s.imageChanged.emit(s.current_id)

    def set_theme(self, mode: str) -> None:
        self.settings.setValue("theme", mode)
        apply_theme(mode)
        for key, act in self.tool_actions.items():          # icons follow the text colour
            act.setIcon(tool_icon(key))
        self.canvas.viewport().update()

    def _initial_sizes(self) -> None:
        if self.settings.value("state") is not None and self.settings.value("layout") == LAYOUT_VERSION:
            return
        self.resizeDocks([self.docks["Files"], self.docks["Display"]], [300, 360], Qt.Orientation.Horizontal)

    def _fit_to_screen(self, panels: bool = False) -> None:
        """Keep the window on the screen it opens on (a saved size can be larger than this screen);
        ``panels``: also stack a saved panel layout that is too wide for it."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        frame = self.frameGeometry().size() - self.geometry().size()
        if panels:
            self._fit_panels(avail.width() - frame.width(), avail.height() - frame.height())
        if self.isMaximized() or self.isFullScreen():
            return
        w = min(self.width(), avail.width() - frame.width())
        h = min(self.height(), avail.height() - frame.height())
        self.resize(max(w, 400), max(h, 300))
        g = self.frameGeometry()
        if not avail.contains(g):
            g.moveTopLeft(avail.topLeft())
            self.move(g.topLeft())

    def _fit_panels(self, max_w: int, max_h: int) -> None:
        """
        A saved layout with panels side by side (or stacked) can need a window larger than this
        screen; Windows then cannot place the window.  Stack the panels of each side as tabs until
        the window's minimum size fits.
        """
        def too_big() -> bool:
            m = self.minimumSizeHint()
            return m.width() > max_w or m.height() > max_h
        if not too_big():
            return
        areas = (Qt.DockWidgetArea.RightDockWidgetArea, Qt.DockWidgetArea.LeftDockWidgetArea,
                 Qt.DockWidgetArea.BottomDockWidgetArea, Qt.DockWidgetArea.TopDockWidgetArea)
        for area in areas:
            docks = [d for d in self.docks.values() if not d.isFloating() and not d.isHidden()
                     and self.dockWidgetArea(d) == area]
            for d in docks[1:]:
                self.tabifyDockWidget(docks[0], d)
            if docks:
                docks[0].raise_()
            if not too_big():
                return
        for d in self.docks.values():                     # still too big: the bottom panels go first
            if not d.isFloating() and not d.isHidden() and self.dockWidgetArea(d) == Qt.DockWidgetArea.BottomDockWidgetArea:
                d.hide()
        if too_big():
            self.session.warn("The panels need more room than this screen has: close some (View > Panels).")

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(f"dock_{title}")
        d.setWidget(widget)
        self.addDockWidget(area, d)
        self.docks[title] = d
        return d

    # ------------------------------------------------------------------ tools
    def _build_tools(self) -> None:
        tb = QToolBar("Tools", self)
        tb.setObjectName("tools")
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)
        group = QActionGroup(self)
        self.tool_actions: dict[str, QAction] = {}
        for key in TOOL_ORDER:
            cls = TOOLS[key]
            act = QAction(tool_icon(key), cls.title, self)
            act.setCheckable(True)
            act.setShortcut(QKeySequence(cls.shortcut))
            act.setToolTip(f"{cls.title} ({cls.shortcut})\n{cls.hint}")
            act.triggered.connect(lambda _=False, k=key: self.canvas.set_tool(k))
            group.addAction(act)
            tb.addAction(act)
            self.tool_actions[key] = act
        self.tool_toolbar = tb

    def _on_tool_changed(self, key: str) -> None:
        act = self.tool_actions.get(key)
        if act is not None and not act.isChecked():
            act.setChecked(True)
        t = TOOLS[key]
        own = t.deletable is not Tool.deletable
        self.tool_hint.setText(t.hint + (" Del deletes the hovered one." if own else ""))
        pw = getattr(self, "_profile_window", None)
        if key == "measure" and (pw is None or not pw.isVisible()):
            self.show_profiles()                     # the profiles of the lines appear as they are drawn
        self.canvas.setFocus()

    # ------------------------------------------------------------------ menus
    def _act(self, menu: QMenu, text: str, slot, shortcut=None, tip: str = "") -> QAction:  # noqa: ANN001
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setStatusTip(tip)
            a.setToolTip(tip)
        a.triggered.connect(lambda _=False: self._guard(slot))
        menu.addAction(a)
        return a

    def _guard(self, slot) -> None:  # noqa: ANN001
        try:
            slot()
        except Exception as exc:  # noqa: BLE001 - errors are reported, the app keeps running
            self.session.error(f"{exc}")
            traceback.print_exc()
            QMessageBox.warning(self, APP_NAME, f"{exc}")

    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._act(f, "Open data folder…", self.open_folder, QKeySequence.StandardKey.Open)
        self._act(f, "Open data folder with subfolders…", lambda: self.open_folder(recursive=True),
                  tip="All data files below the folder, e.g. several measurement days for one training set")
        self._act(f, "Open project…", self.open_project)
        self.recent_menu = f.addMenu("Recent folders")
        self._fill_recent()
        f.addSeparator()
        self._act(f, "Save project", self.save, QKeySequence.StandardKey.Save)
        self._act(f, "Save project as…", self.save_as, QKeySequence.StandardKey.SaveAs)
        self._act(f, "Rescan folder for new files", self.session.rescan)
        f.addSeparator()
        self._act(f, "Import annotations (old CSV)…", self.import_csv)
        self._act(f, "Export annotations (old CSV)…", self.export_csv)
        f.addSeparator()
        self._act(f, "Export image…", lambda: self.export_images(False), "Ctrl+E")
        self._act(f, "Export all images…", lambda: self.export_images(True))
        self._act(f, "Publication figure…", self.publication_figure, "Ctrl+Shift+E",
                  tip="Lay out the image for print: width, fonts, scale bar, colour bar; save PNG/TIFF/PDF/SVG or copy")
        self._act(f, "Copy figure", self.copy_image, "Ctrl+Shift+C",
                  tip="Copy the image as a publication figure (the style last used in Publication figure…)")
        f.addSeparator()
        self._act(f, "Quit", self.close, QKeySequence.StandardKey.Quit)

        e = mb.addMenu("&Edit")
        self.undo_action = self._act(e, "Undo", self.undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._act(e, "Redo", self.redo, QKeySequence.StandardKey.Redo)
        e.addSeparator()
        self._act(e, "Select all", self.select_all, QKeySequence.StandardKey.SelectAll)
        self._act(e, "Delete selected", self.delete_selected)
        if REVIEW:
            e.addSeparator()
            self._act(e, "Accept selected proposals", self.review.accept, "A")
            self._act(e, "Reject selected proposals", self.review.reject, "R")
            self._act(e, "Next proposal to review", lambda: self.review.next_pending(1), "N")
            self._act(e, "Previous proposal", lambda: self.review.next_pending(-1), "Shift+N")
        e.addSeparator()
        for k in range(1, 10):                      # window-level shortcuts, not listed in the menu
            a = QAction(f"Label {k}", self)
            a.setShortcut(QKeySequence(str(k)))
            a.triggered.connect(lambda _=False, k=k: self._guard(lambda: self.label_key(k)))
            self.addAction(a)
        self._act(e, "Remove label from selection", lambda: self.label_key(0), "0")

        v = mb.addMenu("&View")
        self._act(v, "Fit to window", self.canvas.fit, "F")
        self._act(v, "Actual pixels (100 %)", lambda: self.canvas.set_zoom(1.0), "Ctrl+1")
        self._act(v, "Zoom in", lambda: self.canvas.set_zoom(self.canvas.zoom * 1.25), QKeySequence.StandardKey.ZoomIn)
        self._act(v, "Zoom out", lambda: self.canvas.set_zoom(self.canvas.zoom / 1.25), QKeySequence.StandardKey.ZoomOut)
        self._act(v, "Previous file", lambda: self.browser.step(-1), "PgUp")
        self._act(v, "Next file", lambda: self.browser.step(1), "PgDown")
        v.addSeparator()
        self.a_scalebar = self._toggle(v, "Scale bar", True, lambda on: self._set_canvas("show_scale_bar", on))
        self.a_colorbar = self._toggle(v, "Colorbar", False, lambda on: self._set_canvas("show_colorbar", on))
        self.a_labels = self._toggle(v, "Label names", False, lambda on: self._set_layer("show_labels", on))
        self.a_orient = self._toggle(v, "Orientation arrows", True, lambda on: self._set_layer("show_orientation", on))
        self.a_fill = self._toggle(v, "Fill features", True, lambda on: self._set_layer("fill_alpha", 28 if on else 0))
        if REVIEW:
            self.a_rejected = self._toggle(v, "Rejected proposals", False, self._show_rejected)
        self.a_smooth = self._toggle(v, "Smooth pixels", False, lambda on: (self._set_canvas("smooth", on), self.canvas.render()))
        v.addSeparator()
        self._act(v, "Line profiles…", self.show_profiles, "Ctrl+P")
        self._act(v, "3D view…", self.show_surface)
        v.addSeparator()
        theme_menu = v.addMenu("Theme")
        group = QActionGroup(self)
        current = self.settings.value("theme", "system")
        for key, text in (("system", "Same as the system"), ("light", "Light"), ("dark", "Dark")):
            a = QAction(text, self, checkable=True)
            a.setChecked(key == current)
            a.triggered.connect(lambda _=False, k=key: self.set_theme(k))
            group.addAction(a)
            theme_menu.addAction(a)
        panels = v.addMenu("Panels")
        for d in self.docks.values():
            panels.addAction(d.toggleViewAction())
        panels.addAction(self.tool_toolbar.toggleViewAction())

        p = mb.addMenu("&Process")
        self._act(p, "Fourier filter…", self.open_fourier)
        self._act(p, "Set pixel size…", self.set_pixel_size)
        self._act(p, "Apply processing to all images", self.processing.processing._apply_all)
        self._act(p, "Apply display to all images", self.processing.display._apply_all)

        d = mb.addMenu("&Detect")
        self._act(d, "Tune adapt pipeline…", self.tune_adapt, tip="Threshold / morphology pipeline tuned on this image")
        self._act(d, "Run adapt pipeline on this image", lambda: self.run_adapt(False), "Ctrl+D")
        self._act(d, "Run adapt pipeline on all images", lambda: self.run_adapt(True))
        self.review_adapt = QAction("Review adapt results", self, checkable=True)
        self.review_adapt.setToolTip("On: results are proposals to accept or reject in the Review panel. "
                                     "Off: they are accepted at once.")
        self.review_adapt.setStatusTip(self.review_adapt.toolTip())
        self.review_adapt.setChecked(self.settings.value("adapt_review", True, type=bool))
        self.review_adapt.toggled.connect(lambda on: self.settings.setValue("adapt_review", on))
        if REVIEW:
            d.addAction(self.review_adapt)
        d.addSeparator()
        self.yolo_actions = [self._act(d, "Detect with YOLO model…", self.run_yolo)]
        d.addSeparator()
        self._act(d, "Centre lines → outlines…", self.lines_to_outlines,
                  tip="Turn centre lines into polygon features of a given width")
        self._act(d, "Estimate orientations of selected features", lambda: self.auto_orient(False))
        self._act(d, "Estimate orientations of all features", lambda: self.auto_orient(True))

        a = mb.addMenu("&Analyze")
        self._act(a, "ROI analysis…", self.show_analysis)
        a.addSeparator()
        self._act(a, "Split shapes into centre lines", self.split_ribbons,
                  tip="Skeletonise the selected shapes, outlines or painted masks (or all on this image), and add one "
                      "centre line per segment, split at junctions and sharp bends; correct them and the skeleton "
                      "analyses use them")
        self._act(a, "Detect herringbone direction", self.detect_herringbone,
                  tip="Measure the Au(111) herringbone ridges on the bare surface of this image and propose one "
                      "herringbone axis per domain")
        a.addSeparator()
        self._act(a, "Export all measurements…", self.measurements.export)
        self._act(a, "Check annotations", self.check_annotations,
                  tip="Validation rules of every annotation kind (too small, outside the image, missing pairs …)")

        self._build_surface_menu(mb)

        m = mb.addMenu("&Machine learning")
        self._act(m, "Export dataset / train…", self.train)
        self._act(m, "Evaluate a detector run…", self.evaluate)
        self._act(m, "Export feature clips…", self.export_clips)
        m.addSeparator()
        self.install_yolo_action = self._act(m, "Install YOLO…", self.install_yolo,
                                             tip="Download Ultralytics YOLO and PyTorch (about 0.7 GB) to detect and "
                                                 "train with YOLO models")
        for menu in (d, m):
            menu.setToolTipsVisible(True)            # says why a YOLO command is disabled
            menu.aboutToShow.connect(self._update_yolo_actions)
        self._update_yolo_actions()

        h = mb.addMenu("&Help")
        self._act(h, "Tutorials…", self.show_tutorials, "F1",
                  tip="Guided tours in the program: getting started and the advanced tools")
        self._act(h, "Getting started tour", lambda: self.start_tutorial("getting_started"))
        h.addSeparator()
        self._act(h, "Keyboard shortcuts", self.show_shortcuts)
        self._act(h, "Check for updates…", lambda: self.updates.check(),
                  tip="See whether a newer ScanBound is out on GitHub, with what changed")
        self._act(h, "About", lambda: QMessageBox.about(self, APP_NAME, f"{APP_NAME} {software_version()}\n"
                                                         "Scanning probe image analysis and annotation."))

    @property
    def updates(self):  # noqa: ANN201
        """The update checks and installs (app/updates.py), made on first use."""
        if getattr(self, "_updates", None) is None:
            from .updates import Updates
            self._updates = Updates(self)
        return self._updates

    def _toggle(self, menu: QMenu, text: str, on: bool, slot) -> QAction:  # noqa: ANN001
        a = QAction(text, self)
        a.setCheckable(True)
        a.setChecked(on)
        a.toggled.connect(slot)
        menu.addAction(a)
        return a

    def _set_canvas(self, attr: str, value) -> None:  # noqa: ANN001
        setattr(self.canvas, attr, value)
        self.canvas.viewport().update()

    def _set_layer(self, attr: str, value) -> None:  # noqa: ANN001
        self.canvas.layer.set_option(attr, value)

    def _show_rejected(self, on: bool) -> None:
        self.session.hidden_statuses = set() if on else {"rejected"}
        self.canvas.layer.refresh()

    # ------------------------------------------------------------------ project
    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for path in self._recent():
            self._act(self.recent_menu, path, lambda p=path: self._open_path(p))

    def _recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        return [p for p in (v if isinstance(v, list) else [v]) if p]

    def _remember(self, path: str) -> None:
        r = [path] + [p for p in self._recent() if p != path]
        self.settings.setValue("recent", r[:10])
        self._fill_recent()

    def _confirm_discard(self) -> bool:
        if not self.session.project.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, "Save the changes to the project first?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                                 | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard

    def open_folder(self, recursive: bool = False) -> None:
        if not self._confirm_discard():
            return
        d = QFileDialog.getExistingDirectory(self, "Open data folder", self.settings.value("last_dir", ""))
        if d:
            self._open_path(d, recursive)

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        p, _ = QFileDialog.getOpenFileName(self, "Open project", self.settings.value("last_dir", ""),
                                           "ScanBound project (*.json)")
        if p:
            self._open_path(p)

    def _open_path(self, path: str, recursive: bool = False) -> None:
        """Open a folder or project file.  The folder scan runs in the background with a progress dialog
        that can cancel it, and stops at MAX_FILES files to ask before opening a huge tree."""
        from ..io.data import ScanStopped
        from ..project import Project
        is_dir = os.path.isdir(path)
        limit = int(self.settings.value("max_files", MAX_FILES))
        if is_dir:
            recursive = recursive or Session.saved_recursive(path)
        stop = {"flag": False}
        dlg = QProgressDialog("Looking for data files…", "Cancel", 0, 0, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(400)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.canceled.connect(lambda: stop.update(flag=True))

        def work(progress):  # noqa: ANN001 - worker thread
            opts = dict(limit=limit + 1, cancelled=lambda: stop["flag"],
                        progress=lambda d, n: progress((d, n)))
            if is_dir:
                return Project.open_folder(path, recursive, **opts)
            p = Project.load(path)
            p.scan(bool(p.settings.get("recursive")), **opts)
            return p

        def on_progress(v) -> None:  # noqa: ANN001
            d, n = v
            dlg.setLabelText(f"Looking for data files…\n{n} file(s) in {d} folder(s)")

        def done(p) -> None:  # noqa: ANN001
            dlg.close()
            if len(p.files) > limit:
                p = self._too_many(p, path, limit, recursive and is_dir)
                if p is None:
                    return
            try:
                if is_dir:
                    self.session.open_folder(path, recursive, project=p)
                    self.settings.setValue("last_dir", path)
                else:
                    self.session.open_project(path, project=p)
                    self.settings.setValue("last_dir", os.path.dirname(path))
                self._remember(path)
            except Exception as exc:  # noqa: BLE001
                self.session.error(f"Cannot open {path}: {exc}")

        def failed(tb: str) -> None:
            dlg.close()
            if stop["flag"] or ScanStopped.__name__ in tb:
                self.session.info(f"Opening {path} was cancelled")
                return
            last = tb.strip().splitlines()[-1]
            self.session.error(f"Cannot open {path}: {last}")
            QMessageBox.warning(self, APP_NAME, f"Cannot open {path}:\n{last}")

        run_job(work, done=done, failed=failed, progress=on_progress)
        dlg.show()

    def _too_many(self, p, path: str, limit: int, was_recursive: bool):  # noqa: ANN001
        """Ask what to do with a folder that holds more than ``limit`` data files; returns the project
        to open (trimmed to ``limit`` files) or None."""
        box = QMessageBox(QMessageBox.Icon.Warning, APP_NAME,
                          f"{path} holds more than {limit} data files.\n\nOpening this many makes browsing slow. "
                          f"You can open the first {limit}, or open a subfolder instead.", parent=self)
        first = box.addButton(f"Open the first {limit}", QMessageBox.ButtonRole.AcceptRole)
        top = box.addButton("Only the top folder", QMessageBox.ButtonRole.ActionRole) if was_recursive else None
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is first:
            for rel in list(p.files)[limit:]:
                if not any(e.file == rel for e in p.images.values()):
                    del p.files[rel]
            self.session.info(f"Opened the first {limit} data files of {path}; the rest were left out")
            return p
        if top is not None and clicked is top:
            QTimer.singleShot(0, lambda: self._open_path_flat(path))
        return None

    def _open_path_flat(self, path: str) -> None:
        from ..project import Project
        p = Project.open_folder(path, False, limit=int(self.settings.value("max_files", MAX_FILES)))
        p.settings.pop("recursive", None)
        self.session.open_folder(path, False, project=p)
        self.settings.setValue("last_dir", path)
        self._remember(path)

    def save(self) -> bool:
        try:
            self.session.save()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"Could not save the project:\n{exc}")
            return False
        self._update_title()
        return True

    def save_as(self) -> None:
        start = self.session.project.path or os.path.join(self.session.project.root, DEFAULT_NAME)
        p = save_file_name(self, "Save project as", start, "ScanBound project (*.json)")
        if p:
            self.session.save(p)
            self._update_title()

    def _update_title(self) -> None:
        p = self.session.project
        where = p.path or p.root
        self.setWindowTitle(f"{APP_NAME} — {where}{' *' if p.dirty else ''}")

    def import_csv(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Import annotations", self.session.project.root, "CSV (*.csv)")
        if not p:
            return
        n, problems = self.session.project.import_legacy_annotations(p)
        self.session.labels_edited()
        for msg in problems[:20]:
            self.session.warn(msg)
        self.session.info(f"Imported {n} annotation(s) from {os.path.basename(p)}"
                          + (f" ({len(problems)} problem(s), see the log)" if problems else ""))
        self.browser._update_counts(set(self.browser.items))

    def export_csv(self) -> None:
        p = save_file_name(self, "Export annotations", os.path.join(self.session.project.root,
                                                                     "annotations.csv"), "CSV (*.csv)")
        if p:
            n = self.session.project.export_legacy_annotations(p)
            self.session.info(f"Wrote {n} row(s) to {p} (ground truth only)")

    def export_images(self, all_images: bool) -> None:
        from ..figures import export_image
        s = self.session
        default = os.path.join(s.project.root, "ScanBound_image_output")
        d = QFileDialog.getExistingDirectory(self, "Export to folder", default)
        if not d:
            return
        ids = [s.project.entry(i).id for i in s.image_ids()] if all_images else [s.current_id]
        colorbar = QMessageBox.question(self, "Export", "Add a colorbar?") == QMessageBox.StandardButton.Yes
        done = 0
        for i in ids:
            try:
                rel, ch = s.project.split_image_id(i)
                stem = os.path.splitext(os.path.basename(rel))[0]
                safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in ch)
                export_image(s.image(i), s.project.entry(i).display, os.path.join(d, f"{stem}_{safe}.png"),
                             scale_bar=self.canvas.show_scale_bar, colorbar=colorbar)
                done += 1
            except Exception as exc:  # noqa: BLE001
                s.warn(f"{i}: {exc}")
        s.info(f"Exported {done} image(s) to {d} (with metadata.csv and .json provenance)")

    def copy_image(self) -> None:
        from .dialogs.figure import copy_figure
        if self.session.image() is None:
            return
        copy_figure(self.session)

    def _view_crop(self) -> Optional[tuple]:
        img = self.canvas.image
        if img is None:
            return None
        r = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
        h, w = img.shape
        x0, y0, x1, y1 = max(0.0, r.left()), max(0.0, r.top()), min(float(w), r.right()), min(float(h), r.bottom())
        if x1 - x0 >= w - 0.5 and y1 - y0 >= h - 0.5:
            return None                              # the whole image is in view
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def publication_figure(self) -> None:
        from .dialogs.figure import FigureDialog
        if self.session.image() is None:
            return
        FigureDialog(self.session, self._view_crop(), self).exec()

    # ------------------------------------------------------------------ edit
    def undo(self) -> None:
        store = self.session.store
        text = store.undo_text()
        if store.undo():
            self.session.info(f"Undo: {text}")
        self._update_undo()

    def redo(self) -> None:
        store = self.session.store
        text = store.redo_text()
        if store.redo():
            self.session.info(f"Redo: {text}")
        self._update_undo()

    def _update_undo(self) -> None:
        store = self.session.store
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(store.can_undo)
            self.undo_action.setText(f"Undo {store.undo_text()}" if store.can_undo else "Undo")
            self.redo_action.setEnabled(store.can_redo)
            self.redo_action.setText(f"Redo {store.redo_text()}" if store.can_redo else "Redo")

    def select_all(self) -> None:
        hidden = self.session.hidden_statuses
        self.session.select([a.id for a in self.session.annotations() if a.status not in hidden])

    def delete_selected(self) -> None:
        sel = list(self.session.selection)
        if sel:
            self.session.store.remove(sel, description=f"Delete {len(sel)} annotation(s)")

    def label_key(self, k: int) -> None:
        s = self.session
        name = None if k == 0 else s.project.labels.by_key(k)
        if k and name is None:
            return
        if s.selection:
            s.store.set_label(list(s.selection), name)
        else:
            s.set_current_label(name)

    # ------------------------------------------------------------------ image
    def _on_image(self, image_id: str) -> None:
        rel, ch = self.session.project.split_image_id(image_id) if image_id else ("", "")
        self.image_label.setText(rel)
        self.channel.blockSignals(True)
        self.channel.clear()
        try:
            self.channel.addItems(self.session.channels())
        except Exception:  # noqa: BLE001
            pass
        self.channel.setCurrentText(ch)
        self.channel.blockSignals(False)
        img = self.canvas.image
        if img is not None:
            warn = [q.message for q in img.quality() if q.level in ("warning", "error")]
            if warn:
                self.session.warn(f"{rel}: " + "; ".join(warn))
        self.spectra.update_positions()

    def _on_channel(self, ch: str) -> None:
        if ch:
            self.session.set_channel(ch)

    def _on_cursor(self, info) -> None:  # noqa: ANN001
        if not info:
            self.cursor_label.setText("")
            return
        text = f"x {info['x']:.1f}  y {info['y']:.1f} px"
        if "physical" in info:
            text += f"   ({info['physical']})"
        text += f"   value {info['value']}"
        self.cursor_label.setText(text)

    def _on_message(self, text: str, level: str) -> None:
        self.statusBar().showMessage(text, 6000)

    def center_on(self, ann_id: str) -> None:
        a = self.session.store.get(ann_id)
        if a is None:
            return
        x0, y0, x1, y1 = a.geometry.bbox()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        vr = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
        if not vr.contains(QPointF(cx, cy)) or (x1 - x0) > vr.width() or (y1 - y0) > vr.height():
            self.canvas.centerOn(QPointF(cx, cy))

    def _context_menu(self, pos: QPointF, global_pos) -> None:  # noqa: ANN001
        s = self.session
        hits = [h for h in s.store.hit_test(s.current_id, pos.x(), pos.y(), self.canvas.tolerance())
                if h.status not in s.hidden_statuses]
        if hits and hits[0].id not in s.selection:
            s.select([hits[0].id])
        menu = QMenu(self)
        if s.selection:
            lab = menu.addMenu("Label")
            for name in [None] + list(s.project.labels.names):
                act = lab.addAction(name or "(no label)")
                act.triggered.connect(lambda _=False, n=name: s.store.set_label(list(s.selection), n))
            if REVIEW and any(a.status in ("proposed", "rejected") for a in s.selected()):
                menu.addAction("Accept", self.review.accept)
            if REVIEW and any(a.status != "manual" for a in s.selected()):
                menu.addAction("Reject", self.review.reject)
            menu.addAction("Estimate orientation", lambda: self.auto_orient(False))
            if any(a.kind == "feature" for a in s.selected()):
                hand = menu.addMenu("Handedness")
                for text, value in (("Left (L)", "L"), ("Right (R)", "R"), ("Not set", None)):
                    hand.addAction(text, lambda v=value: self.set_handedness(v))
            if any(a.orientation is not None for a in s.selected()):
                menu.addAction("Flip orientation", self.flip_orientation)
                menu.addAction("Clear orientation", lambda: s.store.update_many(list(s.selection), "Clear orientation",
                                                                               orientation=None))
            vec = next((a for a in s.selected() if a.kind == "lattice_vector"), None)
            if vec is not None:
                menu.addAction("Lattice vectors…", lambda: self.surface.edit_lattice_vectors(vec.attrs.get("lattice")))
            menu.addSeparator()
            menu.addAction("Delete", self.delete_selected)
        else:
            menu.addAction("Fit to window", self.canvas.fit)
            menu.addAction("Line profiles…", self.show_profiles)
            menu.addAction("Lattice vectors…", lambda: self.surface.edit_lattice_vectors())
        menu.exec(global_pos)

    # ------------------------------------------------------------------ process
    def set_pixel_size(self) -> None:
        img = self.session.raw_image()
        if img is None:
            return
        cur = img.pixel_size[0] * 1e9 if img.calibrated and img.x.unit == "m" else 0.1
        v, ok = QInputDialog.getDouble(self, "Pixel size", "Pixel size in x (nm), 0 = use the file's value:", cur, 0.0,
                                       1e6, 6)
        if not ok:
            return
        if v <= 0:
            self.session.set_calibration(None)
            return
        h, w = img.shape
        aspect = img.pixel_size[1] / img.pixel_size[0] if img.calibrated else 1.0
        vy, ok = QInputDialog.getDouble(self, "Pixel size", "Pixel size in y (nm):", v * aspect, 0.0, 1e6, 6)
        if not ok:
            return
        unc, ok = QInputDialog.getDouble(self, "Pixel size", "Relative uncertainty of this calibration (%):", 2.0,
                                         0.0, 100.0, 2)
        if ok:
            self.session.set_calibration(v * 1e-9, vy * 1e-9, unc / 100.0)
            self.session.info(f"Pixel size set to {v:g} × {vy:g} nm (± {unc:g} %)")

    def open_fourier(self) -> None:
        from .widgets.fourier_filter import FourierCard
        panel = self.processing.processing.panel
        card = next((c for c in panel.cards if isinstance(c, FourierCard)), None)
        if card is None:
            card = panel.add_step("fourier", emit=False)
        self.docks["Display"].raise_()
        self.processing.setCurrentIndex(1)
        self.edit_fourier(card)

    def edit_fourier(self, card) -> None:  # noqa: ANN001
        from ..display import ImageStats as DS, render
        from ..processing import ProcessingChain
        from .widgets.fourier_filter import FourierFilterDialog
        s = self.session
        raw = s.raw_image()
        if raw is None:
            return
        panel = self.processing.processing.panel
        k = panel.index_of(card)
        full = self.processing.processing.chain()
        before = ProcessingChain(full.steps[:k], full.registry, full.nan_policy)
        data = before.apply(raw, strict=False).filled("nearest")
        dx, dy = raw.pixel_size
        px = math.sqrt(dx * dy) * 1e9 if raw.calibrated and raw.x.unit == "m" else None
        settings = s.entry.display

        def rend(img):  # noqa: ANN001
            return render(img, DS(img), settings)

        values = card.values()
        dlg = FourierFilterDialog(data, px, rend, values.get("shapes"), values.get("mode"), values.get("smooth", 1.0), self)
        if dlg.exec():
            card.set_filter(*dlg.get_filter())
            card.enable_cb.setChecked(True)
            shapes, mode, _ = dlg.get_filter()
            s.info(f"Fourier filter: {len(shapes)} region(s), {mode}")

    def show_profiles(self) -> None:
        from .widgets.line_profile import ProfileWindow
        if self._profile_window is None:
            self._profile_window = ProfileWindow(self._profile_data, self)
            self._profile_window.clearRequested.connect(self._clear_profiles)
        self._profile_window.refresh()
        self._profile_window.show()
        self._profile_window.raise_()
        if self.canvas.tool is not self.canvas.tools["measure"] and \
                not any(a.kind == "measurement_line" for a in self.session.annotations()):
            self.canvas.set_tool("measure")

    def _refresh_profiles(self) -> None:
        if self._profile_window is not None and self._profile_window.isVisible():
            self._profile_window.refresh()

    def _profile_data(self) -> dict:
        s = self.session
        img = s.image()
        palette = ["#00E5FF", "#FF6D00", "#76FF03", "#F50057", "#FFEA00", "#D500F9"]
        lines = [a for a in s.annotations() if a.kind == "measurement_line"]
        return {"image": img, "name": img.quantity if img is not None else "",
                "lines": [(f"line {k + 1}", palette[k % len(palette)], a.geometry) for k, a in enumerate(lines)]}

    def _clear_profiles(self) -> None:
        ids = [a.id for a in self.session.annotations() if a.kind == "measurement_line"]
        if ids:
            self.session.store.remove(ids, description="Clear profiles")

    def show_surface(self) -> None:
        from ..display import ImageStats as DS, levels, render
        from .widgets.surface_view import SurfaceViewDialog
        s = self.session

        def provider() -> dict:
            img = s.image()
            if img is None:
                return {}
            st = DS(img.data)
            px = img.pixel_size[0] * 1e9 if img.calibrated and img.x.unit == "m" else None
            return {"data": img.data, "px": px, "unit": img.unit, "rgb": render(img.data, st, s.entry.display),
                    "levels": levels(st, s.entry.display), "title": s.current_id}
        try:
            if self._surface_window is None:
                self._surface_window = SurfaceViewDialog(provider, self)
                s.imageChanged.connect(lambda _i: self._surface_window.isVisible() and self._surface_window.refresh())
            else:
                self._surface_window.refresh()
        except Exception as exc:  # noqa: BLE001 - e.g. no OpenGL
            s.error(f"3D view unavailable: {exc}")
            return
        self._surface_window.show()
        self._surface_window.raise_()

    # ------------------------------------------------------------------ detect
    def tune_adapt(self) -> None:
        from .dialogs.detect import tune_adapt_pipeline
        tune_adapt_pipeline(self.session, self)

    def run_adapt(self, all_images: bool) -> None:
        from .dialogs.detect import ADAPT_KEY, run_adapt, tune_adapt_pipeline
        s = self.session
        if not s.project.settings.get(ADAPT_KEY) and not tune_adapt_pipeline(s, self):
            return
        ids = [s.project.entry(i).id for i in s.image_ids()] if all_images else [s.current_id]
        review = REVIEW and self.review_adapt.isChecked()

        def done(n: int) -> None:
            if review and n:
                self._show_review()
        run_adapt(s, self, ids, review=review, on_done=done)

    def _update_yolo_actions(self) -> None:
        from .yolo_install import INSTALL_HINT, yolo_available
        ok = yolo_available()
        for a in self.yolo_actions:
            a.setEnabled(ok)
            a.setToolTip("" if ok else INSTALL_HINT)
            a.setStatusTip("" if ok else INSTALL_HINT)
        self.install_yolo_action.setVisible(not ok)

    def install_yolo(self) -> None:
        from .yolo_install import YoloInstallDialog
        dlg = YoloInstallDialog(self)
        dlg.installed.connect(self._update_yolo_actions)
        dlg.installed.connect(lambda: self.session.info("YOLO installed; the YOLO commands are enabled."))
        dlg.exec()

    def run_yolo(self) -> None:
        from .dialogs.detect import YoloDialog
        dlg = YoloDialog(self.session, self)
        if dlg.exec():
            dlg.run()
            if REVIEW:
                self._show_review()

    def _show_review(self) -> None:
        if "Review" in self.docks:
            self.docks["Review"].show()
            self.docks["Review"].raise_()

    def auto_orient(self, everything: bool) -> None:
        from ..measurements import estimate_orientation
        s = self.session
        img = s.image()
        if img is None:
            return
        anns = s.annotations() if everything else s.selected()
        anns = [a for a in anns if a.kind == "feature" and a.status != "rejected"]
        if not anns:
            s.info("Select features first (or use 'all features').")
            return
        data = img.filled("nearest")
        dx, dy = img.pixel_size
        scale = (1.0, dy / dx) if img.calibrated and dx > 0 else (1.0, 1.0)
        fold = self.measurements.fold.value()
        changes = {}
        for a in anns:
            g = a.geometry
            pts = g.points if g.type == "point" else g.outline()
            ang = estimate_orientation(data, pts, closed=g.closed or g.type == "point", polar=fold == 1, scale=scale)
            if ang is not None:
                changes[a.id] = ang
        with s.store.transaction(f"Estimate {len(changes)} orientation(s)"):
            for i, ang in changes.items():
                s.store.update(i, orientation=ang)
        s.info(f"Estimated {len(changes)} orientation(s) from the image (check the arrows; Flip in the context menu)")

    def lines_to_outlines(self) -> None:
        from ..annotations.convert import lines_to_outlines
        s = self.session
        img = s.image()
        lines = [a for a in (s.selected() or s.annotations()) if a.kind == "centre_line"]
        if img is None or not lines:
            s.info("No centre lines (select some, or none for all on this image).")
            return
        unit = "nm" if img.calibrated and img.x.unit == "m" else "px"
        width, ok = QInputDialog.getDouble(self, "Centre lines → outlines", f"Width ({unit}):", 2.0, 0.01, 1e4, 2)
        if not ok:
            return
        buffer, ok = QInputDialog.getDouble(self, "Centre lines → outlines", f"Extend the ends by ({unit}):", 0.5, 0.0,
                                            1e4, 2)
        if not ok:
            return
        k = 1e-9 if unit == "nm" else 1.0
        pairs = lines_to_outlines(lines, width * k, buffer * k, img.pixel_size if unit == "nm" else (1.0, 1.0))
        keep = QMessageBox.question(self, "Centre lines → outlines", "Keep the centre lines as well?") \
            == QMessageBox.StandardButton.Yes
        with s.store.transaction(f"Convert {len(pairs)} centre line(s)"):
            s.store.add([new for _old, new in pairs])
            if not keep:
                s.store.remove([old.id for old, _new in pairs], cascade=False)
        s.info(f"Converted {len(pairs)} centre line(s) to outlines {width:g} {unit} wide")

    def split_ribbons(self) -> None:
        from ..annotations.convert import outlines_to_centre_lines
        from ..ribbon_graph import RibbonParams
        s = self.session
        img = s.image()
        feats = [a for a in (s.selected() or s.annotations()) if a.kind == "feature" and a.geometry.closed
                 and a.status != "rejected"]
        if img is None or not feats:
            s.info("No shapes (select some, or none for all on this image).")
            return
        kink, ok = QInputDialog.getDouble(self, "Split shapes into centre lines",
                                          "Split at bends sharper than (°):", 35.0, 5.0, 120.0, 0)
        if not ok:
            return
        params = RibbonParams(kink_angle=kink, straight_tolerance=kink)
        children: dict[str, list] = {}
        for a in s.annotations():
            if a.kind == "centre_line" and a.parent:
                children.setdefault(a.parent, []).append(a)
        kept = [f for f in feats if any(c.status in GROUND_TRUTH for c in children.get(f.id, []))]
        todo = [f for f in feats if f not in kept]
        dx, dy = img.pixel_size
        pairs = outlines_to_centre_lines(todo, params, (dx, dy) if img.calibrated and dx > 0 and dy > 0 else (1, 1))
        new = [c for _f, cs in pairs for c in cs]
        old = [c.id for f in todo for c in children.get(f.id, []) if c.status in ("proposed", "rejected")]
        with s.store.transaction(f"Split {len(todo)} shape(s) into {len(new)} centre line(s)"):
            if old:
                s.store.remove(old, cascade=False)
            s.store.add(new)
        msg = (f"Proposed {len(new)} centre line(s) for {len(todo)} shape(s): accept or correct them in Review"
               if REVIEW else f"Added {len(new)} centre line(s) for {len(todo)} shape(s)")
        if kept:
            msg += f"; {len(kept)} shape(s) already have accepted centre lines and were left alone"
        s.info(msg)
        if new and REVIEW:
            self._show_review()

    def detect_herringbone(self) -> None:
        import cv2
        import numpy as np
        from ..annotations import Annotation, Origin, Segment
        from ..herringbone import AU111_PERIOD_NM, herringbone_axes, herringbone_domains, herringbone_field
        from ..model.provenance import software_version
        s = self.session
        img = s.image()
        if img is None:
            return
        dx, dy = img.pixel_size
        if not (img.calibrated and img.x.unit == "m" and dx > 0 and dy > 0):
            s.warn("The herringbone needs a calibrated pixel size (Process > Set pixel size).")
            return
        period, ok = QInputDialog.getDouble(self, "Detect herringbone direction", "Herringbone period (nm):",
                                            AU111_PERIOD_NM, 2.0, 30.0, 2)
        if not ok:
            return
        data = np.asarray(img.filled("nearest"), float)
        h, w = data.shape
        sq = math.sqrt(dx * dy)
        fx, fy = dx / sq, dy / sq                       # work on square pixels
        ww, hh = max(1, int(round(w * fx))), max(1, int(round(h * fy)))
        mask = np.zeros((h, w), np.uint8)
        for a in s.annotations():
            if a.kind in ("feature", "centre_line") and a.status != "rejected" and a.geometry.type != "point":
                pts = np.round((np.asarray(a.geometry.outline(), float) - 0.5) * 16).astype(np.int32)
                if a.geometry.closed:
                    cv2.fillPoly(mask, [pts], 1, shift=4)
                else:
                    cv2.polylines(mask, [pts], False, 1, thickness=3, shift=4)
        if (fx, fy) != (1.0, 1.0):
            data = cv2.resize(data, (ww, hh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (ww, hh), interpolation=cv2.INTER_NEAREST)
        try:
            field = herringbone_field(data, sq * 1e9, period, exclude=mask.astype(bool))
        except ValueError as e:
            s.warn(str(e))
            return
        domains = herringbone_domains(field)
        axes = herringbone_axes(field, domains=domains)        # every domain, large ones several times
        if not axes:
            s.warn(f"No herringbone found on this image ({100 * field.coverage:.0f}% of it shows a clear stripe "
                   "pattern). Draw the direction with the Herringbone axis tool (W) instead.")
            return
        origin = Origin("detector", "herringbone", software_version(), "", None,
                        {"period_nm": period, "peaks": [[round(p, 1), round(sh, 3)] for p, sh in field.peaks],
                         "coverage": round(field.coverage, 3), "domains": len(domains)})
        new = [Annotation(s.current_id, "herringbone_axis", Segment([(x0 / fx, y0 / fy), (x1 / fx, y1 / fy)]),
                          status="proposed", origin=origin, attrs={"share": round(share, 3)})
               for (x0, y0), (x1, y1), share in axes]
        old = [a.id for a in s.annotations() if a.kind == "herringbone_axis" and (a.status in ("proposed", "rejected")
               or (not REVIEW and _untouched_result(a, "herringbone")))]
        with s.store.transaction(f"Propose {len(new)} herringbone axis/axes"):
            if old:
                s.store.remove(old, cascade=False)
            s.store.add(new)
        s.info(f"{len(new)} herringbone axes in {len(domains)} domain(s), ridges along "
               + ", ".join(f"{p:.0f}° ({100 * sh:.0f}%)" for p, sh in field.peaks)
               + (f", period {field.period_nm:.1f} nm" if field.period_nm else "")
               + (": accept, move or reject the proposed axes in Review" if REVIEW else ""))
        if REVIEW:
            self._show_review()

    def export_clips(self) -> None:
        from ..ml.clips import export_clips
        s = self.session
        d = QFileDialog.getExistingDirectory(self, "Export clips to", s.project.root)
        if not d:
            return
        per, ok = QInputDialog.getInt(self, "Feature clips", "Clips per label (and angle bin):", 20, 1, 100000)
        if not ok:
            return
        ctx, ok = QInputDialog.getDouble(self, "Feature clips", "Context around the box (×):", 1.6, 1.0, 10.0, 2)
        if not ok:
            return
        binw, ok = QInputDialog.getInt(self, "Feature clips", "Orientation bin (°, 0 = no bins):", 0, 0, 180)
        if not ok:
            return
        n = export_clips(s.project, d, context=ctx, per_group=per, angle_bin=binw or None)
        s.info(f"Wrote {n} clip(s) to {d}")

    def flip_orientation(self) -> None:
        s = self.session
        with s.store.transaction("Flip orientation"):
            for a in s.selected():
                if a.orientation is not None:
                    s.store.update(a.id, orientation=(a.orientation + 180.0) % 360.0)

    # ------------------------------------------------------------------ analyze / ML
    def show_analysis(self) -> None:
        from ..analysis import analysis_data
        from .widgets.roi_analysis_dialog import RoiAnalysisDialog
        s = self.session

        def provider(scope: str):
            if scope == "current":
                ids = [s.current_id] if s.current_id else []
            else:
                ids = sorted({a.image_id for a in s.store if a.status in GROUND_TRUTH})
            return analysis_data(s.project, ids)
        if self._analysis_window is None:
            self._analysis_window = RoiAnalysisDialog(provider, self)
        self._analysis_window.refresh_labels()
        self._analysis_window.show()
        self._analysis_window.raise_()

    def check_annotations(self) -> None:
        s = self.session
        img = s.image()
        issues = s.store.issues(s.current_id, img.shape if img is not None else None)
        if not issues:
            s.info("No problems found on this image.")
            return
        s.select(list(dict.fromkeys(i.annotation for i in issues if getattr(i, "annotation", None))))
        for i in issues[:50]:
            s.warn(f"{i}")
        s.info(f"{len(issues)} issue(s) on this image (selected; details in the log)")
        self.docks["Log"].raise_()

    def train(self) -> None:
        from .dialogs.ml import TrainDialog
        TrainDialog(self.session, self).show()

    def evaluate(self) -> None:
        from .dialogs.evaluate import EvaluateDialog
        EvaluateDialog(self.session, self).show()

    # ------------------------------------------------------------------ spectra
    def _show_spectrum(self, rel: str) -> None:
        d = self.docks["Spectroscopy"]
        if not d.isVisible():
            d.show()
            self.resizeDocks([d], [max(220, self.height() // 3)], Qt.Orientation.Vertical)
        self.docks["Spectroscopy"].raise_()
        self.spectra.show_file(rel)

    def _show_positions(self, positions) -> None:  # noqa: ANN001
        """Spectrum tip positions: ticked ones as dots in their plot colour, the others as grey rings."""
        from PyQt6.QtGui import QBrush, QColor, QPen
        from PyQt6.QtWidgets import QGraphicsEllipseItem
        for it in self._marker_items:
            self.canvas.scene().removeItem(it)
        self._marker_items = []
        sx = 1.0 / max(self.canvas.zoom, 1e-6)                     # screen px -> scene units (x)
        sy = sx / max(self.canvas.aspect, 1e-6)                    # keep the dots round on screen
        for rel, x, y, color in sorted(positions, key=lambda p: p[3] is not None):   # coloured on top
            r = 5.5 if color else 4.5
            it = QGraphicsEllipseItem(x - r * sx, y - r * sy, 2 * r * sx, 2 * r * sy)
            if color:
                pen = QPen(QColor("white"), 1.5)
                it.setBrush(QBrush(QColor(color)))
            else:
                pen = QPen(QColor(200, 200, 200), 1.5)
                it.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            pen.setCosmetic(True)
            it.setPen(pen)
            it.setToolTip(os.path.basename(rel))
            it.setZValue(500 if color else 499)
            self.canvas.scene().addItem(it)
            self._marker_items.append(it)
        self._last_positions = positions

    # ------------------------------------------------------------------ surface
    def _build_surface_menu(self, mb) -> None:  # noqa: ANN001
        from ..surface import workflow as wf
        sm = mb.addMenu("&Surface")
        self._act(sm, "Surface panel", lambda: (self.docks["Surface"].show(), self.docks["Surface"].raise_()))
        self._act(sm, "Fit lattice", self.surface.fit_lattice, "Ctrl+L",
                  tip="Find, index and fit the substrate lattice of this image")
        self._act(sm, "Lattice vectors…", lambda: self.surface.edit_lattice_vectors(),
                  tip="Type in a1 and a2 of this image when you know them")
        self._act(sm, "Calibrate pixel size from the lattice…", self.surface.calibrate)
        sm.addSeparator()
        for text, fn in (("Surface structure (lattice, symmetry, strain, domains)", wf.structure_explorer),
                         ("Adsorption sites and orientations", wf.adsorption_map),
                         ("Molecular assembly", wf.assembly_analysis),
                         ("Lattice defects", wf.defect_map),
                         ("Detect step edges", wf.step_edges),
                         ("Terraces", wf.terrace_map),
                         ("Scan quality", wf.quality_check)):
            self._act(sm, text, lambda fn=fn: self.surface.run(fn))
        self._act(sm, "Explain the selected feature", self.surface.explain)
        self._act(sm, "Build 3D model (.xyz)…", self.surface.build_model,
                  tip="Molecules from .xyz files on the annotated positions, on the fitted substrate, in a 3D window")
        sm.addSeparator()
        self._act(sm, "Register images…", self.register_images,
                  tip="Align another image (other bias, ncAFM, later frame) with this one")
        self._act(sm, "Cross-channel correlation…", self.correlation_workbench,
                  tip="Measure the molecules in several registered images and group them")
        self._act(sm, "Track changes over time…", self.evolution_tracker,
                  tip="Follow molecules through a series of registered images")

    def show_surface_report(self, report) -> None:  # noqa: ANN001
        from .dialogs.surface_report import ReportWindow
        w = ReportWindow(self.session, report, self, on_proposals=self.add_proposals)
        w.show()
        self.surface.update_overlay()

    def add_proposals(self, anns: list) -> None:
        s = self.session
        names = {a.label for a in anns if a.label}
        for n in names:
            if n not in s.project.labels.names:
                s.project.labels.add(n)
        s.labels_edited()
        s.store.add(anns, f"Add {len(anns)} surface proposal(s)" if REVIEW else f"Add {len(anns)} surface result(s)")
        if REVIEW:
            self._show_review()
            s.info(f"Added {len(anns)} proposal(s): accept or reject them in the Review panel.")
        else:
            s.info(f"Added {len(anns)} annotation(s).")

    def set_handedness(self, value) -> None:  # noqa: ANN001
        s = self.session
        sel = [a for a in s.selected() if a.kind == "feature"]
        with s.store.transaction("Set handedness"):
            for a in sel:
                attrs = dict(a.attrs)
                if value is None:
                    attrs.pop("handedness", None)
                else:
                    attrs["handedness"] = value
                s.store.replace(a.evolve(attrs=attrs))

    def register_images(self) -> None:
        from ..surface.store import SurfaceStore
        from ..surface.workflow import register_images
        from .dialogs.surface_tools import RegisterDialog
        s = self.session
        if not s.current_id:
            return
        dlg = RegisterDialog(s, self)
        if not dlg.exec() or dlg.other.currentData() is None:
            return
        other = dlg.other.currentData()
        s.image(other)
        reg = register_images(SurfaceStore(s.project), s.current_id, other, dlg.method.currentData())
        s._check_dirty()
        s.info(f"Registered: {reg.describe()}")
        QMessageBox.information(self, APP_NAME, f"{reg.describe()}\n\nStored in the project; used by the "
                                                "correlation and tracking analyses.")

    def _choose_images(self, title: str, hint: str, ordered: bool) -> list:
        from .dialogs.surface_tools import ImageChooser
        dlg = ImageChooser(self.session, title, hint, ordered=ordered, parent=self)
        if not dlg.exec():
            return []
        ids = dlg.chosen()
        for i in ids:
            self.session.image(i)
        return ids

    def correlation_workbench(self) -> None:
        from ..surface import workflow as wf
        ids = self._choose_images("Cross-channel correlation", "Tick the images to compare (registered to the "
                                  "reference: other channels, biases, ncAFM).", False)
        if ids:
            self.surface.run(wf.correlation_workbench, ids)

    def evolution_tracker(self) -> None:
        from ..surface import workflow as wf
        ids = self._choose_images("Track changes over time", "Tick the frames (registered to the first).", True)
        if ids:
            self.surface.run(wf.evolution_tracker, ids)

    # ------------------------------------------------------------------ help / close
    @property
    def guide(self):  # noqa: ANN201
        if getattr(self, "_guide", None) is None:
            from .tutorial import Guide
            self._guide = Guide(self)
        return self._guide

    def start_tutorial(self, key: str) -> None:
        from .tutorial.content import by_key
        t = by_key(key)
        if t is not None:
            self.guide.start(t)

    def show_tutorials(self) -> None:
        from .tutorial.chooser import TutorialChooser
        dlg = TutorialChooser(self)
        if dlg.exec() and dlg.chosen is not None:
            self.guide.start(dlg.chosen)

    def welcome(self) -> None:
        """First start: offer the getting-started tour (Help → Tutorials… has it at any time)."""
        from .tutorial.chooser import Welcome
        dlg = Welcome(self)
        dlg.exec()
        if dlg.choice == "tour":
            self.start_tutorial("getting_started")
        elif dlg.choice == "browse":
            self.show_tutorials()

    def show_shortcuts(self) -> None:
        rows = [f"{TOOLS[k].shortcut}\t{TOOLS[k].title}" for k in TOOL_ORDER]
        rows += ["", "1 … 9\tlabel the selection (or choose the label to draw with)", "0\tremove the label"]
        if REVIEW:
            rows += ["A / R\taccept / reject the selected proposals", "N / Shift+N\tnext / previous proposal to review"]
        rows += ["Del\tdelete the selection", "Arrows\tnudge the selection (Shift: 10 px)", "Esc\tcancel / deselect",
                 "Space+drag, middle drag\tpan", "Wheel\tzoom", "F\tfit", "PgUp / PgDown\tprevious / next file",
                 "Ctrl+Z / Ctrl+Y\tundo / redo", "Ctrl+S\tsave the project", "F1\ttutorials"]
        QMessageBox.information(self, "Keyboard shortcuts", "\n".join(rows))

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if not self._confirm_discard():
            event.ignore()
            return
        if getattr(self, "_guide", None) is not None:
            self._guide.stop(False)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState(LAYOUT_VERSION))
        self.settings.setValue("layout", LAYOUT_VERSION)
        event.accept()


CRASH_LOG = os.path.join(os.path.expanduser("~"), ".scanbound", "crash.log")


def _install_excepthook() -> None:
    """Report an unexpected error instead of closing: PyQt6 ends the program on an exception that escapes
    a slot.  The traceback goes to the terminal, the Log panel, a dialog and CRASH_LOG; a hard crash
    (in Qt itself) still leaves its Python stack in CRASH_LOG."""
    import datetime
    import faulthandler
    import traceback

    log = None
    try:
        os.makedirs(os.path.dirname(CRASH_LOG), exist_ok=True)
        log = open(CRASH_LOG, "a", encoding="utf-8", buffering=1)
        log.write(f"\n=== ScanBound {software_version()} started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        faulthandler.enable(log, all_threads=True)
    except OSError:
        faulthandler.enable()

    def hook(etype, value, tb):  # noqa: ANN001
        text = "".join(traceback.format_exception(etype, value, tb))
        sys.stderr.write(text)
        if log is not None:
            log.write(text)
        if issubclass(etype, KeyboardInterrupt):
            QApplication.quit()
            return
        for w in QApplication.topLevelWidgets():
            if isinstance(w, MainWindow):
                w.session.error(f"Unexpected error: {value}")
                break
        box = QMessageBox(QMessageBox.Icon.Critical, "Unexpected error",
                          f"{etype.__name__}: {value}\n\nScanBound kept running. The details are in {CRASH_LOG}.")
        box.setDetailedText(text)
        box.exec()

    sys.excepthook = hook

    # Qt's own warnings and fatal errors (a hard crash in Qt says why just before it) go to CRASH_LOG too
    from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
    seen: dict[str, int] = {}

    def qt_messages(mode, context, message):  # noqa: ANN001
        sys.stderr.write(message + "\n")
        if log is None or mode == QtMsgType.QtDebugMsg or mode == QtMsgType.QtInfoMsg:
            return
        seen[message] = n = seen.get(message, 0) + 1
        if n <= 3:                                   # a repeated warning once is enough
            try:
                log.write(f"{datetime.datetime.now():%H:%M:%S} Qt {mode.name[2:-3].lower()}: {message}\n")
            except (OSError, ValueError):
                pass
    qInstallMessageHandler(qt_messages)


def run(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if QApplication.instance() is None:
        # one OpenGL resource space for every 3D view: pyqtgraph compiles each shader once and reuses it
        # in all GL widgets, which is only valid when their contexts are shared
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        configure_opengl()                     # SCANBOUND_OPENGL=core / legacy / software overrides the GL context
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("ScanBound")
    from .branding import apply_branding
    apply_branding(app)
    _install_excepthook()
    apply_theme(str(QSettings("ScanBound", "ScanBound").value("theme", "system")))
    w = MainWindow()
    w.show()
    from .tutorial.chooser import should_welcome
    if should_welcome() and not os.environ.get("SCANBOUND_NO_WELCOME"):
        QTimer.singleShot(600, lambda: w._guard(w.welcome))
    else:                                  # a new version on GitHub? (in the background, silent when offline)
        QTimer.singleShot(0, lambda: w._guard(w.updates.check_at_startup))
    target = next((a for a in argv[1:] if not a.startswith("-")), None)
    if target:
        QTimer.singleShot(0, lambda: w._guard(lambda: w._open_path(os.path.abspath(target))))
    return app.exec()
