import copy
import csv
import math
import os
import re

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QImage
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QFileDialog, QGraphicsPixmapItem, QGroupBox,
                             QHBoxLayout, QHeaderView, QInputDialog, QPushButton, QSpacerItem, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import filters

from core import image_display as idp
from core.display_controls import DisplayControls, fill_colormap_combo
from core.line_profile import ProfileWindow
from core.microscopy_io import FORMATS, find_scans, fmt_length, fmt_si, length_to_nm, load_file
from core.spm_processing import ProcessingPanel
from ui.PyQtImageViewer.QtImageViewer import QtImageViewer
from ui.pages.SPMImageProcess_ui import Ui_Form


def _safe_name(text):
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("_") or "channel"


def _value_str(value, unit):
    """Data value with a sensible SI prefix (handles data already stored in nm, pm, ...)."""
    nm = length_to_nm(value, unit) if unit and unit.lower() != "m" else None
    if nm is not None:
        return fmt_length(nm)
    return fmt_si(value, unit)


class SPMImageProcess(QWidget):
    # dirSent = QtCore.pyqtSignal(str)  # Signal to emit the directory path

    def __init__(self, main_window=None):
        super(SPMImageProcess, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.main_window = main_window
        # Connect pushButton to open directory dialog
        self.ui.pushButton.clicked.connect(self.choose_directory)
        self.ui.label_2.setText("Select a directory to view files")
        self.ui.stackedWidget.removeWidget(self.ui.page)
        self.ui.stackedWidget.removeWidget(self.ui.page_2)

        self.browserText = []
        self.current_viewer = None  # To keep track of the currently displayed image viewer
        self.dir_path = None
        self.saved_output_dir = None  # Directory to save JPG conversions
        self.files = []             # MicroscopyFile per list entry (same order as listWidget / stackedWidget)
        self.shown_channel = []     # channel name shown by each viewer
        self.display_px = []        # nm per displayed pixel of each viewer (None = uncalibrated)
        self.selected_index = 0  # To keep track of the selected image index
        self.selected_channel_name = None  # To keep track of the selected channel name
        self.scaleBarActive = False  # To track if the scale bar is active

        # Connect listWidget item selection to image display
        self.ui.listWidget.itemSelectionChanged.connect(self.display_selected_image)
        self.ui.listWidget_2.itemSelectionChanged.connect(self.update_selected_channel)
        self.ui.pushButton_4.clicked.connect(self.image_processing_ui_show)
        self.ui.checkBox_2.stateChanged.connect(self.changeScaleBarState)
        self.ui.pushButton_6.clicked.connect(self.send_to_imageLabeling)
        self.ui.pushButton_7.clicked.connect(self.save_image)
        self.ui.pushButton_15.clicked.connect(self.save_all_images)
        self.ui.pushButton_8.clicked.connect(self.copy_to_clipboard)
        self.ui.pushButton_2.clicked.connect(self.set_manual_scale)
        self.ui.pushButton_16.clicked.connect(self.clear_log)

        # Image processing: reorderable, toggleable steps (core/spm_processing.py)
        self.processing = ProcessingPanel()
        process_ui_layout = self.ui.widget_10.layout()
        if process_ui_layout is None:
            process_ui_layout = QVBoxLayout(self.ui.widget_10)
        process_ui_layout.addWidget(self.processing)
        self.ui.label_7.hide()                       # the panel has its own title
        self.ui.widget_9.setMaximumWidth(360)
        self.ui.widget_9.hide()  # Hide the image processing widget initially
        self.processing.changed.connect(self.apply_mod)
        self.processing.applyAllRequested.connect(self.apply_mod_all)
        self.processing.closeRequested.connect(self.close_process_menu)
        self.processing.fourierEditRequested.connect(self.edit_fourier_filter)
        self._base_cache = {}       # index -> (channel key, display image, nm/px) before processing
        self._pipe_cache = {}       # index -> (id of the display image, per-step cache)

        # Line profiles: drawn as vector ROIs on the viewer (core/line_profile.py)
        self.profile_mode = False
        self.profile_window = None
        self.ui.pushButton_5.setCheckable(True)
        self.ui.pushButton_5.setText("Line profile")
        self.ui.pushButton_5.setToolTip("Draw lines on the image to see the height (Z) along them")
        self.ui.pushButton_5.toggled.connect(self.set_profile_mode)

        # 3D view (core/surface_view.py)
        self.surface_window = None
        self.view3d_button = QPushButton("3D view", parent=self.ui.widget_7)
        self.view3d_button.setToolTip("Interactive 3D surface of the current image")
        self.view3d_button.clicked.connect(self.open_3d_view)
        self.ui.verticalLayout_6.insertWidget(self.ui.verticalLayout_6.indexOf(self.ui.pushButton_5) + 1,
                                              self.view3d_button)
        self.fourier_button = QPushButton("Fourier filter…", parent=self.ui.widget_7)
        self.fourier_button.setToolTip("Filter frequencies interactively on the 2D Fourier transform "
                                       "(periodic noise, low / high / band-pass). Added as a processing step.")
        self.fourier_button.clicked.connect(self.open_fourier_filter)
        self.ui.verticalLayout_6.insertWidget(self.ui.verticalLayout_6.indexOf(self.view3d_button) + 1,
                                              self.fourier_button)

        self.processed = []         # processed float image behind each viewer
        self.stats = []             # ImageStats of each processed image
        self.colorbarActive = False
        self.display_controls = DisplayControls()
        self.display_controls.changed.connect(self.on_display_changed)
        fill_colormap_combo(self.ui.comboBox, "gray")
        self.ui.comboBox.currentIndexChanged.connect(self.setColorMap)
        self.ui.checkBox.toggled.connect(self.changeColorbarState)

        self._build_metadata_panel()
        self.ui.pushButton.setToolTip("Supported: " + ", ".join(f"{f.name} ({' '.join(f.extensions)})"
                                                              for f in FORMATS))

    @property
    def spm_objects(self):
        return self.files

    # ------------------------------------------------------------------ metadata panel

    def _build_metadata_panel(self):
        """Metadata table on the right, below the buttons (replaces the empty spacer)."""
        for i in reversed(range(self.ui.verticalLayout_4.count())):
            if isinstance(self.ui.verticalLayout_4.itemAt(i), QSpacerItem):
                self.ui.verticalLayout_4.takeAt(i)
        box = QGroupBox("File metadata", parent=self.ui.widget_2)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        self.metadata_table = QTableWidget(0, 2, box)
        self.metadata_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.metadata_table.verticalHeader().setVisible(False)
        self.metadata_table.verticalHeader().setDefaultSectionSize(20)
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metadata_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.metadata_table.setWordWrap(False)
        self.metadata_table.setAlternatingRowColors(True)
        self.metadata_full = QCheckBox("Show full header", box)
        self.metadata_full.toggled.connect(self.show_metadata)
        copy_btn = QPushButton("Copy", box)
        copy_btn.clicked.connect(self.copy_metadata)
        row = QHBoxLayout()
        row.addWidget(self.metadata_full)
        row.addStretch(1)
        row.addWidget(copy_btn)
        lay.addWidget(self.metadata_table, 1)
        lay.addLayout(row)
        box.setTitle("")
        box.setFlat(True)
        self.side_tabs = QTabWidget(self.ui.widget_2)
        self.side_tabs.addTab(self.display_controls, "Display")
        self.side_tabs.addTab(box, "Metadata")
        self.ui.verticalLayout.addWidget(self.side_tabs, 1)

    def show_metadata(self, *_):
        table = self.metadata_table
        table.setRowCount(0)
        if not self.files or not 0 <= self.selected_index < len(self.files):
            return
        f = self.files[self.selected_index]
        ch = f.channel(self.shown_channel[self.selected_index]) or f.default()
        rows = [("File", f.display_name), ("Format", f.format), ("Channel", ch.name),
                ("Data unit", ch.unit or "–")]
        try:
            data = ch.data
            rows.append(("Pixels", f"{data.shape[1]} × {data.shape[0]}"))
            finite = data[np.isfinite(data)]
            if finite.size:
                rows.append(("Data range", f"{_value_str(finite.min(), ch.unit)} … {_value_str(finite.max(), ch.unit)}"))
            if finite.size < data.size:
                rows.append(("Missing data", f"{100 * (1 - finite.size / data.size):.1f} % (incomplete scan)"))
        except Exception as e:
            rows.append(("Data", f"unreadable: {e}"))
        if ch.calibrated:
            tag = " (manual)" if ch.manual_scale else ""
            rows.append(("Pixel size", f"{fmt_length(ch.px_size_x)} × {fmt_length(ch.px_size_y)}{tag}"))
            w, h = ch.scan_size()
            rows.append(("Scan size", f"{fmt_length(w)} × {fmt_length(h)}"))
            px = self.display_px[self.selected_index]
            note = "" if abs(ch.px_size_x / ch.px_size_y - 1) < 1e-3 else " (resampled to square pixels)"
            rows.append(("Displayed pixel size", f"{fmt_length(px)}{note}"))
        else:
            rows.append(("Pixel size", "not calibrated – use \"Set Manual Scale\""))
        sections = [("Measurement", rows), ("File metadata", list(f.metadata.items()))]
        if self.metadata_full.isChecked():
            sections.append(("Full header", list(f.header.items())))

        bold = QFont()
        bold.setBold(True)
        for title, items in sections:
            if not items:
                continue
            r = table.rowCount()
            table.insertRow(r)
            head = QTableWidgetItem(title)
            head.setFont(bold)
            head.setBackground(QColor("#dde6f0"))
            table.setItem(r, 0, head)
            table.setSpan(r, 0, 1, 2)
            for key, value in items:
                r = table.rowCount()
                table.insertRow(r)
                k, v = QTableWidgetItem(str(key)), QTableWidgetItem(str(value).replace("\n", "; "))
                k.setToolTip(str(key))
                v.setToolTip(str(value))
                table.setItem(r, 0, k)
                table.setItem(r, 1, v)
        table.resizeColumnToContents(0)
        table.setColumnWidth(0, min(table.columnWidth(0), 140))

    def copy_metadata(self):
        lines = []
        for r in range(self.metadata_table.rowCount()):
            cells = [self.metadata_table.item(r, c) for c in range(2)]
            lines.append("\t".join(c.text() for c in cells if c is not None))
        QGuiApplication.clipboard().setText("\n".join(lines))

    # ------------------------------------------------------------------ loading

    def choose_directory(self, dir_path=None):
        if not dir_path:
            dir_path = QFileDialog.getExistingDirectory(self, "Select Directory",
                                                        self.dir_path or os.getcwd())
        if not dir_path:
            return
        self.dir_path = dir_path
        self.ui.listWidget.blockSignals(True)
        self.ui.listWidget.clear()
        self.ui.listWidget.blockSignals(False)
        self.ui.listWidget_2.clear()
        self.files, self.shown_channel, self.display_px = [], [], []
        self.processed, self.stats = [], []
        self._base_cache, self._pipe_cache = {}, {}
        while self.ui.stackedWidget.count() > 0:
            widget = self.ui.stackedWidget.widget(0)
            self.ui.stackedWidget.removeWidget(widget)
            widget.deleteLater()
        self.current_viewer = None

        scans = find_scans(dir_path)
        skipped, formats = [], {}
        for n, path in enumerate(scans, 1):
            self.ui.label_2.setText(f"Please wait, loading image {n}/{len(scans)}: {os.path.basename(path)}")
            QApplication.processEvents()
            try:
                f = load_file(path)
                ch = f.default()
                image, px = self.prepare_display(ch)
                stats = idp.ImageStats(image)
                viewer = self.make_Viewer(spmImage=image, stats=stats)
            except Exception as e:
                skipped.append(f"{os.path.basename(path)}: {e}")
                continue
            self.files.append(f)
            self.shown_channel.append(ch.name)
            self.display_px.append(px)
            self.processed.append(image)
            self.stats.append(stats)
            self.ui.stackedWidget.addWidget(viewer)
            self.ui.listWidget.addItem(f.display_name)
            item = self.ui.listWidget.item(self.ui.listWidget.count() - 1)
            item.setToolTip(f"{path}\n{f.format}, {len(f.channels)} channel(s)")
            formats[f.format] = formats.get(f.format, 0) + 1

        self.ui.label.setText(f"Selected Directory: {os.path.basename(dir_path)}")
        summary = ", ".join(f"{k}: {v}" for k, v in formats.items()) or "none"
        self.updateBrowserText(f"Loaded directory: {dir_path}\n{len(self.files)} file(s) loaded ({summary}).")
        if skipped:
            self.updateBrowserText(f"Skipped {len(skipped)} file(s):\n" + "\n".join(skipped[:20])
                                   + ("\n..." if len(skipped) > 20 else ""))
        if not self.files:
            self.ui.label_2.setText("No supported microscopy files found in this directory.")
            self.show_metadata()
            return
        self.ui.label_2.setText("All images loaded!")
        self.ui.listWidget.setCurrentRow(0)

    def updateBrowserText(self, text):
        self.browserText.append(text)
        self.ui.textBrowser.setText('\n \n'.join(self.browserText))
        self.ui.textBrowser.verticalScrollBar().setValue(self.ui.textBrowser.verticalScrollBar().maximum())

    def clear_log(self):
        self.browserText = []
        self.ui.textBrowser.clear()

    def to_qimage(self, data, stats=None):
        """Data -> displayed QImage using the current display settings (levels, gamma, colormap)."""
        stats = stats or idp.ImageStats(data)
        rgb = idp.render(data, stats, self.display_controls.settings)
        h, w = rgb.shape[:2]
        if rgb.ndim == 2:
            return QImage(rgb.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    def make_Viewer(self, imgPath=None, spmImage=None, stats=None):
        viewer = QtImageViewer()
        # Set viewer's aspect ratio mode.
        viewer.aspectRatioMode = QtCore.Qt.AspectRatioMode.KeepAspectRatio

        # Set the viewer's scroll bar behaviour.
        viewer.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        viewer.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        viewer.selectionButton = QtCore.Qt.MouseButton.LeftButton  # Drag to select an area.
        # viewer.regionZoomButton = QtCore.Qt.MouseButton.LeftButton  # set to None to disable
        viewer.zoomOutButton = QtCore.Qt.MouseButton.RightButton  # set to None to disable
        viewer.wheelZoomFactor = 1.25  # Set to None or 1 to disable
        viewer.panButton = QtCore.Qt.MouseButton.MiddleButton  # set to None to disable

        if imgPath:
            viewer.open(filepath=imgPath)
        else:
            viewer.setImage(image=self.to_qimage(spmImage, stats) if isinstance(spmImage, np.ndarray) else spmImage)
        viewer.leftMouseButtonReleased.connect(handleLeftClick)
        viewer.setLabelList(self.PROFILE_LABELS)            # colours of the profile lines
        viewer.roisChanged.connect(lambda action, rois, v=viewer: self._on_viewer_rois(v, action, rois))
        self._apply_viewer_mode(viewer)
        viewer.mousePositionOnImageChanged.connect(self.show_value_under_cursor)
        return viewer

    def show_value_under_cursor(self, pos):
        """Tip bar readout: pixel position, physical position and data value."""
        if not self.files or not 0 <= self.selected_index < len(self.processed):
            return
        data = self.processed[self.selected_index]
        x, y = pos.x(), pos.y()
        if not (0 <= y < data.shape[0] and 0 <= x < data.shape[1]):
            return
        ch = self.files[self.selected_index].channel(self.shown_channel[self.selected_index])
        unit = ch.unit if ch is not None else ""
        st = self.stats[self.selected_index]
        factor, unit_label = idp.display_unit(unit, max(abs(st.r_lo), abs(st.r_hi)))
        text = f"x={x}, y={y} px"
        px = self.display_px[self.selected_index]
        if px is not None:
            text += f"  ({fmt_length(x * px, 4)}, {fmt_length(y * px, 4)})"
        text += f"   value = {data[y, x] * factor:.5g} {unit_label}".rstrip()
        self.ui.label_2.setText(text)

    def prepare_display(self, channel):
        """
        Display image of a channel and the size of one displayed pixel in nm.
        Incomplete scan lines (NaN) are cropped, and non-square pixels are resampled to the
        finer of the two pixel sizes so distances are the same along x and y.
        """
        img_data = self.crop_NaN(channel.data)
        if img_data.size == 0:
            raise ValueError("image contains no valid data")
        px_x, px_y = channel.px_size_x, channel.px_size_y
        if not channel.calibrated:
            return img_data, None
        if abs(px_x / px_y - 1) < 1e-3:
            return img_data, px_x
        rows, cols = img_data.shape
        if px_x > px_y:
            img_data = zoom(img_data, (1, px_x / px_y), order=1)
            return img_data, cols * px_x / img_data.shape[1]
        img_data = zoom(img_data, (px_y / px_x, 1), order=1)
        return img_data, px_x

    def crop_NaN(self, img_data):
        """
        Remove scan lines containing NaN values (incomplete scans); remaining NaNs are filled
        with the mean so filters keep working.
        """
        img_data = np.asarray(img_data, dtype=float)
        if np.isnan(img_data).any():
            img_data = img_data[~np.isnan(img_data).any(axis=1)]
            if img_data.size == 0:
                return img_data
        if not np.isfinite(img_data).all():
            img_data = np.where(np.isfinite(img_data), img_data, np.nanmean(img_data[np.isfinite(img_data)]))
        return img_data

    def update_image(self, channel_name=None, initialize=True, modify=None, index=None):
        """
        Re-render the image at `index` (default: the selected one) with the given channel and
        modifications, replacing its viewer.
        """
        index = self.selected_index if index is None else index
        if not 0 <= index < len(self.files):
            return
        f = self.files[index]
        ch = (f.channel(channel_name) if channel_name else None) or f.default()
        try:
            # display image of the channel (cropped / square pixels), reused while unchanged
            key = (ch.name, ch.px_size_x, ch.px_size_y)
            cached = self._base_cache.get(index)
            if cached is None or cached[0] != key:
                base, px = self.prepare_display(ch)
                self._base_cache[index] = (key, base, px)
            else:
                _, base, px = cached
            # processing steps; unchanged leading steps come from the per-image cache
            pipe = self._pipe_cache.get(index)
            if pipe is None or pipe[0] != id(base):
                pipe = (id(base), [])
                self._pipe_cache[index] = pipe
            newImage = self.processing.run(base, pipe[1])
            stats = idp.ImageStats(newImage)
            qimage = self.to_qimage(newImage, stats)
        except Exception as e:
            self.updateBrowserText(f"Could not display {f.display_name} / {ch.name}: {e}")
            return

        # Same viewer, new pixels: keeps its zoom and the profile lines drawn on it
        viewer = self.ui.stackedWidget.widget(index)
        if viewer is None:
            viewer = self.make_Viewer(spmImage=qimage)
            self.ui.stackedWidget.insertWidget(index, viewer)
        else:
            if viewer.pixmap() is None or viewer.pixmap().size() != qimage.size():
                viewer.zoomStack = []
            viewer.setImage(qimage)
        self.shown_channel[index] = ch.name
        self.display_px[index] = px
        self.processed[index] = newImage
        self.stats[index] = stats
        if index == self.selected_index:
            self.ui.stackedWidget.setCurrentIndex(index)
            self.current_viewer = viewer
            self.display_controls.set_image(stats, ch.unit)
            self.refresh_info()

    def refresh_info(self):
        self.showImageStats()
        self.showScaleBar()
        self.update_colorbar()
        self.show_metadata()
        if self.profile_window is not None and self.profile_window.isVisible():
            self.profile_window.refresh()
        if self.surface_window is not None:
            self.surface_window.page_changed()

    # ------------------------------------------------------------------ line profiles

    PROFILE_LABELS = [f"Profile {i}" for i in range(1, 13)]

    def _apply_viewer_mode(self, viewer):
        """Profile mode: left-drag draws profile lines; otherwise left-drag zooms (default)."""
        if self.profile_mode:
            viewer.regionZoomButton = None
            viewer.drawROI = "VectorROI"
        else:
            viewer.drawROI = None
            viewer.regionZoomButton = QtCore.Qt.MouseButton.LeftButton

    def set_profile_mode(self, on):
        self.profile_mode = bool(on)
        for i in range(self.ui.stackedWidget.count()):
            self._apply_viewer_mode(self.ui.stackedWidget.widget(i))
        if on:
            self.ui.label_2.setText("Profile mode: left-drag to draw a line, drag its ends to adjust, right click "
                                    "deletes it. Click 'Line profile' again to go back to zooming.")
            if self.profile_window is None:
                self.profile_window = ProfileWindow(self._profile_data, self)
                self.profile_window.clearRequested.connect(self.clear_profiles)
            self.profile_window.show()
            self.profile_window.raise_()
            self.profile_window.refresh()
        else:
            self.ui.label_2.setText("Zoom mode: drag to zoom, right click to zoom out.")

    def _on_viewer_rois(self, viewer, action, rois):
        if action == "drawn":
            used = {r.label for r in viewer.rois("VectorROI")}
            free = [l for l in self.PROFILE_LABELS if l not in used]
            viewer.setROILabel(rois, free[0] if free else self.PROFILE_LABELS[-1])
            return                              # the relabel triggers the refresh
        if viewer is self.current_viewer and self.profile_window is not None and self.profile_window.isVisible():
            self.profile_window.refresh()

    def clear_profiles(self):
        if self.current_viewer is not None:
            self.current_viewer.clearROIs(emit=True)

    def _profile_data(self):
        if not self.files or self.current_viewer is None or not 0 <= self.selected_index < len(self.processed):
            return {}
        from ui.PyQtImageViewer.QtImageViewer import label_color
        idx = self.selected_index
        ch = self.files[idx].channel(self.shown_channel[idx])
        lines = []
        for r in self.current_viewer.rois("VectorROI"):
            line = r.line()
            lines.append((r.label, label_color(r.label, self.PROFILE_LABELS), line.x1(), line.y1(), line.x2(), line.y2()))
        return {"data": self.processed[idx], "px": self.display_px[idx], "unit": ch.unit if ch else "",
                "name": ch.name if ch else "", "lines": lines}

    # ------------------------------------------------------------------ Fourier filter

    def open_fourier_filter(self):
        """Edit the (first) Fourier step of the pipeline, adding one if there is none."""
        if not self.files:
            return
        from core.fourier_filter import FourierCard
        card = next((c for c in self.processing.cards if isinstance(c, FourierCard)), None)
        if card is None:
            card = self.processing.add_step("fourier", emit=False)
        self.edit_fourier_filter(card)

    def edit_fourier_filter(self, card):
        """Open the interactive editor with the image as it enters this step of the pipeline."""
        from core.fourier_filter import FourierFilterDialog
        from core.image_adapt_dialog import run_pipeline
        if not self.files or self.selected_index not in self._base_cache:
            return
        idx = self.selected_index
        _, base, px = self._base_cache[idx]
        before = self.processing.steps()[:self.processing.index_of(card)]
        data, _ = run_pipeline(np.asarray(base, dtype=float), before)
        settings = self.display_controls.settings

        def render(img):
            return idp.render(img, idp.ImageStats(img), settings)

        values = card.values()
        dlg = FourierFilterDialog(data, px, render, values.get("shapes"), values.get("mode"),
                                  values.get("smooth", 1.0), self)
        if dlg.exec():
            card.set_filter(*dlg.get_filter())
            card.enable_cb.setChecked(True)
            self.image_processing_ui_show() if not self.ui.widget_9.isVisible() else None
            shapes, mode, _ = dlg.get_filter()
            self.updateBrowserText(f"Fourier filter: {len(shapes)} region(s), {mode}.")

    # ------------------------------------------------------------------ 3D view

    def open_3d_view(self):
        if not self.files:
            return
        from core.surface_view import SurfaceViewDialog
        try:
            if self.surface_window is None:
                self.surface_window = SurfaceViewDialog(self._surface_data, self)
            else:
                self.surface_window.refresh()
        except Exception as e:          # e.g. no OpenGL on this machine
            self.updateBrowserText(f"3D view unavailable: {e}")
            return
        self.surface_window.show()
        self.surface_window.raise_()

    def _surface_data(self):
        if not self.files or not 0 <= self.selected_index < len(self.processed):
            return {}
        idx = self.selected_index
        f = self.files[idx]
        ch = f.channel(self.shown_channel[idx])
        data = self.processed[idx]
        return {"data": data, "px": self.display_px[idx], "unit": ch.unit if ch else "",
                "rgb": idp.render(data, self.stats[idx], self.display_controls.settings),
                "levels": idp.levels(self.stats[idx], self.display_controls.settings),
                "title": f"{f.display_name} / {ch.name if ch else ''}"}

    # ------------------------------------------------------------------ display (levels, colormap, colorbar)

    def on_display_changed(self, final=True):
        """Recolour the selected image with the new display settings (no reprocessing)."""
        if not self.files or self.current_viewer is None:
            return
        idx = self.selected_index
        self.current_viewer.setImage(self.to_qimage(self.processed[idx], self.stats[idx]))
        self.update_colorbar(regenerate=final)
        if final and self.surface_window is not None:
            self.surface_window.page_changed()

    def setColorMap(self, *_):
        name = self.ui.comboBox.currentData()
        if not name:
            return
        self.display_controls.set_cmap(name)
        self.on_display_changed(True)

    def changeColorbarState(self, state):
        self.colorbarActive = bool(state)
        self.update_colorbar()

    def update_colorbar(self, viewer=None, index=None, regenerate=True):
        """
        Show / hide the colorbar to the right of the image (outside the data, so it never hides
        features; it is included when the image is saved).  While a slider is being dragged the
        existing bar is kept (regenerate=False) and only redrawn when the drag ends.
        """
        viewer = viewer or self.current_viewer
        index = self.selected_index if index is None else index
        if viewer is None or not viewer.hasImage() or not 0 <= index < len(self.files):
            return
        img_rect = QRectF(viewer.pixmap().rect())
        item = getattr(viewer, "_colorbar_item", None)
        if not self.colorbarActive:
            if item is not None:
                viewer.scene.removeItem(item)
                viewer._colorbar_item = None
            if viewer.sceneRect() != img_rect:
                viewer.setSceneRect(img_rect)
                viewer.updateViewer()
            return
        if item is None or regenerate:
            ch = self.files[index].channel(self.shown_channel[index])
            # The scene is in image pixels and usually shown enlarged, so render the bar at
            # ~2400 px tall and draw it scaled down: sharp text at any zoom and in saved images.
            oversample = min(8.0, max(2.0, 2400.0 / img_rect.height()))
            rgba = idp.colorbar_image(self.stats[index], self.display_controls.settings, img_rect.height(),
                                      label=ch.name if ch is not None else "", unit=ch.unit if ch is not None else "",
                                      oversample=oversample)
            h, w = rgba.shape[:2]
            pixmap = QPixmap.fromImage(QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy())
            if item is None:
                item = QGraphicsPixmapItem()
                item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
                viewer.scene.addItem(item)
                viewer._colorbar_item = item
            item.setPixmap(pixmap)
            item.setScale(1.0 / oversample)
        bar = item.sceneBoundingRect()
        gap = max(4.0, img_rect.width() * 0.02)
        item.setPos(img_rect.width() + gap, (img_rect.height() - bar.height()) / 2)
        full = QRectF(0, 0, img_rect.width() + gap + bar.width(), img_rect.height())
        if viewer.sceneRect() != full:
            viewer.setSceneRect(full)
            viewer.zoomStack = []
            viewer.updateViewer()

    def display_selected_image(self, initialize=True):
        selected_items = self.ui.listWidget.selectedItems()
        if not selected_items:
            return
        self.selected_index = self.ui.listWidget.row(selected_items[0])
        self.ui.stackedWidget.setCurrentIndex(self.selected_index)
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.display_channels()
        self.apply_mod()        # renders the selected channel with the active modifications

    def display_channels(self):
        f = self.files[self.selected_index]
        names = f.channel_names()
        wanted = self.selected_channel_name if self.selected_channel_name in names else f.default().name
        self.ui.listWidget_2.blockSignals(True)
        self.ui.listWidget_2.clear()
        self.ui.listWidget_2.addItems(names)
        self.ui.listWidget_2.setCurrentRow(names.index(wanted))
        self.ui.listWidget_2.blockSignals(False)
        self.shown_channel[self.selected_index] = wanted

    def update_selected_channel(self):
        """
        Update the selected channel image in the viewer.
        """
        selected = self.ui.listWidget_2.selectedItems()
        if not selected or not self.files:
            return
        self.selected_channel_name = selected[0].text()
        self.apply_mod()

    def _current_channel_name(self, index=None):
        index = self.selected_index if index is None else index
        f = self.files[index]
        if self.selected_channel_name and f.channel(self.selected_channel_name):
            return self.selected_channel_name
        return self.shown_channel[index] or f.default().name

    # ------------------------------------------------------------------ saving

    def save_image(self):
        """
        Save the displayed image (with modifications and overlays such as the scale bar) as a JPG
        and record its pixel scale in metadata.csv next to it.
        """
        if not self.files or self.current_viewer is None:
            return None
        f = self.files[self.selected_index]
        channel = self.shown_channel[self.selected_index]

        self.saved_output_dir = os.path.join(self.dir_path, "ScanBound_image_output")
        os.makedirs(self.saved_output_dir, exist_ok=True)
        file = f"{f.stem}_{_safe_name(channel)}.jpg"
        full_path = os.path.join(self.saved_output_dir, file)

        # Render the whole scene: image, overlays (scale bar) and the colorbar beside the image.
        viewer = self.current_viewer
        image_size = viewer.image().size()
        source = viewer.sceneRect()
        # With a colorbar the export is enlarged by an integer factor (data pixels become crisp
        # k x k blocks) so its text stays legible; the pixel scale written below follows.
        k = 1
        if getattr(viewer, "_colorbar_item", None) is not None:
            k = max(1, math.ceil(1024 / max(1, image_size.height())))
        output_image = QImage(int(round(source.width() * k)), int(round(source.height() * k)),
                              QImage.Format.Format_RGB32)
        output_image.fill(QColor("white"))
        painter = QPainter(output_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        profiles = [r for r in viewer.rois() if r.isVisible()]
        for r in profiles:                       # profile lines are a measuring aid, not part of the image
            r.setVisible(False)
        viewer.scene.render(painter, QRectF(output_image.rect()), source)
        for r in profiles:
            r.setVisible(True)
        painter.end()
        output_image.save(full_path, format='JPG', quality=100)

        px = self.display_px[self.selected_index]
        px = None if px is None else px / k
        self._write_metadata_row(os.path.join(self.saved_output_dir, 'metadata.csv'), {
            "Filename": file,
            "Channel": channel,
            "Modifications": self.processing.summary(),
            "Dimensions": f"{image_size.width() * k} x {image_size.height() * k}",
            "Pixel Scale X (nm)": "" if px is None else px,
            "Pixel Scale Y (nm)": "" if px is None else px,
        })
        return full_path  # Return the path of the saved image for confirmation

    @staticmethod
    def _write_metadata_row(metadata_file, row):
        """Insert or replace the row for row['Filename'] (the pixel scale may have changed)."""
        fields = ["Filename", "Channel", "Modifications", "Dimensions", "Pixel Scale X (nm)", "Pixel Scale Y (nm)"]
        rows = []
        if os.path.exists(metadata_file):
            with open(metadata_file, newline='', encoding='utf-8-sig') as fh:
                reader = csv.DictReader(fh)
                fields += [c for c in (reader.fieldnames or []) if c not in fields]
                rows = [r for r in reader if r.get("Filename") != row["Filename"]]
        rows.append(row)
        with open(metadata_file, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def save_all_images(self):
        """
        Save all images with the selected channel (where available) and the active modifications.
        """
        if not self.files:
            return
        starting_index = self.selected_index
        for idx in range(len(self.files)):
            self.selected_index = idx
            self.update_image(self._current_channel_name(idx), index=idx)
            self.current_viewer = self.ui.stackedWidget.widget(idx)
            self.showScaleBar()
            self.save_image()
        self.selected_index = starting_index
        self.ui.stackedWidget.setCurrentIndex(starting_index)
        self.current_viewer = self.ui.stackedWidget.widget(starting_index)
        self.refresh_info()
        self.updateBrowserText(f"Saved {len(self.files)} images to {self.saved_output_dir}")

    # ------------------------------------------------------------------ scale

    def showImageStats(self):
        if not self.files:
            return
        f = self.files[self.selected_index]
        ch = f.channel(self.shown_channel[self.selected_index]) or f.default()
        px = self.display_px[self.selected_index]
        if px is None:
            self.ui.label_5.setText("Pixel scale: not calibrated")
            self.ui.label_6.setText("Scan size: unknown")
            return
        self.ui.label_5.setText(f"Pixel scale: {px:.5g} nm/px" + (" (manual)" if ch.manual_scale else ""))
        w, h = ch.scan_size()
        self.ui.label_6.setText(f"Scan size: {fmt_length(w)} × {fmt_length(h)}")

    def set_manual_scale(self):
        if not self.files:
            return
        f = self.files[self.selected_index]
        current = self.display_px[self.selected_index] or 1.0
        value, ok = QInputDialog.getDouble(self, "Set pixel scale",
                                           f"Pixel size of {f.display_name} (nm per pixel):",
                                           current, 1e-6, 1e6, 6)
        if not ok:
            return
        f.set_manual_scale(value)
        self.updateBrowserText(f"Manual pixel scale for {f.display_name}: {value:g} nm/px")
        self.apply_mod()

    def changeScaleBarState(self, state):
        self.scaleBarActive = bool(state)
        self.showScaleBar()  # Update the scale bar visibility based on the new state

    @staticmethod
    def _nice_bar_length(width_nm, target_fraction=0.2):
        """1-2-5 series length close to target_fraction of the image width."""
        target = width_nm * target_fraction
        exponent = math.floor(math.log10(target))
        candidates = [m * 10 ** e for e in (exponent - 1, exponent, exponent + 1) for m in (1, 2, 5)]
        valid = [c for c in candidates if 0.05 * width_nm <= c <= 0.4 * width_nm] or candidates
        return min(valid, key=lambda c: abs(math.log(c) - math.log(target)))

    def showScaleBar(self):
        if self.current_viewer is None:
            return
        self.current_viewer.removeItems()
        if not self.scaleBarActive or not self.files:
            return
        px = self.display_px[self.selected_index]
        if px is None:
            self.ui.label_2.setText("No pixel scale for this image: set one with \"Set Manual Scale\".")
            return
        size = self.current_viewer.image().size()
        width_px, height_px = size.width(), size.height()
        bar_nm = self._nice_bar_length(width_px * px)
        bar_px = bar_nm / px
        white = QColor("#FFFFFF")
        font = max(6, int(round(height_px * 0.035)))
        bar_h = max(2, int(round(height_px * 0.018)))
        margin = max(3, int(round(width_px * 0.02)))
        self.current_viewer.addScaleBar(x=margin, y=height_px - margin - bar_h, length=bar_px, height=bar_h,
                                        color=white)
        self.current_viewer.addTextOverlay(fmt_length(bar_nm, 3), margin, height_px - margin - bar_h - font * 2.1,
                                           font_size=font, color=white)

    def copy_to_clipboard(self):
        if self.current_viewer is None:
            return
        full_path = self.save_image()  # Save the current image first

        # Now copy the saved image to the clipboard
        from PIL import Image
        import io
        import win32clipboard

        image = Image.open(full_path)
        output = io.BytesIO()
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]  # BMP file header is 14 bytes,
        output.close()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()

    # ------------------------------------------------------------------ processing

    def image_processing_ui_show(self):
        """
        Show the image processing menu.
        """
        if not self.ui.widget_9.isVisible():
            self.ui.widget_9.show()
        else:
            self.ui.widget_9.hide()

    def apply_mod(self):
        """
        Re-render the selected image with the processing steps of the panel.
        """
        if not self.files:
            return
        self.update_image(channel_name=self._current_channel_name())

    def close_process_menu(self):
        """
        Close the image processing menu (the steps stay active).
        """
        self.ui.widget_9.hide()

    def apply_mod_all(self):
        """
        Process all loaded images with the current steps.
        """
        if not self.files:
            return
        for idx in range(len(self.files)):
            self.update_image(channel_name=self._current_channel_name(idx), index=idx)
        self.ui.stackedWidget.setCurrentIndex(self.selected_index)
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.updateBrowserText(f"Processed {len(self.files)} image(s): {self.processing.summary() or 'no steps'}")

    def reset_mods(self):
        """
        Back to the default processing steps (all off).
        """
        self.processing.reset()

    # ------------------------------------------------------------------ hand-off

    def send_to_imageLabeling(self):
        """
        Send the current dir to the image labeling module.
        """
        from pages_functions.ImageLabeling import ImageLabeling
        from PyQt6.QtWidgets import QMessageBox

        if not self.dir_path:
            return
        self.saved_output_dir = os.path.join(self.dir_path, "ScanBound_image_output")
        os.makedirs(self.saved_output_dir, exist_ok=True)

        existing = os.listdir(self.saved_output_dir)
        missing = [f for f in self.files
                   if not any(n.startswith(f.stem + "_") and n.lower().endswith(".jpg") for n in existing)]
        if missing:
            reply = QMessageBox.question(
                self,
                "Save Images",
                f"{len(missing)} image(s) have not been saved as JPG yet. Would you like to save all now? "
                "If not, then the label module will open with existing JPG files.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Images for labelling must contain only the data: no colorbar strip beside them.
                colorbar = self.colorbarActive
                self.colorbarActive = False
                try:
                    self.save_all_images()
                finally:
                    self.colorbarActive = colorbar
                    self.update_colorbar()

        if self.main_window:
            self.main_window.ImageLabeling_btn.click()
            for i in range(self.main_window.ui.tabWidget.count()):
                widget = self.main_window.ui.tabWidget.widget(i)
                if isinstance(widget, ImageLabeling):
                    widget.choose_directory(self.saved_output_dir)
                    break


def handleLeftClick(x, y):
    row = int(y)
    column = int(x)
    print("Clicked on image pixel (row="+str(row)+", column="+str(column)+")")
