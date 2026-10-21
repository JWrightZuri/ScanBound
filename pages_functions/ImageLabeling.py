import pandas as pd
from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QInputDialog, QTreeWidgetItem, QListWidgetItem, QApplication, \
    QPushButton, QCheckBox, QToolButton, QMenu, QSizePolicy
import math
import PyQt6.QtGui as QtGui
import PyQt6.QtCore as QtCore
from ui.pages.ImageLabeling_ui import Ui_Form
from ui.popUps.automaticAnnotation_ui import Ui_Form as Ui_AutomaticAnnotations
from ui.popUps.predictions_ui import Ui_Form as Ui_Predictions
from PyQt6.QtWidgets import QFileDialog, QLabel, QGridLayout, QTableWidgetItem
import os
from collections import Counter
from core.label_editor import LabelEditor, LabelManager
from ui.PyQtImageViewer.QtImageViewer import (QtImageViewer, ANNOTATION_COLUMNS, FRAME_COLUMNS, EMPTY_LABEL,
                                              SYMMETRY_LABEL, ORIENT_MODE, label_color)
import numpy as np
import csv
from core.first_pass_segment import run_first_pass_segment
from core.second_pass_segment import run_second_pass_segment
from skimage.io import imread
import matplotlib.pyplot as plt
from PyQt6.QtGui import QColor
import scipy.stats as stats


def polyline_stats(xy, px_scale=None):
    """
    Length and straightness of a polyline given as (N, 2) points.
    Returns (length_nm or None, length_px, r_squared, linearity_percent).
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    seg_length = 0.0
    r_squared = None
    linearity = None
    if len(xy) >= 2:
        x, y = xy[:, 0], xy[:, 1]
        seg_d = np.hypot(np.diff(x), np.diff(y))
        seg_length = float(seg_d.sum())

        # Regress over the polyline resampled at ~1 px spacing: simplified lines have
        # only a few vertices, so a fit through the vertices alone would be meaningless
        # (r² = 1 for two points).
        cum = np.concatenate([[0.0], np.cumsum(seg_d)])
        if cum[-1] > 0:
            ts = np.linspace(0.0, cum[-1], max(int(np.ceil(cum[-1])) + 1, 2))
            xd, yd = np.interp(ts, cum, x), np.interp(ts, cum, y)
        else:
            xd, yd = x, y
        try:
            r_squared = float(stats.linregress(xd, yd).rvalue ** 2)
        except ValueError:      # all x identical: perfectly vertical segment
            r_squared = 1.0

        ideal_length = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        if ideal_length == 0 or seg_length == 0:
            linearity = 100.0
        else:
            linearity = float(np.clip(ideal_length / seg_length * 100.0, 0, 100.0))

    length_nm = seg_length * px_scale if px_scale else None
    return length_nm, seg_length, r_squared, linearity


def thicken_polyline(points, half_width, end_buffer=0.0):
    """Polygon outline (M, 2) of a polyline widened by half_width on each side and extended
    by end_buffer at both ends."""
    p = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    if end_buffer > 0:
        for end, nxt in ((0, 1), (-1, -2)):
            d = p[end] - p[nxt]
            n = np.hypot(*d)
            if n > 0:
                p[end] = p[end] + d / n * end_buffer
    tang = np.empty_like(p)
    tang[0] = p[1] - p[0]
    tang[-1] = p[-1] - p[-2]
    tang[1:-1] = p[2:] - p[:-2]
    length = np.hypot(tang[:, 0], tang[:, 1])
    safe = np.where(length > 0, length, 1.0)
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1) / safe[:, None]
    normals[length == 0] = 0.0
    left = p + normals * half_width
    right = p - normals * half_width
    return np.vstack([left, right[::-1]])


# pyuic6.exe .\ImageLabeling.ui -o .\ImageLabeling_ui.py
class ImageLabeling(QWidget):
    def __init__(self, main_window=None):
        super(ImageLabeling, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.main_window = main_window

        # ROI changes are coalesced: label counts / examples are refreshed once per burst
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(30)
        self._refresh_timer.timeout.connect(self._refresh_roi_panels)
        self._syncing_label = False   # True while the label list is updated to mirror a selection

        # Label list: one model (LabelManager) + its editor panel (core/label_editor.py).
        # Every change is applied at once and announced by signals handled below.
        self.label_manager = LabelManager(self)
        self.label_editor = LabelEditor(self.label_manager,
                                        color_for=lambda name: label_color(name, self.labelList),
                                        count_provider=self._label_counts, parent=self)
        self.label_input_dialog = self.label_editor           # shown in the side panel (widget_9)
        label_ui_layout = self.ui.widget_10.layout()
        if label_ui_layout is None:
            label_ui_layout = QVBoxLayout(self.ui.widget_10)
        label_ui_layout.addWidget(self.label_editor)
        self.label_manager.changed.connect(self._on_labels_changed)
        self.label_manager.renamed.connect(self._on_label_renamed)
        self.label_manager.removed.connect(self._on_label_removed)
        self.label_editor.labelActivated.connect(self._select_label_row)
        self.label_editor.saveExamplesRequested.connect(self.saveLabelExamples)
        self.label_editor.closeRequested.connect(self.closeLabelInput)

        # Create automaticAnnotations popup
        self.autoAnnotations_ui = Ui_AutomaticAnnotations()
        self.autoAnnotations_dialog = QWidget(self)
        self.autoAnnotations_ui.setupUi(self.autoAnnotations_dialog)
        autoAnnotations_layout = self.ui.widget_10.layout()
        if autoAnnotations_layout is None:
            autoAnnotations_layout = QVBoxLayout(self.ui.widget_10)
        autoAnnotations_layout.addWidget(self.autoAnnotations_dialog)
        self.ui.widget_9.hide()

        # Create predictions popup
        self.predictions_ui = Ui_Predictions()
        self.predictions_dialog = QWidget(self)
        self.predictions_ui.setupUi(self.predictions_dialog)
        predictions_layout = self.ui.widget_10.layout()
        if predictions_layout is None:
            predictions_layout = QVBoxLayout(self.ui.widget_10)
        predictions_layout.addWidget(self.predictions_dialog)
        self.ui.widget_9.hide()

        # Image adaptation: mask pipeline + ROI generation dialog (core/image_adapt_dialog.py)
        self.ui.pushButton_20.clicked.connect(self.adaptImage_show)
        self.adapted_image_params = None   # last pipeline used in the dialog (re-opened with it)
        self.roi_settings = None           # RoiSettings chosen in the dialog (also used by auto_segment_to_rois)

        # Display all widgets in stackedWidget
        self.ui.stackedWidget.removeWidget(self.ui.page)
        self.ui.stackedWidget.removeWidget(self.ui.page_2)
        self.image_paths = []  # Store full paths to images
        self.data_dir = None

        # Connect pushButton to open directory dialog
        self.ui.pushButton.clicked.connect(self.choose_directory)
        self.ui.label_2.setText("Select a directory to view files")
        self.browserText = []
        self.metadata_dict = {}
        # self.current_viewer._pxScale= None
        self.symmetryVectorMode = False
        self.symmetryVector = []  # Track the symmetry vector(s)
        # Connect listWidget item selection to image display
        # self.ui.listWidget.itemSelectionChanged.connect(self.display_selected_image)

        # Connect listWidget item selection to image display
        self.ui.listWidget.itemSelectionChanged.connect(self.display_selected_image)

        self.getActiveMode = 'Selection'  # Default mode is 'Selection'
        self.ui.pushButton_11.clicked.connect(self.activate_selection_mode)
        self.ui.pushButton_10.clicked.connect(self.activate_zoom_mode)
        self.ui.pushButton_7.clicked.connect(self.activate_bounding_box_mode)
        self.ui.pushButton_8.clicked.connect(self.activate_vector_mode)
        self.ui.pushButton_9.clicked.connect(self.activate_point_mode)
        self.ui.pushButton_14.clicked.connect(self.activate_line_segment_mode)
        self.ui.pushButton_13.clicked.connect(self.activate_polygon_mode)

        self.ui.pushButton_2.clicked.connect(self.setScale)
        self.ui.pushButton_5.clicked.connect(self.load_annotations)
        self.ui.pushButton_3.clicked.connect(self.save_annotations)
        self.ui.pushButton_12.clicked.connect(self.clearAnnotations)
        # self.ui.pushButton_6.clicked.connect(self.runAutoSegment)
        # self.ui.pushButton_6.clicked.connect(self.auto_segment_to_rois)
        # self.ui.pushButton_6.clicked.connect(self.runGNRSegmentation)
        self.ui.pushButton_15.clicked.connect(self.saveLabelClips)
        self.ui.pushButton_16.clicked.connect(self.clearActivityLog)
        self.ui.pushButton_18.clicked.connect(self.collectSymmetryVector)
        # self.ui.pushButton_19.clicked.connect(self.showResults)
        self.roi_analysis_button = QPushButton("Analyze ROIs", parent=self.ui.widget_2)
        self.roi_analysis_button.setToolTip("Coverage, size distribution and label composition per label, "
                                            "for the current image or all loaded images.")
        self.ui.verticalLayout.insertWidget(self.ui.verticalLayout.indexOf(self.ui.pushButton_18) + 1,
                                            self.roi_analysis_button)
        self.roi_analysis_button.clicked.connect(self.open_roi_analysis)
        self._roi_analysis_dialog = None
        self._build_orientation_tools()
        # self.ui.radioButton.clicked.connect(self.toggleAutoVector)
        # self.ui.pushButton_21.setVisible(False)  # Hide the GNR button unless GNR tag is active for a label
        # self.ui.pushButton_21.clicked.connect(self.convert_all_lines_to_polygons)

        # Buttons for side_ui_popup
        self.ui.pushButton_4.clicked.connect(self.label_ui_show)
        # Quick label actions directly on the label picker (right click)
        self.ui.listWidget_2.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.listWidget_2.customContextMenuRequested.connect(self._label_picker_menu)
        self.ui.listWidget_2.setToolTip("Label for new ROIs (number keys 0-9). Picking a label relabels the "
                                        "selected ROIs. Right click to add / rename / remove labels.")
        self.current_label = None  # Track currently selected label
        self.labels_ready = False
        # Connect label selection changes
        self.ui.listWidget_2.itemSelectionChanged.connect(self.on_label_selection_changed)
        self.distinguish_rois = QCheckBox("Vary shades within a label", parent=self.ui.widget_7)
        self.distinguish_rois.setToolTip("Draw neighbouring ROIs of the same label in lighter / darker shades of the "
                                         "label colour so touching or overlapping ROIs can be told apart.")
        self.distinguish_rois.setChecked(True)
        self.distinguish_rois.toggled.connect(self.set_distinguish_rois)
        self.ui.verticalLayout_6.insertWidget(self.ui.verticalLayout_6.indexOf(self.ui.listWidget_2) + 1,
                                              self.distinguish_rois)
    
        self.current_viewer = None  # Track the current viewer for ROI operations
        # automaticAnnotations widget
        # self.ui.pushButton_20.clicked.connect(self.annotations_ui_show)
        # self.autoAnnotations_ui.pushButton.clicked.connect(self.closeAutoAnnotations)

        # predictions widget
        self.ui.pushButton_22.clicked.connect(self.predictions_ui_show)
        self.predictions_ui.pushButton_4.clicked.connect(self.choosePredictionModel)
        self.predictions_ui.pushButton_2.clicked.connect(self.makePrediction)
        self.predictions_ui.pushButton.clicked.connect(self.closePredictions)
        self.yolo_model_path = None  # Store the selected prediction model path

        # Ensure all popups are hidden initially
        self.closeLabelInput()
        self.closeAutoAnnotations()
        self.closePredictions()

        # Enable keyboard focus and shortcuts
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # Label list from the last session (empty on first use)
        saved = QtCore.QSettings("ScanBound", "ImageLabeling").value("labels", [])
        if isinstance(saved, str):
            saved = [saved]
        self.label_manager.set_labels(saved or [])
        self.label_editor.previews.toggled.connect(lambda on: on and self.getLabelExamples())

        # Auto load data (temporary)
        self.choose_directory(dir_path=r"C:\Users\wrja\Desktop\STM_Data\GNR_files\20260716_AuMica_694a_44974_44980_44981\saved_output")
        self.autoVector = False

    def choose_directory(self, dir_path):
        if not dir_path:
            dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", directory= self.data_dir if self.data_dir else os.getcwd())
            if not dir_path or not os.path.isdir(dir_path):
                self.updateBrowserText("No directory selected.")
                return
        if dir_path:
            self.data_dir = dir_path
            self.ui.listWidget.clear()
            self.image_paths = []
            # Clear all widgets from stackedWidget before loading new images
            while self.ui.stackedWidget.count() > 0:
                widget = self.ui.stackedWidget.widget(0)
                self.ui.stackedWidget.removeWidget(widget)
                widget.deleteLater()
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')
            for item in os.listdir(dir_path):
                if item.lower().endswith(image_extensions):
                    self.ui.listWidget.addItem(item)
                    full_path = os.path.join(dir_path, item)
                    self.image_paths.append(full_path)

            # Change the text of label
            self.ui.label.setText(f"Selected Directory: {dir_path.split('/')[-1]}")
            self.updateBrowserText(f"Loaded directory: {dir_path}")
        # Check for metadata file in the selected directory
        metadata_file = None
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.csv', '.json')) and 'meta' in fname.lower():
                metadata_file = os.path.join(dir_path, fname)
                break

        self.metadata_dict = {}
        if metadata_file:
            if metadata_file.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(metadata_file)
                    self.metadata_dict = df.to_dict(orient='list')
                    self.updateBrowserText(f"Loaded metadata from {os.path.basename(metadata_file)}")
                except Exception as e:
                    self.updateBrowserText(f"Failed to load metadata: {e}")
            # elif metadata_file.lower().endswith('.json'):
            #     try:
            #         with open(metadata_file, 'r') as f:
            #             self.metadata_dict = json.load(f)
            #         self.updateBrowserText(f"Loaded metadata from {metadata_file}")
            #     except Exception as e:
            #         self.updateBrowserText(f"Failed to load metadata: {e}")
        self.current_viewer = None   # the old viewers (and their ROIs) were removed above
        self.load_images()
        self.get_scan_dim()  # Load the pixel scale for each image based on metadata

    def updateBrowserText(self, text):
        self.browserText.append(text)
        self.ui.textBrowser.setText('\n \n'.join(self.browserText))
        self.ui.textBrowser.verticalScrollBar().setValue(self.ui.textBrowser.verticalScrollBar().maximum())  # Scroll to the bottom

    def load_images(self):
        for path in self.image_paths:
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
            
            # Load an image file to be displayed (will popup a file dialog).
            viewer.open(filepath=path)
            viewer.leftMouseButtonReleased.connect(handleLeftClick)
            viewer.roisChanged.connect(self.onRoisChanged)
            viewer.roiSelected.connect(self.roiSelection)
            viewer._imgPath = path  # Store the image path in the viewer instance
            viewer.setLabelList(self.labelList)
            if hasattr(self, "distinguish_rois"):
                viewer.setDistinguishROIs(self.distinguish_rois.isChecked())
            self.ui.stackedWidget.addWidget(viewer)
            self.ui.label_2.setText("Press the box button to draw bounding boxes on the image")

    def _viewers(self):
        """All image viewers, in the same order as self.image_paths."""
        viewers = []
        for i in range(self.ui.stackedWidget.count()):
            widget = self.ui.stackedWidget.widget(i)
            if isinstance(widget, QtImageViewer):
                viewers.append(widget)
        return viewers

    def _current_viewer(self):
        widget = self.ui.stackedWidget.currentWidget()
        return widget if isinstance(widget, QtImageViewer) else None

    @property
    def annotations_df(self):
        """
        All annotations of all images (columns ANNOTATION_COLUMNS), built from the ROIs held
        by the viewers, which are the single source of truth.  Each viewer caches its own
        table until its ROIs change, so this is cheap to call.
        """
        frames = [f for f in (v.annotationsFrame() for v in self._viewers()) if len(f)]
        if not frames:
            return pd.DataFrame(columns=FRAME_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def set_distinguish_rois(self, enabled):
        for viewer in self._viewers():
            viewer.setDistinguishROIs(enabled)

    def _set_label_list_on_viewers(self):
        for viewer in self._viewers():
            viewer.setLabelList(self.labelList)

    def _populate_label_widget(self):
        """Fill the label picker (listWidget_2) with "Empty Label" + self.labelList, with colour icons
        that match the ROI colours."""
        self.ui.listWidget_2.blockSignals(True)
        self.ui.listWidget_2.clear()
        self.ui.listWidget_2.addItem(EMPTY_LABEL)
        for label_idx, label in enumerate(self.labelList, 1):
            pixmap = QtGui.QPixmap(20, 20)
            pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setBrush(QtGui.QBrush(QColor(label_color(label, self.labelList))))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(2, 2, 16, 16)
            painter.end()
            list_item = QListWidgetItem(f"{label_idx}: {label}")
            list_item.setIcon(QtGui.QIcon(pixmap))
            self.ui.listWidget_2.addItem(list_item)
        self.ui.listWidget_2.blockSignals(False)
        self._set_label_list_on_viewers()

    def _schedule_refresh(self):
        self._refresh_timer.start()

    def _refresh_roi_panels(self):
        self.labelTextDisplay()
        self.getLabelExamples()
        self.label_editor.update_counts()

    # ------------------------------------------------------------------ label list

    @property
    def labelList(self):
        """The label list (a copy); change it through self.label_manager or by assigning a new list."""
        return self.label_manager.labels()

    @labelList.setter
    def labelList(self, labels):
        self.label_manager.set_labels(labels)

    def _label_counts(self):
        """Number of ROIs per label over all images."""
        counts = Counter()
        for viewer in self._viewers():
            counts.update(r.label for r in viewer.rois())
        return dict(counts)

    def _on_labels_changed(self):
        """Label list changed (added / removed / renamed / reordered): update picker, colours, settings."""
        current = self.get_current_label()
        self._populate_label_widget()
        if current and (current == EMPTY_LABEL or current in self.label_manager):
            self._select_label_row(current)
        QtCore.QSettings("ScanBound", "ImageLabeling").setValue("labels", self.labelList)
        if hasattr(self, "ui"):
            self._schedule_refresh()

    def _rois_with_label(self, label):
        return [(v, [r for r in v.rois() if r.label == label]) for v in self._viewers()]

    def _on_label_renamed(self, old, new):
        n = 0
        for viewer, rois in self._rois_with_label(old):
            if rois:
                viewer.setROILabel(rois, new)
                n += len(rois)
        if self.current_viewer is not None and self.current_viewer._currentLabel == old:
            self.current_viewer._currentLabel = new
        self.updateBrowserText(f"Label '{old}' renamed to '{new}'" + (f" ({n} ROIs relabelled)." if n else "."))

    def _on_label_removed(self, name, how):
        n = 0
        for viewer, rois in self._rois_with_label(name):
            if not rois:
                continue
            n += len(rois)
            if how == "delete":
                viewer.deleteROIs(rois)
            else:
                viewer.setROILabel(rois, EMPTY_LABEL)
        if self.current_viewer is not None and self.current_viewer._currentLabel == name:
            self.current_viewer._currentLabel = EMPTY_LABEL
        what = "deleted" if how == "delete" else "set to Empty Label"
        self.updateBrowserText(f"Label '{name}' removed" + (f"; {n} ROI(s) {what}." if n else "."))

    def _label_picker_menu(self, pos):
        """Right-click menu of the label picker: add / rename / remove without opening the panel."""
        item = self.ui.listWidget_2.itemAt(pos)
        name = None
        if item is not None and self.ui.listWidget_2.row(item) > 0:
            text = item.text()
            name = text.split(": ", 1)[1] if ": " in text else text
        menu = QMenu(self)
        menu.addAction("Add label…", self._quick_add_label)
        if name:
            menu.addAction(f"Rename '{name}'…", lambda: self._quick_rename_label(name))
            menu.addAction(f"Remove '{name}'…", lambda: (self.label_editor.select(name),
                                                         self.label_editor._remove_selected()))
        menu.addSeparator()
        menu.addAction("Edit labels…", self.open_label_panel)
        menu.exec(self.ui.listWidget_2.mapToGlobal(pos))

    def _quick_add_label(self):
        text, ok = QInputDialog.getText(self, "Add label", "New label name:")
        if not ok:
            return
        error = self.label_manager.validate(text)
        if error:
            self.updateBrowserText(error)
            return
        self.label_manager.add(text.strip())
        self._select_label_row(text.strip())

    def _quick_rename_label(self, name):
        text, ok = QInputDialog.getText(self, "Rename label", f"New name for '{name}':", text=name)
        if not ok or text.strip() == name:
            return
        error = self.label_manager.validate(text, allow_existing=True)
        if error:
            self.updateBrowserText(error)
            return
        self.label_manager.rename(name, text.strip())

    def open_label_panel(self):
        if not self.label_editor.isVisible():
            self.label_ui_show()

    def display_selected_image(self):
        selected_items = self.ui.listWidget.selectedItems()
        if not selected_items:
            return
        selected_index = self.ui.listWidget.row(selected_items[0])
        self.ui.stackedWidget.setCurrentIndex(selected_index)
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.current_viewer.setLabelList(self.labelList)
        self.current_viewer._currentLabel = self.get_current_label() or EMPTY_LABEL
        self.labelTextDisplay(os.path.basename(self.image_paths[selected_index]))
        # if self.metadata_dict:
        self.autoSetScale()
        # else:
        #     self.setScale()
        #     self.ui.label_5.setText("Scale: No metadata available, set manually.")
        # Maintaining the active mode while switching images
        activate = {
            'Selection': self.activate_selection_mode,
            'Zoom': self.activate_zoom_mode,
            'Bbox': self.activate_bounding_box_mode,
            'Vector': self.activate_vector_mode,
            'Point': self.activate_point_mode,
            'LineSegment': self.activate_line_segment_mode,
            'Polygon': self.activate_polygon_mode,
            'Orientation': self.activate_orientation_mode,
        }.get(self.getActiveMode)
        if activate is not None:
            activate()

    def roiSelection(self, rois):
        """
        Called when the user changes the ROI selection in the viewer.
        Mirrors the label of the selection in the label list (without relabelling anything),
        shows line segment statistics and handles symmetry-vector picking.
        :param rois: list of selected ROI items (may be empty).
        """
        if not rois:
            return
        viewer = rois[0]._viewer
        if self.symmetryVectorMode and len(rois) == 1 and rois[0].roiType == "VectorROI":
            self.setSymmetryVector(rois[0])
            return

        if len(rois) == 1 and rois[0].roiType == "LineSegmentROI":
            length_nm, length_px, r_squared, linearity = polyline_stats(rois[0].pointsArray(), viewer._pxScale)
            r2_text = f"{r_squared:.2f}" if r_squared is not None else "n/a"
            if length_nm is not None:
                self.updateBrowserText(f"Selected segment stats: \n L={length_nm:.2f} nm, \n r²={r2_text}, \n Linearity={linearity:.2f}")
            else:
                self.updateBrowserText(f"Selected segment stats: \n L={length_px:.2f} pixels, \n r²={r2_text}, \n Linearity={linearity:.2f}")
        elif len(rois) > 1:
            self.ui.label_2.setText(f"{len(rois)} ROIs selected. Pick a label to relabel them all, Delete to remove them.")

        labels = {roi.label for roi in rois}
        if len(labels) == 1:
            self._select_label_row(labels.pop())

    def _select_label_row(self, label):
        """Select `label` in the label list without relabelling the selected ROIs."""
        target = 0   # "Empty Label"
        for i in range(1, self.ui.listWidget_2.count()):
            text = self.ui.listWidget_2.item(i).text()
            name = text.split(": ", 1)[1] if ": " in text else text
            if name == label:
                target = i
                break
        self._syncing_label = True
        try:
            self.ui.listWidget_2.setCurrentRow(target)
        finally:
            self._syncing_label = False
        self.current_label = self.get_current_label()
        if self.current_viewer is not None:
            self.current_viewer._currentLabel = self.current_label or EMPTY_LABEL

    def active_button_color(self, widget):
        #This will be used to change the background color of the button for the active mode
        # Reset all buttons in widget_5 to grey background except the active one
        for btn in self.ui.widget_5.findChildren(QWidget):
            if hasattr(btn, "setStyleSheet"):
                btn.setStyleSheet("background-color: #D3D3D3;")
        widget.setStyleSheet("background-color: #ADD8E6;")

    def _activate_mode(self, mode, draw_roi, button, message, zoom=False, select=False):
        viewer = self._current_viewer()
        if viewer is None:
            return
        self.current_viewer = viewer
        left, right = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton
        viewer.regionZoomButton = left if zoom else None
        viewer.selectionButton = left if select else None
        viewer.zoomOutButton = right if (zoom or select) else None
        viewer.drawROI = draw_roi
        self.getActiveMode = mode
        self.ui.label_2.setText(message)
        self.active_button_color(button)

    def activate_selection_mode(self):
        self._activate_mode('Selection', None, self.ui.pushButton_11,
                            "Selection mode: click or drag to select ROIs (Ctrl adds), drag to move, drag handles to edit. "
                            "Delete removes, a label picks relabels the selection.", select=True)

    def activate_zoom_mode(self):
        self._activate_mode('Zoom', None, self.ui.pushButton_10,
                            "Zoom mode activated. Click and drag to zoom the image. Pan with middle mouse.", zoom=True)

    def activate_bounding_box_mode(self):
        self._activate_mode('Bbox', "RectROI", self.ui.pushButton_7,
                            "Bounding box mode activated. Draw a box on the image, drag corners to resize. "
                            "Pan with middle mouse. Right click to delete.")

    def activate_vector_mode(self):
        self._activate_mode('Vector', "VectorROI", self.ui.pushButton_8,
                            "Vector mode activated. Draw a vector on the image. Right click to delete.")

    def activate_point_mode(self):
        self._activate_mode('Point', "PointROI", self.ui.pushButton_9,
                            "Point mode activated. Click to place points on the image. Right click to delete.")

    def activate_line_segment_mode(self):
        self._activate_mode('LineSegment', "LineSegmentROI", self.ui.pushButton_14,
                            "Line segment mode: click to add points, double click or Enter to finish, "
                            "Backspace removes the last point, Esc cancels. Double click a line to insert a vertex.")

    def activate_orientation_mode(self):
        self._activate_mode('Orientation', ORIENT_MODE, self.orientation_mode_button,
                            "Orientation mode: drag inside an ROI from tail to head to set its orientation, drag an "
                            "arrow tip to turn it, right click clears. Selected ROIs: ] / [ rotate 5° (Shift: 1°), F flips.")

    # ------------------------------------------------------------------ orientations

    def _build_orientation_tools(self):
        """Orientation mode button (tool column) and the 'Orientation tools' menu (right panel)."""
        self.orientation_mode_button = QPushButton("↗", parent=self.ui.widget_5)
        self.orientation_mode_button.setMaximumSize(QtCore.QSize(50, 16777215))
        self.orientation_mode_button.setToolTip("Orientation mode (Alt+8): drag inside an ROI to set its orientation")
        font = self.orientation_mode_button.font()
        font.setPointSize(14)
        self.orientation_mode_button.setFont(font)
        self.ui.verticalLayout_2.addWidget(self.orientation_mode_button)
        self.orientation_mode_button.clicked.connect(self.activate_orientation_mode)

        button = QToolButton(parent=self.ui.widget_2)
        button.setText("Orientation tools")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        menu = QMenu(button)
        auto = menu.addMenu("Auto-orient from the image")
        auto.addAction("Selected ROIs", lambda: self.auto_orient("selected"))
        auto.addAction("ROIs with the current label", lambda: self.auto_orient("label"))
        auto.addAction("All ROIs of this image", lambda: self.auto_orient("all"))
        auto.addAction("All ROIs of all images", lambda: self.auto_orient("everything"))
        auto.addSeparator()
        self.orient_polar = auto.addAction("Head / tail direction (polar)")
        self.orient_polar.setCheckable(True)
        self.orient_polar.setChecked(True)
        self.orient_polar.setToolTip("Off: only the axis (0-180°)")
        self.orient_dark = auto.addAction("Features are dark")
        self.orient_dark.setCheckable(True)
        self.orient_only_unset = auto.addAction("Only ROIs without an orientation")
        self.orient_only_unset.setCheckable(True)
        self.orient_only_unset.setChecked(True)
        menu.addAction("Convert vectors inside ROIs to orientations", self.convert_vectors_to_orientations)
        menu.addAction("Clear orientation of selected ROIs", self.clear_selected_orientations)
        menu.addSeparator()
        self.show_orient = menu.addAction("Show orientation arrows")
        self.show_orient.setCheckable(True)
        self.show_orient.setChecked(True)
        self.show_orient.toggled.connect(lambda on: [v.setShowOrientations(on) for v in self._viewers()])
        button.setMenu(menu)
        self.ui.verticalLayout.insertWidget(self.ui.verticalLayout.indexOf(self.roi_analysis_button) + 1, button)
        self.orientation_tools_button = button

    def _image_array(self, viewer):
        idx = self.ui.stackedWidget.indexOf(viewer)
        return imread(self.image_paths[idx], as_gray=True).astype(float)

    def auto_orient(self, scope="selected"):
        """Estimate ROI orientations from the image (intensity-weighted principal axis)."""
        from core.orientation import estimate_orientation
        viewers = self._viewers() if scope == "everything" else [self._current_viewer()]
        polar = self.orient_polar.isChecked()
        total = skipped = 0
        for viewer in viewers:
            if viewer is None:
                continue
            if scope == "selected":
                rois = viewer.selectedROIs()
            elif scope == "label":
                rois = [r for r in viewer.rois() if r.label == (self.get_current_label() or EMPTY_LABEL)]
            else:
                rois = viewer.rois()
            rois = [r for r in rois if r.roiType != "VectorROI"]
            if self.orient_only_unset.isChecked() and scope != "selected":
                rois = [r for r in rois if r.angle is None]
            if not rois:
                continue
            image = self._image_array(viewer)
            for roi in rois:
                angle = estimate_orientation(image, roi.outlinePoints(), closed=roi.closed, polar=polar,
                                             bright=not self.orient_dark.isChecked(),
                                             point_radius=4 * viewer._pointRadius)
                if angle is None:
                    skipped += 1
                    continue
                viewer.setROIAngle([roi], angle, emit=False)
                total += 1
            viewer.roisChanged.emit("modified", rois)
        self.updateBrowserText(f"Auto-oriented {total} ROI(s)" + (f", {skipped} without a clear feature" if skipped else "")
                               + (". Check the arrows; F flips a selected ROI." if polar else " (axis only)."))

    def convert_vectors_to_orientations(self):
        """
        Old workflow: a VectorROI drawn inside a box.  Every vector (except the symmetry vector)
        whose two end points lie inside an ROI gives that ROI its orientation, and is removed.
        """
        from core.orientation import vector_angle
        converted = left = 0
        for viewer in self._viewers():
            vectors = [v for v in viewer.rois("VectorROI") if v.label != SYMMETRY_LABEL]
            targets = [r for r in viewer.rois() if r.roiType != "VectorROI" and r.closed]
            done = []
            for vec in vectors:
                line = vec.line()
                p1, p2 = line.p1(), line.p2()
                inside = [r for r in targets
                          if r.path().contains(p1 - r.pos()) and r.path().contains(p2 - r.pos())]
                if not inside:
                    left += 1
                    continue
                box = min(inside, key=lambda r: r.area())
                viewer.setROIAngle([box], vector_angle(p1.x(), p1.y(), p2.x(), p2.y()), emit=False)
                done.append(vec)
                converted += 1
            if done:
                viewer.deleteROIs(done)
        self.updateBrowserText(f"Converted {converted} vector(s) into ROI orientations"
                               + (f"; {left} vector(s) were not inside an ROI and were kept." if left else "."))
        self._schedule_refresh()

    def clear_selected_orientations(self):
        viewer = self._current_viewer()
        if viewer is not None and viewer.selectedROIs():
            viewer.setROIAngle(viewer.selectedROIs(), None)

    def reference_angle(self, viewer=None):
        """Reference direction (degrees) for orientations: the symmetry vector of the image, else
        the last symmetry vector that was set, else None (image x axis)."""
        from core.orientation import vector_angle
        if viewer is not None:
            sym = [r for r in viewer.rois("VectorROI") if r.label == SYMMETRY_LABEL]
            if sym:
                line = sym[0].line()
                return vector_angle(line.x1(), line.y1(), line.x2(), line.y2())
        if self.symmetryVector and len(self.symmetryVector) == 2:
            dx, dy = self.symmetryVector
            if dx or dy:
                return math.degrees(math.atan2(dy, dx)) % 360.0
        return None

    def activate_polygon_mode(self):
        self._activate_mode('Polygon', "PolygonROI", self.ui.pushButton_13,
                            "Polygon mode: click to add points, double click or Enter to finish (>=3 points). "
                            "Right click a vertex to remove it, double click an edge to insert one.")

    def setScale(self):
        # scale_factor, ok = QInputDialog.getDouble(self, "Set pixel to nm scale for current image.")
        ok = False
        scale_factor = None
        scale_factor, ok = QInputDialog.getDouble(self, "Set pixel to nm scale for current image.", "Scale (nm/pixel):", 0.1, 0.001, 100, 2, step=0.001)

        if ok:
            self.current_viewer = self.ui.stackedWidget.currentWidget()
            self.current_viewer._pxScale = scale_factor
            self.updateBrowserText(f"Scale set to {scale_factor:.2f} nm/pixel")
            width_real = self.current_viewer._imgShape[0] * scale_factor
            height_real = self.current_viewer._imgShape[1] * scale_factor
            self.ui.label_5.setText(
                            f"Scale: {scale_factor:.2f} nm/pixel \n "
                            f"({width_real:.1f} x {height_real:.1f} nm)"
                        )
            
            metadata_file = os.path.join(self.data_dir, 'metadata.csv')
            file = os.path.basename(self.current_viewer._imgPath)
            channel = 'N/A'  # Placeholder for channel information
            writeHeader = False
            writeLine = False
            if not os.path.exists(metadata_file):
                writeHeader = True
                writeLine = True
            else:
                # Only append metadata if filename is not already present --- Need to see how to properly do this, should I overwrite existing entries or force the first to remain?
                with open(metadata_file, 'r') as f:
                    if file not in f.read():
                        writeLine = True
            if writeLine:
                with open(metadata_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                                    "Filename", "Channel", "Modifications", "Dimensions", "Pixel Scale X (nm)", "Pixel Scale Y (nm)"
                                    ])
                    if writeHeader:
                        writer.writeheader()
                    writer.writerow({
                        "Filename": file,
                        "Channel": channel,
                        "Modifications": '',
                        "Dimensions": f"{self.current_viewer._imgShape[0]} x {self.current_viewer._imgShape[1]}",
                        "Pixel Scale X (nm)": self.current_viewer._pxScale,
                        "Pixel Scale Y (nm)": self.current_viewer._pxScale
                        })

    
    def autoSetScale(self):
        """
        Automatically set the scale based on the metadata of the current image.
        """
        self.current_viewer = self.ui.stackedWidget.currentWidget()

        if self.current_viewer._pxScale is not None:
            self.ui.label_5.setText(
                f"Scale: {self.current_viewer._pxScale:.2f} nm/pixel \n "
                f"({self.current_viewer._imgShape[0] * self.current_viewer._pxScale:.1f} x {self.current_viewer._imgShape[1] * self.current_viewer._pxScale:.1f} nm)"
            )
        else:
            self.ui.label_5.setText("Scale: Please set manually.")
        

    def get_scan_dim(self):
        for viewer in self.ui.stackedWidget.findChildren(QtImageViewer):
            filename = os.path.basename(viewer._imgPath)
            if self.metadata_dict and 'Filename' in self.metadata_dict and 'Pixel Scale X (nm)' in self.metadata_dict:
                if filename in self.metadata_dict['Filename']:
                    idx = self.metadata_dict['Filename'].index(filename)
                    pixel_size_x = self.metadata_dict['Pixel Scale X (nm)'][idx]
                    pixel_size_y = self.metadata_dict['Pixel Scale Y (nm)'][idx]
                    viewer._pxScale = pixel_size_x
                else:
                    viewer._pxScale = None
            else:
                viewer._pxScale = None

    def label_ui_show(self):
        """
        Show the label input menu.
        """
        if not self.ui.widget_9.isVisible():
            self.ui.label_7.setText("Label Input Menu")
            self.ui.widget_9.show()
            self.label_input_dialog.show()
            # self.updateBrowserText("Label input dialog opened.")
        elif self.label_input_dialog.isVisible():
            self.ui.widget_9.hide()
            self.label_input_dialog.hide()
        elif self.autoAnnotations_dialog.isVisible():
            self.autoAnnotations_dialog.hide()
            self.ui.label_7.setText("Label Input Menu")
            self.label_input_dialog.show()

    def annotations_ui_show(self):
        """
        Show the annotations input menu.
        """
        if not self.ui.widget_9.isVisible():
            self.ui.label_7.setText("Auto Annotations Menu")
            self.ui.widget_9.show()
            self.autoAnnotations_dialog.show()
        elif self.autoAnnotations_dialog.isVisible():
            self.ui.widget_9.hide()
            self.autoAnnotations_dialog.hide()
        elif self.label_input_dialog.isVisible():
            self.label_input_dialog.hide()
            self.ui.label_7.setText("Auto Annotations Menu")
            self.autoAnnotations_dialog.show()

    def closeLabelInput(self):
        self.ui.widget_9.hide()
        self.label_input_dialog.hide()

    def closeAutoAnnotations(self):
        self.ui.widget_9.hide()
        self.autoAnnotations_dialog.hide()

    def predictions_ui_show(self):
        """
        Show the predictions input menu.
        """
        if not self.ui.widget_9.isVisible():
            self.ui.label_7.setText("Predictions Menu")
            self.ui.widget_9.show()
            self.predictions_dialog.show()
        elif self.predictions_dialog.isVisible():
            self.ui.widget_9.hide()
            self.predictions_dialog.hide()
        elif self.label_input_dialog.isVisible():
            self.label_input_dialog.hide()
            self.ui.label_7.setText("Predictions Menu")
            self.predictions_dialog.show()

    def closePredictions(self):
        self.ui.widget_9.hide()
        self.predictions_dialog.hide()

    def adaptImage_show(self):
        """
        Open the image adaptation dialog for the current image.

        Tune the mask pipeline (filters, threshold, morphology, size filter ...) and the
        ROI settings with a live preview of the distinct objects, and choose in the "Labels"
        tab which label each object gets (one label, threshold rules on shape measures,
        classes of one measure or automatic groups).  When "Apply" is pressed every object
        is added as a ROI with its assigned label (objects set to "Skip" are left out).
        The settings are remembered, so the dialog re-opens with the last pipeline.
        """
        from core.image_adapt_dialog import ImageAdaptDialog

        current_index = self.ui.stackedWidget.currentIndex()
        if not 0 <= current_index < len(self.image_paths):
            self.updateBrowserText("No image selected.")
            return
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        filename = os.path.basename(self.image_paths[current_index])
        image = imread(self.image_paths[current_index], as_gray=True)

        current = self.get_current_label()
        if not current or current == EMPTY_LABEL:
            current = self.labelList[0] if self.labelList else "object"
        dlg = ImageAdaptDialog(image, parent=self, params=self.adapted_image_params,
                               labels=self.labelList,
                               label_colors={l: label_color(l, self.labelList) for l in self.labelList},
                               px_scale=getattr(self.current_viewer, "_pxScale", None),
                               current_label=current)
        if not dlg.exec():          # Cancel / window closed
            return

        # Remember what was used (also feeds auto_segment_to_rois via self.roi_settings)
        self.adapted_image_params = dlg.get_params()
        self.roi_settings = dlg.get_roi_settings()

        if dlg.get_mask() is None:
            self.updateBrowserText(
                "Image adaptation has no active Threshold step, so there is no binary mask to create ROIs from."
            )
            return
        labeled = dlg.get_labeled_objects()
        if not labeled:
            self.updateBrowserText("No objects to add (none found, or all groups set to 'Skip').")
            return
        self._add_objects_as_rois([o for o, _ in labeled], filename, labels=[l for _, l in labeled])

    def _add_objects_as_rois(self, objects, filename, labels=None):
        """
        Add segmented objects (core.roi_extraction.SegObject list) to the current image as ROIs.
        ``labels`` gives one label per object; without it the currently selected label is used
        (first label / "object" as fallback).  Labels not yet in the label list are added to it.
        Returns the number of objects added.
        """
        if not objects:
            self.updateBrowserText("No objects found, no ROIs were added.")
            return 0

        if labels is None:
            label = self.get_current_label()
            if not label or label == EMPTY_LABEL:
                label = self.labelList[0] if self.labelList else "object"
            labels = [label] * len(objects)

        new_labels = self.label_manager.merge([l for l in dict.fromkeys(labels) if l and l != EMPTY_LABEL])
        if new_labels:
            self.updateBrowserText(f"New label(s) added to the label list: {', '.join(new_labels)}")

        viewer = self.current_viewer
        def coords(o):
            if o.roi_type == "RectROI":
                return o.rect
            if o.roi_type == "PointROI":
                return tuple(float(v) for v in o.points[0])
            return o.points

        rois = [viewer.createROI(o.roi_type, coords(o), label) for o, label in zip(objects, labels)]
        viewer.addROIs(rois)          # one batch, one change signal

        kind = objects[0].roi_type
        n_vertices = sum(o.n_vertices for o in objects)
        counts = Counter(labels)
        per_label = ", ".join(f"{l}: {n}" for l, n in counts.most_common())
        self.updateBrowserText(
            f"Added {len(objects)} {kind}s ({per_label}; {n_vertices} vertices in total)."
        )
        return len(objects)

    def label_ui_close(self):
        """Close the label panel (labels are applied as soon as they are edited)."""
        self.closeLabelInput()

    def load_annotations(self, prediction_results=False, classes=None, replace=True):
        """
        Load annotations from a CSV file (or from a prediction DataFrame) into the viewers.
        :param replace: clear the existing ROIs of every image that receives annotations.
        """
        if hasattr(self.ui, "radioButton"):
            self.ui.radioButton.setChecked(False)
        self.autoVector = False
        source = "predictions"
        if not isinstance(prediction_results, pd.DataFrame):
            if self.data_dir is None:
                self.data_dir = ""
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", self.data_dir, "CSV Files (*.csv)",)
            if not file_path:
                self.updateBrowserText("No annotation file selected.")
                return
            try:
                load_annos = pd.read_csv(file_path)
            except Exception as e:
                self.updateBrowserText(f"Failed to read {file_path}: {e}")
                return
            source = file_path
        else:
            load_annos = prediction_results

        missing = {'type', 'x1', 'y1', 'dx', 'dy', 'file'} - set(load_annos.columns)
        if missing:
            self.updateBrowserText(f"Annotation file is missing columns: {', '.join(sorted(missing))}")
            return

        # Add the labels of the loaded annotations to the label list (existing labels keep their
        # place, colour and number key)
        if classes is None:
            found = [str(l) for l in load_annos['label'].dropna().unique()] if 'label' in load_annos.columns else []
        else:
            found = [str(label) for label in classes.values()]
        found = sorted((l for l in found if l not in (EMPTY_LABEL, SYMMETRY_LABEL)),
                       key=lambda x: (not x[0].isdigit(), x.lower()))
        added = self.label_manager.merge(found)
        if added:
            self.updateBrowserText(f"Labels added from the annotations: {', '.join(added)}")

        # Group once by file instead of filtering the table for every image
        by_file = {name: group for name, group in load_annos.groupby('file', sort=False)}
        n_loaded = 0
        n_images = 0
        for idx, viewer in enumerate(self._viewers()):
            filename = os.path.basename(self.image_paths[idx])
            image_annotations = by_file.get(filename)
            if image_annotations is None or image_annotations.empty:
                continue
            if replace:
                viewer.clearROIs()
            n_loaded += len(viewer.loadROIs(image_annotations, None, "loaded"))
            n_images += 1

        # Restore the symmetry vector if one was saved
        for viewer in self._viewers():
            sym = [r for r in viewer.rois("VectorROI") if r.label == SYMMETRY_LABEL]
            if sym:
                line = sym[0].line()
                self.symmetryVector = [line.dx(), -line.dy()]
                self.updateBrowserText(f"Symmetry vector restored: dx={line.dx():.2f}, dy={-line.dy():.2f}")
                break

        self.updateBrowserText(f"Loaded {n_loaded} ROIs on {n_images} image(s) from {os.path.basename(str(source))}")
        self._refresh_roi_panels()

    def save_annotations(self):
        # Save annotations to a file
        if self.data_dir is None:
            self.data_dir = ""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", self.data_dir, "CSV Files (*.csv)")
        if not file_path:
            file_path = f"{self.data_dir}/annotations.csv"
        if file_path:
            # Implement saving logic here
            try:
                self.annotations_df.to_csv(file_path, index=False)
                self.updateBrowserText(f"Saved annotations to {file_path}")
            except Exception as e:
                self.updateBrowserText(f"Save Failed: {e} \n Make sure file is not opened.")

    def clearAnnotations(self):
        """
        Clear all annotations from the current image.
        """
        viewer = self._current_viewer()
        if viewer is None:
            return
        self.current_viewer = viewer
        viewer.clearROIs(emit=True)
        self.updateBrowserText("Cleared all annotations from the current image.")
        self._refresh_roi_panels()

    def onRoisChanged(self, action, rois):
        """
        Called once per ROI operation of a viewer (action: drawn, added, removed, modified,
        relabeled).  The annotation table is derived from the viewers, so only the side panels
        need refreshing (coalesced), plus the interactive extras for freshly drawn ROIs.
        """
        if action == "drawn" and rois:
            roi = rois[0]
            if self.symmetryVectorMode and roi.roiType == "VectorROI":
                self.setSymmetryVector(roi)
            elif self.autoVector and roi.roiType == "RectROI":
                self._auto_vector_for(roi)
        self._schedule_refresh()

    def _auto_vector_for(self, roi):
        """Detect the chirality of a freshly drawn 9AP box and add its orientation vector."""
        label = roi.label
        if label not in ('9AP (None)', '9AP (S)', '9AP (R)'):
            return
        viewer = roi._viewer
        current_index = self.ui.stackedWidget.indexOf(viewer)
        if not 0 <= current_index < len(self.image_paths):
            return
        rect = roi.rect()
        x1, y1 = int(rect.left()), int(rect.top())
        x2, y2 = int(rect.right()), int(rect.bottom())
        bbox = (y1, x1, y2, x2)
        filename = os.path.basename(self.image_paths[current_index])
        image = imread(self.image_paths[current_index], as_gray=True)
        img = (image - image.min()) / (image.max() - image.min())
        cropped_img = img[y1:y2, x1:x2]
        chirality, vector = self.detectChirality(cropped_img, pad=0, filename=filename, label=label, bbox=bbox, plot_results=False)
        if vector:
            # the detected direction becomes the box's own orientation (no separate VectorROI)
            from core.orientation import vector_angle
            _, _, vx1, vy1, vx2, vy2, _, _ = vector[0]
            viewer.setROIAngle([roi], vector_angle(vx1, vy1, vx2, vy2))

    def labelTextDisplay(self, filename=None):
        """Label counts and average line segment length of the current image (from its ROIs)."""
        viewer = self._current_viewer()
        if viewer is None:
            self.ui.textBrowser_2.setText("No ROIs or labels assigned yet.")
            return
        text_lines = []
        rois = viewer.rois()
        label_counts = Counter(r.label for r in rois if r.label and r.label != EMPTY_LABEL)
        if label_counts:
            text_lines.append("Label counts:")
            text_lines += [f"{label}: {count}" for label, count in label_counts.most_common()]
        if len(rois) and len(label_counts) == 0:
            text_lines.append(f"{len(rois)} unlabeled ROIs")

        segments = [r for r in rois if r.roiType == "LineSegmentROI"]
        if segments and viewer._pxScale is not None:
            lengths = [float(np.hypot(*np.diff(r.pointsArray(), axis=0).T).sum()) for r in segments]
            avgLength = float(np.mean(lengths)) * viewer._pxScale
            text_lines.append(f"Avg GNR length: {avgLength:.2f} nm") # Should be better abstracted

        self.ui.textBrowser_2.setText('\n'.join(text_lines) if text_lines else "No ROIs or labels assigned yet.")

    def get_current_label(self):
        """Get the currently selected label from listWidget_2"""
        selected_label = self.ui.listWidget_2.selectedItems()
        if selected_label:
            # Extract just the label text (remove the "index: " prefix)
            full_text = selected_label[0].text()
            if ": " in full_text:
                return full_text.split(": ", 1)[1]
            return full_text
        return None

    def on_label_selection_changed(self):
        """Handle when label selection changes in listWidget_2: new ROIs get this label, and
        the ROIs selected in the viewer are relabelled."""
        self.current_label = self.get_current_label()
        viewer = self._current_viewer()
        if viewer is None:
            return
        self.current_viewer = viewer
        viewer._currentLabel = self.current_label or EMPTY_LABEL
        if self._syncing_label:
            return
        selected = viewer.selectedROIs()
        if selected and self.current_label is not None:
            self.labelUpdate(selected, self.current_label)

    def labelUpdate(self, rois, label):
        """
        Assign a label to ROIs (single ROI or list) of the current image.
        """
        viewer = self._current_viewer()
        if viewer is None:
            self.updateBrowserText("No image selected.")
            return
        if not isinstance(rois, (list, tuple)):
            rois = [rois]
        if not rois or not label:
            self.updateBrowserText("No matching annotation found for labeling.")
            return
        changed = viewer.setROILabel(rois, label)
        if len(changed) > 1:
            self.updateBrowserText(f"Relabeled {len(changed)} ROIs as '{label}'.")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for label selection"""
        key = event.key()
        modifiers = event.modifiers()

        # Handle number keys (0-9) for label selection or mode switching with Alt
        if QtCore.Qt.Key.Key_0 <= key <= QtCore.Qt.Key.Key_9:
            index = key - QtCore.Qt.Key.Key_0
            if modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
                modes = {
                    1: self.activate_selection_mode,
                    2: self.activate_zoom_mode,
                    3: self.activate_bounding_box_mode,
                    4: self.activate_vector_mode,
                    5: self.activate_point_mode,
                    6: self.activate_line_segment_mode,
                    7: self.activate_polygon_mode,
                    8: self.activate_orientation_mode,
                }
                if index in modes:
                    modes[index]()
                return
            else:
                if index < self.ui.listWidget_2.count():
                    self.ui.listWidget_2.setCurrentRow(index)
                    self.on_label_selection_changed()
                    return

        # Handle Escape key to deselect ROIs
        if key == QtCore.Qt.Key.Key_Escape:
            viewer = self._current_viewer()
            if viewer is not None and viewer.selectedROIs():
                viewer.clearSelection()
                self.updateBrowserText("ROI deselected.")
                return

        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            if key == QtCore.Qt.Key.Key_S:
                self.save_annotations()
        # Call parent's keyPressEvent for other keys
        super().keyPressEvent(event)

    def getLabelExamples(self):
        """
        Get examples of labels from the label input dialog, add cropped images to widget.
        Only show labels that have actual bounding boxes in the current image.
        """
        # Clear all widgets from the grid layout
        for i in reversed(range(self.label_editor.preview_grid.count())):
            self.label_editor.preview_grid.itemAt(i).widget().setParent(None)
        if not self.label_editor.previews.isChecked():
            return              # previews section collapsed: nothing to show

        viewer = self._current_viewer()
        if viewer is None or not viewer.hasImage():
            return
        pixmap = viewer.pixmap()

        # First bounding box of every (non-empty) label
        first_box = {}
        for roi in viewer.rois("RectROI"):
            if roi.label and roi.label != EMPTY_LABEL and roi.label not in first_box:
                first_box[roi.label] = roi

        for row, (label, roi) in enumerate(first_box.items()):
            cropped = pixmap.copy(roi.rect().toRect())
            label_text_widget = QLabel(f"{label}")
            label_text_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            image_widget = QLabel()
            image_widget.setPixmap(cropped.scaled(100, 100, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            # 2 columns: image, label
            self.label_editor.preview_grid.addWidget(image_widget, row, 0)
            self.label_editor.preview_grid.addWidget(label_text_widget, row, 1)
        if self.labelList and set(self.labelList).issubset(set(first_box)):
            self.labels_ready = True

    def saveLabelExamples(self):
        """
        Save cropped images and parameter csv file for future loading.
        """
        current_index = self.ui.stackedWidget.currentIndex()
        if 0 <= current_index < len(self.image_paths):
            image_path = self.image_paths[current_index]
            filename = os.path.basename(image_path)
            label_annos = self.annotations_df
            label_annos['scale'] = self.current_viewer._pxScale
            save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Label Examples")
            if save_dir:
                # Save cropped images
                for i in range(self.label_editor.preview_grid.rowCount()):
                    try:
                        label_widget = self.label_editor.preview_grid.itemAtPosition(i, 1).widget()
                        image_widget = self.label_editor.preview_grid.itemAtPosition(i, 0).widget()
                        label_text = label_widget.text()
                        pixmap = image_widget.pixmap()
                        if pixmap:
                            cropped_image_path = os.path.join(save_dir, f"{filename}_{label_text}.png")
                            pixmap.save(cropped_image_path)
                            print(f"Saved cropped image for label '{label_text}' at {cropped_image_path}")
                    except Exception as e:
                        print(f"Error saving cropped image for label '{label_text}': {e}")
                # Save parameters to CSV
                params_file_path = os.path.join(save_dir, "object_parameters.csv")
                label_annos.to_csv(params_file_path, index=False)
                print(f"Saved parameters to {params_file_path}")
                self.updateBrowserText(f"Label examples saved to {save_dir}")
    
    def clearActivityLog(self):
        """
        Clear the activity log in the text browser.
        """
        self.browserText = []
        self.ui.textBrowser.setText("")
        
    def runAutoSegment(self):
        """
        Run the first pass segmentation algorithm on the current image.
        """
        # self.autoVector = False
        feature_properties, num_bins, thresholds = self.load_feature_parameters()
        if self.labelList is None or len(self.labelList) == 0:
            self.ui.label_2.setText("No labels set. Please set labels before running auto-segmentation.")
            return
        
        if self.autoAnnotations_ui.checkBox_3.isChecked():
            vacancy_detection = True
        else: 
            vacancy_detection = False
        # if "vacancy" in [label.lower() for label in self.labelList] or \
        #     "vacancies" in [label.lower() for label in self.labelList]:
        #     vacancy_detection = True
        # self.ui.radioButton.setChecked(False) # Disable autovector 
        self.autoVector = False
        current_index = self.ui.stackedWidget.currentIndex()
        self.activate_bounding_box_mode()
        self.ui.label_2.setText("Running auto-segmentation... Please wait.")
        self.updateBrowserText(f"Auto-segmentation started for {os.path.basename(self.image_paths[current_index])}.")

        gamma = None
        threshold = None
        plot_results = False
        sig = None
        if self.autoAnnotations_ui.checkBox.isChecked():
            gamma = self.autoAnnotations_ui.doubleSpinBox.value()
        if self.autoAnnotations_ui.checkBox_2.isChecked():
            threshold = self.autoAnnotations_ui.doubleSpinBox_2.value()
        if self.autoAnnotations_ui.checkBox_4.isChecked():
            sig = self.autoAnnotations_ui.doubleSpinBox_3.value()
        if self.ui.checkBox.isChecked():
            plot_results = True
        else:
            plot_results = False
        if 0 <= current_index < len(self.image_paths):
            filename = os.path.basename(self.image_paths[current_index])
            # self.annotations_df = self.annotations_df[self.annotations_df['file'] != filename]
            self.current_viewer = self.ui.stackedWidget.currentWidget()
            image = imread(self.image_paths[current_index], as_gray=True)
            bin_indices, props, vacancies, img_laplace = run_first_pass_segment(image, 
                                   vacancy_detection=vacancy_detection, 
                                   feature_properties=feature_properties, 
                                   num_bins=num_bins, 
                                   thresholds=thresholds,
                                   gamma=gamma,
                                   threshold=threshold, sigma=sig,
                                   plot_results=plot_results)
            # return
            # If there are any names from load_labels not already in labelList, add them
            load_labels = bin_indices.keys()
            if load_labels is not None:
                for label_name in load_labels:
                    if label_name == 'Empty Label':
                        continue
                    if label_name == "9AP": # Need to ultimately abstract this to work for any chiral label
                        continue
                    if label_name not in self.label_manager:
                        self.label_manager.add(label_name)
            pad = 2
            vectorList = []
            roi_df = pd.DataFrame(columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])
            for label_key, label_idx in bin_indices.items():
                for i in label_idx:
                    label = label_key
                    if label.lower() == 'empty label':
                        continue
                    region = props[i]
                    # x1, y1, x2, y2 = region.bbox
                    y1, x1, y2, x2 = region.bbox  # Adjusted for (y, x) order
                    dx = x2 - x1
                    dy = y2 - y1
                    if label == '9AP' or label == '9AP (none)':
                        if label == '9AP (none)':
                            label = '9AP'
                        cropped_img = img_laplace[y1-pad:y2+pad, x1-pad:x2+pad]
                        chirality, vectorData = self.detectChirality(cropped_img, pad, filename, label, region.bbox, plot_results=False)
                        # print("9AP detected, running second pass segmentation.")
                        label = f"{label} ({chirality})"
                        roi_df = pd.concat([roi_df, pd.DataFrame(vectorData, columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])], ignore_index=True)
                    roiData = [[None, 'RectROI', x1 - pad, y1 - pad, dx + 2 * pad, dy + 2 * pad, filename, label]]
                    roi_df = pd.concat([roi_df, pd.DataFrame(roiData, columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])], ignore_index=True)
                    self.current_label = label
                    self.current_viewer._currentLabel = label
                    # self.current_viewer.loadROIs(roi_df, os.path.basename(self.image_paths[current_index]), "added")
                    # auto_anno_df = pd.concat([auto_anno_df, new_df], ignore_index=True)
            if vacancy_detection:
                for vacancy in vacancies:
                    y, x = vacancy
                    roiData = [[None, 'PointROI', x, y, 0, 0, os.path.basename(self.image_paths[current_index]), 'vacancy']]
                    roi_df = pd.concat([roi_df, pd.DataFrame(roiData, columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])], ignore_index=True)
                    self.current_label = 'vacancy'
                    self.current_viewer._currentLabel = 'vacancy'
                    # self.current_viewer.loadROIs(roi_df, os.path.basename(self.image_paths[current_index]), "added")
            
            self.current_viewer.loadROIs(roi_df, os.path.basename(self.image_paths[current_index]), "added")
            # self.autoVector = True
            self.ui.label_2.setText("Auto-segmentation completed and bounding boxes drawn.")
            self.updateBrowserText("Auto-segmentation Ran.")
        else:
            self.updateBrowserText("No image selected for auto-segmentation.")
     
    def load_feature_parameters(self):
        """
        Load feature parameters from a CSV file.
        """
        # stm_data_directory = r'C:\Users\wrja\Desktop\STM_Data\\'  # Directory where images are stored
        # csv_path = stm_data_directory + 'stm_labeled_annotations.csv'
        # Find the object_parameters file in user_data directory
        user_data_dir = os.getcwd() + r"\Data"
        feature_parameter_dir = None
        csv_path = None
        # Search for a file containing 'object_parameters' in its name
        for root, dirs, files in os.walk(user_data_dir):
            for file in files:
                if "object_parameters" in file and file.endswith(".csv"):
                    feature_parameter_dir = root
                    csv_path = os.path.join(root, file)
                    break
            if csv_path:
                break

        print(f"Searching for parameter file... Found: {csv_path}")
        if not pd.io.common.file_exists(csv_path):
            print(f"CSV file not found at {csv_path}. Please ensure the file exists.")
            return None

        df = pd.read_csv(csv_path)
        # If any label value is '9AP (None)', change it to '9AP'
        try:
            df['label'] = df['label'].replace('9AP (None)', '9AP')
        except Exception as e:
            print(f"Error occurred while replacing labels: {e}")
        dx = df['dx'].tolist()
        dy = df['dy'].tolist()
        
        px_areas = [dx[i] * dy[i] for i in range(len(dx))]
        scale = df['scale'].iloc[0] if 'scale' in df.columns else 1.0  # Default scale if not present
        areas_real = [area * (scale ** 2) for area in px_areas] # Convert pixel area to real units
        # # Convert area back to pixel units using the currentPxSize from the viewer
        # if hasattr(self, 'currentPxSize') and self.current_viewer._pxScale and self.current_viewer._pxScale != 0:
        #     areas = [area / (self.current_viewer._pxScale ** 2) for area in areas]
        # else:
        #     areas = [area for area in px_areas]  # fallback to pixel area if scale not available
        aspect_ratios = [dx[i] / dy[i] if dy[i] != 0 else 0 for i in range(len(dy))]
        labels = df['label'].tolist()
        print("Labels:", labels)
        print("Scale:", scale)
        print("Areas (real):", areas_real)
        print("Aspect Ratios:", aspect_ratios)

        feature_properties = {
            label: {
                'area': areas_real[i],
                'area_px': px_areas[i],
                'aspect_ratio': aspect_ratios[i],
            }
            for i, label in enumerate(labels)
        }
        # print(f"Feature properties: {feature_properties}")
        # print(f"Feature names: {feature_properties.keys()}")
        # print(f"Feature sizes: {[feature_properties[name]['area'] for name in feature_names]}")
        num_bins = len(labels)  # Number of bins for histogram
        min_feature_size = min(areas_real) #* 0.1  # Minimum size for removing small objects
        background_thresh = 0.5  # Threshold for background removal
        min_maxima_distance = int(min(areas_real))# * 0.1)  # Minimum distance between local maxima
        
        thresholds = {
            'min_feature_size': min_feature_size,
            'background_thresh': background_thresh,
            'min_maxima_distance': min_maxima_distance
        }
        # print(thresholds)
        return feature_properties, num_bins, thresholds

    def collectSymmetryVector(self):
        self.activate_vector_mode()
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.symmetryVectorMode = True

        self.ui.label_2.setText("Click on a drawn vector to set the high symmetry axis")
        self.updateBrowserText("Symmetry vector mode activated. Click on a drawn vector to set the high symmetry axis.")

    def setSymmetryVector(self, roi):
        """
        Use a VectorROI as the high symmetry axis: it is relabelled "Symmetry Vector" (any
        previous symmetry vector of that image goes back to "Empty Label").
        :param roi: the VectorROI.
        """
        self.symmetryVectorMode = False
        if roi is None or roi.roiType != "VectorROI":
            self.updateBrowserText("Selected ROI is not a vector." if roi is not None else "No vector ROI selected.")
            return
        line = roi.line()
        dx, dy = line.dx(), -line.dy()
        self.symmetryVector = [dx, dy]
        viewer = roi._viewer
        previous = [r for r in viewer.rois("VectorROI") if r.label == SYMMETRY_LABEL and r is not roi]
        if previous:
            viewer.setROILabel(previous, EMPTY_LABEL)
        viewer.setROILabel([roi], SYMMETRY_LABEL)
        self.updateBrowserText(f"Symmetry vector set: dx={dx:.2f}, dy={dy:.2f}")
        slope = f"{dy / dx:.4f}" if dx != 0 else "inf"
        self.ui.label_2.setText(f"Symmetry vector set: Slope = {slope}")

    def correctVector(self):
        print("Correcting symmetry vector...")

    def detectChirality(self, cropped_img, pad, filename, label, bbox, plot_results=False):
        """
        Detect chirality of the cropped image.
        :param cropped_img: The cropped image to analyze.
        :return: 'None', 'S' or 'R' based on chirality detection.
        """

        chirality, vector = run_second_pass_segment(cropped_img, plot_results=plot_results)
        # except Exception as e:
        #     print(f"Error in second pass segmentation for {label_key}: {e}")
        #     chirality = 'None'
        #     vector = None
        #     continue  # Skip to next label
        vectorData = []
        # print(cropped_img[0], cropped_img[1], cropped_img[2], cropped_img[3])
        if vector is not None:
            y1, x1, y2, x2 = bbox
            chiral_x0 = vector[0] + x1 - pad
            chiral_y0 = vector[1] + y1 - pad
            chiral_x1 = vector[2] + x1 - pad
            chiral_y1 = vector[3] + y1 - pad
            vectorData = [[None, 'VectorROI', chiral_x0, chiral_y0, chiral_x1, chiral_y1, filename, 'Empty Label']]
            # roi_df = pd.concat([roi_df, pd.DataFrame(vectorData, columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])], ignore_index=True)
        
        return chirality, vectorData
    
    def toggleAutoVector(self):
        if self.ui.radioButton.isChecked():
            self.autoVector = True
            self.ui.label_2.setText("Auto vector mode enabled. Draw 9AP boxes to auto-detect chirality and add vectors.")
            self.updateBrowserText("Auto vector mode enabled. Draw 9AP boxes to auto-detect chirality and add vectors.")
        else:
            self.autoVector = False
            self.ui.label_2.setText("Auto vector mode disabled.")
            self.updateBrowserText("Auto vector mode disabled.")

    def get_vector_angles_within_9AP(self): # Rename to get_results
        """
        Find all VectorROIs (vectors) within 9AP bounding boxes and calculate their angles with respect to the symmetryVector.
        Returns:
            List of tuples: [(roi, angle_deg), ...]
        """
        print("Producing angle histogram...")
        if not self.symmetryVector or len(self.symmetryVector) != 2:
            self.updateBrowserText("Symmetry vector not set.")
            self.ui.label_2.setText("Symmetry vector not set. Please set it before calculating angles.")
            return []

        current_index = self.ui.stackedWidget.currentIndex()
        if 0 > current_index or current_index >= len(self.image_paths):
            self.updateBrowserText("No image selected.")
            return []
        

        results = []
        coverage = []
        molec_count = []
        molec_density = []
        vacancy_count = []
        vac_density = []
        angle_dict = {'S': [], 'R': [], 'None': []}
        i = 0
        for filename in self.annotations_df['file'].unique():
            print(f"Compiling from file: {filename}")
            # filename = os.path.basename(self.image_paths[current_index])
            
            # Get all 9AP bounding boxes
            bbox_df = self.annotations_df[
                (self.annotations_df['file'] == filename) &
                (self.annotations_df['type'] == 'RectROI') &
                (self.annotations_df['label'].str.contains('9AP', case=False, na=False))
            ]

            # Get all vectors (VectorROI)
            vector_df = self.annotations_df[
                (self.annotations_df['file'] == filename) &
                (self.annotations_df['type'] == 'VectorROI')
            ]

            points_df = self.annotations_df[
                (self.annotations_df['file'] == filename) &
                (self.annotations_df['type'] == 'PointROI') &
                (self.annotations_df['label'].str.contains('vacancy', case=False, na=False))
            ]
            
            # Get the dimensions of the scan for coverage calculation

            pix_x_scale = self.current_viewer._pxScale 
            scan_width = self.current_viewer._imgShape[1] * pix_x_scale if self.current_viewer._imgShape is not None else None
            scan_height = self.current_viewer._imgShape[0] * pix_x_scale if self.current_viewer._imgShape is not None else None

            if scan_width is None or scan_height is None:
                self.updateBrowserText(f"Could not determine scan dimensions for {filename}. Skipping coverage calculation.")
                continue
            scan_area = scan_width * scan_height
            feature_properties, num_bins, thresholds = self.load_feature_parameters()
            molec_area = feature_properties['9AP']['area'] # Need to abstract by calling the desired molecules label
            total_molec_area = len(bbox_df) * molec_area
            coverage.append(total_molec_area / scan_area if scan_area > 0 else 0)
            molec_count.append(len(bbox_df))
            molec_density.append(len(bbox_df) / scan_area if scan_area > 0 else 0)

            if len(points_df) > 0:
                vacancy_count.append(len(points_df))
                vac_density.append(len(points_df) / scan_area if scan_area > 0 else 0)

            # Prepare to collect angles by enantiomer
            for _, bbox in bbox_df.iterrows():
                x1, y1, dx, dy = bbox['x1'], bbox['y1'], bbox['dx'], bbox['dy']
                x2, y2 = x1 + dx, y1 + dy
                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])
                # Determine enantiomer type from label
                label = str(bbox['label'])
                if '(S)' in label:
                    enantiomer = 'S'
                elif '(R)' in label:
                    enantiomer = 'R'
                else:
                    enantiomer = 'None'
                # The box's own orientation (set in orientation mode / auto-orient) takes precedence
                own = bbox.get('angle') if 'angle' in bbox.index else None
                if own is not None and np.isfinite(own):
                    sym_vec = np.array(self.symmetryVector)
                    diff = np.radians(own) - np.arctan2(sym_vec[1], sym_vec[0])
                    angle_deg = (360 - np.degrees(diff)) % 360
                    results.append((bbox, angle_deg, enantiomer))
                    angle_dict[enantiomer].append(angle_deg)
                    continue
                # Otherwise: find vectors inside this bounding box (older annotations)
                for _, vec in vector_df.iterrows():
                    vx1, vy1, vx2, vy2 = vec['x1'], vec['y1'], vec['dx'], vec['dy']
                    # Check if both endpoints are inside the bounding box
                    if (xmin <= vx1 <= xmax and ymin <= vy1 <= ymax and xmin <= vx2 <= xmax and ymin <= vy2 <= ymax):
                        # Vector direction
                        v = np.array([vx2 - vx1, -(vy2 - vy1)]) #Temp fix inverting y 
                        sym_vec = np.array(self.symmetryVector)
                        # Calculate angle in degrees (0 to 360)
                        norm_v = np.linalg.norm(v)
                        norm_sym = np.linalg.norm(sym_vec)
                        if norm_v == 0 or norm_sym == 0:
                            angle_deg = None
                        else:
                            # print(f"Vector {i}: v={v}, sym_vec={sym_vec}")
                            # angle_rad = np.arctan2(np.cross(sym_vec, v), np.dot(sym_vec, v))
                            # angle_deg = np.degrees(angle_rad) % 360 
                            angle1 = np.arctan2(v[1], v[0])
                            angle2 = np.arctan2(sym_vec[1], sym_vec[0])
                            diff = angle1 - angle2
                            angle_deg = (360 - np.degrees(diff)) % 360
                            
                            results.append((vec, angle_deg, enantiomer))
                        if angle_deg is not None:
                            angle_dict[enantiomer].append(angle_deg)
                        break  # Assume one vector per bbox for this analysis

        self.updateBrowserText(f"Found {len(results)} total vectors within 9AP bounding boxes.")
        total_R = len(angle_dict['R'])
        total_S = len(angle_dict['S'])
        try:
            ee = (total_S - total_R) / (total_S + total_R)
        except ZeroDivisionError:
            ee = 0
        coverage = np.mean(coverage) * 100  # Convert to percentage
        molec_avg = np.mean(molec_count)
        unknown_molec = np.sum(molec_count) - (total_S + total_R)
        unknown_percent = (unknown_molec / np.sum(molec_count)) * 100 if np.sum(molec_count) > 0 else 0
        print(f"Unknown 9AP: {unknown_percent:.2f}")

        std = np.std(molec_count)
        print(f"9AP count std dev: {std:.2f}")

        vac_avg = np.mean(vacancy_count) if len(vacancy_count) > 0 else 0
        vac_density_avg = np.mean(vac_density) if len(vac_density) > 0 else 0
        vac_density_std = np.std(vac_density) if len(vac_density) > 0 else 0
        molec_density_avg = np.mean(molec_density) if len(molec_density) > 0 else 0
        molec_density_std = np.std(molec_density) if len(molec_density) > 0 else 0
        vac_norm = vac_density_avg / molec_density_avg if molec_density_avg > 0 else 0
        print(f"Average vacancies per image: {vac_avg:.2f}")
        print(f"Average vacancy density: {vac_density_avg:.2e} vacancies/nm^2")
        print(f"Vacancy density std dev: {vac_density_std:.2e}")
        print(f"Average molecule density: {molec_density_avg:.2e} molecules/nm^2")
        print(f"Molecule density std dev: {molec_density_std:.2e}")
        print(f"Vacancies per 50 nm scan: {vac_density_avg * 2500:.2f}")
        print(f"Molecules per 50 nm scan: {molec_density_avg * 2500:.2f}")
        print(f"Vacancy density normalized by molecule coverage: {vac_norm:.2f}")

        angle_bin_N, ok = QInputDialog.getInt(
            self, "Label Histogram", "How many bins?:", value=18, min=1, max=48
        )
        if not ok:
            return
        
        # Plot histogram by enantiomer
        if results:
            bins = np.linspace(0, 360, angle_bin_N)
            plt.figure(figsize=(7, 5))
            # plt.hist(
            # [angle_dict['S'], angle_dict['R']],
            # bins=bins,
            # stacked=True,
            # color=[ '#56B4E9','#CC79A7',],
            # label=['(S)', '(R)']
            # )
            
            plt.hist(angle_dict['S'], bins=bins, alpha =0.7, label='(S)', color='#56B4E9')
            plt.hist(angle_dict['R'], bins=bins, alpha =0.7, label='(R)', color='#CC79A7')

            total_entropy = 0
            # Compute shannon entropy 
            for enantiomer, data in angle_dict.items():
                if enantiomer == 'None':
                    continue

                hist, bin_edges = np.histogram(data, bins=bins)
                # Normalize to get probabilities
                p = hist / hist.sum()
                # Compute Shannon entropy (base 2)
                entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
                normalized_entropy = entropy / np.log2(len(hist))  # Normalize by max entropy

                total_entropy += normalized_entropy
                print(f"Shannon entropy for {enantiomer}: {normalized_entropy:.4f}")
            
            print(f"Avg Shannon entropy: {total_entropy/2:.4f}")
            
            self.updateBrowserText(f"Analyzed Folder: {self.data_dir} ")
            self.updateBrowserText(f"Detected 9AP count: {np.sum(molec_count)}, Avg molecules per image: {int(molec_avg)}, Avg Coverage: {coverage:.2f}%")
            self.updateBrowserText(f"Unknown 9AP: {unknown_percent:.2f}%, Std Dev: {std:.2f}")
            self.updateBrowserText(f"Shannon Entropy (avg): {total_entropy/2:.4f}")
            self.updateBrowserText(f"Enantiomeric Excess (EE): {ee:.2f} (S: {total_S}, R: {total_R})")
            
            
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Count')
            # plt.title(f'Molecule Orientations w.r.t.{self.symmetryVector[1] / self.symmetryVector[0]:.4f}') #for {os.path.basename(self.image_paths[current_index])}.')
            data_dir_parts = self.data_dir.replace('\\', '/').split('/')
            data_dir_parent = data_dir_parts[-2] if len(data_dir_parts) >= 2 else self.data_dir
            plt.title(f'Orientations w.r.t. [-101] for {data_dir_parent}, {os.path.basename(self.image_paths[current_index])}.')
            plt.legend()
            plt.tight_layout()
            # Set x-axis ticks to integer values only
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.show()
            plt.savefig(f"{os.path.basename(self.image_paths[current_index])}_vectors_hist.png")
        self.ui.label_2.setText(f"Found {len(results)} vectors within bounding boxes. See histogram for angle distribution.")
        return results

    def runGNRSegmentation(self):
        print("Characterizing GNRs...")
        if self.ui.checkBox.isChecked():
            plot_results = True
        else:
            plot_results = False
        self.ui.label_2.setText("GNR characterization running.")
        from core.GNR_segment import run_GNR_segment
        current_index = self.ui.stackedWidget.currentIndex()
        self.clearAnnotations()

        if 0 <= current_index < len(self.image_paths):
            filename = os.path.basename(self.image_paths[current_index])
            # self.annotations_df = self.annotations_df[self.annotations_df['file'] != filename]
            self.current_viewer = self.ui.stackedWidget.currentWidget()
            image = imread(self.image_paths[current_index], as_gray=True)

            # Get the dimensions of the scan for coverage calculation
            self.get_scan_dim()
            if self.current_viewer._imgShape[0] is None or self.current_viewer._imgShape[1] is None:
                self.updateBrowserText(f"Could not determine scan dimensions for {filename}. Ignore GNR length evaluation.")

            gnr_segments, gnr_binary, gnr_skeleton, gnr_distance_map, gnr_num_segments, gnr_avg_length, segment_endpoints = run_GNR_segment(image, px_scale=self.current_viewer._pxScale , gamma=None, threshold=None, sigma=None, block_size=11, num_bins=None, plot_results=plot_results)

        return gnr_binary
        # roi_df = pd.DataFrame(columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])

        # for segment in segment_endpoints:
        #     print(f"Adding GNR segment: {segment}")
        #     roiData = []
        #     x1, y1, dx, dy = segment
        #     roiData = [[None, 'LineSegmentROI', x1, y1, dx, dy, filename, self.current_label]]
        #     roi_df = pd.concat([roi_df, pd.DataFrame(roiData, columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])], ignore_index=True)

        # self.current_viewer.loadROIs(roi_df, os.path.basename(self.image_paths[current_index]), "added")
        # self.ui.label_2.setText(f"Number of GNR segments found: {gnr_num_segments}")
        # self.updateBrowserText(f"GNR segmentation found {gnr_num_segments} segments.")
        # self.updateBrowserText(f"Average GNR length: {gnr_avg_length:.2f} nm.")

    def computeLineSegLength(self, segment_roi):
        """
        Stats of one line segment given as its annotation rows (DataFrame with x1, y1 columns).
        Returns (length_nm or None, length_px, r_squared, linearity).
        """
        if segment_roi.empty or segment_roi.iloc[0]['type'] != 'LineSegmentROI':
            return None, 0.0, None, None
        px_scale = self.current_viewer._pxScale if self.current_viewer is not None else None
        return polyline_stats(segment_roi[['x1', 'y1']].to_numpy(dtype=float), px_scale)

    def convert_all_lines_to_polygons(self):
        """
        Converts all LineSegmentROI items in the current image to PolygonROIs
        based on a user-prompted thickness and end buffer.
        """
        viewer = self._current_viewer()
        if viewer is None:
            self.updateBrowserText("No image selected.")
            return
        self.current_viewer = viewer

        lines = viewer.rois("LineSegmentROI")
        if not lines:
            self.updateBrowserText("No line segment ROIs found in the current image.")
            return

        # Prompt for thickness and end buffer
        thickness_nm, ok = QInputDialog.getDouble(
            self, "Convert All Line Segments to Polygons", "Enter polygon thickness (nm):", value=2.0, min=0.1, max=10.0, decimals=2
        )
        if not ok:
            return

        end_buffer_nm, ok2 = QInputDialog.getDouble(
            self, "Convert All Line Segments to Polygons", "Enter end buffer length (nm):", value=0.5, min=0.0, max=5.0, decimals=2
        )
        if not ok2:
            return

        # ROI coordinates are pixels; _pxScale is the physical size of one pixel in nm.
        px_scale = getattr(viewer, '_pxScale', None)
        if px_scale is None or px_scale <= 0:
            self.updateBrowserText("Cannot convert nm to pixels: pixel scale is unavailable.")
            return
        half_t = thickness_nm / px_scale / 2.0
        end_buffer = end_buffer_nm / px_scale

        converted, polygons = [], []
        for roi in lines:
            pts = roi.pointsArray()
            if len(pts) < 2:
                continue
            polygons.append(viewer.createROI("PolygonROI", thicken_polyline(pts, half_t, end_buffer), roi.label))
            converted.append(roi)

        viewer.deleteROIs(converted)
        viewer.addROIs(polygons)
        self.updateBrowserText(f"Successfully converted {len(converted)} line segment(s) to PolygonROIs with thickness {thickness_nm} nm and buffer {end_buffer_nm} nm.")

    def plotLineSegResults(self):
        filename = os.path.basename(self.image_paths[self.ui.stackedWidget.currentIndex()])

        if 'LineSegmentROI' in self.annotations_df['type'].values:
            line_segments = self.annotations_df[
                (self.annotations_df['file'] == filename) &
                (self.annotations_df['type'] == 'LineSegmentROI')
            ]
            lengthList = []
            r_squared_values = []
            totalSegLength = 0.0
            lengthSTD = 0.0
            for _, segment in line_segments.groupby('objectName', sort=False):
                length_nm, segLength, r_squared, linearity = self.computeLineSegLength(segment)
                if length_nm is not None:
                    totalSegLength += length_nm
                    lengthList.append(length_nm)
                if r_squared is not None:
                    r_squared_values.append(r_squared)
            avgLength = totalSegLength / len(line_segments['objectName'].unique()) if len(line_segments['objectName'].unique()) > 0 else 0
  
            if len(lengthList) > 1:
                lengthSTD = float(np.std(lengthList))

            countBins, ok = QInputDialog.getInt(
                        self, "Length Histogram", "How many bins?:", value=10, min=1, max=48
                    )
            if not ok:
                return

            gnrWidth, ok = QInputDialog.getDouble(
                self, "GNR Coverage", "GNR width (nm):", value=2.0, min=0.1, max=1000, decimals=1
            )
            if not ok:
                return
            gnrArea = gnrWidth * totalSegLength 
            imgArea = self.current_viewer._imgShape[0] * self.current_viewer._pxScale * self.current_viewer._imgShape[1] * self.current_viewer._pxScale if self.current_viewer._pxScale else 0
            gnrCoverage = gnrArea / imgArea if imgArea > 0 else 0
            self.updateBrowserText(f"GNR area: {gnrArea:.2f} nm², Coverage: {gnrCoverage:.2f}")

            if len(lengthList) > 0:
                plt.figure(figsize=(8, 5))
                plt.hist(lengthList, bins=countBins, color='steelblue', edgecolor='black', alpha=0.85, density=True, stacked=True)
                plt.xlabel('R [nm]')
                plt.ylabel('Density [1/nm]')
                plt.title(f'Length Distribution - {filename} \n Avg Length: {avgLength:.2f} ± {lengthSTD:.2f} nm')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.show()
                self.updateBrowserText(
                    f"Plotted length histogram. N={len(lengthList)}, Avg={avgLength:.2f} ± {lengthSTD:.2f} nm."
                )
                self.saveResults(filename, avgLength, lengthSTD, gnrCoverage, lengthList, r_squared_values)
            else:
                self.updateBrowserText("No valid GNR lengths found to plot.")
        else:
            self.updateBrowserText("No line segments found for GNR length histogram.")

    def saveResults(self, filename=None, avgLength=None, lengthSTD=None, gnrCoverage=None, lengthList=None, r_squared_values=None):
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.data_dir, f"gnr_results_{timestamp}.csv")
        df = pd.DataFrame({
            'filename': [filename],
            'avg_length': [avgLength],
            'length_std': [lengthSTD],
            'gnr_coverage': [gnrCoverage],
            'length_list': [lengthList],
            'r_squared_values': [r_squared_values]
        })
        savefile = self.data_dir + "\\_results.csv"
        if not os.path.exists(savefile):
            with open(savefile, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(df.columns)
                writer.writerow(df.iloc[0])
        else:
            with open(savefile, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(df.iloc[0])
            self.updateBrowserText(f"GNR results saved to {savefile}")
        self.updateBrowserText(f"Results saved to {savefile}")

    def saveLabelClips(self):
        from datetime import datetime

        if not self.data_dir or not os.path.isdir(self.data_dir):
            self.updateBrowserText("No valid image directory selected.")
            return
        
        if not self.symmetryVector:
            self.updateBrowserText("Symmetry vector not set. Please set it before saving label clips.")
            return

        n_clips, ok = QInputDialog.getInt(
            self, "Save Label Clips", "Clips per (label, angle):", value=5, min=1, max=500
        )
        if not ok:
            return

        context_scale, ok = QInputDialog.getDouble(
            self, "Save Label Clips", "Context scale around bbox:", value=1.6, min=1.0, max=5.0, decimals=2
        )
        if not ok:
            return

        angle_bin_deg, ok = QInputDialog.getInt(
            self, "Save Label Clips", "Bin angle width (deg):", value=10, min=1, max=180
        )
        if not ok:
            return

        def _safe_name(text):
            return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(text)).strip("_") or "unknown"

        def _vector_angle_deg(vec_row):
            vx = float(vec_row["dx"]) - float(vec_row["x1"])
            vy = -(float(vec_row["dy"]) - float(vec_row["y1"]))  # image y-axis correction
            if vx == 0 and vy == 0:
                return None
            

            # Vector direction
            v = np.array([vx, vy]) #Temp fix inverting y 
            sym_vec = np.array(self.symmetryVector)
            # Calculate angle in degrees (0 to 360)
            norm_v = np.linalg.norm(v)
            norm_sym = np.linalg.norm(sym_vec)
            if norm_v == 0 or norm_sym == 0:
                angle_deg = None
            else:
                angle1 = np.arctan2(v[1], v[0])
                angle2 = np.arctan2(sym_vec[1], sym_vec[0])
                diff = angle1 - angle2
                angle_deg = (360 - np.degrees(diff)) % 360
                
            # return (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0
            return angle_deg

        def _bbox_contains_line(bbox_row, line_row):
            x1 = float(bbox_row["x1"])
            y1 = float(bbox_row["y1"])
            x2 = x1 + float(bbox_row["dx"])
            y2 = y1 + float(bbox_row["dy"])
            xmin, xmax = sorted((x1, x2))
            ymin, ymax = sorted((y1, y2))

            lx1, ly1 = float(line_row["x1"]), float(line_row["y1"])
            lx2, ly2 = float(line_row["dx"]), float(line_row["dy"])
            return (
                xmin <= lx1 <= xmax
                and ymin <= ly1 <= ymax
                and xmin <= lx2 <= xmax
                and ymin <= ly2 <= ymax
            )

        rect_df = self.annotations_df[
            (self.annotations_df["type"] == "RectROI")
            & (self.annotations_df["label"].notna())
            & (self.annotations_df["label"] != "Empty Label")
        ].copy()

        if rect_df.empty:
            self.updateBrowserText("No labeled bounding boxes found.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join(self.data_dir, f"label_clips_{timestamp}")
        os.makedirs(out_root, exist_ok=True)

        image_cache = {}
        counters = {}
        saved_total = 0

        for filename in rect_df["file"].dropna().unique():
            image_path = os.path.join(self.data_dir, filename)
            if not os.path.isfile(image_path):
                self.updateBrowserText(f"Image not found: {filename}")
                continue

            if filename not in image_cache:
                image_cache[filename] = imread(image_path)

            img = image_cache[filename]
            h, w = img.shape[:2]

            file_rects = rect_df[rect_df["file"] == filename]
            file_lines = self.annotations_df[
                (self.annotations_df["file"] == filename)
                & (self.annotations_df["type"] == "VectorROI")
            ]

            for _, bbox in file_rects.iterrows():
                label = str(bbox["label"])
                x = float(bbox["x1"])
                y = float(bbox["y1"])
                bw = float(bbox["dx"])
                bh = float(bbox["dy"])

                # The box's own orientation, else the first vector inside the box (older annotations)
                angle = None
                own = bbox.get("angle") if "angle" in bbox.index else None
                if own is not None and np.isfinite(own):
                    sym = np.array(self.symmetryVector)
                    angle = (360 - (own - np.degrees(np.arctan2(sym[1], sym[0])))) % 360
                for _, line in ([] if angle is not None else file_lines.iterrows()):
                    if _bbox_contains_line(bbox, line):
                        angle = _vector_angle_deg(line)
                        break

                if angle is None:
                    angle_key = "angle_unknown"
                else:
                    binned = int(round(angle / angle_bin_deg) * angle_bin_deg) % 360
                    angle_key = f"angle_{binned:03d}deg"

                group_key = (_safe_name(label), angle_key)
                if counters.get(group_key, 0) >= n_clips:
                    continue

                cx = x + bw / 2.0
                cy = y + bh / 2.0
                crop_w = max(1.0, bw * context_scale)
                crop_h = max(1.0, bh * context_scale)

                x0 = max(0, int(np.floor(cx - crop_w / 2.0)))
                y0 = max(0, int(np.floor(cy - crop_h / 2.0)))
                x1 = min(w, int(np.ceil(cx + crop_w / 2.0)))
                y1 = min(h, int(np.ceil(cy + crop_h / 2.0)))

                if x1 <= x0 or y1 <= y0:
                    continue

                clip = img[y0:y1, x0:x1]
                label_dir = os.path.join(out_root, _safe_name(label), angle_key)
                os.makedirs(label_dir, exist_ok=True)

                idx = counters.get(group_key, 0) + 1
                clip_name = f"{os.path.splitext(filename)[0]}_{_safe_name(label)}_{angle_key}_{idx:03d}.png"
                clip_path = os.path.join(label_dir, clip_name)

                if clip.ndim == 2:
                    plt.imsave(clip_path, clip, cmap="gray")
                else:
                    plt.imsave(clip_path, clip)

                counters[group_key] = idx
                saved_total += 1

        self.updateBrowserText(f"Saved {saved_total} clips to {out_root}")

    def open_roi_analysis(self):
        """Open (or bring to front) the ROI analysis window (core/roi_analysis_dialog.py)."""
        from core.roi_analysis_dialog import RoiAnalysisDialog
        if self._roi_analysis_dialog is None:
            self._roi_analysis_dialog = RoiAnalysisDialog(self._roi_analysis_data, parent=self)
        else:
            self._roi_analysis_dialog.refresh_labels()
        self._roi_analysis_dialog.show()
        self._roi_analysis_dialog.raise_()
        self._roi_analysis_dialog.activateWindow()

    def _roi_analysis_data(self, scope):
        """Analysis input for the current image ("current") or every loaded image ("all")."""
        from core.roi_analysis import AnalysisData, ImageInfo
        if scope == "current":
            viewer = self._current_viewer()
            viewers = [viewer] if viewer is not None else []
        else:
            viewers = self._viewers()
        images, frames = {}, []
        for viewer in viewers:
            if not viewer.hasImage() or not viewer._imgPath:
                continue
            file = os.path.basename(viewer._imgPath)
            pixmap = viewer.pixmap()
            path = viewer._imgPath
            images[file] = ImageInfo(file, pixmap.width(), pixmap.height(), viewer._pxScale or None,
                                     ref_angle=self.reference_angle(viewer),
                                     image=lambda p=path: imread(p, as_gray=True))   # loaded only for maps
            frames.append(viewer.annotationsFrame())
        frames = [f for f in frames if len(f)]
        annotations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FRAME_COLUMNS)
        colors = {label: label_color(label, self.labelList) for label in self.labelList}
        colors[SYMMETRY_LABEL] = "#F5E400"
        return AnalysisData(annotations, images, colors)

    def showResults(self):
        if any("9AP" in str(label) for label in self.labelList):
            self.get_vector_angles_within_9AP()
        if any("GNR" in str(label) for label in self.labelList):
            print("Plotting GNR length histogram...")
            self.plotLineSegResults()

    def choosePredictionModel(self):
        from ultralytics import YOLO
        options = QFileDialog.Option.DontUseNativeDialog
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model Weights",
            "",
            "PyTorch Model Files (*.pt);;All Files (*)",
            options=options,

        )
        if model_path:
            self.ui.label_2.setText(f"Select classes to predict, then run your ML predictions.")
            self.updateBrowserText(f"Loaded YOLO model weights: {model_path}")
            self.yolo_model_path = model_path
            self.predictions_ui.label.setText(f"Model: {os.path.basename(model_path)}")
            model = YOLO(model_path)
            classes = model.names
            table = self.predictions_ui.tableWidget
            table.setRowCount(0)
            class_names = classes.values() if isinstance(classes, dict) else classes
            class_names = [str(label) for label in classes.values()
                                            if str(label) != 'Empty Label']
            for row, class_name in enumerate(class_names):
                table.insertRow(row)
                item = QTableWidgetItem(str(class_name))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                table.setItem(row, 0, item)
            
        else:
            self.ui.label_2.setText("No model selected.")
            self.updateBrowserText("No YOLO model weights selected.")
            self.predictions_ui.label.setText("Model: None")

    def auto_segment_to_rois(self):
        """
        Quick auto-segmentation of the current image (mask from runGNRSegmentation) into
        one ROI per distinct object.  For full control use the image adaptation dialog
        (adaptImage_show), which shares the same ROI generation (core.roi_extraction)
        and the same self.roi_settings.
        """
        import dataclasses
        from core.roi_extraction import RoiSettings, extract_objects

        current_index = self.ui.stackedWidget.currentIndex()
        if 0 <= current_index < len(self.image_paths):
            filename = os.path.basename(self.image_paths[current_index])
        else:
            self.updateBrowserText("No image selected.")
            return

        # Prompt user for target ROI conversion format
        roi_options = ["PolygonROI", "RectROI", "LineSegmentROI", "PointROI"]
        choice, ok = QInputDialog.getItem(
            self, "Auto Threshold Segmentation", "Select output ROI type:", roi_options, 0, False
        )
        if not ok:
            return

        cleaned = self.runGNRSegmentation()     # binary mask (also clears existing annotations)
        settings = dataclasses.replace(self.roi_settings or RoiSettings(), roi_type=choice)
        objects = extract_objects(np.asarray(cleaned, dtype=bool), settings)
        self._add_objects_as_rois(objects, filename)

    def makePrediction(self):
        from core.yolo_predict import predict_bounding_boxes
        try:
            if not self.yolo_model_path:
                print("No YOLO model weights selected.")
                return
            weights = self.yolo_model_path
            image = self.ui.stackedWidget.currentWidget()._imgPath
        except Exception as e:
            print(f"Error occurred while setting up prediction: {e}")
            return
        self.ui.label_2.setText("Running prediction...")
        self.updateBrowserText("Running prediction on " + image.split("/")[-1] + " using weights: " + weights.split("/")[-1])

        selected_classes = []
        for row in range(self.predictions_ui.tableWidget.rowCount()):
            item = self.predictions_ui.tableWidget.item(row, 0)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected_classes.append(row)  # Use row index as class ID
                # class_name = item.text()
                # class_id = next((key for key, value in self.class_mapping.items() if value == class_name), None)
                # if class_id is not None:
                #     selected_classes[class_id] = class_name
        prediction_results, classes = predict_bounding_boxes(
            model_path=weights,
            image_path=image,
            conf_threshold=0.3,
            selected_classes = selected_classes,
            save_annotated_image=False,
            output_image_path="prediction_output.jpg"
        )

        # prediction_results.to_csv("prediction_results.csv", index=False)
        self.load_annotations(prediction_results=prediction_results, classes=classes,
                              replace=self.predictions_ui.checkBox.isChecked())
        self.ui.label_2.setText("Prediction complete.")
            

def handleLeftClick(x, y):
    row = int(y)
    column = int(x)
    print("Clicked on image pixel (row="+str(row)+", column="+str(column)+")")