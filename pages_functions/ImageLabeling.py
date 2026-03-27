from PyQt6.QtWidgets import QWidget, QVBoxLayout, QInputDialog, QTreeWidgetItem, QListWidgetItem, QApplication
import PyQt6.QtGui as QtGui
import PyQt6.QtCore as QtCore
from ui.pages.ImageLabeling_ui import Ui_Form
from ui.popUps.labelTextInput_ui import Ui_Form as Ui_LabelTextInput
from ui.popUps.automaticAnnotation_ui import Ui_Form as Ui_AutomaticAnnotations
from PyQt6.QtWidgets import QFileDialog, QLabel, QGridLayout
import os
from ui.PyQtImageViewer.QtImageViewer import QtImageViewer
import numpy as np
import csv
import pandas as pd
from core.first_pass_segment import run_first_pass_segment
from core.second_pass_segment import run_second_pass_segment
from skimage.io import imread
import matplotlib.pyplot as plt
from PyQt6.QtGui import QColor

# pyuic6.exe .\ImageLabeling.ui -o .\ImageLabeling_ui.py
class ImageLabeling(QWidget):
    def __init__(self, main_window=None):
        super(ImageLabeling, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.main_window = main_window

        # Create and Hide the label text input widget
        self.label_input_ui = Ui_LabelTextInput()
        self.label_input_dialog = QWidget(self)
        self.label_input_ui.setupUi(self.label_input_dialog)
        label_ui_layout = self.ui.widget_10.layout()
        if label_ui_layout is None:
            label_ui_layout = QVBoxLayout(self.ui.widget_10)
        label_ui_layout.addWidget(self.label_input_dialog)

        # Create automaticAnnotations popup
        self.autoAnnotations_ui = Ui_AutomaticAnnotations()
        self.autoAnnotations_dialog = QWidget(self)
        self.autoAnnotations_ui.setupUi(self.autoAnnotations_dialog)
        autoAnnotations_layout = self.ui.widget_10.layout()
        if autoAnnotations_layout is None:
            autoAnnotations_layout = QVBoxLayout(self.ui.widget_10)
        autoAnnotations_layout.addWidget(self.autoAnnotations_dialog)
        self.ui.widget_9.hide()

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
        self.currentPxSize= None
        self.labelList = []
        self.selected_roi = None  # Track currently selected ROI
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
        self.ui.pushButton_2.clicked.connect(self.setScale)
        self.ui.pushButton_5.clicked.connect(self.load_annotations)
        self.ui.pushButton_3.clicked.connect(self.save_annotations)
        self.ui.pushButton_12.clicked.connect(self.clearAnnotations)
        self.ui.pushButton_6.clicked.connect(self.runAutoSegment)
        self.ui.pushButton_16.clicked.connect(self.clearActivityLog)
        self.ui.pushButton_18.clicked.connect(self.collectSymmetryVector)
        self.ui.pushButton_19.clicked.connect(self.get_vector_angles_within_9AP)
        self.ui.radioButton.clicked.connect(self.toggleAutoVector)
        self.ui.pushButton_21.setVisible(False)  # Hide the GNR button unless GNR tag is active for a label
        self.ui.pushButton_21.clicked.connect(self.runGNRSegmentation)
        self.annotations_df = pd.DataFrame(columns=['objectName', 'type','x1', 'y1', 'dx', 'dy', 'file', 'label'])

        # Buttons for side_ui_popup
        self.ui.pushButton_4.clicked.connect(self.label_ui_show)
        self.label_input_ui.lineEdit.returnPressed.connect(self.setLabels)
        self.label_input_ui.pushButton_3.clicked.connect(self.clearLabelsInput)
        self.label_input_ui.pushButton_4.clicked.connect(self.label_ui_close)
        self.label_input_ui.pushButton.clicked.connect(self.saveLabels)
        self.label_input_ui.pushButton_2.clicked.connect(self.loadLabels)
        # self.label_input_ui.listWidget.keyPressEvent = self.create_label_list_keypress_handler(self.label_input_ui.listWidget)
        self.label_input_ui.treeWidget.setEditTriggers(self.label_input_ui.treeWidget.EditTrigger.DoubleClicked | 
                                                       self.label_input_ui.treeWidget.EditTrigger.SelectedClicked)
        self.label_input_ui.pushButton_5.clicked.connect(self.saveLabelExamples)
        self.current_label = None  # Track currently selected label
        self.labels_ready = False
        # Connect label selection changes
        self.ui.listWidget_2.itemSelectionChanged.connect(self.on_label_selection_changed)
    

        # automaticAnnotations widget
        self.ui.pushButton_20.clicked.connect(self.annotations_ui_show)
        self.autoAnnotations_ui.pushButton.clicked.connect(self.closeAutoAnnotations)


        # Enable keyboard focus and shortcuts
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # Auto load data (temporary)
        # self.choose_directory(dir_path=r"C:\Users\wrja\Desktop\STM_Data\Orito_Reaction\PdGaA(111)_Pd1_CX3-2\20240420_PdGaA111_279a_5min_RT_RT_120C_10min_148C_10min\saved_output")
        self.autoVector = False
        self.colorList = [
        QColor("#D81B60"),       # Custom color (hex code)
        QColor("#1E88E5"),       # Another custom color
        QColor("#FFC107"),
        QColor("#4CDBB0"), 
        QColor("#D55E00"),
        QColor("#CC79A7"),
        QColor("#F0E442"),
        QColor("#56B4E9"),
        QColor("#009E73"),
        QColor("#E69F00"),
        QColor("#0072B2"), 
        ]

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
        if self.annotations_df is not None:
            self.annotations_df = self.annotations_df[0:0]  # Clear existing annotations
        self.load_images()
        self.autoSetScale()

    def updateBrowserText(self, text):
        self.browserText.append(text)
        self.ui.textBrowser.setText('\n'.join(self.browserText))

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
            viewer.roiList.connect(self.roiUpdate)
            viewer.roiSelected.connect(self.roiSelection)

            self.ui.stackedWidget.addWidget(viewer)
            self.ui.label_2.setText("Press the box button to draw bounding boxes on the image")

    def display_selected_image(self):
        selected_items = self.ui.listWidget.selectedItems()
        if not selected_items:
            return
        selected_index = self.ui.listWidget.row(selected_items[0])
        self.ui.stackedWidget.setCurrentIndex(selected_index)
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.current_viewer._labelList = self.labelList
        self.labelTextDisplay(os.path.basename(self.image_paths[selected_index]))
        if self.metadata_dict:
            self.autoSetScale()
        # Maintaining the active mode while switching images
        if self.getActiveMode == 'Zoom':
            self.activate_zoom_mode()
        if self.getActiveMode == 'Bbox':
            self.activate_bounding_box_mode()
        if self.getActiveMode == 'Vector':
            self.activate_vector_mode()
        if self.getActiveMode == 'Point':
            self.activate_point_mode()

    def roiSelection(self, roi):
        """
        Handle the selection of a ROI (Region of Interest) in the viewer.
        :param roi: The selected ROI item.
        """
        self.selected_roi = roi
        # When a ROI is selected, update the label selection to match its label in annotations_df
        label = None
        if self.annotations_df is not None and not self.annotations_df.empty:
            match = self.annotations_df[self.annotations_df['objectName'] == roi[0]]
            if not match.empty and 'label' in match.columns:
                label = match.iloc[0]['label']
        # Find the label in listWidget_2 and select it, or select "Empty Label" if not found
        label_found = False
        for i in range(self.ui.listWidget_2.count()):
            item_text = self.ui.listWidget_2.item(i).text()
            if label and item_text.endswith(label):
                self.ui.listWidget_2.setCurrentRow(i)
                label_found = True
                break
        if not label_found:
            self.ui.listWidget_2.setCurrentRow(0)  # Select "Empty Label"
        if self.symmetryVectorMode:
            self.setSymmetryVector(roi[0])
   
    def active_button_color(self, widget):
        #This will be used to change the background color of the button for the active mode
        # Reset all buttons in widget_5 to grey background except the active one
        for btn in self.ui.widget_5.findChildren(QWidget):
            if hasattr(btn, "setStyleSheet"):
                btn.setStyleSheet("background-color: #D3D3D3;")
        widget.setStyleSheet("background-color: #ADD8E6;")

    def activate_selection_mode(self):
        if not hasattr(self.ui.stackedWidget.currentWidget(), "regionZoomButton"):
            return
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.current_viewer.regionZoomButton = None  # set to None to disable
        self.current_viewer.zoomOutButton = QtCore.Qt.MouseButton.RightButton
        self.ui.label_2.setText("Selection mode activated. Click and drag to select an area.")
        # self.updateBrowserText(f"Selection mode activated.")
        self.getActiveMode = 'Selection'
        self.current_viewer.drawROI = None
        # self.ui.pushButton_11.setStyleSheet("background-color: #ADD8E6;")
        self.active_button_color(self.ui.pushButton_11)

    def activate_zoom_mode(self):
        # Check if the current viewer has a regionZoomButton attribute set
        if not hasattr(self.ui.stackedWidget.currentWidget(), "regionZoomButton"):
            return
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.current_viewer.regionZoomButton = QtCore.Qt.MouseButton.LeftButton
        self.current_viewer.zoomOutButton = QtCore.Qt.MouseButton.RightButton
        self.current_viewer.selectionButton = None  # Disable selection
        self.ui.label_2.setText("Zoom mode activated. Click and drag to zoom the image. Pan with middle mouse.")
        # self.updateBrowserText(f"Zoom mode activated.")
        self.getActiveMode = 'Zoom'
        self.current_viewer.drawROI = None
        self.active_button_color(self.ui.pushButton_10)

    def activate_bounding_box_mode(self):
        # Check if the current viewer has a regionZoomButton attribute set
        if not hasattr(self.ui.stackedWidget.currentWidget(), "regionZoomButton"):
            return
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.current_viewer.regionZoomButton = None  # set to None to disable
        self.current_viewer.selectionButton = None
        self.current_viewer.zoomOutButton = None
        self.ui.label_2.setText("Bounding box mode activated. Draw a box on the image. Pan with middle mouse. Right click to delete.")
        # self.updateBrowserText(f"Bbox mode activated.")
        self.getActiveMode = 'Bbox'
        # Variables to store box coordinates
        self.current_viewer.box_start = None
        self.current_viewer.box_end = None
        
        self.current_viewer.drawROI = "RectROI"  # Set to "Rect" for rectangle drawing
        self.active_button_color(self.ui.pushButton_7)
        # self.current_viewer.roiAdded.connect(roiOutput)
   
    def activate_vector_mode(self):
        # Check if the current viewer has a regionZoomButton attribute set
        if not hasattr(self.ui.stackedWidget.currentWidget(), "regionZoomButton"):
            return
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.current_viewer.regionZoomButton = None
        self.current_viewer.selectionButton = None
        self.current_viewer.zoomOutButton = None
        self.ui.label_2.setText("Vector mode activated. Draw a vector on the image.")
        # self.updateBrowserText(f"Vector mode activated.")
        self.getActiveMode = 'Vector'
        self.current_viewer.drawROI = "LineROI"  # Set to "Line" for vector drawing
        self.current_viewer._currentLabel = 'Empty Label'  # Default label for new vectors
        self.active_button_color(self.ui.pushButton_8)

    def activate_point_mode(self):
        # Check if the current viewer has a regionZoomButton attribute set
        if not hasattr(self.ui.stackedWidget.currentWidget(), "regionZoomButton"):
            return
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.current_viewer.regionZoomButton = None
        self.current_viewer.selectionButton = None
        self.current_viewer.zoomOutButton = None
        self.ui.label_2.setText("Point mode activated. Click to place points on the image.")
        # self.updateBrowserText(f"Point mode activated.")
        self.getActiveMode = 'Point'
        self.current_viewer.drawROI = "PointROI"  # Set to "Point" for point drawing
        self.active_button_color(self.ui.pushButton_9)

    def setScale(self):
        scale_factor, ok = QInputDialog.getDouble(self, "Set pixel to real units scale for all images.", value=1.0, min=0.1, max=10.0)
        if ok:
            self.current_viewer = self.ui.stackedWidget.currentWidget()
            self.current_viewer.setScale(scale_factor)
            # self.updateBrowserText(f"Scale set to {scale_factor:.2f}x")
    
    def autoSetScale(self):
        """
        Automatically set the scale based on the metadata of the current image.
        """
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        # Try to set scale from metadata_dict if filename matches
        filename = None
        current_index = self.ui.stackedWidget.currentIndex()
        if 0 <= current_index < len(self.image_paths):
            filename = os.path.basename(self.image_paths[current_index])
        
        pixel_size_x, width_real, height_real = self.get_scan_dim(filename)
        
        if pixel_size_x is not None:
            self.ui.label_5.setText(
                f"Scale: {pixel_size_x:.5f} nm/pixel \n "
                f"({width_real:.1f} x {height_real:.1f} nm)"
            )
            self.currentPxSize= pixel_size_x

    def get_scan_dim(self, filename):
        if self.metadata_dict and 'Filename' in self.metadata_dict and 'Pixel Scale X (nm)' in self.metadata_dict:
            if filename in self.metadata_dict['Filename']:
                idx = self.metadata_dict['Filename'].index(filename)
                pixel_size_x = self.metadata_dict['Pixel Scale X (nm)'][idx]
                pixel_size_y = self.metadata_dict['Pixel Scale Y (nm)'][idx]
                # self.ui.label_5.setText(f"Scale: {pixel_size:.5f} nm/pixel")
                # Show image dimensions in real units (width x height)
                width_px = int(str(self.metadata_dict['Dimensions'][idx]).split('x')[0])
                height_px = int(str(self.metadata_dict['Dimensions'][idx]).split('x')[1])
                width_real = width_px * pixel_size_x
                height_real = height_px * pixel_size_y
            else:
                self.updateBrowserText(f"No metadata available for the {filename}.")
                return None, None, None
        else:
            self.updateBrowserText(f"No metadata available for the {filename}.")
            return None, None, None
        return pixel_size_x, width_real, height_real

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

    def closeAutoAnnotations(self):
        self.ui.widget_9.hide()
        self.autoAnnotations_dialog.hide()

    def setLabels(self):
        """ Set the labels from the input dialog and update the label list.
        """
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        if not self.current_viewer:
            self.updateBrowserText("No images loaded.")
            self.ui.label_2.setText("No images loaded yet for labeling. Please choose a directory first.")
            return

        # self.label_input_ui.listWidget.addItem(self.label_input_ui.lineEdit.text()) 
        label_item = QTreeWidgetItem([self.label_input_ui.lineEdit.text()])
        label_item.setFlags(label_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)  # Make the item editable
        self.label_input_ui.treeWidget.addTopLevelItem(label_item)
        self.label_input_ui.lineEdit.clear()
        self.label_input_ui.lineEdit.clear()
    
    def saveLabels(self):
        """
        Save the labels to a CSV file.
        """
        # Temp fix for setting labelList from treeWidget
        self.label_ui_close()
        self.ui.widget_9.show()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Labels", "", "CSV Files (*.csv)")
        # file_path = os.path.join(os.getcwd(), "data", "labels.csv")
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for label in self.labelList:
                    writer.writerow([label])
            self.updateBrowserText(f"Labels saved to {file_path}")
            self.label_input_ui.label_3.setText(f"Saved {len(self.labelList)} labels to {file_path}")
    
    def loadLabels(self):
        """
        Load labels from a CSV file and populate the label input dialog.
        """
        # if os.path.exists(os.path.join(os.getcwd(), "data", "labels.csv")):
        #     file_path = os.path.join(os.getcwd(), "data", "labels.csv")
        # else:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Labels", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                self.labelList = [row[0] for row in reader if row]
                # self.label_input_ui.listWidget.clear()
                # self.label_input_ui.listWidget.addItems(self.labelList)
                self.label_input_ui.treeWidget.clear()
                self.label_input_ui.treeWidget.addTopLevelItems([QTreeWidgetItem([label]) for label in self.labelList])
            self.label_input_ui.label_3.setText(f"Loaded {len(self.labelList)} labels from {file_path}")

    def clearLabelsInput(self):
        # self.label_input_ui.listWidget.clear()
        self.label_input_ui.treeWidget.clear()
        # self.labelList = []
        print("Labels cleared")

    def eventFilter(self, obj, event):
        # Allow editing label text on double click
        if obj == self.label_input_ui.treeWidget and event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            item = obj.itemAt(event.pos())
            print(f"Double clicked on item: {item.text(0) if item else 'None'}")
            if item:
                obj.editItem(item, 0)
            return True
        return super().eventFilter(obj, event)

    def label_ui_close(self):
        """
        Close the label input dialog and update the main UI.
        """
        # self.label_input_dialog.close()
        # self.labelList.append(self.label_input_ui.lineEdit.text())
        # self.current_viewer._labelList = self.labelList

        self.ui.widget_9.hide()
        # Get each row entry from treeWidget and append to labelList (including label and any tags in parenthesis)
        self.labelList = []
        for i in range(self.label_input_ui.treeWidget.topLevelItemCount()):
            item = self.label_input_ui.treeWidget.topLevelItem(i)
            row_texts = [item.text(col) for col in range(item.columnCount())]
            if row_texts:
                # If any part of the row contains 'chiral', append '(S)' and '(R)' variants
                label_base = row_texts[0]
                if any('chiral' in text.lower() for text in row_texts):  
                    self.labelList.append(label_base + " (None)")
                    self.labelList.append(label_base + " (S)")
                    self.labelList.append(label_base + " (R)")
                    continue  # Skip normal label_text append below
                if any('gnr' in text.lower() for text in row_texts):
                    self.labelList.append(label_base + " (GNR)")
                    self.ui.pushButton_21.setVisible(True)  # Show the GNR button
                    continue  # Skip normal label_text append below
            label_text = label_base
            self.labelList.append(label_text)

        self.ui.listWidget_2.clear()
        label_idx = 1
        self.ui.listWidget_2.addItem("Empty Label")
        for i, label in enumerate(self.labelList):
            # Create a colored circle icon for each label using colorList
            color = self.colorList[i % len(self.colorList)]
            pixmap = QtGui.QPixmap(20, 20)
            pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(2, 2, 16, 16)
            painter.end()
            item = QtGui.QStandardItem(f"{label_idx}: {label}")
            # If using QListWidget, set icon directly
            list_item = QListWidgetItem(f"{label_idx}: {label}")
            list_item.setIcon(QtGui.QIcon(pixmap))
            self.ui.listWidget_2.addItem(list_item)
            label_idx += 1
        self.current_viewer._labelList = self.labelList
        self.updateBrowserText(f"Labels set: {', '.join(self.labelList)}")

    def load_annotations(self):
        current_index = self.ui.stackedWidget.currentIndex()
        self.ui.radioButton.setChecked(False)
        if 0 <= current_index < len(self.image_paths):
            image = os.path.basename(self.image_paths[current_index])
        # Load annotations from a file
        if self.data_dir is None:
            self.data_dir = ""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", self.data_dir, "CSV Files (*.csv)",)
        if file_path:
            # Load the annotations and display them on the current image
            load_annos = pd.read_csv(file_path)
            # Set labelList to any unique labels present in the loaded annotations
            if 'label' in load_annos.columns:
                self.labelList = [str(l) for l in load_annos['label'].dropna().unique()]
                if 'Empty Label' in self.labelList:
                    self.labelList.remove('Empty Label')
                    # print("Removed label: Empty Label")
                if 'Symmetry Vector' in self.labelList:
                    self.labelList.remove('Symmetry Vector')
                    # print("Removed label: Symmetry Vector")
                self.labelList = sorted(self.labelList, key=lambda x: (not x[0].isdigit(), x.lower()))
                # self.label_input_ui.listWidget.clear()
                # self.label_input_ui.listWidget.addItems(self.labelList)
                self.label_input_ui.treeWidget.clear()
                self.label_input_ui.treeWidget.addTopLevelItems([QTreeWidgetItem([label]) for label in self.labelList])
                self.ui.listWidget_2.clear()
                self.ui.listWidget_2.addItem("Empty Label")
                label_idx = 1
                for idx, label in enumerate(self.labelList, 1):
                    # self.ui.listWidget_2.addItem(f"{idx}: {label}")
                    # Create a colored circle icon for each label using colorList
                    color = self.colorList[idx - 1 % len(self.colorList)]
                    pixmap = QtGui.QPixmap(20, 20)
                    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
                    painter = QtGui.QPainter(pixmap)
                    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                    painter.setBrush(QtGui.QBrush(color))
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.drawEllipse(2, 2, 16, 16)
                    painter.end()
                    item = QtGui.QStandardItem(f"{label_idx}: {label}")
                    # If using QListWidget, set icon directly
                    list_item = QListWidgetItem(f"{label_idx}: {label}")
                    list_item.setIcon(QtGui.QIcon(pixmap))
                    self.ui.listWidget_2.addItem(list_item)
                    label_idx += 1
            hold_viewer = self.ui.stackedWidget.currentWidget()
            # Iterate through each viewer and add their annotations if available
            for idx in range(self.ui.stackedWidget.count()):
                self.ui.stackedWidget.setCurrentIndex(idx)
                current_index = self.ui.stackedWidget.currentIndex()
                self.current_viewer = self.ui.stackedWidget.widget(idx)
                self.current_viewer._labelList = self.labelList
                filename = os.path.basename(self.image_paths[idx])
                image_annotations = load_annos[load_annos['file'] == filename]
                if not image_annotations.empty:
                    self.current_viewer.clearROIs()
                    self.current_viewer.loadROIs(image_annotations, filename, "added")
                filename = os.path.basename(self.image_paths[current_index])
                self.labelTextDisplay(filename)  # Update label text display
            # for index, row in self.annotations_df.iterrows():
            #     label = row['label'] if 'label' in row else None
            #     self.current_viewer.clearROIs()
            #     self.current_viewer._currentLabel = label
            #     self.current_viewer.loadROIs(self.annotations_df, image, "loaded")
                    # if label == "Symmetry Vector":
                    #     self.setSymmetryVector(row)
            self.updateBrowserText(f"Loaded annotations from {file_path}")
            self.current_viewer = hold_viewer  # Return to the originally active viewer
            self.ui.stackedWidget.setCurrentWidget(hold_viewer)
            # self.ui.stackedWidget.setCurrentIndex()

    def save_annotations(self):
        # Save annotations to a file
        # file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "CSV Files (*.csv)")
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
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        current_index = self.ui.stackedWidget.currentIndex()
        self.current_viewer.clearROIs()
        filename = os.path.basename(self.image_paths[current_index])
        self.annotations_df = self.annotations_df[self.annotations_df['file'] != filename]
        self.updateBrowserText("Cleared all annotations from the current image.")
        self.labelTextDisplay(os.path.basename(self.image_paths[current_index]))

    def roiUpdate(self, roiList: list) -> None:
        """
        Updates the region of interest (ROI) information based on the current viewer's state.
        Parameters:
            roiList (list): A list containing ROI data for different ROI types.
        Behavior:
            - Checks the current ROI type and change status from the viewer.
            - If no ROI of the current type exists, the function returns immediately.
            - If an ROI of type "Rect" is added, updates the browser text and the internal bbox dictionary.
            - If an ROI of type "Rect" is deleted, updates the browser text and the internal bbox dictionary.
        """
        # 'type','x1', 'y1', 'dx', 'dy', 'file'
        # Get the filename of the active viewer
        current_index = self.ui.stackedWidget.currentIndex()
        if 0 <= current_index < len(self.image_paths):
            filename = os.path.basename(self.image_paths[current_index])
            # print(f"Active viewer filename: {filename}")
        else:
            filename = None


        label = self.current_viewer._currentLabel if self.current_viewer._currentLabel else self.get_current_label()

        roiData = []
        roi = roiList[0] if roiList else None  # Get the first ROI from the list
        roiType = roi.__class__.__name__
        roiChanged = self.current_viewer.roiChanged
        self.selected_roi = roi
        if self.symmetryVectorMode == True:
            self.symmetryVectorMode = False  # Turn off symmetry vector mode after setting
            self.setSymmetryVector(roi)
            return
        if roiChanged == "deleted":
            if roi is not None:
                # Remove rows with the same objectName (roi instance) from annotations_df
                self.annotations_df = self.annotations_df[self.annotations_df['objectName'] != roi]
                # self.updateBrowserText(f"{roiType[:-3]} deleted")
                self.labelTextDisplay(filename)  # Update label text display 
                self.selected_roi = None  # Clear selected ROI
                return
        else:
            if roiType == "RectROI":
                x1 = roi.rect().x()
                y1 = roi.rect().y()
                dx = roi.rect().width()
                dy = roi.rect().height()
                # Append ROI(s) and filename to self.annotations_df
                # self.annotations_df = pd.DataFrame(bboxData, columns=['type', 'x1', 'y1', 'dx', 'dy', 'file'])
            if roiType == "LineROI":
                x1 = roi.line().x1()
                y1 = roi.line().y1()
                dx = roi.line().x2()
                dy = roi.line().y2()
                # self.annotations_df = pd.DataFrame(bboxData, columns=['type', 'x1', 'y1', 'x2', 'y2', 'file'])
            if roiType == "PointROI":
                x1 = roi.rect().x()
                y1 = roi.rect().y()
                dx = 0  # Point has no width
                dy = 0  # Point has no height
                # self.annotations_df = pd.DataFrame(pointData, columns=['type', 'x1', 'y1', 'dx', 'dy', 'file'])
            
            # Figuring out a way to have chirality detection run whenever drawing a new 9AP
            # box, and auto vector drawing when S or R is determined by user. TBA
            # if label == "9AP" or label == "9AP (S)" or label == "9AP (R)":
            #     image = imread(filename, as_gray=True)
            #     pad = 5
            #     cropped_img = image[y1-pad:dy+pad, x1-pad:dx+pad]
            #     chirality, vectorData = self.detectChirality(cropped_img, pad)
            #     print("9AP detected, running second pass segmentation.")                        
            #     if label == "9AP":
            #         label = f"{label} ({chirality})"
            #     roi_df = pd.concat([roi_df, pd.DataFrame(vectorData, columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])], ignore_index=True)
            #     self.current_viewer.loadROIs(roi_df, os.path.basename(self.image_paths[current_index]), "added")

            roiData.append([roi, roiType, x1, y1, dx, dy, filename, label])
            new_df = pd.DataFrame(roiData, columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])
            self.annotations_df = pd.concat([self.annotations_df, new_df], ignore_index=True)
            self.labelTextDisplay(filename)  # Update label text display
            
            if self.labels_ready is False:
                self.getLabelExamples()

            if self.autoVector == True:
                roi_df = pd.DataFrame(columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])
                if label == '9AP (None)' or label == '9AP (S)' or label == '9AP (R)':
                    self.autoVector = False  # Prevent recursive calls
                    # Run second pass segmentation
                    y2 = y1 + dy
                    x2 = x1 + dx
                    y1 = int(y1)
                    x1 = int(x1)
                    y2 = int(y2)
                    x2 = int(x2)
                    bbox = (y1, x1, y2, x2)
                    print(bbox)
                    image = imread(self.image_paths[current_index], as_gray=True)
                    img = (image - image.min()) / (image.max() - image.min())
                    cropped_img = img[y1:y2, x1:x2]
                    chirality, vector = self.detectChirality(cropped_img, pad=0, filename=filename, label=label, bbox=bbox, plot_results=False)
                    print(vector)
                    roi_df = pd.concat([roi_df, pd.DataFrame(vector, columns=['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label'])], ignore_index=True)
                    self.current_viewer.loadROIs(roi_df, os.path.basename(self.image_paths[current_index]), "added")
                    self.autoVector = True  # Re-enable for future use

    def labelTextDisplay(self, filename):
        # Display label counts in textBrowser_2
        # Count ROIs by type
        roi_counts = self.annotations_df[self.annotations_df['file'] == filename]['type'].value_counts()
        text_lines = []
        # if label and label != "Empty Label":
        #     self.getLabelExamples()
        # If labels exist, count by label as well
        if 'label' in self.annotations_df.columns:
            label_counts = self.annotations_df[
                (self.annotations_df['file'] == filename) &
                (self.annotations_df['label'] != 'Empty Label')
            ]['label'].value_counts()
            if not label_counts.empty:
                text_lines.append("Label counts:")
                text_lines += [f"{label}: {count}" for label, count in label_counts.items()]

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
        """Handle when label selection changes in listWidget_2"""
        self.current_label = self.get_current_label()
        self.current_viewer._currentLabel = self.current_label  # Update the viewer's current label
        # If there's a selected ROI, update its label
        if self.selected_roi is not None and self.current_label is not None:
            self.labelUpdate(self.selected_roi, self.current_label)

    def labelUpdate(self, roi, label):
        """
        Update the label of the selected ROI.
        :param roi: The selected ROI item.
        :param label: The label to assign to the ROI.
        """
        if roi and label:
            if isinstance(roi, list):
                roi = roi[0]
            # Update the label in the annotations DataFrame
            current_index = self.ui.stackedWidget.currentIndex()
            if 0 <= current_index < len(self.image_paths):
                roiType = roi.__class__.__name__
                try: 
                    # Update the label in the annotations DataFrame
                    self.annotations_df.loc[self.annotations_df['objectName'] == roi[0], 'label'] = label
                    self.getLabelExamples()    
                except Exception as e:
                    print(e)
                    print(f"No label available")
                self.labelTextDisplay(os.path.basename(self.image_paths[current_index]))    
                self.current_viewer.deleteROI(roi)  # Remove the old ROI
                # Create a new ROI with the updated label for correct color
                self.current_viewer.addROIs(roi)
            else:
                self.updateBrowserText("No image selected.")
        else:
            self.updateBrowserText("No matching annotation found for labeling.")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for label selection"""
        key = event.key()
        modifiers = event.modifiers()

        # Handle number keys (0-9) for label selection or mode switching with Shift
        if QtCore.Qt.Key.Key_0 <= key <= QtCore.Qt.Key.Key_9:
            index = key - QtCore.Qt.Key.Key_0
            if modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
                # Switch modes based on number key (1-5)
                if index == 1:
                    self.activate_selection_mode()
                elif index == 2:
                    self.activate_zoom_mode()
                elif index == 3:
                    self.activate_bounding_box_mode()
                elif index == 4:
                    self.activate_vector_mode()
                elif index == 5:
                    self.activate_point_mode()
                # Add more modes if needed
                return
            else:
                if index < self.ui.listWidget_2.count():
                    self.ui.listWidget_2.setCurrentRow(index) 
                    self.on_label_selection_changed()
                    return
        
        # Handle Escape key to deselect ROI
        if key == QtCore.Qt.Key.Key_Escape:
            if self.selected_roi is not None:
                self.selected_roi = None
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
        Only show labels that have actual annotations in the current image.
        """
        # Clear all widgets from the grid layout
        for i in reversed(range(self.label_input_ui.gridLayout.count())): 
            self.label_input_ui.gridLayout.itemAt(i).widget().setParent(None)

        current_index = self.ui.stackedWidget.currentIndex()
        
        if 0 <= current_index < len(self.image_paths):
            image_path = self.image_paths[current_index]
            filename = os.path.basename(image_path)
            image = QtGui.QImage(image_path)
            pixmap = QtGui.QPixmap.fromImage(image)
            
            # Get only labels that have actual annotations in the current image
            current_image_annotations = self.annotations_df[
                (self.annotations_df['type'] == 'RectROI') &
                (self.annotations_df['file'] == filename) &
                (self.annotations_df['label'].notna()) &  # Exclude NaN labels
                (self.annotations_df['label'] != 'Empty Label')  # Exclude empty labels
            ]
            
            # Get unique labels that have annotations
            labels_with_annotations = current_image_annotations['label'].unique()
            
            for idx, label in enumerate(labels_with_annotations):
                # Find the first instance of this label
                matches = current_image_annotations[current_image_annotations['label'] == label]
                if not matches.empty:
                    first_instance = matches.iloc[0]
                    x, y, dx, dy = first_instance['x1'], first_instance['y1'], first_instance['dx'], first_instance['dy']
                    cropped = pixmap.copy(int(x), int(y), int(dx), int(dy))
                    
                    # Create label text widget
                    label_text_widget = QLabel(f"{label}")
                    label_text_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    
                    # Create image widget
                    image_widget = QLabel()
                    image_widget.setPixmap(cropped.scaled(100, 100, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
                    
                    # Add both widgets to the grid (2 columns: image, label)
                    row = idx
                    self.label_input_ui.gridLayout.addWidget(image_widget, row, 0)
                    self.label_input_ui.gridLayout.addWidget(label_text_widget, row, 1)
                    
                    # print(f"Added label '{label}' example to grid at row {row}")
            if set(self.labelList).issubset(set(labels_with_annotations)):
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
            label_annos['scale'] = self.currentPxSize
            save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Label Examples")
            if save_dir:
                # Save cropped images
                for i in range(self.label_input_ui.gridLayout.rowCount()):
                    try:
                        label_widget = self.label_input_ui.gridLayout.itemAtPosition(i, 1).widget()
                        image_widget = self.label_input_ui.gridLayout.itemAtPosition(i, 0).widget()
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
        self.ui.radioButton.setChecked(False) # Disable autovector 
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
                    if label_name not in self.labelList:
                        self.labelList.append(label_name)
                        # self.label_input_ui.listWidget.addItem(label_name)
                        self.label_input_ui.treeWidget.addTopLevelItem(QTreeWidgetItem([label_name]))
                        self.ui.listWidget_2.addItem(f"{len(self.labelList)}: {label_name}")
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
        # if hasattr(self, 'currentPxSize') and self.currentPxSize and self.currentPxSize != 0:
        #     areas = [area / (self.currentPxSize ** 2) for area in areas]
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
        Set the symmetry vector based on the clicked position.
        :param x: The x-coordinate of the click.
        :param y: The y-coordinate of the click.
        """
        print("Setting symmetry vector...")
        try:
            if roi is None:
                self.updateBrowserText("No vector ROI selected.")
                return
            # For a LineROI, get the vector's dx and dy
            # print(f"Selected ROI: {roi.__class__.__name__}")
            if roi.__class__.__name__ == "LineROI":
                x1 = roi.line().x1()
                y1 = roi.line().y1()
                x2 = roi.line().x2()
                y2 = roi.line().y2()
                dx = (x2 - x1)
                dy = -(y2 - y1)
                self.symmetryVector = [dx, dy]
                self.updateBrowserText(f"Symmetry vector set: dx={dx}, dy={dy}")
                drawer = self.current_viewer.drawROI
                # Temporarily set drawROI to "LineROI" to remove the old vector
                self.current_viewer.drawROI = "LineROI"
                self.current_viewer.deleteROI(roi)  # Remove the old vector ROI
                self.current_viewer.roichanged = "loaded"  # Mark as added for roiUpdate
                self.current_viewer.addROIs(roi, setColor = "#F5E400", labelOverride = "Symmetry Vector")
                self.current_viewer.drawROI = drawer
            else:
                self.updateBrowserText("Selected ROI is not a vector.")
            self.symmetryVectorMode = False
            slope = dy / dx
            self.ui.label_2.setText(f"Symmetry vector set: Slope = {slope}")
        except Exception as e:
            print(f"Error setting symmetry vector: {e}")
            self.updateBrowserText(f"Error setting symmetry vector: {e}")
            self.symmetryVectorMode = False

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
            vectorData = [[None, 'LineROI', chiral_x0, chiral_y0, chiral_x1, chiral_y1, filename, 'Empty Label']]
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
        Find all LineROIs (vectors) within 9AP bounding boxes and calculate their angles with respect to the symmetryVector.
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
            # Get all vectors (LineROI)
            vector_df = self.annotations_df[
                (self.annotations_df['file'] == filename) &
                (self.annotations_df['type'] == 'LineROI')
            ]
            
            # Get the dimensions of the scan for coverage calculation
            pix_x_scale, scan_width, scan_height = self.get_scan_dim(filename)
            if scan_width is None or scan_height is None:
                self.updateBrowserText(f"Could not determine scan dimensions for {filename}. Skipping coverage calculation.")
                continue
            scan_area = scan_width * scan_height
            feature_properties, num_bins, thresholds = self.load_feature_parameters()
            molec_area = feature_properties['9AP']['area'] # Need to abstract by calling the desired molecules label
            total_molec_area = len(bbox_df) * molec_area
            coverage.append(total_molec_area / scan_area if scan_area > 0 else 0)
            molec_count.append(len(bbox_df))

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
                # Find vectors inside this bounding box
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
                            print(f"Vector {i}: v={v}, sym_vec={sym_vec}")
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
        ee = (total_S - total_R) / (total_S + total_R)
        coverage = np.mean(coverage) * 100  # Convert to percentage
        molec_avg = np.mean(molec_count)
        unknown_molec = np.sum(molec_count) - (total_S + total_R)
        unknown_percent = (unknown_molec / np.sum(molec_count)) * 100 if np.sum(molec_count) > 0 else 0
        print(f"Unknown 9AP: {unknown_percent:.2f}")

        std = np.std(molec_count)
        print(f"9AP count std dev: {std:.2f}")


        # Plot stacked histogram by enantiomer
        if results:
            bins = np.linspace(0, 360, 36)
            plt.figure(figsize=(7, 5))
            plt.hist(
            [angle_dict['S'], angle_dict['R']],
            bins=bins,
            stacked=True,
            color=[ '#56B4E9','#CC79A7',],
            label=['(S)', '(R)']
            )
            
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
            plt.title(f'Molecule Orientations w.r.t. Symmetry Vector [-101]')
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
        self.ui.label_2.setText("GNR characterization running.")
        from core.GNR_segment import run_GNR_segment
        current_index = self.ui.stackedWidget.currentIndex()
        
        if 0 <= current_index < len(self.image_paths):
            filename = os.path.basename(self.image_paths[current_index])
            # self.annotations_df = self.annotations_df[self.annotations_df['file'] != filename]
            self.current_viewer = self.ui.stackedWidget.currentWidget()
            image = imread(self.image_paths[current_index], as_gray=True)

            # Get the dimensions of the scan for coverage calculation
            px_scale, scan_width, scan_height = self.get_scan_dim(filename)
            if scan_width is None or scan_height is None:
                self.updateBrowserText(f"Could not determine scan dimensions for {filename}. Igrnoe GNR length evaluation.")

            gnr_rois = run_GNR_segment(image, px_scale=px_scale, sigma=None, block_size=11, num_bins=None, thresholds=None, plot_results=True)
        self.ui.label_2.setText(f"Number of GNR segments found: {gnr_rois[3]}")
        self.updateBrowserText(f"GNR segmentation found {gnr_rois[3]} segments.")
        self.updateBrowserText(f"Average GNR length: {gnr_rois[4]:.2f} nm.")
        

def handleLeftClick(x, y):
    row = int(y)
    column = int(x)
    print("Clicked on image pixel (row="+str(row)+", column="+str(column)+")")