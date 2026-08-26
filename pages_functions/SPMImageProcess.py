from tkinter import Image

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from ui.PyQtImageViewer.QtImageViewer import QtImageViewer
from PyQt6 import QtCore
from PyQt6.QtGui import QBrush, QImage
from ui.pages.SPMImageProcess_ui import Ui_Form
from ui.pages.ImageLabeling_ui import Ui_Form as Ui_ImageLabeling
from ui.popUps.imageProcessing_ui import Ui_Form as Ui_imageProcessing
from PyQt6.QtWidgets import QFileDialog
import os
import pySPM as spm
from matplotlib import pyplot as plt
import io
import sys
import numpy as np
import copy
from scipy.ndimage import zoom
from scipy import ndimage
import csv
from skimage import filters
import cv2

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
        self.current_spm = None
        self.dir_path = None
        self.saved_output_dir = None  # Directory to save JPG conversions
        self.spm_objects = []  # List to store SPM image objects
        self.selected_index = 0  # To keep track of the selected image index
        self.selected_channel = None  # To keep track of the selected channel
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

        # Set up the image processing UI
        self.process_image_ui = Ui_imageProcessing()
        self.process_image_dialog = QWidget(self)
        self.ui.label_7.setText("Image Processing Menu")
        self.process_image_ui.setupUi(self.process_image_dialog)
        process_ui_layout = self.ui.widget_10.layout()
        if process_ui_layout is None:
            process_ui_layout = QVBoxLayout(self.ui.widget_10)
        process_ui_layout.addWidget(self.process_image_dialog)
        self.ui.widget_9.hide()  # Hide the image processing widget initially

        self.checkedMods = []
        self.oldCheckedMods = []
        self.process_image_ui.checkBox.stateChanged.connect(self.apply_mod)
        self.process_image_ui.checkBox_2.stateChanged.connect(self.apply_mod)
        self.process_image_ui.checkBox_3.stateChanged.connect(self.apply_mod)
        self.process_image_ui.checkBox_4.stateChanged.connect(self.apply_mod)
        self.process_image_ui.checkBox_7.stateChanged.connect(self.apply_mod)
        self.process_image_ui.checkBox_8.stateChanged.connect(self.apply_mod)
        self.process_image_ui.checkBox_9.stateChanged.connect(self.apply_mod)

        self.process_image_ui.pushButton_3.clicked.connect(self.apply_mod_all)
        self.process_image_ui.pushButton_4.clicked.connect(self.reset_mods)
        self.process_image_ui.pushButton_5.clicked.connect(self.close_process_menu)

        self.kernel_strength = 1.0
        self.process_image_ui.doubleSpinBox_2.valueChanged.connect(self.update_kernel_strength)
        self.process_image_ui.label_2.setText("Strength")

        self.gauss_sigma = 1
        self.process_image_ui.doubleSpinBox.valueChanged.connect(self.update_gauss_sigma)
        self.process_image_ui.label.setText("Sigma")
        
        self.dog_sigma = 1
        self.process_image_ui.horizontalSlider_3.valueChanged.connect(self.update_dog_sigma)
        self.process_image_ui.label_3.setText("Sigma: {}".format(self.dog_sigma))

        all_cmaps = plt.colormaps()
        self.cmap = "gray"
        self.ui.comboBox.addItems(all_cmaps)
        self.ui.comboBox.currentTextChanged.connect(self.setColorMap)
        self.ui.comboBox.setCurrentText("gray")
        

    def choose_directory(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        # self.jpg_folder = self.dir_path + "/jpg_images"
        self.ui.label_2.setText("Please wait, loading images...")

        if self.dir_path:
            self.ui.listWidget.clear()
            self.ui.listWidget_2.clear()
            self.spm_objects.clear()  # Clear previous SPM objects
            # Remove all widgets from stackedWidget
            while self.ui.stackedWidget.count() > 0:
                widget = self.ui.stackedWidget.widget(0)
                self.ui.stackedWidget.removeWidget(widget)
                widget.deleteLater()
            image_extensions = ('.sxm', '.Z_mtrx') #'.png', '.jpg', 
                                #'.jpeg', '.bmp', '.gif', '.tif', '.tiff')
            self.spm_images = {}  
            for item in os.listdir(self.dir_path):
                self.ui.label_2.setText("Please wait, loading image... " + item)
                if item.lower().endswith(image_extensions):
                    full_path = os.path.join(self.dir_path, item)
                    if item.lower().endswith('.sxm'):
                        sxm_obj = spm.SXM(filename=full_path)
                        self.spm_objects.append(sxm_obj) 
                    self.ui.listWidget.addItem(item)

            # Change the text of label
            self.ui.label.setText(f"Selected Directory: {self.dir_path.split('/')[-1]}")
            self.updateBrowserText(f"Loaded directory: {self.dir_path}")

        self.ui.listWidget.setCurrentRow(0)
        self.ui.listWidget_2.setCurrentRow(0)
        self.load_images_initial()
        self.showImageStats()

    def updateBrowserText(self, text):
        self.browserText.append(text)
        self.ui.textBrowser.setText('\n'.join(self.browserText))

    def make_Viewer(self, imgPath=None, spmImage=None):
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

        if self.cmap != "gray":
            norm = (spmImage - np.min(spmImage)) / (np.max(spmImage) - np.min(spmImage) + 1e-9)
            colored = plt.cm.get_cmap(self.cmap)(norm)
            # Convert to 8-bit RGB
            colored = (colored[:, :, :3] * 255).astype(np.uint8)
            # Convert to QImage
            height, width, _ = colored.shape
            spmImage = QImage(colored.data, width, height, 3 * width, QImage.Format.Format_RGB888)

        if imgPath:
            viewer.open(filepath=imgPath)
        else:
            # img_data = np.flipud(spmImage)
            viewer.setImage(image=spmImage)
        viewer.leftMouseButtonReleased.connect(handleLeftClick)
        return viewer

    def load_images_initial(self):
        for spm_obj in self.spm_objects:
            try:
                if spm_obj.filename.lower().endswith(('.png', '.jpg', '.jpeg', 
                                        '.bmp', '.gif', '.tif', '.tiff')):
                    print(spm_obj.filename)
                    viewer = self.make_Viewer(imgPath=spm_obj.filename)
                    continue

                # Use the 'Z' channel for initial display
                channel = 'Z'
                img_channel= spm_obj.get_channel(channel)   
                img_channel_mod = copy.deepcopy(img_channel)  # Create a copy to avoid modifying the original
                # Get pixel data and stretch axes by pixel-to-real size ratio
                img_data = self.crop_NaN(img_channel_mod)
                img_data = self.stretch_image(img_data, img_channel_mod)
                if spm_obj.header['SCAN_DIR'][0][0] == 'up':
                    img_data = np.flipud(img_data)
                viewer = self.make_Viewer(imgPath=None, spmImage=img_data)
                self.ui.stackedWidget.addWidget(viewer)
            except Exception as e:
                print(f"Error loading scan: {e} {spm_obj.filename}")
                continue
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        # self.display_channels()
        self.ui.label_2.setText("All images loaded!")

    def crop_NaN(self, img_channel_mod):
        """
        Crop the image to remove NaN values.
        """
        # Find rows and columns that contain only non-NaN values
        # Check for NaN values in img_data and crop if necessary
        img_data = img_channel_mod.pixels
        if np.isnan(img_data).any():
            # valid_rows = np.where(~np.isnan(img_data).all(axis=1))[0]
            # valid_cols = np.where(~np.isnan(img_data).all(axis=0))[0]
            img_data_cleaned = img_data[~np.isnan(img_data).any(axis=1)]
            # img_data = img_data[np.ix_(valid_rows, valid_cols)]
        else: 
            img_data_cleaned = img_data
        return img_data_cleaned

    def stretch_image(self, img_data, img_channel_mod):
        """
        Stretch the image data so the pixel distances match the physical distances.
        """
        pixel_extent = img_channel_mod.pxs()
        w = pixel_extent[0][0] 
        h = pixel_extent[1][0] 
        zoom_factor_y = w / h
        img_data_stretched = zoom(img_data, (1, zoom_factor_y))  # Stretch height by zoom factor
        return img_data_stretched

    def update_image(self, channel_name, initialize=True, modify=[]):
        """
        Update the current image in the stacked widget after image modification.
        """
        if self.current_viewer is None:
            return

        current_viewer = self.current_viewer

        self.ui.stackedWidget.removeWidget(current_viewer)
        current_viewer.deleteLater()
        
        spm_obj = self.spm_objects[self.selected_index]
        

        # modStateChange = False
        # if self.oldCheckedMods != modify:
        #     self.oldCheckedMods = modify
        #     modStateChange = True
        
        img_channel = spm_obj.get_channel(channel_name)

        if self.scaleBarActive:
            img_channel.add_scale(length=50)

        img_channel_mod = copy.deepcopy(img_channel)

        if spm_obj.filename.lower().endswith(('.png', '.jpg', '.jpeg', 
                                      '.bmp', '.gif', '.tif', '.tiff')):
            imgPath=spm_obj.filename
            spmImage=img_channel_mod.pixels
        else:
            img_data = self.crop_NaN(img_channel_mod)
            img_data_stretched = self.stretch_image(img_data, img_channel_mod)
            imgPath=None
            spmImage=img_data_stretched
            if spm_obj.header['SCAN_DIR'][0][0] == 'up':
                spmImage = np.flipud(spmImage)
        # spmImage = np.flipud(spmImage)  # Flip the image vertically
        # Apply modifications if any
        newImage = spmImage
        modifiedImageData = None
        if modify:
            for mod in modify:
                if mod == 'Correct Slope':
                    modifiedImageData = self.correct_slope(newImage)
                elif mod == 'Correct Plane':
                    modifiedImageData = self.correct_plane(newImage)
                elif mod == 'Correct Lines':
                    modifiedImageData = self.correct_lines(newImage)
                elif mod == 'Scar Removal':
                    modifiedImageData = self.filter_scars_removal(newImage)
                elif mod == 'Correct Median':
                    modifiedImageData = self.correct_median_diff(newImage)
                elif mod == 'Sharpen':
                    modifiedImageData = self.sharpen_image(newImage)
                elif mod == 'Gaussian':
                    modifiedImageData = self.gaussian_filter(newImage)
                elif mod == 'DoG':
                    modifiedImageData = self.apply_DoG(newImage)
                newImage = modifiedImageData #Should consider that this does not take into account the order of operations
        else:
            modifiedImageData = newImage
        try:
            new_viewer = self.make_Viewer(imgPath=imgPath, spmImage=modifiedImageData)
        except Exception as e:
            print(f"Error loading scan: {e} {imgPath}")
            self.spm_objects.pop(self.selected_index)
            self.ui.listWidget.takeItem(self.selected_index)
            return
        self.ui.stackedWidget.insertWidget(self.selected_index, new_viewer)
        self.current_viewer = new_viewer

        self.display_selected_image(initialize=initialize)

    def display_selected_image(self, initialize=True):
        selected_items = self.ui.listWidget.selectedItems()
        if not selected_items:
            return
        self.selected_index = self.ui.listWidget.row(selected_items[0])
        self.ui.stackedWidget.setCurrentIndex(self.selected_index)
        self.current_viewer = self.ui.stackedWidget.currentWidget()
        self.showImageStats()
        self.showScaleBar()
        if initialize is True:
            if self.spm_objects:
                self.display_channels()
                self.apply_mod()
            else:
                self.ui.listWidget_2.clear()
                self.ui.listWidget_2.addItem("No channels available (Default to Z)")

    def display_channels(self):
        try:
            selected_channel_idx = self.ui.listWidget_2.row(self.selected_channel[0]) if self.selected_channel else 0
        except: 
            selected_channel_idx = 0
        self.ui.listWidget_2.clear()
        spm_object = self.spm_objects[self.selected_index]
        # Capture stdout temporarily
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        spm_object.list_channels()
        sys.stdout = old_stdout

        # Split captured output into lines, ignoring the first two lines
        channelList = buffer.getvalue().strip().split('\n')[2:]
        for channel in channelList:
            self.ui.listWidget_2.addItem(channel)
        self.ui.listWidget_2.setCurrentRow(selected_channel_idx)

    def update_selected_channel(self):
        """
        Update the selected channel image in the viewer.
        """
        self.selected_channel = self.ui.listWidget_2.selectedItems()
        if not self.selected_channel:
            return
        self.selected_channel_name = self.selected_channel[0].text().replace('-', '').strip()
        self.ui.listWidget_2.setCurrentItem(self.selected_channel[0])
        self.update_image(channel_name=self.selected_channel_name, initialize=False)

    def save_image(self):
        """
        Save the specified channel of the SXM image data as a JPG file.
        Applies all checked modifications.
        Crops the image if there are NaN slices so modifications still work.
        """
        spm_obj = self.spm_objects[self.selected_index]
        channel = self.selected_channel_name if self.selected_channel_name else 'Z'
        
        self.saved_output_dir = os.path.join(self.dir_path, "saved_output")
        if not os.path.exists(self.saved_output_dir):
            os.makedirs(self.saved_output_dir)

        file = os.path.basename(spm_obj.filename.replace('.sxm', f'_{channel}.jpg'))

        full_path = os.path.join(self.saved_output_dir, file)
         

        # self.current_viewer.image().save(full_path, format='JPG', quality=100, )

        # Capture the visible viewer contents so any scene items/overlays
        # (such as scale bars or annotations) are included in the saved image.
        from PyQt6.QtGui import QImage, QPainter

        output_image = self.current_viewer.image()
        scene = self.current_viewer.scene
        painter = QPainter(output_image)
        scene.render(painter)
        painter.end()
        output_image.save(full_path, format='JPG', quality=100)

        # Additionally append metadata to a text file
        metadata_file = os.path.join(self.saved_output_dir, 'metadata.csv')
        spm_object = spm_obj.get_channel(channel)
        # if spm_obj.header['SCAN_DIR'][0][0] == 'up':
        #     spm_object = np.flipud(spm_object)
        pxScale_og = spm_object.pxs()
        pxScaleX = pxScale_og[0][0]
        pxScaleY = pxScale_og[1][0]
        # Only append metadata if filename is not already present
        should_write = True
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                if file in f.read():
                    should_write = False
        if should_write:
            # Check if file exists to write header only once
            write_header = not os.path.exists(metadata_file)
            with open(metadata_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "Filename", "Channel", "Modifications", "Dimensions", "Pixel Scale X (nm)", "Pixel Scale Y (nm)"
                ])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "Filename": file,
                    "Channel": channel,
                    "Modifications": ', '.join(self.checkedMods),
                    "Dimensions": f"{spm_object.size['pixels']['x']} x {spm_object.size['pixels']['y']}",
                    "Pixel Scale X (nm)": pxScaleX,
                    "Pixel Scale Y (nm)": pxScaleY
                })
                # f.write("Date: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
        return full_path  # Return the path of the saved image for confirmation

    def save_all_images(self):
        """
        Save all images in the current directory with modifications applied.
        """
        starting_index = self.selected_index  # Store the starting index to return to it later
        for idx, spm_obj in enumerate(self.spm_objects):
            self.selected_index = idx
            self.current_viewer = self.ui.stackedWidget.widget(idx)
            if self.scaleBarActive:
                self.showScaleBar()
            self.selected_channel_name = self.selected_channel_name if self.selected_channel_name else 'Z'
            self.save_image()
        self.selected_index = starting_index  # Return to the starting index
        self.current_viewer = self.ui.stackedWidget.widget(self.selected_index)
        self.updateBrowserText("All images saved successfully.")

    def getImageStats(self, spm_object):
        """
        Get image statistics such as pixel scale and scan dimensions.
        :param spm_object: SPM image object
        :return: Dictionary containing pixel scale and scan dimensions
        """
        pxScale_og = spm_object.pxs()
        pxScaleX = pxScale_og[0][0]
        pxScaleY = pxScale_og[1][0]
        pxScaleUnit = pxScale_og[0][1]
        objectSizeX = spm_object.size['pixels']['x']
        objectSizeY = spm_object.size['pixels']['y']

        return {
            "Pixel Scale X": pxScaleX,
            "Pixel Scale Y": pxScaleY,
            "Pixel Scale Unit": pxScaleUnit,
            "Scan Dimension X (nm)": objectSizeX,
            "Scan Dimension Y (nm)": objectSizeY
        }

    def showImageStats(self):
        if self.current_viewer:
            spm_object = self.spm_objects[self.selected_index].get_channel('Z')
            # if spm_obj.header['SCAN_DIR'][0][0] == 'up':
            #     img_channel_mod = np.flipud(img_channel_mod)
            
            pxScaleX, pxScaleY, pxScaleUnit, objectSizeX, objectSizeY = self.getImageStats(spm_object).values()
            self.ui.label_5.setText(f"Pixel scale: 1 Pixel = {pxScaleX:.5g} {pxScaleY:.5g} {pxScaleUnit}")
            self.ui.label_6.setText(f"Scan Dim: {objectSizeX*pxScaleX:.5g} x {objectSizeY*pxScaleY:.5g} nm")
            
    def changeScaleBarState(self, state):
        self.scaleBarActive = not self.scaleBarActive
        self.showScaleBar()  # Update the scale bar visibility based on the new state

    def showScaleBar(self):
        if self.current_viewer is None:
            return
        if self.scaleBarActive:
            pxScaleX, pxScaleY, pxScaleUnit, objectSizeX, objectSizeY = self.getImageStats(self.spm_objects[self.selected_index].get_channel('Z')).values()
            
            scaleLength = objectSizeX * pxScaleX  # Length of the scale bar in nm
            actual_px_height = self.current_viewer.image().size().height() # Needed for incomplete scans
            import math
            # Optimize the length of the scale bar to be a reasonable fraction of the image width
            target_fraction=0.2
            min_fraction=0.05
            max_fraction=0.4
            allowed_sizes=[0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
            target_len = scaleLength * target_fraction
            # Only consider bars that actually fit reasonably in the frame
            valid = [s for s in allowed_sizes
                    if min_fraction * scaleLength <= s <= max_fraction * scaleLength]

            if not valid:
                # Nothing fits the fraction window -- fall back to whatever
                # allowed size is closest to the image width itself, clipped
                # to the smallest/largest option.
                valid = allowed_sizes

            # Compare in log-space so ratio-distance matters, not absolute distance
            bestScale = min(valid, key=lambda s: abs(math.log(s) - math.log(target_len)))
            print()
            self.current_viewer.addTextOverlay(f"{bestScale:.5g} nm", 2, actual_px_height - 40, font_size=14, color = self.current_viewer.colorList[-2])
            self.current_viewer.addScaleBar(x = 5, y = actual_px_height - 15, length= objectSizeX * (bestScale / scaleLength), height=10, color=self.current_viewer.colorList[-2], thickness= actual_px_height * 0.005)

        elif self.current_viewer is not None:
            self.current_viewer.removeItems()

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


    def image_processing_ui_show(self):
        """
        Show the image processing menu.
        """
        if not self.ui.widget_9.isVisible():
            self.ui.widget_9.show()
            # self.updateBrowserText("Label input dialog opened.")
        else:
            self.ui.widget_9.hide()

    def apply_mod(self):
        """
        Apply a modification based on the checkbox state.
        """
        if self.current_viewer is None:
            return
        
        self.checkedMods = []
        # Iterate through all checkboxes in widget_3 and add checked ones to self.checkedMods
        try:
            for i in range(self.process_image_ui.widget_3.layout().count()):
                widget = self.process_image_ui.widget_3.layout().itemAt(i).widget()
                if hasattr(widget, "isChecked") and widget.isChecked():
                    self.checkedMods.append(widget.text())
            self.update_image(channel_name=self.selected_channel_name, initialize=False, modify=self.checkedMods)
        except Exception as e:
            print(f"Error applying modifications: {e}")
            return
    
    def close_process_menu(self):
        """
        Close the image processing menu.
        """
        self.ui.widget_9.hide()
        # self.updateBrowserText("Image processing menu closed.")
        # # Reset the checked modifications
        # self.checkedMods = []
        # self.oldCheckedMods = []
        # for i in range(self.process_image_ui.widget_3.layout().count()):
        #     widget = self.process_image_ui.widget_3.layout().itemAt(i).widget()
        #     if hasattr(widget, "setChecked"):
        #         widget.setChecked(False)
   
    def apply_mod_all(self):
        """
        Apply all selected modifications to all loaded images.
        """
        if not self.checkedMods:
            return
        og_viewer = self.current_viewer
        for idx, spm_obj in enumerate(self.spm_objects):
            channel_name = self.selected_channel_name if self.selected_channel_name else 'Z'
            self.current_viewer = self.ui.stackedWidget.widget(idx)
            self.selected_index = idx
            self.update_image(channel_name=channel_name, initialize=False, modify=self.checkedMods)

        self.ui.stackedWidget.setCurrentIndex(self.selected_index)
        self.current_viewer = og_viewer
        # self.display_selected_image(initialize=False)

    def reset_mods(self):
        """
        Reset all modifications to their default state.
        """
        self.checkedMods = []
        self.oldCheckedMods = []
        for i in range(self.process_image_ui.widget_3.layout().count()):
            widget = self.process_image_ui.widget_3.layout().itemAt(i).widget()
            if hasattr(widget, "setChecked"):
                widget.setChecked(False)
        spm_obj = self.spm_objects[self.selected_index]
        self.update_image(spm_obj=spm_obj, channel_name=self.selected_channel_name, initialize=False, modify=self.checkedMods)

    def setColorMap(self):
        self.cmap = self.ui.comboBox.currentText()
        self.update_image(channel_name=self.selected_channel_name, initialize=False, modify=self.checkedMods)

   

    def correct_slope(self, spmImage):
        # Correct the image by subtracting a fitted slope along the y-axis
        New = copy.deepcopy(spmImage)
        s = np.mean(spmImage, axis=1)
        i = np.arange(s.shape[0])
        fit = np.polyfit(i, s, 1)
        New -= np.tile(
            np.polyval(fit, i).reshape(np.shape(spmImage)[0], 1),
            np.shape(spmImage)[1],
        )
        # New = np.flipud(New)  # Flip the image vertically
        return New
 
    def correct_plane(self, spmImage, mask=None):
        # Correct the image by subtracting a fitted plane
        x = np.arange(np.shape(spmImage)[0])
        y = np.arange(np.shape(spmImage)[1])
        # Flipping here
        X0, Y0 = np.meshgrid(y,x)
        Z0 = spmImage
        if mask is not None:
            X = X0[mask]
            Y = Y0[mask]
            Z = Z0[mask]
        else:
            X = X0
            Y = Y0
            Z = Z0
        A = np.column_stack((np.ones(Z.ravel().size), X.ravel(), Y.ravel()))
        c, resid, rank, sigma = np.linalg.lstsq(A, Z.ravel(), rcond=-1)

        New = copy.deepcopy(spmImage)
        # Ensure all operands have the same shape as spmImage
        plane = c[0] * np.ones_like(spmImage) + c[1] * X0 + c[2] * Y0
        New -= plane
        return New
    
    def correct_lines(self, spmImage):
        # Correct the image by removing linear features
        New = copy.deepcopy(spmImage)
        New -= np.tile(
            np.mean(spmImage, axis=1).T, (np.shape(spmImage)[1], 1)
        ).T
        return New
  
    def correct_median_diff(self, spmImage):
        N = spmImage
        # Difference of the pixel between two consecutive row
        N2 = N - np.vstack([N[:1, :], N[:-1, :]])
        # Take the median of the difference and cumsum them
        C = np.cumsum(np.median(N2, axis=1))
        # Extend the vector to a matrix (row copy)
        D = np.tile(C, (N.shape[0], 1)).T

        try:
            New = N - D
        except ValueError:
            New = N
            print("Error occurred while correcting median difference")
        return New

    def update_kernel_strength(self, value):
        self.kernel_strength = value
        # self.process_image_ui.label_2.setText("Kernel Strength: {}".format(self.kernel_strength))
        if self.process_image_ui.checkBox_9.isChecked():
            self.apply_mod()
    
    def sharpen_image(self, spmImage):
        """
        Apply sharpening to an image.
        
        Parameters:
            img (numpy.ndarray): Input image (BGR).
            strength (float): Sharpening strength. 
                            0.0 = no effect, 
                            1.0 = normal sharpening, 
                            >1.0 = stronger sharpening.
        
        Returns:
            numpy.ndarray: Sharpened image.
        """
        strength = self.kernel_strength
        # img = np.array(spmImage, dtype=np.float32)
        # Base sharpening kernel
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype=np.float32)
        
        # Blend between identity (no sharpening) and sharpening kernel
        identity = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float32)
        
        # Interpolate between identity and sharpening kernel
        kernel = identity * (1.0 - strength) + kernel * strength
        
        # Apply filter
        sharpened = cv2.filter2D(spmImage, -1, kernel)
        return sharpened

    def update_gauss_sigma(self, value):
        self.gauss_sigma = value
        # self.process_image_ui.label.setText("Sigma: {}".format(self.gauss_sigma))
        if self.process_image_ui.checkBox_8.isChecked():
            self.apply_mod()

    def gaussian_filter(self, spmImage):
        sigma = self.gauss_sigma
        New = filters.gaussian(spmImage, sigma=sigma)
        return New

    def update_dog_sigma(self, value):
        self.dog_sigma = value
        self.process_image_ui.label_3.setText("Sigma: {}".format(self.dog_sigma))
        if self.process_image_ui.checkBox_2.isChecked():
            self.apply_mod()

    def apply_DoG(self, spmImage):
        sigma1 = self.dog_sigma
        sigma2 = sigma1 * 1.6
        New = filters.difference_of_gaussians(spmImage, low_sigma=sigma1, high_sigma=sigma2)
        return New

    def filter_scars_removal(self, spmImage):
        # Apply a median filter to remove scars
        img = np.array(spmImage, dtype=np.float64)
        print(np.shape(img))

        filtered = ndimage.median_filter(img, size=3)
        return filtered

    def send_to_imageLabeling(self):
        """
        Send the current dir to the image labeling module.
        """
        from pages_functions.ImageLabeling import ImageLabeling
        from PyQt6.QtWidgets import QMessageBox

        # Check if JPG files exist for all images in the selected directory
        self.saved_output_dir = os.path.join(self.dir_path, "saved_output")
        if not os.path.exists(self.saved_output_dir):
            os.makedirs(self.saved_output_dir)

        jpg_files_exist = True

        for spm_obj in self.spm_objects:
            # channel = self.selected_channel_name if self.selected_channel_name else 'Z'
            channel = 'Z'  # Default to Z channel
            jpg_filename = os.path.basename(spm_obj.filename.replace('.sxm', f'_{channel}.jpg'))
            jpg_path = os.path.join(self.saved_output_dir, jpg_filename)
            if not os.path.exists(jpg_path):
                jpg_files_exist = False
                break

        if not jpg_files_exist:
            reply = QMessageBox.question(
                self,
                "Save Images",
                "JPG files for all images not found. Would you like to save all now?" \
                "If not, then the label module will open with existing JPG files.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_all_images()


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