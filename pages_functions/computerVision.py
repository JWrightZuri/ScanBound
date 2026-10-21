from PyQt6.QtWidgets import QWidget, QVBoxLayout, QInputDialog, QTreeWidgetItem, QListWidgetItem, QApplication, QTableWidgetItem
import PyQt6.QtGui as QtGui
import PyQt6.QtCore as QtCore
from PyQt6.QtCore import QThread, QObject, pyqtSignal
from core.yolo_prep import convert_csv_to_yolo, split_yolo_dataset
# from core.yolo_prep import augment_yolo_dataset
from core.yolo_prep import get_classes
from core.yolo_predict import predict_bounding_boxes
from ui.popUps.setupML_ui import Ui_Form as Ui_setupML
from ui.pages.computerVision_ui import Ui_Form
from PyQt6.QtWidgets import QFileDialog, QLabel, QGridLayout
import os
import shutil
import sys
import yaml
import pandas as pd
from pathlib import Path
import torch

class EmittingStream(QObject):
    """
    A minimal file-like object that emits a Qt signal for every write.

    This lets us redirect sys.stdout (e.g. verbose print() output from the
    YOLO/ultralytics training code) into a Qt widget. Because it emits a
    signal rather than touching the widget directly, it's safe to use even
    when the writes happen on a background QThread: Qt automatically
    delivers the signal to the main thread via a queued connection.
    """
    text_written = pyqtSignal(str)

    def write(self, text):
        if text:
            self.text_written.emit(str(text))
            print(text, end='')  # Optional: also print to the console for debugging
            # print("Emitting text from stream")  # Optional: also print to the console for debugging

    def flush(self):
        # Required for file-like stream compatibility; nothing to flush.
        pass

class Logger(object):
    """
    A simple logger that writes to both the console and a log file.
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

class TrainingWorker(QObject):
    """
    Runs yolo_train.train_yolo_model() on a background thread so the UI
    stays responsive during training.
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, yolo_train_module, save_dir, classes, data_yaml_path, model_weights,
                 epochs, batch_size, project_name, run_name, device, augmentations, augmentations_custom):
        super().__init__()

        results_dir = yolo_train_module.train_yolo_model(
            data_yaml_path=data_yaml_path,
            save_dir=save_dir,
            classes=classes,
            model_weights=model_weights,
            epochs=epochs,
            batch_size=batch_size,
            project=project_name,
            name=run_name,
            augmentations=augmentations,
            augmentations_custom=augmentations_custom,
            device=device

        )
        self.finished.emit(str(results_dir))


class computerVision(QWidget):
    def __init__(self, main_window=None):
        super(computerVision, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.main_window = main_window

        # Create and hide the setupML widget
        self.setupML_ui = Ui_setupML()
        self.setupML_dialog = QWidget(self)
        self.setupML_ui.setupUi(self.setupML_dialog)
        setupML_layout = self.ui.widget_7.layout()
        if setupML_layout is None:
            setupML_layout = QVBoxLayout(self.ui.widget_7)
        setupML_layout.addWidget(self.setupML_dialog)
        self.ui.widget_7.hide()  # Hide the setupML widget initially
        self.setupML_ui.pushButton_2.clicked.connect(self.close_setupML)  
        self.setupML_ui.pushButton.clicked.connect(self.choose_directories)  
        self.setupML_ui.pushButton_5.clicked.connect(self.clear_directories) 
        self.setupML_ui.pushButton_3.clicked.connect(self.setup_yolo)   
        self.setupML_ui.pushButton_4.clicked.connect(self.set_base_directory)   
        # self.setupML_ui.pushButton_4.clicked.connect(self.set_base_directory)   
        self.setupML_ui.treeWidget.setAcceptDrops(True)
        self.setupML_ui.treeWidget.installEventFilter(self)
    

        self.class_mapping = pd.DataFrame(columns=["label", "roi_count", "type"])  # Initialize class_mapping as an empty DataFrame

        self.valid_directories = []
        self.base_dir = r".\data\user_data\yolo_training"  # Default base directory for YOLO training data
        self.setupML_ui.label_2.setText(f"Destination: {self.base_dir}")
        self.results_dir = None  # Initialize results_dir to None
        self.project = "yolo_training"  # Default project name
        self.name = "detector" # Default run name
        # self.yolo_data = os.path.join(self.base_dir, self.model_name) if self.model_name else os.path.join(self.base_dir, "myModel")
        self.yolo_data = None
        self.setupML_ui.label_2.setText(f"Output location: {self.base_dir}")
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
        self.annotation_extensions = {".csv", ".yolo"}

        self.ui.pushButton.clicked.connect(self.load_inputs)
        self.ui.pushButton_2.clicked.connect(self.open_setupML)
        self.ui.pushButton_4.clicked.connect(self.run_training)
        self.ui.pushButton_5.clicked.connect(self.return_to_imageLabeling)
        # self.ui.pushButton_7.clicked.connect(self.set_base_directory)
        self.ui.lineEdit.textChanged.connect(self.update_dirs)
        self.ui.lineEdit_2.textChanged.connect(self.update_dirs)
        self.ui.label_3.hide()


        # self.populate_chart_selection()
        # self.ui.comboBox.currentIndexChanged.connect(lambda: self.update_displayed_chart(self.ui.label_4, self.ui.comboBox.currentText()))
        # self.ui.comboBox_7.currentIndexChanged.connect(lambda: self.update_displayed_chart(self.ui.label_19, self.ui.comboBox_7.currentText()))


    def eventFilter(self, watched, event):
        """Accept dropped local folders on the setup directory tree."""
        if watched is self.setupML_ui.treeWidget:
            if event.type() in (QtCore.QEvent.Type.DragEnter,
                                QtCore.QEvent.Type.DragMove):
                if event.mimeData().hasUrls() and any(
                    url.isLocalFile() and os.path.isdir(url.toLocalFile())
                    for url in event.mimeData().urls()
                ):
                    event.acceptProposedAction()
                    return True
            elif event.type() == QtCore.QEvent.Type.Drop:
                directories = [
                    url.toLocalFile()
                    for url in event.mimeData().urls()
                    if url.isLocalFile() and os.path.isdir(url.toLocalFile())
                ]
                if directories:
                    self.add_directory(directories)
                    event.acceptProposedAction()
                    return True

        return super().eventFilter(watched, event)

    def set_base_directory(self):
        # Open a dialog to select the base directory for ML models
        base_directory = QFileDialog.getExistingDirectory(
            self,
            "Select Destination for ML training data",
            os.getcwd(),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks | QFileDialog.Option.DontUseNativeDialog
        )
        if base_directory:
            self.base_dir = base_directory
            print(f"Base directory set to: {self.base_dir}")
            self.setupML_ui.label_2.setText(f"Destination: {self.base_dir}")

    def load_inputs(self):
        # Open a dialog to select the output directory for training results
        input_directory = QFileDialog.getExistingDirectory(
            self,
            "Select input Directory for YOLO Training",
            os.getcwd(),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks | QFileDialog.Option.DontUseNativeDialog
        )
        if input_directory:
            self.base_dir = os.path.dirname(input_directory)
            self.yolo_data = input_directory
            print(f"Input directory set to: {self.yolo_data}")
            self.project = os.path.basename(input_directory)  # Use the last part of the path as the project name
            self.ui.lineEdit_2.setText(str(self.project))
            self.update_dirs()
            self.ui.label_3.show()
            self.update_results_table()
            self.populate_chart_selection()

        else:
            print("No input directory selected.")
            return

        self.ui.listWidget.clear()
        class_list = self.ui.listWidget
        try:
            f = yaml.safe_load(open(os.path.join(self.yolo_data, "dataset.yaml"), "r"))
        except Exception as e:
            print(f"Error no .yaml file found: {e}")
            return
        class_names = f.get("names", {})
        print(class_names)
        for class_name in class_names.values():
            if class_name == 'Empty Label':
                continue
            item = QListWidgetItem(str(class_name))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            class_list.addItem(item)

        self.ui.pushButton_4.setEnabled(True)  # Enable the training button after loading inputs
    
    def update_dirs(self):
        # self.ui.label_10.setText(f"Input FData: {self.yolo_data}")
        self.project = self.ui.lineEdit_2.text() if self.ui.lineEdit_2.text() else "yolo_training"
        self.name = self.ui.lineEdit.text() if self.ui.lineEdit.text() else "trained_model"
        self.results_dir = os.path.join(str(self.base_dir),self.project, self.name)  # Update results_dir to point to the expected output location
        
        self.ui.label_3.setText(f"Output destination: \n {self.results_dir}")


    def choose_directories(self):
        # Open a dialog to select multiple directories
        directories = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directories for YOLO Training",
            os.getcwd(),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks | QFileDialog.Option.DontUseNativeDialog
        )
        if directories:
            self.add_directory([directories])

    def add_directory(self, directories=None):
        # Directories selection. They must contain a combination of images and annotations in native or YOLO format.

        if directories is None:
            return
        
        for directory in directories:
            if not os.path.isdir(directory):
                print(f"Skipping {directory}: Not a valid directory.")
                continue
            if not any(
                os.path.splitext(filename)[1].lower() in self.image_extensions
                for filename in os.listdir(directory)
            ):
                print(f"Skipping {directory}: No images found.")
                continue
            if not any(
                os.path.splitext(filename)[1].lower() in self.annotation_extensions
                for filename in os.listdir(directory)
            ):
                print(f"Skipping {directory}: No annotations found.")
                continue
            if directory in self.valid_directories:
                print(f"Skipping {directory}: Already added.")
                continue

            self.valid_directories.append(directory)

            file_path = os.path.join(directory, "annotations.csv")
            # with open(file_path, "r") as f:
        
            new_classes = get_classes(file_path)

            if self.class_mapping.empty:
                self.class_mapping = new_classes
            elif new_classes is not None:
                self.class_mapping = pd.concat([self.class_mapping, new_classes]).drop_duplicates().reset_index(drop=True)
                if {"label", "roi_count"}.issubset(self.class_mapping.columns):
                    self.class_mapping["roi_count"] = pd.to_numeric(
                        self.class_mapping["roi_count"], errors="coerce"
                    ).fillna(0)
                    self.class_mapping = (
                        self.class_mapping.groupby("label", as_index=False, sort=False)
                        .agg({"roi_count": "sum", "type": "first"})
                    )
            
        print(self.class_mapping)
        # Populate the directory column with each valid directory and its files.
        self.setupML_ui.treeWidget.clear()
        for directory in self.valid_directories:
            directory_item = QTreeWidgetItem([directory])
            self.setupML_ui.treeWidget.addTopLevelItem(directory_item)

            for filename in sorted(os.listdir(directory)):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    directory_item.addChild(QTreeWidgetItem([filename]))

            directory_item.setExpanded(True)

        self.setupML_ui.tableWidget.clearContents()
        self.setupML_ui.tableWidget.setRowCount(0)
        class_table = self.setupML_ui.tableWidget
        # class_table.clear()
        for row, class_values in enumerate(self.class_mapping.itertuples(index=False, name=None)):
            class_table.insertRow(row)
            for column, value in enumerate(class_values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                # item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                # item.setCheckState(QtCore.Qt.CheckState.Checked)
                class_table.setItem(row, column, item)

            
        return self.valid_directories

    def clear_directories(self):
        self.valid_directories = []
        self.setupML_ui.treeWidget.clear()
        self.setupML_ui.tableWidget.clearContents()
        self.setupML_ui.tableWidget.setRowCount(0)

    def open_setupML(self):
        self.ui.widget_7.show()  # Show the setupML widget
        # self.setupML_dialog.show()  # Show the setupML dialog

    def close_setupML(self):
        self.ui.widget_7.hide()  # Hide the setupML widget
        # self.setupML_dialog.hide()  # Hide the setupML dialog

    def setup_yolo(self):
        print("Process to set up YOLO on your PC...")

        if self.valid_directories is None or len(self.valid_directories) == 0:
            print("No valid directories selected. Please select directories containing images and annotations.")
            return

        self.yolo_data = self.base_dir
        self.update_dirs()
        self.ui.label_3.show()


        os.mkdir(self.yolo_data) if not os.path.exists(self.yolo_data) else None
        os.mkdir(os.path.join(self.yolo_data, "images")) if not os.path.exists(os.path.join(self.yolo_data, "images")) else None
        os.mkdir(os.path.join(self.yolo_data, "labels")) if not os.path.exists(os.path.join(self.yolo_data, "labels")) else None

        self.generate_yaml()

        for directory in self.valid_directories:
            if ".yolo" in [os.path.splitext(filename)[1].lower() for filename in os.listdir(directory)]:
                print(f"Directory {directory} contains YOLO annotations.")
            elif ".csv" in [os.path.splitext(filename)[1].lower() for filename in os.listdir(directory)]:
                print(f"Directory {directory} contains native annotations. Converting to YOLO format...")
                # TODO: Make this more robust to handle different file names and formats.
                file_path = os.path.join(directory, "annotations.csv")
                # Need to generalize this to handle different annotation file names and formats. For now, assuming a standard name.
                convert_csv_to_yolo(
                    input_dir=directory,
                    csv_path=file_path,
                    output_dir=self.yolo_data,
                )            

        # Next split data in training and validation sets.
        split_yolo_dataset(
            data_dir = self.yolo_data,
            val_ratio = 0.2,
            seed = 42
            )

        # Next create image augmentations for the training set. This is optional and can be skipped if desired.
        # augment_yolo_dataset(
        #     images_dir = os.path.join(self.yolo_data, "images"),
        #     labels_dir = os.path.join(self.yolo_data, "labels"),
        #     output_images_dir = os.path.join(self.yolo_data, "images", "train"),
        #     output_labels_dir = os.path.join(self.yolo_data, "labels", "train"),
        #     num_aug_per_image = 2
        # )
        self.ui.listWidget.clear()
        class_list = self.ui.listWidget
        for class_name in self.class_mapping["label"]:
            if str(class_name) == 'Empty Label':
                continue
            item = QListWidgetItem(str(class_name))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            class_list.addItem(item)
        
        self.update_results_table()
        self.ui.pushButton_4.setEnabled(True)  # Enable the training button after loading inputs


    def generate_yaml(self):
        # Construct dataset.yaml schema
            
        # Format class mapping for YAML ({class_id: name})
        print(f"Generating dataset.yaml with class mapping: {self.class_mapping}")
        yaml_class_mapping = {index: row["label"] for index, row in self.class_mapping.iterrows()}

        yaml_content = {
            "path": os.path.abspath(self.yolo_data), # This creates issues when the training is run on a different machine. Consider using relative paths or a more flexible approach.
            "train": "images\\train",
            "val": "images\\val",
            "names": yaml_class_mapping
        }

        yaml_file = self.yolo_data + "\\dataset.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

        print(f"Config generated at: {yaml_file}")

    
    def custom_augment(self, method="default"):
        import albumentations as A

        if method == "default":
            transform = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.2, 0.2),
                    rotate=(-15, 15),
                    shear=(-10, 10),
                    p=0.5            
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ]
            return transform
    #    elif method == "Auto"


    def run_training(self):
        from core import yolo_train
        print(self.yolo_data)
        if self.yolo_data is None:
            print(f"No YOLO data directory provided.")
            return
        

        # Guard against starting a second training run while one is already in progress.
        existing_thread = getattr(self, "_training_thread", None)
        if existing_thread is not None and existing_thread.isRunning():
            self.ui.textBrowser.append("--Training is already in progress--")
            return

        # self.project = "ScanBound_YOLO_Training"
        # self.project = self.ui.lineEdit_2.text() if self.ui.lineEdit_2.text() else "yolo_training"
        # self.name = self.setupML_ui.lineEdit.text() if self.setupML_ui.lineEdit.text() else "rectroi_detector"
        yolo_model = self.ui.comboBox_2.currentText() if self.ui.comboBox_2.currentText() else "yolo26n.pt"
        device_name = self.ui.comboBox_5.currentText() if self.ui.comboBox_5.currentText() else "cpu"
        epochs = int(self.ui.spinBox_4.value()) if self.ui.spinBox_4.value() else 50
        batch_size = int(self.ui.spinBox_5.value()) if self.ui.spinBox_5.value() else 16
        
        if device_name == "gpu":
            device_name = "cuda"
        if device_name == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            device_name = "cpu"

        augmentation_strategy = self.ui.comboBox_4.currentText().lower() if self.ui.comboBox_4.currentText() else "randaugment"
        augmentations_custom = None

        if augmentation_strategy == "custom":
            augmentation_strategy = "none"  # Disable default augmentations if custom is selected.
            augmentations_custom = self.custom_augment(method="default")

            

        self.ui.textBrowser.append(f"--Starting YOLO training with data from: {self.yolo_data}--")
        self.ui.textBrowser.append("Training may take several minutes to complete. Please be patient.")
        self.ui.progressBar.setValue(0)
        # Disable the training button so it can't be triggered again mid-run.
        self.ui.pushButton_4.setEnabled(False)

        selected_classes = {}
        for index in range(self.ui.listWidget.count()):
            item = self.ui.listWidget.item(index)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected_classes[index] = item.text()

        print(f"Selected classes for training: {selected_classes}")
        # Redirect stdout so the verbose print() output from the training
        # code (e.g. ultralytics epoch/loss logging) is captured and shown
        # in the textBrowser instead of only going to the console.
        # self._original_stdout = sys.stdout
        # self._stdout_stream = EmittingStream()
        # self._stdout_stream.text_written.connect(self._append_training_log)
        # sys.stdout = self._stdout_stream
        sys.stdout = Logger()
        # Set up the worker + thread. Keep references on self so they aren't
        # garbage collected while the thread is running.

        self._training_thread = QThread(self)
        self._training_worker = TrainingWorker(
            yolo_train_module=yolo_train,
            data_yaml_path=self.yolo_data + r"\dataset.yaml",
            save_dir=self.yolo_data,
            model_weights=yolo_model,
            classes=list(selected_classes.keys()),
            epochs=epochs,
            batch_size=batch_size,
            project_name=self.project,
            run_name=self.name,
            device=device_name, 
            augmentations=augmentation_strategy,
            augmentations_custom=augmentations_custom
        )

        self._training_worker.moveToThread(self._training_thread)

        self._training_thread.started.connect(self._training_worker.run)
        self._training_worker.finished.connect(self._on_training_finished)
        self._training_worker.error.connect(self._on_training_error)
        # Once the worker is done (success or error), stop and clean up the thread.
        self._training_worker.finished.connect(self._training_thread.quit)
        self._training_worker.error.connect(self._training_thread.quit)
        self._training_worker.finished.connect(self._training_worker.deleteLater)
        self._training_worker.error.connect(self._training_worker.deleteLater)
        self._training_thread.finished.connect(self._training_thread.deleteLater)

        self._training_thread.start()


    def populate_chart_selection(self):
        # Populate the chart selection combo box with available charts
        # results_dir = os.path.join(self.yolo_data, "runs", "detect")
        # results_dir = Path(r"C:\Users\wrja\Desktop\ScanBound\data\user_data\rectroi_detector-3")

        if self.results_dir is None or not os.path.exists(self.results_dir):
            print("Results directory not found. Please run training first.")
            return
        image_names = [file.name for file in self.results_dir.iterdir() if file.suffix.lower() in self.image_extensions]

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(image_names)
        self.ui.comboBox_7.clear()
        self.ui.comboBox_7.addItems(image_names)
       

    def update_displayed_chart(self, label, chart_path):
        if self.results_dir is None or not os.path.exists(self.results_dir):
            print("Results directory not found. Please run training first.")
            return
        # Update the displayed chart in the given widget
        # results_dir = r"C:\Users\wrja\Desktop\ScanBound\data\user_data\rectroi_detector-3"
        chart_name = os.path.join(self.results_dir, chart_path)
        pixmap = QtGui.QPixmap(chart_name)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        # self.setCentralWidget(label)
        # self.resize(pixmap.width(), pixmap.height())

    def update_results_table(self):
        self.ui.tableView.clearContents()
        self.ui.tableView.setModel(None)

        if not self.yolo_data:
            return
        
        try:
            results_path = os.path.join(self.yolo_data, "results.csv")
        except Exception as e:
            print(f"Error occurred while creating results path: {e}")
            return

        try:
            results = pd.read_csv(results_path)
        except (OSError, pd.errors.ParserError) as error:
            print(f"Could not load results.csv: {error}")
            return

        model = QtGui.QStandardItemModel(self.ui.tableView)
        model.setColumnCount(len(results.columns))
        model.setHorizontalHeaderLabels([str(column) for column in results.columns])
        for row in results.itertuples(index=False, name=None):
            model.appendRow([
                QtGui.QStandardItem("" if pd.isna(value) else str(value))
                for value in row
            ])

        self.ui.tableView.setModel(model)
        self.ui.tableView.resizeColumnsToContents()
        
    def _append_training_log(self, text):
        # Slot connected to EmittingStream.text_written. Runs on the main/UI
        # thread even though the signal is emitted from the worker thread,
        # so it's safe to touch the widget here.
        text = text.rstrip("\n")
        if text == "":
            return
        self.ui.textBrowser.append(text)

    def _on_training_finished(self, results_dir):
        self._restore_stdout()
        self.ui.textBrowser.append(f"--Training results: {results_dir}--")
        self.ui.progressBar.setValue(100)
        self.ui.label_5.setText(f"Training completed.")
        self.ui.pushButton_4.setEnabled(True)
        self.populate_chart_selection()
        self.update_results_table()

    def _on_training_error(self, error_message):
        self._restore_stdout()
        self.ui.textBrowser.append(f"--Training failed: {error_message}--")
        self.ui.pushButton_4.setEnabled(True)

    def _restore_stdout(self):
        original_stdout = getattr(self, "_original_stdout", None)
        if original_stdout is not None:
            sys.stdout = original_stdout
            self._original_stdout = None


    def package_inputs(self):
        print("Packaging input files into a zip to run on a another machine...")

        # Copy the yolo_train.py script to the yolo_data directory
        source_path = os.path.join("core", "yolo_train.py")
        destination_path = os.path.join(self.yolo_data, "yolo_train.py")
        shutil.copy2(source_path, destination_path)

        # Zip yolo_annotations into user_data/yolo_annotations.zip.
        shutil.make_archive(os.path.join("data", "user_data", "prepared_yolo_files"), "zip", self.yolo_data)

        print("Packaged input files into a zip to run on a cuda-enabled machine. Saved at: data/user_data/prepared_yolo_files.zip")


    def return_to_imageLabeling(self):
        """
        Returns the user to the ImageLabeling page.
        """
        from pages_functions.ImageLabeling import ImageLabeling

        if self.main_window:
            self.main_window.ImageLabeling_btn.click()
