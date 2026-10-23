"""
Train a detector on the project's annotations: export a dataset, train, look at the results.

Only ground truth is exported (manual + accepted).  The dataset folder gets a manifest that ties
every training picture to its raw file, processing chain and annotations; the trained model gets
``scanbound_train.json`` with the same information, so predictions are made the way the training
pictures were.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget, QPlainTextEdit, QProgressBar, QPushButton,
                             QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget)

from ...ml.dataset import export_yolo
from ...ml.render import RENDER_MODES, RenderSpec
from ..qtutil import run_job
from ..session import Session
from ..yolo_install import INSTALL_HINT, YoloInstallDialog, yolo_available


def _path_row(edit: QLineEdit, pick) -> QWidget:  # noqa: ANN001
    b = QPushButton("…")
    b.clicked.connect(pick)
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(edit, 1)
    lay.addWidget(b)
    return w


class TrainDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train a detector")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(1000, 720)
        self.session = session
        st = session.project.settings.get("train", {})
        root = session.project.root

        # ---------------------------------------------------------------- dataset
        self.out = QLineEdit(st.get("dataset", os.path.join(root, "scanbound_dataset")))
        self.task = QComboBox()
        self.task.addItems(["detect", "segment"])
        self.task.setCurrentText(st.get("task", "detect"))
        self.task.setToolTip("detect: boxes; segment: outlines (needs a -seg base model such as yolo26n-seg.pt)")
        self.render = QComboBox()
        self.render.addItems(list(RENDER_MODES))
        self.render.setCurrentText(st.get("render", "display_rgb"))
        self.render.setToolTip("display_rgb: as shown (levels + colormap)\ndisplay_gray: levels and gamma, no colormap\n"
                               "data_gray: the data min-max scaled, ignoring display settings")
        self.val = QDoubleSpinBox()
        self.val.setRange(0.0, 0.9)
        self.val.setSingleStep(0.05)
        self.val.setValue(float(st.get("val_ratio", 0.2)))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.setValue(int(st.get("seed", 42)))
        self.empty = QCheckBox("Include reviewed images without annotations (negatives)")
        self.empty.setChecked(bool(st.get("include_empty", True)))
        from ..features import REVIEW
        self.empty.setVisible(REVIEW)                   # images are only marked reviewed in the Review panel
        self.labels = QListWidget()
        self.labels.setMaximumHeight(110)
        for name in session.project.labels.names:
            self.labels.addItem(name)
            it = self.labels.item(self.labels.count() - 1)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
        ds = QGroupBox("Dataset (ground truth only)")
        f1 = QFormLayout(ds)
        f1.addRow("Folder", _path_row(self.out, lambda: self._pick_dir(self.out)))
        f1.addRow("Task", self.task)
        f1.addRow("Picture", self.render)
        f1.addRow("Validation share", self.val)
        f1.addRow("Split seed", self.seed)
        f1.addRow("Classes", self.labels)
        f1.addRow(self.empty)

        # ---------------------------------------------------------------- training
        default_w = st.get("weights") or _find_default_weights(root)
        self.weights = QLineEdit(default_w)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.epochs.setValue(int(st.get("epochs", 100)))
        self.imgsz = QSpinBox()
        self.imgsz.setRange(64, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(int(st.get("imgsz", 640)))
        self.batch = QSpinBox()
        self.batch.setRange(-1, 512)
        self.batch.setValue(int(st.get("batch", 16)))
        self.batch.setToolTip("-1: automatic (GPU only)")
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["cpu", "0", "mps"])
        self.device.setCurrentText(st.get("device", "cpu"))
        self.patience = QSpinBox()
        self.patience.setRange(0, 1000)
        self.patience.setValue(int(st.get("patience", 50)))
        self.runs = QLineEdit(st.get("runs", os.path.join(root, "scanbound_runs")))
        self.name = QLineEdit(st.get("name", "detector"))
        tr = self.training = QGroupBox("Training (Ultralytics YOLO)")
        f2 = QFormLayout(tr)
        f2.addRow("Base weights", _path_row(self.weights, self._pick_weights))
        f2.addRow("Epochs", self.epochs)
        f2.addRow("Image size", self.imgsz)
        f2.addRow("Batch", self.batch)
        f2.addRow("Device", self.device)
        f2.addRow("Early stop patience", self.patience)
        f2.addRow("Runs folder", _path_row(self.runs, lambda: self._pick_dir(self.runs)))
        f2.addRow("Run name", self.name)

        self.b_export = QPushButton("Export dataset")
        self.b_export.clicked.connect(lambda: self.start(train=False))
        self.b_train = QPushButton("Export && train")
        self.b_train.setDefault(True)
        self.b_train.clicked.connect(lambda: self.start(train=True))
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        buttons = QHBoxLayout()
        buttons.addWidget(self.progress, 1)
        buttons.addWidget(self.b_export)
        buttons.addWidget(self.b_train)

        self.yolo_note = QWidget()
        nl = QHBoxLayout(self.yolo_note)
        nl.setContentsMargins(0, 0, 0, 0)
        note = QLabel("Training needs YOLO, which is not installed. Dataset export works without it.")
        note.setWordWrap(True)
        b_install = QPushButton("Install YOLO…")
        b_install.clicked.connect(self._install_yolo)
        nl.addWidget(note, 1)
        nl.addWidget(b_install)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(ds)
        ll.addWidget(self.yolo_note)
        ll.addWidget(tr)
        ll.addStretch(1)
        ll.addLayout(buttons)

        # ---------------------------------------------------------------- results
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.charts = QListWidget()
        self.charts.currentTextChanged.connect(self._show_chart)
        self.chart = QLabel()
        self.chart.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sc = QScrollArea()
        sc.setWidget(self.chart)
        sc.setWidgetResizable(True)
        chart_page = QSplitter()
        chart_page.addWidget(self.charts)
        chart_page.addWidget(sc)
        chart_page.setSizes([180, 600])
        self.table = QTableWidget()
        open_run = QPushButton("Open a finished run…")
        open_run.clicked.connect(self._open_run)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.log, "Log")
        self.tabs.addTab(chart_page, "Charts")
        self.tabs.addTab(self.table, "Metrics per epoch")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.tabs, 1)
        rl.addWidget(open_run, 0, Qt.AlignmentFlag.AlignRight)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 580])
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self.save_dir: Optional[str] = None
        self._set_yolo(yolo_available())

    def _set_yolo(self, ok: bool) -> None:
        """Training controls are shown but disabled until YOLO is installed."""
        self.yolo_ok = ok
        self.yolo_note.setVisible(not ok)
        self.training.setEnabled(ok)
        self.b_train.setEnabled(ok)
        self.b_train.setToolTip("" if ok else INSTALL_HINT)

    def _install_yolo(self) -> None:
        dlg = YoloInstallDialog(self)
        dlg.installed.connect(lambda: self._set_yolo(True))
        dlg.exec()

    # ------------------------------------------------------------------ pickers
    def _pick_dir(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder", edit.text())
        if d:
            edit.setText(d)

    def _pick_weights(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Base weights", os.path.dirname(self.weights.text()), "Weights (*.pt)")
        if p:
            self.weights.setText(p)

    def _say(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _classes(self) -> list[str]:
        return [self.labels.item(i).text() for i in range(self.labels.count())
                if self.labels.item(i).checkState() == Qt.CheckState.Checked]

    def _remember(self) -> None:
        self.session.project.settings["train"] = {
            "dataset": self.out.text(), "task": self.task.currentText(), "render": self.render.currentText(),
            "val_ratio": self.val.value(), "seed": self.seed.value(), "include_empty": self.empty.isChecked(),
            "weights": self.weights.text(), "epochs": self.epochs.value(), "imgsz": self.imgsz.value(),
            "batch": self.batch.value(), "device": self.device.currentText(), "patience": self.patience.value(),
            "runs": self.runs.text(), "name": self.name.text()}
        self.session.project.dirty = True

    # ------------------------------------------------------------------ run
    def start(self, train: bool) -> None:
        self._remember()
        out = self.out.text().strip()
        if os.path.isdir(out) and os.listdir(out):
            from PyQt6.QtWidgets import QMessageBox
            r = QMessageBox.question(self, "Dataset folder not empty",
                                     f"{out} already contains files. Files with the same names are overwritten. Continue?")
            if r != QMessageBox.StandardButton.Yes:
                return
        try:
            report = export_yolo(self.session.project, out, labels=self._classes() or None, task=self.task.currentText(),
                                 val_ratio=self.val.value(), seed=self.seed.value(),
                                 render=RenderSpec(self.render.currentText()), include_empty=self.empty.isChecked())
        except Exception as exc:  # noqa: BLE001
            self._say(f"Export failed: {exc}")
            self.session.error(f"Dataset export failed: {exc}")
            return
        self._say(f"Dataset: {report.summary()}\n  {report.yaml_path}")
        for s in report.skipped[:20]:
            self._say(f"  skipped: {s}")
        self.session.info(f"Dataset exported to {out}: {report.summary()}")
        if not train:
            return
        if report.counts.get("val", 0) == 0:
            self._say("Warning: no validation images (the dataset has one file, or the validation share is 0).")
        weights = self.weights.text().strip()
        if not weights:
            self._say("Choose base weights (e.g. yolo26n.pt).")
            return
        self._train(report.yaml_path, weights)

    def _train(self, yaml_path: str, weights: str) -> None:
        from ...ml.train import train
        self.b_train.setEnabled(False)
        self.b_export.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, self.epochs.value())
        self.progress.setValue(0)
        self._say(f"Training {os.path.basename(weights)} for {self.epochs.value()} epochs on {self.device.currentText()}…")
        settings = dict(weights=weights, epochs=self.epochs.value(), imgsz=self.imgsz.value(), batch=self.batch.value(),
                        device=self.device.currentText(), project=self.runs.text(), name=self.name.text(),
                        patience=self.patience.value(), workers=0 if sys.platform in ("win32", "darwin") else 2)
        run = self.session.project.log_run("train", self.name.text(), {"data": yaml_path, **settings}, [])

        def work(progress=None):
            return train(yaml_path, on_epoch=lambda e, n, m: progress and progress((e, n, m)), **settings)

        def on_progress(v) -> None:  # noqa: ANN001
            e, n, m = v
            self.progress.setValue(e)
            keys = [k for k in m if "mAP50" in k or "precision" in k or "recall" in k]
            self._say(f"epoch {e}/{n}  " + "  ".join(f"{k.split('/')[-1]}={float(m[k]):.3f}" for k in keys[:4]))

        def done(save_dir: str) -> None:
            self._finish()
            run.summary = {"save_dir": save_dir}
            self.session.project.dirty = True
            self._say(f"Done. Weights: {os.path.join(save_dir, 'weights', 'best.pt')}")
            self.session.info(f"Training finished: {save_dir}")
            self.session.project.settings.setdefault("yolo", {})["weights"] = os.path.join(save_dir, "weights", "best.pt")
            self.load_run(save_dir)

        def failed(tb: str) -> None:
            self._finish()
            self._say(tb)
            self.session.error("Training failed: " + tb.strip().splitlines()[-1])

        run_job(work, done=done, failed=failed, progress=on_progress)

    def _finish(self) -> None:
        self.b_train.setEnabled(self.yolo_ok)
        self.b_export.setEnabled(True)
        self.progress.setVisible(False)

    # ------------------------------------------------------------------ results
    def _open_run(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Training run folder", self.runs.text())
        if d:
            self.load_run(d)

    def load_run(self, save_dir: str) -> None:
        self.save_dir = save_dir
        self.charts.clear()
        for name in sorted(os.listdir(save_dir)):
            if name.lower().endswith((".png", ".jpg")):
                self.charts.addItem(name)
        if self.charts.count():
            self.charts.setCurrentRow(0)
            self.tabs.setCurrentIndex(1)
        path = os.path.join(save_dir, "results.csv")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            if rows:
                head = [h.strip() for h in rows[0]]
                self.table.setColumnCount(len(head))
                self.table.setHorizontalHeaderLabels(head)
                self.table.setRowCount(len(rows) - 1)
                for r, row in enumerate(rows[1:]):
                    for c, v in enumerate(row):
                        self.table.setItem(r, c, QTableWidgetItem(v.strip()))

    def _show_chart(self, name: str) -> None:
        if not name or not self.save_dir:
            return
        pm = QPixmap(os.path.join(self.save_dir, name))
        self.chart.setPixmap(pm.scaledToWidth(min(pm.width(), 1100), Qt.TransformationMode.SmoothTransformation))


def _find_default_weights(root: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (root, os.getcwd(), os.path.normpath(os.path.join(here, "..", "..", "..", "..", ".."))):
        p = os.path.join(d, "yolo26n.pt")
        if os.path.isfile(p):
            return p
    return "yolo26n.pt"
