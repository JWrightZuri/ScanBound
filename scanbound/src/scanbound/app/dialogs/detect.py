"""
Run detectors and turn their results into proposals for review.

* the adapt pipeline (tuned interactively in the Image Adaptation dialog, saved in the project)
* a trained YOLO model

Runs happen in a worker thread; each run is logged in the project (its settings and images) and
every proposal links to it, so results can be traced and evaluated per run.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QHBoxLayout, QLineEdit, QProgressDialog, QPushButton, QSpinBox, QWidget)

from ...annotations import Annotation
from ...config import DetectionConfig
from ...detect import Detector
from ...model.provenance import params_digest
from ...proposals import detection_summary, detector_input, propose
from ..features import REVIEW
from ..qtutil import run_job
from ..session import Session

ADAPT_KEY = "adapt_params"


def tune_adapt_pipeline(session: Session, parent: QWidget) -> bool:
    """Open the Image Adaptation dialog on the current image; keeps its settings in the project."""
    from ..widgets.image_adapt_dialog import ImageAdaptDialog
    img = session.image()
    if img is None:
        return False
    gray, _valid = detector_input(img, session.entry.display, "display")
    dx, dy = img.pixel_size
    px_nm = math.sqrt(dx * dy) * 1e9 if img.calibrated and img.x.unit == "m" else None
    labels = session.project.labels
    dlg = ImageAdaptDialog(gray, parent, params=session.project.settings.get(ADAPT_KEY), labels=list(labels.names),
                           label_colors={n: labels.color(n) for n in labels.names}, px_scale=px_nm,
                           current_label=session.current_label)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return False
    session.project.settings[ADAPT_KEY] = dlg.get_params()
    session.project.dirty = True
    session.info("Adapt pipeline settings saved in the project")
    return True


def untouched_result(a: Annotation, name: Optional[str] = None) -> bool:
    """An accepted detector / model result that nobody has edited since it was added (``name``: of
    that detector).  While review is off, a new run replaces these as it replaced pending proposals."""
    return (a.status == "accepted" and a.origin.method != "manual" and not a.modified
            and (name is None or a.origin.name == name))


def _replace_pending(session: Session, image_id: str, method: str, names: Optional[set] = None) -> None:
    old = [a.id for a in session.store.for_image(image_id) if a.origin.method == method
           and (a.status == "proposed" or (not REVIEW and names and any(untouched_result(a, n) for n in names)))]
    if old:
        session.store.remove(old, cascade=True, description="Replace earlier results" if not REVIEW
                             else "Replace unreviewed proposals")


def _add_results(session: Session, results: dict[str, list[Annotation]], method: str, replace: bool,
                 description: str, accept: bool = False) -> int:
    """Add the proposals of a run (replacing earlier pending ones); ``accept`` accepts them at once."""
    n = 0
    with session.store.transaction(description):
        for image_id, anns in results.items():
            if replace:
                _replace_pending(session, image_id, method, {a.origin.name for a in anns})
            session.store.add(anns)
            if accept and anns:
                session.store.accept([a.id for a in anns])
            n += len(anns)
    return n


def run_in_background(session: Session, parent: QWidget, title: str, ids: list[str],
                      work: Callable[[str], list[Annotation]], finish: Callable[[dict], None]) -> None:
    """``work(image_id)`` in a thread for each image, with a cancellable progress dialog."""
    progress = QProgressDialog(title, "Cancel", 0, len(ids), parent)
    progress.setMinimumDuration(300)
    progress.setValue(0)
    state = {"cancel": False}
    progress.canceled.connect(lambda: state.update(cancel=True))

    def job(progress_cb=None, **_):
        out, errors = {}, []
        for k, i in enumerate(ids):
            if state["cancel"]:
                break
            try:
                out[i] = work(i)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
                errors.append(f"{i}: {exc}")
            if progress_cb:
                progress_cb(k + 1)
        return out, errors

    def done(res) -> None:  # noqa: ANN001
        progress.reset()
        out, errors = res
        for e in errors[:20]:
            session.warn(e)
        finish(out)

    def failed(tb: str) -> None:
        progress.reset()
        session.error(tb.strip().splitlines()[-1])

    run_job(lambda progress=None: job(progress), done=done, failed=failed,
            progress=lambda v: progress.setValue(int(v)))


def run_adapt(session: Session, parent: QWidget, ids: list[str], replace: bool = True, review: bool = True,
              on_done: Optional[Callable[[int], None]] = None) -> None:
    """Run the adapt pipeline on ``ids``.  With ``review`` the objects come in as proposals for the Review
    panel; without it they are accepted at once (still marked as made by the detector)."""
    params = session.project.settings.get(ADAPT_KEY)
    if not params:
        session.warn("Tune the adapt pipeline first (Detect → Tune adapt pipeline…)")
        return
    lab = params.get("labeling") if isinstance(params, dict) else None
    if isinstance(lab, dict) and lab.get("single_label") and lab["single_label"] not in session.project.labels:
        # a label of earlier work that this project no longer has: use the current label instead
        params = {**params, "labeling": {**lab, "single_label": session.current_label}}
    cfg = DetectionConfig.from_dict(params)
    detector = Detector(cfg)
    run = session.project.log_run("detect", "adapt pipeline", {"config": params, "digest": params_digest(params)}, ids)
    project = session.project
    default_label = session.current_label
    # images are loaded in the GUI thread (the project cache is not thread safe), detection runs in the worker
    inputs = {}
    for i in ids:
        try:
            iid = project.entry(i).id
            inputs[iid] = (project.processed_image(iid), project.entry(iid).display)
        except Exception as exc:  # noqa: BLE001
            session.warn(f"{i}: {exc}")

    def work(image_id: str) -> list[Annotation]:
        img, display = inputs[image_id]
        anns, det = propose(detector, img, image_id=image_id, display=display, run_id=run.id,
                            default_label=default_label)
        return anns

    def finish(out: dict) -> None:
        n = _add_results(session, out, "detector", replace, "Adapt pipeline " + ("proposals" if review else "results"),
                         accept=not review)
        run.summary = {"proposals": n, "images": len(out), "reviewed": review}
        if review:
            session.info(f"Adapt pipeline: {n} proposal(s) on {len(out)} image(s) — review them in the Review panel")
        else:
            session.info(f"Adapt pipeline: {n} object(s) on {len(out)} image(s)")
        if on_done is not None:
            on_done(n)

    run_in_background(session, parent, "Detecting…", list(inputs), work, finish)


class YoloDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detect with a YOLO model")
        self.session = session
        s = session.project.settings.get("yolo", {})
        self.weights = QLineEdit(s.get("weights", ""))
        browse = QPushButton("…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.weights, 1)
        row.addWidget(browse)
        w = QWidget()
        w.setLayout(row)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 1.0)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(float(s.get("conf", 0.25)))
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.05, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(float(s.get("iou", 0.5)))
        self.imgsz = QSpinBox()
        self.imgsz.setRange(0, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(int(s.get("imgsz", 0)))
        self.imgsz.setSpecialValueText("as trained")
        self.scope = QComboBox()
        self.scope.addItems(["Current image", "All images"])
        self.replace = QCheckBox("Replace unreviewed proposals of earlier model runs" if REVIEW
                                 else "Replace unedited results of earlier model runs")
        self.replace.setChecked(True)
        form = QFormLayout(self)
        form.addRow("Weights (.pt)", w)
        form.addRow("Confidence ≥", self.conf)
        form.addRow("Overlap (NMS IoU)", self.iou)
        form.addRow("Image size", self.imgsz)
        form.addRow("Images", self.scope)
        form.addRow(self.replace)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Model weights", os.path.dirname(self.weights.text()),
                                              "PyTorch weights (*.pt)")
        if path:
            self.weights.setText(path)

    def run(self) -> None:
        weights = self.weights.text().strip()
        if not os.path.isfile(weights):
            self.session.error(f"Weights not found: {weights}")
            return
        s = self.session
        s.project.settings["yolo"] = {"weights": weights, "conf": self.conf.value(), "iou": self.iou.value(),
                                      "imgsz": self.imgsz.value()}
        ids = [s.current_id] if self.scope.currentIndex() == 0 else [s.project.entry(i).id for i in s.image_ids()]
        run_yolo(s, self.parent() or self, ids, weights, self.conf.value(), self.iou.value(), self.imgsz.value() or None,
                 self.replace.isChecked())


def run_yolo(session: Session, parent: QWidget, ids: list[str], weights: str, conf: float = 0.25, iou: float = 0.5,
             imgsz: Optional[int] = None, replace: bool = True) -> None:
    from ...ml.predict import predict_annotations
    project = session.project
    run = project.log_run("predict", f"yolo:{os.path.basename(weights)}",
                          {"weights": weights, "conf": conf, "iou": iou, "imgsz": imgsz}, ids)
    inputs = {}
    for i in ids:
        try:
            inputs[i] = (project.processed_image(i), project.entry(i).display)
        except Exception as exc:  # noqa: BLE001
            session.warn(f"{i}: {exc}")

    def work(image_id: str) -> list[Annotation]:
        img, display = inputs[image_id]
        return predict_annotations(weights, img, display=display, image_id=image_id, conf=conf, iou=iou, imgsz=imgsz,
                                   run_id=run.id)

    def finish(out: dict) -> None:
        n = _add_results(session, out, "model", replace, "Model proposals")
        run.summary = {"proposals": n, "images": len(out)}
        session.info(f"{os.path.basename(weights)}: {n} {'proposal' if REVIEW else 'object'}(s) on {len(out)} image(s)")

    run_in_background(session, parent, "Running the model…", list(inputs), work, finish)
