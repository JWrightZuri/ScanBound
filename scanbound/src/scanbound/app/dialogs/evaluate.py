"""Evaluate a detector run against ground truth: precision, recall, F1 and AP per class."""

from __future__ import annotations

import math

from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit,
                             QPushButton, QVBoxLayout)

from ...annotations import GROUND_TRUTH
from ...ml.evaluate import evaluate, review_stats
from ..features import REVIEW
from ..session import Session
from ..theme import mark


class EvaluateDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Evaluate a detector")
        self.resize(760, 520)
        self.session = session
        self.run = QComboBox()
        runs = [r for r in session.project.runs if r.kind in ("detect", "predict")]
        for r in reversed(runs):
            n = sum(1 for a in session.store if a.origin.run == r.id)
            self.run.addItem(f"{r.name} — {n} result(s) — {r.id}", r.id)
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.05, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.5)
        self.mode = QComboBox()
        self.mode.addItems(["box", "mask"])
        self.mode.setToolTip("box: overlap of bounding boxes; mask: overlap of the outlines (slower, stricter)")
        go = QPushButton("Evaluate")
        go.clicked.connect(self.evaluate)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        note = QLabel("Ground truth: manual annotations and accepted proposals on the images of the run. Proposals that "
                      "were accepted unchanged count as correct, so for an unbiased estimate evaluate on images that "
                      "were annotated by hand. Rejected proposals count as false positives.")
        note.setWordWrap(True)
        mark(note, "muted")
        form = QFormLayout()
        form.addRow("Run", self.run)
        form.addRow("IoU threshold", self.iou)
        form.addRow("Overlap of", self.mode)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(go)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addLayout(row)
        lay.addWidget(self.out, 1)

    def evaluate(self) -> None:
        run_id = self.run.currentData()
        if not run_id:
            return
        store = self.session.store
        preds = [a for a in store if a.origin.run == run_id and a.kind == "feature"]
        images = {a.image_id for a in preds}
        run = self.session.project.run(run_id)
        if run is not None:
            images |= {self.session.project.entry(i).id for i in run.images if i}
        gt = [a for a in store if a.image_id in images and a.kind == "feature" and a.status in GROUND_TRUTH
              and not (a.origin.run == run_id and (a.status != "accepted" or (not REVIEW and not a.modified)))]
        ev = evaluate(gt, preds, self.iou.value(), self.mode.currentText())
        rs = review_stats(store, run_id)
        text = ev.table()
        if REVIEW:
            text += (f"\n\nReview of this run: {rs['accepted']} accepted unchanged, {rs['accepted_edited']} corrected, "
                     f"{rs['rejected']} rejected, {rs['pending']} not reviewed")
            if not math.isnan(rs["acceptance_rate"]):
                text += f" (acceptance {100 * rs['acceptance_rate']:.0f} %)"
            if rs["pending"]:
                text += "\nUnreviewed proposals count as predictions but their correctness is not known yet."
        else:
            text += "\n\nResults of this run you have not edited are not counted as ground truth."
        self.out.setPlainText(text)
        self.session.info(f"Evaluation of run {run_id}: mAP@{self.iou.value():g} = {ev.map:.3f}")
