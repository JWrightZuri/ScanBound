"""
image_adapt_dialog.py
=====================

A PyQt6 dialog that shows an image and, below it, a customizable *pipeline* of
processing steps (each with its own sliders / combo boxes / checkboxes).
Every change re-runs the pipeline and updates the preview in real time, so you
can tune the steps until thresholding leaves only the important features as a
clean binary image.

Features
--------
* Add any step (several of the same kind are allowed) from the "+ Add step" menu.
* Remove, reorder (up / down), collapse and enable / disable every step.
* Steps are applied top to bottom. Gray steps work on a float image, "binary"
  steps (morphology, size filter, ...) must come after a Threshold step.
* Save / load the whole pipeline (steps, order, enabled flags, values) as JSON.
* View the result, the original, or both side by side.
* "Show distinct objects": every connected object of the binary result gets its own
  colour and ID, with the exact ROI (rectangle / polygon / centre line) that
  ImageLabeling.auto_segment_to_rois will generate drawn on top, plus the vertex
  count.  The ROI options (padding, simplification, vertex cap ...) are saved with
  the pipeline under the "roi" key.

Standalone use
--------------
    python image_adapt_dialog.py [image_file]

Use inside ImageLabeling (replace your adaptImage_show / update_adapted_image)
------------------------------------------------------------------------------
    from .image_adapt_dialog import ImageAdaptDialog

    def adaptImage_show(self):
        idx = self.ui.stackedWidget.currentIndex()
        image = imread(self.image_paths[idx], as_gray=True)
        dlg = ImageAdaptDialog(image, parent=self,
                               params=getattr(self, "adapted_image_params", None))
        if dlg.exec():                               # user pressed "Apply"
            self.adapted_image_params = dlg.get_params()   # JSON-serialisable dict
            self.adapted_binary = dlg.get_mask()           # bool ndarray (or None)
            self.adapted_objects = dlg.get_objects()       # list[SegObject] (ROI geometry)
            self.roi_settings = dlg.get_roi_settings()     # used by auto_segment_to_rois

Needs roi_extraction.py next to this file (or in core/).
Dependencies: PyQt6, numpy, opencv-python, scipy, scikit-image
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QToolButton,
    QVBoxLayout, QWidget,
)

from scanbound.roi import ROI_TYPES, RoiSettings, SegObject, mask_components, extract_objects  # noqa: F401
from . import _ol as ol
from .label_assignment_panel import LabelAssignmentPanel

MAX_PREVIEW_OBJECTS = 1500              # above this only the colouring is drawn, no outlines

# =============================================================================
# Image helpers, processing steps, step registry and pipeline runner
#
# These live in the UI-independent ``scanbound`` package; the names below are kept so existing
# imports from this module still work.  Add new steps in scanbound/src/scanbound/steps/adapt.py.
# =============================================================================

from scanbound._image import display_u8 as _display_u8, norm01 as _norm01, to_float_gray  # noqa: E402,F401
from scanbound._image import to_u8 as _u8, to_u16 as _u16  # noqa: E402,F401
from scanbound.pipeline import run_pipeline  # noqa: E402,F401
from scanbound.render import _hex_rgb, object_palette as _object_palette, render_objects  # noqa: E402,F401
from scanbound.steps.adapt import (  # noqa: E402,F401
    ADAPT_STEPS as STEP_REGISTRY, DEFAULT_ADAPT_PIPELINE as DEFAULT_PIPELINE,
    THRESHOLD_METHODS as _THRESH_METHODS, f_background, f_clahe, f_contrast, f_fill_holes, f_gamma,
    f_gaussian, f_invert, f_line_align, f_median, f_morph, f_rescale, f_size_filter, f_skeleton,
    f_threshold, f_unsharp, smooth_background as _smooth_background,
)
from scanbound.steps.registry import Param, StepSpec  # noqa: E402,F401



def _numpy_to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """2D (gray) or HxWx3 (RGB) uint8 array -> QImage (deep copy)."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    fmt = (QtGui.QImage.Format.Format_Grayscale8 if arr.ndim == 2
           else QtGui.QImage.Format.Format_RGB888)
    return QtGui.QImage(arr.data, w, h, arr.strides[0], fmt).copy()




# =============================================================================
# Widgets
# =============================================================================

class ParamControl(QWidget):
    """One parameter editor: slider + spin box, combo box, or checkbox."""
    changed = pyqtSignal()

    def __init__(self, p: Param, parent=None):
        super().__init__(parent)
        self.p = p
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = self.spin = self.widget = None

        if p.kind == "choice":
            self.widget = QComboBox()
            self.widget.addItems(list(p.choices))
            self.widget.setCurrentText(str(p.default))
            self.widget.currentIndexChanged.connect(lambda _: self.changed.emit())
            lay.addWidget(self.widget)
        elif p.kind == "bool":
            self.widget = QCheckBox()
            self.widget.setChecked(bool(p.default))
            self.widget.toggled.connect(lambda _: self.changed.emit())
            lay.addWidget(self.widget)
            lay.addStretch(1)
        else:
            is_float = p.kind == "float"
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, int(round((p.maximum - p.minimum) / p.step)))
            self.spin = QDoubleSpinBox() if is_float else QSpinBox()
            if is_float:
                self.spin.setDecimals(p.decimals)
            self.spin.setRange(p.minimum, p.maximum)
            self.spin.setSingleStep(p.step)
            self.spin.setKeyboardTracking(False)   # only fire on Enter / focus out
            self.spin.setFixedWidth(82)
            lay.addWidget(self.slider, 1)
            lay.addWidget(self.spin)
            self.set_value(p.default)
            self.slider.valueChanged.connect(self._on_slider)
            self.spin.valueChanged.connect(self._on_spin)
        if p.tooltip:
            self.setToolTip(p.tooltip)

    # -- helpers ---------------------------------------------------------
    def _snap(self, v):
        p = self.p
        i = int(round((float(v) - p.minimum) / p.step))
        i = max(0, min(i, int(round((p.maximum - p.minimum) / p.step))))
        v = p.minimum + i * p.step
        return float(v) if p.kind == "float" else int(v)

    def _on_slider(self, i):
        v = self._snap(self.p.minimum + i * self.p.step)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.changed.emit()

    def _on_spin(self, v):
        v = self._snap(v)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((v - self.p.minimum) / self.p.step)))
        self.slider.blockSignals(False)
        self.changed.emit()

    # -- public ----------------------------------------------------------
    def value(self):
        if self.p.kind == "choice":
            return self.widget.currentText()
        if self.p.kind == "bool":
            return self.widget.isChecked()
        return self.spin.value()

    def set_value(self, v):
        """Set without emitting `changed`."""
        if self.p.kind == "choice":
            if str(v) in self.p.choices:
                self.widget.blockSignals(True)
                self.widget.setCurrentText(str(v))
                self.widget.blockSignals(False)
        elif self.p.kind == "bool":
            self.widget.blockSignals(True)
            self.widget.setChecked(bool(v))
            self.widget.blockSignals(False)
        else:
            v = self._snap(v)
            self.spin.blockSignals(True)
            self.slider.blockSignals(True)
            self.spin.setValue(v)
            self.slider.setValue(int(round((v - self.p.minimum) / self.p.step)))
            self.spin.blockSignals(False)
            self.slider.blockSignals(False)


class StepCard(QFrame):
    """One pipeline step: header (enable, reorder, remove) + parameter rows."""
    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)
    move_requested = pyqtSignal(object, int)

    def __init__(self, spec: StepSpec, params: Optional[dict] = None,
                 enabled: bool = True, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 6)
        outer.setSpacing(4)

        # header
        header = QHBoxLayout()
        self.enable_cb = QCheckBox(spec.title)
        font = self.enable_cb.font()
        font.setBold(True)
        self.enable_cb.setFont(font)
        self.enable_cb.setChecked(enabled)
        self.enable_cb.setToolTip(spec.description or spec.title)
        self.enable_cb.toggled.connect(self._on_enabled)
        header.addWidget(self.enable_cb)
        header.addStretch(1)

        def tool_btn(text, tip, slot):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            return b

        self.collapse_btn = tool_btn("▾", "Collapse / expand", self._toggle_collapse)
        header.addWidget(self.collapse_btn)
        header.addWidget(tool_btn("▲", "Move up", lambda: self.move_requested.emit(self, -1)))
        header.addWidget(tool_btn("▼", "Move down", lambda: self.move_requested.emit(self, +1)))
        header.addWidget(tool_btn("✕", "Remove step", lambda: self.remove_requested.emit(self)))
        outer.addLayout(header)

        # body
        self.body = QWidget()
        grid = QGridLayout(self.body)
        grid.setContentsMargins(18, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)
        self.rows: dict[str, tuple[QLabel, ParamControl]] = {}
        for r, p in enumerate(spec.params):
            label = QLabel(p.label)
            ctrl = ParamControl(p)
            if params and p.key in params:
                ctrl.set_value(params[p.key])
            ctrl.changed.connect(self._on_param)
            grid.addWidget(label, r, 0)
            grid.addWidget(ctrl, r, 1)
            self.rows[p.key] = (label, ctrl)
        outer.addWidget(self.body)
        if not spec.params:
            self.body.hide()
            self.collapse_btn.hide()

        self.status = QLabel()
        self.status.setWordWrap(True)
        mark(self.status, "error")
        self.status.hide()
        outer.addWidget(self.status)

        self._update_visibility()
        self._on_enabled(enabled, emit=False)

    # -- state -----------------------------------------------------------
    def values(self) -> dict:
        return {k: c.value() for k, (_, c) in self.rows.items()}

    def is_enabled(self) -> bool:
        return self.enable_cb.isChecked()

    def set_error(self, msg: Optional[str]):
        self.status.setText(msg or "")
        self.status.setVisible(bool(msg))

    def to_dict(self) -> dict:
        return {"type": self.spec.key, "enabled": self.is_enabled(), "params": self.values()}

    # -- slots -----------------------------------------------------------
    def _update_visibility(self):
        vals = self.values()
        for p in self.spec.params:
            if p.visible_if is not None:
                show = bool(p.visible_if(vals))
                label, ctrl = self.rows[p.key]
                label.setVisible(show)
                ctrl.setVisible(show)

    def _on_param(self):
        self._update_visibility()
        self.changed.emit()

    def _on_enabled(self, checked, emit=True):
        self.body.setEnabled(checked)
        if not checked:
            self.set_error(None)
        if emit:
            self.changed.emit()

    def _toggle_collapse(self):
        visible = not self.body.isVisible()
        self.body.setVisible(visible)
        self.collapse_btn.setText("▾" if visible else "▸")


class ImagePreview(QLabel):
    """Label that keeps its image scaled to the available space."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._qimg: Optional[QtGui.QImage] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background-color: #1e1e1e;")

    def set_image(self, qimg: QtGui.QImage):
        self._qimg = qimg
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._qimg is None or self.width() < 4 or self.height() < 4:
            return
        upscaling = self._qimg.width() < self.width() and self._qimg.height() < self.height()
        mode = (Qt.TransformationMode.FastTransformation if upscaling
                else Qt.TransformationMode.SmoothTransformation)
        pm = QtGui.QPixmap.fromImage(self._qimg).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, mode)
        self.setPixmap(pm)


from scanbound.roi import ROI_PARAMS  # noqa: E402
from ..theme import mark  # noqa: E402
from ..qtutil import save_file_name

OBJECT_PARAMS = list(ROI_PARAMS)        # ROI editor rows (defined next to RoiSettings)


class ObjectOptions(QGroupBox):
    """Checkable group: show distinct objects + settings of the ROIs generated for them."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Show distinct objects (ROI preview)", parent)
        self.setCheckable(True)
        self.setChecked(True)
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)
        self.rows: dict[str, tuple[QLabel, ParamControl]] = {}
        for r, p in enumerate(OBJECT_PARAMS):
            label, ctrl = QLabel(p.label), ParamControl(p)
            ctrl.changed.connect(self._on_param)
            grid.addWidget(label, r, 0)
            grid.addWidget(ctrl, r, 1)
            self.rows[p.key] = (label, ctrl)
        self.toggled.connect(lambda _: self.changed.emit())
        self._update_visibility()

    def _update_visibility(self):
        vals = self.values()
        for p in OBJECT_PARAMS:
            if p.visible_if is not None:
                show = bool(p.visible_if(vals))
                label, ctrl = self.rows[p.key]
                label.setVisible(show)
                ctrl.setVisible(show)

    def _on_param(self):
        self._update_visibility()
        self.changed.emit()

    def values(self) -> dict:
        return {k: c.value() for k, (_, c) in self.rows.items()}

    def settings(self) -> RoiSettings:
        return RoiSettings.from_dict(self.values())

    def to_dict(self) -> dict:
        return {"enabled": self.isChecked(), **self.values()}

    def set_dict(self, data: dict):
        self.blockSignals(True)
        self.setChecked(bool(data.get("enabled", True)))
        self.blockSignals(False)
        for k, (_, c) in self.rows.items():
            if k in data:
                c.set_value(data[k])
        self._update_visibility()


# =============================================================================
# Dialog
# =============================================================================

PARAM_FILE_FORMAT = "image_adapt_pipeline"


class ImageAdaptDialog(QDialog):
    def __init__(self, image: np.ndarray, parent=None, params: Optional[dict] = None,
                 labels: Optional[list] = None, label_colors: Optional[dict] = None,
                 px_scale: Optional[float] = None, current_label: Optional[str] = None):
        """
        labels / label_colors / current_label: label list of the labelling page (for the
        "Labels" tab); px_scale: nm per pixel of the image (None = measures in pixels).
        """
        super().__init__(parent)
        self.setWindowTitle("Image Adaptation")
        self.resize(1000, 1000)
        self._fitted = False

        self._source = to_float_gray(image)
        self._result: np.ndarray = self._source
        self._cards: list[StepCard] = []
        self._objects: Optional[list] = None      # SegObjects of the current result
        self._features = None                     # measurements of self._objects (DataFrame)
        self._step_cache: list = []               # per-step outputs for incremental re-runs
        self._objects_key = None                  # (mask content, ROI settings) the objects belong to
        self._extract_cache: dict = {}            # per-object ROI geometry, reused while unchanged
        self._components = None                   # connected components of the current mask
        self._dim_base = None                     # dimmed RGB original for the object preview
        self._px_scale = px_scale if px_scale and px_scale > 0 else None
        self._n_components = 0
        self._obj_msg = ""

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)      # debounce: coalesce rapid slider moves
        self._timer.timeout.connect(self._recompute)

        # ---- top: preview ------------------------------------------------
        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Result", "Original", "Side by side"])
        self.view_combo.currentIndexChanged.connect(self._update_display)
        bar.addWidget(self.view_combo)
        bar.addStretch(1)
        self.big_btn = QToolButton()
        self.big_btn.setText("⤢ Large preview")
        self.big_btn.setCheckable(True)
        self.big_btn.setToolTip("Give the preview almost the whole window (click again to restore)")
        self.big_btn.toggled.connect(self._toggle_large_preview)
        bar.addWidget(self.big_btn)
        top_l.addLayout(bar)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        top_l.addWidget(self.info_label)
        self.preview = ImagePreview()
        top_l.addWidget(self.preview, 1)

        # ---- bottom: step cards + buttons --------------------------------
        bottom = QWidget()
        bot_l = QVBoxLayout(bottom)
        bot_l.setContentsMargins(0, 0, 0, 0)

        self.obj_opts = ObjectOptions()
        self.obj_opts.changed.connect(self._schedule)
        bot_l.addWidget(self.obj_opts)

        container = QWidget()
        self.cards_layout = QVBoxLayout(container)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

        pipeline_tab = QWidget()
        pipe_l = QVBoxLayout(pipeline_tab)
        pipe_l.setContentsMargins(0, 0, 0, 0)
        pipe_l.addWidget(scroll, 1)

        self.label_panel = LabelAssignmentPanel(labels or [], label_colors or {}, self._px_scale, current_label)
        self.label_panel.assignmentChanged.connect(self._update_display)
        self.label_panel.featuresNeeded.connect(self._update_features)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(pipeline_tab, "Processing pipeline")
        self.tabs.addTab(self.label_panel, "Labels")
        self.tabs.currentChanged.connect(self._on_tab)
        bot_l.addWidget(self.tabs, 1)

        row1 = QHBoxLayout()
        self.add_btn = QToolButton()
        self.add_btn.setText("＋ Add step")
        self.add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.add_btn)
        for key, spec in STEP_REGISTRY.items():
            act = menu.addAction(spec.title)
            act.setToolTip(spec.description)
            act.triggered.connect(lambda _=False, k=key: self.add_step(k))
        self.add_btn.setMenu(menu)
        row1.addWidget(self.add_btn)
        row1.addStretch(1)
        for text, slot in (("Load…", self._load_dialog), ("Save…", self._save_dialog),
                           ("Reset", self.reset)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            row1.addWidget(b)
        pipe_l.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addStretch(1)
        cancel = QPushButton("Cancel")
        apply_ = QPushButton("Apply")
        for b in (cancel, apply_):
            b.setAutoDefault(False)
            row2.addWidget(b)
        cancel.clicked.connect(self.reject)
        apply_.clicked.connect(self.accept)

        # ---- assemble ----------------------------------------------------
        # The controls sit in a scroll area so the preview can be dragged almost to the full
        # window height (the controls then scroll instead of limiting the splitter).
        bottom_scroll = QScrollArea()
        bottom_scroll.setWidgetResizable(True)
        bottom_scroll.setFrameShape(QFrame.Shape.NoFrame)
        bottom_scroll.setWidget(bottom)
        bottom_scroll.setMinimumHeight(60)
        bottom.setMinimumHeight(420)
        self.splitter = splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 440])
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.handle(1).setToolTip("Drag to resize the preview")
        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)
        root.addLayout(row2)

        if params:
            self.set_params(params)
        else:
            self.reset()

    # ------------------------------------------------------------------
    # Pipeline editing
    # ------------------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        if not self._fitted:                  # first show: fit the screen it opens on, centred
            self._fitted = True
            screen = self.screen() or QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                frame = self.frameGeometry().size() - self.size()     # title bar and borders
                w = min(self.width(), int(avail.width() * 0.9) - frame.width())
                h = min(self.height(), int(avail.height() * 0.9) - frame.height())
                self.resize(max(w, self.minimumSizeHint().width()), max(h, 300))
                g = self.frameGeometry()
                g.moveCenter(avail.center())
                self.move(g.topLeft())
            self.splitter.setSizes([int(self.splitter.height() * 0.55), int(self.splitter.height() * 0.45)])

    def add_step(self, key: str, params: Optional[dict] = None, enabled: bool = True,
                 schedule: bool = True) -> StepCard:
        card = StepCard(STEP_REGISTRY[key], params, enabled)
        card.changed.connect(self._schedule)
        card.remove_requested.connect(self._remove_card)
        card.move_requested.connect(self._move_card)
        self.cards_layout.insertWidget(len(self._cards), card)
        self._cards.append(card)
        if schedule:
            self._schedule()
        return card

    def _remove_card(self, card: StepCard):
        self._cards.remove(card)
        self._dispose_card(card)
        self._schedule()

    def _move_card(self, card: StepCard, delta: int):
        i = self._cards.index(card)
        j = i + delta
        if not 0 <= j < len(self._cards):
            return
        self._cards.insert(j, self._cards.pop(i))
        self.cards_layout.removeWidget(card)
        self.cards_layout.insertWidget(j, card)
        self._schedule()

    def clear_steps(self):
        for card in self._cards:
            self._dispose_card(card)
        self._cards.clear()

    def _dispose_card(self, card: StepCard):
        self.cards_layout.removeWidget(card)
        card.hide()
        card.setParent(None)
        card.deleteLater()

    def reset(self):
        self.obj_opts.set_dict({"enabled": True, **RoiSettings().to_dict()})
        self.clear_steps()
        for key, enabled, params in DEFAULT_PIPELINE:
            self.add_step(key, params, enabled, schedule=False)
        self._schedule()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        """JSON-serialisable description of the whole pipeline."""
        return {"format": PARAM_FILE_FORMAT, "version": 1,
                "steps": [c.to_dict() for c in self._cards],
                "roi": self.obj_opts.to_dict(),
                "labeling": self.label_panel.config()}

    def set_params(self, data: dict) -> list[str]:
        """Replace the pipeline. Returns a list of warnings (unknown steps etc.)."""
        warnings: list[str] = []
        steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(steps, list):
            raise ValueError("Not a valid pipeline file (missing 'steps' list).")
        if isinstance(data.get("roi"), dict):
            self.obj_opts.set_dict(data["roi"])
        if isinstance(data.get("labeling"), dict):
            self.label_panel.set_config(data["labeling"])
        self.clear_steps()
        for entry in steps:
            key = entry.get("type") if isinstance(entry, dict) else None
            if key not in STEP_REGISTRY:
                warnings.append(f"Skipped unknown step type: {key!r}")
                continue
            self.add_step(key, entry.get("params") or {}, bool(entry.get("enabled", True)),
                          schedule=False)
        self._schedule()
        return warnings

    def save_params(self, path: str):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.get_params(), fh, indent=2)

    def load_params(self, path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as fh:
            return self.set_params(json.load(fh))

    def _save_dialog(self):
        path = save_file_name(
            self, "Save parameters", "image_adapt_params.json", "Parameter files (*.json)")
        if not path:
            return
        try:
            self.save_params(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load parameters", "", "Parameter files (*.json);;All files (*)")
        if not path:
            return
        try:
            warnings = self.load_params(path)
        except (OSError, ValueError) as exc:    # JSONDecodeError is a ValueError
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        if warnings:
            QMessageBox.warning(self, "Loaded with warnings", "\n".join(warnings))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _flush(self):
        """Run a pending (debounced) recompute now, so results are never stale."""
        if self._timer.isActive():
            self._timer.stop()
            self._recompute()

    def accept(self):
        self._flush()
        super().accept()

    def get_result(self) -> np.ndarray:
        """Final image: bool array if a Threshold step is active, else float [0,1]."""
        self._flush()
        return self._result

    def get_mask(self) -> Optional[np.ndarray]:
        """Final binary mask (bool), or None when no threshold step produced one."""
        self._flush()
        return self._result if self._result.dtype == np.bool_ else None

    def get_roi_settings(self) -> RoiSettings:
        """ROI generation settings currently chosen in the dialog."""
        return self.obj_opts.settings()

    def get_objects(self) -> list:
        """SegObjects (one per distinct object) for the current mask and ROI settings."""
        mask = self.get_mask()
        return extract_objects(mask, self.get_roi_settings()) if mask is not None else []

    def get_labeling(self) -> dict:
        """Label assignment configuration chosen in the "Labels" tab."""
        return self.label_panel.config()

    def get_labeled_objects(self) -> list:
        """
        [(SegObject, label)] for the current mask, ROI settings and label assignment.
        Objects whose group is set to "Skip" are left out.
        """
        mask = self.get_mask()
        if mask is None:
            return []
        objects = self.get_objects()
        config = self.get_labeling()
        if config.get("mode", "single") == "single":
            label = config.get("single_label")
            return [(o, label) for o in objects] if label is not None else []
        features = ol.measure_objects(mask, objects, self._px_scale, self._source)
        labels = ol.assign_labels(features, config)
        return [(o, labels.get(o.id)) for o in objects if labels.get(o.id) is not None]

    # ------------------------------------------------------------------
    # Processing + display
    # ------------------------------------------------------------------
    def _schedule(self):
        self._timer.start()

    def _recompute(self):
        steps = [(c.spec, c.values(), c.is_enabled()) for c in self._cards]
        # unchanged leading steps are taken from the cache (tuning a late step is instant)
        self._result, errors = run_pipeline(self._source, steps, self._step_cache)
        for i, card in enumerate(self._cards):
            card.set_error(errors.get(i))

        settings = self.obj_opts.settings()
        is_bin = self._result.dtype == np.bool_
        content = (hashlib.blake2b(np.packbits(self._result).tobytes(), digest_size=16).digest()
                   if is_bin else id(self._result))
        key = (content, self._result.shape, self.obj_opts.isChecked(), tuple(sorted(settings.to_dict().items())))
        if key == self._objects_key:
            self._update_display()          # same mask and ROI settings: objects are still valid
            return
        mask_changed = self._objects_key is None or self._objects_key[:2] != key[:2]
        self._objects_key = key
        self._objects, self._obj_msg = None, ""
        self._features = None
        if is_bin:
            if mask_changed or self._components is None:
                self._components = mask_components(self._result)
            self._n_components = self._components[0] - 1
            if self.obj_opts.isChecked():
                if self._n_components > MAX_PREVIEW_OBJECTS:
                    self._obj_msg = (f"{self._n_components} objects: too many to outline, "
                                     "tighten the pipeline (size filter / threshold).")
                else:
                    try:
                        self._objects = extract_objects(self._result, settings, cache=self._extract_cache,
                                                        components=self._components)
                    except Exception as exc:
                        self._obj_msg = f"ROI generation failed: {type(exc).__name__}: {exc}"
        else:
            self._components, self._n_components = None, 0
        self._update_features()

    def _toggle_large_preview(self, on):
        total = sum(self.splitter.sizes())
        if on:
            self._sizes_before = self.splitter.sizes()
            self.splitter.setSizes([total - 70, 70])
        else:
            self.splitter.setSizes(getattr(self, "_sizes_before", [total // 2, total - total // 2]))

    def _on_tab(self, _index=None):
        if self.tabs.currentWidget() is self.label_panel and self._features is None:
            self._update_features()

    def _update_features(self):
        """Measure the objects (only when the label assignment needs it) and re-assign labels."""
        needed = self._objects and (self.label_panel.mode_key() != "single"
                                    or self.tabs.currentWidget() is self.label_panel)
        if needed and self._features is None:
            try:
                self._features = ol.measure_objects(self._result, self._objects, self._px_scale, self._source)
            except Exception as exc:
                # report instead of handing the panel a stand-in (it would ask for features again)
                self._obj_msg = f"Measuring objects failed: {type(exc).__name__}: {exc}"
                self.label_panel.set_features(None)
                return
        elif not needed:
            self._features = None
        if self._features is None and self._objects:
            # cheap stand-in so the single-label preview and counts still work
            import pandas as pd
            self._features = None
            self.label_panel.features = pd.DataFrame(index=[o.id for o in self._objects])
            self.label_panel.recompute()
            return
        self.label_panel.set_features(self._features)      # emits assignmentChanged -> display

    def _update_display(self):
        mode = self.view_combo.currentText()
        is_bin = self._result.dtype == np.bool_
        show_objs = is_bin and self.obj_opts.isChecked()

        orig = _display_u8(self._source)
        if show_objs:
            colors = None
            assignment = self.label_panel.assignment
            if self._objects and assignment:
                colors = {oid: (None if lab is None else self.label_panel.color(lab))
                          for oid, lab in assignment.items()}
            if self._dim_base is None:
                self._dim_base = np.repeat((_u8(self._source) * 0.35).astype(np.uint8)[..., None], 3, axis=2)
            res = render_objects(self._source, self._result, self._objects, colors,
                                 components=self._components, base=self._dim_base)
        else:
            res = _display_u8(self._result)

        if mode == "Original":
            shown = orig
        elif mode == "Side by side":
            if res.ndim == 3:
                orig = np.repeat(orig[..., None], 3, axis=2)
            gap = np.full((orig.shape[0], 6) + orig.shape[2:], 128, np.uint8)
            shown = np.hstack([orig, gap, res])
        else:
            shown = res
        self.preview.set_image(_numpy_to_qimage(shown))

        h, w = self._source.shape
        parts = [f"{w} × {h} px"]
        if is_bin:
            parts.append(f"foreground {self._result.mean() * 100:.2f} %")
            parts.append(f"{self._n_components} objects")
            if self._objects is not None and self._objects:
                n_v = sum(o.n_vertices for o in self._objects)
                kind = self._objects[0].roi_type
                if kind in ("RectROI", "PointROI"):
                    parts.append(f"{len(self._objects)} {kind}s")
                elif kind == "MaskROI":
                    parts.append(f"{len(self._objects)} MaskROIs, {sum(o.area for o in self._objects)} px")
                else:
                    parts.append(f"{kind}: {n_v} vertices total, "
                                 f"{n_v / len(self._objects):.1f} per object")
            if self._obj_msg:
                parts.append(self._obj_msg)
        else:
            parts.append("no threshold step active (gray result)")
        self.info_label.setText("   |   ".join(parts))


# =============================================================================
# Standalone entry point
# =============================================================================

def _demo_image() -> np.ndarray:
    """Noisy rods on a tilted background, handy for trying the tool."""
    rng = np.random.default_rng(0)
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w]
    img = 0.3 * (xx / w) + 0.2 * (yy / h)
    layer = np.zeros((h, w), np.float32)
    for _ in range(25):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        ang = rng.uniform(0, np.pi)
        ln = rng.integers(30, 90)
        p2 = (int(p1[0] + ln * np.cos(ang)), int(p1[1] + ln * np.sin(ang)))
        cv2.line(layer, p1, p2, 1.0, int(rng.integers(3, 6)))
    img = img + 0.35 * cv2.GaussianBlur(layer, (0, 0), 1.2) + rng.normal(0, 0.08, (h, w))
    return img.astype(np.float32)


def main():
    app = QApplication(sys.argv)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        path, _ = QFileDialog.getOpenFileName(
            None, "Select an image (Cancel = demo image)", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)")
    if path:
        from skimage.io import imread
        image = imread(path, as_gray=True)
    else:
        image = _demo_image()

    dlg = ImageAdaptDialog(image)
    if dlg.exec():
        print("Applied parameters:")
        print(json.dumps(dlg.get_params(), indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()