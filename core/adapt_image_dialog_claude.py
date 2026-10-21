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

Standalone use
--------------
    python image_adapt_dialog.py [image_file]

Use inside ImageLabeling (replace your adaptImage_show / update_adapted_image)
------------------------------------------------------------------------------
    from core.image_adapt_dialog import ImageAdaptDialog

    def adaptImage_show(self):
        idx = self.ui.stackedWidget.currentIndex()
        image = imread(self.image_paths[idx], as_gray=True)
        dlg = ImageAdaptDialog(image, parent=self,
                               params=getattr(self, "adapted_image_params", None))
        if dlg.exec():                               # user pressed "Apply"
            self.adapted_image_params = dlg.get_params()   # JSON-serialisable dict
            self.adapted_binary = dlg.get_mask()           # bool ndarray (or None)

Dependencies: PyQt6, numpy, opencv-python, scikit-image
"""

from __future__ import annotations

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
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QToolButton,
    QVBoxLayout, QWidget,
)

# =============================================================================
# Image helpers
# =============================================================================


def _norm01(img: np.ndarray) -> np.ndarray:
    """Linearly rescale to [0, 1] (float32). Constant images become zeros."""
    img = np.nan_to_num(img.astype(np.float32, copy=False))
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _u16(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def to_float_gray(image: np.ndarray) -> np.ndarray:
    """Any numpy image (gray / RGB / RGBA, any dtype) -> float32 gray in [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        else:
            arr = arr[..., 0]
    return _norm01(arr)


def _numpy_to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """2D uint8 array -> QImage (deep copy, so the numpy buffer can be freed)."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    return QtGui.QImage(arr.data, w, h, arr.strides[0],
                        QtGui.QImage.Format.Format_Grayscale8).copy()


def _display_u8(img: np.ndarray) -> np.ndarray:
    """Bool -> black/white, float -> 0..255 gray."""
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    return _u8(img)


# =============================================================================
# Processing steps
#
# Every step function has the signature  func(image, **params) -> image
# where the parameter names match the `key` of the step's Param entries.
# "gray" steps get/return a float32 image in [0, 1]; "binary" steps get/return
# a bool image; the threshold step converts gray -> bool.
# =============================================================================

def f_line_align(img, method):
    ref = np.median(img, axis=1, keepdims=True) if method == "Median" \
        else img.mean(axis=1, keepdims=True)
    return _norm01(img - ref)


def f_background(img, sigma):
    return _norm01(img - cv2.GaussianBlur(img, (0, 0), float(sigma)))


def f_contrast(img, low, high):
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        return img
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def f_gamma(img, gamma):
    return np.power(np.clip(img, 0, 1), gamma).astype(np.float32)


def f_gaussian(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), float(sigma))


def f_median(img, size):
    if size <= 1:
        return img
    return cv2.medianBlur(_u8(img), int(size)).astype(np.float32) / 255.0


def f_unsharp(img, sigma, amount):
    if sigma <= 0 or amount == 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), float(sigma))
    return np.clip(img + amount * (img - blur), 0, 1).astype(np.float32)


def f_clahe(img, clip, tiles):
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    return clahe.apply(_u16(img)).astype(np.float32) / 65535.0


def f_invert(img):
    return (1.0 - img).astype(np.float32)


def f_threshold(img, method, features, value, block, offset):
    from skimage.filters import threshold_otsu, threshold_triangle

    # keep the window valid for small images
    block = int(min(block, (min(img.shape) // 2) * 2 - 1))
    block = max(block, 3)

    try:
        if method == "Otsu":
            t = threshold_otsu(img)
        elif method == "Triangle":
            t = threshold_triangle(img)
        elif method == "Manual":
            t = value
        elif method == "Adaptive (Gaussian)":
            t = cv2.GaussianBlur(img, (block, block), 0)
        else:  # "Adaptive (Mean)"
            t = cv2.blur(img, (block, block))
    except ValueError:  # e.g. constant image
        t = 0.5

    off = 0.0 if method == "Manual" else offset
    # positive offset = stricter in both modes
    if features == "Bright":
        return img > (t + off)
    return img < (t - off)


def f_morph(mask, op, size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
    code = {"Open": cv2.MORPH_OPEN, "Close": cv2.MORPH_CLOSE,
            "Erode": cv2.MORPH_ERODE, "Dilate": cv2.MORPH_DILATE}[op]
    out = cv2.morphologyEx(mask.astype(np.uint8), code, kernel, iterations=int(iterations))
    return out > 0


def f_size_filter(mask, min_size, max_size):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_size
    if max_size > 0:
        keep &= areas <= max_size
    keep[0] = False  # background
    return keep[labels]


def f_fill_holes(mask, max_area):
    h, w = mask.shape
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    touches_border = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    fill = (~touches_border) & (area <= max_area)
    fill[0] = False
    return mask | fill[labels]


def f_skeleton(mask):
    from skimage.morphology import skeletonize
    return skeletonize(mask)


# =============================================================================
# Step registry (this is where you add new steps / parameters)
# =============================================================================

@dataclass
class Param:
    key: str
    label: str
    kind: str                       # "float" | "int" | "odd" | "choice" | "bool"
    default: object
    minimum: float = 0
    maximum: float = 1
    step: float = 1
    decimals: int = 2
    choices: tuple = ()
    visible_if: Optional[Callable[[dict], bool]] = None
    tooltip: str = ""


@dataclass
class StepSpec:
    key: str
    title: str
    needs: str                      # "gray" or "binary"
    func: Callable
    params: list = field(default_factory=list)
    description: str = ""


_THRESH_METHODS = ("Otsu", "Triangle", "Manual", "Adaptive (Gaussian)", "Adaptive (Mean)")

STEP_REGISTRY: dict[str, StepSpec] = {s.key: s for s in [
    # ---- gray image steps ---------------------------------------------------
    StepSpec("line_align", "Line align", "gray", f_line_align, [
        Param("method", "Method", "choice", "Median", choices=("Median", "Mean")),
    ], "Subtract each scan row's median/mean (removes STM scan-line offsets)."),
    StepSpec("background", "Background subtraction", "gray", f_background, [
        Param("sigma", "Sigma (px)", "float", 30.0, 1, 200, 1, 0),
    ], "Subtract a heavily blurred copy to flatten slow background variations."),
    StepSpec("contrast", "Contrast stretch", "gray", f_contrast, [
        Param("low", "Low %", "float", 1.0, 0, 49, 0.5, 1),
        Param("high", "High %", "float", 99.0, 51, 100, 0.5, 1),
    ], "Clip to percentiles and rescale to the full range."),
    StepSpec("gamma", "Gamma", "gray", f_gamma, [
        Param("gamma", "Gamma", "float", 1.0, 0.1, 5.0, 0.05, 2),
    ], "Power-law intensity curve (<1 brightens, >1 darkens)."),
    StepSpec("gaussian", "Gaussian blur", "gray", f_gaussian, [
        Param("sigma", "Sigma (px)", "float", 1.0, 0, 10, 0.1, 1),
    ], "Smooth noise."),
    StepSpec("median", "Median blur", "gray", f_median, [
        Param("size", "Kernel size", "odd", 3, 1, 31, 2, 0),
    ], "Edge-preserving noise removal."),
    StepSpec("unsharp", "Unsharp mask", "gray", f_unsharp, [
        Param("sigma", "Sigma (px)", "float", 2.0, 0.1, 20, 0.1, 1),
        Param("amount", "Amount", "float", 1.0, 0, 5, 0.1, 1),
    ], "Sharpen edges."),
    StepSpec("clahe", "CLAHE", "gray", f_clahe, [
        Param("clip", "Clip limit", "float", 2.0, 1, 40, 0.5, 1),
        Param("tiles", "Tile grid", "int", 8, 2, 32, 1, 0),
    ], "Local adaptive histogram equalisation."),
    StepSpec("invert", "Invert", "gray", f_invert, [], "Flip bright and dark."),

    # ---- gray -> binary -----------------------------------------------------
    StepSpec("threshold", "Threshold", "gray", f_threshold, [
        Param("method", "Method", "choice", "Otsu", choices=_THRESH_METHODS),
        Param("features", "Features are", "choice", "Bright", choices=("Bright", "Dark")),
        Param("value", "Value", "float", 0.5, 0, 1, 0.01, 2,
              visible_if=lambda p: p["method"] == "Manual"),
        Param("block", "Block size", "odd", 31, 3, 301, 2, 0,
              visible_if=lambda p: p["method"].startswith("Adaptive")),
        Param("offset", "Offset", "float", 0.0, -0.5, 0.5, 0.005, 3,
              visible_if=lambda p: p["method"] != "Manual",
              tooltip="Positive = stricter (fewer pixels pass)."),
    ], "Turns the image into a binary mask."),

    # ---- binary steps -------------------------------------------------------
    StepSpec("morph", "Morphology", "binary", f_morph, [
        Param("op", "Operation", "choice", "Open", choices=("Open", "Close", "Erode", "Dilate")),
        Param("size", "Kernel size", "int", 3, 1, 25, 1, 0),
        Param("iterations", "Iterations", "int", 1, 1, 10, 1, 0),
    ], "Open removes specks, close fills gaps, erode/dilate shrink/grow."),
    StepSpec("size_filter", "Object size filter", "binary", f_size_filter, [
        Param("min_size", "Min area (px)", "int", 30, 0, 5000, 1, 0),
        Param("max_size", "Max area (px, 0=off)", "int", 0, 0, 100000, 10, 0),
    ], "Drop connected objects that are too small / too large."),
    StepSpec("fill_holes", "Fill holes", "binary", f_fill_holes, [
        Param("max_area", "Max hole area (px)", "int", 50, 0, 10000, 1, 0),
    ], "Fill enclosed background holes up to a given size."),
    StepSpec("skeleton", "Skeletonize", "binary", f_skeleton, [],
             "Thin every object to a 1 px wide centre line."),
]}

# Pipeline loaded on startup / "Reset": (step key, enabled, param overrides)
DEFAULT_PIPELINE = [
    ("gaussian", True, {"sigma": 1.0}),
    ("background", True, {"sigma": 40.0}),
    ("threshold", True, {"method": "Otsu"}),
    ("morph", False, {}),
    ("size_filter", True, {"min_size": 30}),
]


def run_pipeline(source: np.ndarray, steps: list) -> tuple[np.ndarray, dict]:
    """
    steps: list of (StepSpec, params dict, enabled bool)
    Returns (result image, {step_index: error message}).
    """
    cur = source
    errors: dict[int, str] = {}
    for i, (spec, params, enabled) in enumerate(steps):
        if not enabled:
            continue
        is_bin = cur.dtype == np.bool_
        if spec.needs == "binary" and not is_bin:
            errors[i] = "Needs a binary image: put a Threshold step above this one."
            continue
        if spec.needs == "gray" and is_bin:
            cur = cur.astype(np.float32)
        try:
            cur = spec.func(cur, **params)
        except Exception as exc:  # keep the UI alive, show the problem on the card
            errors[i] = f"{type(exc).__name__}: {exc}"
    return cur, errors


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
        self.status.setStyleSheet("color: #d9534f;")
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


# =============================================================================
# Dialog
# =============================================================================

PARAM_FILE_FORMAT = "image_adapt_pipeline"


class ImageAdaptDialog(QDialog):
    def __init__(self, image: np.ndarray, parent=None, params: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Image Adaptation")
        self.resize(960, 960)

        self._source = to_float_gray(image)
        self._result: np.ndarray = self._source
        self._cards: list[StepCard] = []

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
        self.info_label = QLabel()
        bar.addWidget(self.info_label)
        top_l.addLayout(bar)
        self.preview = ImagePreview()
        top_l.addWidget(self.preview, 1)

        # ---- bottom: step cards + buttons --------------------------------
        bottom = QWidget()
        bot_l = QVBoxLayout(bottom)
        bot_l.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        self.cards_layout = QVBoxLayout(container)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        bot_l.addWidget(scroll, 1)

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
        bot_l.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addStretch(1)
        cancel = QPushButton("Cancel")
        apply_ = QPushButton("Apply")
        for b in (cancel, apply_):
            b.setAutoDefault(False)
            row2.addWidget(b)
        cancel.clicked.connect(self.reject)
        apply_.clicked.connect(self.accept)
        bot_l.addLayout(row2)

        # ---- assemble ----------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 480])
        splitter.setChildrenCollapsible(False)
        root = QVBoxLayout(self)
        root.addWidget(splitter)

        if params:
            self.set_params(params)
        else:
            self.reset()

    # ------------------------------------------------------------------
    # Pipeline editing
    # ------------------------------------------------------------------
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
                "steps": [c.to_dict() for c in self._cards]}

    def set_params(self, data: dict) -> list[str]:
        """Replace the pipeline. Returns a list of warnings (unknown steps etc.)."""
        warnings: list[str] = []
        steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(steps, list):
            raise ValueError("Not a valid pipeline file (missing 'steps' list).")
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
        path, _ = QFileDialog.getSaveFileName(
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
    def get_result(self) -> np.ndarray:
        """Final image: bool array if a Threshold step is active, else float [0,1]."""
        return self._result

    def get_mask(self) -> Optional[np.ndarray]:
        """Final binary mask (bool), or None when no threshold step produced one."""
        return self._result if self._result.dtype == np.bool_ else None

    # ------------------------------------------------------------------
    # Processing + display
    # ------------------------------------------------------------------
    def _schedule(self):
        self._timer.start()

    def _recompute(self):
        steps = [(c.spec, c.values(), c.is_enabled()) for c in self._cards]
        self._result, errors = run_pipeline(self._source, steps)
        for i, card in enumerate(self._cards):
            card.set_error(errors.get(i))
        self._update_display()

    def _update_display(self):
        mode = self.view_combo.currentText()
        orig = _display_u8(self._source)
        res = _display_u8(self._result)
        if mode == "Original":
            shown = orig
        elif mode == "Side by side":
            gap = np.full((orig.shape[0], 6), 128, np.uint8)
            shown = np.hstack([orig, gap, res])
        else:
            shown = res
        self.preview.set_image(_numpy_to_qimage(shown))

        h, w = self._source.shape
        text = f"{w} × {h} px"
        if self._result.dtype == np.bool_:
            n_obj = cv2.connectedComponents(self._result.astype(np.uint8), connectivity=8)[0] - 1
            text += f"   |   foreground {self._result.mean() * 100:.2f} %   |   {n_obj} objects"
        else:
            text += "   |   no threshold step active (gray result)"
        self.info_label.setText(text)


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