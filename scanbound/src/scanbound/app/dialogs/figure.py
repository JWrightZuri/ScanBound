"""Publication figure dialog: style the current image for print, preview it, save it or copy it."""

from __future__ import annotations

import json
import os
from typing import Optional

from PyQt6.QtCore import QByteArray, QMimeData, QSettings, Qt, QTimer
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ...annotations import GROUND_TRUTH
from ...publication import COLORBARS, POSITIONS, WIDTHS_MM, FigureStyle, figure_png, figure_svg, render_figure, \
    save_figure
from ..session import Session
from ..theme import mark

SETTINGS_KEY = "figure_style"
PREVIEW_PX = 520                                  # preview width on screen


def saved_style() -> FigureStyle:
    raw = QSettings("ScanBound", "ScanBound").value(SETTINGS_KEY, "")
    try:
        return FigureStyle.from_dict(json.loads(raw) if raw else None)
    except (TypeError, ValueError):
        return FigureStyle()


def remember_style(style: FigureStyle) -> None:
    d = style.to_dict()
    d["crop"] = None                              # the view is per image, not a preference
    QSettings("ScanBound", "ScanBound").setValue(SETTINGS_KEY, json.dumps(d))


def build_figure(session: Session, style: FigureStyle):  # noqa: ANN201 - a matplotlib Figure
    img = session.image()
    if img is None:
        raise ValueError("no image is open")
    anns = [a for a in session.annotations() if a.status in GROUND_TRUTH] if style.annotations else []
    colors = {name: session.label_color(name) for name in session.project.labels.names}
    return render_figure(img, session.entry.display, style, annotations=anns, label_colors=colors)


def copy_figure(session: Session, style: Optional[FigureStyle] = None) -> None:
    """Put the figure on the clipboard: a PNG at the style's resolution, plus SVG for apps that take it."""
    style = style or saved_style()
    fig = build_figure(session, style)
    png = figure_png(fig, style)
    data = QMimeData()
    qimg = QImage.fromData(png, "PNG")
    data.setImageData(qimg)
    data.setData("image/png", QByteArray(png))
    try:
        data.setData("image/svg+xml", QByteArray(figure_svg(fig)))
    except Exception:  # noqa: BLE001 - the bitmap alone is fine
        pass
    QGuiApplication.clipboard().setMimeData(data)
    session.info(f"Figure copied ({qimg.width()} × {qimg.height()} px at {style.dpi} dpi)")


class FigureDialog(QDialog):
    def __init__(self, session: Session, view_crop: Optional[tuple] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Publication figure")
        self.session = session
        self.view_crop = view_crop
        st = saved_style()

        self.width_choice = QComboBox()
        for text, mm in WIDTHS_MM.items():
            self.width_choice.addItem(text, mm)
        self.width_choice.addItem("Custom", 0.0)
        self.width_mm = QDoubleSpinBox()
        self.width_mm.setRange(20, 400)
        self.width_mm.setSuffix(" mm")
        self.width_mm.setValue(st.width_mm)
        presets = [self.width_choice.itemData(i) for i in range(self.width_choice.count())]
        k = next((i for i, mm in enumerate(presets) if abs(mm - st.width_mm) < 1e-6), len(presets) - 1)
        self.width_choice.setCurrentIndex(k)
        self.width_choice.currentIndexChanged.connect(self._width_preset)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 2400)
        self.dpi.setSingleStep(100)
        self.dpi.setValue(st.dpi)
        self.dpi.setSuffix(" dpi")
        self.font_name = QComboBox()
        self.font_name.setEditable(True)
        self.font_name.addItems(["Arial", "Helvetica", "Calibri", "Times New Roman", "DejaVu Sans"])
        self.font_name.setCurrentText(st.font)
        self.font_size = QDoubleSpinBox()
        self.font_size.setRange(4, 24)
        self.font_size.setSingleStep(0.5)
        self.font_size.setSuffix(" pt")
        self.font_size.setValue(st.font_size)

        self.scale_bar = QCheckBox("Scale bar")
        self.scale_bar.setChecked(st.scale_bar)
        self.sb_pos = QComboBox()
        self.sb_pos.addItems(POSITIONS)
        self.sb_pos.setCurrentText(st.scale_bar_position)
        self.sb_len = QDoubleSpinBox()
        self.sb_len.setRange(0, 1e6)
        self.sb_len.setDecimals(3)
        self.sb_len.setSpecialValueText("automatic")
        self.sb_len.setSuffix(" nm")
        self.sb_len.setValue(st.scale_bar_length * 1e9)
        self.sb_color = QComboBox()
        self.sb_color.addItems(["white", "black"])
        self.sb_color.setCurrentText(st.scale_bar_color)
        self.sb_box = QCheckBox("Box behind the bar")
        self.sb_box.setChecked(st.scale_bar_box)

        self.colorbar = QComboBox()
        self.colorbar.addItems(COLORBARS)
        self.colorbar.setCurrentText(st.colorbar)
        self.cb_label = QLineEdit(st.colorbar_label)
        self.cb_label.setPlaceholderText("automatic, e.g. Z (pm)")
        self.panel = QLineEdit(st.panel_label)
        self.panel.setPlaceholderText("e.g. a")
        self.panel.setMaximumWidth(60)
        self.axes = QCheckBox("Axes in physical units")
        self.axes.setChecked(st.axes)
        self.anns = QCheckBox("Annotations (manual + accepted)")
        self.anns.setChecked(st.annotations)
        self.ann_w = QDoubleSpinBox()
        self.ann_w.setRange(0.1, 5)
        self.ann_w.setSingleStep(0.1)
        self.ann_w.setSuffix(" pt")
        self.ann_w.setValue(st.annotation_width)
        self.crop = QCheckBox("Only the area shown in the viewer")
        self.crop.setEnabled(view_crop is not None)
        self.crop.setChecked(False)

        form = QFormLayout()
        form.addRow("Width", self.width_choice)
        form.addRow("", self.width_mm)
        form.addRow("Resolution", self.dpi)
        form.addRow("Font", self.font_name)
        form.addRow("Font size", self.font_size)
        form.addRow(self.scale_bar)
        form.addRow("  Position", self.sb_pos)
        form.addRow("  Length", self.sb_len)
        form.addRow("  Colour", self.sb_color)
        form.addRow("", self.sb_box)
        form.addRow("Colour bar", self.colorbar)
        form.addRow("  Label", self.cb_label)
        form.addRow("Panel letter", self.panel)
        form.addRow(self.axes)
        form.addRow(self.anns)
        form.addRow("  Line width", self.ann_w)
        form.addRow(self.crop)
        controls = QWidget()
        controls.setLayout(form)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumWidth(PREVIEW_PX + 20)
        scroll = self.scroll = QScrollArea()
        scroll.setWidget(self.preview)
        scroll.setWidgetResizable(True)
        self.info = mark(QLabel(""), "muted")
        self.info.setWordWrap(True)

        save = QPushButton("Save…")
        save.setToolTip("PNG or TIFF at the chosen resolution, or vector PDF / SVG (editable text)")
        save.clicked.connect(self.save)
        copy = QPushButton("Copy to clipboard")
        copy.clicked.connect(self.copy)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self.info, 1)
        buttons.addWidget(copy)
        buttons.addWidget(save)
        buttons.addWidget(close)
        top = QHBoxLayout()
        top.addWidget(controls)
        top.addWidget(scroll, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(top, 1)
        lay.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refresh)
        for w in (self.width_mm, self.dpi, self.font_size, self.sb_len, self.ann_w):
            w.valueChanged.connect(self._timer.start)
        for w in (self.font_name, self.sb_pos, self.sb_color, self.colorbar):
            w.currentTextChanged.connect(self._timer.start)
        for w in (self.cb_label, self.panel):
            w.textChanged.connect(self._timer.start)
        for w in (self.scale_bar, self.sb_box, self.axes, self.anns, self.crop):
            w.toggled.connect(self._timer.start)
        self._width_preset()
        self.resize(1000, 720)
        self.refresh()

    def _width_preset(self, *_) -> None:
        mm = self.width_choice.currentData()
        custom = not mm
        self.width_mm.setEnabled(custom)
        if not custom:
            self.width_mm.blockSignals(True)
            self.width_mm.setValue(mm)
            self.width_mm.blockSignals(False)
            self._timer.start()

    def figure_style(self) -> FigureStyle:
        return FigureStyle(width_mm=self.width_mm.value(), dpi=self.dpi.value(), font=self.font_name.currentText(),
                           font_size=self.font_size.value(), scale_bar=self.scale_bar.isChecked(),
                           scale_bar_position=self.sb_pos.currentText(), scale_bar_length=self.sb_len.value() * 1e-9,
                           scale_bar_color=self.sb_color.currentText(), scale_bar_box=self.sb_box.isChecked(),
                           colorbar=self.colorbar.currentText(), colorbar_label=self.cb_label.text().strip(),
                           annotations=self.anns.isChecked(), annotation_width=self.ann_w.value(),
                           panel_label=self.panel.text().strip(), axes=self.axes.isChecked(),
                           crop=self.view_crop if self.crop.isChecked() else None)

    def refresh(self) -> None:
        st = self.figure_style()
        try:
            fig = build_figure(self.session, st)
        except Exception as exc:  # noqa: BLE001
            self.info.setText(f"Cannot draw the figure: {exc}")
            return
        w_in, h_in = fig.get_size_inches()
        avail_w = max(200, self.scroll.viewport().width() - 10)
        avail_h = max(200, self.scroll.viewport().height() - 10)
        png = figure_png(fig, st, dpi=max(20, int(min(avail_w / w_in, avail_h / h_in))))   # the whole figure fits
        self.preview.setPixmap(QPixmap.fromImage(QImage.fromData(png, "PNG")))
        h_mm = h_in * 25.4
        px = (round(w_in * st.dpi), round(h_in * st.dpi))
        self.info.setText(f"{st.width_mm:g} × {h_mm:.0f} mm · {px[0]} × {px[1]} px at {st.dpi} dpi")

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._timer.start()

    def save(self) -> None:
        st = self.figure_style()
        rel = self.session.project.split_image_id(self.session.current_id)[0]
        stem = os.path.splitext(os.path.basename(rel))[0] or "figure"
        start = os.path.join(QSettings("ScanBound", "ScanBound").value("figure_dir", self.session.project.root),
                             stem + ".png")
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", start,
                                              "PNG (*.png);;TIFF (*.tif *.tiff);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".png"
        fig = build_figure(self.session, st)
        save_figure(fig, path, st, image=self.session.image(), display=self.session.entry.display)
        QSettings("ScanBound", "ScanBound").setValue("figure_dir", os.path.dirname(path))
        remember_style(st)
        self.session.info(f"Figure saved to {path}")
        self.info.setText(f"Saved {os.path.basename(path)}")

    def copy(self) -> None:
        st = self.figure_style()
        copy_figure(self.session, st)
        remember_style(st)
        self.info.setText("Copied to the clipboard")

    def accept(self) -> None:
        remember_style(self.figure_style())
        super().accept()
