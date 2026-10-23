"""
surface_view.py
===============

Interactive 3D surface view of an SPM image (OpenGL, via pyqtgraph), similar to Gwyddion's
3D window: drag to rotate, wheel to zoom, Ctrl / middle drag to pan.  The surface uses the
page's colormap and display levels.

``provider()`` returns a dict:
    data    2D array in its physical unit (processed image)
    px      nm per pixel or None
    unit    data unit ("m", "A", ...)
    rgb     (H, W, 3) uint8 colours as shown on the page (optional)
    levels  (low, high) display levels of the page (optional, used to clip spikes)
    title   window title text
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

from scanbound import display as idp
from scanbound.io.microscopy import fmt_length, length_to_nm
from ..theme import mark
from ..qtutil import save_file_name


def _downsample(arr, max_size):
    """Area-average down to at most max_size pixels on the long side (keeps the aspect)."""
    import cv2
    h, w = arr.shape[:2]
    f = max(h, w) / float(max_size)
    if f <= 1:
        return arr, 1.0
    size = (max(2, int(round(w / f))), max(2, int(round(h / f))))
    return cv2.resize(arr, size, interpolation=cv2.INTER_AREA), w / size[0]


class SurfaceViewDialog(QDialog):
    def __init__(self, provider, parent=None):
        super().__init__(parent)
        import pyqtgraph.opengl as gl
        self._gl = gl
        self.setWindowTitle("3D view")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(1100, 760)
        self._provider = provider
        self._surface = None
        self._decor = []
        self._extent = 1.0
        self._zrange_true = None          # z range in nm when the data is a length (true scale)

        from .glview import gl_view_class
        self.view = gl_view_class()()
        self.view.setBackgroundColor("k")

        # ---- controls
        self.zscale = QSlider(Qt.Orientation.Horizontal)
        self.zscale.setRange(-400, 200)            # log scale: 10**(v/100) x auto (true scale of flat surfaces is far below auto)
        self.zscale.setValue(0)
        self.zscale.setToolTip("Height exaggeration")
        self.zscale.valueChanged.connect(self._rebuild)
        self.zscale_label = QLabel()
        true_btn = QPushButton("True scale")
        true_btn.setToolTip("Heights to the same scale as the lateral dimensions (lengths only)")
        true_btn.clicked.connect(self._true_scale)
        auto_btn = QPushButton("Auto")
        auto_btn.setToolTip("Height range = 30 % of the image size")
        auto_btn.clicked.connect(lambda: self.zscale.setValue(0))

        self.resolution = QComboBox()
        for n in (128, 256, 512, 1024):
            self.resolution.addItem(f"{n} px", n)
        self.resolution.setCurrentIndex(1)
        self.resolution.setToolTip("Mesh resolution (higher = finer, slower)")
        self.resolution.currentIndexChanged.connect(self.refresh)
        self.lighting = QCheckBox("Lighting")
        self.lighting.setChecked(True)
        self.lighting.toggled.connect(self._rebuild)
        self.grid = QCheckBox("Grid and axes")
        self.grid.setChecked(True)
        self.grid.toggled.connect(self._rebuild)
        self.background = QComboBox()
        self.background.addItems(["Black", "White", "Gray"])
        self.background.currentTextChanged.connect(self._set_background)
        self.clip = QCheckBox("Clip to display levels")
        self.clip.setChecked(True)
        self.clip.setToolTip("Limit heights to the page's display levels, so spikes and crash lines do not "
                             "flatten the surface (as in the 2D view). Off: true data range.")
        self.clip.toggled.connect(self.refresh)
        self.follow = QCheckBox("Follow the page")
        self.follow.setChecked(True)
        self.follow.setToolTip("Update when the image, filters or colours change on the page")

        views = QHBoxLayout()
        for text, el, az in (("Top", 90, -90), ("Iso", 35, -60), ("Front", 8, -90), ("Side", 8, 0)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, e=el, a=az: self.set_camera(e, a))
            views.addWidget(b)

        save = QPushButton("Save image…")
        save.clicked.connect(self.save_image)
        update = QPushButton("Update now")
        update.clicked.connect(self.refresh)

        box = QGroupBox("Height")
        g = QGridLayout(box)
        g.addWidget(self.zscale, 0, 0, 1, 2)
        g.addWidget(self.zscale_label, 1, 0, 1, 2)
        g.addWidget(auto_btn, 2, 0)
        g.addWidget(true_btn, 2, 1)
        box2 = QGroupBox("Display")
        g2 = QGridLayout(box2)
        g2.addWidget(QLabel("Resolution"), 0, 0)
        g2.addWidget(self.resolution, 0, 1)
        g2.addWidget(QLabel("Background"), 1, 0)
        g2.addWidget(self.background, 1, 1)
        g2.addWidget(self.lighting, 2, 0, 1, 2)
        g2.addWidget(self.grid, 3, 0, 1, 2)
        g2.addWidget(self.clip, 4, 0, 1, 2)
        box3 = QGroupBox("View")
        v3 = QVBoxLayout(box3)
        v3.addLayout(views)
        hint = QLabel("Drag: rotate · Wheel: zoom · Ctrl/middle drag: pan")
        hint.setWordWrap(True)
        mark(hint, "muted")
        v3.addWidget(hint)
        self.info = QLabel()
        self.info.setWordWrap(True)

        side = QWidget()
        side.setMaximumWidth(270)
        sl = QVBoxLayout(side)
        sl.addWidget(box)
        sl.addWidget(box2)
        sl.addWidget(box3)
        sl.addWidget(self.follow)
        sl.addWidget(update)
        sl.addWidget(save)
        sl.addWidget(self.info)
        sl.addStretch(1)

        lay = QHBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addWidget(side)
        self._mesh = None
        self.refresh()
        self.set_camera(35, -60)

    # ------------------------------------------------------------------ data
    def refresh(self, *_):
        info = self._provider() or {}
        data = info.get("data")
        if data is None:
            self.info.setText("No image.")
            return
        data = np.nan_to_num(np.asarray(data, dtype=float), nan=float(np.nanmin(data)))
        levels = info.get("levels")
        if self.clip.isChecked() and levels is not None:
            data = np.clip(data, *levels)
        n = self.resolution.currentData()
        z, factor = _downsample(data, n)
        rgb = info.get("rgb")
        if rgb is not None:
            rgb = np.asarray(rgb)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[..., None], 3, axis=2)
            rgb, _ = _downsample(rgb.astype(np.float32), n)
        px = info.get("px")
        step = (px or 1.0) * factor                     # nm (or px) per mesh cell
        unit = info.get("unit", "")
        to_nm = length_to_nm(1.0, unit)                 # None when the data is not a length
        zr = float(z.max() - z.min()) or 1.0
        self._zrange_true = zr * to_nm if (to_nm and px) else None
        self._mesh = {"z": z, "rgb": rgb, "step": step, "px": px, "unit": unit, "zr": zr}
        rows, cols = z.shape
        self._extent = max(rows, cols) * step
        self.setWindowTitle(f"3D view – {info.get('title', '')}")
        self._rebuild()

    def _auto_height(self):
        return 0.3 * self._extent

    def _height_scale(self):
        """Factor from data units to scene units (scene units = nm or px of the lateral axes)."""
        m = self._mesh
        return self._auto_height() / m["zr"] * 10 ** (self.zscale.value() / 100.0)

    def _true_scale(self):
        if not self._zrange_true:
            self.info.setText("True scale needs height data in a length unit and a pixel scale.")
            return
        ratio = self._zrange_true / self._auto_height()           # true height / auto height
        self.zscale.setValue(int(round(100 * math.log10(max(ratio, 1e-6)))))

    def _rebuild(self, *_):
        gl = self._gl
        m = self._mesh
        if m is None:
            return
        for item in [self._surface] + self._decor:
            if item is not None:
                self.view.removeItem(item)
        self._decor = []
        z = m["z"]
        rows, cols = z.shape
        s = self._height_scale()
        # GL y points up: flip rows so the surface seen from the top looks like the image
        zz = ((z - z.min()) * s)[::-1].T
        x = np.arange(cols) * m["step"]
        y = np.arange(rows) * m["step"]
        colors = None
        if m["rgb"] is not None:
            rgb = m["rgb"][::-1].transpose(1, 0, 2) / 255.0
            colors = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], axis=2)
        else:
            t = (zz - zz.min()) / (np.ptp(zz) or 1)
            colors = np.stack([t, t, t, np.ones_like(t)], axis=2)
        self._surface = gl.GLSurfacePlotItem(x=x, y=y, z=zz, colors=colors.reshape(-1, 4),
                                             shader="shaded" if self.lighting.isChecked() else None,
                                             smooth=True, computeNormals=self.lighting.isChecked())
        self.view.addItem(self._surface)
        height = float(zz.max())
        if self.grid.isChecked():
            self._add_frame(cols * m["step"], rows * m["step"], height)
        self.view.opts["center"] = QVector3D(cols * m["step"] / 2, rows * m["step"] / 2, height / 2)
        exag = ""
        if self._zrange_true:
            exag = f" (×{height / self._zrange_true:.3g} vs true scale)"
        self.zscale_label.setText(f"Height ×{10 ** (self.zscale.value() / 100):.2f} auto{exag}")
        lateral = "nm" if m["px"] else "px"
        factor, zunit = idp.display_unit(m["unit"], m["zr"])
        self.info.setText(f"{cols} × {rows} mesh · {fmt_length(cols * m['step']) if m['px'] else f'{cols * m['step']:.0f} px'}"
                          f" wide · height range {m['zr'] * factor:.4g} {zunit}")

    def _add_frame(self, w, h, height):
        gl = self._gl
        fg = (0.15, 0.15, 0.15, 1) if self.background.currentText() == "White" else (0.85, 0.85, 0.85, 1)
        grid = gl.GLGridItem()
        grid.setSize(w, h)
        grid.setSpacing(w / 10, h / 10)
        grid.translate(w / 2, h / 2, 0)
        grid.setColor((128, 128, 128, 90))
        self.view.addItem(grid)
        self._decor.append(grid)
        corner = np.array([[0, 0, 0], [w, 0, 0], [0, 0, 0], [0, h, 0], [0, 0, 0], [0, 0, height]], float)
        axes = gl.GLLinePlotItem(pos=corner, color=fg, width=2, mode="lines")
        self.view.addItem(axes)
        self._decor.append(axes)
        m = self._mesh
        if hasattr(gl, "GLTextItem"):
            def label(pos, text):
                t = gl.GLTextItem(pos=pos, text=text, color=tuple(int(c * 255) for c in fg))
                self.view.addItem(t)
                self._decor.append(t)
            if m["px"]:
                label((w, 0, 0), fmt_length(w, 3))
                label((0, h, 0), fmt_length(h, 3))
            else:
                label((w, 0, 0), f"{w:.0f} px")
                label((0, h, 0), f"{h:.0f} px")
            factor, zunit = idp.display_unit(m["unit"], m["zr"])
            label((0, 0, height), f"{m['zr'] * factor:.3g} {zunit}")

    # ------------------------------------------------------------------ view
    def set_camera(self, elevation, azimuth):
        self.view.setCameraPosition(distance=1.7 * self._extent, elevation=elevation, azimuth=azimuth)

    def _set_background(self, text):
        self.view.setBackgroundColor({"Black": "k", "White": "w", "Gray": (60, 60, 60)}[text])
        self._rebuild()

    def save_image(self):
        path = save_file_name(self, "Save 3D view", "surface_3d.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.view.grabFramebuffer().save(path)

    def page_changed(self):
        """Called by the page when its image or display changed."""
        if self.isVisible() and self.follow.isChecked():
            self.refresh()
