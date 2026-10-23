"""
fourier_filter.py
=================

Interactive 2D Fourier filtering for the SPM page (like Gwyddion's 2D FFT filter).

The filter is a set of shapes drawn on the (centred) Fourier transform.  They either mark
frequencies to remove (stop) or the only frequencies to keep (pass).  Shapes are stored in
frequency units (cycles / pixel, centre = 0, Nyquist = ±0.5), so a filter applies to images of
any size, and are mirrored automatically because the transform of real data is
point-symmetric.  The filter is a normal step of the processing pipeline ("Fourier filter"):
it can be reordered, toggled, saved with the pipeline and applied to all images.

Shapes:  ("ellipse", cx, cy, rx, ry)   ("rect", x0, y0, x1, y1)   ("poly", ((x, y), ...))
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget)
from scipy import ndimage

from .image_adapt_dialog import Param, StepCard, StepSpec  # noqa: F401

# Mask and filter maths (no Qt) live in scanbound.steps.fourier.
from scanbound.steps.fourier import (  # noqa: E402,F401
    FOURIER_STEP, MODES, as_shapes, build_mask, find_peaks, fourier_filter, frequency_grid, log_spectrum,
    s_fourier,
)
from ..theme import mark



class FourierCard(StepCard):
    """Pipeline card of the Fourier step: the usual controls + the drawn shapes + 'Edit filter…'."""
    edit_requested = pyqtSignal(object)

    def __init__(self, spec, params=None, enabled=True, parent=None):
        super().__init__(spec, params, enabled, parent)
        self.shapes = as_shapes((params or {}).get("shapes"))
        self.summary = QLabel()
        mark(self.summary, "muted")
        edit = QPushButton("Edit filter…")
        edit.setAutoDefault(False)
        edit.clicked.connect(lambda: self.edit_requested.emit(self))
        grid = self.body.layout()
        r = grid.rowCount()
        grid.addWidget(self.summary, r, 0)
        grid.addWidget(edit, r, 1)
        self.body.show()
        self.collapse_btn.show()
        self._update_summary()

    def _update_summary(self):
        n = len(self.shapes)
        self.summary.setText(f"{n} region{'s' if n != 1 else ''}" if n else "no regions yet")

    def values(self):
        v = super().values()
        v["shapes"] = getattr(self, "shapes", ())     # the base constructor asks before shapes exist
        return v

    def set_filter(self, shapes, mode, smooth):
        self.shapes = as_shapes(shapes)
        self.rows["mode"][1].set_value(mode)
        self.rows["smooth"][1].set_value(smooth)
        self._update_summary()
        self.changed.emit()


# =============================================================================
# Editor dialog
# =============================================================================

class FourierFilterDialog(QDialog):
    """
    data: image entering the filter (physical units); px: nm per pixel (or None);
    render(array) -> (H, W[, 3]) uint8 image as shown on the page.
    """

    def __init__(self, data, px=None, render=None, shapes=(), mode=MODES[0], smooth=1.0, parent=None):
        super().__init__(parent)
        from .image_adapt_dialog import ImagePreview
        self.setWindowTitle("Fourier filter")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(1250, 760)
        self.data = np.asarray(data, dtype=float)
        self.px = px
        self.render = render
        rows, cols = self.data.shape
        self._shape = (rows, cols)

        spec = log_spectrum(self.data)
        lo, hi = np.percentile(spec, [1, 99.9])
        self._spec_u8 = (np.clip((spec - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)

        # ---- Fourier transform with drawable regions (the app's own canvas and tools)
        from scanbound.model import Image as _Image
        from ..canvas import ImageCanvas
        from ..session import StandaloneSession
        self._fft_session = StandaloneSession(_Image(self._spec_u8.astype(float), id="fft::spectrum"))
        self._fft_session.set_current_label("Region")
        self.fft_view = ImageCanvas(self._fft_session)
        self.fft_view.show_scale_bar = False
        self.fft_view.setMinimumSize(420, 420)
        self.fft_view.reload()
        self.fft_view.layer.set_image(self._fft_session.current_id)
        self._fft_session.annotationsChanged.connect(lambda *_: self._schedule())
        self.fft_view.cursorMoved.connect(self._show_frequency)

        tools = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        for text, roi, tip in (("Ellipse", "ellipse", "Drag to draw an elliptical region"),
                               ("Rectangle", "rect", "Drag to draw a rectangular region"),
                               ("Polygon", "polygon", "Click points, double click to finish"),
                               ("Select / move", "select", "Select, move and resize regions; Delete removes")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, r=roi: self._set_tool(r))
            self.tool_group.addButton(b)
            tools.addWidget(b)
        self.tool_group.buttons()[0].setChecked(True)
        self._set_tool("ellipse")
        clear = QPushButton("Clear")
        clear.setAutoDefault(False)
        clear.clicked.connect(lambda: self._fft_session.store.clear_image(self._fft_session.current_id))
        tools.addWidget(clear)
        self.freq_label = QLabel(" ")
        mark(self.freq_label, "muted")
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Fourier transform</b> (log power; regions are mirrored automatically)"))
        ll.addLayout(tools)
        ll.addWidget(self.fft_view, 1)
        ll.addWidget(self.freq_label)

        # ---- result preview
        self.preview = ImagePreview()
        self.preview.setMinimumSize(420, 420)
        self.show_what = QComboBox()
        self.show_what.addItems(["Filtered", "Removed part (original − filtered)", "Original"])
        self.show_what.currentIndexChanged.connect(self._update_preview)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Result</b>"))
        top.addStretch(1)
        top.addWidget(self.show_what)
        rl.addLayout(top)
        rl.addWidget(self.preview, 1)
        self.stats_label = QLabel(" ")
        mark(self.stats_label, "muted")
        rl.addWidget(self.stats_label)

        # ---- settings
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.setCurrentText(mode)
        self.mode.currentIndexChanged.connect(self._schedule)
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 10)
        self.smooth.setSingleStep(0.5)
        self.smooth.setValue(smooth)
        self.smooth.setSuffix(" px")
        self.smooth.valueChanged.connect(self._schedule)
        unit = "nm" if px else "px"
        self.cut1 = QDoubleSpinBox()
        self.cut2 = QDoubleSpinBox()
        for sb, val in ((self.cut1, 4.0), (self.cut2, 1.0)):
            sb.setRange(0.001, 1e6)
            sb.setDecimals(3)
            sb.setSuffix(f" {unit}")
            sb.setValue(val * (px * 10 if px else 1))
        presets = QGroupBox("Presets (cutoff = feature period)")
        pg = QGridLayout(presets)
        pg.addWidget(QLabel("Period A"), 0, 0)
        pg.addWidget(self.cut1, 0, 1)
        pg.addWidget(QLabel("Period B"), 0, 2)
        pg.addWidget(self.cut2, 0, 3)
        for i, (text, tip, slot) in enumerate((
                ("Low-pass", "Keep features larger than period A (smoothing)", self._lowpass),
                ("High-pass", "Remove features larger than period A (background)", self._highpass),
                ("Band-pass", "Keep features between periods A and B", lambda: self._band(True)),
                ("Band-stop", "Remove features between periods A and B", lambda: self._band(False)))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setAutoDefault(False)
            b.clicked.connect(slot)
            pg.addWidget(b, 1, i)
        peaks = QGroupBox("Periodic noise")
        pk = QHBoxLayout(peaks)
        self.peak_threshold = QDoubleSpinBox()
        self.peak_threshold.setRange(2, 30)
        self.peak_threshold.setValue(6)
        self.peak_threshold.setPrefix("threshold ")
        self.peak_threshold.setSuffix(" σ")
        self.peak_count = QSpinBox()
        self.peak_count.setRange(1, 40)
        self.peak_count.setValue(8)
        self.peak_count.setPrefix("max ")
        find = QPushButton("Find peaks")
        find.setToolTip("Mark isolated bright spots of the spectrum (e.g. electrical noise) for removal")
        find.setAutoDefault(False)
        find.clicked.connect(self._find_peaks)
        pk.addWidget(self.peak_threshold)
        pk.addWidget(self.peak_count)
        pk.addWidget(find)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Mode"))
        settings.addWidget(self.mode, 1)
        settings.addWidget(QLabel("Edge softness"))
        settings.addWidget(self.smooth)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        lay.addLayout(settings)
        opts = QHBoxLayout()
        opts.addWidget(presets, 3)
        opts.addWidget(peaks, 2)
        lay.addLayout(opts)
        lay.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._recompute)
        self._filtered = self.data
        warning = self._background_warning()
        if warning:
            note = QLabel(warning)
            note.setWordWrap(True)
            mark(note, "warn").setStyleSheet("font-weight: bold;")
            lay.insertWidget(0, note)
        self.set_shapes(shapes)
        self._recompute()

    def _background_warning(self):
        """Text when tilt / background dominates the input (its spectrum is then hard to use)."""
        F = np.abs(np.fft.fft2(self.data - self.data.mean())) ** 2
        fx, fy = frequency_grid(self.data.shape)
        low = np.fft.ifftshift(np.hypot(fx, fy) < 2.5 / max(self.data.shape))
        share = F[low].sum() / max(F.sum(), 1e-300)
        if share > 0.5:
            return (f"{share * 100:.0f} % of the signal is tilt / long-range background. The filter works on the "
                    "image as it enters this step: enable 'Plane level' and/or 'Align rows' above the Fourier "
                    "filter for a clearer spectrum and preview.")
        return ""

    # ------------------------------------------------------------------ coordinates
    def _to_freq(self, x, y):
        rows, cols = self._shape
        return (x - 0.5 - cols // 2) / cols, (y - 0.5 - rows // 2) / rows

    def _to_scene(self, fx, fy):
        rows, cols = self._shape
        return fx * cols + cols // 2 + 0.5, fy * rows + rows // 2 + 0.5

    def _period_text(self, f):
        if f <= 0:
            return "∞"
        period_px = 1.0 / f
        return f"{period_px * self.px:.4g} nm" if self.px else f"{period_px:.4g} px"

    def _show_frequency(self, info):
        if not info:
            self.freq_label.setText(" ")
            return
        fx, fy = self._to_freq(info["x"], info["y"])
        f = math.hypot(fx, fy)
        angle = math.degrees(math.atan2(-fy, fx)) % 180
        self.freq_label.setText(f"period {self._period_text(f)} · direction {angle:.1f}° · "
                                f"f = ({fx:.4f}, {fy:.4f}) 1/px")

    # ------------------------------------------------------------------ shapes <-> regions
    def _set_tool(self, tool):
        self.fft_view.set_tool(tool)

    def shapes(self):
        from scanbound.annotations import Ellipse, Polygon, Rect
        out = []
        for a in self._fft_session.annotations():
            g = a.geometry
            if isinstance(g, Rect):
                (x0, y0), (x1, y1) = self._to_freq(g.x, g.y), self._to_freq(g.x + g.w, g.y + g.h)
                if isinstance(g, Ellipse):
                    out.append(("ellipse", (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2))
                else:
                    out.append(("rect", x0, y0, x1, y1))
            elif isinstance(g, Polygon):
                out.append(("poly", tuple(self._to_freq(x, y) for x, y in g.points)))
        return as_shapes(out)

    def set_shapes(self, shapes):
        from scanbound.annotations import Annotation, Ellipse, Polygon, Rect
        sess = self._fft_session
        anns = []
        for s in as_shapes(shapes):
            if s[0] in ("ellipse", "rect"):
                if s[0] == "ellipse":
                    _, cx, cy, rx, ry = s
                    (x0, y0), (x1, y1) = self._to_scene(cx - rx, cy - ry), self._to_scene(cx + rx, cy + ry)
                    g = Ellipse.from_corners(x0, y0, x1, y1)
                else:
                    _, a, b, c, d = s
                    (x0, y0), (x1, y1) = self._to_scene(a, b), self._to_scene(c, d)
                    g = Rect.from_corners(x0, y0, x1, y1)
            elif s[0] == "poly":
                g = Polygon([self._to_scene(x, y) for x, y in s[1]])
            else:
                continue
            anns.append(Annotation(sess.current_id, "feature", g, label="Region"))
        with sess.store.transaction("Set regions"):
            sess.store.clear_image(sess.current_id)
            if anns:
                sess.store.add(anns)
        self._schedule()

    # ------------------------------------------------------------------ presets
    def _freq_of(self, spin):
        value = spin.value()
        period_px = value / self.px if self.px else value
        return 1.0 / max(period_px, 1e-9)

    def _circle(self, f):
        return ("ellipse", 0.0, 0.0, f, f)

    def _lowpass(self):
        self.mode.setCurrentText(MODES[1])
        self.set_shapes([self._circle(self._freq_of(self.cut1))])

    def _highpass(self):
        self.mode.setCurrentText(MODES[0])
        self.set_shapes([self._circle(self._freq_of(self.cut1))])

    def _band(self, keep):
        fa, fb = sorted((self._freq_of(self.cut1), self._freq_of(self.cut2)))
        # annulus fa..fb as a polygon ring is awkward: use two circles via a polygon with a hole-free
        # approximation -> outer circle polygon minus inner circle polygon
        t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ring = [(fb * math.cos(a), fb * math.sin(a)) for a in t] + [(fb, 0.0)] + \
               [(fa * math.cos(a), -fa * math.sin(a)) for a in t] + [(fa, 0.0)]
        self.mode.setCurrentText(MODES[1] if keep else MODES[0])
        self.set_shapes([("poly", tuple(ring))])

    def _find_peaks(self):
        found = find_peaks(self.data, self.peak_threshold.value(), self.peak_count.value())
        if not found:
            self.stats_label.setText("No isolated peaks above the threshold.")
            return
        self.mode.setCurrentText(MODES[0])
        self.set_shapes(list(self.shapes()) + found)
        self.stats_label.setText(f"Marked {len(found)} peak(s) for removal.")

    # ------------------------------------------------------------------ preview
    def _schedule(self, *_):
        self._timer.start()

    def _recompute(self):
        shapes = self.shapes()
        mode, smooth = self.mode.currentText(), self.smooth.value()
        mask = build_mask(self._shape, shapes, mode, smooth)
        # FFT display: removed frequencies tinted red
        rgb = np.repeat(self._spec_u8[..., None], 3, axis=2).astype(float)
        removed = 1.0 - mask
        rgb[..., 1] *= 1 - 0.75 * removed
        rgb[..., 2] *= 1 - 0.75 * removed
        rgb[..., 0] = rgb[..., 0] * (1 - 0.4 * removed) + 110 * removed
        rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
        h, w = rgb.shape[:2]
        self.fft_view.show_rgb(rgb)
        self._filtered = fourier_filter(self.data, shapes, mode, smooth) if shapes else self.data
        kept = float(mask.mean()) * 100
        diff = self.data - self._filtered
        self.stats_label.setText(f"{len(shapes)} region(s) · {kept:.1f} % of the spectrum kept · "
                                 f"RMS removed {np.sqrt(np.mean(diff ** 2)):.3g} (data units)")
        self._update_preview()

    def _update_preview(self, *_):
        which = self.show_what.currentIndex()
        img = self._filtered if which == 0 else (self.data - self._filtered if which == 1 else self.data)
        if self.render is not None:
            u8 = self.render(img)
        else:
            lo, hi = np.percentile(img, [0.5, 99.5])
            u8 = (np.clip((img - lo) / max(hi - lo, 1e-30), 0, 1) * 255).astype(np.uint8)
        u8 = np.ascontiguousarray(u8)
        h, w = u8.shape[:2]
        fmt = QImage.Format.Format_Grayscale8 if u8.ndim == 2 else QImage.Format.Format_RGB888
        self.preview.set_image(QImage(u8.data, w, h, u8.strides[0], fmt).copy())

    def get_filter(self):
        """(shapes, mode, edge softness) chosen in the dialog."""
        return self.shapes(), self.mode.currentText(), self.smooth.value()
