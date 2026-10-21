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

from core.image_adapt_dialog import Param, StepCard, StepSpec

MODES = ("Remove selected (stop)", "Keep only selected (pass)")


# =============================================================================
# Mask and filter (no Qt)
# =============================================================================

def frequency_grid(shape):
    """(fx, fy) in cycles / pixel of the fftshift-ed transform of an image of `shape`."""
    rows, cols = shape
    fx = (np.arange(cols) - cols // 2) / cols
    fy = (np.arange(rows) - rows // 2) / rows
    return np.meshgrid(fx, fy)


def _shape_mask(fx, fy, shape):
    kind = shape[0]
    if kind == "ellipse":
        _, cx, cy, rx, ry = shape
        rx, ry = max(abs(rx), 1e-9), max(abs(ry), 1e-9)
        return ((fx - cx) / rx) ** 2 + ((fy - cy) / ry) ** 2 <= 1.0
    if kind == "rect":
        _, x0, y0, x1, y1 = shape
        return (fx >= min(x0, x1)) & (fx <= max(x0, x1)) & (fy >= min(y0, y1)) & (fy <= max(y0, y1))
    if kind == "poly":
        from matplotlib.path import Path
        pts = np.asarray(shape[1], float)
        if len(pts) < 3:
            return np.zeros(fx.shape, bool)
        inside = Path(pts).contains_points(np.column_stack([fx.ravel(), fy.ravel()]))
        return inside.reshape(fx.shape)
    return np.zeros(fx.shape, bool)


def build_mask(shape, shapes, mode=MODES[0], smooth=0.0):
    """Transfer function (0..1) on the fftshift-ed grid of an image of `shape`."""
    fx, fy = frequency_grid(shape)
    selected = np.zeros(shape, bool)
    for s in shapes or ():
        selected |= _shape_mask(fx, fy, s) | _shape_mask(-fx, -fy, s)      # point-symmetric
    keep = selected if mode == MODES[1] else ~selected
    mask = keep.astype(float)
    if smooth > 0:
        mask = ndimage.gaussian_filter(mask, smooth)
    return mask


def fourier_filter(z, shapes, mode=MODES[0], smooth=0.0):
    """Filtered image (same unit); the mean is always kept."""
    z = np.asarray(z, dtype=float)
    if not shapes:
        return z.copy()
    mean = float(np.mean(z))
    F = np.fft.fftshift(np.fft.fft2(z - mean))
    out = np.fft.ifft2(np.fft.ifftshift(F * build_mask(z.shape, shapes, mode, smooth))).real
    return out + mean


def log_spectrum(z, window=False):
    """log10 |F|² of the mean-free image, centred.  window=True applies a Hann window first
    (less spectral leakage from the image edges; used to detect peaks)."""
    z = np.asarray(z, dtype=float)
    z = z - np.mean(z)
    if window:
        z = z * np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(z))
    return np.log10(np.abs(F) ** 2 + 1e-30)


def find_peaks(z, threshold=5.0, max_peaks=12, exclude_radius=0.03, radius=0.01):
    """
    Ellipses around isolated bright peaks of the spectrum (periodic noise / lattice), each
    peak once (its mirror is implied).  threshold: in standard deviations above the median.
    """
    spec = log_spectrum(z, window=True)
    fx, fy = frequency_grid(spec.shape)
    centre = np.hypot(fx, fy) < exclude_radius
    centre |= (np.abs(fx) > 0.48) | (np.abs(fy) > 0.48)          # outermost border: leakage / aliasing
    # periodic noise = sharp spike above its *local* surroundings (the broad spectrum of the real
    # features is bright too, but smooth), so compare with a local median background
    work = spec - ndimage.median_filter(spec, size=15)
    work[centre] = 0.0
    local_max = work == ndimage.maximum_filter(work, size=7)
    med = np.median(work)
    sd = 1.4826 * np.median(np.abs(work - med)) or work.std() or 1.0
    cand = np.argwhere(local_max & (work > med + threshold * sd))
    cand = sorted(cand, key=lambda rc: -work[rc[0], rc[1]])
    shapes, seen = [], []
    for r, c in cand:
        x, y = float(fx[r, c]), float(fy[r, c])
        if any(math.hypot(x - a, y - b) < 2 * radius or math.hypot(x + a, y + b) < 2 * radius for a, b in seen):
            continue
        seen.append((x, y))
        shapes.append(("ellipse", x, y, radius, radius))
        if len(shapes) >= max_peaks:
            break
    return shapes


def as_shapes(value):
    """Shapes from JSON lists back to hashable tuples."""
    out = []
    for s in value or ():
        if not s:
            continue
        if s[0] == "poly":
            out.append(("poly", tuple(tuple(float(v) for v in p) for p in s[1])))
        else:
            out.append((s[0],) + tuple(float(v) for v in s[1:]))
    return tuple(out)


def s_fourier(z, mode, smooth, shapes=()):
    return fourier_filter(z, as_shapes(shapes), mode, smooth)


FOURIER_STEP = StepSpec("fourier", "Fourier filter", "gray", s_fourier, [
    Param("mode", "Mode", "choice", MODES[0], choices=MODES),
    Param("smooth", "Edge softness (px)", "float", 1.0, 0.0, 10.0, 0.5, 1,
          tooltip="Soften the mask edges to avoid ringing."),
], "Remove (or keep only) frequency regions drawn on the Fourier transform: periodic noise, "
   "low-pass / high-pass / band-pass.")


class FourierCard(StepCard):
    """Pipeline card of the Fourier step: the usual controls + the drawn shapes + 'Edit filter…'."""
    edit_requested = pyqtSignal(object)

    def __init__(self, spec, params=None, enabled=True, parent=None):
        super().__init__(spec, params, enabled, parent)
        self.shapes = as_shapes((params or {}).get("shapes"))
        self.summary = QLabel()
        self.summary.setStyleSheet("color: #555;")
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
        from core.image_adapt_dialog import ImagePreview
        from ui.PyQtImageViewer.QtImageViewer import QtImageViewer
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

        # ---- Fourier transform with drawable regions
        self.fft_view = QtImageViewer()
        self.fft_view.setMinimumSize(420, 420)
        self.fft_view.setLabelList(["Region"])
        self.fft_view._currentLabel = "Region"
        self.fft_view.distinguishROIs = False
        self.fft_view.regionZoomButton = None
        self.fft_view.selectionButton = Qt.MouseButton.LeftButton
        self.fft_view.roisChanged.connect(lambda *_: self._schedule())
        self.fft_view.mousePositionOnImageChanged.connect(self._show_frequency)
        self.fft_view.setImage(self._spec_u8)

        tools = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        for text, roi, tip in (("Ellipse", "EllipseROI", "Drag to draw an elliptical region"),
                               ("Rectangle", "RectROI", "Drag to draw a rectangular region"),
                               ("Polygon", "PolygonROI", "Click points, double click to finish"),
                               ("Select / move", None, "Select, move and resize regions; Delete removes")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, r=roi: self._set_tool(r))
            self.tool_group.addButton(b)
            tools.addWidget(b)
        self.tool_group.buttons()[0].setChecked(True)
        self._set_tool("EllipseROI")
        clear = QPushButton("Clear")
        clear.setAutoDefault(False)
        clear.clicked.connect(lambda: self.fft_view.clearROIs(emit=True))
        tools.addWidget(clear)
        self.freq_label = QLabel(" ")
        self.freq_label.setStyleSheet("color: #555;")
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
        self.stats_label.setStyleSheet("color: #555;")
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
            note.setStyleSheet("color: #8a4b00; font-weight: bold;")
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

    def _show_frequency(self, pos):
        if pos.x() < 0:
            self.freq_label.setText(" ")
            return
        fx, fy = self._to_freq(pos.x() + 0.5, pos.y() + 0.5)
        f = math.hypot(fx, fy)
        angle = math.degrees(math.atan2(-fy, fx)) % 180
        self.freq_label.setText(f"period {self._period_text(f)} · direction {angle:.1f}° · "
                                f"f = ({fx:.4f}, {fy:.4f}) 1/px")

    # ------------------------------------------------------------------ shapes <-> regions
    def _set_tool(self, roi_type):
        self.fft_view.drawROI = roi_type

    def shapes(self):
        out = []
        for r in self.fft_view.rois():
            if r.roiType in ("EllipseROI", "RectROI"):
                rect = r.rect()
                (x0, y0), (x1, y1) = self._to_freq(rect.left(), rect.top()), self._to_freq(rect.right(), rect.bottom())
                if r.roiType == "RectROI":
                    out.append(("rect", x0, y0, x1, y1))
                else:
                    out.append(("ellipse", (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2))
            elif r.roiType == "PolygonROI":
                out.append(("poly", tuple(self._to_freq(x, y) for x, y in r.pointsArray())))
        return as_shapes(out)

    def set_shapes(self, shapes):
        v = self.fft_view
        v.clearROIs()
        rois = []
        for s in as_shapes(shapes):
            if s[0] == "ellipse":
                _, cx, cy, rx, ry = s
                x0, y0 = self._to_scene(cx - rx, cy - ry)
                x1, y1 = self._to_scene(cx + rx, cy + ry)
                rois.append(v.createROI("EllipseROI", (x0, y0, x1 - x0, y1 - y0), "Region"))
            elif s[0] == "rect":
                _, a, b, c, d = s
                x0, y0 = self._to_scene(a, b)
                x1, y1 = self._to_scene(c, d)
                rois.append(v.createROI("RectROI", (x0, y0, x1 - x0, y1 - y0), "Region"))
            elif s[0] == "poly":
                rois.append(v.createROI("PolygonROI", [self._to_scene(x, y) for x, y in s[1]], "Region"))
        if rois:
            v.addROIs(rois, emit=False)
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
        self.fft_view.setImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())
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
