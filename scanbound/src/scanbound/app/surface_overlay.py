"""
Canvas overlay of the surface analyses: fitted lattice sites (coloured by how far they sit from the
ideal lattice), the ideal lattice, the unit cell, the aligned structure model's atoms and sites, and
a colour map (strain, rotation, domains).  Drawn in screen pixels so markers keep their size at any zoom.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem

from .qtutil import to_qimage

SITE_OK = "#2ECC71"
SITE_EXTRA = "#FF4FD8"
PREVIEW = "#39FF14"
IDEAL = "#FFD400"
CELL = "#00E5FF"
LAYER_COLORS = ["#FF8A00", "#7FA7FF", "#B0B0B0", "#9E7BFF"]
SITE_COLORS = {"top": "#FF5252", "bridge": "#40C4FF", "hollow": "#69F0AE", "custom": "#FFD740"}


def _ramp(v: np.ndarray, vmax: float) -> np.ndarray:
    """0 -> green, vmax -> red (RGB 0..255)."""
    t = np.clip(np.nan_to_num(v / max(vmax, 1e-30), nan=0.0), 0, 1)[:, None]
    return (np.array([46, 204, 113]) * (1 - t) + np.array([255, 64, 64]) * t).astype(int)


class SurfaceOverlay(QGraphicsItem):
    def __init__(self) -> None:
        super().__init__()
        self.setZValue(400)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.fit: Any = None
        self.alignment: Any = None
        self.shape: tuple[int, int] = (1, 1)
        self.show_sites = True
        self.show_ideal = False
        self.show_cell = True
        self.show_model = False
        self.show_model_sites = False
        self._ideal = np.zeros((0, 2))
        self._atoms: list[tuple[Any, np.ndarray]] = []
        self._model_sites: list[tuple[Any, np.ndarray]] = []
        self._site_colors = np.zeros((0, 3), int)
        self._preview = np.zeros((0, 2))           # candidate sites while tuning the detection
        self._preview_on = np.zeros(0, bool)       # ... and which of them pass the threshold
        self.map_item = QGraphicsPixmapItem()
        self.map_item.setZValue(5)
        self.map_item.setOpacity(0.55)
        self.map_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.map_item.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)

    # ------------------------------------------------------------------ data
    def set_data(self, fit: Any, alignment: Any, shape: tuple[int, int]) -> None:
        self.prepareGeometryChange()
        self.fit, self.alignment, self.shape = fit, alignment, tuple(shape)
        self._ideal = np.zeros((0, 2))
        self._atoms, self._model_sites = [], []
        if fit is not None:
            try:
                self._ideal = fit.predicted(shape)[1]
            except Exception:  # noqa: BLE001
                pass
            d = np.asarray(fit.displacement, float)
            ok = np.isfinite(d)
            vmax = 3 * float(np.median(d[ok])) if ok.any() else 1.0
            self._site_colors = _ramp(d, vmax)
        if alignment is not None:
            try:
                self._atoms = alignment.atom_pixels(shape)
                self._model_sites = self._site_pixels(alignment, shape)
            except Exception:  # noqa: BLE001 - e.g. too many model cells: leave the model out
                self._atoms, self._model_sites = [], []
        self.update()

    @staticmethod
    def _site_pixels(al: Any, shape: tuple[int, int]) -> list[tuple[Any, np.ndarray]]:
        rows, cols = shape
        fit = al.fit
        corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]], float)
        f = fit.to_fractional(corners)
        lo, hi = np.floor(f.min(0)) - 1, np.ceil(f.max(0)) + 1
        m, n = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        cells = np.column_stack([m.ravel(), n.ravel()])
        out = []
        for s, p in al.site_fractional():
            px = fit.to_pixel(cells + p)
            ok = (px[:, 0] >= 0) & (px[:, 0] <= cols) & (px[:, 1] >= 0) & (px[:, 1] <= rows)
            out.append((s, px[ok]))
        return out

    def set_preview(self, xy: Optional[np.ndarray], accepted: Optional[np.ndarray] = None,
                    shape: Optional[tuple[int, int]] = None) -> None:
        """Candidate sites before fitting: green dots pass the threshold, grey ones do not. None hides them."""
        self.prepareGeometryChange()
        if shape is not None:
            self.shape = tuple(shape)
        self._preview = np.zeros((0, 2)) if xy is None else np.asarray(xy, float).reshape(-1, 2)
        self._preview_on = (np.ones(len(self._preview), bool) if accepted is None
                            else np.asarray(accepted, bool).reshape(-1))
        self.update()

    def set_map(self, values: Optional[np.ndarray], cmap: str = "RdBu_r", vmax: Optional[float] = None,
                categorical: bool = False) -> None:
        """A per-pixel map under the markers (NaN = transparent); None removes it."""
        if values is None:
            self.map_item.setVisible(False)
            return
        import matplotlib
        v = np.asarray(values, float)
        ok = np.isfinite(v)
        if categorical:
            cm = matplotlib.colormaps["tab10"]
            rgba = cm(np.where(ok, np.mod(np.nan_to_num(v), 10) / 10 + 0.05, 0))
        else:
            vm = vmax or (float(np.nanpercentile(np.abs(v[ok]), 98)) if ok.any() else 1.0) or 1.0
            rgba = matplotlib.colormaps[cmap](np.clip((np.nan_to_num(v) / vm + 1) / 2, 0, 1))
        rgba[..., 3] = np.where(ok, 1.0, 0.0)
        self.map_item.setPixmap(QPixmap.fromImage(to_qimage((rgba * 255).astype(np.uint8))))
        self.map_item.setVisible(True)

    # ------------------------------------------------------------------ painting
    def boundingRect(self) -> QRectF:
        rows, cols = self.shape
        return QRectF(-20, -20, cols + 40, rows + 40)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ANN001
        if len(self._preview):
            self._paint_preview(painter, option)
        fit = self.fit
        if fit is None:
            return
        t = painter.worldTransform()
        exposed = option.exposedRect if option is not None else self.boundingRect()
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # level of detail: markers closer than ~8 screen pixels would only hide the image
        c0 = fit.to_pixel(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        q = [t.map(QPointF(float(x), float(y))) for x, y in c0]
        spacing = min(np.hypot(q[1].x() - q[0].x(), q[1].y() - q[0].y()), np.hypot(q[2].x() - q[0].x(), q[2].y() - q[0].y()))
        detail = spacing >= 8.0

        def visible(P: np.ndarray) -> np.ndarray:
            if not len(P):
                return np.zeros(0, bool)
            return ((P[:, 0] >= exposed.left() - 2) & (P[:, 0] <= exposed.right() + 2)
                    & (P[:, 1] >= exposed.top() - 2) & (P[:, 1] <= exposed.bottom() + 2))

        if detail and self.show_model and self._atoms:
            for atom, P in sorted(self._atoms, key=lambda ap: -ap[0].layer):
                c = QColor(LAYER_COLORS[min(atom.layer, len(LAYER_COLORS) - 1)])
                r = 4.5 if atom.layer == 0 else 3.0
                painter.setPen(QPen(c.darker(150), 1.0))
                painter.setBrush(QBrush(c) if atom.layer == 0 else Qt.BrushStyle.NoBrush)
                for x, y in P[visible(P)]:
                    q = t.map(QPointF(float(x), float(y)))
                    painter.drawEllipse(q, r, r)
        if detail and self.show_model_sites and self._model_sites:
            for s, P in self._model_sites:
                c = QColor(SITE_COLORS.get(s.kind, SITE_COLORS["custom"]))
                painter.setPen(QPen(c, 1.4))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for x, y in P[visible(P)]:
                    q = t.map(QPointF(float(x), float(y)))
                    painter.drawRect(QRectF(q.x() - 2.5, q.y() - 2.5, 5, 5))
        if detail and self.show_ideal and len(self._ideal):
            painter.setPen(QPen(QColor(IDEAL), 1.0))
            for x, y in self._ideal[visible(self._ideal)]:
                q = t.map(QPointF(float(x), float(y)))
                painter.drawLine(QPointF(q.x() - 3, q.y()), QPointF(q.x() + 3, q.y()))
                painter.drawLine(QPointF(q.x(), q.y() - 3), QPointF(q.x(), q.y() + 3))
        if detail and self.show_sites:
            xy = fit.sites.xy
            vis = visible(xy)
            on = fit.on_lattice
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in np.nonzero(vis)[0]:
                q = t.map(QPointF(float(xy[i, 0]), float(xy[i, 1])))
                if on[i]:
                    r, g, b = self._site_colors[i]
                    painter.setPen(QPen(QColor(int(r), int(g), int(b)), 1.4))
                    painter.drawEllipse(q, 3.2, 3.2)
                else:
                    painter.setPen(QPen(QColor(SITE_EXTRA), 1.6))
                    painter.drawLine(QPointF(q.x() - 3, q.y() - 3), QPointF(q.x() + 3, q.y() + 3))
                    painter.drawLine(QPointF(q.x() - 3, q.y() + 3), QPointF(q.x() + 3, q.y() - 3))
        if self.show_cell:
            # the cell nearest the image centre, so it is in view
            rows, cols = self.shape
            uv = np.round(fit.to_fractional(np.array([[cols / 2, rows / 2]]))[0])
            pts = fit.to_pixel(np.array([uv, uv + [1, 0], uv + [1, 1], uv + [0, 1]]))
            poly = QPolygonF([t.map(QPointF(float(x), float(y))) for x, y in pts])
            painter.setPen(QPen(QColor(CELL), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(poly)
            painter.drawText(poly[1] + QPointF(4, 0), "a1")
            painter.drawText(poly[3] + QPointF(4, 0), "a2")
        painter.restore()

    def _paint_preview(self, painter: QPainter, option) -> None:  # noqa: ANN001
        t = painter.worldTransform()
        r = option.exposedRect if option is not None else self.boundingRect()
        P = self._preview
        vis = (P[:, 0] >= r.left() - 2) & (P[:, 0] <= r.right() + 2) & (P[:, 1] >= r.top() - 2) & (P[:, 1] <= r.bottom() + 2)
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        for on, col, rad in ((False, QColor(160, 160, 160, 150), 1.6), (True, QColor(PREVIEW), 2.4)):
            painter.setBrush(QBrush(col))
            for x, y in P[vis & (self._preview_on == on)]:
                painter.drawEllipse(t.map(QPointF(float(x), float(y))), rad, rad)
        painter.restore()
