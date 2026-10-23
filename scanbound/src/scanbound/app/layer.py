"""
Draws the annotation store on the canvas.

One light ``QGraphicsItem`` per annotation of the current image, updated incrementally from store
events (only the changed annotations are touched, so editing one of thousands stays instant).
Items paint in device (screen) coordinates: lines, handles and arrowheads keep their on-screen
size at every zoom and with non-square pixels.

Style: the label decides the colour; the review status the line style (manual / accepted solid,
proposed dashed, rejected dotted grey).  Selected annotations are drawn thicker with handles.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem

from ..annotations import Annotation, Ellipse, Geometry, KINDS, Mask, Point, Rect, StoreEvent
from .qtutil import qcolor
from .tools import Preview, handles

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QGraphicsScene
    from .session import Session

HANDLE = 4.0            # half size of a vertex handle, screen px
REJECTED = "#8c8c8c"
REFERENCE = "#F5E400"


def _outline(g: Geometry) -> np.ndarray:
    return g.outline(64) if isinstance(g, Ellipse) else g.outline()


def _device_poly(painter: QPainter, pts: np.ndarray) -> QPolygonF:
    t = painter.worldTransform()
    return QPolygonF([t.map(QPointF(float(x), float(y))) for x, y in pts])


def _arrow_head(painter: QPainter, tip: QPointF, tail: QPointF, size: float = 9.0) -> None:
    dx, dy = tip.x() - tail.x(), tip.y() - tail.y()
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return
    ux, uy = dx / n, dy / n
    left = QPointF(tip.x() - size * ux + size * 0.45 * uy, tip.y() - size * uy - size * 0.45 * ux)
    right = QPointF(tip.x() - size * ux - size * 0.45 * uy, tip.y() - size * uy + size * 0.45 * ux)
    painter.drawPolygon(QPolygonF([tip, left, right]))


def _mask_image(g: Mask, color: QColor, alpha: int) -> QImage:
    """The mask's pixels in ``color`` (cached on the geometry, which never changes)."""
    cache = g.__dict__.setdefault("_qimages", {})
    key = (color.rgb(), alpha)
    img = cache.get(key)
    if img is None:
        h, w = g.bits.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[g.bits] = (color.red(), color.green(), color.blue(), alpha)
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        if len(cache) > 4:
            cache.clear()
        cache[key] = img
    return img


def _paint_mask(painter: QPainter, g: Mask, color: QColor, width: float, style: str, fill_alpha: int,
                selected: bool) -> None:
    if fill_alpha:
        painter.drawImage(QRectF(*g.bbox()[:2], *g.bits.shape[::-1]), _mask_image(g, color, min(255, 2 * fill_alpha)))
    t = painter.worldTransform()
    path = QPainterPath()
    outer, holes = g.contours
    for c in outer + holes:
        path.addPolygon(QPolygonF([t.map(QPointF(float(x), float(y))) for x, y in c]))
        path.closeSubpath()
    painter.save()
    painter.resetTransform()
    pen = QPen(color, width + (1.2 if selected else 0.0))
    pen.setCosmetic(True)
    pen.setStyle({"dash": Qt.PenStyle.DashLine, "dot": Qt.PenStyle.DotLine}.get(style, Qt.PenStyle.SolidLine))
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.restore()


def paint_geometry(painter: QPainter, g: Geometry, color: QColor, *, width: float = 1.6, style: str = "solid",
                   fill_alpha: int = 0, directed: bool = False, selected: bool = False,
                   orientation: Optional[float] = None) -> None:
    """Paint ``g`` (in scene coordinates, through the painter's transform) in device coordinates."""
    if isinstance(g, Mask):
        _paint_mask(painter, g, color, width, style, fill_alpha, selected)
        if orientation is None:
            return
    t = painter.worldTransform()
    pts = _outline(g)
    painter.save()
    painter.resetTransform()
    pen = QPen(color, width + (1.2 if selected else 0.0))
    pen.setCosmetic(True)
    pen.setStyle({"solid": Qt.PenStyle.SolidLine, "dash": Qt.PenStyle.DashLine,
                  "dot": Qt.PenStyle.DotLine}.get(style, Qt.PenStyle.SolidLine))
    painter.setPen(pen)
    poly = QPolygonF([t.map(QPointF(float(x), float(y))) for x, y in pts])
    if isinstance(g, Point):
        c = poly[0]
        r = 4.5 + (1.5 if selected else 0)
        painter.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
        painter.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
        painter.drawEllipse(c, r * 0.6, r * 0.6)
    elif isinstance(g, Mask):
        pass                                              # drawn above; only its orientation arrow here
    elif g.closed:
        if fill_alpha:
            fc = QColor(color)
            fc.setAlpha(fill_alpha)
            painter.setBrush(QBrush(fc))
        painter.drawPolygon(poly)
    else:
        painter.drawPolyline(poly)
        if directed and len(poly) >= 2:
            painter.setBrush(QBrush(color))
            solid = QPen(pen)
            solid.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(solid)
            _arrow_head(painter, poly[len(poly) - 1], poly[len(poly) - 2])
    if orientation is not None and g.closed:
        cx, cy = g.centroid()
        c = t.map(QPointF(cx, cy))
        b = poly.boundingRect()
        r = max(8.0, 0.45 * max(b.width(), b.height()))
        a = math.radians(orientation)
        tip = QPointF(c.x() + r * math.cos(a), c.y() - r * math.sin(a))
        solid = QPen(color, width)
        solid.setCosmetic(True)
        painter.setPen(solid)
        painter.setBrush(QBrush(color))
        painter.drawLine(c, tip)
        _arrow_head(painter, tip, c, 7.0)
    if selected:
        painter.setPen(QPen(QColor("white"), 1.0))
        painter.setBrush(QBrush(color))
        for x, y in handles(g):
            p = t.map(QPointF(float(x), float(y)))
            painter.drawRect(QRectF(p.x() - HANDLE, p.y() - HANDLE, 2 * HANDLE, 2 * HANDLE))
    painter.restore()


class AnnotationItem(QGraphicsItem):
    def __init__(self, layer: "AnnotationLayer", ann: Annotation):
        super().__init__()
        self.layer = layer
        self.ann = ann
        self.selected = False
        self._bbox = QRectF()
        self.set_annotation(ann)

    def set_annotation(self, ann: Annotation) -> None:
        self.prepareGeometryChange()
        self.ann = ann
        x0, y0, x1, y1 = ann.geometry.bbox()
        self._bbox = QRectF(x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))
        area = max(ann.geometry.area(), 0.0)
        self.setZValue(10.0 - math.log1p(area) if ann.geometry.closed else 20.0)
        self.update()

    def boundingRect(self) -> QRectF:
        m = self.layer.margin
        if self.ann.orientation is not None:
            m = max(m, 0.5 * max(self._bbox.width(), self._bbox.height()) + m)
        r = self._bbox.adjusted(-m, -m, m, m)
        if self.layer.show_labels and self.ann.label and self.ann.kind == "feature":
            u = self.layer.margin / 10.0                                    # scene units per screen pixel
            r = r.united(QRectF(self._bbox.x(), self._bbox.y() - 20 * u, (9 * len(self._label_text()) + 10) * u, 20 * u))
        if self.ann.kind == "lattice_vector":                               # room for the 'a1 · reference' tag
            u = self.layer.margin / 10.0
            r = r.adjusted(-30 * u, -20 * u, 130 * u, 20 * u)
        return r

    def _label_text(self) -> str:
        a = self.ann
        conf = getattr(a.origin, "confidence", None)
        return str(a.label or "") + (f" {conf:.2f}" if isinstance(conf, (int, float)) else "")

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ANN001
        a = self.ann
        layer = self.layer
        if a.status in layer.session_hidden():
            return
        kind = KINDS.get(a.kind)
        if a.kind == "reference_axis":
            color = QColor(REFERENCE)
        elif a.status == "rejected":
            color = QColor(REJECTED)
        elif a.label is None and kind is not None and kind.color:
            color = QColor(kind.color)
        else:
            color = qcolor(layer.session.label_color(a.label))
        style = {"proposed": "dash", "rejected": "dot"}.get(a.status, "solid")
        if a.kind == "measurement_line":
            style = "solid"
            color = QColor("#00E5FF")
        fill = layer.fill_alpha if a.kind == "feature" and a.status != "rejected" else 0
        paint_geometry(painter, a.geometry, color, style=style, fill_alpha=fill,
                       directed=bool(kind and kind.directed), selected=self.selected,
                       orientation=a.orientation if layer.show_orientation and a.kind == "feature" else None)
        if a.kind == "lattice_vector" and len(a.geometry.points) >= 2:
            self._paint_lattice_tag(painter, color)
        if layer.show_labels and a.label and a.kind == "feature":
            t = painter.worldTransform()
            x0, y0, _, _ = a.geometry.bbox()
            p = t.map(QPointF(x0, y0))
            painter.save()
            painter.resetTransform()
            painter.setPen(QPen(color))
            painter.drawText(QPointF(p.x(), p.y() - 3), self._label_text())
            painter.restore()


    def _paint_lattice_tag(self, painter: QPainter, color: QColor) -> None:
        """'a1' / 'a2' at the head of a lattice vector, and 'reference' on the one orientations use."""
        from ..measurements.report import reference_axis
        a = self.ann
        role = a.attrs.get("role") or ""
        tag = role
        try:
            ref = reference_axis(self.layer.session.store.for_image(a.image_id))
        except Exception:  # noqa: BLE001
            ref = None
        if ref is not None and ref.id == a.id:
            tag = (tag + " · reference").strip(" ·")
        name = str(a.attrs.get("lattice", ""))
        if name and name != "lattice 1":
            tag = f"{tag} ({name})" if tag else name
        if not tag:
            return
        t = painter.worldTransform()
        (x0, y0), (x1, y1) = a.geometry.points[0], a.geometry.points[-1]
        p = t.map(QPointF(float(x1), float(y1)))
        q = t.map(QPointF(float(x0), float(y0)))
        d = QPointF(p.x() - q.x(), p.y() - q.y())
        n = max((d.x() ** 2 + d.y() ** 2) ** 0.5, 1e-6)
        painter.save()
        painter.resetTransform()
        painter.setPen(QPen(color))
        painter.drawText(QPointF(p.x() + 6 * d.x() / n - 4, p.y() + 6 * d.y() / n + 4), tag)
        painter.restore()


class PreviewItem(QGraphicsItem):
    """The active tool's in-progress shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.previews: list[Preview] = []
        self._rect = QRectF()
        self.setZValue(1000)

    def set_previews(self, previews: list[Preview], margin: float) -> None:
        self.prepareGeometryChange()
        self.previews = list(previews)
        r = QRectF()
        for p in self.previews:
            x0, y0, x1, y1 = p.geometry.bbox()
            r = r.united(QRectF(x0, y0, max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)))
        self._rect = r.adjusted(-margin, -margin, margin, margin)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ANN001
        for p in self.previews:
            if p.style == "band":
                paint_geometry(painter, p.geometry, QColor("#3daee9"), width=1.0, style="dash", fill_alpha=30)
            elif p.style == "mask":
                paint_geometry(painter, p.geometry, qcolor(p.color or "#ffffff"), width=1.2, fill_alpha=60)
            elif p.style == "edit":
                paint_geometry(painter, p.geometry, QColor("#ffffff"), width=1.2, style="dash")
            else:
                paint_geometry(painter, p.geometry, qcolor(p.color or "#ffffff"), width=1.6, style="solid")


class AnnotationLayer:
    """Keeps one item per annotation of the current image in sync with the store."""

    def __init__(self, scene: "QGraphicsScene", session: "Session"):
        self.scene = scene
        self.session = session
        self.items: dict[str, AnnotationItem] = {}
        self.image_id = ""
        self.margin = 2.0                 # scene units; updated with the zoom
        self.fill_alpha = 28
        self.show_labels = False
        self.show_orientation = True
        session.annotationsChanged.connect(self.on_event)
        session.selectionChanged.connect(self.on_selection)

    def session_hidden(self) -> set:
        return getattr(self.session, "hidden_statuses", set())

    def set_image(self, image_id: str) -> None:
        self.image_id = image_id
        self.rebuild()

    def rebuild(self) -> None:
        for it in self.items.values():
            self.scene.removeItem(it)
        self.items = {}
        if self.image_id:
            for a in self.session.store.for_image(self.image_id):
                self._add(a)
        self.on_selection()

    def _add(self, a: Annotation) -> None:
        it = AnnotationItem(self, a)
        self.items[a.id] = it
        self.scene.addItem(it)

    def on_event(self, event: StoreEvent) -> None:
        if self.image_id not in event.image_ids:
            return
        if event.source == "load" or len(event.changes) > 2000:
            self.rebuild()
            return
        for c in event.changes:
            if c.before is not None and c.before.image_id == self.image_id and \
                    (c.after is None or c.after.image_id != self.image_id):
                it = self.items.pop(c.before.id, None)
                if it is not None:
                    self.scene.removeItem(it)
            if c.after is not None and c.after.image_id == self.image_id:
                it = self.items.get(c.after.id)
                if it is None:
                    self._add(c.after)
                else:
                    it.set_annotation(c.after)
        if any(a is not None and a.kind == "lattice_vector" for c in event.changes for a in (c.before, c.after)):
            for it in self.items.values():                 # the 'reference' tag may have moved
                if it.ann.kind == "lattice_vector":
                    it.update()
        self.on_selection()

    def on_selection(self) -> None:
        sel = set(self.session.selection)
        for i, it in self.items.items():
            s = i in sel
            if s != it.selected:
                it.selected = s
                it.update()

    def set_margin(self, margin: float) -> None:
        if abs(margin - self.margin) / max(self.margin, 1e-9) < 0.05:
            return
        for it in self.items.values():
            it.prepareGeometryChange()
        self.margin = margin

    def refresh(self) -> None:
        for it in self.items.values():
            it.update()

    def set_option(self, attr: str, value) -> None:  # noqa: ANN001
        """Change a display setting (show_labels, fill_alpha, ...) and repaint, keeping the items
        (rebuilding them all at once from a menu toggle crashed Qt with many objects).  Qt must hear
        of a bounding box change before it happens."""
        for it in self.items.values():
            it.prepareGeometryChange()
        setattr(self, attr, value)
        for it in self.items.values():
            it.update()
