"""Small line icons for the annotation tools, drawn with QPainter in the current text colour (so they
suit light and dark themes and stay sharp at any screen scaling)."""

from __future__ import annotations

import math
from typing import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QApplication

SIZE = 20                                     # logical px; drawn at 2× / 3× for high-DPI screens


def _arrow_head(p: QPainter, tip: QPointF, angle: float, size: float = 4.5) -> None:
    a1, a2 = angle + math.radians(150), angle - math.radians(150)
    p.drawPolygon(QPolygonF([tip, tip + QPointF(size * math.cos(a1), size * math.sin(a1)),
                             tip + QPointF(size * math.cos(a2), size * math.sin(a2))]))


def _select(p: QPainter) -> None:
    pts = [(5, 3), (5, 16), (8.5, 12.5), (11, 18), (13, 17), (10.5, 11.5), (15.5, 11.5)]
    p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))


def _pan(p: QPainter) -> None:
    c = QPointF(10, 10)
    for ang in (0, 90, 180, 270):
        a = math.radians(ang)
        tip = c + QPointF(7.5 * math.cos(a), 7.5 * math.sin(a))
        p.drawLine(c, tip)
        _arrow_head(p, tip, a, 3.5)


def _rect(p: QPainter) -> None:
    p.drawRect(QRectF(3.5, 5.5, 13, 9))


def _ellipse(p: QPainter) -> None:
    p.drawEllipse(QRectF(3, 5, 14, 10))


def _polygon(p: QPainter) -> None:
    pts = [QPointF(10, 3.5), QPointF(16.5, 8.5), QPointF(14, 16.5), QPointF(6, 16.5), QPointF(3.5, 8.5)]
    p.drawPolygon(QPolygonF(pts))
    _dots(p, pts)


def _point(p: QPainter) -> None:
    p.drawLine(QPointF(10, 3), QPointF(10, 7))
    p.drawLine(QPointF(10, 13), QPointF(10, 17))
    p.drawLine(QPointF(3, 10), QPointF(7, 10))
    p.drawLine(QPointF(13, 10), QPointF(17, 10))
    _dots(p, [QPointF(10, 10)], 2.0)


def _centre_line(p: QPainter) -> None:
    pts = [QPointF(3, 15), QPointF(8, 7), QPointF(12.5, 12), QPointF(17, 4.5)]
    p.drawPolyline(QPolygonF(pts))
    _dots(p, pts)


def _measure(p: QPainter) -> None:
    p.drawLine(QPointF(3, 13), QPointF(17, 13))
    p.drawLine(QPointF(3, 9.5), QPointF(3, 16.5))
    p.drawLine(QPointF(17, 9.5), QPointF(17, 16.5))
    for x in (6.5, 10, 13.5):
        p.drawLine(QPointF(x, 13), QPointF(x, 10.8))
    p.drawLine(QPointF(5, 5.5), QPointF(15, 5.5))
    _arrow_head(p, QPointF(15.5, 5.5), 0.0, 3.0)
    _arrow_head(p, QPointF(4.5, 5.5), math.pi, 3.0)


def _orientation(p: QPainter) -> None:
    p.drawEllipse(QRectF(4, 6.5, 12, 7))
    tip = QPointF(17.5, 4)
    p.drawLine(QPointF(2.5, 16), tip)
    _arrow_head(p, tip, math.atan2(4 - 16, 17.5 - 2.5), 4.0)


def _lattice(p: QPainter) -> None:
    o = QPointF(4, 16)
    for ang in (-8.0, -68.0):
        a = math.radians(ang)
        tip = o + QPointF(12.5 * math.cos(a), 12.5 * math.sin(a))
        p.drawLine(o, tip)
        _arrow_head(p, tip, a, 3.8)
    _dots(p, [o], 1.6)


def _step_edge(p: QPainter) -> None:
    # two terraces seen from the side, the edge marked by a dot
    p.drawPolyline(QPolygonF([QPointF(2.5, 15.5), QPointF(9, 15.5), QPointF(9, 8.5), QPointF(17.5, 8.5)]))
    pen = p.pen()
    dashed = QPen(pen)
    dashed.setDashPattern([1.5, 1.5])
    p.setPen(dashed)
    p.drawLine(QPointF(9, 3), QPointF(9, 17.5))
    p.setPen(pen)


def _region(p: QPainter) -> None:
    pen = p.pen()
    dashed = QPen(pen)
    dashed.setDashPattern([2.0, 1.5])
    p.setPen(dashed)
    pts = [QPointF(3, 6), QPointF(12, 3), QPointF(17, 10), QPointF(11, 17), QPointF(4, 14)]
    p.drawPolygon(QPolygonF(pts))
    p.setPen(pen)
    _dots(p, [QPointF(8, 9), QPointF(12, 11), QPointF(9, 13)], 1.1)


def _herringbone(p: QPainter) -> None:
    # two zig-zag ridge pairs, like the reconstruction, with the marked direction as an arrow
    pen = p.pen()
    thin = QPen(pen)
    thin.setWidthF(max(pen.widthF() * 0.7, 1.0))
    p.setPen(thin)
    for dy in (0.0, 4.0):
        p.drawPolyline(QPolygonF([QPointF(2.5, 4 + dy), QPointF(8, 9 + dy), QPointF(13, 4 + dy), QPointF(17.5, 8.5 + dy)]))
    p.setPen(pen)
    p.drawLine(QPointF(3, 17.5), QPointF(16, 17.5))
    _arrow_head(p, QPointF(16.5, 17.5), 0.0, 3.2)


def _paint(p: QPainter) -> None:
    # a brush over a blob of painted pixels
    brush = p.brush()
    fill = QColor(p.pen().color())
    fill.setAlpha(110)
    p.setBrush(fill)
    p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in
                             ((3, 11), (7, 11), (7, 9), (10, 9), (10, 13), (8, 13), (8, 17), (3, 17))]))
    p.setBrush(brush)
    p.drawLine(QPointF(10.5, 9.5), QPointF(16.5, 3.5))
    p.drawLine(QPointF(9.5, 8.0), QPointF(12.0, 10.5))


def _dots(p: QPainter, pts, r: float = 1.3) -> None:  # noqa: ANN001
    brush = p.brush()
    p.setBrush(p.pen().color())
    for q in pts:
        p.drawEllipse(q, r, r)
    p.setBrush(brush)


PAINTERS: dict[str, Callable[[QPainter], None]] = {
    "select": _select, "pan": _pan, "rect": _rect, "ellipse": _ellipse, "polygon": _polygon, "point": _point,
    "centre_line": _centre_line, "measure": _measure, "orientation": _orientation,
    "lattice": _lattice, "step_edge": _step_edge, "region": _region, "herringbone": _herringbone,
    "paint": _paint,
}


def tool_icon(key: str, color: QColor | None = None) -> QIcon:
    """The icon of a tool, in ``color`` (default: the theme's text colour)."""
    paint = PAINTERS.get(key)
    icon = QIcon()
    if paint is None:
        return icon
    if color is None:
        app = QApplication.instance()
        color = app.palette().buttonText().color() if app is not None else QColor("black")
    for scale in (1, 2, 3):
        pm = QPixmap(SIZE * scale, SIZE * scale)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(scale, scale)
        pen = QPen(color, 1.4)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if key == "select":                                     # a filled pointer reads best
            p.setBrush(color)
        paint(p)
        p.end()
        pm.setDevicePixelRatio(scale)
        icon.addPixmap(pm)
    return icon
