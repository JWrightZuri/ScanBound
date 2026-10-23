"""
The image canvas: one image, its annotations, the active tool.

Scene coordinates are image pixel coordinates (pixel (c, r) covers [c, c+1] x [r, r+1]), whatever
the zoom.  Non-square pixels are shown with their true aspect through the view transform, so the
picture looks like the surface and angles on screen are physical angles.  Pixels without data are
transparent (the dark checker background shows through).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPen, QPixmap, QTransform
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..display import DisplaySettings, render_rgba
from ..model import Image
from ..model.units import format_value
from .layer import AnnotationLayer, PreviewItem
from .qtutil import to_qimage
from .session import Session
from .tools import TOOLS, Tool

_KEYS = {Qt.Key.Key_Escape: "escape", Qt.Key.Key_Return: "return", Qt.Key.Key_Enter: "enter",
         Qt.Key.Key_Backspace: "backspace", Qt.Key.Key_Delete: "delete", Qt.Key.Key_Left: "left",
         Qt.Key.Key_Right: "right", Qt.Key.Key_Up: "up", Qt.Key.Key_Down: "down",
         Qt.Key.Key_BracketLeft: "[", Qt.Key.Key_BracketRight: "]"}
_CURSORS = {"cross": Qt.CursorShape.CrossCursor, "arrow": Qt.CursorShape.ArrowCursor,
            "open_hand": Qt.CursorShape.OpenHandCursor}


def _mods(m: Qt.KeyboardModifier) -> frozenset:
    out = set()
    if m & Qt.KeyboardModifier.ShiftModifier:
        out.add("shift")
    if m & Qt.KeyboardModifier.ControlModifier:
        out.add("ctrl")
    if m & Qt.KeyboardModifier.AltModifier:
        out.add("alt")
    return frozenset(out)


def _checker() -> QBrush:
    pm = QPixmap(16, 16)
    pm.fill(QColor("#2b2b2b"))
    p = QPainter(pm)
    p.fillRect(0, 0, 8, 8, QColor("#353535"))
    p.fillRect(8, 8, 8, 8, QColor("#353535"))
    p.end()
    return QBrush(pm)


class ImageCanvas(QGraphicsView):
    cursorMoved = pyqtSignal(object)            # dict: x, y (pixels), physical text, value text; or None
    zoomChanged = pyqtSignal(float)
    contextRequested = pyqtSignal(QPointF, object)   # scene position, global QPoint
    toolChanged = pyqtSignal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setBackgroundBrush(_checker())
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.FastTransformation)   # true pixels
        self.pixmap_item.setZValue(-100)
        self.scene_.addItem(self.pixmap_item)
        self.preview_item = PreviewItem()
        self.scene_.addItem(self.preview_item)
        self.layer = AnnotationLayer(self.scene_, session)
        self.overlays: list = []                  # extra items (spectrum positions ...)

        self.image: Optional[Image] = None
        self.zoom = 1.0
        self.aspect = 1.0                         # displayed height / width of one pixel
        self.show_scale_bar = True
        self.show_colorbar = False
        # overlay boxes can be dragged; positions are the box centre as a fraction of the view
        self.overlay_pos: dict[str, tuple[float, float]] = dict(OVERLAY_DEFAULTS)
        saved = QSettings("ScanBound", "ScanBound").value("overlay_pos")
        if isinstance(saved, dict):
            for k, v in saved.items():
                if k in self.overlay_pos and isinstance(v, (list, tuple)) and len(v) == 2:
                    self.overlay_pos[k] = (float(v[0]), float(v[1]))
        self._overlay_rects: dict[str, QRectF] = {}
        self._overlay_drag: Optional[tuple[str, QPointF]] = None
        self.smooth = False
        self.override: Optional[np.ndarray] = None
        self._pan: Optional[QPointF] = None
        self._space = False
        self._auto_fit = True                     # fit on resize until the user zooms
        self.tools: dict[str, Tool] = {k: cls(session, self.tolerance) for k, cls in TOOLS.items()}
        self.tool: Tool = self.tools["select"]

        session.imageChanged.connect(self._on_image_changed)
        session.imageDataChanged.connect(lambda i: i == session.current_id and self.reload())
        session.displayChanged.connect(lambda i, final: i == session.current_id and self.render())

    # ------------------------------------------------------------------ tools
    def tolerance(self) -> float:
        return 6.0 / max(self.zoom, 1e-9)

    def set_tool(self, key: str) -> None:
        self.tool.cancel()
        self.tool = self.tools[key]
        self.viewport().setCursor(QCursor(_CURSORS.get(self.tool.cursor, Qt.CursorShape.CrossCursor)))
        self._sync_preview()
        self.toolChanged.emit(key)

    def _sync_preview(self) -> None:
        self.preview_item.set_previews(self.tool.preview, 12.0 / max(self.zoom, 1e-9))

    # ------------------------------------------------------------------ image
    def _on_image_changed(self, image_id: str) -> None:
        self.tool.cancel()
        self._sync_preview()
        first = self.image is None
        old_shape = self.image.shape if self.image is not None else None
        self.reload()
        self.layer.set_image(image_id)
        if self.image is not None and (first or self.image.shape != old_shape):
            self.fit()

    def reload(self) -> None:
        try:
            self.image = self.session.image()
        except Exception as exc:  # noqa: BLE001 - shown, the app keeps running
            self.image = None
            self.session.error(f"Cannot show {self.session.current_id}: {exc}")
        if self.image is None:
            self.pixmap_item.setPixmap(QPixmap())
            self.scene_.setSceneRect(QRectF(0, 0, 1, 1))
            return
        h, w = self.image.shape
        dx, dy = self.image.pixel_size
        aspect = dy / dx if dx > 0 and dy > 0 and self.image.calibrated else 1.0
        if abs(aspect - self.aspect) > 1e-9:
            self.aspect = aspect
            self._apply_transform()
        m = max(w, h) * 0.25
        self.scene_.setSceneRect(QRectF(-m, -m, w + 2 * m, h + 2 * m))
        self.render()

    def show_rgb(self, rgb: Optional[np.ndarray]) -> None:
        """Show this picture instead of rendering the image data (same shape; None goes back)."""
        self.override = None if rgb is None else np.ascontiguousarray(rgb)
        self.render()

    def render(self) -> None:
        if self.image is None:
            return
        if self.override is not None:
            self.pixmap_item.setPixmap(QPixmap.fromImage(to_qimage(self.override)))
            return
        entry = self.session.entry
        s = entry.display if entry is not None else DisplaySettings()
        stats = self.session.display_stats()
        rgba = render_rgba(self.image.data, stats, s)
        self.pixmap_item.setPixmap(QPixmap.fromImage(to_qimage(rgba)))
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation if self.smooth
                                               else Qt.TransformationMode.FastTransformation)

    def rendered_rgba(self) -> Optional[np.ndarray]:
        if self.image is None:
            return None
        entry = self.session.entry
        return render_rgba(self.image.data, self.session.display_stats(), entry.display if entry else DisplaySettings())

    # ------------------------------------------------------------------ zoom
    def _apply_transform(self) -> None:
        self.setTransform(QTransform.fromScale(self.zoom, self.zoom * self.aspect))
        self.layer.set_margin(10.0 / max(self.zoom, 1e-9))
        self._sync_preview()
        self.zoomChanged.emit(self.zoom)

    def set_zoom(self, z: float, anchor: Optional[QPointF] = None) -> None:
        self._auto_fit = False
        z = float(min(max(z, 0.01), 400.0))
        if anchor is None:
            anchor = self.mapToScene(self.viewport().rect().center())
        self.zoom = z
        self._apply_transform()
        self.centerOn(anchor)

    def fit(self) -> None:
        self._auto_fit = True
        if self.image is None:
            return
        h, w = self.image.shape
        # the size without scroll bars: measuring the viewport itself made some images flip the bars on and
        # off forever (each resize refitted, which toggled the bars, which resized the viewport ...)
        size = self.maximumViewportSize()
        vw, vh = max(size.width() - 8, 10), max(size.height() - 8, 10)
        zoom = max(0.01, min(vw / w, vh / (h * self.aspect)))
        if QTransform.fromScale(zoom, zoom * self.aspect) != self.transform():
            self.zoom = zoom
            self._apply_transform()
        self.centerOn(QPointF(w / 2, h / 2))

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit()

    def wheelEvent(self, event) -> None:  # noqa: ANN001
        self._auto_fit = False
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        anchor = self.mapToScene(event.position().toPoint())
        self.zoom = float(min(max(self.zoom * (1.25 ** steps), 0.01), 400.0))
        self._apply_transform()
        # keep the point under the mouse in place
        new = self.mapToScene(event.position().toPoint())
        d = new - anchor
        self.translate_view(d.x(), d.y())

    def translate_view(self, dx: float, dy: float) -> None:
        c = self.mapToScene(self.viewport().rect().center())
        self.centerOn(QPointF(c.x() - dx, c.y() - dy))

    # ------------------------------------------------------------------ mouse
    def _scene(self, event) -> QPointF:  # noqa: ANN001
        return self.mapToScene(event.position().toPoint())

    # ------------------------------------------------------------------ movable overlays
    def _overlay_at(self, pos: QPointF) -> Optional[str]:
        for key, r in self._overlay_rects.items():
            if r.contains(pos):
                return key
        return None

    def _place(self, key: str, w: float, h: float) -> QRectF:
        """The rectangle of an overlay box of size w × h, kept inside the view."""
        vw, vh = self.viewport().width(), self.viewport().height()
        fx, fy = self.overlay_pos.get(key, OVERLAY_DEFAULTS[key])
        x = min(max(fx * vw - w / 2, 2.0), max(vw - w - 2, 2.0))
        y = min(max(fy * vh - h / 2, 2.0), max(vh - h - 2, 2.0))
        r = QRectF(x, y, w, h)
        self._overlay_rects[key] = r
        return r

    def reset_overlays(self) -> None:
        self.overlay_pos = dict(OVERLAY_DEFAULTS)
        QSettings("ScanBound", "ScanBound").setValue("overlay_pos", {k: list(v) for k, v in self.overlay_pos.items()})
        self.viewport().update()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        btn = event.button()
        if btn == Qt.MouseButton.LeftButton:
            key = self._overlay_at(event.position())
            if key is not None:                    # drag the scale bar / colour bar
                r = self._overlay_rects[key]
                self._overlay_drag = (key, event.position() - r.center())
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return
        if btn == Qt.MouseButton.MiddleButton or (btn == Qt.MouseButton.LeftButton and
                                                  (self._space or self.tool.name == "pan")):
            self._pan = event.position()
            self._auto_fit = False
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        p = self._scene(event)
        name = "left" if btn == Qt.MouseButton.LeftButton else "right" if btn == Qt.MouseButton.RightButton else ""
        if name and self.tool.press(p.x(), p.y(), name, _mods(event.modifiers())):
            self._sync_preview()
            return
        if btn == Qt.MouseButton.RightButton:
            # open the menu after this event: a nested menu loop inside a mouse handler, with the view
            # changing under it (Fit to window ...), can bring Qt down on Windows
            gp = event.globalPosition().toPoint()
            QTimer.singleShot(0, lambda: self.contextRequested.emit(p, gp))

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._overlay_drag is not None:
            key, off = self._overlay_drag
            c = event.position() - off
            vw, vh = max(self.viewport().width(), 1), max(self.viewport().height(), 1)
            self.overlay_pos[key] = (min(max(c.x() / vw, 0.0), 1.0), min(max(c.y() / vh, 0.0), 1.0))
            self.viewport().update()
            return
        if not event.buttons() and self._overlay_at(event.position()) is not None:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        elif not event.buttons() and self._pan is None and not self._space:
            self.viewport().setCursor(QCursor(_CURSORS.get(self.tool.cursor, Qt.CursorShape.CrossCursor)))
        if self._pan is not None:
            d = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        p = self._scene(event)
        if self.tool.move(p.x(), p.y(), _mods(event.modifiers())):
            self._sync_preview()
        self._report_cursor(p)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._overlay_drag is not None:
            self._overlay_drag = None
            QSettings("ScanBound", "ScanBound").setValue("overlay_pos",
                                                         {k: list(v) for k, v in self.overlay_pos.items()})
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if self._pan is not None:
            self._pan = None
            self.viewport().setCursor(QCursor(_CURSORS.get(self.tool.cursor, Qt.CursorShape.CrossCursor)))
            return
        p = self._scene(event)
        btn = event.button()
        name = "left" if btn == Qt.MouseButton.LeftButton else "right" if btn == Qt.MouseButton.RightButton else ""
        if name and self.tool.release(p.x(), p.y(), name, _mods(event.modifiers())):
            self._sync_preview()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton:
            return
        key = self._overlay_at(event.position())
        if key is not None:                          # double-click an overlay: back to its corner
            self.overlay_pos[key] = OVERLAY_DEFAULTS[key]
            QSettings("ScanBound", "ScanBound").setValue("overlay_pos",
                                                         {k: list(v) for k, v in self.overlay_pos.items()})
            self.viewport().update()
            return
        p = self._scene(event)
        if self.tool.double_click(p.x(), p.y(), _mods(event.modifiers())):
            self._sync_preview()

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        self.cursorMoved.emit(None)
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        name = _KEYS.get(event.key())
        if name and self.tool.key(name, _mods(event.modifiers())):
            self._sync_preview()
            return
        if name == "escape" and self.session.selection:     # any tool: Esc deselects
            self.session.select([])
            return
        if name in ("delete", "backspace"):        # a drawing tool deletes the hovered shape (Mac: Backspace)
            pos = self.viewport().mapFromGlobal(QCursor.pos())
            if self.viewport().rect().contains(pos):
                p = self.mapToScene(pos)
                if self.tool.delete_at(p.x(), p.y()):
                    return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.viewport().setCursor(QCursor(_CURSORS.get(self.tool.cursor, Qt.CursorShape.CrossCursor)))
            return
        super().keyReleaseEvent(event)

    def _report_cursor(self, p: QPointF) -> None:
        img = self.image
        if img is None:
            return
        h, w = img.shape
        x, y = p.x(), p.y()
        if not (0 <= x < w and 0 <= y < h):
            self.cursorMoved.emit(None)
            return
        c, r = int(x), int(y)
        v = float(img.data[r, c])
        info = {"x": x, "y": y, "col": c, "row": r,
                "value": "no data" if not math.isfinite(v) else format_value(v, img.unit, 4)}
        if img.calibrated:
            info["physical"] = (format_value(img.x.to_physical(x), img.x.unit, 4) + ", "
                                + format_value(img.y.to_physical(y), img.y.unit, 4))
        self.cursorMoved.emit(info)

    # ------------------------------------------------------------------ overlay
    def _draw_colorbar(self, painter: QPainter) -> None:
        from ..display import levels
        entry = self.session.entry
        stats = self.session.display_stats()
        if entry is None or stats is None:
            return
        s = entry.display
        lo, hi = levels(stats, s)
        painter.save()
        painter.resetTransform()
        vh = self.viewport().height()
        h = int(min(260, vh * 0.45))
        box = self._place("colorbar", 92, h + 12)
        x, y = box.x() + 6, box.y() + 6
        painter.fillRect(box, QColor(0, 0, 0, 120))
        painter.drawPixmap(QRectF(x, y, 14, h).toRect(), _colorbar_pixmap(s.cmap, s.reverse))
        painter.setPen(QPen(QColor("white")))
        f = QFont()
        f.setPointSizeF(8.5)
        painter.setFont(f)
        unit = self.image.unit
        for t, v in ((0.0, hi), (0.5, (lo + hi) / 2), (1.0, lo)):
            painter.drawText(QPointF(x + 20, y + t * h + 4), format_value(v, unit, 4))
        painter.restore()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        self._overlay_rects = {}
        if self.show_colorbar and self.image is not None and self.override is None:
            self._draw_colorbar(painter)
        if not self.show_scale_bar or self.image is None or not self.image.calibrated:
            return
        painter.save()
        painter.resetTransform()
        vw, vh = self.viewport().width(), self.viewport().height()
        dx = self.image.pixel_size[0]
        unit = self.image.x.unit
        screen_per_unit = self.zoom / dx                 # screen px per physical unit
        target = 0.18 * vw / screen_per_unit
        length = _nice(target)
        bar = length * screen_per_unit
        box = self._place("scale_bar", bar + 12, 30)
        x0, y = box.x() + 6, box.y() + 20
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(box, QColor(0, 0, 0, 120))
        painter.fillRect(QRectF(x0, y, bar, 4), QColor("white"))
        painter.setPen(QPen(QColor("white")))
        f = QFont()
        f.setPointSizeF(9)
        painter.setFont(f)
        painter.drawText(QRectF(x0 - 6, y - 19, bar + 12, 16), Qt.AlignmentFlag.AlignCenter,
                         format_value(length, unit, 3))
        painter.restore()


OVERLAY_DEFAULTS = {"scale_bar": (0.86, 0.95), "colorbar": (0.06, 0.25)}   # box centres, fraction of the view


def _colorbar_pixmap(cmap: str, reverse: bool) -> QPixmap:
    from ..display import lut
    table = lut(cmap, reverse)[::-1]                       # top = high values
    arr = np.ascontiguousarray(np.repeat(table[:, None, :], 12, axis=1))
    return QPixmap.fromImage(to_qimage(arr))


def _nice(v: float) -> float:
    if v <= 0 or not math.isfinite(v):
        return 1.0
    e = math.floor(math.log10(v))
    for m in (5, 2, 1):
        if m * 10 ** e <= v:
            return m * 10 ** e
    return 10 ** e
