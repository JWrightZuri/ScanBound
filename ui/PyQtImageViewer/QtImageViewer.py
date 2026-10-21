""" QtImageViewer.py: PyQt image viewer widget based on QGraphicsView with mouse zooming/panning and ROIs.

ROI design
----------
* Every ROI is a plain QGraphicsPathItem with no Python paint/shape/boundingRect overrides,
  so Qt draws and spatially indexes them entirely in C++.  This keeps the view responsive
  with thousands of ROIs.
* Hover/selection highlights and vertex handles are drawn by one overlay item, and only for
  the few ROIs that need them (instead of one child item per polygon vertex).
* All interaction (picking, selection, moving, vertex editing, drawing) is handled by the
  viewer, using a BSP-indexed bounding-box query followed by an exact distance test, with a
  pick tolerance in screen pixels that is independent of zoom.
* The viewer owns the ROI state.  Each ROI has a stable string id (``roiId``, saved as
  ``objectName``) and a ``label``.  ``annotationsFrame()`` serialises all ROIs into the
  annotation table format and is cached until the ROIs change.
* Changes are reported once per operation through ``roisChanged(action, rois)`` with action
  in {"drawn", "added", "removed", "modified", "relabeled"}; bulk operations emit one signal.

Mouse / keyboard (ROI related)
------------------------------
Selection mode : click selects (Ctrl toggles), drag on empty space rubber-band selects,
                 drag a selected ROI to move it, drag a handle to move a vertex/corner,
                 right-click a handle removes that vertex, right-click elsewhere zooms out.
Draw modes     : left draws (polyline/polygon: click to add vertices, double-click or Enter
                 to finish, Backspace removes the last vertex, Esc / right-click cancels);
                 ROIs of the active type can be selected, moved and edited as above and are
                 deleted with right-click.
Any mode       : Delete removes selected ROIs, Esc clears the selection, Ctrl+A selects all
                 editable ROIs, double-click on a line/polygon edge inserts a vertex.
"""

import math
import os.path
import uuid

import numpy as np

try:
    from PyQt6.QtCore import Qt, QRectF, QRect, QPoint, QPointF, QLineF, pyqtSignal, QEvent, QSize
    from PyQt6.QtGui import QImage, QPixmap, QPainterPath, QMouseEvent, QPainter, QPen, QBrush, QColor, \
        QFont, QPolygonF
    from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QFileDialog, QSizePolicy, \
        QGraphicsItem, QGraphicsRectItem, QGraphicsPathItem, QGraphicsTextItem, QRubberBand
except ImportError:
    try:
        from PyQt5.QtCore import Qt, QRectF, QRect, QPoint, QPointF, QLineF, pyqtSignal, QEvent, QSize
        from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QMouseEvent, QPainter, QPen, QBrush, QColor, \
            QFont, QPolygonF
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QFileDialog, QSizePolicy, \
            QGraphicsItem, QGraphicsRectItem, QGraphicsPathItem, QGraphicsTextItem, QRubberBand
    except ImportError:
        raise ImportError("Requires PyQt (version 5 or 6)")

# qimage2ndarray is optional: useful for displaying numpy 2d arrays as images.
try:
    import qimage2ndarray
except ImportError:
    qimage2ndarray = None

__author__ = "Marcel Goldschen-Ohm <marcel.goldschen@gmail.com>"
__version__ = '3.0.0'


# =============================================================================
# Shared constants
# =============================================================================

EMPTY_LABEL = "Empty Label"
SYMMETRY_LABEL = "Symmetry Vector"
ANNOTATION_COLUMNS = ['objectName', 'type', 'x1', 'y1', 'dx', 'dy', 'file', 'label']
ANGLE_COLUMN = 'angle'          # ROI orientation in degrees (CCW from the image x axis), empty if unset
FRAME_COLUMNS = ANNOTATION_COLUMNS + [ANGLE_COLUMN]
ORIENT_MODE = "Orientation"     # drawROI value of the orientation tool

# Index 0 is used for "Empty Label" / unknown labels, label i of the label list uses 1 + i.
ROI_COLORS = [
    "#B9B1B4", "#D81B60", "#1E88E5", "#D49713", "#4CDBB0", "#D55E00", "#CC79A7", "#F0E442",
    "#56B4E9", "#009E73", "#E69F00", "#0072B2", "#F5E400", "#FFFFFF", "#000000",
]
VECTOR_COLOR = "#000000"
SYMMETRY_COLOR = "#F5E400"

ROI_PEN_WIDTH = 2       # cosmetic (screen pixels)
PICK_PX = 6             # pick tolerance around outlines (screen pixels)
HANDLE_PX = 4           # half size of a vertex handle (screen pixels)
MIN_DRAG_PX = 3         # mouse travel before a press becomes a drag (screen pixels)
MAX_HANDLE_ROIS = 25    # vertex handles are only shown when at most this many ROIs are selected


# HSL lightness offsets used to tell apart nearby ROIs of one label; step 0 is the label colour.
SHADE_STEPS = (0.0, 0.24, -0.20, 0.38, -0.32, 0.12, -0.10)


def shade_color(base, step):
    """`base` with its lightness shifted by SHADE_STEPS[step] (hue kept)."""
    offset = SHADE_STEPS[step % len(SHADE_STEPS)]
    if offset == 0:
        return QColor(base)
    h, s, l, a = base.getHslF()
    lightness = l + offset
    if not 0.12 <= lightness <= 0.92:          # no room on that side: go the other way
        lightness = l - offset
    return QColor.fromHslF(max(h, 0.0), s, min(max(lightness, 0.06), 0.95), a)


def label_color(label, label_list):
    """Colour (hex string) of a label, consistent between the viewer and label legends."""
    if label in label_list:
        return ROI_COLORS[1 + label_list.index(label) % (len(ROI_COLORS) - 1)]
    return ROI_COLORS[0]


def _new_id(roi_type):
    return f"{roi_type}_{uuid.uuid4().hex[:12]}"


def _as_points(points):
    """(N, 2) float array from an array, a list of (x, y) or a list of QPointF."""
    if points is None:
        return np.zeros((0, 2))
    if isinstance(points, np.ndarray):
        return points.astype(float).reshape(-1, 2).copy()
    out = [(p.x(), p.y()) if hasattr(p, "x") and callable(p.x) else (float(p[0]), float(p[1]))
           for p in points]
    return np.array(out, dtype=float).reshape(-1, 2)


def _dedupe(pts, eps=1e-3):
    """Drop consecutive (near) duplicate points."""
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), bool)
    keep[1:] = np.hypot(*np.diff(pts, axis=0).T) > eps
    return pts[keep]


def _segment_projection(pts, x, y, closed):
    """Per-segment (distance, t, a, b) from (x, y) to the polyline/polygon through pts."""
    if closed:
        a, b = pts, np.roll(pts, -1, axis=0)
    else:
        a, b = pts[:-1], pts[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / denom, 0.0, 1.0)
    px = a[:, 0] + t * ab[:, 0]
    py = a[:, 1] + t * ab[:, 1]
    return np.hypot(px - x, py - y), t, a, b


def _distance_to_outline(pts, x, y, closed):
    if len(pts) == 0:
        return math.inf
    if len(pts) == 1:
        return math.hypot(pts[0, 0] - x, pts[0, 1] - y)
    return float(_segment_projection(pts, x, y, closed)[0].min())


def _clean_label(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return EMPTY_LABEL
    text = str(value)
    return text if text and text.lower() != "nan" else EMPTY_LABEL


def _clean_name(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return text if text and text.lower() not in ("nan", "none") else None


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


# =============================================================================
# Viewer
# =============================================================================

class QtImageViewer(QGraphicsView):
    leftMouseButtonPressed = pyqtSignal(float, float)
    leftMouseButtonReleased = pyqtSignal(float, float)
    middleMouseButtonPressed = pyqtSignal(float, float)
    middleMouseButtonReleased = pyqtSignal(float, float)
    rightMouseButtonPressed = pyqtSignal(float, float)
    rightMouseButtonReleased = pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    rightMouseButtonDoubleClicked = pyqtSignal(float, float)

    viewChanged = pyqtSignal()
    mousePositionOnImageChanged = pyqtSignal(QPoint)

    # ROI signals
    roiSelected = pyqtSignal(list)          # user changed the selection: list of selected ROIs
    roisChanged = pyqtSignal(str, list)     # (action, rois), one emit per operation

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self._image = None
        self.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.regionZoomButton = Qt.MouseButton.LeftButton
        self.zoomOutButton = Qt.MouseButton.RightButton
        self.panButton = Qt.MouseButton.MiddleButton
        self.wheelZoomFactor = 1.25

        self.selectionButton = Qt.MouseButton.LeftButton
        self.zoomStack = []

        self._isPanning = False
        self._pixelPosition = QPoint()
        self._scenePosition = QPointF()

        # ROI state
        self._rois = {}             # roiId -> roi (insertion ordered = drawing order)
        self._selected = {}         # roiId -> roi
        self._hovered = None
        self._drawing = None        # ROI being drawn (not registered yet)
        self._drag = None           # active mouse drag (dict)
        self._drawROI = None
        self._labels = []
        self.distinguishROIs = True     # shades of the label colour for neighbouring ROIs
        self.showOrientations = True    # draw orientation arrows of ROIs that have an angle
        self._currentLabel = EMPTY_LABEL
        self._revision = 0
        self._frameCache = (None, None)
        self._pens = {}
        self._cursor = None
        self._pointRadius = 4.0     # scene units, updated from the image size
        self._arrowSize = 6.0       # scene units, updated from the image size

        self._band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
        self._bandOrigin = QPoint()

        self._overlay = _ROIOverlay(self)
        self.scene.addItem(self._overlay)

        self._imgShape = []
        self._pxScale = None
        self._imgPath = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        return QSize(900, 600)

    # ------------------------------------------------------------------ image

    def hasImage(self):
        return self._image is not None

    def clearImage(self):
        if self.hasImage():
            self.scene.removeItem(self._image)
            self._image = None

    def pixmap(self):
        if self.hasImage():
            return self._image.pixmap()
        return None

    def image(self):
        if self.hasImage():
            return self._image.pixmap().toImage()
        return None

    def returnImage(self):
        if self.hasImage():
            arr = qimage2ndarray.byte_view(self._image.pixmap().toImage())
            return arr
        return None

    def setImage(self, image):
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        elif type(image) is np.ndarray:
            if qimage2ndarray is not None:
                qimage = qimage2ndarray.array2qimage(image, True)
                pixmap = QPixmap.fromImage(qimage)
            else:
                image = image.astype(np.float32)
                image -= image.min()
                image /= image.max()
                image *= 255
                image[image > 255] = 255
                image[image < 0] = 0
                image = image.astype(np.uint8)
                height, width = image.shape
                bytes = image.tobytes()
                qimage = QImage(bytes, width, height, QImage.Format.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qimage)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage, QPixmap, or numpy.ndarray.")

        if self.hasImage():
            self._image.setPixmap(pixmap)
        else:
            self._image = self.scene.addPixmap(pixmap)

        rect = QRectF(pixmap.rect())
        self.setSceneRect(rect)
        size = max(rect.width(), rect.height(), 1.0)
        self._pointRadius = max(2.0, size * 0.008)
        self._arrowSize = max(3.0, size * 0.015)
        margin = size * 0.25 + 50
        self._overlay.setBounds(rect.adjusted(-margin, -margin, margin, margin))
        for roi in self._rois.values():
            if isinstance(roi, (PointROI, VectorROI)):
                roi._rebuild()
        self.updateViewer()

    def addTextOverlay(self, text, x=5, y=5, width=0, height=0, font_size=12,
                       color=Qt.GlobalColor.white):
        if not self.hasImage():
            return None

        text_item = QGraphicsTextItem(text)
        text_item.setPos(x, y)
        font = QFont("Arial", font_size, QFont.Weight.Bold)
        text_item.setFont(font)
        text_item.setDefaultTextColor(color)
        self.scene.addItem(text_item)
        return text_item

    def addScaleBar(self, x=5, y=5, length=100, height=5, color=Qt.GlobalColor.white, background_color=None, border_color=None, thickness=1):
        if not self.hasImage():
            return None

        rect_item = QGraphicsRectItem(x, y, length, height)
        rect_item.setBrush(QBrush(color))

        if background_color is not None or border_color is not None:
            if border_color is not None:
                pen = QPen(border_color)
                pen.setWidth(thickness)
                rect_item.setPen(pen)
            else:
                rect_item.setPen(QPen(Qt.PenStyle.NoPen))

            if background_color is not None:
                rect_item.setBrush(QBrush(background_color))
            else:
                rect_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        self.scene.addItem(rect_item)
        return rect_item

    def removeItems(self):
        for item in self.scene.items():
            if isinstance(item, (QGraphicsTextItem, QGraphicsRectItem)):
                self.scene.removeItem(item)

    def open(self, filepath=None):
        if filepath is None:
            filepath, dummy = QFileDialog.getOpenFileName(self, "Open image file.")
        if len(filepath) and os.path.isfile(filepath):
            image = QImage(filepath)
            self._imgShape = [image.width(), image.height()]
            self.setImage(image)

    # ------------------------------------------------------------------ zoom

    def updateViewer(self):
        if not self.hasImage():
            return
        if len(self.zoomStack):
            self.fitInView(self.zoomStack[-1], self.aspectRatioMode)
        else:
            self.fitInView(self.sceneRect(), self.aspectRatioMode)

    def clearZoom(self):
        if len(self.zoomStack) > 0:
            self.zoomStack = []
            self.updateViewer()
            self.viewChanged.emit()

    def resizeEvent(self, event):
        self.updateViewer()

    def wheelEvent(self, event):
        if self.wheelZoomFactor is not None:
            if self.wheelZoomFactor == 1:
                return
            if event.angleDelta().y() < 0:
                if len(self.zoomStack) == 0:
                    self.zoomStack.append(self.sceneRect())
                elif len(self.zoomStack) > 1:
                    del self.zoomStack[:-1]
                zoomRect = self.zoomStack[-1]
                center = zoomRect.center()
                zoomRect.setWidth(zoomRect.width() / self.wheelZoomFactor)
                zoomRect.setHeight(zoomRect.height() / self.wheelZoomFactor)
                zoomRect.moveCenter(center)
                self.zoomStack[-1] = zoomRect.intersected(self.sceneRect())
                self.updateViewer()
                self.viewChanged.emit()
            else:
                if len(self.zoomStack) == 0:
                    return
                if len(self.zoomStack) > 1:
                    del self.zoomStack[:-1]
                zoomRect = self.zoomStack[-1]
                center = zoomRect.center()
                zoomRect.setWidth(zoomRect.width() * self.wheelZoomFactor)
                zoomRect.setHeight(zoomRect.height() * self.wheelZoomFactor)
                zoomRect.moveCenter(center)
                self.zoomStack[-1] = zoomRect.intersected(self.sceneRect())
                if self.zoomStack[-1] == self.sceneRect():
                    self.zoomStack = []
                self.updateViewer()
                self.viewChanged.emit()
            event.accept()
            return

        QGraphicsView.wheelEvent(self, event)

    # ------------------------------------------------------------------ mode

    @property
    def drawROI(self):
        """ROI type drawn by the left mouse button, or None for selection / zoom mode."""
        return self._drawROI

    @drawROI.setter
    def drawROI(self, roiType):
        if roiType == self._drawROI:
            return
        self._cancelDrawing()
        self._endDrag()
        self._drawROI = roiType
        self._refreshInteractivity()

    def setROIsAreMovable(self):
        """Kept for compatibility: interactivity now follows the active mode automatically."""
        self._refreshInteractivity()

    def _mode(self):
        if self._drawROI is not None:
            return "draw"
        if self.regionZoomButton is not None:
            return "zoom"
        return "select"

    def _isInteractive(self, roi):
        if roi is self._drawing:
            return False
        mode = self._mode()
        if mode == "draw":
            if self._drawROI == ORIENT_MODE:
                return roi.roiType != "VectorROI"
            return roi.roiType == self._drawROI
        return mode == "select"

    def _refreshInteractivity(self):
        self._setHovered(None)
        keep = [r for r in self._selected.values() if self._isInteractive(r)]
        if len(keep) != len(self._selected):
            self._setSelection(keep)
        self._overlay.update()

    # ------------------------------------------------------------------ labels / style

    @property
    def _labelList(self):
        return self._labels

    @_labelList.setter
    def _labelList(self, labels):
        self.setLabelList(labels)

    def setLabelList(self, labels):
        """Set the label list that determines ROI colours (recolours ROIs if it changed)."""
        labels = list(labels or [])
        if labels == self._labels:
            return
        self._labels = labels
        for roi in self._rois.values():
            self._styleROI(roi)
        self._overlay.update()

    def colorFor(self, roi):
        """The label colour of an ROI."""
        label = roi.label
        if label == SYMMETRY_LABEL:
            return QColor(SYMMETRY_COLOR)
        if roi.roiType == "VectorROI" and label in (None, EMPTY_LABEL):
            return QColor(VECTOR_COLOR)
        return QColor(label_color(label, self._labels))

    def strokeColor(self, roi):
        """Colour the ROI is drawn with: its label colour, or a shade of it (see setDistinguishROIs)."""
        base = self.colorFor(roi)
        if self.distinguishROIs and self._shadeable(roi):
            return shade_color(base, roi._shade)
        return base

    def setDistinguishROIs(self, enabled):
        """
        Draw neighbouring ROIs of the same label in different shades of the label colour
        (same hue, different lightness).  Shades are assigned so that ROIs close to each other
        differ; an isolated ROI keeps the exact label colour.
        """
        enabled = bool(enabled)
        if enabled == self.distinguishROIs:
            return
        self.distinguishROIs = enabled
        for roi in self._rois.values():
            self._styleROI(roi)
        self._overlay.update()

    @staticmethod
    def _shadeable(roi):
        return roi.roiType != "VectorROI" and roi.label != SYMMETRY_LABEL

    def _assignShade(self, roi):
        """Greedy colouring: the first shade not used by a nearby ROI of the same label."""
        roi._shade = 0
        if not self._shadeable(roi):
            return
        r = roi.sceneBoundingRect()
        m = 0.35 * max(r.width(), r.height()) + 2.0
        used = {}
        for item in self.scene.items(r.adjusted(-m, -m, m, m), Qt.ItemSelectionMode.IntersectsItemBoundingRect):
            if item is roi or not isinstance(item, BaseROI) or item.label != roi.label or not self.hasROI(item):
                continue
            used[item._shade] = used.get(item._shade, 0) + 1
        free = [k for k in range(len(SHADE_STEPS)) if k not in used]
        roi._shade = free[0] if free else min(range(len(SHADE_STEPS)), key=lambda k: used.get(k, 0))

    def _reshade(self, rois):
        for roi in rois:
            if self.hasROI(roi):
                self._assignShade(roi)
                self._styleROI(roi)

    def _pen(self, color):
        key = color.name()
        pen = self._pens.get(key)
        if pen is None:
            pen = QPen(color, ROI_PEN_WIDTH)
            pen.setCosmetic(True)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            self._pens[key] = pen
        return pen

    def _styleROI(self, roi):
        color = self.strokeColor(roi)
        roi.setPen(self._pen(color))
        if roi.roiType == "PolygonROI":
            fill = QColor(color)
            fill.setAlpha(40)
            roi.setBrush(QBrush(fill))
        elif roi.roiType == "VectorROI":
            roi.setBrush(QBrush(color))
        elif self.distinguishROIs and roi.roiType in ("RectROI", "EllipseROI"):
            fill = QColor(color)       # faint fill in the ROI's own shade: separates touching boxes
            fill.setAlpha(30)
            roi.setBrush(QBrush(fill))
        else:
            roi.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._updateArrow(roi)

    # ------------------------------------------------------------------ orientation

    def _updateArrow(self, roi):
        """Create / update / hide the orientation arrow drawn as a child of the ROI."""
        arrow = getattr(roi, "_arrow", None)
        if roi.angle is None or not self.showOrientations:
            if arrow is not None:
                arrow.setVisible(False)
            return
        if arrow is None:
            arrow = QGraphicsPathItem(roi)
            arrow.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            roi._arrow = arrow
        arrow.setVisible(True)
        color = self.strokeColor(roi)
        arrow.setPen(self._pen(color))
        arrow.setBrush(QBrush(color))
        arrow.setPath(roi.orientationPath())

    def setShowOrientations(self, show):
        self.showOrientations = bool(show)
        for roi in self._rois.values():
            self._updateArrow(roi)
        self._overlay.update()

    def setROIAngle(self, rois, angle, emit=True):
        """Set (degrees, CCW from the image x axis) or clear (None) the orientation of ROIs."""
        if isinstance(rois, BaseROI):
            rois = [rois]
        changed = [r for r in rois if self.hasROI(r) and r.roiType != "VectorROI"]
        for roi in changed:
            roi.angle = None if angle is None else float(angle) % 360.0
            self._updateArrow(roi)
        if changed:
            self._revision += 1
            self._updateOverlay(changed)
            if emit:
                self.roisChanged.emit("modified", changed)
        return changed

    def rotateROIs(self, rois, delta, emit=True):
        """Rotate the orientation of ROIs by `delta` degrees (unset orientations start at 0)."""
        changed = [r for r in rois if self.hasROI(r) and r.roiType != "VectorROI"]
        for roi in changed:
            roi.angle = ((roi.angle or 0.0) + delta) % 360.0
            self._updateArrow(roi)
        if changed:
            self._revision += 1
            self._updateOverlay(changed)
            if emit:
                self.roisChanged.emit("modified", changed)
        return changed

    def _orientROIs(self):
        """ROIs whose orientation tip can be dragged: selected (few) or hovered, with an angle."""
        out = []
        if len(self._selected) <= MAX_HANDLE_ROIS:
            out = [r for r in self._selected.values() if r.angle is not None and self._isInteractive(r)]
        hov = self._hovered
        if hov is not None and hov.angle is not None and self._isInteractive(hov) and hov.roiId not in self._selected:
            out.append(hov)
        return out if self.showOrientations else []

    def _orientHandleAt(self, viewPos):
        p = self.mapToScene(viewPos)
        r = self._tol(HANDLE_PX + 3)
        for roi in self._orientROIs():
            tx, ty = roi.orientationTip()
            if math.hypot(tx + roi.x() - p.x(), ty + roi.y() - p.y()) <= r:
                return roi
        return None

    @staticmethod
    def _angleBetween(p0, p1):
        """Screen angle (degrees CCW, y axis pointing up on screen) of the direction p0 -> p1."""
        return math.degrees(math.atan2(-(p1.y() - p0.y()), p1.x() - p0.x())) % 360.0

    def _labelForNew(self, roiType):
        if roiType == "VectorROI":
            return EMPTY_LABEL
        return self._currentLabel or EMPTY_LABEL

    # ------------------------------------------------------------------ ROI store

    @property
    def ROIs(self):
        """ROIs grouped by type: {roiType: [roi, ...]} (a fresh dict, for reading)."""
        out = {t: [] for t in ("EllipseROI", "RectROI", "VectorROI", "LineSegmentROI", "PolygonROI", "PointROI")}
        for roi in self._rois.values():
            out.setdefault(roi.roiType, []).append(roi)
        return out

    def rois(self, roiType=None):
        """List of ROIs (optionally of one type) in drawing order."""
        if roiType is None:
            return list(self._rois.values())
        return [r for r in self._rois.values() if r.roiType == roiType]

    def roiCount(self):
        return len(self._rois)

    def hasROI(self, roi):
        return roi is not None and self._rois.get(roi.roiId) is roi

    def createROI(self, roiType, coords, label=None, roiId=None):
        """
        Create (but do not add) an ROI.  ``coords`` depends on the type:
        RectROI / EllipseROI (x, y, w, h); PointROI (x, y); VectorROI (x1, y1, x2, y2);
        LineSegmentROI / PolygonROI sequence of (x, y) or an (N, 2) array.
        """
        cls = ROI_CLASSES[roiType]
        return cls.fromCoords(self, coords, label=label, roiId=roiId)

    def _register(self, roi, label=None):
        if label is not None:
            roi.label = label
        if roi.label is None:
            roi.label = self._labelForNew(roi.roiType)
        if not roi.roiId or roi.roiId.startswith("<") or roi.roiId in self._rois:
            roi.roiId = _new_id(roi.roiType)
        roi._viewer = self
        roi._drawing = False
        self._assignShade(roi)
        self._styleROI(roi)
        if roi.scene() is not self.scene:
            self.scene.addItem(roi)
        self._rois[roi.roiId] = roi

    def addROIs(self, rois, label=None, emit=True, labelOverride=None, setColor=None):
        """
        Add one ROI or an iterable of ROIs.  ROIs keep their own label unless ``label`` is
        given; ROIs without a label get the current label.  Emits one ``roisChanged("added")``.
        """
        if isinstance(rois, BaseROI):
            rois = [rois]
        label = label if label is not None else labelOverride
        added = []
        for roi in rois:
            if self.hasROI(roi):
                continue
            self._register(roi, label)
            added.append(roi)
        if added:
            self._revision += 1
            if emit:
                self.roisChanged.emit("added", added)
        return added

    def addROI(self, roi, label=None, emit=True):
        return self.addROIs([roi], label=label, emit=emit)

    def deleteROIs(self, rois, emit=True):
        if isinstance(rois, BaseROI):
            rois = [rois]
        removed = []
        selectionChanged = False
        for roi in list(rois):
            if not self.hasROI(roi):
                continue
            del self._rois[roi.roiId]
            if self._selected.pop(roi.roiId, None) is not None:
                selectionChanged = True
            if self._hovered is roi:
                self._hovered = None
            removed.append(roi)
        if not removed:
            return []
        self._updateOverlay(removed)
        self._removeItems(removed)
        self._revision += 1
        if emit:
            self.roisChanged.emit("removed", removed)
            if selectionChanged:
                self.roiSelected.emit(self.selectedROIs())
        return removed

    def _removeItems(self, items):
        for item in items:
            self.scene.removeItem(item)

    def deleteROI(self, roi, emit=True):
        return self.deleteROIs([roi], emit=emit)

    def clearROIs(self, emit=False):
        self._cancelDrawing()
        self._endDrag()
        rois = list(self._rois.values())
        if not rois:
            return []
        self._rois = {}
        self._selected = {}
        self._hovered = None
        self._removeItems(rois)
        self._overlay.update()
        self._revision += 1
        if emit:
            self.roisChanged.emit("removed", rois)
        return rois

    def setROILabel(self, rois, label, emit=True):
        if isinstance(rois, BaseROI):
            rois = [rois]
        changed = [r for r in rois if self.hasROI(r) and r.label != label]
        for roi in changed:
            roi.label = label
        self._reshade(changed)
        if changed:
            self._revision += 1
            self._updateOverlay(changed)
            if emit:
                self.roisChanged.emit("relabeled", changed)
        return changed

    def loadROIs(self, df, image=None, method="added"):
        """
        Create ROIs from annotation rows (columns objectName, type, x1, y1, dx, dy, file, label)
        and add them in one batch.  ``image`` filters rows on the 'file' column.  With
        method="loaded" no change signal is emitted.  Returns the added ROIs.
        """
        if df is None or len(df) == 0:
            return []
        if image is not None and "file" in df.columns:
            df = df[df["file"] == image]
        rois = self.roisFromFrame(df)
        return self.addROIs(rois, emit=(method != "loaded"))

    def roisFromFrame(self, df):
        """ROIs (not added) from annotation rows.  Vertex ROIs are grouped by objectName."""
        rois = []
        groups = {}
        has_name = "objectName" in df.columns
        has_label = "label" in df.columns
        has_angle = ANGLE_COLUMN in df.columns

        def with_angle(roi, angle):
            if angle is not None:
                roi.angle = angle % 360.0
            return roi

        for rec in df.itertuples(index=False):
            roiType = str(rec.type).strip()
            label = _clean_label(rec.label) if has_label else EMPTY_LABEL
            name = _clean_name(rec.objectName) if has_name else None
            x1, y1, dx, dy = _num(rec.x1), _num(rec.y1), _num(rec.dx), _num(rec.dy)
            angle = None
            if has_angle:
                a = getattr(rec, ANGLE_COLUMN)
                try:
                    angle = float(a) if a is not None and math.isfinite(float(a)) else None
                except (TypeError, ValueError):
                    angle = None
            if roiType in ("LineSegmentROI", "PolygonROI"):
                group = groups.setdefault((roiType, name), [label, name, [], angle])
                group[2].append((x1, y1))
            elif roiType in ("RectROI", "EllipseROI"):
                rois.append(with_angle(self.createROI(roiType, (x1, y1, dx, dy), label, name), angle))
            elif roiType == "PointROI":
                rois.append(with_angle(self.createROI(roiType, (x1, y1), label, name), angle))
            elif roiType in ("VectorROI", "LineROI"):   # LineROI: legacy name
                rois.append(self.createROI("VectorROI", (x1, y1, dx, dy), label, name))
        for (roiType, _), (label, name, pts, angle) in groups.items():
            pts = _dedupe(np.array(pts, dtype=float))
            if len(pts) >= ROI_CLASSES[roiType].minPoints:
                rois.append(with_angle(self.createROI(roiType, pts, label, name), angle))
        return rois

    def annotationsFrame(self, filename=None):
        """
        All ROIs as annotation rows (a cached DataFrame; treat it as read-only).
        RectROI/EllipseROI: x1, y1, dx, dy = top-left, width, height.  PointROI: centre.
        VectorROI: x1, y1, dx, dy = start x, start y, end x, end y.
        LineSegmentROI/PolygonROI: one row per vertex.
        """
        import pandas as pd
        if filename is None:
            filename = os.path.basename(self._imgPath) if self._imgPath else ""
        key = (self._revision, filename)
        if self._frameCache[0] == key:
            return self._frameCache[1]
        rows = []
        for roi in self._rois.values():
            angle = float("nan") if roi.angle is None else roi.angle
            rows.extend(r + (angle,) for r in roi.toRows(filename))
        frame = pd.DataFrame(rows, columns=FRAME_COLUMNS)
        self._frameCache = (key, frame)
        return frame

    # ------------------------------------------------------------------ selection

    def selectedROIs(self):
        return list(self._selected.values())

    def selectROIs(self, rois, emit=True):
        self._setSelection(rois, emit=emit)

    def clearSelection(self, emit=True):
        self._setSelection([], emit=emit)

    def _setSelection(self, rois, emit=True):
        new = {}
        for roi in rois:
            if self.hasROI(roi):
                new[roi.roiId] = roi
        old = self._selected
        if new.keys() == old.keys():
            return
        self._selected = new
        self._updateOverlay(list(old.values()) + list(new.values()))
        if emit:
            self.roiSelected.emit(list(new.values()))

    def _setHovered(self, roi):
        if roi is self._hovered:
            return
        old, self._hovered = self._hovered, roi
        self._updateOverlay([r for r in (old, roi) if r is not None])

    def _updateOverlay(self, rois=None):
        if rois is None or len(rois) > 300:
            self._overlay.update()
            return
        rect = QRectF()
        for roi in rois:
            rect = rect.united(roi.sceneBoundingRect())
            if roi._arrow is not None:
                rect = rect.united(roi.mapRectToScene(roi.childrenBoundingRect()))
        if rect.isNull():
            return
        m = self._tol(2 * HANDLE_PX + ROI_PEN_WIDTH + 8)
        self._overlay.update(rect.adjusted(-m, -m, m, m))

    def _handleROIs(self):
        out = []
        if self._drawROI == ORIENT_MODE:
            return out                  # orientation tool: only the arrow tips are editable
        if len(self._selected) <= MAX_HANDLE_ROIS:
            out = [r for r in self._selected.values() if r.editable and self._isInteractive(r)]
        hov = self._hovered
        if hov is not None and hov.editable and self._isInteractive(hov) and hov.roiId not in self._selected:
            out.append(hov)
        if self._drawing is not None and isinstance(self._drawing, _VertexROI):
            out.append(self._drawing)
        return out

    # ------------------------------------------------------------------ picking

    def _scale(self):
        return max(abs(self.transform().m11()), 1e-9)

    def _tol(self, px=PICK_PX):
        """Screen distance in pixels -> scene units at the current zoom."""
        return px / self._scale()

    def roiAt(self, viewPos):
        """Interactive ROI under a viewport position (outline hits beat interior hits)."""
        p = self.mapToScene(viewPos)
        t = self._tol()
        area = QRectF(p.x() - t, p.y() - t, 2 * t, 2 * t)
        best, bestKey = None, None
        for item in self.scene.items(area, Qt.ItemSelectionMode.IntersectsItemBoundingRect,
                                     Qt.SortOrder.DescendingOrder):
            if not isinstance(item, BaseROI) or not self._isInteractive(item) or not self.hasROI(item):
                continue
            x, y = p.x() - item.x(), p.y() - item.y()
            d = item.edgeDistance(x, y)
            if d <= t:
                key = (0, d)
            elif item.closed and item.path().contains(QPointF(x, y)):
                key = (1, item.area())
            else:
                continue
            if bestKey is None or key < bestKey:
                best, bestKey = item, key
        return best

    def _handleAt(self, viewPos):
        """(roi, handle index) of the vertex handle under a viewport position, or None."""
        p = self.mapToScene(viewPos)
        r = self._tol(HANDLE_PX + 2)
        best, bestD = None, math.inf
        for roi in self._handleROIs():
            if roi is self._drawing:
                continue
            hp = roi.handlePoints()
            if len(hp) == 0:
                continue
            d = np.hypot(hp[:, 0] + roi.x() - p.x(), hp[:, 1] + roi.y() - p.y())
            i = int(np.argmin(d))
            if d[i] <= r and d[i] < bestD:
                best, bestD = (roi, i), d[i]
        return best

    # ------------------------------------------------------------------ drawing

    def _isPolyDrawing(self):
        return isinstance(self._drawing, _VertexROI)

    def _startDrawing(self, scenePos, viewPos):
        cls = ROI_CLASSES.get(self._drawROI)
        if cls is None:
            return
        self._setSelection([], emit=True)
        p = (scenePos.x(), scenePos.y())
        if cls is PointROI:
            roi = PointROI(self, [p], label=self._labelForNew("PointROI"))
            self._commitDrawn(roi)
            return
        roi = cls(self, [p, p], label=self._labelForNew(cls.roiType))
        roi._drawing = True
        self._styleROI(roi)
        self.scene.addItem(roi)
        self._drawing = roi
        if isinstance(roi, _VertexROI):
            self._drag = {"kind": "polyclick", "startView": QPoint(viewPos)}
        else:
            self._drag = {"kind": "shape", "start": QPointF(scenePos), "startView": QPoint(viewPos)}

    def _commitDrawn(self, roi):
        roi.finishEdit()
        self._drawing = None
        self._register(roi)
        self._revision += 1
        self._setSelection([roi], emit=False)
        self.roisChanged.emit("drawn", [roi])

    def _cancelDrawing(self):
        roi = self._drawing
        if roi is None:
            return
        self._drawing = None
        if roi.scene() is self.scene:
            self.scene.removeItem(roi)
        self._overlay.update()

    def _finishPolyDrawing(self):
        roi = self._drawing
        if roi is None:
            return
        pts = _dedupe(roi._pts[:-1])        # drop the rubber-band preview point
        if len(pts) >= roi.minPoints:
            roi._drawing = False
            roi.setPoints(pts)
            self._commitDrawn(roi)
        else:
            self._cancelDrawing()

    def _endDrag(self):
        self._drag = None
        self._band.hide()

    # ------------------------------------------------------------------ mouse

    def _setCursor(self, shape):
        if shape != self._cursor:
            self._cursor = shape
            self.viewport().setCursor(shape)

    def mousePressEvent(self, event):
        dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                             | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        if event.modifiers() == dummyModifiers:
            QGraphicsView.mousePressEvent(self, event)
            event.accept()
            return

        self.setFocus()
        button = event.button()
        viewPos = event.pos()
        scenePos = self.mapToScene(viewPos)

        if (self.panButton is not None) and (button == self.panButton):
            self._pixelPosition = viewPos
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            if self.panButton == Qt.MouseButton.LeftButton:
                QGraphicsView.mousePressEvent(self, event)
            else:
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                dummyEvent = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(viewPos), Qt.MouseButton.LeftButton,
                                         event.buttons(), dummyModifiers)
                self.mousePressEvent(dummyEvent)
            sceneViewport = self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())
            self._scenePosition = sceneViewport.topLeft()
            event.accept()
            self._isPanning = True
            return

        self._emitClick(button, scenePos, pressed=True)
        mode = self._mode()

        if button == Qt.MouseButton.LeftButton:
            event.accept()
            if mode == "zoom":
                self._startBand(viewPos, "zoom")
                return
            if self._isPolyDrawing():
                roi = self._drawing
                roi.updateLastPoint(scenePos)
                roi.addPoint(scenePos)
                self._drag = {"kind": "polyclick", "startView": QPoint(viewPos)}
                self._updateOverlay([roi])
                return
            tip_roi = self._orientHandleAt(viewPos)
            if tip_roi is not None:
                self._drag = {"kind": "orient", "roi": tip_roi, "startView": QPoint(viewPos), "moved": False}
                return
            hit = self._handleAt(viewPos)
            if hit is not None:
                roi, index = hit
                self._drag = {"kind": "vertex", "roi": roi, "index": index,
                              "startView": QPoint(viewPos), "moved": False}
                return
            roi = self.roiAt(viewPos)
            if self._drawROI == ORIENT_MODE:
                # orientation tool: drag inside an ROI from its tail towards its head
                if roi is None:
                    self._setSelection([])
                    return
                if roi.roiId not in self._selected:
                    self._setSelection([roi])
                self._drag = {"kind": "orientdraw", "roi": roi, "start": QPointF(scenePos),
                              "startView": QPoint(viewPos), "moved": False}
                return
            if roi is not None:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    sel = [r for r in self._selected.values() if r is not roi]
                    if len(sel) == len(self._selected):
                        sel.append(roi)
                    self._setSelection(sel)
                    return
                if roi.roiId not in self._selected:
                    self._setSelection([roi])
                rois = [r for r in self._selected.values() if self._isInteractive(r)]
                self._drag = {"kind": "move", "roi": roi, "rois": rois, "start": QPointF(scenePos),
                              "startView": QPoint(viewPos), "moved": False}
                return
            if mode == "draw":
                self._startDrawing(scenePos, viewPos)
                return
            additive = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if not additive:
                self._setSelection([])
            if self.selectionButton is not None:
                self._startBand(viewPos, "band", additive=additive)
            return

        if button == Qt.MouseButton.RightButton:
            event.accept()
            if self._drawing is not None:
                self._cancelDrawing()
                self._endDrag()
                return
            hit = self._handleAt(viewPos)
            if hit is not None and hit[0].canRemoveVertices:
                self._removeVertex(*hit)
                return
            if mode == "draw":
                roi = self.roiAt(viewPos)
                if roi is not None:
                    if self._drawROI == ORIENT_MODE:
                        self.setROIAngle([roi], None)        # orientation tool: clear, keep the ROI
                    else:
                        self.deleteROIs([roi])
                return
            if (self.zoomOutButton is not None) and (button == self.zoomOutButton):
                if len(self.zoomStack):
                    self.zoomStack.pop()
                    self.updateViewer()
                    self.viewChanged.emit()
            return

        QGraphicsView.mousePressEvent(self, event)

    def _startBand(self, viewPos, kind, additive=False):
        self._bandOrigin = QPoint(viewPos)
        self._band.setGeometry(QRect(self._bandOrigin, QSize()))
        self._band.show()
        self._drag = {"kind": kind, "startView": QPoint(viewPos), "additive": additive}

    def _removeVertex(self, roi, index):
        self._updateOverlay([roi])
        if roi.removeVertex(index):
            self._revision += 1
            self._updateOverlay([roi])
            self.roisChanged.emit("modified", [roi])
        else:
            self.deleteROIs([roi])

    def mouseMoveEvent(self, event):
        viewPos = event.pos()
        scenePos = self.mapToScene(viewPos)
        drag = self._drag

        if self._isPanning:
            QGraphicsView.mouseMoveEvent(self, event)
            if len(self.zoomStack) > 0:
                sceneViewport = self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())
                delta = sceneViewport.topLeft() - self._scenePosition
                self._scenePosition = sceneViewport.topLeft()
                self.zoomStack[-1].translate(delta)
                self.zoomStack[-1] = self.zoomStack[-1].intersected(self.sceneRect())
                self.updateViewer()
                self.viewChanged.emit()
        elif drag is not None:
            kind = drag["kind"]
            if kind in ("move", "vertex", "orient", "orientdraw") and not drag["moved"]:
                if (viewPos - drag["startView"]).manhattanLength() >= MIN_DRAG_PX:
                    drag["moved"] = True
            if kind in ("orient", "orientdraw") and drag["moved"]:
                roi = drag["roi"]
                if kind == "orient":         # drag the arrow tip around the ROI centre
                    cx, cy = roi.centre()
                    angle = self._angleBetween(QPointF(cx + roi.x(), cy + roi.y()), scenePos)
                else:                        # direction of the drag
                    angle = self._angleBetween(drag["start"], scenePos)
                self._updateOverlay([roi])
                roi.angle = angle
                self._updateArrow(roi)
                self._updateOverlay([roi])
            elif kind == "move" and drag["moved"]:
                rois = drag["rois"]
                self._updateOverlay(rois)
                delta = scenePos - drag["start"]
                for roi in rois:
                    roi.setPos(delta)
                self._updateOverlay(rois)
            elif kind == "vertex" and drag["moved"]:
                roi = drag["roi"]
                self._updateOverlay([roi])
                roi.moveHandle(drag["index"], scenePos.x() - roi.x(), scenePos.y() - roi.y())
                self._updateOverlay([roi])
            elif kind == "shape" and self._drawing is not None:
                start = drag["start"]
                self._drawing.setPoints([(start.x(), start.y()), (scenePos.x(), scenePos.y())])
            elif kind in ("band", "zoom"):
                self._band.setGeometry(QRect(self._bandOrigin, viewPos).normalized())
        if self._isPolyDrawing():
            self._updateOverlay([self._drawing])
            self._drawing.updateLastPoint(scenePos)
            self._updateOverlay([self._drawing])

        if drag is None and not self._isPanning and not event.buttons():
            hit = self._handleAt(viewPos)
            roi = hit[0] if hit is not None else self.roiAt(viewPos)
            self._setHovered(roi)
            if self._drawing is not None:
                self._setCursor(Qt.CursorShape.CrossCursor)
            elif hit is not None:
                self._setCursor(Qt.CursorShape.SizeAllCursor)
            elif roi is not None:
                self._setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self._setCursor(Qt.CursorShape.CrossCursor)

        if self.sceneRect().contains(scenePos):
            x = int(round(scenePos.x() - 0.5))
            y = int(round(scenePos.y() - 0.5))
            imagePos = QPoint(x, y)
        else:
            imagePos = QPoint(-1, -1)
        self.mousePositionOnImageChanged.emit(imagePos)

        if self._isPanning:
            return
        QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                             | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        if event.modifiers() == dummyModifiers:
            QGraphicsView.mouseReleaseEvent(self, event)
            event.accept()
            return

        button = event.button()
        viewPos = event.pos()
        scenePos = self.mapToScene(viewPos)

        if (self.panButton is not None) and (button == self.panButton):
            if self.panButton == Qt.MouseButton.LeftButton:
                QGraphicsView.mouseReleaseEvent(self, event)
            else:
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                self._cursor = None
                dummyEvent = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(viewPos),
                                         Qt.MouseButton.LeftButton, event.buttons(), dummyModifiers)
                self.mouseReleaseEvent(dummyEvent)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            if len(self.zoomStack) > 0:
                sceneViewport = self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())
                delta = sceneViewport.topLeft() - self._scenePosition
                self.zoomStack[-1].translate(delta)
                self.zoomStack[-1] = self.zoomStack[-1].intersected(self.sceneRect())
                self.viewChanged.emit()
            event.accept()
            self._isPanning = False
            return

        drag = self._drag
        if button == Qt.MouseButton.LeftButton and drag is not None:
            self._drag = None
            self._finishDrag(drag, viewPos, scenePos, event.modifiers())
            event.accept()

        self._emitClick(button, scenePos, pressed=False)
        QGraphicsView.mouseReleaseEvent(self, event)

    def _finishDrag(self, drag, viewPos, scenePos, modifiers):
        kind = drag["kind"]
        travel = (viewPos - drag["startView"]).manhattanLength()

        if kind == "move":
            rois = drag["rois"]
            if drag["moved"]:
                for roi in rois:
                    roi.bakePosition()
                self._reshade(rois)
                self._revision += 1
                self._updateOverlay(rois)
                self.roisChanged.emit("modified", rois)
            elif len(self._selected) > 1:
                self._setSelection([drag["roi"]])

        elif kind == "vertex":
            roi = drag["roi"]
            if drag["moved"]:
                roi.finishEdit()
                self._reshade([roi])
                self._revision += 1
                self._updateOverlay([roi])
                self.roisChanged.emit("modified", [roi])

        elif kind in ("orient", "orientdraw"):
            roi = drag["roi"]
            if drag["moved"] and self.hasROI(roi):
                self._revision += 1
                self._updateOverlay([roi])
                self.roisChanged.emit("modified", [roi])

        elif kind == "shape":
            roi = self._drawing
            if roi is None:
                return
            delta = viewPos - drag["startView"]
            if isinstance(roi, VectorROI):
                big = math.hypot(delta.x(), delta.y()) >= 5
            else:
                big = abs(delta.x()) >= 2 and abs(delta.y()) >= 2
            if big:
                start = drag["start"]
                roi.setPoints([(start.x(), start.y()), (scenePos.x(), scenePos.y())])
                self._commitDrawn(roi)
            else:
                self._cancelDrawing()

        elif kind == "polyclick":
            if self._isPolyDrawing() and travel >= MIN_DRAG_PX:   # press-drag-release adds a segment
                self._drawing.updateLastPoint(scenePos)
                self._drawing.addPoint(scenePos)

        elif kind == "band":
            self._band.hide()
            rect = self._band.geometry()
            if rect.width() > 3 and rect.height() > 3:
                sceneRect = self.mapToScene(rect).boundingRect()
                hits = [it for it in self.scene.items(sceneRect, Qt.ItemSelectionMode.ContainsItemBoundingRect,
                                                      Qt.SortOrder.AscendingOrder)
                        if isinstance(it, BaseROI) and self.hasROI(it) and self._isInteractive(it)]
                if drag.get("additive"):
                    hits = list(self._selected.values()) + hits
                self._setSelection(hits)

        elif kind == "zoom":
            self._band.hide()
            rect = self._band.geometry()
            if rect.width() > 3 and rect.height() > 3:
                zoomRect = self.mapToScene(rect).boundingRect().intersected(self.sceneRect())
                if zoomRect.isValid() and (zoomRect != self.sceneRect()):
                    self.zoomStack.append(zoomRect)
                    self.updateViewer()
                    self.viewChanged.emit()

    def mouseDoubleClickEvent(self, event):
        button = event.button()
        viewPos = event.pos()
        scenePos = self.mapToScene(viewPos)

        if button == Qt.MouseButton.LeftButton:
            if self._isPolyDrawing():
                self._drag = None
                self._finishPolyDrawing()
                event.accept()
                return
            if self._mode() != "zoom":
                roi = self.roiAt(viewPos)
                if roi is not None and roi.canRemoveVertices:
                    x, y = scenePos.x() - roi.x(), scenePos.y() - roi.y()
                    if roi.edgeDistance(x, y) <= self._tol():
                        self._drag = None
                        roi.insertVertexNear(x, y)
                        self._revision += 1
                        self._setSelection([roi])
                        self._updateOverlay([roi])
                        self.roisChanged.emit("modified", [roi])
                        event.accept()
                        return

        if (self.zoomOutButton is not None) and (button == self.zoomOutButton):
            self.clearZoom()
            event.accept()
            return

        if button == Qt.MouseButton.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif button == Qt.MouseButton.RightButton:
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mouseDoubleClickEvent(self, event)

    def _emitClick(self, button, scenePos, pressed):
        x, y = scenePos.x(), scenePos.y()
        if button == Qt.MouseButton.LeftButton:
            (self.leftMouseButtonPressed if pressed else self.leftMouseButtonReleased).emit(x, y)
        elif button == Qt.MouseButton.MiddleButton:
            (self.middleMouseButtonPressed if pressed else self.middleMouseButtonReleased).emit(x, y)
        elif button == Qt.MouseButton.RightButton:
            (self.rightMouseButtonPressed if pressed else self.rightMouseButtonReleased).emit(x, y)

    def keyPressEvent(self, event):
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if key == Qt.Key.Key_Escape and (self._drawing is not None or self._selected):
            if self._drawing is not None:
                self._cancelDrawing()
                self._endDrag()
            else:
                self.clearSelection()
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._isPolyDrawing():
            self._finishPolyDrawing()
            event.accept()
            return
        if key == Qt.Key.Key_Backspace and self._isPolyDrawing():
            roi = self._drawing
            if len(roi._pts) > 2:
                roi.setPoints(np.delete(roi._pts, -2, axis=0))
                self._updateOverlay([roi])
            else:
                self._cancelDrawing()
            event.accept()
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._selected and self._drawing is None:
            self.deleteROIs(list(self._selected.values()))
            event.accept()
            return
        if key == Qt.Key.Key_A and ctrl and self._mode() != "zoom":
            self._setSelection([r for r in self._rois.values() if self._isInteractive(r)])
            event.accept()
            return
        # orientation of the selected ROIs: ] / [ rotate +-5 deg (with Shift +-1 deg), F flips 180 deg
        rotate = {Qt.Key.Key_BracketRight: 5.0, Qt.Key.Key_BracketLeft: -5.0,
                  Qt.Key.Key_BraceRight: 1.0, Qt.Key.Key_BraceLeft: -1.0}
        if self._selected and self._drawing is None and (key in rotate or key == Qt.Key.Key_F) and not ctrl:
            rois = list(self._selected.values())
            if key == Qt.Key.Key_F:
                self.rotateROIs([r for r in rois if r.angle is not None], 180.0)
            else:
                step = rotate[key]
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and abs(step) == 5.0:
                    step = step / 5.0
                self.rotateROIs(rois, step)
            event.accept()
            return
        QGraphicsView.keyPressEvent(self, event)

    def enterEvent(self, event):
        self._setCursor(Qt.CursorShape.CrossCursor)

    def leaveEvent(self, event):
        self._setHovered(None)
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self._cursor = None

    # ------------------------------------------------------------------ overlay painting

    def _paintOverlay(self, painter):
        handleRois = self._handleROIs()
        orientRois = self._orientROIs()
        hovered = self._hovered if self._hovered is not None and self.hasROI(self._hovered) else None
        if not self._selected and hovered is None and not handleRois:
            return
        if orientRois:
            # round handle at each arrow tip: drag it to turn the orientation
            h = (HANDLE_PX + 1) / (abs(painter.worldTransform().m11()) or 1.0)
            pen = QPen(QColor(0, 0, 0), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 235, 59)))
            for roi in orientRois:
                tx, ty = roi.orientationTip()
                painter.drawEllipse(QPointF(tx + roi.x(), ty + roi.y()), h, h)
        scale = abs(painter.worldTransform().m11()) or 1.0
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if hovered is not None and hovered.roiId not in self._selected:
            path = hovered.path().translated(hovered.pos())
            if hovered.closed:
                painter.fillPath(path, QBrush(QColor(128, 128, 128, 80)))
            else:
                pen = QPen(QColor(255, 255, 255, 110), ROI_PEN_WIDTH + 5)
                pen.setCosmetic(True)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPath(path)

        if self._selected:
            dash = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)
            dash.setCosmetic(True)
            for roi in self._selected.values():
                path = roi.path().translated(roi.pos())
                glow = self.colorFor(roi)
                glow.setAlpha(120)
                pen = QPen(glow, ROI_PEN_WIDTH + 5)
                pen.setCosmetic(True)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPath(path)
                painter.setPen(dash)
                painter.drawPath(path)

        if handleRois:
            h = HANDLE_PX / scale
            pen = QPen(QColor(0, 0, 0), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 255, 255)))
            for roi in handleRois:
                ox, oy = roi.x(), roi.y()
                pts = roi.handlePoints()
                if roi is self._drawing:
                    pts = pts[:-1]
                for x, y in pts:
                    painter.drawRect(QRectF(x + ox - h, y + oy - h, 2 * h, 2 * h))


class _ROIOverlay(QGraphicsItem):
    """Single item that paints hover/selection highlights and vertex handles for the viewer."""

    def __init__(self, viewer):
        QGraphicsItem.__init__(self)
        self._viewer = viewer
        self._bounds = QRectF()
        self.setZValue(1e9)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def setBounds(self, rect):
        self.prepareGeometryChange()
        self._bounds = QRectF(rect)

    def boundingRect(self):
        return self._bounds

    def paint(self, painter, option, widget=None):
        self._viewer._paintOverlay(painter)


# =============================================================================
# ROI items
# =============================================================================

class BaseROI(QGraphicsPathItem):
    """
    Base ROI: geometry is an (N, 2) array of points in scene coordinates (``_pts``), the
    drawn path is rebuilt from it.  While being dragged as a whole the item is offset
    with setPos(); the offset is baked into the points when the drag ends.
    """
    roiType = "BaseROI"
    closed = True               # interior is part of the ROI (picking, hover fill)
    editable = True             # shows vertex/corner handles
    canRemoveVertices = False
    minPoints = 1

    def __init__(self, viewer, points=None, label=None, roiId=None):
        QGraphicsPathItem.__init__(self)
        self._viewer = viewer
        self.label = label
        self.roiId = roiId
        self._drawing = False
        self._shade = 0
        self.angle = None               # orientation, degrees CCW from the image x axis (None = unset)
        self._arrow = None
        self._pts = _as_points(points)
        self.setZValue(1)
        self._rebuild()

    @classmethod
    def fromCoords(cls, viewer, coords, label=None, roiId=None):
        return cls(viewer, coords, label=label, roiId=roiId)

    # geometry -----------------------------------------------------------------

    def _buildPath(self):
        raise NotImplementedError

    def _rebuild(self):
        self.setPath(self._buildPath())
        if self.angle is not None and self._arrow is not None and self._viewer is not None:
            self._viewer._updateArrow(self)

    # orientation -----------------------------------------------------------------

    def centre(self):
        """Centre of the ROI in local coordinates (bounding box centre of its points)."""
        if not len(self._pts):
            return 0.0, 0.0
        lo, hi = self._pts.min(axis=0), self._pts.max(axis=0)
        return float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2

    def _arrowLength(self):
        r = self.path().boundingRect()
        size = max(r.width(), r.height())
        minimum = 4 * (self._viewer._pointRadius if self._viewer is not None else 4.0)
        return max(0.8 * size, minimum)

    def orientationTip(self):
        """Arrow tip (local coordinates) of the orientation."""
        cx, cy = self.centre()
        half = self._arrowLength() / 2
        a = math.radians(self.angle or 0.0)
        return cx + half * math.cos(a), cy - half * math.sin(a)

    def orientationPath(self):
        """Arrow through the ROI centre pointing along `angle` (screen y axis points down)."""
        cx, cy = self.centre()
        length = self._arrowLength()
        a = math.radians(self.angle or 0.0)
        dx, dy = math.cos(a), -math.sin(a)
        tail = (cx - dx * length / 2, cy - dy * length / 2)
        tip = (cx + dx * length / 2, cy + dy * length / 2)
        head = min(0.32 * length, max(3.0, 0.25 * length))
        path = QPainterPath()
        path.moveTo(*tail)
        path.lineTo(*tip)
        for side in (math.pi / 7, -math.pi / 7):
            ang = math.atan2(dy, dx) + math.pi + side
            if side > 0:
                path.moveTo(*tip)
                path.lineTo(tip[0] + head * math.cos(ang), tip[1] + head * math.sin(ang))
            else:
                path.lineTo(tip[0] + head * math.cos(ang), tip[1] + head * math.sin(ang))
        path.closeSubpath()
        return path

    def _changed(self):
        viewer = self._viewer
        if viewer is not None and viewer.hasROI(self):
            viewer._revision += 1

    def setPoints(self, points):
        self._pts = _as_points(points)
        self._rebuild()
        self._changed()

    def pointsArray(self):
        """Points in scene coordinates as an (N, 2) array (copy)."""
        return self._pts + np.array([self.x(), self.y()])

    def points(self):
        return [QPointF(x, y) for x, y in self.pointsArray()]

    def outlinePoints(self):
        """Outline in scene coordinates: 4 corners of a box, sampled ellipse, vertices otherwise."""
        pts, _ = self._outline()
        return np.asarray(pts, dtype=float) + np.array([self.x(), self.y()])

    def bakePosition(self):
        if self.x() or self.y():
            self._pts = self._pts + np.array([self.x(), self.y()])
            self.setPos(0, 0)
            self._rebuild()
            self._changed()

    def area(self):
        r = self.path().boundingRect()
        return r.width() * r.height()

    def _outline(self):
        return self._pts, self.closed

    def edgeDistance(self, x, y):
        pts, closed = self._outline()
        return _distance_to_outline(pts, x, y, closed and len(pts) > 2)

    # editing ------------------------------------------------------------------

    def handlePoints(self):
        return self._pts

    def moveHandle(self, index, x, y):
        self._pts[index] = (x, y)
        self._rebuild()

    def finishEdit(self):
        """Normalise geometry after drawing / handle dragging."""
        self._changed()

    def removeVertex(self, index):
        return False

    def insertVertexNear(self, x, y):
        return None

    def toRows(self, filename):
        raise NotImplementedError


class RectROI(BaseROI):
    """Axis aligned rectangle, stored as two opposite corners."""
    roiType = "RectROI"

    @classmethod
    def fromCoords(cls, viewer, coords, label=None, roiId=None):
        x, y, w, h = (float(v) for v in coords[:4])
        return cls(viewer, [(x, y), (x + w, y + h)], label=label, roiId=roiId)

    def _localRect(self):
        if len(self._pts) < 2:
            return QRectF()
        (x0, y0), (x1, y1) = self._pts[0], self._pts[1]
        return QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized()

    def _buildPath(self):
        path = QPainterPath()
        if len(self._pts) >= 2:
            path.addRect(self._localRect())
        return path

    def rect(self):
        """Normalised rectangle in scene coordinates."""
        return self._localRect().translated(self.pos())

    def setRect(self, x, y, w, h):
        self.setPoints([(x, y), (x + w, y + h)])

    def handlePoints(self):
        if len(self._pts) < 2:
            return self._pts
        (x0, y0), (x1, y1) = self._pts[0], self._pts[1]
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def moveHandle(self, index, x, y):
        self._pts[0 if index in (0, 3) else 1, 0] = x
        self._pts[0 if index in (0, 1) else 1, 1] = y
        self._rebuild()

    def finishEdit(self):
        r = self._localRect()
        self._pts = np.array([[r.left(), r.top()], [r.right(), r.bottom()]])
        self._rebuild()
        self._changed()

    def _outline(self):
        return self.handlePoints(), True

    def toRows(self, filename):
        r = self.rect()
        return [(self.roiId, self.roiType, r.x(), r.y(), r.width(), r.height(), filename, self.label)]


class EllipseROI(RectROI):
    """Ellipse inscribed in a rectangle (same storage and handles as RectROI)."""
    roiType = "EllipseROI"

    def _buildPath(self):
        path = QPainterPath()
        if len(self._pts) >= 2:
            path.addEllipse(self._localRect())
        return path

    def _outline(self):
        r = self._localRect()
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        pts = np.stack([r.center().x() + r.width() / 2 * np.cos(t),
                        r.center().y() + r.height() / 2 * np.sin(t)], axis=1)
        return pts, True


class PointROI(BaseROI):
    """Single point, drawn as a small circle (radius scales with the image size)."""
    roiType = "PointROI"
    editable = False

    @classmethod
    def fromCoords(cls, viewer, coords, label=None, roiId=None):
        return cls(viewer, [(float(coords[0]), float(coords[1]))], label=label, roiId=roiId)

    def _radius(self):
        return self._viewer._pointRadius if self._viewer is not None else 4.0

    def _buildPath(self):
        path = QPainterPath()
        if len(self._pts):
            r = self._radius()
            path.addEllipse(QPointF(*self._pts[0]), r, r)
        return path

    def center(self):
        x, y = self.pointsArray()[0]
        return QPointF(x, y)

    def rect(self):
        r = self._radius()
        c = self.center()
        return QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)

    def edgeDistance(self, x, y):
        if not len(self._pts):
            return math.inf
        return max(0.0, math.hypot(self._pts[0, 0] - x, self._pts[0, 1] - y) - self._radius())

    def toRows(self, filename):
        x, y = self.pointsArray()[0]
        return [(self.roiId, self.roiType, x, y, 0, 0, filename, self.label)]


class VectorROI(BaseROI):
    """Arrow from the first to the second point."""
    roiType = "VectorROI"
    closed = False

    @classmethod
    def fromCoords(cls, viewer, coords, label=None, roiId=None):
        x1, y1, x2, y2 = (float(v) for v in coords[:4])
        return cls(viewer, [(x1, y1), (x2, y2)], label=label, roiId=roiId)

    def _buildPath(self):
        path = QPainterPath()
        if len(self._pts) < 2:
            return path
        (x1, y1), (x2, y2) = self._pts[0], self._pts[1]
        path.moveTo(x1, y1)
        path.lineTo(x2, y2)
        length = math.hypot(x2 - x1, y2 - y1)
        if length > 0:
            size = min(length * 0.35, self._viewer._arrowSize if self._viewer is not None else 6.0)
            angle = math.atan2(y2 - y1, x2 - x1)
            path.moveTo(x2, y2)
            path.lineTo(x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
            path.lineTo(x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
            path.closeSubpath()
        return path

    def line(self):
        (x1, y1), (x2, y2) = self.pointsArray()[:2]
        return QLineF(x1, y1, x2, y2)

    def setLine(self, x1, y1, x2, y2):
        self.setPoints([(x1, y1), (x2, y2)])

    def _outline(self):
        return self._pts[:2], False

    def toRows(self, filename):
        (x1, y1), (x2, y2) = self.pointsArray()[:2]
        return [(self.roiId, self.roiType, x1, y1, x2, y2, filename, self.label)]


class _VertexROI(BaseROI):
    """Shared implementation of polyline / polygon ROIs (one row per vertex when saved)."""
    canRemoveVertices = True

    def _buildPath(self):
        path = QPainterPath()
        if len(self._pts):
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in self._pts]))
            if self.closed and not self._drawing and len(self._pts) >= 3:
                path.closeSubpath()
        return path

    def addPoint(self, pt):
        self._pts = np.vstack([self._pts, [[pt.x(), pt.y()]]])
        self._rebuild()

    def updateLastPoint(self, pt):
        if len(self._pts):
            self._pts[-1] = (pt.x(), pt.y())
            self._rebuild()

    def removeVertex(self, index):
        if len(self._pts) - 1 < self.minPoints:
            return False
        self._pts = np.delete(self._pts, index, axis=0)
        self._rebuild()
        self._changed()
        return True

    def insertVertexNear(self, x, y):
        if len(self._pts) < 2:
            return None
        closed = self.closed and len(self._pts) > 2
        dist, t, a, b = _segment_projection(self._pts, x, y, closed)
        k = int(np.argmin(dist))
        p = a[k] + t[k] * (b[k] - a[k])
        self._pts = np.insert(self._pts, k + 1, p, axis=0)
        self._rebuild()
        self._changed()
        return k + 1

    def toRows(self, filename):
        return [(self.roiId, self.roiType, x, y, 0, 0, filename, self.label) for x, y in self.pointsArray()]


class LineSegmentROI(_VertexROI):
    """Open polyline."""
    roiType = "LineSegmentROI"
    closed = False
    minPoints = 2

    def setLine(self, x1, y1, x2, y2):
        self.setPoints([(x1, y1), (x2, y2)])


class PolygonROI(_VertexROI):
    """Closed polygon with a translucent fill."""
    roiType = "PolygonROI"
    closed = True
    minPoints = 3


ROI_CLASSES = {
    "RectROI": RectROI,
    "EllipseROI": EllipseROI,
    "PointROI": PointROI,
    "VectorROI": VectorROI,
    "LineSegmentROI": LineSegmentROI,
    "PolygonROI": PolygonROI,
}


if __name__ == '__main__':
    import sys
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    viewer = QtImageViewer()
    viewer.open()
    viewer.show()
    sys.exit(app.exec())
