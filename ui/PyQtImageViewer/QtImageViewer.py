""" QtImageViewer.py: PyQt image viewer widget based on QGraphicsView with mouse zooming/panning and ROIs.

"""

import os.path

try:
    from PyQt6.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal, QEvent, QSize
    from PyQt6.QtGui import QImage, QPixmap, QPainterPath, QMouseEvent, QPainter, QPen, QBrush, QColor, QFont
    from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QFileDialog, QSizePolicy, \
        QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPathItem, \
        QGraphicsPolygonItem, QGraphicsTextItem
except ImportError:
    try:
        from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal, QEvent, QSize
        from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QMouseEvent, QPainter, QPen, QBrush, QColor, QFont
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QFileDialog, QSizePolicy, \
            QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPathItem, \
            QGraphicsPolygonItem, QGraphicsTextItem
    except ImportError:
        raise ImportError("Requires PyQt (version 5 or 6)")

# numpy is optional: only needed if you want to display numpy 2d arrays as images.
try:
    import numpy as np
except ImportError:
    np = None
import math

# qimage2ndarray is optional: useful for displaying numpy 2d arrays as images.
try:
    import qimage2ndarray
except ImportError:
    qimage2ndarray = None

__author__ = "Marcel Goldschen-Ohm <marcel.goldschen@gmail.com>"
__version__ = '2.0.0'


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
    roiSelected = pyqtSignal(list)
    roiDict = pyqtSignal(dict)
    roiList = pyqtSignal(list)

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self._image = None
        self.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.regionZoomButton = Qt.MouseButton.LeftButton
        self.zoomOutButton = Qt.MouseButton.RightButton
        self.panButton = Qt.MouseButton.MiddleButton
        self.wheelZoomFactor = 1.25

        self.selectionButton = Qt.MouseButton.LeftButton
        self.selectedROI = None
        self.zoomStack = []

        self._isZooming = False
        self._isPanning = False
        self._isSelecting = False
        self._pixelPosition = QPoint()
        self._scenePosition = QPointF()

        self.ROIs = {
            "EllipseROI": [],
            "RectROI": [],
            "VectorROI": [],
            "LineSegmentROI": [],
            "PolygonROI": [],
            "PointROI": []
        }
        self.ROIs_selected = {
            "EllipseROI": [],
            "RectROI": [],
            "VectorROI": [],
            "LineSegmentROI": [],
            "PolygonROI": [],
            "PointROI": []
        }

        self._roiStartPos = None
        self._roiEndPos = None
        self._currentRectROI = None
        self._currentVectorROI = None
        self._currentLineSegmentROI = None
        self.drawROI = None

        self.roiChanged = ""

        self.colorList = [
            QColor("#B9B1B4"),
            QColor("#D81B60"),
            QColor("#1E88E5"),
            QColor("#D49713"),
            QColor("#4CDBB0"), 
            QColor("#D55E00"),
            QColor("#CC79A7"),
            QColor("#F0E442"),
            QColor("#56B4E9"),
            QColor("#009E73"),
            QColor("#E69F00"),
            QColor("#0072B2"), 
            QColor("#F5E400"),
            QColor("#FFFFFF"),
            QColor("#000000")
            ]
        self._labelList = []
        self._imgShape = []
        self._pxScale = None
        self._imgPath = None
        self._currentLabel = "Empty Label"
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        return QSize(900, 600)

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
        elif (np is not None) and (type(image) is np.ndarray):
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

        self.setSceneRect(QRectF(pixmap.rect()))
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

    def mousePressEvent(self, event):
        dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                             | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        if event.modifiers() == dummyModifiers:
            QGraphicsView.mousePressEvent(self, event)
            event.accept()
            return

        if self.drawROI is not None:
            self.selectedROI = None
            item = self.itemAt(event.pos())

            if item and (item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable) and item.__class__.__name__ == self.drawROI:
                # While drawing a polyline, ignore the in-progress item so clicks keep adding vertices.
                if self.drawROI == "LineSegmentROI" and item is self._currentLineSegmentROI:
                    pass
                else:
                    QGraphicsView.mousePressEvent(self, event)
                    self.selectedROI = item
                    return

            if self.drawROI == "EllipseROI":
                pass
            elif self.drawROI == "RectROI":
                if event.button() == Qt.MouseButton.LeftButton:
                    self._roiStartPos = self.mapToScene(event.pos())
                    self._currentRectROI = RectROI(self)
                    self._currentRectROI.setRect(self._roiStartPos.x(), self._roiStartPos.y(), 0, 0)
                    self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
                    QGraphicsView.mousePressEvent(self, event)
                    event.accept()
                else:
                    event.ignore()
            elif self.drawROI == "VectorROI":
                if event.button() == Qt.MouseButton.LeftButton:
                    self._roiStartPos = self.mapToScene(event.pos())
                    self._currentVectorROI = VectorROI(self)
                    self._currentVectorROI.setLine(self._roiStartPos.x(), self._roiStartPos.y(),
                                                 self._roiStartPos.x(), self._roiStartPos.y())
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    pen = QPen(Qt.GlobalColor.black)
                    pen.setCosmetic(True)
                    pen.setWidth(4)
                    self._currentVectorROI.setPen(pen)
                    self.scene.addItem(self._currentVectorROI)
                    event.accept()
            elif self.drawROI == "LineSegmentROI":
                if event.button() == Qt.MouseButton.LeftButton:
                    pos = self.mapToScene(event.pos())
                    if self._currentLineSegmentROI is None:
                        # First click: start new polyline
                        self.setDragMode(QGraphicsView.DragMode.NoDrag)
                        self._currentLineSegmentROI = LineSegmentROI(self)
                        self._currentLineSegmentROI.addPoint(pos)
                        self._currentLineSegmentROI.addPoint(pos)
                        pen = QPen(Qt.GlobalColor.black)
                        pen.setCosmetic(True)
                        pen.setWidth(2)
                        self._currentLineSegmentROI.setPen(pen)
                        self.scene.addItem(self._currentLineSegmentROI)
                        QGraphicsView.mousePressEvent(self, event)
                    else:
                        # Next click: fix vertex and extend new segment preview
                        self._currentLineSegmentROI.updateLastPoint(pos)
                        self._currentLineSegmentROI.addPoint(pos)
                    event.accept()
                    return
            elif self.drawROI == "PolygonROI":
                pass
        elif self.drawROI is None:
            if (self.regionZoomButton is not None) and (event.button() == self.regionZoomButton):
                self._pixelPosition = event.pos()
                self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
                QGraphicsView.mousePressEvent(self, event)
                event.accept()
                self._isZooming = True
                return
            elif (self.selectionButton is not None) and (event.button() == self.selectionButton):
                self._pixelPosition = event.pos()
                self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
                QGraphicsView.mousePressEvent(self, event)
                event.accept()
                self._isSelecting = True
                return

        if (self.zoomOutButton is not None) and (event.button() == self.zoomOutButton):
            if len(self.zoomStack):
                self.zoomStack.pop()
                self.updateViewer()
                self.viewChanged.emit()
            event.accept()
            return

        if (self.panButton is not None) and (event.button() == self.panButton):
            self._pixelPosition = event.pos()
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            if self.panButton == Qt.MouseButton.LeftButton:
                QGraphicsView.mousePressEvent(self, event)
            else:
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier
                                                     | Qt.KeyboardModifier.ControlModifier
                                                     | Qt.KeyboardModifier.AltModifier
                                                     | Qt.KeyboardModifier.MetaModifier)
                dummyEvent = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(event.pos()), Qt.MouseButton.LeftButton,
                                         event.buttons(), dummyModifiers)
                self.mousePressEvent(dummyEvent)
            sceneViewport = self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())
            self._scenePosition = sceneViewport.topLeft()
            event.accept()
            self._isPanning = True
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middleMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                             | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        if event.modifiers() == dummyModifiers:
            QGraphicsView.mouseReleaseEvent(self, event)
            event.accept()
            return

        if (self.panButton is not None) and (event.button() == self.panButton):
            if self.panButton == Qt.MouseButton.LeftButton:
                QGraphicsView.mouseReleaseEvent(self, event)
            else:
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier
                                                     | Qt.KeyboardModifier.ControlModifier
                                                     | Qt.KeyboardModifier.AltModifier
                                                     | Qt.KeyboardModifier.MetaModifier)
                dummyEvent = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(event.pos()),
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

        if self.drawROI is not None:
            if self.selectedROI is not None:
                QGraphicsView.mouseReleaseEvent(self, event)
                self.selectedROI = None
                return
            if event.button() == Qt.MouseButton.RightButton:
                if self._currentLineSegmentROI:
                    self.scene.removeItem(self._currentLineSegmentROI)
                    self._currentLineSegmentROI = None
                else:
                    self.deleteROI(self.roiSelected)
                QGraphicsView.mousePressEvent(self, event)
                event.accept()
                return
            elif self.drawROI == "PointROI":
                if event.button() == Qt.MouseButton.LeftButton:
                    self._roiStartPos = self.mapToScene(event.pos())
                    self._currentPointROI = PointROI(self)
                    radius = self.size().width() // 100
                    self._currentPointROI.setRect(self._roiStartPos.x()- radius/2, self._roiStartPos.y()- radius/2,
                                                  radius, radius)
                    self.addROIs(self._currentPointROI)
                    QGraphicsView.mouseReleaseEvent(self, event)
                    event.accept()
                    return
            elif self.drawROI == "EllipseROI":
                pass
            elif self.drawROI == "RectROI" and self._currentRectROI and self._roiStartPos:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._roiEndPos = self.mapToScene(event.pos())
                    dx = self._roiEndPos.x() - self._roiStartPos.x()
                    dy = self._roiEndPos.y() - self._roiStartPos.y()
                    if dx == 0 or dy == 0:
                        QGraphicsView.mouseReleaseEvent(self, event)
                        return
                    self._currentRectROI.setRect(self._roiStartPos.x(), self._roiStartPos.y(), dx, dy)
                    self.addROIs(self._currentRectROI)
                self._currentRectROI = None
                QGraphicsView.mouseReleaseEvent(self, event)
                event.accept()
                return
            elif self.drawROI == "VectorROI":
                if event.button() == Qt.MouseButton.LeftButton and self._currentVectorROI:
                    self._roiEndPos = self.mapToScene(event.pos())
                    dx = self._roiEndPos.x() - self._roiStartPos.x()
                    dy = self._roiEndPos.y() - self._roiStartPos.y()
                    min_length = 5
                    if math.hypot(dx, dy) >= min_length:
                        self._currentVectorROI.setLine(self._roiStartPos.x(), self._roiStartPos.y(),
                                                        self._roiEndPos.x(), self._roiEndPos.y())
                        self.addROIs(self._currentVectorROI)
                    else:
                        self.scene.removeItem(self._currentVectorROI)
                self._currentVectorROI = None
                QGraphicsView.mouseReleaseEvent(self, event)
                event.accept()
                return
            elif self.drawROI == "LineSegmentROI":
                if event.button() == Qt.MouseButton.LeftButton:
                    pos = self.mapToScene(event.pos())
                    if self._currentLineSegmentROI is not None:
                        self._currentLineSegmentROI.updateLastPoint(pos)
                        self._currentLineSegmentROI.addPoint(pos)
                return

            elif self.drawROI == "PolygonROI":
                pass
        
        elif self.drawROI is None:
            if (self.regionZoomButton is not None) and (event.button() == self.regionZoomButton):
                QGraphicsView.mouseReleaseEvent(self, event)
                if self._isZooming: 
                    zoomRect = self.scene.selectionArea().boundingRect().intersected(self.sceneRect())
                    self.scene.setSelectionArea(QPainterPath())
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    zoomPixelWidth = abs(event.pos().x() - self._pixelPosition.x())
                    zoomPixelHeight = abs(event.pos().y() - self._pixelPosition.y())
                    if zoomPixelWidth > 3 and zoomPixelHeight > 3:
                        if zoomRect.isValid() and (zoomRect != self.sceneRect()):
                            self.zoomStack.append(zoomRect)
                            self.updateViewer()
                            self.viewChanged.emit()
                    self._isZooming = False 
                event.accept()
                return
            elif (self.selectionButton is not None) and (event.button() == self.selectionButton):
                QGraphicsView.mouseReleaseEvent(self, event)
                selectionRect = self.scene.selectionArea().boundingRect().intersected(self.sceneRect())
                self.scene.setSelectionArea(QPainterPath())
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                selectPixelWidth = abs(event.pos().x() - self._pixelPosition.x())
                selectPixelHeight = abs(event.pos().y() - self._pixelPosition.y())
                if selectPixelWidth > 3 and selectPixelHeight > 3:
                    if selectionRect.isValid() and (selectionRect != self.sceneRect()):
                        self.ROIs_selected = {k: [] for k in self.ROIs.keys()}
                        for roi_type, roi_list in self.ROIs.items():
                            for roi in roi_list:
                                if hasattr(roi, "boundingRect"):
                                    roi_rect = roi.mapToScene(roi.boundingRect()).boundingRect()
                                elif hasattr(roi, "line"):
                                    roi_rect = roi.mapToScene(roi.line().boundingRect()).boundingRect()
                                else:
                                    continue
                                if selectionRect.contains(roi_rect):
                                    self.ROIs_selected[roi_type].append(roi)
                        self.roiSelected.emit([self.ROIs_selected])
                        event.accept()
                        self._isSelecting = False
                        return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middleMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        if self.drawROI == "LineSegmentROI" and getattr(self, "_currentLineSegmentROI", None):
            if event.button() == Qt.MouseButton.LeftButton:
                raw_pts = self._currentLineSegmentROI.points()
                cleaned_pts = []
                for p in raw_pts:
                    if not cleaned_pts or (cleaned_pts[-1] - p).manhattanLength() > 1e-4:
                        cleaned_pts.append(p)

                self._currentLineSegmentROI.setPoints(cleaned_pts)
                if len(cleaned_pts) >= 2:
                    self.addROIs(self._currentLineSegmentROI)
                else:
                    self.scene.removeItem(self._currentLineSegmentROI)

                self._currentLineSegmentROI = None
                event.accept()
                return

        if (self.zoomOutButton is not None) and (event.button() == self.zoomOutButton):
            self.clearZoom()
            event.accept()
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mouseDoubleClickEvent(self, event)

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

    def mouseMoveEvent(self, event):
        if self.drawROI == "VectorROI" and getattr(self, "_currentVectorROI", None) and (event.buttons() & Qt.MouseButton.LeftButton):
            currentPos = self.mapToScene(event.pos())
            self._currentVectorROI.setPen(QPen(self._currentVectorROI.pen().color(), 3))
            self._currentVectorROI.setLine(
                self._roiStartPos.x(), self._roiStartPos.y(),
                currentPos.x(), currentPos.y(), 
            )
        elif self.drawROI == "LineSegmentROI" and getattr(self, "_currentLineSegmentROI", None):
            currentPos = self.mapToScene(event.pos())
            self._currentLineSegmentROI.updateLastPoint(currentPos)

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

        scenePos = self.mapToScene(event.pos())
        if self.sceneRect().contains(scenePos):
            x = int(round(scenePos.x() - 0.5))
            y = int(round(scenePos.y() - 0.5))
            imagePos = QPoint(x, y)
        else:
            imagePos = QPoint(-1, -1)
        self.mousePositionOnImageChanged.emit(imagePos)

        QGraphicsView.mouseMoveEvent(self, event)

    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.CrossCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def addROIs(self, roi, setColor=None, labelOverride=None):
        roiType = roi.__class__.__name__
        if roiType == 'VectorROI':
            self._currentLabel = 'Empty Label'
            if labelOverride is None:
                setColor = "#000000"
        if labelOverride is not None:
            self._currentLabel = labelOverride
        i = self._labelList.index(self._currentLabel)+1 if self._currentLabel in self._labelList else 0
        if setColor:
            color = QColor(setColor) 
        else:    
            color = self.colorList[i]
        pen = QPen(color)
        pen.setWidth(3)
        pen.setCosmetic(True)
        roi.setPen(pen)

        if roi.scene() != self.scene:
            self.scene.addItem(roi)

        self.ROIs[roiType].append(roi)
        if self.roiChanged == "loaded":
            return
        else:
            self.roiChanged = "added"
            self.roiList.emit([roi])

    def deleteROI(self, roi):
        roiType = self.drawROI
        if roiType is None:
            roiType = roi.__class__.__name__
        if roi in self.ROIs[roiType]:
            self.scene.removeItem(roi)
            self.ROIs[roiType].remove(roi)
            self.roiChanged = "deleted"
            self.roiList.emit([roi])
            del roi

    def clearROIs(self):
        if any(len(rois) > 0 for rois in self.ROIs.values()):
            for roiType in self.ROIs:
                for roi in self.ROIs[roiType]:
                    self.scene.removeItem(roi)
                self.ROIs[roiType].clear()

    def roiClicked(self, roi):
        roiType = roi.__class__.__name__
        for object in self.ROIs[roiType]:
            if roi is object:
                self.roiSelected.emit([roi])
                break

    def setROIsAreMovable(self):
        for roiType, roiList in self.ROIs.items():
            if roiType == self.drawROI or self.drawROI is None:
                for roi in roiList:
                    roi.setFlags(roi.flags() | QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                    roi.setFlags(roi.flags() | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            else:
                for roi in roiList:
                    roi.setFlags(roi.flags() & ~QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                    roi.setFlags(roi.flags() & ~QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

    def loadROIs(self, df, image, method):
        self.roiChanged = method
        line_segment_groups = {}
        line_segment_labels = {}
        for index, row in df.iterrows():
            self._currentLabel = row['label'] if 'label' in row else "Empty Label"
            if row['file'] == image:
                if row['type'] == 'EllipseROI':
                    roi = EllipseROI(self)
                    roi.setRect(row['x1'], row['y1'], row['dx'], row['dy'])
                    self.addROIs(roi)
                elif row['type'] == 'PointROI':
                    roi = PointROI(self)
                    radius = 10
                    roi.setRect(row['x1']-radius/2, row['y1']-radius/2, radius, radius)
                    self.addROIs(roi)
                elif row['type'] == 'VectorROI':
                    roi = VectorROI(self)
                    roi.setLine(row['x1'], row['y1'], row['dx'], row['dy'])
                    self.addROIs(roi)
                elif row['type'] == 'LineSegmentROI':
                    object_name = row['objectName'] if 'objectName' in row else None
                    if object_name not in line_segment_groups:
                        line_segment_groups[object_name] = []
                        line_segment_labels[object_name] = self._currentLabel
                    line_segment_groups[object_name].append(QPointF(row['x1'], row['y1']))
                elif row['type'] == 'RectROI':
                    roi = RectROI(self)
                    roi.setRect(row['x1'], row['y1'], row['dx'], row['dy'])
                    self.addROIs(roi)

        for object_name, points in line_segment_groups.items():
            if points:
                self._currentLabel = line_segment_labels.get(object_name, "Empty Label")
                roi = LineSegmentROI(self)
                roi.setPoints(points)
                self.addROIs(roi)


class EllipseROI(QGraphicsEllipseItem):
    def __init__(self, viewer):
        QGraphicsItem.__init__(self)
        self._viewer = viewer
        pen = QPen(Qt.GlobalColor.blue)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsMovable)

    def mousePressEvent(self, event):
        QGraphicsItem.mousePressEvent(self, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._viewer.roiClicked(self)
        if event.button() == Qt.MouseButton.RightButton:
            self._viewer.deleteROI(self)


class PointROI(QGraphicsEllipseItem):
    def __init__(self, viewer):
        QGraphicsItem.__init__(self)
        self._viewer = viewer
        pen = QPen(Qt.GlobalColor.blue)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsMovable)

    def mousePressEvent(self, event):
        QGraphicsItem.mousePressEvent(self, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._viewer.roiClicked(self)
        if event.button() == Qt.MouseButton.RightButton:
            self._viewer.deleteROI(self)


class RectROI(QGraphicsRectItem):
    def __init__(self, viewer):
        QGraphicsItem.__init__(self)
        self._viewer = viewer
        pen = QPen(Qt.GlobalColor.yellow)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsMovable)

        self.setAcceptHoverEvents(True)

        self._defaultBrush = QBrush(Qt.BrushStyle.NoBrush)
        self._hoverBrush = QBrush(Qt.GlobalColor.gray, Qt.BrushStyle.SolidPattern)
        self._hoverBrush.setColor(Qt.GlobalColor.gray)
        color = self._hoverBrush.color()
        color.setAlpha(100)
        self._hoverBrush.setColor(color)

        self.setBrush(self._defaultBrush)

    def hoverEnterEvent(self, event):
        self.setBrush(self._hoverBrush)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(self._defaultBrush)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        QGraphicsItem.mousePressEvent(self, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._viewer.roiClicked(self)
        if event.button() == Qt.MouseButton.RightButton:
            self._viewer.deleteROI(self)


class VectorROI(QGraphicsLineItem):
    def __init__(self, viewer):
        QGraphicsItem.__init__(self)
        self._viewer = viewer
        pen = QPen(Qt.GlobalColor.black)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsMovable)
        self.arrow_size = 3

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        line = self.line()
        if line.length() == 0:
            return
        angle = math.atan2(line.dy(), line.dx())
        p2 = line.p2()
        arrow_p1 = p2 - QPointF(
            self.arrow_size * math.cos(angle - math.pi / 6),
            self.arrow_size * math.sin(angle - math.pi / 6)
        )
        arrow_p2 = p2 - QPointF(
            self.arrow_size * math.cos(angle + math.pi / 6),
            self.arrow_size * math.sin(angle + math.pi / 6)
        )
        arrow_head = [p2, arrow_p1, arrow_p2]
        painter.setBrush(self.pen().color())
        painter.drawPolygon(*arrow_head)

    def mousePressEvent(self, event):
        if self._viewer.drawROI == "VectorROI" and getattr(self._viewer, "_currentVectorROI", None) is self:
            event.ignore()
            return

        QGraphicsItem.mousePressEvent(self, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._viewer.roiClicked(self)
        if event.button() == Qt.MouseButton.RightButton:
            self._viewer.deleteROI(self)


class LineSegmentROI(QGraphicsPathItem):
    """A polyline segment ROI."""

    def __init__(self, viewer):
        QGraphicsItem.__init__(self)
        self._viewer = viewer
        self._points = []
        pen = QPen(Qt.GlobalColor.red)
        pen.setCosmetic(True)
        pen.setWidth(3)
        self.setPen(pen)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsMovable)

    def setPoints(self, points):
        self._points = [QPointF(p) for p in points]
        self.updatePath()

    def addPoint(self, pt):
        self._points.append(QPointF(pt))
        self.updatePath()

    def updateLastPoint(self, pt):
        if self._points:
            self._points[-1] = QPointF(pt)
            self.updatePath()

    def points(self):
        return self._points

    def updatePath(self):
        path = QPainterPath()
        if self._points:
            path.moveTo(self._points[0])
            for pt in self._points[1:]:
                path.lineTo(pt)
        self.setPath(path)

    def setLine(self, x1, y1, x2, y2):
        self.setPoints([QPointF(x1, y1), QPointF(x2, y2)])

    def mousePressEvent(self, event):
        if self._viewer.drawROI == "LineSegmentROI" and getattr(self._viewer, "_currentLineSegmentROI", None) is self:
            event.ignore()
            return

        QGraphicsItem.mousePressEvent(self, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._viewer.roiClicked(self)
        if event.button() == Qt.MouseButton.RightButton:
            self._viewer.deleteROI(self)


class PolygonROI(QGraphicsPolygonItem):
    def __init__(self, viewer):
        QGraphicsItem.__init__(self)
        self._viewer = viewer
        pen = QPen(Qt.GlobalColor.yellow)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsMovable)

    def mousePressEvent(self, event):
        QGraphicsItem.mousePressEvent(self, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._viewer.roiClicked(self)


if __name__ == '__main__':
    import sys
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        from PyQt5.QtWidgets import QApplication

    def handleViewChange():
        print("viewChanged")

    app = QApplication(sys.argv)
    viewer = QtImageViewer()
    viewer.open()
    viewer.show()
    sys.exit(app.exec())