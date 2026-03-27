from PyQt6.QtWidgets import QLabel, QMenu
from PyQt6.QtGui import QAction, QMouseEvent
from PyQt6.QtCore import Qt

class ContextMenuLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.context_menu = QMenu(self)
        action1 = QAction("Action 1", self)
        action2 = QAction("Action 2", self) 
        action3 = QAction("Action 3", self)

        action1.triggered.connect(self.action1_triggered)
        action2.triggered.connect(self.action2_triggered)
        action3.triggered.connect(self.action3_triggered)

        self.file_menu = QMenu(self, title="File")
        action1 = self.file_menu.addAction("Save")
        action2 = self.file_menu.addAction("Open")
        action3 = self.file_menu.addAction("Delete")

        self.context_menu.addMenu(self.file_menu)

        self.show()
        
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.show_context_menu(event.pos())
        elif event.button() == Qt.MouseButton.RightButton:
            self.show_context_menu(event.pos())
        else:
            super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """
        Override the context menu event to show the custom context menu.
        """
        self.context_menu.exec(event.globalPos())

    def action1_triggered(self):
        """
        Action 1 triggered.
        """
        print("Action 1 triggered")
    def action2_triggered(self):
        """
        Action 2 triggered.
        """
        print("Action 2 triggered")
    def action3_triggered(self):
        """
        Action 3 triggered.
        """
        print("Action 3 triggered")
