from PyQt6.QtWidgets import QWidget

from ui.pages.computerVision_ui import Ui_Form


class computerVision(QWidget):
    def __init__(self):
        super(computerVision, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)