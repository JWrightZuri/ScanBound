from PyQt6.QtWidgets import QWidget

from ui.pages.IP1_ui import Ui_Form


class IP1(QWidget):
    def __init__(self):
        super(IP1, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)