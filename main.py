import six.moves
import dateutil.tz
import dateutil.rrule
# now safe to import PySide6 and everything else

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMenu
from PyQt6.QtGui import QAction, QTransform, QIcon
from PyQt6.QtCore import Qt, QTimer, QElapsedTimer

from ui.main_window_ui import Ui_MainWindow
from pages_functions.home import Home
from pages_functions.ImageLabeling import ImageLabeling
from pages_functions.SPMImageProcess import SPMImageProcess
from pages_functions.IP1 import IP1
from pages_functions.computerVision import computerVision
# from pages_functions.tozota import Tozota


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.settings_widget.hide()  # Hide the settings widget initially
        self.ui.search_frame.hide()  # Hide the search frame until implementation
        ## ============================================
        ## Get all buttons in the main menu window
        ## ============================================
        self.home_btn = self.ui.pushButton
        self.SPMImageProcess_btn = self.ui.pushButton_17
        self.ImageLabeling_btn = self.ui.pushButton_2
        self.IP1_btn = self.ui.pushButton_3
        self.computerVision_btn = self.ui.pushButton_6

        self.settings_btn = self.ui.user_label

        ## ============================================
        # Connect left and right mouse button clicks to show_settings_menu (temporary)
        # self.settings_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # self.settings_btn.customContextMenuRequested.connect(self.show_settings_menu)
        # self.settings_btn.mousePressEvent = self.handle_settings_btn_click

        self.settings_btn.mousePressEvent = self.handle_settings_btn_click

        ## ============================================
        ## Create dict for menu buttons and tab windows
        ## ============================================
        self.menu_btns_dict = {
            self.home_btn: Home,
            self.SPMImageProcess_btn: SPMImageProcess,
            self.ImageLabeling_btn: ImageLabeling,
            self.IP1_btn: IP1,
            self.computerVision_btn: computerVision,
        }
        ## ============================================ 
        ## Show home window when starting app
        ## ==============================u==============
        self.show_home_window()
        ## ============================================
        ## Connect signal and slot
        ## ============================================
        self.ui.tabWidget.tabCloseRequested.connect(self.close_tab)
        self.home_btn.clicked.connect(self.show_selected_window)
        self.ImageLabeling_btn.clicked.connect(self.show_selected_window)
        self.SPMImageProcess_btn.clicked.connect(self.show_selected_window)
        self.IP1_btn.clicked.connect(self.show_selected_window)
        self.computerVision_btn.clicked.connect(self.show_selected_window)

    # def show_settings_menu(self, pos):
    #     menu = QMenu(self)
    #     action1 = QAction("Settings", self)
    #     action2 = QAction("Exit", self)

    #     action1.triggered.connect(self.action1_triggered)
    #     action2.triggered.connect(self.closeApplication)  # Close the application
    #     menu.addAction(action1)
    #     menu.addAction(action2)
    #     menu.exec(self.settings_btn.mapToGlobal(pos))

    def action1_triggered(self):
        """
        Action 1 triggered.
        """
        print("Action 1 triggered")
    
    def closeApplication(self):
        """
        Close the application.
        """
        print("Application closed")
        self.close()

    def handle_settings_btn_click(self, event): #This just routes the left mouse click back to the show_settings_menu context menu (temporary)
        if hasattr(self.settings_btn, "pixmap") and callable(getattr(self.settings_btn, "pixmap", None)):
            original_pixmap = self.settings_btn.pixmap()
            if original_pixmap:
                timer = QTimer(self)
                elapsed = QElapsedTimer()
                elapsed.start()
                duration = 1000 # milliseconds
                interval = 50    # ms per frame
                angle_step = 360 * interval / duration
                self._rotation_angle = 0

                def rotate_pixmap():
                    if elapsed.elapsed() < duration:
                        self._rotation_angle += angle_step
                        transform = QTransform().rotate(self._rotation_angle)
                        rotated = original_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
                        self.settings_btn.setPixmap(rotated)
                    else:
                        self.settings_btn.setPixmap(original_pixmap)
                        timer.stop()

                timer.timeout.connect(rotate_pixmap)
                timer.start(interval)
        if event.button() == Qt.MouseButton.LeftButton:
            # self.show_settings_menu(event.pos())
            if not self.ui.settings_widget.isVisible():
                self.ui.settings_widget.show()
            else:
                self.ui.settings_widget.hide()
        else:
            # Call the default handler for other mouse buttons
            super(type(self.settings_btn), self.settings_btn).mousePressEvent(event)

    def show_home_window(self):
        """
        Function for showing the home window
        :return: None
        """
        result = self.open_tab_flag(self.home_btn)
        self.set_btn_checked(self.home_btn)

        if result[0]:
            self.ui.tabWidget.setCurrentIndex(result[1])
        else:
            tab_title = self.home_btn.text()
            curIndex = self.ui.tabWidget.addTab(Home(), tab_title)
            self.ui.tabWidget.setCurrentIndex(curIndex)
            self.ui.tabWidget.setVisible(True)

    def set_btn_checked(self, btn):
        """
        Set the status of selected button to checked and set all others to unchecked
        :param btn: button to set checked
        :return: 
        """
        for button in self.menu_btns_dict.keys():
            if button != btn.text():
                button.setChecked(False)
            else:
                button.setChecked(True)

    def show_selected_window(self):
        """
        Show the selected window based on the button clicked
        :return: None
        """
        btn = self.sender()
        
        result = self.open_tab_flag(btn.text())
        self.set_btn_checked(btn)

        if result[0]:
            self.ui.tabWidget.setCurrentIndex(result[1])
        else:
            tab_title = btn.text()
            curIndex = self.ui.tabWidget.addTab(self.menu_btns_dict[btn](main_window=self), tab_title)
            self.ui.tabWidget.setCurrentIndex(curIndex)
            self.ui.tabWidget.setVisible(True)

    def close_tab(self, index):
        """
        Close the clicked tab in tabWidget
        :return: None
        """
        self.ui.tabWidget.removeTab(index)

        if self.ui.tabWidget.count() == 0:
            self.ui.toolBox.setCurrentIndex(0)
            self.show_home_window()

    def open_tab_flag(self, btn_text):
        """
        Check if the selected window is shown or not
        :param btn: button text
        :return: None
        """
        open_tab_count = self.ui.tabWidget.count()

        for i in range(open_tab_count):
            tab_title = self.ui.tabWidget.tabText(i)
            if tab_title == btn_text:
                return True, i
            else:
                continue
        return False, None


if __name__ == '__main__':
    import sys



    app = QApplication(sys.argv)


    main_window = MyWindow()
    # main_window.setWindowTitle("MOSAIC") # - Molecular Scan Analysis and Characterization")
    # main_window.setWindowIcon(QIcon("static/Mosaic_logo.ico"))
    
    # Show the main window
    main_window.show()
    
    sys.exit(app.exec())

    ##  .\spm-venv\Lib\site-packages\PySide6\rcc.exe .\static\resource.qrc -g python -o .\static\resource_rc.py
    ## pyside6-rcc -o resource.py resource.qrc      
    ## pyuic6.exe .\main_window.ui -o .\main_window_ui.py  
    ## pyuic6.exe .\ImageLabeling.ui -o .\ImageLabeling_ui.py   