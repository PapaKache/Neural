import sys
import os
from PyQt5 import QtWidgets
from  Ui_main import Ui_MainWindow
from main_function import MainWindow
app = QtWidgets.QApplication(sys.argv)
dlg = MainWindow()
#dlg.showFullScreen()
dlg.show()
sys.exit(app.exec_())


