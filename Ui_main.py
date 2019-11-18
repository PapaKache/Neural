# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\mnist\main.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(275, 645)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButtonTrain = QtWidgets.QPushButton(self.centralWidget)
        self.pushButtonTrain.setGeometry(QtCore.QRect(10, 420, 251, 41))
        self.pushButtonTrain.setObjectName("pushButtonTrain")
        self.calendarWidget = QtWidgets.QCalendarWidget(self.centralWidget)
        self.calendarWidget.setGeometry(QtCore.QRect(10, 10, 248, 197))
        self.calendarWidget.setObjectName("calendarWidget")
        self.textEditResult = QtWidgets.QTextEdit(self.centralWidget)
        self.textEditResult.setGeometry(QtCore.QRect(10, 230, 251, 71))
        self.textEditResult.setObjectName("textEditResult")
        self.pushButtonInference = QtWidgets.QPushButton(self.centralWidget)
        self.pushButtonInference.setGeometry(QtCore.QRect(10, 320, 251, 41))
        self.pushButtonInference.setObjectName("pushButtonInference")
        self.progressBarPercent = QtWidgets.QProgressBar(self.centralWidget)
        self.progressBarPercent.setGeometry(QtCore.QRect(10, 560, 261, 23))
        self.progressBarPercent.setProperty("value", 24)
        self.progressBarPercent.setObjectName("progressBarPercent")
        self.progressBarAcc = QtWidgets.QProgressBar(self.centralWidget)
        self.progressBarAcc.setGeometry(QtCore.QRect(10, 590, 261, 23))
        self.progressBarAcc.setProperty("value", 24)
        self.progressBarAcc.setObjectName("progressBarAcc")
        self.horizontalSlider = QtWidgets.QSlider(self.centralWidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(10, 530, 231, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.labelTime = QtWidgets.QLabel(self.centralWidget)
        self.labelTime.setGeometry(QtCore.QRect(253, 530, 21, 20))
        self.labelTime.setObjectName("labelTime")
        self.pushButtonStop = QtWidgets.QPushButton(self.centralWidget)
        self.pushButtonStop.setGeometry(QtCore.QRect(10, 470, 251, 41))
        self.pushButtonStop.setObjectName("pushButtonStop")
        self.pushButtonAcc = QtWidgets.QPushButton(self.centralWidget)
        self.pushButtonAcc.setGeometry(QtCore.QRect(10, 370, 251, 41))
        self.pushButtonAcc.setObjectName("pushButtonAcc")
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "  "))
        self.pushButtonTrain.setText(_translate("MainWindow", "训练"))
        self.pushButtonInference.setText(_translate("MainWindow", "预测"))
        self.labelTime.setText(_translate("MainWindow", "0"))
        self.pushButtonStop.setText(_translate("MainWindow", "停止"))
        self.pushButtonAcc.setText(_translate("MainWindow", "当前准确率"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
