# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'App1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets, QtTest


def paass(label):



    class Ui_MainWindow(object):
        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(800, 600)
            MainWindow.setAutoFillBackground(False)
            MainWindow.setStyleSheet("color: rgb(0, 0, 0);\n"
    "background-color: rgb(0, 0, 0);")
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
            self.plainTextEdit.setGeometry(QtCore.QRect(-190, 280, 104, 71))
            self.plainTextEdit.setObjectName("plainTextEdit")
            self.Logo = QtWidgets.QLabel(self.centralwidget)
            self.Logo.setGeometry(QtCore.QRect(60, 0, 601, 151))
            self.Logo.setStyleSheet("background-image: url(C:/Users/SIN/Desktop/Python/face-mask-detector/LOGO-EQYS.png);")
            self.Logo.setText("")
            self.Logo.setPixmap(QtGui.QPixmap("LOGO-EQYS.png"))
            self.Logo.setObjectName("Logo")
            self.label_2 = QtWidgets.QLabel(self.centralwidget)
            self.label_2.setGeometry(QtCore.QRect(130, 220, 1000, 91))
            self.label_2.setStyleSheet("color: rgb(85, 255, 0);\n"
    "font: 48pt \"MS Shell Dlg 2\";\n"
    "")
            self.label_2.setObjectName("label_2")
            self.label_3 = QtWidgets.QLabel(self.centralwidget)
            self.label_3.setGeometry(QtCore.QRect(130, 320, 101, 101))
            self.label_3.setStyleSheet("color: rgb(255, 255, 255);\n"
    "font: 48pt \"MS Shell Dlg 2\";")
            self.label_3.setObjectName("label_3")
            self.ValorTemperatura = QtWidgets.QLabel(self.centralwidget)
            self.ValorTemperatura.setGeometry(QtCore.QRect(240, 320, 391, 101))
            self.ValorTemperatura.setStyleSheet("font: 48pt \"MS Shell Dlg 2\";\n"
    "color: rgb(255, 255, 255);")
            self.ValorTemperatura.setText("")
            self.ValorTemperatura.setObjectName("ValorTemperatura")
            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 21))
            self.menubar.setObjectName("menubar")
            MainWindow.setMenuBar(self.menubar)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)
            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)



        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            self.label_2.setText(_translate("MainWindow", "Tiene Mascarilla "))
            self.label_3.setText(_translate("MainWindow", "T°:"))




    if __name__ == "__main__":
        import sys
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        QtTest.QTest.qWait(1500)
