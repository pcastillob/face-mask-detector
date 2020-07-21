from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import threading
import xlsxwriter
import csv
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
from PyQt5 import QtGui, QtTest, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTimeEdit
from PyQt5.QtCore import QTime
import sys
 
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Información"
        self.InitWindow()  
 
    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.showMaximized()
        self.MyTime()
        self.show()


    def MyTime(self):
        label = QLabel()
        label.setText("Tiene mascarilla")
        label2 = QLabel()
        label2.setText("T°:")
        label2.setFont(QtGui.QFont("Sanserif", 50))
        label2.setStyleSheet("background-color:black;color:white;")
        label3 = QLabel(self)
        pixmap = QPixmap('LOGO-EQYS.png')
        label3.setPixmap(pixmap)
        label3.setStyleSheet("background-color:black;")
        vbox = QVBoxLayout()
        label.setFont(QtGui.QFont("Sanserif", 50))
        vbox.addWidget(label3)
        vbox.addWidget(label)
        vbox.addWidget(label2)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label3.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("background-color:black; color:lightgreen;")
        self.setLayout(vbox)
        QtTest.QTest.qWait(1500)



 
 
 
 
App = QApplication(sys.argv)
window = Window()
