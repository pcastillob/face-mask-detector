from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import pyttsx3
import os
import threading
import csv
import logging
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
from PyQt5 import QtGui, QtTest, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTimeEdit
from PyQt5.QtCore import QTime
import sys
from openpyxl import load_workbook
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import Qt 

#logging.basicConfig(level=logging.DEBUG, format='%(threadName)s: %(message)s')

def voz(mascarillaBool):
   # logging.info('Ejecutando voz')
    engine = pyttsx3.init()
    engine.setProperty('rate',140)
    if mascarillaBool==0:
        engine.say("ponte mascarilla")
        engine.runAndWait()
    else:
        engine.say("siga adelante")
        engine.runAndWait()

def paass(label,mascarillaBool):
        class Window(QWidget):
            def __init__(self):
                super().__init__()
                self.setStyleSheet("background-color: black;")
                self.setWindowTitle("no title") 
                self.showMaximized()
                self.MyTime()
            def MyTime(self):
                vbox = QVBoxLayout()
                label3 = QLabel(self)
                vbox.addWidget(label3)
              #  logging.info('Ejecutando UI')
                if (mascarillaBool==0):
                    label = QLabel()
                    label.setText("Ponte mascarilla")
                    label.setFont(QtGui.QFont("Sanserif", 50))
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    label.setStyleSheet("background-color:black; color:red;")            
                    vbox.addWidget(label)

                if (mascarillaBool==1):
                    label = QLabel()
                    label.setText("Mascarilla puesta correctamente")
                    label.setFont(QtGui.QFont("Sanserif", 50))
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    label.setStyleSheet("background-color:black; color:lightgreen;")
                    vbox.addWidget(label)
                    
                label2 = QLabel()
                label2.setText("T°:")
                label2.setFont(QtGui.QFont("Sanserif", 50))
                label2.setStyleSheet("background-color:black;color:white;")
                pixmap = QPixmap('LOGO-EQYS.png')
                label3.setPixmap(pixmap)
                label3.setStyleSheet("background-color:black;")
                vbox.addWidget(label2)
                label3.setAlignment(QtCore.Qt.AlignCenter)
                self.setLayout(vbox)
                
                QtTest.QTest.qWait(2000)   
        App = QApplication(sys.argv)
        dialog = QDialog()
        dialog.showFullScreen()
        window = Window()
        #sys.exit(App.exec())
 
 
#if __name__ == "__main__":
# textTovoice(sys.argv[0]) 
