from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import pyttsx3
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
from openpyxl import load_workbook
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import Qt 

 

def paass(label):

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
                filepath1="Datos.xlsx"
        # load demo.xlsx 
                wb1=load_workbook(filepath1)
# select demo.xlsx
                sheet1=wb1.active
# set value for cell A1=1
                engine = pyttsx3.init()
                if (sheet1['A1'].value==0):
                    label = QLabel()
                    label.setText("Ponte mascarilla")
                    label.setFont(QtGui.QFont("Sanserif", 50))
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    label.setStyleSheet("background-color:black; color:red;")
                    engine.setProperty('rate',140)
                    engine.say("ponte mascarilla")
                    vbox.addWidget(label)
                # Se reproduce la voz
                    engine.runAndWait()
                if (sheet1['A1'].value==1):
                    label = QLabel()
                    label.setText("Mascarilla puesta correctamente")
                    label.setFont(QtGui.QFont("Sanserif", 50))
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    label.setStyleSheet("background-color:black; color:lightgreen;")
                    engine.setProperty('rate',140)
                    engine.say("siga adelante")
                    vbox.addWidget(label)
                # Se reproduce la voz
                    engine.runAndWait()
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
