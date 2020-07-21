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
#from detect_mask_video import tempppp


#print(tempppp)
 

def paass2(label):

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
                
                            
                #sys.exit(1)
                #window.destroy()

            #def 

         
            def MyTime(self):
                label = QLabel()
                label.setText("Bienvenido")
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
                label.setStyleSheet("background-color:black; color:green;")
                self.setLayout(vbox)
                vbox = QVBoxLayout()
                time = QTime()
                time.setHMS(13,15,40)
                #engine = pyttsx3.init()
                #txtTovoice = input("Ingresa tu texto => ")
                # Se genera la voz a partir de un texto
                #engine.setProperty('rate',200)
                #print(texto)
                #engine.say("Bienvenido")
                # Se reproduce la voz
                #engine.runAndWait()
                #QtTest.QTest.qWait(2)
         
             
        App = QApplication(sys.argv)
        window = Window()
        #sys.exit(App.exec())
 
def paass(label):

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
                
                            
                #sys.exit(1)
                #window.destroy()

            #def 

         
            def MyTime(self):
                
                label = QLabel()
                label.setText("Ponte mascarilla")
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
                label.setStyleSheet("background-color:black; color:red;")
                self.setLayout(vbox)
                vbox = QVBoxLayout()
                time = QTime()
                time.setHMS(13,15,40)
                #engine = pyttsx3.init()
                #txtTovoice = input("Ingresa tu texto => ")
                # Se genera la voz a partir de un texto
                #engine.setProperty('rate',200)
                #print(texto)
                #engine.say("ponte mascarilla")
                # Se reproduce la voz
                #engine.runAndWait()
                #QtTest.QTest.qWait(2)
         
             
        App = QApplication(sys.argv)
        window = Window()

 
#if __name__ == "__main__":
# textTovoice(sys.argv[0]) 
