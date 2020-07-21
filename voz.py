import pyttsx3
import sys
import cv2
import os
import time
import threading
import xlsxwriter
import csv
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
from PyQt5 import QtGui, QtTest
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTimeEdit
from PyQt5.QtCore import QTime
import sys
 

def paass(label):
    imagen=cv2.imread("C:/Users/SIN/Desktop/Python/face-mask-detector/Image/1.jpg",1)
    cv2.namedWindow("Reserva", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Reserva",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Reserva",imagen)
    # Se inicia el motor de voz
    engine = pyttsx3.init()
    #txtTovoice = input("Ingresa tu texto => ")
    # Se genera la voz a partir de un texto
    engine.setProperty('rate',140)
    #print(texto)
    engine.say("wena")
    # Se reproduce la voz
    engine.runAndWait()
#sys.exit()    

   
#cv2.destroyWindows()
App = QApplication(sys.argv)
window = paass()