
import pyttsx3
import sys
import cv2
import os
import time
from sys import exit


def paass(label):
    imagen=cv2.imread("Image/Plan3.png",1)
    cv2.namedWindow("Reserva", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Reserva",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Reserva",imagen)
    # Se inicia el motor de voz
    engine = pyttsx3.init()
    #txtTovoice = input("Ingresa tu texto => ")
    # Se genera la voz a partir de un texto
    engine.setProperty('rate',140)
    #print(texto)
    engine.say("Tus datos")
    # Se reproduce la voz
    engine.runAndWait()
    

    exit()


#if __name__ == "__main__":
 #   textTovoice(sys.argv[1])