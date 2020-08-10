# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import voz2
import os
import threading
import pyttsx3
import sys
import xlsxwriter
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from openpyxl import load_workbook
from utilidades import image_resize
from datetime import datetime
import serial
sexo = int(0)
arduino = serial.Serial('COM3', 9600, timeout=0)
Contador1 = int(0)
Contador2 = int(0)
Contador3 = int(0)
identificador=int(0)
#pab: funcion para calcular el área de la cara detectada
def getArea(startY,endY,startX,endX):
	largo=abs(startY-endY)
	ancho=abs(startX-endX)
	return largo*ancho

def detect_and_predict_mask(frame, faceNet, maskNet):

	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (80, 80),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	area=0

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			#print("detectó uno")
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			#pab: este es el código para calcular sólo una cara
			#	  primero se crea la variable areaAux, que tendra un valor 0 (linea 56),
			# 	  cada vez que detecte una cara en el frame, se calculará el área de dicha cara.
			# 	  La primera cara calculada siempre pasará el if, por lo que se guardará los datos de la cara en faces, su coordenada en locs y su área en area
			#     la siguiente cara se le calculará su área y se comparará con la ya guardada (areaAux vs area respectivamente)
			#     si esta cara tiene mayor área, entonces se eliminan los elementos de la lista de caras y la lista de coordenadas (cara y su ubicación en los pixeles)
			#	  luego sólo predicirá si esa cara con mayor área tiene o no mascara (linea 105)
			areaAux=getArea(startY,endY,startX,endX)
			if areaAux > area:
				area=areaAux
				faces.clear()
				faces.append(face)
				locs.clear()
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on all
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
executor = ThreadPoolExecutor(max_workers=1)
contador = 0
totalMask= 0
totalSinMask= 0
contadorFrames = 0
while True:

	#pab: ajustar el width hasta que se vea bien, con 400 o 600 lo veía poco nitido, igual se puede reducir el 1080
	frame = vs.read()
	frame = imutils.resize(frame, width=500)	
	
	#flipHorizontal = cv2.flip(frame,1)

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# pab: probé y detecta 24 frames por segundo, 
	# 	   si no funciona bien o tu camara tiene otro fps
	# 	   cambiar ese 48 por fps*2 para tener un margen de 2 segundos
	contadorFrames +=1
	if contadorFrames == 48:
		contador=0
		totalMask=0
		totalSinMask=0
		contadorFrames=0
	#if  (totalMask == 0 ) or (totalSinMask == 0):
	#	cv.
	cadena = arduino.readline()
	if(cadena.decode() != '' and cadena.decode() != '0' and cadena.decode() != '1'):
		temp=cadena.decode()
	if(cadena.decode() == '0'):
		temp="0"
	if(cadena.decode() == ''):
		temp="0"
	if(cadena.decode() == '1'):
		temp="99.9"

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Tiene mascarilla" if mask > withoutMask else "No tiene mascarilla"
		color = (0, 255, 0) if label == "Tiene mascarilla" else (0, 0, 255)
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
		contadorFrames=0
		totalMask +=  mask
		totalSinMask +=  withoutMask
		contador = contador+1
		#pab: este contador indica que se tomarán 15 fotos de la persona para precedir si lleva o no máscara
		#	  se puede cambiar a gusto, antes lo tenia en 20 y para probar la ui lo dejaba en 1
		if(cadena.decode()!='' && cadena.decode()!='S' !=cadena.decode()!='s' ):
			if	contador == 5:
				contador = 0
				print("Entró al if")
				print("Total sin mask: " + str(totalSinMask))
				print("Total con mask: " + str(totalMask))
				if totalMask > totalSinMask and float(temp)<37.4:
					print("resultado: con mascarilla")
					executor.submit(voz2.voz,1,float(temp),sexo)	
					voz2.MostrarUI(float(temp),1)
					arduino.write('a'.encode())
				elif totalMask > totalSinMask and float(temp)>=37.4:
					print("resultado: con mascarilla")
					executor.submit(voz2.voz,1,float(temp),sexo)	
					voz2.MostrarUI(float(temp),0)
					arduino.write('a'.encode())
				else:
					print("resultado: sin mascarilla")
					executor.submit(voz2.voz,0,float(temp),sexo)
					voz2.MostrarUI(float(temp),0)
					arduino.write('a'.encode())

		# display the label and bounding box rectangle on the output
		# frame
		
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	#pab: Poner la marca de agua reemplazando los pixeles afectados
	
	#configuraciones de formato
	cv2.namedWindow("Detector de mascarilla", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("Detector de mascarilla",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
	# show the output frame
	cv2.imshow("Detector de mascarilla", frame)
    
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	if key == ord("f"):
		sexo=0
	if key == ord("m"):
		sexo=1
# do a bit of cleanup
#cv2.destroyAllWindows()
#vs.stop()