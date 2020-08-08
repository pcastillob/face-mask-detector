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
#from PIL import Image,ImageFont,ImageDraw
import voz2
import os
import threading
import pyttsx3
import sys
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from openpyxl import load_workbook
from utilidades import image_resize

from datetime import datetime





# construct the argument parser and parse the arguments




class Applicacion:
	def __init__(self):
		self.ap = argparse.ArgumentParser()
		self.ap.add_argument("-f", "--face", type=str,
			default="face_detector",
			help="path to face detector model directory")
		self.ap.add_argument("-m", "--model", type=str,
			default="mask_detector.model",
			help="path to trained face mask detector model")
		self.ap.add_argument("-c", "--confidence", type=float, default=0.5,
			help="minimum probability to filter weak detections")
		self.args = vars(self.ap.parse_args())

		# load our serialized face detector model from disk
		self.prototxtPath = os.path.sep.join([self.args["face"], "deploy.prototxt"])
		self.weightsPath = os.path.sep.join([self.args["face"],
			"res10_300x300_ssd_iter_140000.caffemodel"])
		self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)

		# load the face mask detector model from disk
		self.maskNet = load_model(self.args["model"])

		# initialize the video stream and allow the camera sensor to warm up
		self.vs = VideoStream(src=0).start()
		time.sleep(2.0)

		# loop over the frames from the video stream
		self.executor = ThreadPoolExecutor(max_workers=1)
		self.contador = 0
		self.totalMask= 0
		self.totalSinMask= 0
		self.contadorFrames = 0

		# Crear elementos gráficos
		#self.root = tk.TK()
		self.video_loop()

	def video_loop(self):
		while True:
			frame = self.vs.read()
			frame = imutils.resize(frame,width=400)	
			#flipHorizontal = cv2.flip(frame,1)
			(locs, preds) = self.detect_and_predict_mask(frame, self.faceNet, self.maskNet)
			self.contadorFrames +=1
			if self.contadorFrames == 48:
				self.contador=0
				self.totalMask=0
				self.totalSinMask=0
				self.contadorFrames=0
			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred
				label = "Tiene mascarilla" if mask > withoutMask else "No tiene mascarilla"
				color = (0, 255, 0) if label == "Tiene mascarilla" else (0, 0, 255)
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
				self.contadorFrames=0
				self.totalMask +=  mask
				self.totalSinMask +=  withoutMask
				self.contador = self.contador+1
				if	self.contador == 5:
					self.contador = 0
					print("Entró al if")
					print("Total sin mask: " + str(self.totalSinMask))
					print("Total con mask: " + str(self.totalMask))
					if self.totalMask > self.totalSinMask:
						print("resultado: con mascarilla")
						self.executor.submit(voz2.voz,1)	
						voz2.MostrarUI(37,1)	
					else:
						print("resultado: sin mascarilla")
						self.executor.submit(voz2.voz,0)
						voz2.MostrarUI(38.5,0)		
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			#pab: Poner la marca de agua reemplazando los pixeles afectados
			#configuraciones de formato
			#cv2.namedWindow("ventana", cv2.WND_PROP_FULLSCREEN)
			#cv2.setWindowProperty("ventana",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
			#cv2.namedWindow("ventana", cv2.WINDOW_NORMAL)
			#cv2.setWindowProperty("ventana",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
			#cv2.resizeWindow('ventana', 2000, 2000) 
			#cv2.setWindowProperty("ventana",CV2_WND_PROP_ASPECTRATIO,CV_WINDOW_FREERATIO)
			#cv2.setWindowProperty("ventana",cv2.WND_PROP_FULLSCREEN,cv2.WND_PROP_FULLSCREEN)

			cv2.namedWindow("ventana", cv2.WINDOW_NORMAL)
			cv2.setWindowProperty("ventana",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

			#cvSetWindowProperty("ventana", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
			cv2.imshow("ventana", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

	def getArea(self,startY,endY,startX,endX):
		largo=abs(startY-endY)
		ancho=abs(startX-endX)
		return largo*ancho

	def detect_and_predict_mask(self,frame, faceNet, maskNet):
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (80, 80),(104.0, 177.0, 123.0))
		faceNet.setInput(blob)
		detections = faceNet.forward()
		faces = []
		locs = []
		preds = []
		area=0

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > self.args["confidence"]:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				areaAux=self.getArea(startY,endY,startX,endX)
				if areaAux > area:
					area=areaAux
					faces.clear()
					faces.append(face)
					locs.clear()
					locs.append((startX, startY, endX, endY))
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on all
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			faces = np.array(faces, dtype="float32")
			preds = maskNet.predict(faces, batch_size=32)
		return (locs, preds)

# do a bit of cleanup
#cv2.destroyAllWindows()
#vs.stop()
app = Applicacion()