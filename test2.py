import logging
import time
import tkinter
from tkinter import PhotoImage
from tkinter.font import Font
import argparse
import imutils
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from queue import Full, Queue, Empty
from threading import Thread, Event
#from MYPIL import ImageTk, Image
import cv2
import voz2
import os
import pygame
import sys


logger = logging.getLogger("VideoStream")


def setup_webcam_stream(src=0):
    cap = cv2.VideoCapture(src)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Camera dimensions: {width, height}")
    logger.info(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    grabbed, frame = cap.read()  # Read once to init
    if not grabbed:
        raise IOError("Cannot read video stream.")
    return cap, width, height

def cargar_modelo():
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
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])
    return faceNet,maskNet

def video_stream_loop(video_stream: cv2.VideoCapture, img_process: Queue, stop_event: Event,pausa_event: Event):
    print("VIDEO STREAM LOOP")
    while not stop_event.is_set():
        try:
            success, img = video_stream.read()
            if success:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if not pausa_event.is_set():
                    img_process.put(img,timeout=30)
                else:
                    time.sleep(5)
        except Full:
            pass  # try again with a newer frame
    print("STOP DEL VIDEO")
    video_stream.release()
    sys.exit()

def getArea(startY,endY,startX,endX):
	largo=abs(startY-endY)
	ancho=abs(startX-endX)
	return largo*ancho

def detect_and_predict_mask(frame,faceNet,maskNet):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,1.0, (80,80), (104.0,177.0,123.0))
    faceNet.setInput(blob)
    detections=faceNet.forward()
    faces=[]
    locs=[]
    preds = []
    area=0
    
    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            (startX,startY) = (max(0,startX),max(0,startY))
            (endX,endY) = (min(w-1,endX),min(h-1,endY))
            face=frame[startY:endY,startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            areaAux=getArea(startY,endY,startX,endX)
            if areaAux > area:
                area=areaAux
                faces.clear()
                faces.append(face)
                locs.clear()
                locs.append((startX,startY,endX,endY))

    if len(faces)>0:
        faces = np.array(faces,dtype="float32")
        preds=maskNet.predict(faces,batch_size=32)
    return (locs,preds)

def processing_loop(input_queue: Queue,resultado_queue: Queue,  stop_event: Event,pausa_event: Event, faceNet , maskNet,audio_aprobado,audio_rechazado):
    print("PROCESSING")
    contador = 0
    totalMask= 0
    totalSinMask= 0
    contadorFrames = 0
    while not stop_event.is_set():
        try:
            img = input_queue.get(timeout=20)
            #img = imutils.resize(img, width=500)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = img[:, ::-1]  # mirror
            #time.sleep(0.01)  # simulate some processing time
            (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
            contadorFrames +=1
            if contadorFrames == 24:
                contador=0
                totalMask=0
                totalSinMask=0
                contadorFrames=0
           # print(preds)
            for (box, pred) in zip(locs, preds):
                (mask, withoutMask) = pred
                # determine the class label and color we'll use to draw
                # the bounding box and text                
                contadorFrames=0
                totalMask +=  mask
                totalSinMask +=  withoutMask
                contador = contador+1
                if	contador == 15:
                    contador = 0
                    print("Entró al if")
                    print("Total sin mask: " + str(totalSinMask))
                    print("Total con mask: " + str(totalMask))
                    pausa_event.set()
                    #modificar acá para poner el codigo del sensor
                    if totalMask > totalSinMask:
                        totalMask=0
                        totalSinMask=0
                        #TIENE MASCARILLA
                        print("resultado: con mascarilla")
                        resultado_queue.put_nowait("pasa")
                        audio_aprobado.play()
                        for n in range(input_queue.qsize()):
                            input_queue.get_nowait()
                        #voz2.voz(1)
                        time.sleep(5)
                        pausa_event.clear()	
                    else:
                        totalMask=0
                        totalSinMask=0
                        #NO TIENE MASCARILLA
                        print("resultado: sin mascarilla")
                        resultado_queue.put_nowait("denegado")
                        audio_rechazado.play()
                        for n in range(input_queue.qsize()):
                            input_queue.get_nowait()
                        #voz2.voz(0)
                        time.sleep(5)
                        pausa_event.clear()
            # We need a timeout here to not get stuck when no images are retrieved from the queue
           # output_queue.put(img, timeout=33)
        except Full:
            pass  # try again with a newer frame
    print("STOP DEL PROCESSING")
    sys.exit()

class App:
    def __init__(self, window, resultado_queue:Queue, resultado_event: Event):
        print("iniciando UI")
        self.window = window
        #self.window.overrideredirect(True)
        #self.window.wm_attributes("-fullscreen","true")
        self.resultado_evento = resultado_event
        self.resultado_queue = resultado_queue
        self.fontStyle = Font(family="Arial", size=18)
        # Create a canvas that can fit the above video source size
        self.imagen=PhotoImage(file="negro-blank.png")
        self.label = tkinter.Label(window)
        self.label.pack()
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()
        self.window.mainloop()

    def update(self):        
        try:
            if self.resultado_evento.is_set():
                res = self.resultado_queue.get(timeout=0.2)
                archivo = "negro-green.png" if res == "pasa" else "negro-red.png"
                self.imagen=PhotoImage(file=archivo)
                self.label.configure(image=self.imagen)
                self.label.image=self.imagen
                self.window.after(5000,self.update)
            else:
                self.imagen=PhotoImage(file="negro-blank.png")
                self.label.configure(image=self.imagen)
                self.label.image=self.imagen
                #time.sleep(0.3)
                self.window.after(self.delay,self.update)
        except Empty:
            pass



def main():
    faceNet,maskNet=cargar_modelo()
    stream, width, height = setup_webcam_stream(0)
    resultado_queue = Queue()
    #evento_ui = Event()

    cola_processing = Queue()
    pygame.init()
    audio_aprobado = pygame.mixer.Sound('Audios/Masc/Francisco1.ogg')
    audio_rechazado = pygame.mixer.Sound('Audios/Masc/Francisco2.ogg')
    #processed_queue = Queue(maxsize=1000)
    stop_event = Event()
    pausa_event= Event()
    try:
        Thread(name="hilo1",target=video_stream_loop, args=[stream, cola_processing, stop_event,pausa_event]).start()
        Thread(name="hilo2",target=processing_loop, args=[cola_processing ,resultado_queue, stop_event,pausa_event, faceNet, maskNet, audio_aprobado,audio_rechazado]).start()
        App(tkinter.Tk(), resultado_queue,pausa_event)
    finally:
        print("APRETANDO STOP")
        stop_event.set()

    print(f"UI queue: {resultado_queue.qsize()}")
    print(f"process queue: {cola_processing.qsize()}")
    
  #  print(f"Processed queue: {processed_queue.qsize()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()