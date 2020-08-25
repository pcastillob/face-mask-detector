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
from queue import Full, Queue, Empty, LifoQueue
from threading import Thread, Event
import cv2
import voz2
import os
import pygame
import sys
import serial
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=0)


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

def detect_and_predict_mask(frame, faceNet, maskNet):

    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (50, 50),
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
        if confidence > 50:
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
            #     primero se crea la variable areaAux, que tendra un valor 0 (linea 56),
            #     cada vez que detecte una cara en el frame, se calculará el área de dicha cara.
            #     La primera cara calculada siempre pasará el if, por lo que se guardará los datos de la cara en faces, su coordenada en locs y su área en area
            #     la siguiente cara se le calculará su área y se comparará con la ya guardada (areaAux vs area respectivamente)
            #     si esta cara tiene mayor área, entonces se eliminan los elementos de la lista de caras y la lista de coordenadas (cara y su ubicación en los pixeles)
            #     luego sólo predicirá si esa cara con mayor área tiene o no mascara (linea 105)
            #areaAux=getArea(startY,endY,startX,endX)
            #if areaAux > area:
            #area=areaAux
            #faces.clear()
            faces.append(face)
            #locs.clear()
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on all
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=2)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def getTemp(temp_event: Event,stop_event: Event,tempe_queue:Queue):
    contador = 0
    while not stop_event.is_set():
        if not temp_event.is_set():
            cadena = arduino.readline()
            time.sleep(1)
            try:
                var = cadena.decode()
                if(cadena != b'' and cadena!=b'100\r\n' and cadena!=b'0\r\n'):
                    print("VAR: ...")    
                    temp_event.set()
                    tempe_queue.put(var)
                elif(cadena == b'1'):
                    temp=99.9
                    #cadena!=b'100\r\n'
                elif (cadena == b'0'):
                    temp=0
                        #print("EVENT TEMPERATURA:")
                        #print(temp_event.is_set())
                    #print("EVENT TEMPERATURA:")
                    #print(temp_event.is_set())
                elif():
                    pass
                else:
                    temp=0
            except:
                pass
def processing_loop(input_queue: Queue,resultado_queue: Queue,  stop_event: Event,pausa_event: Event, faceNet , maskNet,audio_aprobado,audio_pongaseMask,audio_tempElevada,audio_noCumpleReq,temp_event,tempe_queue):
    print("PROCESSING")
    contador = 0
    temp=0
    totalMask= 0
    totalSinMask= 0
    contadorFrames = 0
    while not stop_event.is_set():
        try:
            img = input_queue.get(timeout=15)
            img = imutils.resize(img, width=400)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            contadorFrames +=1
            if contadorFrames == 48:
                contador=0
                totalMask=0
                totalSinMask=0
                contadorFrames=0
                temp_event.clear()
            if temp_event.is_set():
                print("Hay evento temepratura")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
                for (box, pred) in zip(locs, preds):
                    (mask, withoutMask) = pred
                    # determine the class label and color we'll use to draw
                    # the bounding box and text                
                    contadorFrames=0
                    print("DETECTÓ UNA CARA")
                    totalMask +=  mask
                    totalSinMask +=  withoutMask
                    contador +=  1
                    if  contador == 5:
                        contador = 0
                        print("Entró al if")
                        print("Total sin mask: " + str(totalSinMask))
                        print("Total con mask: " + str(totalMask))
                        pausa_event.set()
                        temp = tempe_queue.get()
                        print(str(temp))
                        #modificar acá para poner el codigo del sensor
                        if totalMask > totalSinMask:
                            if  37.4 > float(temp):   
                                totalMask=0
                                totalSinMask=0
                                print("resultado: con mascarilla y temp aceptable")
                                resultado_queue.put_nowait("pasa")
                                audio_aprobado.play()
                                for n in range(input_queue.qsize()):
                                    input_queue.get_nowait()
                                #voz2.voz(1)
                                time.sleep(5)
                                pausa_event.clear()
                                temp_event.clear()
                            else:
                                totalMask=0
                                totalSinMask=0
                                #TIENE MASCARILLA
                                print("resultado: con mascarilla pero temp elevada")
                                resultado_queue.put_nowait("denegado")
                                audio_tempElevada.play()
                                for n in range(input_queue.qsize()):
                                    input_queue.get_nowait()
                                #voz2.voz(1)
                                time.sleep(5)
                                pausa_event.clear()
                                temp_event.clear()
                        else:
                            if  37.4 > float(temp):
                                totalMask=0
                                totalSinMask=0
                                print("resultado: sin mascarilla y buena temp")
                                resultado_queue.put_nowait("denegado")
                                audio_pongaseMask.play()
                                for n in range(input_queue.qsize()):
                                    input_queue.get_nowait()
                                time.sleep(5)
                                pausa_event.clear()
                                temp_event.clear()
                            else:
                                totalMask=0
                                totalSinMask=0
                                print("resultado: sin mascarilla y temp elevada")
                                resultado_queue.put_nowait("denegado")
                                audio_noCumpleReq.play()
                                for n in range(input_queue.qsize()):
                                    input_queue.get_nowait()
                                time.sleep(5)
                                pausa_event.clear()
                                temp_event.clear()
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
        self.font = Font(family="Arial",size=40)
        #self.window.overrideredirect(True)
        self.window.wm_attributes("-fullscreen","true")
        self.resultado_evento = resultado_event
        self.resultado_queue = resultado_queue
        self.fontStyle = Font(family="Arial", size=18)
        # Create a canvas that can fit the above video source size
        self.imagen=PhotoImage(file="negro-blank.png")
        self.label = tkinter.Label(window,compound=tkinter.CENTER)
        self.label.pack()
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()
        #self.window
        self.window.mainloop()

    def update(self):        
        try:
            hora = time.strftime("%H:%M")
            if self.resultado_evento.is_set():
                res = self.resultado_queue.get()
                archivo = "negro-green.png" if res == "pasa" else "negro-red.png"
                self.imagen=PhotoImage(file=archivo)
                self.label.configure(image=self.imagen,text="\n\n\n\n\n\n\n\n\n\n\n"+hora,fg="white",font=self.font)
                self.label.image=self.imagen
                self.window.after(5000,self.update)
            else:
                self.imagen=PhotoImage(file="negro-blank.png")
                self.label.configure(image=self.imagen,text="\n\n\n\n\n\n\n\n\n\n\n"+hora,fg="white",font=self.font)
                self.label.image=self.imagen
                #time.sleep(0.3)
                self.window.after(self.delay,self.update)
        except Empty:
            pass



def main():
    faceNet,maskNet=cargar_modelo()
    stream, width, height = setup_webcam_stream(0)
    resultado_queue = Queue()
    tempe_queue=Queue()    
    cola_processing = Queue()
    pygame.init()
    audio_aprobado = pygame.mixer.Sound('Audios/Masc/Francisco1.ogg')
    audio_pongaseMask = pygame.mixer.Sound('Audios/Masc/Francisco2.ogg')
    audio_tempElevada = pygame.mixer.Sound('Audios/Masc/Francisco3.ogg')
    audio_noCumpleReq = pygame.mixer.Sound('Audios/Masc/Francisco4.ogg')
    #processed_queue = Queue(maxsize=1000)
    stop_event = Event()
    pausa_event= Event()
    temp_event=Event()
    try:
        Thread(name="hilo1",target=video_stream_loop, args=[stream, cola_processing, stop_event,pausa_event]).start()
        Thread(name="hilo2",target=processing_loop, args=[cola_processing ,resultado_queue, stop_event,pausa_event, faceNet, maskNet, audio_aprobado,audio_pongaseMask,audio_tempElevada,audio_noCumpleReq,temp_event,tempe_queue]).start()
        Thread(name="hilo3",target=getTemp,args=[temp_event,stop_event,tempe_queue]).start()
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