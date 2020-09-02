import logging
import time
import tkinter
from tkinter import PhotoImage
from tkinter.font import Font
import argparse
import imutils
from imutils.video import VideoStream
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

#src=0 indica la cámara 0, puede haber más en el equipo
def setup_webcam_stream(src=0):
    cap = VideoStream(src).start()
        #raise IOError("Cannot read video stream.")
    return cap


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


def getArea(startY,endY,startX,endX):
    largo=abs(startY-endY)
    ancho=abs(startX-endX)
    return largo*ancho

def detect_and_predict_mask(frame, faceNet, maskNet):

    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    #Modificar el blob (numero,numero) para cambiar el tamaño de la detección, con blob(60,60) detecta una persona a unos 30cm, con blob (80,80) a unos 60cm
    blob = cv2.dnn.blobFromImage(frame, 1.0, (60, 60),
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
        if confidence > 0.5:
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
            faces.append(face)
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

#función para obtener la temperatura.
def getTemp():
    #print("entró al getTemp")
    cadena = arduino.readline()
    var = cadena.decode()
    #Este es la condición requerida
    if(cadena != b'' and cadena!=b'100\r\n' and cadena!=b'0\r\n'):
        #print("obtuvo la temperatura")
        print("TEMPERATURA: "+ str(var))
        return var
    elif(cadena == b'1'):
        return 99.9
    elif (cadena == b'0'):
        return 0
    else:
        return 0


'''
Función de proceso principal,
Primnero se definen variables de condición instanciadas en 0 o un valor que no afecte el funcionamiento.

Hay un loop principal que está activo mientras el evento Stop no esté activo, si este se activa se detiene todo el procesamiento.

Detecta (>=5) mascarilla en un rostro hasta que obtiene temperatura, entonces da el resultado


'''
    
def processing_loop(video_stream: VideoStream,resultado_queue: Queue,  stop_event: Event,pausa_event: Event, faceNet , maskNet,audio_aprobado,audio_pongaseMask,audio_tempElevada,audio_noCumpleReq,cargando_event: Event):
    print("PROCESSING")
    contador = 0
    temp=0
    totalMask= 0
    totalSinMask= 0
    contadorFrames = 0
    temp_bool = True #Hay o no temperatura
    while not stop_event.is_set():
        try:
            pausa_event.clear()
            #puede ser 
            #img = imutils.resize(img, width=400)
            #temp = 0
            #cargando_event.set()
            
            #Si se obtiene una temperatura entre 35 y 41, temp bool queda false, de esa forma no hace otra lectura de temperatura, esta temperatura
            #se guarda hasta que el programa entregue el resultado o si hay 48 imagenes sin detectar un rostro. 
            if temp_bool:
                temp = getTemp()
                if float(temp) > 35 and float(temp) < 41:
                    temp_bool = False
                    if not cargando_event.is_set():
                        cargando_event.set()
            #print(temp)
            img=video_stream.read()
            #print("imagen de la cámara")
            contadorFrames +=1
            #Si hay 48 imagenes sin detectar un rostro, se reinician las variables para empezar el funcionamiento desde cero
            if contadorFrames == 48:
                contador=0
                totalMask=0
                totalSinMask=0
                contadorFrames=0
                temp_bool=True
                cargando_event.clear()
            (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                (mask, withoutMask) = pred
                contadorFrames=0
                #print("detectó una cara")
                #contadores resultado de la detección de mascarilla
                totalMask +=  mask
                totalSinMask +=  withoutMask
                contador +=  1
                #si ha hecho 5 o más detecciones está listo para seguir el funcionamiento del procesamiento
                if  contador >= 5:
                    if not cargando_event.is_set():
                        cargando_event.set()
                    #Si ha leido la temperatura puede continuar el funcionamiento del procesamiento
                    if not temp_bool:
                        #el procesamiento termina acá, se reinician las variables condicionales y se entrega el resultado en los if posteriores
                        contador = 0
                        temp_bool = True
                        print("Entró al if")
                        print("Total sin mask: " + str(totalSinMask))
                        print("Total con mask: " + str(totalMask))
                        if float(temp) > 35 and float(temp) < 41:
                            pausa_event.set()
                            #modificar acá para poner el codigo del sensor
                            if totalMask > totalSinMask:
                                if  37.4 > float(temp):   
                                    totalMask=0
                                    totalSinMask=0
                                    print("resultado: con mascarilla y temp aceptable")
                                    resultado_queue.put_nowait("pasa")
                                    audio_aprobado.play()
                                    #voz2.voz(1)
                                    time.sleep(5)
                                    totalSinMask = 0
                                    totalMask =0
                                    pausa_event.clear()
                                    listo = False
                                else:
                                    totalMask=0
                                    totalSinMask=0
                                    #TIENE MASCARILLA
                                    print("resultado: con mascarilla pero temp elevada")
                                    resultado_queue.put_nowait("elevada")
                                    audio_tempElevada.play()
                                    #voz2.voz(1)
                                    listo = False
                                    time.sleep(17)
                                    pausa_event.clear()
                            else:
                                if  37.4 > float(temp):
                                    totalMask=0
                                    totalSinMask=0
                                    print("resultado: sin mascarilla y buena temp")
                                    resultado_queue.put_nowait("denegado")
                                    audio_pongaseMask.play()
                                    listo = False
                                    time.sleep(5)
                                    totalSinMask = 0
                                    totalMask = 0
                                    pausa_event.clear()
                                else:
                                    totalMask=0
                                    totalSinMask=0
                                    print("resultado: sin mascarilla y temp elevada")
                                    resultado_queue.put_nowait("elevada")
                                    audio_noCumpleReq.play()            
                                    time.sleep(17)
                                    pausa_event.clear()
                                    totalSinMask = 0
                                    totalMask = 0
                                    listo = False
        # We need a timeout here to not get stuck when no images are retrieved from the queue
                # output_queue.put(img, timeout=33)
        except Full:
            pass  # try again with a newer frame
    print("STOP DEL PROCESSING")
    sys.exit()

class App:
    def __init__(self, window, resultado_queue:Queue, resultado_event: Event,cargando_event: Event):
        print("iniciando UI")
        self.window = window
        self.font = Font(family="Arial",size=40)
        #self.window.overrideredirect(True)
        self.cargando_event = cargando_event
        self.window.wm_attributes("-fullscreen","true")
        self.resultado_evento = resultado_event
        self.resultado_queue = resultado_queue
        self.fontStyle = Font(family="Arial", size=18)
        # Create a canvas that can fit the above video source size
        self.imagen=PhotoImage(file="negro-blank.png")
        self.label = tkinter.Label(window,compound=tkinter.CENTER)
        self.label.pack()
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 100
        self.num=0
        self.update()
        #self.window
        self.window.mainloop()
    def update_parpadeo(self):
        #print("enbtró al parpadeo")
        hora = time.strftime("%H:%M")
        archivo = "negro-red.png" if self.num%2 == 0 else "negro-blank.png"
        self.imagen=PhotoImage(file=archivo)
        self.label.configure(image=self.imagen,text="\n\n\n\n\n\n\n\n\n\n\n"+hora,fg="white",font=self.font)
        self.label.image=self.imagen
        self.num +=1
        if self.num==34:
            self.num=0
            self.window.after(500,self.update)
        else:
            self.window.after(500,self.update_parpadeo)
        
    def update(self):        
        try:
            hora = time.strftime("%H:%M")
            if self.resultado_evento.is_set():
                res = self.resultado_queue.get()
                if res =="pasa" or res=="denegado":
                    archivo = "negro-green.png" if res == "pasa" else "negro-red.png"
                    self.imagen=PhotoImage(file=archivo)
                    self.label.configure(image=self.imagen,text="\n\n\n\n\n\n\n\n\n\n\n"+hora,fg="white",font=self.font)
                    self.label.image=self.imagen
                    self.window.after(5000,self.update)
                else:
                    self.update_parpadeo()
            else:
                if not self.cargando_event.is_set():
                    txt = "\n\n\n\n\n\n\n\n\n\n\n"+hora
                else:
                    txt = "\n\n"+"Detectando..."+"\n\n\n\n\n\n\n\n\n"+hora

                self.imagen=PhotoImage(file="negro-blank.png")
                self.label.configure(image=self.imagen,text=txt,fg="white",font=self.font)
                self.label.image=self.imagen
                #time.sleep(0.3)
                self.window.after(self.delay,self.update)
        except Empty:
            pass


def main():
    faceNet,maskNet=cargar_modelo() #se carga el modelo
    stream = setup_webcam_stream(0) #se inicia el elemento StreamVideo, el que da las imagenes de la camara
    time.sleep(1) # se da un segundo de espera
    resultado_queue = Queue() #elemento cola (Queue), se comunica con el hilo de la ui para pasarle el resultado de la detección    
    pygame.init() #Se inicializa la libreria mixer, el que reproducie los audios
    audio_aprobado = pygame.mixer.Sound('Audios/Masc/Francisco1.ogg') #se cargan los audios
    audio_pongaseMask = pygame.mixer.Sound('Audios/Masc/Francisco2.ogg')
    audio_tempElevada = pygame.mixer.Sound('Audios/Masc/Francisco3.ogg')
    audio_noCumpleReq = pygame.mixer.Sound('Audios/Masc/Francisco4.ogg')
    #processed_queue = Queue(maxsize=1000)
    stop_event = Event() #Evento stop,  evento compartido entre los hilos, detiene toda ejecucion una vez gatillado
    pausa_event= Event() #Evento pausa, evento compartido que dice a los hilos que ya hay un resultado de 
    cargando_event = Event() #Evento cargando, Le dice a la ui que ponga el texto "cargando"
    
    try:
        #Hilo de ejecuíón que se encarga del procesamiento
        Thread(name="hilo2",target=processing_loop, args=[stream,resultado_queue, stop_event,pausa_event, faceNet, maskNet, audio_aprobado,audio_pongaseMask,audio_tempElevada,audio_noCumpleReq,cargando_event]).start()
        #Hilo principal del codigo, se encarga de la UI, Tkinter no funciona bien en otro hilo que no sea el principal 
        App(tkinter.Tk(), resultado_queue,pausa_event,cargando_event)
    finally:
        print("APRETANDO STOP")
        #si se cierra la ui se gatilla el evento Stop, el que también detiene el procesamiento
        stop_event.set()
    print("Fin del programa")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
