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
from MYPIL import ImageTk, Image
import cv2
import voz2
import os


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

def video_stream_loop(video_stream: cv2.VideoCapture, queue: Queue, stop_event: Event):
    print("VIDEO STREAM LOOP")
    while not stop_event.is_set():
        try:
            success, img = video_stream.read()
            # We need a timeout here to not get stuck when no images are retrieved from the queue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            queue.put(img, timeout=30)
        except Full:
            pass  # try again with a newer frame

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
            face=cv2.resize(face,(244,244))
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

def processing_loop(input_queue: Queue, output_queue: Queue, stop_event: Event, faceNet , maskNet):
    print("PROCESSING")
    contador = 0
    totalMask= 0
    totalSinMask= 0
    contadorFrames = 0
    
    """if input_queue.qsize() >= 70:
                with input_queue.mutex:
                    input_queue.queue.clear()"""
    
    while not stop_event.is_set():
        
        try:
            
           # print("TRY")
            img = input_queue.get()
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
                print("total sin mask: "+str(totalSinMask))
                if	contador == 15:
                    contador = 0
                    print("Entró al if")
                    print("Total sin mask: " + str(totalSinMask))
                    print("Total con mask: " + str(totalMask))
                    if totalMask > totalSinMask:
                        print("resultado: con mascarilla")
                        voz2.voz(1)
                        with output_queue.mutex:
                            output_queue.queue.clear()
                     #   voz2.MostrarUI(37,1)	
                    else:
                        print("resultado: sin mascarilla")
                        voz2.voz(0)
                        with output_queue.mutex:
                            output_queue.queue.clear()
                      #  voz2.MostrarUI(38.5,0)
            # We need a timeout here to not get stuck when no images are retrieved from the queue
            output_queue.put(img, timeout=33)
        except Full:
            pass  # try again with a newer frame


class App:
    def __init__(self, window, window_title, image_queue: Queue, image_dimensions: tuple):
        self.window = window
        self.window.title(window_title)

        self.image_queue = image_queue
        self.fontStyle = Font(family="Arial", size=18)
        # Create a canvas that can fit the above video source size
        self.imagen=PhotoImage(file="LOGO-EQYS.png")
        self.labelArriba = tkinter.Label(window,bg="black",image=self.imagen).pack()
        self.canvas = tkinter.Canvas(window, width=image_dimensions[0], height=image_dimensions[1])
        self.canvas.pack()
        self.labelAbajo = tkinter.Label(window,text="Por favor, acérquese al tótem.",font=self.fontStyle,pady=20).pack()
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()

        self.window.mainloop()

    def update(self):
        try:
            frame = self.image_queue.get(timeout=0.2)  # Timeout to not block this method forever
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.window.after(self.delay, self.update)
        except Empty:
            pass  # try again next time


def main():
    faceNet,maskNet=cargar_modelo()
    stream, width, height = setup_webcam_stream(0)
    webcam_queue = Queue()
    processed_queue = Queue(maxsize=1000)
    stop_event = Event()
    window_name = "FPS Multi Threading"

    try:
        Thread(target=video_stream_loop, args=[stream, webcam_queue, stop_event]).start()
        Thread(target=processing_loop, args=[webcam_queue, processed_queue, stop_event, faceNet, maskNet]).start()
        App(tkinter.Tk(), window_name, webcam_queue, (width, height))
    finally:
        stop_event.set()

    print(f"Webcam queue: {webcam_queue.qsize()}")
    print(f"Processed queue: {processed_queue.qsize()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()