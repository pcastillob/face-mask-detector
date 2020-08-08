from pygame import mixer
from tkinter import *
from tkinter.font import Font
from MYPIL import Image, ImageTk

def voz(mask):
    archivo = 'Audios/Masc/Francisco1.ogg' if mask == 1 else 'Audios/Masc/Francisco2.ogg' 
    mixer.init()
    mixer.music.load(archivo)
    mixer.music.play()

def MostrarUI(T,M):
    root.overrideredirect(True)
    imagen=PhotoImage(file=foto)
    #autodestrucción de la ui en 3 segundos
    #poner las coordenadas de posicion en la ventana
    image.pack()
    root.mainloop()

def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


MostrarUI(36,1)