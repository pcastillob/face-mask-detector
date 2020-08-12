from pygame import mixer
from tkinter import *
from tkinter.font import Font
from MYPIL import ImageTk, Image


def voz(mask):
    archivo = 'Audios/Masc/Francisco1.ogg' if mask == 1 else 'Audios/Masc/Francisco2.ogg' 
    mixer.init()
    mixer.music.load(archivo)
    mixer.music.play()
    

def MostrarUI(T,M):
    foto = "ui-pasa.png" if M ==1 else "ui-mask.png"
    root = Tk()
    
    root.overrideredirect(True)
    fontStyle = Font(family="Arial", size=48)
#    root.wm_attributes('-alpha', 0)  
    #se calcula el posicionamiento de la ventana emergente, esto podría no funcionar el raspberry, en caso negativo comentar esto y la linea 38
    windowWidth = root.winfo_reqwidth()
    windowHeight = root.winfo_reqheight()
    positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
    positionDown = int(root.winfo_screenheight()/3 - windowHeight/2)

    imagen=PhotoImage(file=foto)
    #Poner una imagen y texto encima con el atributo "compound", los colores se asignan con la funcion from_rgb, que toma una tupla (r,g,b) como parámetro
    #el fondo de la ventana tendrá el mismo color que el marco de la imagen (rgb 155,159,162)
    image=Label(root,image=imagen,text="\n\n\n\n\n\n\n\n"+str(T)+"°C",bg=from_rgb((161,184,199)),compound=CENTER,font=fontStyle,fg=from_rgb((14,190,1)) if M==1 else from_rgb((254,0,0)))

    #autodestrucción de la ui en 3,5 segundos
    root.after(3500,lambda:root.destroy())
    #poner las coordenadas de posicion en la ventana
    root.geometry("+{}+{}".format(positionRight-60, positionDown))

    image.pack()
    root.mainloop()

def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

