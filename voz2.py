from pygame import mixer
from tkinter import *
from tkinter.font import Font
from pil import ImageTk, Image

def voz(mascarillaBool):
    if mascarillaBool==1:
        mixer.init()
        mixer.music.load('Audios/Masc/Francisco1.mp3')
        mixer.music.play()
    else:
        mixer.init()
        mixer.music.load('Audios/Masc/Francisco2.mp3')
        mixer.music.play()

def MostrarUI(T,M):
    foto = "ui-pasaste.png" if M ==1 else "ui-denegado.png"
    root = Tk()
    root.overrideredirect(True)
    fontStyle = Font(family="Arial", size=48)
    windowWidth = root.winfo_reqwidth()
    windowHeight = root.winfo_reqheight()
    positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
    positionDown = int(root.winfo_screenheight()/3 - windowHeight/2)
  #  canvas = Canvas(root,height=519,width=309)
    root.image=ImageTk.PhotoImage(Image.open(foto))
    image=Label(root,image=root.image,bg='white')
    texto=Label(root,text=str(T)+"°C",bg="white",bd="0",fg="green",font=fontStyle)
   # canvas.create_image(0,0,anchor=NW,image=image)
    #canvas.pack()
    root.attributes("-transparentcolor", "white")
    #texto.attributes("-transparentcolor", "white")
    root.after(3000,lambda:root.destroy())
    root.geometry("+{}+{}".format(positionRight, positionDown))
    
    image.pack()
    texto.pack(fill=BOTH)
    root.mainloop()

