from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename

from image_reader import *

def openFile():
    filename = askopenfilename()
    print(filename)
    return filename

def gui():
    root = Tk()
    root.title("EiT - Maskinlæring, Gruppe 1")

    img = ImageTk.PhotoImage(Image.open("1968.png"))

    panel = Label(root, image=img)
    panel.pack()

    button = Button(root, text="hei", width=25, command=openFile)
    button.pack()

    mainloop()

if __name__ == '__main__':
    ARRAY_FROM_PATH("1968.png")
    gui()