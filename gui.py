from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from image_reader import *

m = 0
f = 0

def openFile():
    filename = askopenfilename()
    print(filename)
    display_picture(filename)
    

def display_picture(path):
    img = Image.open(path)
    #img = img.resize((250, 250), Image.ANTIALIAS)
    #img123 = ImageTk.PhotoImage(img)
    #panel = Label(Tk(), image=img123)
    #panel.pack()
    
def gui():
    root = Tk()
    root.title("EiT - Maskinlæring, Gruppe 1")

    button = Button(root, text="hei", width=25, command=openFile)
    button.pack()
    
    rb1 = Radiobutton(root, text = "Male", padx = 20, variable = m, value = 1).pack(anchor=tk.W)
    rb2 = Radiobutton(root, text = "Female", padx = 20, variable = f, value = 2).pack(anchor=tk.W)

    mainloop()

if __name__ == '__main__':
    ARRAY_FROM_PATH("1968.png")
    gui()