from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from image_reader import *

m = 0
f = 0
root = Tk()

def openFile():
    filename = askopenfilename()
    display_picture(filename)
    

def display_picture(path):
    global root
    img = Image.open(path)
    img = img.resize((250, 250),Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
    
def gui():
    global root
    root.title("EiT - Maskinlæring, Gruppe 1")

    button = Button(root, text="Open X-ray",pady=-100, width=25, command=openFile)
    button.pack(side = BOTTOM)
    
    rb1 = Radiobutton(root, text = "Male", padx = 20, variable = m, value = 1).pack(anchor=tk.W)
    rb2 = Radiobutton(root, text = "Female", padx = 20, variable = f, value = 2).pack(anchor=tk.W)

    mainloop()

if __name__ == '__main__':
    ARRAY_FROM_PATH("1968.png")
    gui()