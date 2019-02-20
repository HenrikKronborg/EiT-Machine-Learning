from tkinter import *
from PIL import ImageTk, Image

root = Tk()
root.title("EiT - Maskinlæring, Gruppe 1")

img = ImageTk.PhotoImage(Image.open("download.png"))

panel = Label(root, image=img)
panel.pack()

button = Button(root, text="hei", width=25, command=root.destroy)
button.pack()

mainloop()