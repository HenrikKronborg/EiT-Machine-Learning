from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from image_reader import *

#------------------------Parameters----------------------------------

m = 0
f = 0
root = Tk()

#------------------------FileName Class-------------------------------

class FileName:
    filename = ""
    array = []

    def display_picture(self):
        filename = askopenfilename()
        array = ARRAY_FROM_PATH(filename)
        img = Image.open(filename)
        img = img.resize((250, 250),Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(root, image=img)
        panel.image = img
        panel.grid(row = 0, column = 1, rowspan=3, columnspan = 5)

    def predict(self):
        print("hei")
    
def gui():
    root.title("EiT - Maskinlæring, Gruppe 1")
    
    path = FileName
    button = Button(root, text="Open X-ray", width=20, command=path.display_picture)
    button.grid(row = 4, column = 4)
    button2 = Button(root, text="Predict", width=20, command=path.predict)
    button2.grid(row = 4, column = 5)
    
    rb1 = Radiobutton(root, text = "Male", padx = 20, variable = m, value = 1)
    rb2 = Radiobutton(root, text = "Female", padx = 20, variable = f, value = 2)
    rb1.grid(row = 0,column = 0)
    rb2.grid(row = 1,column = 0)
    
    age_corr_label = Label(root, width = 20, text = "Correct Age [months]")
    age_corr_label.grid(row = 0, column = 6)
    age_pred_label = Label(root, width = 20, text = "Age prediction [months]")
    age_pred_label.grid(row = 1, column = 6)
    age_corr = Entry(root, state="disabled", width = 25)
    age_corr.grid(row=0,column=7)
    age_pred = Entry(root, state="disabled", width = 25)
    age_pred.grid(row=1,column=7)
    
    mainloop()

if __name__ == '__main__':
    gui()

