# Source - https://stackoverflow.com/q/70403360
# Posted by 1d10t
# Retrieved 2026-06-23, License - CC BY-SA 4.0

# from tkinter import *

# pixel_dim = 20
# canvas_width = 28 * pixel_dim
# canvas_height = 28 * pixel_dim

# def paint( event ):
#    python_green = "#476042"
#    x1, y1 = ( event.x - 1 ), ( event.y - 1 )
#    x2, y2 = ( event.x + 1 ), ( event.y + 1 )
#    w.create_rectangle()
#    w.create_oval( x1, y1, x2, y2, fill = python_green )

# master = Tk()
# master.title( "Painting using Ovals" )
# w = Canvas(master, 
#            width=canvas_width, 
#            height=canvas_height)
# w.pack(expand = YES, fill = BOTH)
# w.bind( "<B1-Motion>", paint )

# message = Label( master, text = "Press and Drag the mouse to draw" )
# message.pack( side = BOTTOM )
    
# master.mainloop()

# Source - https://stackoverflow.com/a/34011751
# Posted by Steven Summers
# Retrieved 2026-06-23, License - CC BY-SA 3.0

import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk

import numpy as np

import load_mnist
import nn

dim = 28
canvas_size = 600
cell_size = canvas_size // dim

training_inputs = load_mnist.load_data("mnist_data/t10k-images.idx3-ubyte")
training_labels = load_mnist.load_labels("mnist_data/t10k-labels.idx1-ubyte")

image_index = 0
img_data = np.zeros((dim, dim))

def inc_image(event=None):
    global image_index, img_data
    img_data = training_inputs[image_index] / 255
    l_target.config(text=training_labels[image_index])
    image_index += 1
    render()

root = tk.Tk()

font = tkFont.Font(family="Arial", size=25)

l_target = tk.Label(root, height=1, width=10, font=font)
canvas = tk.Canvas(root, height=600, width=600)
l_prediction = tk.Label(root, height=12, width=10, font=font, anchor="w", justify="left", padx=50)

l_target.grid(row=0, column=0)
canvas.grid(row=1, column=0)
l_prediction.grid(row=1, column=1)

img_id = None  # canvas image handle
tk_img = None   # keep reference to avoid garbage collection

def render():
    global tk_img, img_id

    arr = ((1.0 - img_data) * 255).astype(np.uint8)

    pil_img = Image.fromarray(arr, mode="L")

    # Resize to make pixels visible
    pil_img = pil_img.resize((canvas_size, canvas_size), Image.NEAREST)

    tk_img = ImageTk.PhotoImage(pil_img)

    # Draw on canvas (replace previous image)
    if img_id is None:
        img_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)
    else:
        canvas.itemconfig(img_id, image=tk_img)


def lin_paint_fallof(x):
    return np.maximum(0, -0.002 * cell_size * x + 1)

def gauss_paint_fallof(x):
    sigma = 0.5 * cell_size
    return np.exp(-0.45 * (x / sigma)**2) / 2


def paint(event):
    global img_data

    x = event.x
    y = event.y

    dist = np.arange(dim) * cell_size + cell_size / 2

    dist_map = np.sqrt((dist - x)**2 + ((dist - y)**2)[:, np.newaxis])
    modify = gauss_paint_fallof(dist_map)

    img_data += modify
    img_data = np.minimum(1, img_data)

    render()
    predict()

def clear(event=None):
    global img_data
    img_data.fill(0)
    l_target.config(text="")
    render()

network = nn.Sequential([
    nn.Linear(28 * 28, 100),
    nn.Sigmoid(100),
    nn.Linear(100, 10),
    nn.Softmax(10, True),
])

network.load("testnet.npz")

def predict():
    global network, img_data
    prediction = np.stack((np.arange(10), network.forward(img_data.ravel())), axis=1)

    text = ""
    for i, p in np.flip(prediction[prediction[:, 1].argsort()], axis=0):
        text += f"{int(i)}: {int(100 * p)}%\n"
    l_prediction.config(text=text)
    canvas.after(2000, predict)

canvas.bind("<B1-Motion>", paint)
canvas.bind("<Button-1>", paint)

root.bind("c", clear)
root.bind("i", inc_image)

inc_image()
predict()

root.mainloop()
