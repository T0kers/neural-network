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
from PIL import Image, ImageTk
import numpy as np

import load_mnist

dim = 28
canvas_size = 600
cell_size = canvas_size // dim

training_inputs = load_mnist.load_data("mnist_data/t10k-images.idx3-ubyte")
img_data = training_inputs[1] / 255
#img_data = np.zeros((dim, dim))

root = tk.Tk()

c = tk.Canvas(root, height=600, width=1000, bg='gray')
c.pack(fill=tk.BOTH, expand=True)

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
        img_id = c.create_image(0, 0, anchor="nw", image=tk_img)
    else:
        c.itemconfig(img_id, image=tk_img)

def create_grid(event=None):
    c.delete('grid_line') # Will only remove the grid_line

    # Creates all vertical lines at intevals of 100
    for i in range(0, dim + 1):
        x = i / dim * canvas_size
        y = i / dim * canvas_size
        c.create_line([(x, 0), (x, canvas_size)], tag='grid_line')
        c.create_line([(0, y), (canvas_size, y)], tag='grid_line')


def paint(event):
    x = event.x
    y = event.y

    # Convert pixel position -> grid index
    j = min(dim - 1, max(0, x // cell_size))
    i = min(dim - 1, max(0, y // cell_size))

    # Set pixel (black ink)
    img_data[i, j] = 1.0

    render()


def lin_paint_fallof(x):
    return np.maximum(0, -0.002 * cell_size * x + 1)

def gauss_paint_fallof(x):
    sigma = 0.5 * cell_size
    return np.exp(-0.45 * (x / sigma)**2) / 2


def paint2(event):
    global img_data

    x = event.x
    y = event.y

    dist = np.arange(dim) * cell_size + cell_size / 2

    dist_map = np.sqrt((dist - x)**2 + ((dist - y)**2)[:, np.newaxis])
    modify = gauss_paint_fallof(dist_map)

    img_data += modify
    img_data = np.minimum(1, img_data)

    render()

def clear(event=None):
    global img_data
    img_data.fill(0)
    render()

c.bind("<B1-Motion>", paint2)
c.bind("<Button-1>", paint2)

# c.bind('<Configure>', create_grid)
# c.bind('<Configure>', create_grid)
root.bind("c", clear)

render()

root.mainloop()
