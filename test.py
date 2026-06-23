import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

# --------------------
# CONFIG
# --------------------
dim = 28
canvas_size = 560  # display size in pixels (must be divisible-ish by dim)

cell_size = canvas_size // dim

# grayscale image buffer (0 = black, 1 = white)
image = np.zeros((dim, dim), dtype=np.float32)

# --------------------
# TK SETUP
# --------------------
root = tk.Tk()
root.title("PIL Pixel Painter")

c = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
c.pack()

img_id = None  # canvas image handle
tk_img = None   # keep reference to avoid garbage collection


# --------------------
# RENDER FUNCTION
# --------------------
def render():
    global tk_img, img_id

    # Convert 0..1 -> 0..255 (invert so drawing is black)
    arr = (1.0 - image) * 255
    arr = arr.astype(np.uint8)

    # Create PIL image (grayscale)
    pil_img = Image.fromarray(arr, mode="L")

    # Resize to make pixels visible
    pil_img = pil_img.resize((canvas_size, canvas_size), Image.NEAREST)

    # Convert to Tk image
    tk_img = ImageTk.PhotoImage(pil_img)

    # Draw on canvas (replace previous image)
    if img_id is None:
        img_id = c.create_image(0, 0, anchor="nw", image=tk_img)
    else:
        c.itemconfig(img_id, image=tk_img)


# --------------------
# DRAWING LOGIC
# --------------------
def paint(event):
    x = event.x
    y = event.y

    # Convert pixel position -> grid index
    j = min(dim - 1, max(0, x // cell_size))
    i = min(dim - 1, max(0, y // cell_size))

    # Set pixel (black ink)
    image[i, j] = 1.0

    render()


# --------------------
# CLEAR FUNCTION (optional)
# --------------------
def clear(event=None):
    global image
    image.fill(0)
    render()


# --------------------
# BINDINGS
# --------------------
c.bind("<B1-Motion>", paint)   # drag to draw
c.bind("<Button-1>", paint)    # click to draw
root.bind("c", clear)          # press 'c' to clear

# initial render
render()

root.mainloop()