import tkinter as tk
from tkinter import filedialog

import cv2

from matplotlib_viewer import show_images


root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Pilih satu atau beberapa gambar",
    filetypes=[
        ("File gambar", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
        ("Semua file", "*.*"),
    ],
)
root.destroy()

if not file_paths:
    raise SystemExit("Tidak ada gambar yang dipilih.")

images = []
titles = []
for file_path in file_paths:
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {file_path}")
    images.append(image)
    titles.append(file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])

show_images(images, titles)
