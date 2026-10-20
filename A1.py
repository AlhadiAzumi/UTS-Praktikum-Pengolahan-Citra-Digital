import cv2
from matplotlib_viewer import show_images

img = cv2.imread("spid.jpg")
if img is None:
    raise FileNotFoundError("Gambar Citra2.png tidak ditemukan.")

show_images([img], ["Gambar"])