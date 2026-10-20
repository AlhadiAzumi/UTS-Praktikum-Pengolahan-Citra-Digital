import cv2
from matplotlib_viewer import show_images

image = cv2.imread("spid.jpg")
if image is None:
    print("Gagal memuat gambar. Pastikan file Citra2.png ada di folder yang sama.")
else:
    show_images([image], ["Show Image GUI"])