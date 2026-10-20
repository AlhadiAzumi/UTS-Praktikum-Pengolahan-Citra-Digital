import cv2
import numpy as np
from matplotlib_viewer import show_images

gray = cv2.imread("spid.jpg", cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError("Pastikan spid.jpg ada di folder yang sama.")

T, MAKS = 127, 255
jenis = [("binary", cv2.THRESH_BINARY), ("binary_inv", cv2.THRESH_BINARY_INV),
         ("trunc", cv2.THRESH_TRUNC), ("tozero", cv2.THRESH_TOZERO),
         ("tozero_inv", cv2.THRESH_TOZERO_INV)]

cv2.imwrite("hasil_H1_gray.png", gray)
gambar_hasil = [gray]
nama_hasil = ["Asli"]
for nama, tipe in jenis:
    _, hasil = cv2.threshold(gray, T, MAKS, tipe)
    gambar_hasil.append(hasil)
    nama_hasil.append(nama)
    cv2.imwrite(f"hasil_H1_{nama}.png", hasil)

show_images(gambar_hasil, nama_hasil)

# Tugas 1: citra 3 bit (aras 0-7), T = 4
citra = np.array([[3, 0, 1, 5],
                  [7, 6, 0, 4],
                  [2, 7, 0, 6],
                  [1, 3, 5, 5]], dtype=np.uint8)
print("Citra asli:")
print(citra)
for nama, tipe in jenis:
    _, h = cv2.threshold(citra, 4, 7, tipe)
    print(f"\n{nama}:")
    print(h)
