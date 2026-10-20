
import cv2
import numpy as np
from matplotlib_viewer import show_images

# Membaca gambar
gambar = cv2.imread("Citra2.png")

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Mendapatkan tinggi dan lebar gambar
    h, w = gambar.shape[:2]

    # Menentukan pergeseran sebesar seperempat ukuran gambar
    quarter_h, quarter_w = h / 4, w / 4

    # Membuat matriks translasi
    T = np.float32([
        [1, 0, quarter_w],
        [0, 1, quarter_h]
    ])

    # Melakukan translasi
    hasil = cv2.warpAffine(gambar, T, (w, h))

    # Menampilkan gambar asli dan hasil translasi dalam satu figure
    show_images([gambar, hasil], ["Gambar Asli", "Hasil Translasi"])

    # Menyimpan hasil
    cv2.imwrite("hasil_b1.jpg", hasil)
    print("Hasil translasi disimpan sebagai hasil_b1.jpg")
