
import cv2
import numpy as np
from matplotlib_viewer import show_images

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan! Periksa nama dan lokasi gambar.")
else:
    h, w = gambar.shape[:2]

    # Fungsi untuk memutar gambar
    def rotasi(gambar, derajat):
        h, w = gambar.shape[:2]

        # Membuat matriks rotasi dengan pusat gambar
        matriks = cv2.getRotationMatrix2D(
            (w / 2, h / 2), derajat, 0.7
        )

        # Menghitung ukuran gambar hasil rotasi
        cos = abs(matriks[0, 0])
        sin = abs(matriks[0, 1])

        lebar_baru = int((h * sin) + (w * cos))
        tinggi_baru = int((h * cos) + (w * sin))

        # Menyesuaikan posisi pusat gambar
        matriks[0, 2] += (lebar_baru / 2) - (w / 2)
        matriks[1, 2] += (tinggi_baru / 2) - (h / 2)

        # Melakukan rotasi
        hasil = cv2.warpAffine(
            gambar, matriks, (lebar_baru, tinggi_baru)
        )

        return hasil

    # Mencoba beberapa sudut rotasi
    images = []
    titles = []
    for sudut in [-45, 45, -90, 90, 180]:
        hasil_rotasi = rotasi(gambar, sudut)

        images.append(hasil_rotasi)
        titles.append(f"Rotasi {sudut} derajat")
        cv2.imwrite(f"hasil_rotasi_{sudut}.jpg", hasil_rotasi)

    # Melakukan transpose
    hasil_transpose = cv2.transpose(gambar)

    images.extend([gambar, hasil_transpose])
    titles.extend(["Gambar Asli", "Hasil Transpose"])
    cv2.imwrite("hasil_transpose.jpg", hasil_transpose)

    show_images(images, titles)
    print("Semua hasil berhasil disimpan.")
