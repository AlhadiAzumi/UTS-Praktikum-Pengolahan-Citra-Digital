import cv2
from matplotlib_viewer import show_images

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Mengatur kecerahan
    kecerahan = 50
    hasil = cv2.convertScaleAbs(gambar, alpha=1, beta=kecerahan)

    # Menampilkan gambar dalam satu figure
    show_images([gambar, hasil], ["Gambar Asli", "Hasil Pencerahan"])

    # Menyimpan hasil
    cv2.imwrite("hasil_pencerahan.png", hasil)
