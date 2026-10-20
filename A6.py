
import cv2
import numpy as np
from matplotlib_viewer import show_images

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Mengubah gambar menjadi grayscale
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Mencari nilai piksel minimum dan maksimum
    nilai_min = int(np.min(abu_abu))
    nilai_max = int(np.max(abu_abu))

    print("Nilai piksel minimum:", nilai_min)
    print("Nilai piksel maksimum:", nilai_max)

    # Melakukan peregangan kontras
    if nilai_max > nilai_min:
        hasil = (
            (abu_abu.astype(float) - nilai_min)
            / (nilai_max - nilai_min) * 255
        )
        hasil = np.clip(hasil, 0, 255).astype(np.uint8)
    else:
        hasil = abu_abu.copy()

    # Menampilkan gambar dalam satu figure
    show_images(
        [abu_abu, hasil],
        ["Gambar Asli", "Hasil Peregangan Kontras"],
    )

    # Menyimpan hasil
    cv2.imwrite("hasil_a6.jpg", hasil)

    print("Hasil disimpan sebagai hasil_a6.jpg")
