
import cv2
from matplotlib_viewer import show_images

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Mengubah gambar menjadi grayscale
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Menentukan nilai threshold
    threshold = 100

    # Membuat citra biner
    biner = abu_abu.copy()
    biner[abu_abu > threshold] = 255
    biner[abu_abu < threshold] = 0

    # Menampilkan gambar dalam satu figure
    show_images(
        [abu_abu, biner],
        ["Gambar Grayscale", "Hasil Citra Biner"],
    )

    # Menyimpan hasil
    cv2.imwrite("hasil_a8.jpg", biner)
    print("Threshold:", threshold)
    print("Hasil disimpan sebagai hasil_a8.jpg")
