
import cv2
from matplotlib_viewer import show_images

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Mengubah gambar menjadi grayscale
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Membuat citra negatif
    negatif = 255 - abu_abu

    # Menampilkan gambar dalam satu figure
    show_images(
        [abu_abu, negatif],
        ["Gambar Asli Grayscale", "Hasil Citra Negatif"],
    )

    # Menyimpan hasil
    cv2.imwrite("hasil_a7.jpg", negatif)
    print("Hasil citra negatif disimpan sebagai hasil_a7.jpg")
