
import cv2
from matplotlib_viewer import show_images

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Mengubah gambar RGB menjadi grayscale
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Mengatur nilai kontras
    nilai_kontras = 1.5

    # Menyiapkan gambar hasil
    hasil = abu_abu.astype(float) * nilai_kontras

    # Proses clipping
    hasil[hasil > 255] = 255
    hasil[hasil < 0] = 0

    # Mengubah hasil menjadi tipe data gambar
    hasil = hasil.astype("uint8")

    # Menampilkan gambar asli dan hasil dalam satu figure
    show_images(
        [abu_abu, hasil],
        ["Gambar Asli", "Hasil Pengaturan Kontras"],
    )

    # Menyimpan hasil
    cv2.imwrite("hasil_kontras.jpg", hasil)

    print("Nilai kontras:", nilai_kontras)
    print("Hasil disimpan sebagai hasil_kontras.jpg")
