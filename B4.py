
import cv2
from matplotlib_viewer import show_images

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan! Periksa nama dan lokasi gambar.")
else:
    # Mengambil tinggi dan lebar gambar
    tinggi, lebar = gambar.shape[:2]

    # Menentukan titik awal pemotongan
    baris_awal = int(tinggi * 0.1)
    kolom_awal = int(lebar * 0.1)

    # Menentukan titik akhir pemotongan
    baris_akhir = int(tinggi * 0.5)
    kolom_akhir = int(lebar * 0.5)

    # Melakukan pemotongan gambar
    hasil_crop = gambar[
        baris_awal:baris_akhir,
        kolom_awal:kolom_akhir
    ]

    # Menampilkan gambar asli dan hasil crop dalam satu figure
    show_images([gambar, hasil_crop], ["Gambar Asli", "Hasil Crop"])

    # Menampilkan ukuran gambar
    print("Ukuran asli:", lebar, "x", tinggi)
    print(
        "Ukuran hasil crop:",
        hasil_crop.shape[1], "x", hasil_crop.shape[0]
    )

    # Menyimpan hasil crop
    cv2.imwrite("hasil_crop.jpg", hasil_crop)

    print("Hasil crop disimpan sebagai hasil_crop.jpg")
