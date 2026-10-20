
import cv2
from matplotlib_viewer import show_images

# Membaca gambar asli
gambar = cv2.imread("Citra2.png")

if gambar is None:
    print("Gambar tidak ditemukan! Periksa nama dan lokasi gambar.")
else:
    # Mengambil ukuran gambar asli
    tinggi, lebar = gambar.shape[:2]

    print("Ukuran gambar asli:", lebar, "x", tinggi)

    # 1. Zoom In: memperbesar 2 kali
    zoom_in = cv2.resize(
        gambar,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # 2. Zoom Out: memperkecil menjadi setengah
    zoom_out = cv2.resize(
        gambar,
        None,
        fx=0.50,
        fy=0.50
    )

    # 3. Mengubah ukuran menjadi 900 x 400 piksel
    resize_ukuran = cv2.resize(
        gambar,
        (900, 400),
        interpolation=cv2.INTER_AREA
    )

    # Menampilkan gambar dalam satu figure
    show_images(
        [gambar, zoom_in, zoom_out, resize_ukuran],
        ["Gambar Asli", "Zoom In 2x", "Zoom Out 0.5x", "Resize 900 x 400"],
    )

    # Menampilkan ukuran hasil
    print("Ukuran Zoom In:", zoom_in.shape[1], "x", zoom_in.shape[0])
    print("Ukuran Zoom Out:", zoom_out.shape[1], "x", zoom_out.shape[0])
    print("Ukuran Resize:", resize_ukuran.shape[1], "x", resize_ukuran.shape[0])

    # Menyimpan hasil
    cv2.imwrite("hasil_zoom_in.jpg", zoom_in)
    cv2.imwrite("hasil_zoom_out.jpg", zoom_out)
    cv2.imwrite("hasil_resize_900x400.jpg", resize_ukuran)

    print("Semua hasil berhasil disimpan.")
