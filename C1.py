
import cv2
from matplotlib_viewer import show_images

# Membaca dua gambar dalam bentuk grayscale
img1 = cv2.imread("Citra2.png", 0)
img2 = cv2.imread("Citra1.png", 0)

# Memeriksa apakah gambar berhasil dibaca
if img1 is None or img2 is None:
    print("Gambar tidak ditemukan! Periksa nama dan lokasi file.")

elif img1.shape != img2.shape:
    print("Ukuran kedua gambar harus sama.")

else:
    # 1. Operasi penjumlahan
    add_img = cv2.add(img1, img2)

    # 2. Operasi pengurangan
    subtract = cv2.subtract(img1, img2)

    # 3. Operasi perkalian
    multiply = cv2.multiply(img1, img2, scale=1/255)

    # 4. Operasi pembagian
    divide = cv2.divide(img1, img2, scale=255)

    # Menampilkan gambar asli dan hasil operasi dalam satu figure
    show_images(
        [img1, img2, add_img, subtract, multiply, divide],
        ["Image 1", "Image 2", "Penjumlahan", "Pengurangan", "Perkalian", "Pembagian"],
    )

    # Menyimpan hasil operasi
    cv2.imwrite("hasil_penjumlahan.jpg", add_img)
    cv2.imwrite("hasil_pengurangan.jpg", subtract)
    cv2.imwrite("hasil_perkalian.jpg", multiply)
    cv2.imwrite("hasil_pembagian.jpg", divide)

    print("Semua hasil operasi berhasil disimpan.")
