
import cv2
import numpy as np
from konvolusi import convolve
from matplotlib_viewer import show_images

# 1. Membaca gambar dalam bentuk grayscale
img = cv2.imread("spid.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Gambar tidak ditemukan. Pastikan file Citra2.png tersedia.")
    exit()

# 2. Membuat kernel Laplace sesuai modul
laplace = (1.0 / 16) * np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0]
], dtype=np.float32)

# 3. Melakukan konvolusi
hasil = convolve(img.astype(np.float32), laplace)

# 4. Normalisasi agar hasil filter mudah dilihat
hasil_tampil = cv2.normalize(
    hasil, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# 5. Menampilkan gambar asli dan hasil filter dalam satu figure
show_images(
    [img, hasil_tampil],
    ["Gambar Asli", "Hasil Image Sharpening - Laplace"],
)

# 6. Menyimpan hasil
# cv2.imwrite("hasil_sharpening.jpg", hasil_tampil)

# 7. Menampilkan nilai piksel sebelum dan sesudah konvolusi
tinggi, lebar = img.shape
y = tinggi // 2
x = lebar // 2

print("Nilai piksel asli:", int(img[y, x]))
print("Nilai hasil konvolusi:", float(hasil[y, x]))
print("Kernel Laplace:")
print(laplace)
print("Hasil disimpan sebagai hasil_sharpening.jpg")
