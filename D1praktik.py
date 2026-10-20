
import cv2
import numpy as np
from konvolusi import convolve
from matplotlib_viewer import show_images

# Membaca gambar dalam bentuk grayscale
img = cv2.imread("pnl.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Gambar tidak ditemukan. Periksa nama file gambar.jpg.")
    exit()

# Kernel A: semua bobot bernilai 1
kernel_a = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32)

# Kernel B: kernel sesuai modul
kernel_b = np.array([
    [6, 0, -6],
    [6, 1, -6],
    [6, 0, -6]
], dtype=np.float32)

# Melakukan konvolusi
hasil_a = convolve(img.astype(np.float32), kernel_a)
hasil_b = convolve(img.astype(np.float32), kernel_b)

# Menyesuaikan nilai agar hasil mudah ditampilkan
hasil_a = cv2.normalize(
    hasil_a, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

hasil_b = cv2.normalize(
    np.abs(hasil_b), None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Menampilkan gambar dalam satu figure
show_images(
    [img, hasil_a, hasil_b],
    ["Gambar Asli", "Hasil Kernel A", "Hasil Kernel B"],
)

# Menyimpan hasil
# cv2.imwrite("hasil_kernel_a.jpg", hasil_a)
# cv2.imwrite("hasil_kernel_b.jpg", hasil_b)

print("Proses konvolusi selesai.")
print("Hasil tersimpan dalam folder praktikum.")
