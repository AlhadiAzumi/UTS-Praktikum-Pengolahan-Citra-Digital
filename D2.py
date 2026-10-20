
import cv2
import numpy as np
from konvolusi import convolve
from matplotlib_viewer import show_images

# Membaca gambar dalam bentuk grayscale
img = cv2.imread("spid.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Gambar tidak ditemukan. Periksa nama file Citra3.png.")
    exit()

# Kernel mean filter 3 x 3
kernel_3x3 = (1.0 / 9) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32)

# Mean filter menggunakan fungsi konvolusi dari D1
hasil_3x3 = convolve(img.astype(np.float32), kernel_3x3)
hasil_3x3 = np.clip(hasil_3x3, 0, 255).astype(np.uint8)

# Mean filter 2 x 2 menggunakan OpenCV
hasil_2x2 = cv2.blur(img, (2, 2))

# Menampilkan gambar dalam satu figure
show_images(
    [img, hasil_3x3, hasil_2x2],
    ["Gambar Asli", "Mean Filter 3x3", "Mean Filter 2x2"],
)

# # Menyimpan hasil
# cv2.imwrite("hasil_mean_3x3.jpg", hasil_3x3)
# cv2.imwrite("hasil_mean_2x2.jpg", hasil_2x2)

print("Proses Mean Filter selesai.")
print("Hasil tersimpan di folder praktikum.")
