
import cv2
import numpy as np
from konvolusi import convolve
from matplotlib_viewer import show_images

# 1. Membaca gambar dalam bentuk grayscale
img = cv2.imread("spid.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Gambar tidak ditemukan. Pastikan file Citra2.png ada.")
    exit()

# 2. Membuat kernel Gaussian 5x5 sesuai modul
kernel_gaussian = (1.0 / 345) * np.array([
    [1,  5,  7,  5, 1],
    [5, 20, 33, 20, 5],
    [7, 33, 55, 33, 7],
    [5, 20, 33, 20, 5],
    [1,  5,  7,  5, 1]
], dtype=np.float32)

# 3. Melakukan konvolusi
hasil = convolve(img.astype(np.float32), kernel_gaussian)

# 4. Membatasi nilai piksel ke rentang 0-255
hasil = np.clip(hasil, 0, 255).astype(np.uint8)

# 5. Menampilkan hasil dalam satu figure
show_images([img, hasil], ["Gambar Asli", "Hasil Gaussian Filter"])

# # 6. Menyimpan gambar hasil
# cv2.imwrite("hasil_gaussian.jpg", hasil)

# 7. Membandingkan nilai piksel di bagian tengah gambar
tinggi, lebar = img.shape
y = tinggi // 2
x = lebar // 2

print("Nilai piksel sebelum filter:", int(img[y, x]))
print("Nilai piksel setelah filter:", int(hasil[y, x]))
print("Kernel Gaussian:")
print(kernel_gaussian)
print("Hasil disimpan sebagai hasil_gaussian.jpg")
