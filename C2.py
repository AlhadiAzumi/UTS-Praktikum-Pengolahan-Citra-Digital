
import cv2
from matplotlib_viewer import show_images

# Membaca dua gambar berwarna
img1 = cv2.imread("spid.jpg")
img2 = cv2.imread("pnl.jpg")

# Memeriksa apakah gambar berhasil dibaca
if img1 is None or img2 is None:
    print("Gambar tidak ditemukan. Periksa nama dan lokasi gambar.")
    exit()

# Menyamakan ukuran gambar jika berbeda
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Operasi Boolean
op_and = cv2.bitwise_and(img1, img2)
op_or = cv2.bitwise_or(img1, img2)
op_xor = cv2.bitwise_xor(img1, img2)

# Menampilkan gambar asli dan hasil operasi dalam satu figure
show_images(
    [img1, img2, op_and, op_or, op_xor],
    ["Gambar 1", "Gambar 2", "Operasi AND", "Operasi OR", "Operasi XOR"],
)

# Menyimpan hasil operasi
# cv2.imwrite("hasil_and.jpg", op_and)
# cv2.imwrite("hasil_or.jpg", op_or)
# cv2.imwrite("hasil_xor.jpg", op_xor)

print("Operasi Boolean selesai.")
print("Hasil disimpan dalam folder praktikum.")
