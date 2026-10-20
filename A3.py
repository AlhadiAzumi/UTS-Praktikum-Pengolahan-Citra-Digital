import cv2
from matplotlib_viewer import show_images

# Membaca gambar berwarna
gambar = cv2.imread("spid.jpg")

# Mengubah gambar menjadi grayscale
abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

# Menampilkan gambar asli dan hasil grayscale dalam satu figure
show_images([gambar, abu_abu], ["Gambar Asli", "Gambar Grayscale"])

# Menyimpan hasil grayscale
cv2.imwrite("hasil_grayscale.jpg", abu_abu)