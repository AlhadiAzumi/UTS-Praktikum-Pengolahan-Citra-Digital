
import cv2
import matplotlib.pyplot as plt

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Mengubah gambar menjadi grayscale
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Menampilkan gambar dan histogram dalam satu figure
    figure = plt.figure(num=1, figsize=(10, 4), clear=True)
    image_axis, histogram_axis = figure.subplots(1, 2)
    image_axis.imshow(abu_abu, cmap="gray", vmin=0, vmax=255)
    image_axis.set_title("Gambar Grayscale")
    image_axis.axis("off")

    histogram_axis.hist(abu_abu.ravel(), 256, [0, 256])
    histogram_axis.set_title("Histogram Citra Grayscale")
    histogram_axis.set_xlabel("Intensitas Piksel")
    histogram_axis.set_ylabel("Jumlah Piksel")
    histogram_axis.set_xlim([0, 256])
    histogram_axis.grid(True)

    figure.tight_layout()
    plt.show()
