
import cv2
import matplotlib.pyplot as plt

# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Menampilkan gambar asli
    gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)

    # Warna kanal OpenCV: Blue, Green, Red
    warna = ("b", "g", "r")

    # Menampilkan gambar dan histogram RGB dalam satu figure
    figure = plt.figure(num=1, figsize=(10, 4), clear=True)
    image_axis, histogram_axis = figure.subplots(1, 2)
    image_axis.imshow(gambar_rgb)
    image_axis.set_title("Gambar Asli")
    image_axis.axis("off")

    for i, col in enumerate(warna):
        histogram = cv2.calcHist(
            [gambar], [i], None, [256], [0, 256]
        )
        histogram_axis.plot(histogram, color=col)

    histogram_axis.set_title("Histogram Citra RGB")
    histogram_axis.set_xlabel("Intensitas Piksel")
    histogram_axis.set_ylabel("Jumlah Piksel")
    histogram_axis.set_xlim([0, 256])
    histogram_axis.grid(True)
    figure.tight_layout()
    plt.show()
