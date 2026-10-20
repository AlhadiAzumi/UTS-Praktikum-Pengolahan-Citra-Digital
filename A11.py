
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar sebagai grayscale
gambar = cv2.imread("Citra2.png", 0)

if gambar is None:
    print("Gambar tidak ditemukan!")
else:
    # Menghitung histogram citra
    hist, bins = np.histogram(
        gambar.flatten(), 256, [0, 256]
    )

    # Menghitung CDF
    cdf = hist.cumsum()

    # Menormalisasi CDF untuk ditampilkan pada grafik
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Mengabaikan nilai intensitas yang tidak muncul
    cdf_mask = np.ma.masked_equal(cdf, 0)

    # Mengubah CDF menjadi nilai intensitas baru
    cdf_mask = (
        (cdf_mask - cdf_mask.min())
        * 255
        / (cdf_mask.max() - cdf_mask.min())
    )

    # Mengisi nilai yang sebelumnya diabaikan
    tabel = np.ma.filled(cdf_mask, 0).astype("uint8")

    # Membentuk citra hasil equalization
    hasil = tabel[gambar]

    # Menyimpan hasil
    cv2.imwrite("hasil_a11.jpg", hasil)

    # Menampilkan citra, hasil, serta histogram dalam satu figure
    figure = plt.figure(num=1, figsize=(10, 8), clear=True)
    axes = figure.subplots(2, 2)
    axes[0, 0].imshow(gambar, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Citra Grayscale Asli")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(hasil, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Hasil Histogram Equalization")
    axes[0, 1].axis("off")

    axes[1, 0].plot(cdf_normalized, color="b")
    axes[1, 0].hist(
        hasil.flatten(), 256, [0, 256],
        color="r", alpha=0.5
    )
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].set_xlabel("Intensitas Piksel")
    axes[1, 0].set_ylabel("Jumlah Piksel")
    axes[1, 0].set_title("Histogram Equalization")
    axes[1, 0].legend(("CDF", "Histogram Hasil"), loc="upper left")
    axes[1, 0].grid(True)
    axes[1, 1].set_visible(False)

    figure.tight_layout()
    plt.show()

    print("Hasil disimpan sebagai hasil_a11.jpg")
