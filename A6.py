
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_viewer import show_images


def tampilkan_histogram(gambar, judul):
    figure = plt.figure(num=1, figsize=(7, 5), clear=True)
    axis = figure.subplots()
    axis.hist(gambar.ravel(), bins=256, range=(0, 256), color="gray")
    axis.set_title(judul)
    axis.set_xlabel("Intensitas Piksel")
    axis.set_ylabel("Jumlah Piksel")
    axis.set_xlim([0, 256])
    axis.grid(True)
    figure.tight_layout()
    plt.show()


# Membaca gambar
gambar = cv2.imread("spid.jpg")

if gambar is None:
    raise FileNotFoundError("Gambar spid.jpg tidak ditemukan.")
else:
    # Mengubah gambar menjadi grayscale
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Mencari nilai piksel minimum dan maksimum
    nilai_min = int(np.min(abu_abu))
    nilai_max = int(np.max(abu_abu))

    print("Nilai piksel minimum:", nilai_min)
    print("Nilai piksel maksimum:", nilai_max)

    # Melakukan peregangan kontras
    if nilai_max > nilai_min:
        hasil = (
            (abu_abu.astype(float) - nilai_min)
            / (nilai_max - nilai_min) * 255
        )
        hasil = np.clip(hasil, 0, 255).astype(np.uint8)
    else:
        hasil = abu_abu.copy()

    # Menyimpan hasil
    cv2.imwrite("hasil_a6.jpg", hasil)

    print("Hasil disimpan sebagai hasil_a6.jpg")

    pilihan = {
        "1": "Tampilkan gambar grayscale asli",
        "2": "Tampilkan hasil peregangan kontras",
        "3": "Bandingkan gambar asli dan hasil",
        "4": "Tampilkan histogram gambar asli",
        "5": "Tampilkan histogram hasil peregangan",
        "6": "Tampilkan ringkasan gambar dan histogram",
    }

    print("\n=== Menu A6 ===")
    for nomor, keterangan in pilihan.items():
        print(f"{nomor}. {keterangan}")

    while True:
        pilihan_user = input("Pilih opsi (1-6): ").strip()
        if pilihan_user in pilihan:
            break
        print("Pilihan tidak valid. Masukkan angka 1 sampai 6.")

    if pilihan_user == "1":
        show_images([abu_abu], ["Gambar Grayscale Asli"])
    elif pilihan_user == "2":
        show_images([hasil], ["Hasil Peregangan Kontras"])
    elif pilihan_user == "3":
        show_images(
            [abu_abu, hasil],
            ["Gambar Asli", "Hasil Peregangan Kontras"],
        )
    elif pilihan_user == "4":
        tampilkan_histogram(abu_abu, "Histogram Gambar Asli")
    elif pilihan_user == "5":
        tampilkan_histogram(hasil, "Histogram Hasil Peregangan")
    else:
        figure = plt.figure(num=1, figsize=(10, 8), clear=True)
        axes = figure.subplots(2, 2)
        axes[0, 0].imshow(abu_abu, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Gambar Grayscale Asli")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(hasil, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("Hasil Peregangan Kontras")
        axes[0, 1].axis("off")

        for axis, data, judul in (
            (axes[1, 0], abu_abu, "Histogram Gambar Asli"),
            (axes[1, 1], hasil, "Histogram Hasil Peregangan"),
        ):
            axis.hist(data.ravel(), bins=256, range=(0, 256), color="gray")
            axis.set_title(judul)
            axis.set_xlabel("Intensitas Piksel")
            axis.set_ylabel("Jumlah Piksel")
            axis.set_xlim([0, 256])
            axis.grid(True)

        figure.tight_layout()
        plt.show()
