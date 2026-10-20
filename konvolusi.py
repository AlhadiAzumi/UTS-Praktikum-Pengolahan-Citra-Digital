
import cv2
import numpy as np


def convolve(X, F):
    tinggi, lebar = X.shape
    tinggi_kernel, lebar_kernel = F.shape

    pusat_y = tinggi_kernel // 2
    pusat_x = lebar_kernel // 2

    # Menyiapkan citra hasil
    hasil = np.zeros((tinggi, lebar), dtype=np.float32)

    # Menghitung konvolusi setiap piksel
    for i in range(pusat_y, tinggi - (tinggi_kernel - pusat_y) + 1):
        for j in range(pusat_x, lebar - (lebar_kernel - pusat_x) + 1):
            jumlah = 0

            for k in range(tinggi_kernel):
                for l in range(lebar_kernel):
                    piksel = X[i + k - pusat_y, j + l - pusat_x]
                    bobot = F[k, l]
                    jumlah += bobot * piksel

            hasil[i, j] = jumlah

    return hasil


def konvolusi(X, F):
    return convolve(X, F)


def muat_gray(path):
    gambar = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gambar is None:
        raise FileNotFoundError(f"Gambar tidak dapat dibaca: {path}")
    return gambar


def ke_uint8(gambar):
    return np.clip(gambar, 0, 255).astype(np.uint8)


def tambah_noise(gambar, proporsi):
    if not 0 <= proporsi <= 1:
        raise ValueError("Proporsi noise harus berada di antara 0 dan 1.")

    hasil = gambar.copy()
    jumlah = int(gambar.size * proporsi)
    if jumlah == 0:
        return hasil

    indeks = np.random.default_rng().choice(gambar.size, jumlah, replace=False)
    datar = hasil.reshape(-1)
    jumlah_garam = (jumlah + 1) // 2
    datar[indeks[:jumlah_garam]] = 255
    datar[indeks[jumlah_garam:]] = 0
    return hasil
