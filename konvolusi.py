
import numpy as np

def convolve(X, F):
    tinggi, lebar = X.shape
    tinggi_kernel, lebar_kernel = F.shape

    H = tinggi_kernel // 2
    W = lebar_kernel // 2

    # Menyiapkan citra hasil
    hasil = np.zeros((tinggi, lebar), dtype=np.float32)

    # Menghitung konvolusi setiap piksel
    for i in range(H, tinggi - H):
        for j in range(W, lebar - W):
            jumlah = 0

            for k in range(-H, H + 1):
                for l in range(-W, W + 1):
                    piksel = X[i + k, j + l]
                    bobot = F[H + k, W + l]
                    jumlah += bobot * piksel

            hasil[i, j] = jumlah

    return hasil
