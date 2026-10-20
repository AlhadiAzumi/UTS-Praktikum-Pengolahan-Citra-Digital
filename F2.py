import cv2
import numpy as np
from konvolusi import konvolusi, muat_gray, ke_uint8
from matplotlib_viewer import show_images

gray = muat_gray("spid.jpg")

# Langkah 1: reduksi noise (Gaussian 5x5, sigma = 1.4)
gauss = np.array([[0, 1, 2, 1, 0],
                  [1, 3, 5, 3, 1],
                  [2, 5, 9, 5, 2],
                  [1, 3, 5, 3, 1],
                  [0, 1, 2, 1, 0]], dtype=np.float32) / 57
blur = ke_uint8(konvolusi(gray, gauss))

# Langkah 2: gradien dengan operator Sobel
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
gx, gy = konvolusi(blur, sx), konvolusi(blur, sy)
mag = np.sqrt(gx ** 2 + gy ** 2)
mag = mag / mag.max() * 255
theta = np.arctan2(gy, gx)

# Langkah 3: non-maximum suppression
angle = np.degrees(theta)
angle[angle < 0] += 180
H, W = mag.shape
nms = np.zeros_like(mag)
for i in range(1, H - 1):
    for j in range(1, W - 1):
        a = angle[i, j]
        if a < 22.5 or a >= 157.5:
            q, r = mag[i, j + 1], mag[i, j - 1]
        elif a < 67.5:
            q, r = mag[i + 1, j - 1], mag[i - 1, j + 1]
        elif a < 112.5:
            q, r = mag[i + 1, j], mag[i - 1, j]
        else:
            q, r = mag[i - 1, j - 1], mag[i + 1, j + 1]
        if mag[i, j] >= q and mag[i, j] >= r:
            nms[i, j] = mag[i, j]

# Langkah 4: hysteresis thresholding
def hysteresis(citra, weak, strong):
    kuat = citra >= strong
    lemah = (citra >= weak) & ~kuat
    ganda = np.zeros(citra.shape, np.uint8)
    ganda[lemah] = 127
    ganda[kuat] = 255
    _, label = cv2.connectedComponents((kuat | lemah).astype(np.uint8), connectivity=8)
    simpan = np.unique(label[kuat])                     # komponen yang memuat tepi kuat
    akhir = np.isin(label, simpan[simpan > 0]).astype(np.uint8) * 255
    return ganda, akhir

ganda, akhir = hysteresis(nms, weak=20, strong=50)
_, akhir2 = hysteresis(nms, weak=5, strong=15)
_, akhir3 = hysteresis(nms, weak=40, strong=90)
bawaan = cv2.Canny(blur, 80, 200)     # cv2.Canny memakai gradien tanpa normalisasi (skala ~4x)

hasil = {"blur": blur, "gradien": ke_uint8(mag), "nms": ke_uint8(nms), "ambang": ganda,
         "hasil": akhir, "rendah": akhir2, "tinggi": akhir3, "cv2canny": bawaan}
for nama, h in hasil.items():
    cv2.imwrite(f"hasil_F2_{nama}.png", h)

show_images(list(hasil.values()), list(hasil.keys()))

print("Piksel tepi: Canny (20,50) =", int((akhir > 0).sum()),
      "| (5,15) =", int((akhir2 > 0).sum()),
      "| (40,90) =", int((akhir3 > 0).sum()),
      "| cv2.Canny =", int((bawaan > 0).sum()))
