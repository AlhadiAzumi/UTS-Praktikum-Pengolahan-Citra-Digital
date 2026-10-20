import cv2
import numpy as np
from konvolusi import konvolusi, muat_gray, ke_uint8
from matplotlib_viewer import show_images

gray = muat_gray("spid.jpg")

def gradien(img, kx, ky):
    gx = konvolusi(img, kx.astype(np.float32))
    gy = konvolusi(img, ky.astype(np.float32))
    g = np.sqrt(gx * gx + gy * gy)
    return ke_uint8(g / g.max() * 255)           # normalisasi 0-255

sobel = gradien(gray, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                      np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
prewitt = gradien(gray, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
roberts = gradien(gray, np.array([[1, 0], [0, -1]]),
                        np.array([[0, 1], [-1, 0]]))

show_images(
    [gray, sobel, prewitt, roberts],
    ["Asli", "Sobel", "Prewitt", "Roberts"],
)
cv2.imwrite("hasil_F1_gray.png", gray)
cv2.imwrite("hasil_F1_sobel.png", sobel)
cv2.imwrite("hasil_F1_prewitt.png", prewitt)
cv2.imwrite("hasil_F1_roberts.png", roberts)

for nama, h in (("Sobel", sobel), ("Prewitt", prewitt), ("Roberts", roberts)):
    print(f"{nama:8s} rata-rata = {h.mean():6.2f} | piksel tepi kuat (>128) = {(h > 128).sum()}")
