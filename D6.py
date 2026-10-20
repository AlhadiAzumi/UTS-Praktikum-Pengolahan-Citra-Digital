import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from konvolusi import muat_gray, tambah_noise
from matplotlib_viewer import show_images

gray = muat_gray("spid.jpg")
noisy = tambah_noise(gray, 0.05)

jendela = sliding_window_view(noisy, (7, 7))
maks = noisy.copy()
mini = noisy.copy()
maks[3:-3, 3:-3] = jendela.max(axis=(2, 3))      # Max filter 7 x 7
mini[3:-3, 3:-3] = jendela.min(axis=(2, 3))      # Min filter 7 x 7

show_images([noisy, maks, mini], ["Noise", "Max Filter", "Min Filter"])
cv2.imwrite("hasil_D6_max.png", maks)
cv2.imwrite("hasil_D6_min.png", mini)

kernel = np.ones((7, 7), np.uint8)
print("Selisih dengan cv2.dilate :", int(np.abs(maks[3:-3, 3:-3].astype(int) - cv2.dilate(noisy, kernel)[3:-3, 3:-3]).max()))
print("Selisih dengan cv2.erode  :", int(np.abs(mini[3:-3, 3:-3].astype(int) - cv2.erode(noisy, kernel)[3:-3, 3:-3]).max()))
print("Rata-rata intensitas: noise", noisy.mean().round(1), "| max", maks.mean().round(1), "| min", mini.mean().round(1))
