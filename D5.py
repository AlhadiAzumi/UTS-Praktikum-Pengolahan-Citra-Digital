import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from konvolusi import muat_gray, tambah_noise

gray = muat_gray("spid.jpg")
noisy = tambah_noise(gray, 0.05)

# Median filter 7 x 7: nilai ke-25 dari 49 tetangga yang telah diurutkan
jendela = sliding_window_view(noisy, (7, 7))               # (h-6, w-6, 7, 7)
median = np.median(jendela.reshape(*jendela.shape[:2], 49), axis=2).astype(np.uint8)
hasil = noisy.copy()                                       # tepi 3 piksel dibiarkan
hasil[3:-3, 3:-3] = median

cv2.imshow("Noise", noisy)
cv2.imshow("Median 7x7", hasil)
cv2.imwrite("hasil_D5_noise.png", noisy)
cv2.imwrite("hasil_D5_median.png", hasil)

# Contoh modul: lingkungan 3x3 dengan piksel derau bernilai 35
n = np.array([10, 10, 10, 11, 35, 10, 9, 12, 10])
print("Urutan  :", np.sort(n), "-> median =", int(np.median(n)))

pembanding = cv2.medianBlur(noisy, 7)
print("Selisih dengan cv2.medianBlur (bagian dalam):",
      int(np.abs(hasil[3:-3, 3:-3].astype(int) - pembanding[3:-3, 3:-3]).max()))
y, x = [p for p in np.argwhere((noisy != gray) & (noisy == 255)) if 3 <= p[0] < gray.shape[0] - 3 and 3 <= p[1] < gray.shape[1] - 3][10]   # contoh piksel salt
print(f"Piksel derau di ({y},{x}): asli {gray[y, x]} -> noise {noisy[y, x]} -> median {hasil[y, x]}")
cv2.waitKey(0)
cv2.destroyAllWindows()
