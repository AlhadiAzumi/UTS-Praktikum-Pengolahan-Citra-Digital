import cv2
import numpy as np
from konvolusi import muat_gray
from matplotlib_viewer import show_images

gray = muat_gray("spid.jpg")

# Citra biner: bagian gelap (karakter dan langit bagian atas) dibuat putih sebagai objek
_, biner = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

def morfologi(bentuk):
    k = cv2.getStructuringElement(bentuk, (5, 5))
    return {"erosi": cv2.erode(biner, k),
            "dilasi": cv2.dilate(biner, k),
            "opening": cv2.morphologyEx(biner, cv2.MORPH_OPEN, k),
            "closing": cv2.morphologyEx(biner, cv2.MORPH_CLOSE, k)}

hasil = morfologi(cv2.MORPH_CROSS)
cv2.imwrite("hasil_G1_biner.png", biner)
for nama, h in hasil.items():
    cv2.imwrite(f"hasil_G1_{nama}.png", h)

nama_hasil = list(hasil.keys())
gambar_hasil = [biner, *hasil.values()]

print("Jumlah piksel putih (objek):")
print("  biner  :", int((biner > 0).sum()))
for bentuk, nama_bentuk in ((cv2.MORPH_CROSS, "CROSS"), (cv2.MORPH_RECT, "RECT"), (cv2.MORPH_ELLIPSE, "ELLIPSE")):
    h = morfologi(bentuk)
    print(f"  {nama_bentuk:8s}", {n: int((v > 0).sum()) for n, v in h.items()})

# Skeletonizing: erosi berulang, simpan selisih terhadap opening
def skeleton(citra):
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(citra)
    sementara = citra.copy()
    while cv2.countNonZero(sementara) > 0:
        buka = cv2.morphologyEx(sementara, cv2.MORPH_OPEN, k)
        skel = cv2.bitwise_or(skel, cv2.subtract(sementara, buka))
        sementara = cv2.erode(sementara, k)
    return skel

kerangka = skeleton(biner)
cv2.imwrite("hasil_G1_skeleton.png", kerangka)
print("Piksel kerangka (skeleton):", int((kerangka > 0).sum()))

show_images(
    [*gambar_hasil, kerangka],
    ["Biner", *nama_hasil, "Skeleton"],
)
