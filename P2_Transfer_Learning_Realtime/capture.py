"""Langkah 1: ambil citra terkoreksi (undistort) dari kamera eksternal.
Contoh:  python capture.py baut lab-terang --cam 2
SPACE = simpan, Q = keluar. Gunakan '-' (bukan '_') di nama kondisi."""
import argparse, csv, datetime, sys
from pathlib import Path
import cv2
from common import open_camera, Undistorter

p = argparse.ArgumentParser()
p.add_argument("kelas")
p.add_argument("kondisi", help="mis. lab-terang, lab-redup, jendela, bayangan")
p.add_argument("--cam", type=int, default=0)
p.add_argument("--width", type=int, default=640)
p.add_argument("--height", type=int, default=480)
p.add_argument("--root", default="dataset_raw")
p.add_argument("--no-undistort", action="store_true")
a = p.parse_args()
if "_" in a.kelas or "_" in a.kondisi:
    sys.exit("Jangan pakai '_' di kelas/kondisi, pakai '-'.")

und = Undistorter(not a.no_undistort)
cap = open_camera(a.cam, a.width, a.height)
out_dir = Path(a.root) / a.kelas
out_dir.mkdir(parents=True, exist_ok=True)
meta = Path(a.root) / "metadata.csv"
new_meta = not meta.exists()
tanggal = datetime.date.today().strftime("%Y%m%d")
n = len(list(out_dir.glob(f"{a.kelas}_{tanggal}_{a.kondisi}_*.png")))

with open(meta, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new_meta:
        w.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])
    print("SPACE = simpan | Q = keluar")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal membaca frame."); break
        frame = und(frame)
        view = frame.copy()
        cv2.putText(view, f"{a.kelas} | {a.kondisi} | tersimpan: {n}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("capture (undistorted)", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == 32:
            n += 1
            name = f"{a.kelas}_{tanggal}_{a.kondisi}_{n:03d}.png"
            cv2.imwrite(str(out_dir / name), frame)
            w.writerow([name, a.kelas, tanggal, a.kondisi]); f.flush()
            print("simpan", name)
cap.release(); cv2.destroyAllWindows()
