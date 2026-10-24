"""Klasifikasi 2 objek real-time dari kamera eksternal (undistort -> resize -> normalisasi).
python realtime_classify.py --cam 1 --ckpt models/resnet18_partial.pt"""
import argparse, time
import cv2, torch
from common import build_model, open_camera, Undistorter, preprocess_bgr

p = argparse.ArgumentParser()
p.add_argument("--cam", type=int, default=0)
p.add_argument("--ckpt", default="models/resnet18_partial.pt")
p.add_argument("--thr", type=float, default=0.70, help="di bawah ini dianggap tidak yakin")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.ckpt, map_location=dev)
m = build_model(ck["arch"], len(ck["classes"]), False); m.load_state_dict(ck["state"]); m.eval().to(dev)
cap, und = open_camera(a.cam), Undistorter(True)
fps, last = 0.0, time.perf_counter()
with torch.no_grad():
    while True:
        ok, f = cap.read()
        if not ok: break
        f = und(f)
        pr = m(torch.from_numpy(preprocess_bgr(f)).unsqueeze(0).to(dev)).softmax(1)[0]
        c = int(pr.argmax()); conf = float(pr[c])
        label = ck["classes"][c] if conf >= a.thr else "tidak yakin"
        now = time.perf_counter(); fps = 0.9 * fps + 0.1 / (now - last); last = now
        col = (0, 255, 0) if conf >= a.thr else (0, 165, 255)
        cv2.putText(f, f"{label} {conf*100:.0f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(f, f"{fps:.1f} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("realtime classification", f)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release(); cv2.destroyAllWindows()
