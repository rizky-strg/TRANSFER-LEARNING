"""Langkah 4: latensi ResNet-18 vs MobileNetV3-Small + anggaran pipeline nyata.
python latency.py                       (model saja)
python latency.py --cam 1 --ckpt models/resnet18_partial.pt   (pipeline kamera)"""
import argparse, time
import numpy as np, torch
from common import build_model, open_camera, Undistorter, preprocess_bgr

p = argparse.ArgumentParser()
p.add_argument("--runs", type=int, default=100)
p.add_argument("--cam", type=int, default=None)
p.add_argument("--ckpt", default="models/resnet18_partial.pt")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", dev)

def sync():
    if dev == "cuda": torch.cuda.synchronize()

for arch in ["resnet18", "mobilenet_v3_small"]:
    m = build_model(arch, 2, pretrained=False).eval().to(dev)
    x = torch.randn(1, 3, 224, 224, device=dev); ts = []
    with torch.no_grad():
        for i in range(a.runs + 20):
            t = time.perf_counter(); m(x); sync()
            if i >= 20: ts.append((time.perf_counter() - t) * 1000)
    print(f"{arch:20s} inferensi: {np.mean(ts):6.2f} ms (std {np.std(ts):.2f}) -> {1000/np.mean(ts):5.1f} FPS")

if a.cam is not None:
    ck = torch.load(a.ckpt, map_location=dev)
    m = build_model(ck["arch"], len(ck["classes"]), False); m.load_state_dict(ck["state"]); m.eval().to(dev)
    cap, und = open_camera(a.cam), Undistorter(True)
    T = {k: [] for k in ["akuisisi+undistort", "preprocess", "inferensi", "postprocess"]}
    with torch.no_grad():
        for i in range(a.runs + 10):
            t0 = time.perf_counter(); ok, f = cap.read(); f = und(f)
            t1 = time.perf_counter(); x = torch.from_numpy(preprocess_bgr(f)).unsqueeze(0).to(dev)
            t2 = time.perf_counter(); out = m(x); sync()
            t3 = time.perf_counter(); pr = out.softmax(1)[0]; _ = ck["classes"][int(pr.argmax())]
            t4 = time.perf_counter()
            if i >= 10:
                for k, v in zip(T, [t1-t0, t2-t1, t3-t2, t4-t3]): T[k].append(v * 1000)
    tot = sum(np.mean(v) for v in T.values())
    for k, v in T.items(): print(f"{k:20s} {np.mean(v):6.2f} ms")
    print(f"TOTAL pipeline       {tot:6.2f} ms -> {1000/tot:.1f} FPS (target 15 FPS = 67 ms)")
    cap.release()
