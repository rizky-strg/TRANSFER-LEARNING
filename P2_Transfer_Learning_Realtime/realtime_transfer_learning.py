#!/usr/bin/env python3
"""
Realtime Transfer Learning (2 objek) - kamera eksternal index 2.

Alur: kamera -> undistort (realtime_calibration.npz) -> ambil data -> latih 3 mode
(feature / partial / scratch) -> klasifikasi live.

Kontrol:
    1 / 2 / 3 : pilih kelas aktif (botol / kaleng / latar)
    L       : ganti kondisi cahaya (terang/redup/jendela/bayangan)
    SPACE   : ambil 1 foto untuk kelas aktif
    G       : auto-capture ON/OFF (1 foto tiap 0.4 detik)
    T       : latih ResNet-18 & MobileNetV3-Small x 3 mode (6 training) + ukur latensi
    M       : ganti model yang dipakai untuk prediksi
    P       : prediksi live ON/OFF
    S       : simpan screenshot hasil prediksi
    R       : hapus semua data (tekan 2x)
    Q/ESC   : keluar

Semua output ada di folder: output_transfer/
"""
from __future__ import annotations

import csv
import datetime
import json
import random
import shutil
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import (CALIB_FILE, IMG_SIZE, MEAN, STD, Undistorter, build_model,
                    open_camera, preprocess_bgr)

# ============================================================
# CONFIGURATION  (ubah sesuai kebutuhan)
# ============================================================
CAMERA_INDEX = 2
FRAME_W, FRAME_H = 640, 480            # harus sama dengan resolusi kalibrasi
CLASS_NAMES = ("botol", "kaleng", "latar")   # 2 objek + kelas "latar" (bukan objek apa pun)
BACKGROUND_CLASS = "latar"             # jika terprediksi ini -> tidak menampilkan apa-apa
LIGHTING = ["terang", "redup", "jendela", "bayangan"]
TARGET_PER_CLASS = 60
AUTO_INTERVAL = 0.4                    # detik
EPOCHS = 10
BATCH = 16
MODES = ["feature", "partial", "scratch"]
ARCHS = ["resnet18", "mobilenet_v3_small"]   # model yang dibandingkan
CONF_THRESHOLD = 0.85

ROOT = Path("output_transfer")
RAW = ROOT / "dataset_raw"
SPLIT = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
SNAP_DIR = ROOT / "snapshots"
META = RAW / "metadata.csv"

DEV = "cuda" if torch.cuda.is_available() else "cpu"

state = {
    "training": False,
    "status": "",
    "models": {},          # mode -> model (eval)
    "classes": list(CLASS_NAMES),
    "active_model": None,
    "predict": False,
    "best": {},            # mode -> best val acc
}


# ============================================================
# DATA
# ============================================================
def count_images():
    return {c: len(list((RAW / c).glob("*.png"))) if (RAW / c).exists() else 0 for c in CLASS_NAMES}


def save_sample(frame, cls, cond):
    d = RAW / cls
    d.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")
    n = len(list(d.glob(f"{cls}_{today}_{cond}_*.png"))) + 1
    name = f"{cls}_{today}_{cond}_{n:03d}.png"
    cv2.imwrite(str(d / name), frame)            # frame bersih (tanpa teks)
    new = not META.exists()
    with open(META, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])
        w.writerow([name, cls, today, cond])
    return name


def make_split():
    """Bagi train/val berdasarkan kondisi/sesi (cegah data leakage)."""
    rows = list(csv.DictReader(open(META, encoding="utf-8")))
    rows = [r for r in rows if (RAW / r["kelas"] / r["nama_file"]).exists()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tanggal"], r["kondisi_cahaya"])].append(r)
    if len(groups) < 2:
        raise ValueError("Ambil data di minimal 2 kondisi cahaya (tekan L).")
    keys = list(groups)
    chosen = None
    for seed in range(500):
        order = keys[:]
        random.Random(seed).shuffle(order)
        val = []
        cnt = 0
        for k in order[:-1]:
            if cnt >= 0.25 * len(rows):
                break
            val.append(k)
            cnt += len(groups[k])
        val_cls = {r["kelas"] for k in val for r in groups[k]}
        tr_cls = {r["kelas"] for k in keys if k not in val for r in groups[k]}
        if val and val_cls == set(CLASS_NAMES) and tr_cls == set(CLASS_NAMES):
            chosen = val
            break
    if chosen is None:
        raise ValueError("Tiap kelas harus punya foto di >= 2 kondisi cahaya.")
    if SPLIT.exists():
        shutil.rmtree(SPLIT)
    cnt = defaultdict(int)
    for k, items in groups.items():
        sp = "val" if k in chosen else "train"
        for r in items:
            dst = SPLIT / sp / r["kelas"]
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(RAW / r["kelas"] / r["nama_file"], dst / r["nama_file"])
            cnt[(sp, r["kelas"])] += 1
    return [k[1] for k in chosen], dict(cnt)


# ============================================================
# TRAINING (jalan di thread terpisah)
# ============================================================
HEAD = {"resnet18": ("fc.",), "mobilenet_v3_small": ("classifier.",)}
PARTIAL_EXTRA = {"resnet18": ("layer4.",),
                 "mobilenet_v3_small": tuple(f"features.{i}." for i in range(9, 13))}


def trainable_prefixes(arch, mode):
    if mode == "scratch":
        return None
    if mode == "feature":
        return HEAD[arch]
    return HEAD[arch] + PARTIAL_EXTRA[arch]


def train_mode(arch, mode, ds_tr, ds_va, tag=""):
    torch.manual_seed(42)
    classes = ds_tr.classes
    m = build_model(arch, len(classes), pretrained=(mode != "scratch"))
    pref = trainable_prefixes(arch, mode)
    if pref is not None:
        for nm, p in m.named_parameters():
            p.requires_grad = nm.startswith(pref)
    if mode == "scratch":
        groups = [{"params": m.parameters(), "lr": 1e-3}]
    else:
        head = [p for nm, p in m.named_parameters() if nm.startswith(HEAD[arch])]
        groups = [{"params": head, "lr": 1e-3}]
        if mode == "partial":
            body = [p for nm, p in m.named_parameters() if nm.startswith(PARTIAL_EXTRA[arch])]
            groups.append({"params": body, "lr": 1e-4})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    m.to(DEV)
    dl_tr = DataLoader(ds_tr, BATCH, shuffle=True, drop_last=len(ds_tr) > BATCH)
    dl_va = DataLoader(ds_va, BATCH)

    def run(dl, train):
        m.train(train)
        if train and pref is not None:               # BatchNorm lapisan beku tetap eval()
            for nm, mod in m.named_modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm) and not (nm + ".").startswith(pref):
                    mod.eval()
        tot = ok = 0
        loss_sum = 0.0
        with torch.set_grad_enabled(train):
            for x, y in dl:
                x, y = x.to(DEV), y.to(DEV)
                out = m(x)
                loss = loss_fn(out, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                loss_sum += loss.item() * len(y)
                ok += (out.argmax(1) == y).sum().item()
                tot += len(y)
        return loss_sum / tot, ok / tot

    hist = {"arch": arch, "mode": mode, "classes": classes, "train_acc": [], "val_acc": [], "val_loss": []}
    best, first90, t0 = 0.0, None, time.time()
    ckpt = MODEL_DIR / f"{arch}_{mode}.pt"
    for ep in range(1, EPOCHS + 1):
        _, tra = run(dl_tr, True)
        vl, vaa = run(dl_va, False)
        sched.step()
        hist["train_acc"].append(tra)
        hist["val_acc"].append(vaa)
        hist["val_loss"].append(vl)
        if vaa >= 0.9 and first90 is None:
            first90 = ep
        if vaa >= best:
            best = vaa
            torch.save({"state": m.state_dict(), "classes": classes, "arch": arch, "mode": mode}, ckpt)
        state["status"] = f"{tag} {arch}/{mode} ep {ep}/{EPOCHS} train={tra:.2f} val={vaa:.2f}"
        print(state["status"])
    hist.update(best_val_acc=best, train_time_s=time.time() - t0, epoch_ge_90=first90)
    json.dump(hist, open(RESULT_DIR / f"{arch}_{mode}.json", "w"), indent=2)
    return hist


def bench(arch, n_cls):
    m = build_model(arch, n_cls, False).eval().to(DEV)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEV)
    ts = []
    with torch.no_grad():
        for _ in range(10):
            m(x)
        for _ in range(50):
            t = time.perf_counter()
            m(x)
            if DEV == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
    return float(np.mean(ts))


def write_report(results, lat, val_cond, split_cnt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"feature": "tab:blue", "partial": "tab:orange", "scratch": "tab:green"}
    styles = {"resnet18": "-", "mobilenet_v3_small": "--"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, r in results.items():
        x = range(1, len(r["val_acc"]) + 1)
        kw = dict(marker="o", color=colors[r["mode"]], linestyle=styles[r["arch"]], label=key)
        ax[0].plot(x, r["val_acc"], **kw)
        ax[1].plot(x, r["train_acc"], **kw)
    for a, t in zip(ax, ["Akurasi validasi", "Akurasi train"]):
        a.set_title(t); a.set_xlabel("epoch"); a.set_ylim(0, 1.05); a.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "akurasi_per_epoch.png", dpi=150)
    plt.close(fig)

    L = ["# Hasil Transfer Learning: ResNet-18 vs MobileNetV3-Small", "",
         f"Kelas: {', '.join(CLASS_NAMES)}",
         f"Kondisi untuk validasi: {', '.join(val_cond)}",
         f"Jumlah citra: {split_cnt}", "",
         "## Akurasi validasi terbaik", "",
         "| Model | " + " | ".join(MODES) + " |", "| --- | " + " | ".join("---" for _ in MODES) + " |"]
    for arch in ARCHS:
        L.append(f"| {arch} | " + " | ".join(f"{results[f'{arch}-{m}']['best_val_acc']*100:.1f}%" for m in MODES) + " |")
    L += ["", "## Detail", "",
          "| Model | Mode | Akurasi val terbaik | Waktu latih (s) | Epoch akurasi >= 90% |",
          "| --- | --- | --- | --- | --- |"]
    for r in results.values():
        L.append(f"| {r['arch']} | {r['mode']} | {r['best_val_acc']*100:.1f}% | {r['train_time_s']:.0f} | {r['epoch_ge_90'] or '-'} |")
    L += ["", f"## Ukuran dan latensi inferensi model saja ({DEV}, input 224x224, komputer ini)", "",
          "| Model | Parameter (juta) | ms/frame | FPS |", "| --- | --- | --- | --- |"]
    for arch, (ms, par) in lat.items():
        L.append(f"| {arch} | {par:.1f} | {ms:.2f} | {1000/ms:.1f} |")
    L += ["", "![grafik](akurasi_per_epoch.png)"]
    (RESULT_DIR / "hasil_perbandingan.md").write_text("\n".join(L), encoding="utf-8")


def train_all():
    try:
        state["training"] = True
        state["predict"] = False
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        state["status"] = "Membagi data train/val..."
        val_cond, split_cnt = make_split()
        norm = transforms.Normalize(MEAN.tolist(), STD.tolist())
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3), transforms.ToTensor(), norm])
        tf_val = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])
        ds_tr = datasets.ImageFolder(str(SPLIT / "train"), tf_train)
        ds_va = datasets.ImageFolder(str(SPLIT / "val"), tf_val)
        results = {}
        total, k = len(ARCHS) * len(MODES), 0
        for arch in ARCHS:
            for mode in MODES:
                k += 1
                results[f"{arch}-{mode}"] = train_mode(arch, mode, ds_tr, ds_va, tag=f"({k}/{total})")
        state["status"] = "Mengukur latensi..."
        lat = {}
        for a in ARCHS:
            par = sum(p.numel() for p in build_model(a, len(ds_tr.classes), False).parameters()) / 1e6
            lat[a] = (bench(a, len(ds_tr.classes)), par)
        write_report(results, lat, val_cond, split_cnt)
        load_models()
        state["best"] = {k_: r["best_val_acc"] for k_, r in results.items()}
        state["active_model"] = max(state["best"], key=state["best"].get)
        state["predict"] = True
        state["status"] = f"SELESAI. Hasil di {RESULT_DIR}. Model aktif: {state['active_model']}"
        print(state["status"])
    except Exception as e:  # noqa: BLE001
        state["status"] = f"ERROR: {e}"
        print(state["status"])
    finally:
        state["training"] = False


def load_models():
    state["models"].clear()
    for arch in ARCHS:
        for mode in MODES:
            p = MODEL_DIR / f"{arch}_{mode}.pt"
            if p.exists():
                ck = torch.load(p, map_location=DEV)
                m = build_model(ck["arch"], len(ck["classes"]), False)
                m.load_state_dict(ck["state"])
                state["models"][f"{arch}-{mode}"] = m.eval().to(DEV)
                state["classes"] = ck["classes"]
    if state["models"] and state["active_model"] not in state["models"]:
        state["active_model"] = next(iter(state["models"]))


def predict(frame):
    m = state["models"][state["active_model"]]
    x = torch.from_numpy(preprocess_bgr(frame)).unsqueeze(0).to(DEV)
    with torch.no_grad():
        pr = m(x).softmax(1)[0]
    c = int(pr.argmax())
    return state["classes"][c], float(pr[c])


# ============================================================
# DISPLAY
# ============================================================
def put(img, text, org, color=(0, 255, 0), scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    print("=" * 60)
    print("REALTIME TRANSFER LEARNING")
    print("=" * 60)
    print(f"Camera index : {CAMERA_INDEX}\nKelas        : {CLASS_NAMES}\nDevice       : {DEV}\n")
    for d in (RAW, SNAP_DIR, RESULT_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    try:
        und = Undistorter(True, CALIB_FILE)
        cal = np.load(CALIB_FILE)
        if "image_width" in cal.files and (int(cal["image_width"]), int(cal["image_height"])) != (FRAME_W, FRAME_H):
            print(f"PERINGATAN: resolusi kalibrasi {int(cal['image_width'])}x{int(cal['image_height'])} "
                  f"!= {FRAME_W}x{FRAME_H}. Samakan FRAME_W/FRAME_H.")
        print(f"Kalibrasi dimuat dari {CALIB_FILE}")
    except Exception as e:  # noqa: BLE001
        print(f"PERINGATAN: kalibrasi tidak dimuat ({e}). Memakai frame tanpa undistort.")
        und = Undistorter(False)

    load_models()
    if state["models"]:
        state["active_model"] = state["active_model"] or next(iter(state["models"]))
        print("Model tersimpan dimuat:", list(state["models"]))

    cap = open_camera(CAMERA_INDEX, FRAME_W, FRAME_H)
    active, light_i = 0, 0
    auto, last_auto, last_r = False, 0.0, 0.0
    msg, msg_t = "", 0.0
    fps, last = 0.0, time.perf_counter()
    label, conf = "", 0.0

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("Gagal membaca frame dari kamera.")
                break
            frame = und(raw)
            disp = frame.copy()
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            cond = LIGHTING[light_i]
            cls = CLASS_NAMES[active]

            if auto and not state["training"] and time.time() - last_auto > AUTO_INTERVAL:
                save_sample(frame, cls, cond)
                last_auto = time.time()

            if state["predict"] and state["models"] and not state["training"]:
                label, conf = predict(frame)
                if conf >= CONF_THRESHOLD and label != BACKGROUND_CLASS:
                    put(disp, f"{label} {conf*100:.0f}%", (20, 150), (0, 255, 0), 1.0)
                else:
                    label = ""        # tidak ada botol/kaleng -> tidak tampil apa-apa
                put(disp, f"Model: {state['active_model']} | {fps:.1f} FPS", (20, 180), (255, 255, 0))

            cnt = count_images()
            put(disp, f"Kelas aktif [{active+1}]: {cls} | Cahaya: {cond} | Auto: {'ON' if auto else 'OFF'}", (20, 25))
            put(disp, "  ".join(f"{c}: {cnt[c]}/{TARGET_PER_CLASS}" for c in CLASS_NAMES), (20, 55),
                (0, 255, 0) if all(v >= TARGET_PER_CLASS for v in cnt.values()) else (0, 200, 255))
            if state["status"]:
                put(disp, state["status"], (20, 85), (255, 255, 0))
            if msg and time.time() - msg_t < 3:
                put(disp, msg, (20, 115), (255, 255, 255))
            put(disp, "1/2/3=Kelas L=Cahaya SPACE=Foto G=Auto T=Latih M=Model P=Prediksi S=Simpan R=Reset Q=Keluar",
                (10, FRAME_H - 15), (255, 255, 255), 0.4)
            cv2.imshow("Realtime Transfer Learning", disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif ord("1") <= key < ord("1") + len(CLASS_NAMES):
                active = key - ord("1")
            elif key == ord("l"):
                light_i = (light_i + 1) % len(LIGHTING)
            elif key == ord(" "):
                if state["training"]:
                    msg, msg_t = "Sedang training, tunggu selesai.", time.time()
                else:
                    name = save_sample(frame, cls, cond)
                    msg, msg_t = f"Tersimpan: {name}", time.time()
            elif key == ord("g"):
                auto = not auto
            elif key == ord("t"):
                if state["training"]:
                    msg, msg_t = "Training sedang berjalan.", time.time()
                elif min(cnt.values()) < 10:
                    msg, msg_t = "Minimal 10 foto per kelas (target 50).", time.time()
                else:
                    auto = False
                    threading.Thread(target=train_all, daemon=True).start()
            elif key == ord("m"):
                if state["models"]:
                    ms = list(state["models"])
                    state["active_model"] = ms[(ms.index(state["active_model"]) + 1) % len(ms)]
            elif key == ord("p"):
                if state["models"]:
                    state["predict"] = not state["predict"]
                else:
                    msg, msg_t = "Belum ada model. Tekan T untuk melatih.", time.time()
            elif key == ord("s"):
                fn = SNAP_DIR / f"hasil_{label or 'frame'}_{datetime.datetime.now():%H%M%S}.jpg"
                cv2.imwrite(str(fn), disp)
                msg, msg_t = f"Screenshot: {fn.name}", time.time()
                print("Screenshot disimpan:", fn)
            elif key == ord("r"):
                if state["training"]:
                    continue
                if time.time() - last_r < 3:
                    shutil.rmtree(RAW, ignore_errors=True)
                    shutil.rmtree(SPLIT, ignore_errors=True)
                    RAW.mkdir(parents=True, exist_ok=True)
                    msg, msg_t = "Semua data dihapus.", time.time()
                else:
                    last_r = time.time()
                    msg, msg_t = "Tekan R lagi (3 detik) untuk hapus semua data.", time.time()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.\nProgram finished.")


if __name__ == "__main__":
    main()