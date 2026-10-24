"""Tabel hasil 3 mode + grafik akurasi per epoch (untuk README)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = {m: json.load(open(f"results/{m}.json")) for m in ["feature", "partial", "scratch"] if Path(f"results/{m}.json").exists()}
print("| Mode | Akurasi val terbaik | Waktu latih (s) | Epoch akurasi >= 90% |\n| --- | --- | --- | --- |")
for m, r in res.items():
    print(f"| {m} | {r['best_val_acc']*100:.1f}% | {r['train_time_s']:.0f} | {r['epoch_ge_90'] or '-'} |")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m, r in res.items():
    x = range(1, len(r["val_acc"]) + 1)
    ax[0].plot(x, r["val_acc"], marker="o", label=m); ax[1].plot(x, r["train_acc"], marker="o", label=m)
for axx, t in zip(ax, ["Akurasi validasi", "Akurasi train"]):
    axx.set_title(t); axx.set_xlabel("epoch"); axx.set_ylim(0, 1.05); axx.grid(alpha=.3); axx.legend()
plt.tight_layout(); plt.savefig("results/akurasi_per_epoch.png", dpi=150)
print("Grafik: results/akurasi_per_epoch.png")
