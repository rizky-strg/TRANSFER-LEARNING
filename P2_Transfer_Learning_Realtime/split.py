"""Langkah 2: bagi train/val BERDASARKAN KONDISI/SESI (cegah data leakage).
Contoh:  python split.py                      (otomatis ~25% val)
         python split.py --val-kondisi jendela"""
import argparse, csv, random, shutil
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", default="dataset_raw")
p.add_argument("--out", default="dataset")
p.add_argument("--val-kondisi", nargs="*", default=None)
p.add_argument("--val-frac", type=float, default=0.25)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

rows = list(csv.DictReader(open(Path(a.root) / "metadata.csv", encoding="utf-8")))
groups = defaultdict(list)
for r in rows:
    groups[(r["tanggal"], r["kondisi_cahaya"])].append(r)
if len(groups) < 2:
    raise SystemExit("Perlu minimal 2 kondisi/sesi berbeda agar split bebas leakage.")

if a.val_kondisi:
    val_keys = [k for k in groups if k[1] in a.val_kondisi]
else:
    keys = list(groups); random.Random(a.seed).shuffle(keys)
    val_keys, cnt = [], 0
    for k in keys[:-1]:                      # sisakan minimal 1 grup untuk train
        if cnt >= a.val_frac * len(rows): break
        val_keys.append(k); cnt += len(groups[k])

if Path(a.out).exists():
    shutil.rmtree(a.out)
count = defaultdict(int)
for k, items in groups.items():
    split = "val" if k in val_keys else "train"
    for r in items:
        dst = Path(a.out) / split / r["kelas"]; dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(a.root) / r["kelas"] / r["nama_file"], dst / r["nama_file"])
        count[(split, r["kelas"])] += 1
print("Kondisi val:", [k[1] for k in val_keys])
for key in sorted(count): print(key, count[key])
