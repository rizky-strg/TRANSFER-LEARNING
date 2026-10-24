# Tugas 2 (P2) – Transfer Learning 2 Objek dengan Kamera Eksternal

Pipeline: **kamera eksternal → undistort (realtime_calibration.npz) → resize 224×224 → RGB → normalisasi ImageNet → ResNet-18**.
Preprocessing sama persis saat pelatihan dan deployment (lihat `common.py`).

## Setup
Salin semua file ke folder `CameraCalibration` (satu folder dengan `realtime_calibration.npz`), lalu:

    .venv\Scripts\python.exe -m pip install -r requirements_tugas2.txt

Ganti `--cam 1` sesuai hasil `check_cameras.py`. Pilih 2 objek, misal `baut` dan `mur`.

## Langkah
1. **Ambil data** (≥ 50 citra/kelas, variasikan jarak, posisi, orientasi, latar, cahaya). Gunakan minimal 2–3 kondisi berbeda per kelas:

       python capture.py baut lab-terang --cam 1
       python capture.py baut lab-redup  --cam 1
       python capture.py baut jendela    --cam 1
       (ulangi untuk kelas mur)

   SPACE = simpan, Q = keluar. Hasil: `dataset_raw/<kelas>/...png` + `dataset_raw/metadata.csv`.
2. **Bagi train/val per kondisi** (cegah data leakage): `python split.py`
3. **Latih 3 pendekatan:**

       python train.py feature
       python train.py partial
       python train.py scratch
       python report.py        # tabel + results/akurasi_per_epoch.png
4. **Ukur latensi:** `python latency.py` lalu `python latency.py --cam 1 --ckpt models/resnet18_partial.pt`
5. **Demo real-time:** `python realtime_classify.py --cam 1 --ckpt models/resnet18_partial.pt`

## Yang dikumpulkan (link GitHub)
Dokumen desain (maks. 2 hal), `dataset_raw` + `metadata.csv`, tabel 3 mode + grafik, latensi, analisis singkat di README.

## Catatan analisis
- Akurasi val 100% dalam 2 epoch → curigai leakage.
- Bandingkan feature vs partial vs scratch (akurasi, waktu, epoch ≥ 90%).
- Jika gagal di lapangan, cek kondisi cahaya/jarak di metadata.csv.
