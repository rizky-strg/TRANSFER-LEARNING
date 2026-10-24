# Hasil Transfer Learning (ResNet-18)

Kelas: botol, kaleng
Kondisi untuk validasi: jendela, terang
Jumlah citra: {('val', 'botol'): 40, ('val', 'kaleng'): 37, ('train', 'botol'): 20, ('train', 'kaleng'): 23}

| Mode | Akurasi val terbaik | Waktu latih (s) | Epoch akurasi >= 90% |
| --- | --- | --- | --- |
| feature | 93.5% | 105 | 5 |
| partial | 100.0% | 118 | 2 |
| scratch | 90.9% | 155 | 10 |

## Latensi inferensi model saja (cpu, input 224x224)

| Model | ms/frame | FPS |
| --- | --- | --- |
| resnet18 | 101.71 | 9.8 |
| mobilenet_v3_small | 23.99 | 41.7 |

![grafik](akurasi_per_epoch.png)