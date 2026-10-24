# Hasil Transfer Learning: ResNet-18 vs MobileNetV3-Small

Kelas: botol, kaleng, latar
Kondisi untuk validasi: jendela, terang
Jumlah citra: {('val', 'botol'): 69, ('val', 'kaleng'): 66, ('val', 'latar'): 104, ('train', 'botol'): 21, ('train', 'kaleng'): 24, ('train', 'latar'): 12}

## Akurasi validasi terbaik

| Model | feature | partial | scratch |
| --- | --- | --- | --- |
| resnet18 | 73.6% | 77.8% | 60.7% |
| mobilenet_v3_small | 68.2% | 66.5% | 27.6% |

## Detail

| Model | Mode | Akurasi val terbaik | Waktu latih (s) | Epoch akurasi >= 90% |
| --- | --- | --- | --- | --- |
| resnet18 | feature | 73.6% | 293 | - |
| resnet18 | partial | 77.8% | 297 | - |
| resnet18 | scratch | 60.7% | 330 | - |
| mobilenet_v3_small | feature | 68.2% | 104 | - |
| mobilenet_v3_small | partial | 66.5% | 94 | - |
| mobilenet_v3_small | scratch | 27.6% | 114 | - |

## Ukuran dan latensi inferensi model saja (cpu, input 224x224, komputer ini)

| Model | Parameter (juta) | ms/frame | FPS |
| --- | --- | --- | --- |
| resnet18 | 11.2 | 93.14 | 10.7 |
| mobilenet_v3_small | 1.5 | 22.02 | 45.4 |

![grafik](akurasi_per_epoch.png)