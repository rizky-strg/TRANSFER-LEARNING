"""Fungsi bersama: kalibrasi, kamera, preprocessing, model.
Preprocessing HARUS identik saat pelatihan dan deployment:
undistort -> resize 224x224 -> RGB -> normalisasi ImageNet."""
import os
import cv2
import numpy as np

CALIB_FILE = "realtime_calibration.npz"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_calibration(path=CALIB_FILE):
    data = np.load(path)
    def pick(names):
        for n in names:
            if n in data.files:
                return data[n]
        raise KeyError(f"Kunci {names} tidak ada di {path}. Isi file: {data.files}")
    K = pick(["camera_matrix", "mtx", "K", "cameraMatrix", "camera_mtx"])
    D = pick(["dist_coeffs", "dist", "D", "distCoeffs", "distortion", "distortion_coefficients"])
    return K, D


def open_camera(index=0, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera index {index} tidak bisa dibuka (cek check_cameras.py).")
    return cap


class Undistorter:
    def __init__(self, enabled=True, path=CALIB_FILE):
        self.enabled = enabled
        self.K = self.D = None
        if enabled:
            self.K, self.D = load_calibration(path)

    def __call__(self, frame):
        return cv2.undistort(frame, self.K, self.D) if self.enabled else frame


def preprocess_bgr(frame_bgr):
    """frame BGR (OpenCV) -> array float32 CHW siap masuk model."""
    img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def build_model(arch, num_classes, pretrained=True):
    import torch.nn as nn
    from torchvision import models
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(arch)
    return m
