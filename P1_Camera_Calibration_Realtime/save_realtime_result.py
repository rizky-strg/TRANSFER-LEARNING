import cv2
import numpy as np
from pathlib import Path

CAMERA_INDEX = 2
CALIBRATION_FILE = Path("realtime_calibration.npz")
OUTPUT_DIR = Path("output_realtime")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load hasil kalibrasi
data = np.load(CALIBRATION_FILE)

camera_matrix = data["camera_matrix"]
distortion_coefficients = data["distortion_coefficients"]

# Buka kamera eksternal
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Kamera eksternal tidak dapat dibuka.")

print("Kamera eksternal aktif.")
print("Tekan SPACE untuk mengambil gambar.")
print("Tekan Q untuk keluar.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame kamera.")
        break

    # Proses undistortion
    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        distortion_coefficients
    )

    # Gabungkan original dan undistorted
    comparison = np.hstack((frame, undistorted))

    cv2.putText(
        comparison,
        "ORIGINAL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        comparison,
        "UNDISTORTED",
        (660, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Realtime Calibration Result", comparison)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:  # SPACE
        cv2.imwrite(str(OUTPUT_DIR / "realtime_original.jpg"), frame)
        cv2.imwrite(
            str(OUTPUT_DIR / "realtime_undistorted.jpg"),
            undistorted
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "realtime_comparison.jpg"),
            comparison
        )

        print("JPG berhasil disimpan:")
        print(" - output_realtime/realtime_original.jpg")
        print(" - output_realtime/realtime_undistorted.jpg")
        print(" - output_realtime/realtime_comparison.jpg")

cap.release()
cv2.destroyAllWindows()