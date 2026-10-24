import cv2

print("Mencari kamera...")

for index in range(10):
    cap = cv2.VideoCapture(index)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Kamera ditemukan pada index: {index}")

        cap.release()

print("Selesai.")