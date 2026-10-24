import cv2

CAMERA_INDEX = 2

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka.")
    raise SystemExit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Kamera index :", CAMERA_INDEX)
print("Resolusi     :", f"{width} x {height}")
print("FPS          :", fps)

cap.release()