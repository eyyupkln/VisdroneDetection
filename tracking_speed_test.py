
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

video_path = r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\video\16857839_3840_2160_30fps.mp4"
model_path = r"runs/detect/train-6/weights/best.pt"

# CUDA kullanılabiliyor mu kontrol et
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Video açılamadı!")

fps_values = []
frame_count = 0
start_total = time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Önce 720p ile hız testi
    frame = cv2.resize(frame, (1280, 720))

    start = time.perf_counter()

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.25,
        device=0,
        half=True,
        verbose=False
    )[0]

    # Ekran çizimini ölçüme dahil etmiyoruz
    elapsed = time.perf_counter() - start
    fps_values.append(1.0 / max(elapsed, 1e-9))
    frame_count += 1

    if frame_count % 30 == 0:
        print(
            f"Kare: {frame_count} | "
            f"Son FPS: {fps_values[-1]:.2f} | "
            f"Ort. FPS: {np.mean(fps_values):.2f}"
        )

    # Görüntüyü daha küçük pencerede göster
    display = result.plot()
    display = cv2.resize(display, (960, 540))
    cv2.imshow("YOLO + ByteTrack speed test", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

total_time = time.perf_counter() - start_total

print("\n--- SPEED TEST ---")
print("İşlenen kare:", frame_count)
print("Ortalama kare işleme hızı:", round(np.mean(fps_values), 2), "FPS")
print("Toplam geçen süre:", round(total_time, 2), "sn")
print("Video okuma ve gösterim dahil toplam hız:",
      round(frame_count / max(total_time, 1e-9), 2), "FPS")