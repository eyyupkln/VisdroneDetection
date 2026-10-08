import cv2
import time
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
# Ultralytics'in yerleşik Tracker (BoT-SORT / ByteTrack) sistemini içe aktarıyoruz
from ultralytics.trackers import BOTSORT
from ultralytics.engine.results import Boxes



def main():
    # 1. Dosya Yolları ve Video Ayarları
    video_path = r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\video\16857839_3840_2160_30fps.mp4"  # Test için VisDrone'dan veya internetten bir drone videosu
    model_path = "runs/detect/train-6/weights/best.pt"

    cap = cv2.VideoCapture(video_path)

    # 2. SAHI Modelini Yükle
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov11',
        model_path=model_path,
        confidence_threshold=0.45,  # Takipte gürültü istemeyiz, eşiği yüksek tut
        device="cuda:0"
    )

    # 3. BoT-SORT Tracker'ı Başlat (SAHI'den bağımsız olarak)

    from types import SimpleNamespace
    tracker_args = SimpleNamespace(
        tracker_type='botsort',
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        gmc_method='sparseOptFlow',
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        fallback_id=True,
        fuse_score=True,
        # ULTRALYTICS'İN İÇ MİMARİSİNİN BEKLEDİĞİ ZORUNLU PARAMETRELER:
        with_reid=False,  # ReID modelimiz şu an yok, bunu açıkça belirtiyoruz (Hatayı çözen satır)
        model="auto",  # Varsayılan model ataması
        device="cuda:0"  # İşlemin yapılacağı donanım
    )

    # Tracker'ı sadece args objesiyle başlatıyoruz
    tracker = BOTSORT(args=tracker_args)

    # Metrik değişkenleri
    frame_idx        = 0
    fps_history      = []
    total_det_count  = 0
    max_simultaneous = 0
    unique_track_ids = set()

    print("Video isleniyor...")

    while cap.isOpened():
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("Video bitti.")
            break
        frame_idx += 1

        # PERFORMANS İÇİN: 4K (3840x2160) görüntüyü 1080p'ye (1920x1080) düşürüyoruz.
        # Bu sayede SAHI çok daha az parça çıkaracak ve inanılmaz derecede hızlanacak.
        frame = cv2.resize(frame, (1920, 1080))

        # 4. Görüntüyü SAHI ile Dilimle ve Tespitleri Al
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1
        )

        # 5. KÖPRÜ: Tensor'u 'Boxes' Objesine Sarma (CPU Versiyonu)
        detections = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            conf = obj.score.value
            cls_id = obj.category.id
            detections.append([bbox[0], bbox[1], bbox[2], bbox[3], conf, cls_id])

        if len(detections) > 0:
            # .to('cuda:0') SİLİNDİ - Veri CPU'da kalmalı ki Tracker Numpy ile işleyebilsin
            det_tensor = torch.tensor(detections).float()
        else:
            # .to('cuda:0') SİLİNDİ
            det_tensor = torch.empty((0, 6))

        # Tensor'u Boxes objesine sarıyoruz
        yolo_boxes = Boxes(det_tensor, frame.shape[:2])

        # 6. Tracker'ı Güncelle
        tracked_objects = tracker.update(yolo_boxes, frame)

        # Metrik güncelle
        n_dets = len(detections)
        n_tracks = len(tracked_objects)
        total_det_count += n_dets
        max_simultaneous = max(max_simultaneous, n_tracks)

        # FPS
        fps = 1.0 / (time.perf_counter() - t_start + 1e-9)
        fps_history.append(fps)

        # 7. Sonuçları Görüntünün Üzerine Çiz
        for track in tracked_objects:
            x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
            track_id = int(track[4])
            conf = track[5]
            cls = int(track[6])
            unique_track_ids.add(track_id)

            # Nesneyi çerçevele
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # ID Numarasını ve Sınıfı Yaz
            label = f"ID: {track_id} | Cls: {cls}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS + istatistik HUD
        avg_fps_so_far = float(np.mean(fps_history))
        hud = [
            f"FPS: {fps:.1f}  (ort: {avg_fps_so_far:.1f})",
            f"Frame: {frame_idx}  Tracks: {n_tracks}  Dets: {n_dets}",
            f"Toplam track: {len(unique_track_ids)}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

        # Ekranda Göster
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('SAHI + BoT-SORT Tracking', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── SAYISAL ÖZET ─────────────────────────────────────────────
    if fps_history:
        avg_fps = float(np.mean(fps_history))
        min_fps = float(np.min(fps_history))
        max_fps = float(np.max(fps_history))
        avg_dets = total_det_count / max(frame_idx, 1)
        print()
        print("=" * 50)
        print("  TRACKING SAYISAL SONUCLARI")
        print("=" * 50)
        print(f"  Islenen Kare         : {frame_idx}")
        print(f"  Ort. FPS             : {avg_fps:.2f}")
        print(f"  Min FPS              : {min_fps:.2f}")
        print(f"  Max FPS              : {max_fps:.2f}")
        print(f"  Ort. Gecikme (ms)    : {1000/avg_fps:.1f}")
        print(f"  Toplam Tespit        : {total_det_count}")
        print(f"  Ort. Tespit/Kare     : {avg_dets:.1f}")
        print(f"  Benzersiz Track ID   : {len(unique_track_ids)}")
        print(f"  Max Ayni Anda Track  : {max_simultaneous}")
        print("=" * 50)


if __name__ == "__main__":
    main()