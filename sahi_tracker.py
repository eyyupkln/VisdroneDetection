import cv2
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

    print("Video işleniyor...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video bitti.")
            break

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

        # 7. Sonuçları Görüntünün Üzerine Çiz
        for track in tracked_objects:
            x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
            track_id = int(track[4])
            conf = track[5]
            cls = int(track[6])

            # Nesneyi çerçevele
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # ID Numarasını ve Sınıfı Yaz
            label = f"ID: {track_id} | Cls: {cls}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Ekranda Göster
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('SAHI + BoT-SORT Tracking', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()