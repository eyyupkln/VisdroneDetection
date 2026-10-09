import cv2
import time
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.trackers import BOTSORT
from ultralytics.engine.results import Boxes


# ── AYARLAR ──────────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\video\16857839_3840_2160_30fps.mp4"
PT_PATH    = "runs/detect/train-6/weights/best.pt"
ONNX_PATH  = "runs/detect/train-6/weights/best.onnx"

# ── PERFORMANS AYARLARI ───────────────────────────────────────────────────────
# Her kacinci karede bir SAHI tespiti yapilsin?
# 1 = her kare (en dogruu, en yavas) | 3 = iyi denge | 5 = en hizli
DETECT_EVERY_N = 3

# SAHI'ye gonderilecek cozunurluk (4K -> daha kucuk)
# 1280x720 -> ~4 dilim  |  960x540 -> ~2 dilim  |  1920x1080 -> ~9 dilim
INFER_W, INFER_H = 1280, 720

# Ekranda gosterilecek boyut
DISPLAY_W, DISPLAY_H = 1280, 720

# ReID: CPU'da calisir, yogun sahnelerde FPS'i dusurebilir.
# Hiz oncelikli ise False, dogruluk oncelikli ise True yapın.
USE_REID   = False
REID_MODEL = "yolo26n-reid.onnx"

CONF_THRESH = 0.25
DEVICE      = "cuda:0"

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car",
    "van", "truck", "tricycle", "awning-tricycle",
    "bus", "motor"
]

# Sinif renkleri (BGR) - her sinif icin ayri renk
CLASS_COLORS = [
    (0, 255, 180),    # pedestrian  - neon yesil
    (0, 200, 255),    # people      - acik mavi
    (255, 100, 0),    # bicycle     - turuncu
    (0, 120, 255),    # car         - mavi
    (180, 0, 255),    # van         - mor
    (0, 0, 255),      # truck       - kirmizi
    (255, 255, 0),    # tricycle    - sari
    (0, 255, 255),    # awning-tri  - cyan
    (255, 0, 180),    # bus         - pembe
    (100, 255, 100),  # motor       - acik yesil
]
# ─────────────────────────────────────────────────────────────────────────────


def choose_model():
    """
    En iyi mevcut model formatini sec:
    1. ONNX (TensorRT EP ile otomatik hizlanir, ~%40-60 daha hizli)
    2. PyTorch .pt (fallback)
    """
    onnx = Path(ONNX_PATH)
    pt   = Path(PT_PATH)

    if onnx.exists():
        import os
        # [HACK] ONNX Runtime'in GPU DLL'lerini (cuDNN, cuBLAS) bulabilmesi icin
        # PyTorch'un icindeki gizli lib klasorunu gecici olarak sistem PATH'ine ekliyoruz.
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        os.environ['PATH'] = torch_lib + ';' + os.environ.get('PATH', '')

        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            print("[INFO] ONNX + TensorRT EP: En yuksek hiz modu aktif!")
        elif "CUDAExecutionProvider" in providers:
            print("[INFO] ONNX + CUDA EP: GPU hizlandirmali cikarsim (cuDNN hack ile)")
        else:
            print("[INFO] ONNX + CPU EP: (GPU EP bulunamadi)")
        print(f"[INFO] Model: {onnx}")
        return str(onnx), "yolov11"
    else:
        print(f"[INFO] ONNX bulunamadi, PyTorch kullaniliyor: {pt}")
        print("[IPUCU] ONNX export icin:")
        print("  from ultralytics import YOLO")
        print(f"  YOLO('{PT_PATH}').export(format='onnx', device=0)")
        return str(pt), "yolov11"


def make_tracker():
    """ReID destekli BoT-SORT tracker olustur."""
    reid_active = USE_REID and Path(REID_MODEL).exists()
    if USE_REID and not Path(REID_MODEL).exists():
        print(f"[UYARI] ReID modeli bulunamadi: {REID_MODEL}")
        print("[UYARI] ReID devre disi. Indirmek icin bir kez calistirin:")
        print("  from ultralytics.trackers.utils.reid import build_encoder")
        print(f"  build_encoder(True, '{REID_MODEL}')")

    args = SimpleNamespace(
        tracker_type="botsort",
        # Tespit / takip esikleri (optimizasyon ile belirlenmis)
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.4,
        track_buffer=60,
        match_thresh=0.9,
        # Kamera hareketi telafisi (GMC)
        gmc_method="sparseOptFlow",
        # ReID gorunum eslestirmesi
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=reid_active,
        model=REID_MODEL if reid_active else "auto",
        fallback_id=True,
        fuse_score=True,
        device=DEVICE,
    )

    tracker = BOTSORT(args=args)

    if reid_active:
        print(f"[INFO] ReID AKTIF: {REID_MODEL}")
        print("[INFO] Nesneler kisa sure kaybolsa bile gorunum ile yeniden tanimlanacak")
        print("[INFO] ID switch (kimlik kaymasi) minimize edildi")
    else:
        print("[INFO] ReID PASIF: Yalnizca IoU + GMC ile takip")

    return tracker


def draw_track(frame, x1, y1, x2, y2, track_id, cls_id, conf):
    """Renk kodlu bbox + ID + sinif etiketi ciz."""
    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Kose kucuk dolgulari ile kutu
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Sol ve sag kenar vurgu
    cv2.line(frame, (x1, y1), (x1 + 10, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + 10), color, 3)
    cv2.line(frame, (x2, y2), (x2 - 10, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - 10), color, 3)

    # Etiket arkaplan + metin
    cls_name  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
    label     = f"#{track_id} {cls_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    label_y1  = max(y1 - th - 6, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 4, label_y1 + th + 6), color, -1)
    cv2.putText(frame, label, (x1 + 2, label_y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, fps, frame_idx, n_tracks, n_dets, unique_ids, mode_str):
    """Sol ust kose bilgi paneli."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 108), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    color_ok  = (0, 255, 180)
    color_dim = (120, 200, 150)
    lines = [
        (f"Mode : {mode_str}",                         color_ok),
        (f"FPS  : {fps:5.1f}   Frame : {frame_idx}",  color_dim),
        (f"Tracks: {n_tracks:3d}   Dets  : {n_dets}", color_dim),
        (f"Unique IDs tracked: {len(unique_ids)}",     color_dim),
    ]
    for i, (text, col) in enumerate(lines):
        cv2.putText(frame, text, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)


def main():
    print("=" * 55)
    print("  SAHI + BoT-SORT Tracker")
    print("  [ONNX/TRT Hizlandirma + ReID Entegrasyonu]")
    print("=" * 55)

    # 1. Model sec ve yukle
    model_path, model_type = choose_model()
    is_onnx = model_path.endswith(".onnx")

    print("\n[INFO] SAHI detection modeli yukleniyor (FP16)...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=CONF_THRESH,
        device=DEVICE,
        load_at_init=True,
    )
    # FP16 (half precision) optimizasyonu yalnizca PyTorch (.pt) modellerinde gecerlidir
    if model_path.endswith('.pt') and hasattr(detection_model, 'model') and hasattr(detection_model.model, 'half'):
        detection_model.model.half()
        print("[INFO] PyTorch Modeli FP16 (half precision) moduna alindi")

    # 2. Tracker olustur (ReID dahil)
    print()
    tracker = make_tracker()

    mode_str = ("ONNX+TRT" if is_onnx else "PyTorch") + (" +ReID" if USE_REID else "")
    print(f"\n[INFO] Aktif mod: {mode_str}")

    # 3. Video ac
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[HATA] Video acilamadi: {VIDEO_PATH}")
        return

    frame_idx        = 0
    fps_history      = []
    total_det_count  = 0
    max_simultaneous = 0
    unique_track_ids = set()
    cached_det_tensor = torch.empty((0, 6))  # Son tespitler cache'lenir

    print("\n[INFO] Video isleniyor... Cikmak icin Q'ya basin.")
    print(f"[INFO] Tespit her {DETECT_EVERY_N} karede bir yapiliyor, arasi Kalman tahmini.\n")

    while cap.isOpened():
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("\n[INFO] Video tamamlandi.")
            break
        frame_idx += 1

        # 4K -> cikarsim cozunurlugu
        infer_frame = cv2.resize(frame, (INFER_W, INFER_H))

        # ── FRAME-SKIP: Her N karede bir SAHI calistir ────────────────────
        if frame_idx % DETECT_EVERY_N == 1:
            result = get_sliced_prediction(
                infer_frame, detection_model,
                slice_height=640, slice_width=640,
                overlap_height_ratio=0.1, overlap_width_ratio=0.1,
            )
            detections = [
                [*obj.bbox.to_xyxy(), obj.score.value, obj.category.id]
                for obj in result.object_prediction_list
            ]
            cached_det_tensor = torch.tensor(detections).float() if detections else torch.empty((0, 6))
            total_det_count += len(detections)
        # Aradaki karelerde onceki tespitler kullanilir (Kalman tahmin eder)
        # ──────────────────────────────────────────────────────────────────

        det_tensor   = cached_det_tensor
        yolo_boxes   = Boxes(det_tensor, infer_frame.shape[:2])
        tracked_objs = tracker.update(yolo_boxes, infer_frame)

        # Metrikler
        n_dets   = len(det_tensor)
        n_tracks = len(tracked_objs)
        max_simultaneous = max(max_simultaneous, n_tracks)
        fps = 1.0 / (time.perf_counter() - t_start + 1e-9)
        fps_history.append(fps)

        # Ciz: her takip edilen nesne (infer_frame boyutundaki koordinatlar)
        draw_frame = infer_frame.copy()
        for track in tracked_objs:
            x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
            track_id = int(track[4])
            conf     = float(track[5])
            cls_id   = int(track[6])
            unique_track_ids.add(track_id)
            draw_track(draw_frame, x1, y1, x2, y2, track_id, cls_id, conf)

        is_det_frame = (frame_idx % DETECT_EVERY_N == 1)
        mode_full = mode_str + (" [DET]" if is_det_frame else " [TRK]")
        draw_hud(draw_frame, fps, frame_idx, n_tracks, n_dets, unique_track_ids, mode_full)

        display = cv2.resize(draw_frame, (DISPLAY_W, DISPLAY_H))
        cv2.imshow(f"SAHI + BoT-SORT [{mode_str}]", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Terminal ozeti
    if fps_history:
        avg_fps  = float(np.mean(fps_history))
        avg_dets = total_det_count / max(frame_idx, 1)
        print()
        print("=" * 55)
        print(f"  TRACKING SONUCLARI [{mode_str}]")
        print("=" * 55)
        print(f"  Mod                  : {mode_str}")
        print(f"  Islenen Kare         : {frame_idx}")
        print(f"  Ortalama FPS         : {avg_fps:.2f}")
        print(f"  Min / Max FPS        : {min(fps_history):.2f} / {max(fps_history):.2f}")
        print(f"  Ort. Gecikme (ms)    : {1000/avg_fps:.1f}")
        print(f"  Toplam Tespit        : {total_det_count}")
        print(f"  Ort. Tespit/Kare     : {avg_dets:.1f}")
        print(f"  Benzersiz Track ID   : {len(unique_track_ids)}")
        print(f"  Max Esanl. Track     : {max_simultaneous}")
        print("=" * 55)


if __name__ == "__main__":
    main()
