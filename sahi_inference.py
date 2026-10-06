from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import os

def main ():

    image_path = r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\Visdrone\VisDrone2019-DET-val\images\0000001_02999_d_0000005.jpg"
    model_path =r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\runs\detect\train-5\weights\best.pt"

    if not os.path.exists(image_path):
        print("hata: test fotoğrafı bulunamadı.")
        return

    detection_model= AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0"
    )

    print("Fotoğraflar dilimleniyor ve analiz ediliyor..")

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.5
    )

    export_dir = "sahi_outputs"
    os.makedirs(export_dir, exist_ok=True)

    result.export_visuals(export_dir=export_dir, file_name="sahi_result")

    print(f"İşlem tamamlandı.Sonuç görseli '{export_dir}/sahi_result.png' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
