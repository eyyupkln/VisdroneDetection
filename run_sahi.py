from pathlib import Path
import json

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


dataset_dir = Path(r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\Visdrone")

images_dir = (
    dataset_dir /
    "VisDrone2019-DET-val" /
    "images"
)

model_path = (
    r"C:\Users\Monster\source\pycharmProject"
    r"\VisdroneDetection\runs\detect\train-5"
    r"\weights\best.pt"
)


detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.001,
    device="cuda:0"
)


image_paths = sorted(images_dir.glob("*.jpg"))


predictions = []


for index, image_path in enumerate(image_paths, start=1):

    print(f"[{index}/{len(image_paths)}] {image_path.name}")

    result = get_sliced_prediction(
        str(image_path),
        detection_model,

        slice_height=640,
        slice_width=640,

        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,

        verbose=0
    )

    # COCO image id
    image_id = index

    coco_predictions = result.to_coco_predictions(
        image_id=image_id
    )

    predictions.extend(coco_predictions)


output_file = "sahi_predictions.json"

with open(output_file, "w") as f:
    json.dump(predictions, f,default=float)


print()
print("SAHI prediction tamamlandı.")
print(f"Toplam prediction: {len(predictions)}")
print(f"Dosya: {output_file}")