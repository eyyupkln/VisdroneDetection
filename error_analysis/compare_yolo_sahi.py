from pathlib import Path
import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# PATHS
dataset_dir = Path(r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\Visdrone")

images_dir = (
    dataset_dir
    / "VisDrone2019-DET-val"
    / "images"
)

model_path = (
    r"C:\Users\Monster\source\pycharmProject"
    r"\VisdroneDetection\runs\detect\train-6"
    r"\weights\best.pt"
)

output_dir = Path("error_analysis/results")
output_dir.mkdir(parents=True, exist_ok=True)


#
# MODELS

yolo_model = YOLO(model_path)

sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.25,
    device="cuda:0"
)

# CLASS NAMES

class_names = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]


# IMAGE LIST

image_paths = sorted(images_dir.glob("*.jpg"))

# İlk etapta sadece 20 görüntü
image_paths = image_paths[:20]


# PROCESS

for index, image_path in enumerate(image_paths, start=1):

    print(f"[{index}/{len(image_paths)}] {image_path.name}")

    image = cv2.imread(str(image_path))

    # YOLO

    yolo_result = yolo_model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )[0]

    yolo_image = image.copy()

    for box in yolo_result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{class_names[cls]} {conf:.2f}"

        cv2.rectangle(
            yolo_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            yolo_image,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )


    # SAHI

    sahi_result = get_sliced_prediction(
        str(image_path),
        sahi_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )

    sahi_image = image.copy()

    for prediction in sahi_result.object_prediction_list:

        bbox = prediction.bbox

        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)

        cls = prediction.category.id
        conf = prediction.score.value

        label = f"{class_names[cls]} {conf:.2f}"

        cv2.rectangle(
            sahi_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            sahi_image,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )


    # SIDE BY SIDE

    combined = cv2.hconcat([
        yolo_image,
        sahi_image
    ])

    output_path = (
        output_dir
        / f"{index:03d}_{image_path.name}"
    )

    cv2.imwrite(
        str(output_path),
        combined
    )

print("\nError analysis görüntüleri oluşturuldu.")