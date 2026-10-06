from pathlib import Path
import json
import cv2 as cv


dataset_dir = Path(r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\Visdrone")
val_dir = dataset_dir / "VisDrone2019-DET-val"

images_dir = val_dir / "images"
labels_dir = val_dir / "labels"

output_file = val_dir / "visdrone_val_coco.json"


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


coco = {
    "images": [],
    "annotations": [],
    "categories": []
}


# Categories
for class_id, class_name in enumerate(class_names):
    coco["categories"].append({
        "id": class_id,
        "name": class_name,
        "supercategory": "object"
    })


annotation_id = 1

image_paths = sorted(images_dir.glob("*.jpg"))

for image_id, image_path in enumerate(image_paths, start=1):

    image = cv.imread(str(image_path))

    if image is None:
        print(f"Okunamadı: {image_path}")
        continue

    height, width = image.shape[:2]

    coco["images"].append({
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height
    })

    label_path = labels_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    with open(label_path, "r") as f:

        for line in f:

            values = line.strip().split()

            if len(values) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, values)

            cls = int(cls)

            # YOLO normalized -> pixel
            box_width = bw * width
            box_height = bh * height

            center_x = xc * width
            center_y = yc * height

            x = center_x - box_width / 2
            y = center_y - box_height / 2

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls ,
                "bbox": [
                    x,
                    y,
                    box_width,
                    box_height
                ],
                "area": box_width * box_height,
                "iscrowd": 0
            })

            annotation_id += 1


with open(output_file, "w") as f:
    json.dump(coco, f)

print("COCO ground truth oluşturuldu:")
print(output_file)
print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")