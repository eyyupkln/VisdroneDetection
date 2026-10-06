from sahi.scripts.coco_evaluation import evaluate

result = evaluate(
    dataset_json_path=r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\Visdrone\VisDrone2019-DET-val\visdrone_val_coco.json",
    result_json_path="sahi_predictions.json",
    type="bbox",
    classwise=True,
    max_detections=500,
    return_dict=True
)

print(result)
