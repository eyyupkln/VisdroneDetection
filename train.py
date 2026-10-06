from ultralytics import YOLO
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = YOLO("yolo11n.pt")

    results =model.train(
        data= r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\Visdrone\VisDrone.yaml",
        epochs= 20,
        imgsz = 640,
        batch=8,
        device = device,
        workers=4
    )



if __name__ == "__main__":
    main()


