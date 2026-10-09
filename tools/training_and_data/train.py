from ultralytics import YOLO
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = YOLO("yolo11s.pt")

    results =model.train(
        data= r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\Visdrone\VisDrone.yaml",
        epochs= 50,
        imgsz = 800,
        batch=8,
        device = device,
        workers=4
    )



if __name__ == "__main__":
    main()


