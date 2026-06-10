from ultralytics import YOLO

model = YOLO("runs/VisDrone/yolov8s-PRN/train/weights/best.pt")

model.val(
    data="VisDrone.yaml",
    split="test",
    batch=16,
    imgsz=640,
    project="VisDrone/yolov8s-PRN",
)