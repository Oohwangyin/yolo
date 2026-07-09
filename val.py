import os
from ultralytics import YOLO

os.environ["OMP_NUM_THREADS"] = "8"

model = YOLO("runs/VisDrone/yolov8s-FAFM-Lite-CENet/train/weights/best.pt")

model.val(
    data="VisDrone.yaml",
    split="val",
    batch=1,
    imgsz=640,
    conf=0.001,
    iou=0.7,
    max_det=300,
    project="VisDrone/yolov8s-FAFM-Lite-CENet",
    exist_ok=True,
)