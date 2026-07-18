import os
from ultralytics import YOLO

os.environ["OMP_NUM_THREADS"] = "8"

model = YOLO("runs/HIT-UAV/yolov8s/seed0/train/weights/best.pt")

model.val(
    data="HIT-UAV.yaml",
    split="test",
    batch=1,
    imgsz=640,
    conf=0.001,
    iou=0.7,
    max_det=300,
    project="HIT-UAV/yolov8s/seed0",
    exist_ok=True,
)