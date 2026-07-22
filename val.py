import os
from ultralytics import YOLO

os.environ["OMP_NUM_THREADS"] = "8"

model = YOLO("runs/UAVDT/yolov8s/train/weights/best.pt")

model.val(
    data="UAVDT.yaml",
    split="test",
    batch=1,
    imgsz=640,
    conf=0.001,
    iou=0.7,
    max_det=300,
    project="UAVDT/yolov8s",
    exist_ok=False,  # Keep previous results: val -> val-2 -> val-3 -> ...
)