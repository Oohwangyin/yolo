from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8s-FAFM-Lite-DSEB.yaml")

model.train(
    data="HIT-UAV.yaml",
    epochs=200,
    batch=16,
    imgsz=640,
    optimizer="SGD",
    pretrained=False,
    cos_lr=True,
    close_mosaic=20,
    patience=30,
    seed=0,
    deterministic=False,
    device=0,
    workers=8,
    project="HIT-UAV/yolov8s-FAFM-Lite-DSEB/seed0",
    exist_ok=False,
)