from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8s-FAFM-Lite-DiscrimLoss.yaml")

model.train(
    data="VisDrone.yaml",
    epochs=200,
    batch=16,
    imgsz=640,
    patience=30,
    optimizer="SGD",
    pretrained=False,
    cos_lr=True,
    close_mosaic=20,
    project="VisDrone/yolov8s-FAFM-Lite-DiscrimLoss",
    deterministic=False,
)
