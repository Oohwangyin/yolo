from ultralytics import YOLO

model = YOLO("yolov8s-PRN.yaml")

model.train(
    data="VisDrone.yaml",
    epochs=200,
    batch=16,
    imgsz=640,
    patience=50,
    optimizer="SGD",
    pretrained=False,
    cos_lr=True,
    close_mosaic=20,
    project="VisDrone/yolov8s-PRN",
)

# model = YOLO("yolov8s.yaml")

# model.train(
#     data="VEDAI.yaml",
#     epochs=350,
#     batch=32,
#     patience=50,
#     project="VEDAI/yolo8/yolov8s",
#     optimizer="SGD",
#     pretrained=False,
# )


