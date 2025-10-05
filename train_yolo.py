from ultralytics import YOLO

# load a pre-trained YOLOv8 model (nano for speed)
model = YOLO("yolov8n.pt")

# train on your dataset
model.train(
    data="Valorant Map Analyser.v1i.yolov8/data.yaml",  # path to data.yaml
    epochs=50,
    imgsz=640,
    batch=16
)