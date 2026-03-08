from ultralytics import YOLO
m = YOLO("yolov8s.pt")
m.train(data="data.yaml", imgsz=224, epochs=20, batch=1)