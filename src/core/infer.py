from ultralytics import YOLO
m = YOLO("runs/detect/train/weights/best.pt")
m.predict(source="dataset/images/test", conf=0.25, iou=0.6, save=True)