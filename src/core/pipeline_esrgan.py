import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from ultralytics import YOLO
from RealESRGAN import RealESRGAN
from PIL import Image

det_model_path = "runs/detect/train3/weights/best.pt"
clf_checkpoint_path = "clf_models/resnet18_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = YOLO(det_model_path)

checkpoint = torch.load(clf_checkpoint_path, map_location=device)
clf_model = models.resnet18(weights=None)
num_classes = len(checkpoint["class_to_idx"])
clf_model.fc = nn.Linear(clf_model.fc.in_features, num_classes)
clf_model.load_state_dict(checkpoint["model_state_dict"])
clf_model = clf_model.to(device)
clf_model.eval()

idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

transform_clf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

esr_model = RealESRGAN(device, scale=4)
esr_model.load_weights("weights/RealESRGAN_x4.pth", download=False)

def crop_from_box(orig_img, box):
    x1, y1, x2, y2 = box
    h, w = orig_img.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2))
    y2 = min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return orig_img[y1:y2, x1:x2]

def enhance_crop_bgr(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    sr_img = esr_model.predict(pil_img)
    sr_np = np.array(sr_img)
    return cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

def classify_crop(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform_clf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = clf_model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
    class_name = idx_to_class[int(pred_idx.item())]
    return class_name, float(conf.item())

def run_pipeline_on_folder(folder="test_images"):
    paths = []
    for ext in ("*.bmp", "*.jpg", "*.jpeg", "*.png"):
        paths.extend(Path(folder).glob(ext))
    paths = sorted(paths)
    for img_path in paths:
        img_path = str(img_path)
        print("\nImage:", img_path)
        results = detector.predict(source=img_path, conf=0.25, iou=0.5, save=True, verbose=False)
        r = results[0]
        if r.boxes.cls.numel() == 0:
            print("  Aucun élément détecté par YOLO")
            continue
        orig_img = r.orig_img
        for i, (cls_det, conf_det, box) in enumerate(zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy), start=1):
            det_id = int(cls_det)
            det_name = r.names[det_id]
            det_conf = float(conf_det)
            box_xyxy = box.tolist()
            crop = crop_from_box(orig_img, box_xyxy)
            if crop is None or crop.size == 0:
                print(f"  Élément {i}: crop invalide")
                continue
            enhanced = enhance_crop_bgr(crop)
            clf_name, clf_conf = classify_crop(enhanced)
            print(f"  Élément {i}:")
            print(f"    YOLO: {det_name} ({det_conf*100:.1f}%)")
            print(f"    Real-ESRGAN + ResNet: {clf_name} ({clf_conf*100:.1f}%)")

if __name__ == "__main__":
    run_pipeline_on_folder("test_images")