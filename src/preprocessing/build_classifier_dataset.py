import os
import glob
from pathlib import Path
import cv2

classes = ["cacahuatl", "teocomitl", "tepetla", "tlapexohuiloni"]

src_img_root = "source_images_bmp"
src_lbl_root = "source_labels_yolo"
dst_root = "clf_dataset/all"

os.makedirs(dst_root, exist_ok=True)
for cls in classes:
    os.makedirs(os.path.join(dst_root, cls), exist_ok=True)

patterns = ["*.bmp", "*.jpg", "*.jpeg", "*.png"]

def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(img_w - 1, int(cx + bw / 2))
    y2 = min(img_h - 1, int(cy + bh / 2))
    return x1, y1, x2, y2

for cls in classes:
    img_dir = os.path.join(src_img_root, cls)
    lbl_dir = os.path.join(src_lbl_root, cls)
    img_files = []
    for p in patterns:
        img_files.extend(glob.glob(os.path.join(img_dir, p)))
    img_files.sort()
    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = Path(img_path).stem
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            continue
        crop_idx = 0
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cls_name = classes[cls_id]
            out_dir = os.path.join(dst_root, cls_name)
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{stem}_c{crop_idx}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, crop)
            crop_idx += 1