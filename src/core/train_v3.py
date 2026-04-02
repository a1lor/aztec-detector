import os
from pathlib import Path
from ultralytics import YOLO

ROOT      = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "config" / "data_full.yaml"

CONFIG = dict(
    data=str(DATA_YAML),
    model="yolov8s.pt",

    imgsz=640,
    epochs=75,
    patience=15,
    batch=32,           # plus grand batch possible avec RTX 4090 + 45k images
    workers=8,
    device="cuda",
    cache=True,

    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,

    copy_paste=0.3,
    degrees=5.0,
    shear=2.0,
    scale=0.7,
    perspective=0.0001,
    close_mosaic=15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.1,
    mosaic=1.0,
    mixup=0.1,
    erasing=0.3,
    translate=0.1,

    cls=1.5,
    box=7.5,
    dfl=1.5,
    overlap_mask=False,

    save=True,
    save_period=10,
    project=str(ROOT / "models" / "runs" / "detect"),
    name="train_v3_full",
    exist_ok=True,
    val=True,
    plots=True,
    verbose=True,
)

if __name__ == "__main__":
    model = YOLO(CONFIG.pop("model"))
    results = model.train(**CONFIG)
    print("\n✅ Entraînement YOLO v3 terminé — 304 classes.")
    print(f"   Poids : {ROOT}/models/runs/detect/train_v3_full/weights/best.pt")
