import os
from pathlib import Path
from ultralytics import YOLO

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "config" / "data.yaml"

# ── Hyperparamètres (optimisés GPU + anti-biais de classe) ────────────────────
CONFIG = dict(
    data=str(DATA_YAML),
    model="yolov8s.pt",

    # Résolution & entraînement
    imgsz=640,
    epochs=200,
    patience=40,
    batch=16,
    workers=4,
    device="cuda",

    # Optimiseur
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,

    # ── Anti-biais de classe ──────────────────────────────────────────────
    # Focal loss : pénalise les prédictions faciles (souvent classe majoritaire)
    fl_gamma=2.0,
    # Poids de la perte de classification augmenté pour forcer l'apprentissage
    # sur les classes minoritaires (teocomitl, tepetla, tlapexohuiloni)
    cls=1.5,            # était 0.5 → ×3 pour mieux discriminer les 4 classes
    # Copy-paste : copie des objets minoritaires dans d'autres images
    copy_paste=0.4,
    # Mixup : mélange d'images pour régulariser et équilibrer
    mixup=0.15,

    # Augmentations standard
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.1,
    degrees=15.0,
    scale=0.5,
    mosaic=1.0,
    erasing=0.3,
    translate=0.1,

    # Pertes
    box=7.5,
    dfl=1.5,

    # Masques désactivés (labels bbox uniquement, pas segmentation)
    overlap_mask=False,

    # Sauvegarde
    save=True,
    save_period=25,
    project=str(ROOT / "models" / "runs" / "detect"),
    name="train_server",
    exist_ok=True,

    # Validation & plots
    val=True,
    plots=True,
    verbose=True,
)

if __name__ == "__main__":
    model = YOLO(CONFIG.pop("model"))
    results = model.train(**CONFIG)
    print("\n✅ Entraînement YOLO terminé.")
    best = ROOT / "models" / "runs" / "detect" / "train_server" / "weights" / "best.pt"
    print(f"   Meilleurs poids : {best}")
