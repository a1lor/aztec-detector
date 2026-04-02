#!/usr/bin/env python3
"""
finetune_planches.py — Fine-tune le modèle de base sur le dataset synthétique de planches.

Étapes préalables :
  1. Générer le dataset synthétique :
       python src/core/synthetic_compositing.py --n 2000
  2. Lancer ce script :
       python src/core/finetune_planches.py

Le modèle de base (best_fixed.pt) doit être à la racine du projet.
Résultat : models/runs/detect/finetune_planches/weights/best.pt
"""
from pathlib import Path
import torch
from ultralytics import YOLO

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "0"          # GPU NVIDIA
elif torch.backends.mps.is_available():
    DEVICE = "mps"        # Mac M-series
else:
    DEVICE = "cpu"

print(f"Device utilisé : {DEVICE}")

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
BASE_MODEL = ROOT / "best_fixed.pt"          # modèle original mAP=0.766
DATA_YAML  = ROOT / "data" / "synthetic_planches" / "dataset.yaml"

# ── Config fine-tuning ────────────────────────────────────────────────────────
CONFIG = dict(
    data=str(DATA_YAML),

    imgsz=640,
    epochs=15,            # 15 max — évite le catastrophic forgetting
    patience=5,
    batch=8,
    workers=4,
    device=DEVICE,

    optimizer="AdamW",
    lr0=0.00005,          # très bas — on ajuste, on n'efface pas
    lrf=0.01,
    warmup_epochs=2,
    cos_lr=True,

    freeze=15,            # geler les 15 premières couches (backbone complet)

    # Augmentation légère
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.3,
    fliplr=0.3,
    degrees=5.0,
    scale=0.3,
    mosaic=0.5,
    copy_paste=0.0,
    mixup=0.0,

    cls=1.5,
    box=7.5,
    dfl=1.5,

    save=True,
    save_period=5,
    project=str(ROOT / "models" / "runs" / "detect"),
    name="finetune_planches",
    exist_ok=True,
    val=True,
    plots=True,
    verbose=True,
)

if __name__ == "__main__":
    if not BASE_MODEL.exists():
        print(f"❌ Modèle de base introuvable : {BASE_MODEL}")
        print("   Télécharge best_fixed.pt et place-le à la racine du projet.")
        exit(1)

    if not DATA_YAML.exists():
        print(f"❌ Dataset introuvable : {DATA_YAML}")
        print("   Lance d'abord : python src/core/synthetic_compositing.py --n 2000")
        exit(1)

    print(f"Modèle de base : {BASE_MODEL}")
    print(f"Dataset        : {DATA_YAML}")
    print(f"Epochs         : {CONFIG['epochs']} | LR : {CONFIG['lr0']} | Freeze : {CONFIG['freeze']} couches")

    model   = YOLO(str(BASE_MODEL))
    results = model.train(**CONFIG)

    best = ROOT / "models" / "runs" / "detect" / "finetune_planches" / "weights" / "best.pt"
    print(f"\n✅ Fine-tuning terminé.")
    print(f"   Meilleurs poids : {best}")
    print(f"\nPour déployer : copier best.pt → AztecVision/best_fixed.pt")
