#!/usr/bin/env python3
"""
tune.py — Optimisation des hyperparamètres pour YOLOv8 Segmentation
Utilise le tuner intégré d'Ultralytics (algorithme génétique par mutations)
pour explorer l'espace des hyperparamètres automatiquement.
Usage:
  python tune.py --data configs/dataset.yaml --iterations 10 --epochs 20 --device 0
Le tuner va :
  1. Entraîner le modèle avec les hyperparamètres par défaut
  2. Muter les hyperparamètres et ré-entraîner
  3. Garder les meilleures combinaisons
  4. Sauvegarder les résultats dans runs/tune/
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
# Espace de recherche des hyperparamètres
# Le tuner d'Ultralytics va explorer cet espace par mutations
SEARCH_SPACE = {
    # Learning rate
    "lr0": (1e-5, 1e-1),           # LR initiale
    "lrf": (0.001, 0.1),           # LR finale (fraction)
    # Optimiseur
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    # Loss weights
    "box": (0.02, 12.0),           # Poids box loss
    "cls": (0.2, 4.0),             # Poids classification loss
    "dfl": (0.5, 3.0),             # Poids distribution focal loss
    # Augmentation
    "hsv_h": (0.0, 0.1),           # Hue
    "hsv_s": (0.0, 0.9),           # Saturation
    "hsv_v": (0.0, 0.9),           # Value
    "degrees": (0.0, 45.0),        # Rotation
    "translate": (0.0, 0.9),       # Translation
    "scale": (0.0, 0.9),           # Scale
    "shear": (0.0, 10.0),          # Shear
    "flipud": (0.0, 1.0),          # Flip up-down
    "fliplr": (0.0, 1.0),          # Flip left-right
    "mosaic": (0.0, 1.0),          # Mosaic augmentation
    "mixup": (0.0, 1.0),           # MixUp augmentation
    "copy_paste": (0.0, 1.0),      # Copy-Paste (segmentation)
}
def tune(args):
    """Lance l'optimisation des hyperparamètres."""
    model_name = args.model or "yolov8m-seg.pt"
    print("=" * 60)
    print("🔍 OPTIMISATION DES HYPERPARAMÈTRES")
    print("=" * 60)
    print(f"  Modèle      : {model_name}")
    print(f"  Dataset     : {args.data}")
    print(f"  Itérations  : {args.iterations}")
    print(f"  Epochs/iter : {args.epochs}")
    print(f"  Device      : {args.device}")
    print(f"  Batch       : {args.batch}")
    print("=" * 60)
    # Charger le modèle
    model = YOLO(model_name)
    # Lancer le tuning
    # Le tuner intégré utilise un algorithme génétique
    result_grid = model.tune(
        data=str(args.data),
        epochs=args.epochs,
        iterations=args.iterations,
        optimizer="AdamW",
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        cache=True,
        amp=True,
        plots=True,
        save=True,
        project=args.project or "runs/tune",
        name=args.name or "seg_tune",
        exist_ok=True,
        verbose=True,
    )
    print("\n" + "=" * 60)
    print("🏁 TUNING TERMINÉ")
    print("=" * 60)
    print(f"📁 Résultats sauvegardés dans : {args.project or 'runs/tune'}")
    print("\n💡 Utilise les meilleurs hyperparamètres trouvés dans best_hyperparameters.yaml")
    print("   pour lancer un entraînement complet avec train.py")
    return result_grid
def main():
    parser = argparse.ArgumentParser(
        description="Optimisation des hyperparamètres YOLOv8 Segmentation"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Chemin vers dataset.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modèle de base (défaut: yolov8m-seg.pt)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Nombre d'itérations de tuning (défaut: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epochs par itération (défaut: 20, réduit pour le temps)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Taille du batch (défaut: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille des images (défaut: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device (défaut: '0')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Nombre de workers data loading (défaut: 8)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Dossier du projet (défaut: runs/tune)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nom de l'expérience (défaut: seg_tune)",
    )
    args = parser.parse_args()
    tune(args)
if __name__ == "__main__":
    main()
