"""
finetune_planches_v3.py
========================
Fine-tuning YOLOv8 pour la détection de glyphes aztèques sur planches entières.

Améliorations vs v1 (original) :
  - imgsz=1280 (obligatoire — glyphes ~35px sur planches réelles)
  - freeze=8 au lieu de 15 (backbone moins bloqué)
  - Augmentations calibrées pour manuscrits aztèques
  - perspective=0.001 (support souple, distorsion légère)
  - Mosaic activé (critique pour apprendre le contexte dense)
  - Sauvegarde tous les 5 epochs (sécurité Colab)
  - Rapport de fin automatique

Usage :
    python finetune_planches_v3.py --data data/synthetic_v3/dataset.yaml \
                                   --model best_fixed.pt \
                                   --epochs 20
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/synthetic_v3/dataset.yaml",
                   help="Chemin vers dataset.yaml")
    p.add_argument("--model",   default="best_fixed.pt",
                   help="Modèle de base (best_fixed.pt)")
    p.add_argument("--epochs",  type=int, default=20)
    p.add_argument("--imgsz",   type=int, default=1280,
                   help="Taille d'image (1280 recommandé pour planches)")
    p.add_argument("--batch",   type=int, default=4,
                   help="Batch size (réduire si VRAM insuffisante)")
    p.add_argument("--freeze",  type=int, default=8,
                   help="Nombre de couches gelées (8=backbone partiel)")
    p.add_argument("--project", default="runs/finetune")
    p.add_argument("--name",    default="planches_v3")
    return p.parse_args()


def check_model(model_path: str) -> None:
    if not Path(model_path).exists():
        print(f"ERREUR : modèle introuvable : {model_path}")
        print("Assurez-vous que best_fixed.pt est dans le répertoire courant.")
        sys.exit(1)


def check_dataset(data_yaml: str) -> None:
    if not Path(data_yaml).exists():
        print(f"ERREUR : dataset.yaml introuvable : {data_yaml}")
        print("Lancez d'abord : python synthetic_compositing_v3.py")
        sys.exit(1)


def train(args) -> None:
    import torch
    from ultralytics import YOLO

    check_model(args.model)
    check_dataset(args.data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")
    if device == "cpu":
        print("AVERTISSEMENT : CPU détecté — ~2h d'entraînement.")
        print("Sur Colab : Runtime → Change runtime type → GPU (T4)")

    print(f"Modèle    : {args.model}")
    print(f"Dataset   : {args.data}")
    print(f"imgsz     : {args.imgsz}")
    print(f"Epochs    : {args.epochs}")
    print(f"Freeze    : {args.freeze} couches\n")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,           # 1280 = détecte les petits glyphes (~35px)
        batch=args.batch,
        lr0=0.00003,                # très bas : évite catastrophic forgetting
        lrf=0.005,
        warmup_epochs=2,
        freeze=args.freeze,         # 8 = partiel (moins agressif que 15)
        optimizer="AdamW",
        weight_decay=0.0005,
        patience=7,
        device=device,

        # ── Augmentations calibrées pour manuscrits aztèques ──────────────
        degrees=12,                 # rotations légères (glyphes pas trop inclinés)
        translate=0.08,
        scale=0.35,                 # variation taille (glyphes de tailles variables)
        shear=3,                    # légère torsion (support souple)
        perspective=0.0008,         # distorsion perspective (amate = souple)
        fliplr=0.0,                 # DÉSACTIVÉ : glyphes ont une orientation
        flipud=0.0,                 # DÉSACTIVÉ : idem
        mosaic=1.0,                 # CRITIQUE : apprend le contexte dense
        mixup=0.05,
        copy_paste=0.0,
        hsv_h=0.008,                # peu de variation teinte (encre = stable)
        hsv_s=0.25,
        hsv_v=0.45,                 # variation luminosité (encre fanée)
        erasing=0.35,               # glyphes partiellement effacés (réaliste)

        # ── Paramètres de sauvegarde ───────────────────────────────────────
        project=args.project,
        name=args.name,
        exist_ok=True,
        val=True,
        plots=True,
        save=True,
        save_period=5,              # sauvegarde tous les 5 epochs (sécurité Colab)
        verbose=True,
    )

    _print_report(results, args)


def _print_report(results, args) -> None:
    """Affiche un résumé des résultats et les commandes de packaging."""
    out_dir = Path(args.project) / args.name

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)

    # Lire results.csv pour afficher les métriques finales
    csv_path = out_dir / "results.csv"
    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            keys = [k.strip() for k in last.keys()]
            vals = [v.strip() for v in last.values()]
            data = dict(zip(keys, vals))
            print("\nMétriques finales :")
            for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "val/box_loss"]:
                if k in data:
                    print(f"  {k:<30} {data[k]}")

    best = out_dir / "weights" / "best.pt"
    if best.exists():
        size_mb = best.stat().st_size / 1e6
        print(f"\nModèle final : {best} ({size_mb:.1f} MB)")
    else:
        print(f"\nAttention : best.pt non trouvé dans {out_dir}/weights/")

    print(f"""
Commandes pour packager et envoyer à David :

    zip aztec_finetuned_v3.zip \\
        {out_dir}/weights/best.pt \\
        {out_dir}/results.csv \\
        {out_dir}/results.png \\
        {out_dir}/confusion_matrix.png

Taille estimée : ~25 MB
""")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
