#!/usr/bin/env python3
"""
train.py — Entraînement YOLOv8 Segmentation
Entraîne un modèle YOLOv8-seg sur ton dataset personnalisé.
Optimisé pour rester sous 30 minutes par run avec 307 classes / ~46k images.
Usage:
  python train.py --data configs/dataset.yaml --epochs 50 --batch 16 --device 0
Options avancées:
  python train.py --data configs/dataset.yaml --epochs 100 --batch 32 \
    --model yolov8m-seg.pt --imgsz 640 --device 0 --name exp1 --resume
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings
def get_optimal_settings(num_classes: int = 307, finetune: bool = False) -> dict:
    """
    Retourne les hyperparamètres optimisés pour la détection de glyphes aztèques.

    Différences clés par rapport au run initial (best_fixed.pt) :
      - imgsz        : 512 → 640   (meilleure résolution, petits glyphes sur planches)
      - copy_paste   : 0.0 → 0.3   (CRITIQUE : apprend à détecter plusieurs glyphes)
      - degrees      : 0.0 → 5.0   (rotation légère pour les codex)
      - shear        : 0.0 → 2.0   (perspective manuscrit)
      - scale        : 0.5 → 0.7   (plus de variation d'échelle = planches)
      - patience     : 10  → 15    (plus de tolérance avant early stopping)
      - epochs       : 30  → 75    (le run initial s'est arrêté trop tôt)
      - close_mosaic : 10  → 15    (garde la mosaïque plus longtemps)
      - cache        : False → True (déjà prévu, était désactivé au run initial)

    finetune=True : learning rate réduit pour repartir de best_fixed.pt
                    au lieu d'entraîner from scratch.
    """
    lr0 = 0.0002 if finetune else 0.001   # LR faible en fine-tuning
    warmup = 1.0 if finetune else 3.0

    return {
        # --- Modèle ---
        "model_size": "m",                  # yolov8m-seg : bon compromis

        # --- Entraînement ---
        "epochs": 75,                       # ↑ 30→75, early stopping si convergence
        "batch": 16,
        "imgsz": 640,                       # ↑ 512→640 : meilleure détection petits glyphes
        "patience": 15,                     # ↑ 10→15

        # --- Optimiseur ---
        "optimizer": "AdamW",
        "lr0": lr0,                         # 0.001 scratch | 0.0002 finetune
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": warmup,
        "warmup_momentum": 0.8,

        # --- Augmentation ---
        "hsv_h": 0.02,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 5.0,                     # ↑ 0→5 : glyphes légèrement inclinés
        "translate": 0.1,
        "scale": 0.7,                       # ↑ 0.5→0.7 : variation taille sur planches
        "shear": 2.0,                       # ↑ 0→2 : déformation perspective manuscrit
        "perspective": 0.0001,              # ↑ 0→légère
        "flipud": 0.0,                      # pas de flip vertical (codex ont sens)
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.3,                  # ↑ 0→0.3 : CRUCIAL pour multi-glyphes
        "copy_paste_mode": "flip",

        # --- Loss ---
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,

        # --- Performance ---
        "workers": 8,
        "cache": True,                      # cache RAM (était False au run initial)
        "amp": True,
        "cos_lr": True,
        "close_mosaic": 15,                 # ↑ 10→15
    }
class TrainingCallback:
    """
    Callback pour envoyer les métriques de training en temps réel.
    Sauvegarde les métriques dans un fichier JSON pour le frontend.
    """
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.report_dir / "training_metrics.json"
        self.metrics_history = []
        self.start_time = time.time()
    def on_train_epoch_end(self, trainer):
        """Appelé à la fin de chaque epoch."""
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        elapsed = time.time() - self.start_time
        eta = (elapsed / epoch) * (total_epochs - epoch) if epoch > 0 else 0
        metrics = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "timestamp": datetime.now().isoformat(),
        }
        # Ajouter les métriques du trainer si disponibles
        if hasattr(trainer, "loss") and trainer.loss is not None:
            loss_items = trainer.loss.detach().cpu().numpy()
            metrics["box_loss"] = round(float(loss_items[0]), 4)
            metrics["seg_loss"] = round(float(loss_items[1]), 4)
            metrics["cls_loss"] = round(float(loss_items[2]), 4)
            metrics["dfl_loss"] = round(float(loss_items[3]), 4)
        if hasattr(trainer, "lr"):
            metrics["learning_rate"] = {
                k: round(v, 8) for k, v in trainer.lr.items()
            }
        self.metrics_history.append(metrics)
        # Sauvegarder le rapport JSON
        report = {
            "status": "training",
            "model_name": trainer.args.model,
            "dataset": str(trainer.args.data),
            "device": str(trainer.device),
            "progress": round(epoch / total_epochs * 100, 1),
            "current_epoch": epoch,
            "total_epochs": total_epochs,
            "best_fitness": round(float(trainer.best_fitness), 4) if trainer.best_fitness else None,
            "history": self.metrics_history,
        }
        with open(self.metrics_file, "w") as f:
            json.dump(report, f, indent=2)
    def on_train_end(self, trainer):
        """Appelé à la fin de l'entraînement."""
        report = {
            "status": "completed",
            "total_time_seconds": round(time.time() - self.start_time, 1),
            "best_model": str(trainer.best),
            "last_model": str(trainer.last),
            "results_dir": str(trainer.save_dir),
            "history": self.metrics_history,
        }
        with open(self.metrics_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Rapport sauvegardé → {self.metrics_file}")
def train(args):
    """Lance l'entraînement (scratch ou fine-tuning)."""
    finetune = args.finetune
    settings_opt = get_optimal_settings(finetune=finetune)

    # Fine-tuning : on repart de best_fixed.pt
    # Scratch     : on repart de yolov8m-seg.pt (ImageNet pré-entraîné)
    if finetune:
        default_model = "best_fixed.pt"
    else:
        default_model = f"yolov8{settings_opt['model_size']}-seg.pt"

    model_name = args.model or default_model

    mode_label = "FINE-TUNING depuis best_fixed.pt" if finetune else "Entraînement FROM SCRATCH"
    print(f"\n{'='*60}")
    print(f"  {mode_label}")
    print(f"{'='*60}")
    print(f"  Modèle     : {model_name}")
    print(f"  Dataset    : {args.data}")
    print(f"  Device     : {args.device}")
    print(f"  Image size : {args.imgsz or settings_opt['imgsz']}")
    print(f"  Batch      : {args.batch or settings_opt['batch']}")
    print(f"  Epochs     : {args.epochs or settings_opt['epochs']}")
    print(f"  copy_paste : {settings_opt['copy_paste']}")
    print(f"  LR0        : {settings_opt['lr0']}")
    print(f"{'='*60}\n")

    if args.resume and Path(model_name).exists():
        model = YOLO(model_name)
        print("♻️  Reprise de l'entraînement...")
    else:
        model = YOLO(model_name)
        if finetune:
            print("🔁 Fine-tuning depuis le modèle existant...")
        else:
            print("🆕 Nouvel entraînement from scratch...")
    # Setup du callback pour les rapports temps réel
    report_dir = Path(args.report_dir) if args.report_dir else Path("runs/reports")
    callback = TrainingCallback(report_dir)
    model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
    model.add_callback("on_train_end", callback.on_train_end)
    # Lancer l'entraînement
    results = model.train(
        data=str(args.data),
        # Contrôle du temps
        epochs=args.epochs or settings_opt["epochs"],
        patience=settings_opt["patience"],
        # Taille et batch
        imgsz=args.imgsz or settings_opt["imgsz"],
        batch=args.batch or settings_opt["batch"],
        # Optimiseur
        optimizer=settings_opt["optimizer"],
        lr0=settings_opt["lr0"],
        lrf=settings_opt["lrf"],
        momentum=settings_opt["momentum"],
        weight_decay=settings_opt["weight_decay"],
        warmup_epochs=settings_opt["warmup_epochs"],
        warmup_momentum=settings_opt["warmup_momentum"],
        # Augmentation
        hsv_h=settings_opt["hsv_h"],
        hsv_s=settings_opt["hsv_s"],
        hsv_v=settings_opt["hsv_v"],
        degrees=settings_opt["degrees"],
        translate=settings_opt["translate"],
        scale=settings_opt["scale"],
        shear=settings_opt["shear"],
        perspective=settings_opt["perspective"],
        flipud=settings_opt["flipud"],
        fliplr=settings_opt["fliplr"],
        mosaic=settings_opt["mosaic"],
        mixup=settings_opt["mixup"],
        copy_paste=settings_opt["copy_paste"],
        # Loss
        box=settings_opt["box"],
        cls=settings_opt["cls"],
        dfl=settings_opt["dfl"],
        # Performance
        workers=settings_opt["workers"],
        cache=settings_opt["cache"],
        amp=settings_opt["amp"],
        cos_lr=settings_opt["cos_lr"],
        close_mosaic=settings_opt["close_mosaic"],
        # Logging
        device=args.device,
        name=args.name or "seg_train",
        project=args.project or "runs/segment",
        exist_ok=True,
        resume=args.resume,
        verbose=True,
        plots=True,
    )
    # Résumé final
    print("\n" + "=" * 60)
    print("🏁 ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    if results:
        print(f"  📁 Résultats → {model.trainer.save_dir}")
        print(f"  🏆 Meilleur modèle → {model.trainer.best}")
        print(f"  📊 Rapport → {callback.metrics_file}")
    return model
def main():
    parser = argparse.ArgumentParser(
        description="Entraînement YOLOv8 Segmentation"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Chemin vers le fichier dataset.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modèle pré-entraîné (défaut: yolov8m-seg.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Nombre d'epochs (défaut: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Taille du batch (défaut: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Taille des images (défaut: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device(s) : '0', '0,1', 'cpu' (défaut: '0')",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nom de l'expérience (défaut: seg_train)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Dossier du projet (défaut: runs/segment)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Dossier pour les rapports temps réel (défaut: runs/reports)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre un entraînement interrompu",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help=(
            "Fine-tuning depuis best_fixed.pt (lr réduit, warmup court). "
            "Recommandé pour améliorer le modèle existant rapidement. "
            "Sans ce flag : entraînement from scratch depuis yolov8m-seg.pt."
        ),
    )
    args = parser.parse_args()
    train(args)
if __name__ == "__main__":
    main()
