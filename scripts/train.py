#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


class MetricsLogger:
    def __init__(self, out_dir: Path):
        self.path = out_dir / "metrics.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.t0 = time.time()

    def on_train_epoch_end(self, trainer):
        elapsed = time.time() - self.t0
        epoch = trainer.epoch + 1
        entry = {
            "epoch": epoch,
            "elapsed": round(elapsed, 1),
            "eta": round(elapsed / epoch * (trainer.epochs - epoch), 1),
            "timestamp": datetime.now().isoformat(),
        }
        if hasattr(trainer, "loss") and trainer.loss is not None:
            t = trainer.loss.detach().cpu()
            vals = t.tolist() if t.dim() > 0 else [float(t)]
            entry.update(dict(zip(["box", "seg", "cls", "dfl"], [round(v, 4) for v in vals])))
        self.history.append(entry)
        self.path.write_text(json.dumps({
            "status": "training",
            "progress": round(epoch / trainer.epochs * 100, 1),
            "best_fitness": round(float(trainer.best_fitness), 4) if trainer.best_fitness else None,
            "history": self.history,
        }, indent=2))

    def on_train_end(self, trainer):
        self.path.write_text(json.dumps({
            "status": "completed",
            "duration_s": round(time.time() - self.t0, 1),
            "best": str(trainer.best),
            "history": self.history,
        }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--epochs",   type=int, default=75)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--name",     default="seg_train")
    parser.add_argument("--data",     default="configs/dataset.yaml")
    parser.add_argument("--resume",   action="store_true")
    args = parser.parse_args()

    model_path = "best_fixed.pt" if args.finetune else "yolov8m-seg.pt"
    lr0        = 0.0002          if args.finetune else 0.001
    warmup     = 1.0             if args.finetune else 3.0

    model  = YOLO(model_path)
    logger = MetricsLogger(Path("runs/reports"))
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)
    model.add_callback("on_train_end",       logger.on_train_end)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        project="runs/segment",
        exist_ok=True,
        resume=args.resume,
        # Optimizer
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=warmup,
        warmup_momentum=0.8,
        cos_lr=True,
        # Augmentation — tuned for aztec codex glyphs
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.7,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.3,
        copy_paste_mode="flip",
        # Loss weights
        box=7.5, cls=0.5, dfl=1.5,
        # Performance
        workers=8,
        cache=True,
        amp=True,
        close_mosaic=15,
        patience=15,
        verbose=True,
        plots=True,
    )

    # Evaluation on test set
    best = Path(f"runs/segment/{args.name}/weights/best.pt")
    print("\n--- Evaluation test set ---")
    metrics = YOLO(str(best)).val(
        data=args.data,
        split="test",
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
        save_json=True,
    )
    print(f"mAP50      : {metrics.seg.map50:.4f}")
    print(f"mAP50-95   : {metrics.seg.map:.4f}")
    print(f"Precision  : {metrics.seg.mp:.4f}")
    print(f"Recall     : {metrics.seg.mr:.4f}")

    # Upload automatique via transfer.sh + notification téléphone
    import subprocess
    NTFY_TOPIC = "aztec-imran-66832"
    try:
        print("\nUpload du modèle vers transfer.sh...")
        result = subprocess.run(
            ["curl", "--upload-file", str(best), "https://transfer.sh/best.pt"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            print(f"\n*** LIEN DE TELECHARGEMENT (valable 14 jours) ***")
            print(f"  {url}")
            Path("runs/download_link.txt").write_text(url)
            # Notification téléphone
            subprocess.run([
                "curl", "-s", "-d", f"Entrainement termine! Telecharge best.pt ici: {url}",
                f"https://ntfy.sh/{NTFY_TOPIC}"
            ], timeout=15)
            print("Notification envoyée sur ton téléphone !")
        else:
            raise RuntimeError(result.stderr)
    except Exception as e:
        print(f"\nUpload échoué ({e})")
        subprocess.run([
            "curl", "-s", "-d",
            f"Entrainement termine mais upload echoue. Connecte-toi a l'ecole pour recuperer best.pt",
            f"https://ntfy.sh/{NTFY_TOPIC}"
        ], timeout=15)
        print(f"  scp -P 22 lab-7fb3bcfce5@10.94.11.10:{best.resolve()} .")


if __name__ == "__main__":
    main()
