#!/usr/bin/env python3
"""
predict.py — Inférence YOLOv8 Segmentation
Exécute la segmentation sur une image ou un dossier d'images
avec le modèle entraîné.
Usage:
  # Une seule image
  python predict.py --model runs/segment/seg_train/weights/best.pt --source image.png
  # Un dossier d'images
  python predict.py --model runs/segment/seg_train/weights/best.pt --source images/
  # Avec options
  python predict.py --model best.pt --source img.png --conf 0.5 --save-json --save-masks
"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
def predict(args):
    """Exécute l'inférence de segmentation."""
    # Vérifications
    if not Path(args.model).exists():
        print(f"❌ Modèle introuvable : {args.model}")
        sys.exit(1)
    source = Path(args.source)
    if not source.exists():
        print(f"❌ Source introuvable : {args.source}")
        sys.exit(1)
    print(f"🔧 Modèle : {args.model}")
    print(f"📁 Source : {args.source}")
    print(f"🎯 Confiance min : {args.conf}")
    print(f"📐 Image size : {args.imgsz}")
    # Charger le modèle
    model = YOLO(args.model)
    # Exécuter l'inférence
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_txt,
        project=args.project or "runs/predict",
        name=args.name or "seg_predict",
        exist_ok=True,
        retina_masks=True,  # Masques haute résolution
        verbose=True,
    )
    # Post-traitement et sauvegarde JSON
    all_detections = []
    for i, result in enumerate(results):
        img_name = Path(result.path).name
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            masks = result.masks
            for j in range(len(boxes)):
                det = {
                    "image": img_name,
                    "class_id": int(boxes.cls[j]),
                    "class_name": result.names[int(boxes.cls[j])],
                    "confidence": round(float(boxes.conf[j]), 4),
                    "bbox": {
                        "x1": round(float(boxes.xyxy[j][0]), 2),
                        "y1": round(float(boxes.xyxy[j][1]), 2),
                        "x2": round(float(boxes.xyxy[j][2]), 2),
                        "y2": round(float(boxes.xyxy[j][3]), 2),
                    },
                    "bbox_normalized": {
                        "x_center": round(float(boxes.xywhn[j][0]), 4),
                        "y_center": round(float(boxes.xywhn[j][1]), 4),
                        "width": round(float(boxes.xywhn[j][2]), 4),
                        "height": round(float(boxes.xywhn[j][3]), 4),
                    },
                }
                # Ajouter les polygones du masque si disponibles
                if masks is not None and j < len(masks):
                    mask_xy = masks.xy[j]
                    if len(mask_xy) > 0:
                        det["polygon"] = [
                            {"x": round(float(pt[0]), 2), "y": round(float(pt[1]), 2)}
                            for pt in mask_xy
                        ]
                detections.append(det)
        all_detections.append({
            "image": img_name,
            "num_detections": len(detections),
            "detections": detections,
        })
        print(f"  📷 {img_name} → {len(detections)} détections")
    # Sauvegarder les résultats JSON
    if args.save_json:
        output_dir = Path(args.project or "runs/predict") / (args.name or "seg_predict")
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(all_detections, f, indent=2, ensure_ascii=False)
        print(f"\n📊 Résultats JSON → {json_path}")
    # Sauvegarder les masques binaires individuels
    if args.save_masks:
        output_dir = Path(args.project or "runs/predict") / (args.name or "seg_predict") / "masks"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, result in enumerate(results):
            if result.masks is not None:
                img_name = Path(result.path).stem
                for j, mask in enumerate(result.masks.data):
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                    mask_path = output_dir / f"{img_name}_mask_{j}.png"
                    cv2.imwrite(str(mask_path), mask_np)
        print(f"🎭 Masques sauvegardés → {output_dir}")
    # Résumé
    total_dets = sum(d["num_detections"] for d in all_detections)
    print(f"\n✅ Total : {len(all_detections)} images, {total_dets} détections")
    return all_detections
def main():
    parser = argparse.ArgumentParser(
        description="Inférence YOLOv8 Segmentation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le modèle .pt (ex: runs/segment/seg_train/weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image ou dossier d'images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Seuil de confiance minimum (défaut: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="Seuil IoU pour NMS (défaut: 0.7)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille de l'image (défaut: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (défaut: '0')",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Sauvegarder les résultats en JSON",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Sauvegarder les labels YOLO",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Sauvegarder les masques binaires PNG",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Dossier du projet",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nom de l'expérience",
    )
    args = parser.parse_args()
    predict(args)
if __name__ == "__main__":
    main()
