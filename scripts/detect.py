#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def detect_codex(model, image_path: Path, out_dir: Path, conf: float, iou: float):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Impossible de lire : {image_path}")
        return

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    detections = []
    annotated = image.copy()

    for i, box in enumerate(results.boxes):
        cls_id    = int(box.cls[0])
        cls_name  = model.names[cls_id]
        confidence = round(float(box.conf[0]), 3)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        detections.append({
            "id": i,
            "glyph": cls_name,
            "confidence": confidence,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center": {"x": cx, "y": cy},
            "width": x2 - x1,
            "height": y2 - y1,
        })

        # Dessine la boîte et le label
        color = (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Marque le centre
        cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # Sauvegarde image annotée
    img_out = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(img_out), annotated)

    # Sauvegarde rapport JSON
    report = {
        "source": str(image_path),
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "total_detections": len(detections),
        "detections": sorted(detections, key=lambda d: (d["bbox"]["y1"], d["bbox"]["x1"])),
    }
    json_out = out_dir / f"{stem}_report.json"
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Résumé des glyphes trouvés
    counts = {}
    for d in detections:
        counts[d["glyph"]] = counts.get(d["glyph"], 0) + 1

    print(f"\n{'='*50}")
    print(f"Codex : {image_path.name}")
    print(f"{'='*50}")
    print(f"  {len(detections)} glyphe(s) détecté(s) — {len(counts)} type(s) différent(s)\n")
    for glyph, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  • {glyph:<30} x{count}")
    print(f"\n  Image  : {img_out}")
    print(f"  Rapport: {json_out}")
    print(f"{'='*50}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Détection de glyphes aztèques dans des codex")
    parser.add_argument("input",  type=Path, help="Image ou dossier de codex à analyser")
    parser.add_argument("--model", type=str, default="runs/segment/seg_train/weights/best.pt",
                        help="Chemin vers le modèle entraîné")
    parser.add_argument("--out",  type=Path, default=Path("runs/detections"),
                        help="Dossier de sortie pour les images et rapports")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance (défaut: 0.25)")
    parser.add_argument("--iou",  type=float, default=0.45, help="Seuil IoU NMS (défaut: 0.45)")
    args = parser.parse_args()

    model = YOLO(args.model)
    print(f"Modèle chargé : {args.model}")
    print(f"Classes       : {len(model.names)}")

    if args.input.is_dir():
        images = sorted(args.input.glob("*.png")) + sorted(args.input.glob("*.jpg"))
        print(f"\n{len(images)} image(s) trouvée(s) dans {args.input}\n")
        total = 0
        for img_path in images:
            report = detect_codex(model, img_path, args.out, args.conf, args.iou)
            if report:
                total += report["total_detections"]
        print(f"\nTotal : {total} glyphes détectés sur {len(images)} codex")
    else:
        detect_codex(model, args.input, args.out, args.conf, args.iou)


if __name__ == "__main__":
    main()
