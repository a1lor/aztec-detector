"""
inference_planche.py
=====================
Détection de glyphes aztèques sur une planche entière.

Deux modes disponibles :
  1. Direct (imgsz=1280) — pour planches < 2000px
  2. Sliding window — pour grandes planches (> 2000px)

La sliding window découpe la planche en tuiles qui se chevauchent,
lance le modèle sur chaque tuile, puis fusionne avec NMS global.
C'est la méthode la plus robuste pour les grandes planches.

Usage :
    # Mode automatique (choisit le meilleur mode selon la taille)
    python inference_planche.py --model best_finetuned.pt --image planche.jpg

    # Forcer le sliding window
    python inference_planche.py --model best_finetuned.pt --image planche.jpg --mode sliding

    # Sauvegarder les détections en JSON
    python inference_planche.py --model best_finetuned.pt --image planche.jpg --json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


CLASSES = [
    "cacahuatl","teocomitl","tepetla","tlapexohuiloni","toloa","xicolli",
    "xihuitl_2","yauhtli","yopitzontli","zo","azcatl","castillan_totolin",
    "chicuatli","mixoyotl","terre_01","ayocuan","fruto_01","fruto_02",
    "maitl_2","tenchilnahuayo","tlemaitl","iztetl","mantilla","matlactli",
    "peinado_erizado","petlacalli","pochotl","tlilhuahuana","camotli","huilotl",
    "icpatl","mamatlatl","mizquicuahuitl","pachtli","tamalli","tocatl",
    "calcuamimilli","cáliz","elli","huauhtli","mapilli","nenepilli","ocozacatl",
    "potzalli","tizatl","tziuhtli","casa","pechtli","tecciztli","ameyalli",
    "barra","cozcacuauhtli","cuezalin","icopi","molcaxitl","poyomatli",
    "tlalpilli_2","tlapanqui","tlaquimilolli","tlatelli","yahualli","camachalli",
    "copilli","cuetlaxtli","tlahuitolli","zayolin","ichcaxochitl","xocpalli",
    "xoxoqui","cozamatl","cuauhcozcatl","icpalli","miyahuatl","chapolin",
    "cuezcomatl","cuicatl","omitl_1","tamazolin","tlachtli","tlanelhuatl",
    "tlotli","xacalli","chocholli","metztli","pintura_corporal","quetzallalpiloni",
    "tenextli","tianquiztli","amacalli","cuecuextli","iglesia","ilhuicatl",
    "ixtlilli","nacaztli","quequetzalli","xochipalli","yollotl","chitatli",
    "omitl_2","quetzalmiyahuayotl","tlaloc","yacatl","cuitlatl","tecolotl",
    "tlacotl","tlamamallahuiztli","yacametztli","acayetl","tozan","xicalcoliuhqui",
    "xicotli","iztatl","tejuelo_1","tenexcalli","xicalli","acalli","coxolitli",
    "tentetl","matlatl","tlalpiloni","cruz","macuextli","cuechtli","tlacuiloloni",
    "cueyatl","maxaltic","texcalli","tzacualli","cimatl","ehecacehuaztli",
    "ehuatl_2","ocuilin","teopantli","malacatl","tlaxichtli","apantli","teuhtli",
    "huehuetl","matlalli","tzonquemitl","huexocuahuitl","ococuahuitl","octli",
    "tenamitl","cozcapetlatl","cuetlachtli","malinalli","quechtzontli","tonalli",
    "xolotl","oxitl","nextli","tomin","topilli","ayatl","quemitl","tlacuilolli",
    "yohualli","macuahuitl","ohuatl","cipactli","caxitl","cuacuahuitl",
    "yahualiuhqui","eztli","citlalin","cilli","xilotl","huipilli","nacochtli",
    "metl","tlanextli","ichcahuipilli","tzihuactli","xiuhuitzolli","cuaxicalli",
    "oztotl","coyolli","papalotl","nacayotl","nezahualli","cuixtli","maquiztli",
    "nopalli","xolochauhqui","carga_de_yerba","miztli","tezcatl","tlaolli","zolin",
    "chiquihuitl","tepoztli_2","tonatiuh","tlalpilli","olin","otli","tolin",
    "aztatl","maxtlatl","chian","tzontli_2","xihuitl_1","cuaitl","borrado",
    "quecholli","totolteme","teoxihuitl","amatl","tecomatl","chilli","ocotl",
    "matlalin","nochtli","cueitl","olli","tentlapilolli","itzcuintli","tlaxcalli",
    "nenetl","quimilli","etl","cuahuacalli","cuetzpalin","teocuitlatl","tletl",
    "tzintli","xalli","zacatl","coltic","cuahuitl_1","icxitl","ozomatli",
    "cuahuitl_2","iztac","cactli","mecatl","maitl_1","xiquipilli","ixayotl",
    "effacé","totolin","huictli","cemixtlapalli","tlamamalli","tzontli","popoca",
    "xahualli","huitzilin","centli","cihuatl","tlapalli","itztli","línea_enlace",
    "ehecatl","cozcatl","coyotl","mazatl","huitztli","toztli","chalchihuitl",
    "petlatl","xayacatl","tepoztli","tliltic","ixtelolotli","xiuhtic","tototl",
    "tlatoa","michin","ocelotl","quetzalli","xihuitl_3","tecpatl","ihuitl",
    "quiyahuitl","tentli","chimalli","tlantli","xihuitl_4","acatl","macpalli",
    "mixtli","tilmatli","tlalli","comitl","tochtli","coztic","mitl","ayotl",
    "piqui","xocpalmachiyotl","centzontli","xochitl","tepetl","cuauhtli","calli",
    "cohuatl","tetl","tlacatl","atl","macuilli","pantli","ce"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="Chemin vers best_finetuned.pt")
    p.add_argument("--image",   required=True, help="Chemin vers la planche")
    p.add_argument("--conf",    type=float, default=0.20)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--mode",    choices=["auto","direct","sliding"], default="auto")
    p.add_argument("--tile",    type=int, default=1280, help="Taille des tuiles (sliding)")
    p.add_argument("--overlap", type=float, default=0.25, help="Chevauchement tuiles")
    p.add_argument("--out",     default=None, help="Image de sortie (défaut: _detected.jpg)")
    p.add_argument("--json",    action="store_true", help="Exporter les détections en JSON")
    return p.parse_args()


# ─────────────────────── INFÉRENCE DIRECTE ──────────────────────────────────

def detect_direct(model, image_path: str, conf: float, iou_thr: float, imgsz: int = 1280):
    """Inférence standard sur image entière."""
    results = model(image_path, conf=conf, iou=iou_thr, imgsz=imgsz, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "conf": float(box.conf[0]),
                "class_id": int(box.cls[0]),
                "class_name": CLASSES[int(box.cls[0])] if int(box.cls[0]) < len(CLASSES) else "unknown"
            })
    return boxes


# ─────────────────────── SLIDING WINDOW ─────────────────────────────────────

def nms_boxes(boxes: list, iou_threshold: float = 0.45) -> list:
    """NMS global sur toutes les détections fusionnées."""
    if not boxes:
        return []
    import torch
    bboxes = torch.tensor([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes], dtype=torch.float32)
    scores = torch.tensor([b["conf"] for b in boxes], dtype=torch.float32)
    try:
        from torchvision.ops import nms as tv_nms
        keep = tv_nms(bboxes, scores, iou_threshold)
        return [boxes[i] for i in keep.tolist()]
    except ImportError:
        # Fallback : NMS manuel
        keep = []
        order = scores.argsort(descending=True).tolist()
        suppressed = set()
        for i in order:
            if i in suppressed:
                continue
            keep.append(i)
            b1 = boxes[i]
            for j in order:
                if j <= i or j in suppressed:
                    continue
                b2 = boxes[j]
                ix1 = max(b1["x1"], b2["x1"]); iy1 = max(b1["y1"], b2["y1"])
                ix2 = min(b1["x2"], b2["x2"]); iy2 = min(b1["y2"], b2["y2"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter == 0:
                    continue
                a1 = (b1["x2"]-b1["x1"]) * (b1["y2"]-b1["y1"])
                a2 = (b2["x2"]-b2["x1"]) * (b2["y2"]-b2["y1"])
                iou_val = inter / max(a1 + a2 - inter, 1)
                if iou_val > iou_threshold:
                    suppressed.add(j)
        return [boxes[i] for i in keep]


def detect_sliding(model, image_path: str, conf: float, iou_thr: float,
                   tile_size: int = 1280, overlap: float = 0.25) -> list:
    """
    Sliding window : découpe la planche en tuiles qui se chevauchent,
    détecte sur chaque tuile, puis fusionne avec NMS global.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    H, W = img.shape[:2]
    step  = int(tile_size * (1 - overlap))
    all_boxes = []

    print(f"  Planche : {W}×{H}px")
    print(f"  Tuiles  : {tile_size}px, pas={step}px, overlap={overlap:.0%}")

    n_tiles = 0
    for y in range(0, H, step):
        for x in range(0, W, step):
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            tile = img[y:y2, x:x2]

            results = model(tile, conf=conf, iou=iou_thr, imgsz=tile_size, verbose=False)
            n_tiles += 1

            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    cid = int(box.cls[0])
                    all_boxes.append({
                        "x1": int(bx1 + x), "y1": int(by1 + y),
                        "x2": int(bx2 + x), "y2": int(by2 + y),
                        "conf": float(box.conf[0]),
                        "class_id": cid,
                        "class_name": CLASSES[cid] if cid < len(CLASSES) else "unknown"
                    })

    print(f"  Tuiles traitées : {n_tiles}")
    print(f"  Détections brutes : {len(all_boxes)}")

    merged = nms_boxes(all_boxes, iou_threshold=iou_thr)
    print(f"  Après NMS global : {len(merged)}")
    return merged


# ─────────────────────── VISUALISATION ──────────────────────────────────────

def draw_detections(image_path: str, boxes: list, out_path: str) -> None:
    """Dessine les bounding boxes sur l'image et la sauvegarde."""
    img = cv2.imread(image_path)
    if img is None:
        return

    # Palette de couleurs par classe
    rng = np.random.default_rng(0)
    colors = {cid: tuple(int(c) for c in rng.integers(60, 220, 3))
              for cid in range(len(CLASSES))}

    for b in boxes:
        cid   = b["class_id"]
        color = colors.get(cid, (0, 200, 0))
        label = f"{b['class_name']} {b['conf']:.2f}"

        cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)

        # Fond du label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(b["y1"] - 4, th + 4)
        cv2.rectangle(img, (b["x1"], ly - th - 4), (b["x1"] + tw + 4, ly), color, -1)
        cv2.putText(img, label, (b["x1"] + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Image sauvegardée : {out_path}")


# ─────────────────────── MAIN ────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERREUR : ultralytics non installé. Lancez : pip install ultralytics")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"ERREUR : modèle introuvable : {args.model}")
        sys.exit(1)
    if not Path(args.image).exists():
        print(f"ERREUR : image introuvable : {args.image}")
        sys.exit(1)

    print(f"\nModèle : {args.model}")
    print(f"Image  : {args.image}")
    print(f"Conf   : {args.conf} | IoU : {args.iou}")

    model = YOLO(args.model)

    # Choix du mode
    img_check = cv2.imread(args.image)
    if img_check is None:
        print(f"ERREUR : impossible de lire {args.image}")
        sys.exit(1)
    H, W = img_check.shape[:2]
    del img_check

    if args.mode == "auto":
        mode = "sliding" if max(H, W) > 1800 else "direct"
        print(f"Mode auto → {mode} (image {W}×{H}px)")
    else:
        mode = args.mode

    print()
    if mode == "direct":
        boxes = detect_direct(model, args.image, args.conf, args.iou, imgsz=args.tile)
    else:
        boxes = detect_sliding(model, args.image, args.conf, args.iou,
                               tile_size=args.tile, overlap=args.overlap)

    print(f"\n{len(boxes)} glyphes détectés")

    # Afficher les top détections
    if boxes:
        top = sorted(boxes, key=lambda b: b["conf"], reverse=True)[:10]
        print("\nTop 10 détections :")
        for b in top:
            print(f"  {b['class_name']:<30} conf={b['conf']:.3f}  "
                  f"bbox=[{b['x1']},{b['y1']},{b['x2']},{b['y2']}]")

    # Sauvegarder l'image annotée
    out_path = args.out or str(Path(args.image).stem) + "_detected.jpg"
    draw_detections(args.image, boxes, out_path)

    # Export JSON
    if args.json:
        json_path = str(Path(args.image).stem) + "_detections.json"
        with open(json_path, "w") as f:
            json.dump({"image": args.image, "n_detections": len(boxes),
                       "boxes": boxes}, f, indent=2, ensure_ascii=False)
        print(f"  JSON sauvegardé : {json_path}")


if __name__ == "__main__":
    main()
