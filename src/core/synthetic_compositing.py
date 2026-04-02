#!/usr/bin/env python3
"""
synthetic_compositing.py
Génère des images synthétiques de "planches" à partir des crops isolés du dataset original.
Chaque image = plusieurs glyphes collés sur un fond parchemin synthétique, avec annotations YOLO parfaites.

Usage (Google Colab / Kaggle):
    python src/core/synthetic_compositing.py \
        --crops  data/full_yolo_dataset/images/train \
        --labels data/full_yolo_dataset/labels/train \
        --out    data/synthetic_planches \
        --n      2000
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = [
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
    "tlacotl","tlamamallahuiztli","yacametztli","acayetl","tozan",
    "xicalcoliuhqui","xicotli","iztatl","tejuelo_1","tenexcalli","xicalli",
    "acalli","coxolitli","tentetl","matlatl","tlalpiloni","cruz","macuextli",
    "cuechtli","tlacuiloloni","cueyatl","maxaltic","texcalli","tzacualli",
    "cimatl","ehecacehuaztli","ehuatl_2","ocuilin","teopantli","malacatl",
    "tlaxichtli","apantli","teuhtli","huehuetl","matlalli","tzonquemitl",
    "huexocuahuitl","ococuahuitl","octli","tenamitl","cozcapetlatl","cuetlachtli",
    "malinalli","quechtzontli","tonalli","xolotl","oxitl","nextli","tomin",
    "topilli","ayatl","quemitl","tlacuilolli","yohualli","macuahuitl","ohuatl",
    "cipactli","caxitl","cuacuahuitl","yahualiuhqui","eztli","citlalin","cilli",
    "xilotl","huipilli","nacochtli","metl","tlanextli","ichcahuipilli",
    "tzihuactli","xiuhuitzolli","cuaxicalli","oztotl","coyolli","papalotl",
    "nacayotl","nezahualli","cuixtli","maquiztli","nopalli","xolochauhqui",
    "carga_de_yerba","miztli","tezcatl","tlaolli","zolin","chiquihuitl",
    "tepoztli_2","tonatiuh","tlalpilli","olin","otli","tolin","aztatl",
    "maxtlatl","chian","tzontli_2","xihuitl_1","cuaitl","borrado","quecholli",
    "totolteme","teoxihuitl","amatl","tecomatl","chilli","ocotl","matlalin",
    "nochtli","cueitl","olli","tentlapilolli","itzcuintli","tlaxcalli","nenetl",
    "quimilli","etl","cuahuacalli","cuetzpalin","teocuitlatl","tletl","tzintli",
    "xalli","zacatl","coltic","cuahuitl_1","icxitl","ozomatli","cuahuitl_2",
    "iztac","cactli","mecatl","maitl_1","xiquipilli","ixayotl","effacé",
    "totolin","huictli","cemixtlapalli","tlamamalli","tzontli","popoca",
    "xahualli","huitzilin","centli","cihuatl","tlapalli","itztli","línea_enlace",
    "ehecatl","cozcatl","coyotl","mazatl","huitztli","toztli","chalchihuitl",
    "petlatl","xayacatl","tepoztli","tliltic","ixtelolotli","xiuhtic","tototl",
    "tlatoa","michin","ocelotl","quetzalli","xihuitl_3","tecpatl","ihuitl",
    "quiyahuitl","tentli","chimalli","tlantli","xihuitl_4","acatl","macpalli",
    "mixtli","tilmatli","tlalli","comitl","tochtli","coztic","mitl","ayotl",
    "piqui","xocpalmachiyotl","centzontli","xochitl","tepetl","cuauhtli",
    "calli","cohuatl","tetl","tlacatl","atl","macuilli","pantli","ce",
]


def parchment_background(size: int) -> np.ndarray:
    """Génère un fond parchemin synthétique avec bruit et taches."""
    r = random.randint(180, 220)
    g = random.randint(155, 200)
    b = random.randint(95, 140)
    base = np.full((size, size, 3), (b, g, r), dtype=np.uint8)
    noise = np.random.normal(0, 14, base.shape).astype(np.int16)
    bg = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(random.randint(2, 7)):
        cx = random.randint(0, size)
        cy = random.randint(0, size)
        radius = random.randint(15, 90)
        color = (random.randint(80, 150), random.randint(90, 160), random.randint(100, 170))
        cv2.circle(bg, (cx, cy), radius, color, -1)
    bg = cv2.GaussianBlur(bg, (21, 21), 0)
    return bg


def load_crops(crops_dir: Path, labels_dir: Path):
    """Charge tous les crops avec leur classe depuis le dataset original."""
    crops = []
    for img_path in crops_dir.glob("*.png"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue
        class_id = int(lines[0].split()[0])
        crops.append((img_path, class_id))
    for img_path in crops_dir.glob("*.jpg"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue
        class_id = int(lines[0].split()[0])
        crops.append((img_path, class_id))
    return crops


def generate(crops_dir: Path, labels_dir: Path, out_dir: Path,
             n_images: int, canvas_size: int, val_ratio: float):

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    crops = load_crops(crops_dir, labels_dir)
    if not crops:
        print("❌ Aucun crop trouvé. Vérifier --crops et --labels.")
        return

    print(f"✅ {len(crops)} crops chargés.")

    n_val   = int(n_images * val_ratio)
    n_train = n_images - n_val

    for idx in range(n_images):
        split  = "val" if idx < n_val else "train"
        canvas = parchment_background(canvas_size)
        annotations = []
        used_boxes  = []

        n_glyphs = random.randint(4, 14)
        sample   = random.sample(crops, min(n_glyphs, len(crops)))

        for (crop_path, class_id) in sample:
            crop = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
            if crop is None:
                continue

            # Redimensionner aléatoirement entre 35 et 130 px
            target = random.randint(35, 130)
            h, w   = crop.shape[:2]
            scale  = target / max(h, w)
            new_w  = max(1, int(w * scale))
            new_h  = max(1, int(h * scale))
            crop   = cv2.resize(crop, (new_w, new_h))

            # Légère rotation aléatoire
            if random.random() < 0.3:
                angle  = random.uniform(-15, 15)
                M      = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1.0)
                crop   = cv2.warpAffine(crop, M, (new_w, new_h))

            # Chercher une position sans chevauchement
            placed = False
            for _ in range(40):
                x1 = random.randint(5, canvas_size - new_w - 5)
                y1 = random.randint(5, canvas_size - new_h - 5)
                x2 = x1 + new_w
                y2 = y1 + new_h
                overlap = any(
                    not (x2 < bx1 - 5 or x1 > bx2 + 5 or y2 < by1 - 5 or y1 > by2 + 5)
                    for (bx1, by1, bx2, by2) in used_boxes
                )
                if not overlap:
                    placed = True
                    break

            if not placed:
                continue

            used_boxes.append((x1, y1, x2, y2))

            # Coller le crop (avec canal alpha si disponible)
            if crop.ndim == 3 and crop.shape[2] == 4:
                alpha = crop[:, :, 3:4] / 255.0
                rgb   = crop[:, :, :3]
                roi   = canvas[y1:y2, x1:x2]
                canvas[y1:y2, x1:x2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
            elif crop.ndim == 3:
                canvas[y1:y2, x1:x2] = crop[:, :, :3]
            else:
                canvas[y1:y2, x1:x2] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

            # Annotation YOLO normalisée
            cx = (x1 + x2) / 2 / canvas_size
            cy = (y1 + y2) / 2 / canvas_size
            bw = (x2 - x1) / canvas_size
            bh = (y2 - y1) / canvas_size
            annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not annotations:
            continue

        fname = f"synth_{idx:05d}"
        cv2.imwrite(str(out_dir / "images" / split / f"{fname}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_dir / "labels" / split / f"{fname}.txt").write_text("\n".join(annotations))

        if idx % 200 == 0:
            print(f"  [{idx}/{n_images}] split={split}  glyphs={len(annotations)}")

    # Écrire dataset.yaml
    names_str = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml_content = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\nval: images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names_str}\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_content)

    print(f"\n{'='*50}")
    print(f"✅ Terminé : {n_train} train + {n_val} val")
    print(f"📁 Dataset : {out_dir}")
    print(f"\nProchaine étape : python src/core/finetune_planches.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops",  default="data/full_yolo_dataset/images/train")
    parser.add_argument("--labels", default="data/full_yolo_dataset/labels/train")
    parser.add_argument("--out",    default="data/synthetic_planches")
    parser.add_argument("--n",      type=int,   default=2000,
                        help="Nombre d'images synthétiques à générer (défaut 2000)")
    parser.add_argument("--size",   type=int,   default=1280,
                        help="Taille du canvas en pixels (défaut 1280)")
    parser.add_argument("--val",    type=float, default=0.2,
                        help="Ratio val set (défaut 0.2)")
    args = parser.parse_args()
    generate(
        Path(args.crops), Path(args.labels), Path(args.out),
        args.n, args.size, args.val
    )
