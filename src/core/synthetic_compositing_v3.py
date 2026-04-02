"""
synthetic_compositing_v3.py
============================
Générateur de planches synthétiques aztèques — version 3.

Améliorations vs v1 (original) :
  - Fond amate réaliste (fibres, pliures, taches d'humidité)
  - Cases dessinées à la main autour des glyphes (50% des glyphes)
  - Colonnes ordonnées (comme les codex de tribut)
  - Encre multicolore et dégradée (noir, rouge sépia, traces effacées)
  - Rotations légères et distorsion perspective
  - Annotations de texte latin fantôme (lignes horizontales)
  - Chevauchement partiel autorisé (MAX_OVERLAP=0.12)
  - Split train/val automatique (80/20)
  - Support PNG transparent (canal alpha) et JPEG

Usage :
    python synthetic_compositing_v3.py --crops data/full_yolo_dataset/images/train \
                                       --labels data/full_yolo_dataset/labels/train \
                                       --out data/synthetic_v3 \
                                       --n 4000
"""

import cv2
import numpy as np
import random
import argparse
import shutil
from pathlib import Path


# ─────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops",  default="data/full_yolo_dataset/images/train")
    p.add_argument("--labels", default="data/full_yolo_dataset/labels/train")
    p.add_argument("--out",    default="data/synthetic_v3")
    p.add_argument("--n",      type=int, default=4000)
    p.add_argument("--canvas", type=int, default=1280)
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


# ─────────────────────── GÉNÉRATION DE FOND ─────────────────────────────────

def amate_background(size: int) -> np.ndarray:
    """
    Fond amate réaliste (écorce de figuier) basé sur l'analyse
    de la vraie planche uploadée :
      - Fibres horizontales ondulées
      - Trame verticale secondaire légère
      - Grandes taches d'humidité / moisissures
      - Pliure centrale (visible sur la planche)
      - Vignette douce sur les bords
    """
    # Couleur de base : beige-ocre amate (warmer than parchemin)
    base = np.zeros((size, size, 3), dtype=np.float32)
    r = random.randint(195, 232)
    g = random.randint(170, 212)
    b = random.randint(118, 160)
    base[:] = [b, g, r]  # BGR pour OpenCV

    # ── Fibres horizontales (trait caractéristique de l'amate) ──
    n_fibres = random.randint(50, 100)
    for _ in range(n_fibres):
        y_f   = random.randint(0, size - 1)
        thick = random.randint(1, 3)
        dark  = random.uniform(0.82, 0.97)
        freq  = random.uniform(1.5, 5.0)
        amp   = random.randint(3, 10)
        wave  = (np.sin(np.linspace(0, freq * np.pi, size)) * amp).astype(int)
        for dx in range(size):
            yy = int(np.clip(y_f + wave[dx], 0, size - 1))
            y_end = min(yy + thick, size)
            base[yy:y_end, dx] *= dark

    # ── Trame verticale secondaire ──
    n_vert = random.randint(12, 30)
    for _ in range(n_vert):
        x_f   = random.randint(0, size - 1)
        thick = random.randint(1, 2)
        dark  = random.uniform(0.93, 0.99)
        x_end = min(x_f + thick, size)
        base[:, x_f:x_end] *= dark

    # ── Taches d'humidité / zones décolorées ──
    n_taches = random.randint(3, 9)
    Y, X = np.ogrid[:size, :size]
    for _ in range(n_taches):
        cx   = random.randint(0, size)
        cy   = random.randint(0, size)
        rx_e = random.randint(40, 220)
        ry_e = random.randint(25, 160)
        mask = ((X - cx)**2 / max(rx_e**2, 1) + (Y - cy)**2 / max(ry_e**2, 1)) < 1
        delta = random.choice([-1, 1]) * random.uniform(0.04, 0.14)
        base[mask] = np.clip(base[mask] * (1 + delta), 0, 255)

    # ── Pliure (60% des images) ──
    if random.random() < 0.60:
        fold_x = random.randint(size // 4, 3 * size // 4)
        fw = random.randint(2, 7)
        x_end = min(fold_x + fw, size)
        base[:, fold_x:x_end] *= random.uniform(0.65, 0.82)

    # ── Vignette bords ──
    cx2, cy2 = size // 2, size // 2
    dist = np.sqrt(((X - cx2) / (size * 0.55))**2 + ((Y - cy2) / (size * 0.55))**2)
    vignette = 1.0 - np.clip(dist * 0.22, 0, 0.28)
    base *= vignette[:, :, np.newaxis]

    # ── Bruit de grain léger ──
    grain = np.random.normal(0, 4, base.shape).astype(np.float32)
    base = np.clip(base + grain, 0, 255)

    return base.astype(np.uint8)


def add_latin_text_lines(canvas: np.ndarray, size: int) -> None:
    """
    Ajoute des lignes horizontales simulant du texte latin colonial
    (annotation espagnole visible sur la vraie planche).
    Ces lignes apprennent au modèle à ignorer le bruit textuel.
    """
    n_zones = random.randint(1, 3)
    for _ in range(n_zones):
        y_start = random.randint(20, size - 120)
        n_lines = random.randint(3, 8)
        line_h  = random.randint(10, 16)
        ink     = random.choice([(20, 20, 20), (40, 10, 10)])  # noir ou encre brune

        for i in range(n_lines):
            y = y_start + i * line_h
            if y >= size:
                break
            x_start = random.randint(10, 60)
            x_end   = random.randint(size // 2, size - 20)
            # Ligne de base du texte (fine)
            cv2.line(canvas, (x_start, y), (x_end, y), ink, 1)
            # Hastes aléatoires (lettres)
            for _ in range(random.randint(5, 20)):
                xh = random.randint(x_start, x_end)
                h  = random.randint(3, line_h - 2)
                cv2.line(canvas, (xh, y - h), (xh, y), ink, 1)


# ─────────────────────── TRAITEMENT DES CROPS ───────────────────────────────

def load_crops(crops_dir: Path, labels_dir: Path) -> list:
    crops = []
    for img_path in crops_dir.glob("*.png"):
        lp = labels_dir / (img_path.stem + ".txt")
        if not lp.exists():
            continue
        lines = lp.read_text().strip().splitlines()
        if not lines:
            continue
        class_id = int(lines[0].split()[0])
        crops.append((img_path, class_id))
    print(f"  Crops chargés : {len(crops)}")
    return crops


def rotate_crop(crop: np.ndarray, angle: float) -> np.ndarray:
    """Rotation avec conservation du canal alpha."""
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        crop, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )


def apply_ink_degradation(crop: np.ndarray) -> np.ndarray:
    """
    Simule le vieillissement de l'encre :
      - Opacité réduite (encre délavée)
      - Légère teinte sépia/ocre
      - Bruit sur le trait (encre non uniforme)
    """
    result = crop.astype(np.float32)
    fade   = random.uniform(0.55, 1.0)

    if result.shape[2] == 4:
        result[:, :, 3] *= fade
        # Teinte sépia légère sur les pixels foncés
        sepia = random.uniform(0.0, 0.25)
        result[:, :, 0] *= (1 - sepia * 0.15)  # canal B
        result[:, :, 1] *= (1 - sepia * 0.08)  # canal G

    # Bruit de grain sur le trait
    noise = np.random.normal(0, random.uniform(2, 8), result[:, :, :3].shape)
    result[:, :, :3] = np.clip(result[:, :, :3] + noise, 0, 255)

    return result.astype(np.uint8)


def ensure_alpha(crop: np.ndarray) -> np.ndarray:
    """Ajoute un canal alpha si le crop est RGB."""
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if crop.shape[2] == 3:
        alpha = np.ones((*crop.shape[:2], 1), dtype=np.uint8) * 255
        crop  = np.concatenate([crop, alpha], axis=2)
    return crop


# ───────────────────────── PLACEMENT ────────────────────────────────────────

def iou(b1: tuple, b2: tuple) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1)


def draw_hand_box(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """
    Dessine une case à main levée autour d'un glyphe
    (comme dans le Codex Mendoza et la planche uploadée).
    """
    thick = random.randint(1, 3)
    # Noir (80%) ou rouge/ocre colonial (20%)
    ink   = (10, 10, 10) if random.random() < 0.80 else (20, 20, random.randint(120, 180))

    pad = random.randint(4, 14)
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = x2 + pad, y2 + pad

    # 4 côtés avec micro-irrégularité (trait à la main)
    for side in range(4):
        w1 = random.randint(-3, 3)
        w2 = random.randint(-3, 3)
        if side == 0:   # haut
            cv2.line(canvas, (bx1 + w1, by1 + w1), (bx2 + w2, by1 + w2), ink, thick)
        elif side == 1: # bas
            cv2.line(canvas, (bx1 + w1, by2 + w1), (bx2 + w2, by2 + w2), ink, thick)
        elif side == 2: # gauche
            cv2.line(canvas, (bx1 + w1, by1 + w1), (bx1 + w2, by2 + w2), ink, thick)
        else:           # droite
            cv2.line(canvas, (bx2 + w1, by1 + w1), (bx2 + w2, by2 + w2), ink, thick)


def paste_crop(canvas: np.ndarray, crop: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Colle le crop sur le canvas via son canal alpha."""
    h, w = crop.shape[:2]
    ch = y2 - y1
    cw = x2 - x1
    if ch <= 0 or cw <= 0 or h != ch or w != cw:
        return False
    alpha = crop[:, :, 3:4] / 255.0
    rgb   = crop[:, :, :3]
    roi   = canvas[y1:y2, x1:x2]
    canvas[y1:y2, x1:x2] = np.clip(rgb * alpha + roi * (1 - alpha), 0, 255).astype(np.uint8)
    return True


def generate_grid_positions(size: int, n: int) -> list:
    """
    Génère des positions en grille approximative
    (structure colonne, comme dans les codex de tribut).
    """
    cols = random.randint(2, 6)
    rows = max(1, (n + cols - 1) // cols)
    cw   = size // cols
    rh   = size // max(rows, 1)
    positions = []
    for r in range(rows):
        for c in range(cols):
            jx = random.randint(-cw // 5, cw // 5)
            jy = random.randint(-rh // 5, rh // 5)
            positions.append((c * cw + cw // 2 + jx, r * rh + rh // 2 + jy))
    random.shuffle(positions)
    return positions


# ─────────────────────── BOUCLE PRINCIPALE ──────────────────────────────────

def generate_dataset(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    CROPS_DIR  = Path(args.crops)
    LABELS_DIR = Path(args.labels)
    OUT_ROOT   = Path(args.out)
    SIZE       = args.canvas
    N          = args.n

    # Répertoires de sortie (train/val séparés dès le début)
    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Chargement
    print("Chargement des crops...")
    all_crops = load_crops(CROPS_DIR, LABELS_DIR)
    if not all_crops:
        raise RuntimeError(f"Aucun crop trouvé dans {CROPS_DIR}")

    n_train = int(N * 0.80)
    n_val   = N - n_train
    counts  = {"train": n_train, "val": n_val}
    idx     = 0

    for split, n_split in counts.items():
        print(f"\nGénération {split} : {n_split} images...")

        for i in range(n_split):
            canvas      = amate_background(SIZE)
            annotations = []
            used_boxes  = []

            # Texte latin fantôme (30% des images)
            if random.random() < 0.30:
                add_latin_text_lines(canvas, SIZE)

            # Nombre de glyphes (plus dense que v1)
            n_glyphs = random.randint(6, 22)
            samples  = random.choices(all_crops, k=n_glyphs)

            # Disposition : grille (50%) ou aléatoire (50%)
            use_grid  = random.random() < 0.50
            positions = generate_grid_positions(SIZE, n_glyphs) if use_grid else []
            pos_idx   = 0

            for (crop_path, class_id) in samples:
                crop = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
                if crop is None:
                    continue
                crop = ensure_alpha(crop)

                # Taille cible : 28–140px (plus petit que v1 = plus réaliste)
                target = random.randint(28, 140)
                h, w   = crop.shape[:2]
                scale  = target / max(h, w)
                nw     = max(1, int(w * scale))
                nh     = max(1, int(h * scale))
                crop   = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

                # Rotation légère (glyphes en cases = peu de rotation)
                angle = random.uniform(-18, 18)
                crop  = rotate_crop(crop, angle)
                nh, nw = crop.shape[:2]

                # Dégradation de l'encre
                crop = apply_ink_degradation(crop)

                # Trouver une position
                placed = False
                for attempt in range(60):
                    if use_grid and pos_idx < len(positions):
                        gx, gy = positions[pos_idx]
                        x1 = int(np.clip(gx - nw // 2, 8, SIZE - nw - 8))
                        y1 = int(np.clip(gy - nh // 2, 8, SIZE - nh - 8))
                    else:
                        x1 = random.randint(8, max(8, SIZE - nw - 8))
                        y1 = random.randint(8, max(8, SIZE - nh - 8))

                    x2 = x1 + nw
                    y2 = y1 + nh

                    # Chevauchement partiel autorisé (12%)
                    if all(iou((x1, y1, x2, y2), b) < 0.12 for b in used_boxes):
                        placed = True
                        break

                pos_idx += 1
                if not placed:
                    continue

                # Case dessinée à la main (50% des glyphes)
                if random.random() < 0.50:
                    draw_hand_box(canvas, x1, y1, x2, y2)

                # Coller le glyphe
                if not paste_crop(canvas, crop, x1, y1, x2, y2):
                    continue

                used_boxes.append((x1, y1, x2, y2))

                # Annotation YOLO normalisée
                cx_n = (x1 + x2) / 2 / SIZE
                cy_n = (y1 + y2) / 2 / SIZE
                bw_n = (x2 - x1) / SIZE
                bh_n = (y2 - y1) / SIZE
                annotations.append(
                    f"{class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}"
                )

            if not annotations:
                continue

            fname = f"synth_{idx:06d}"
            idx  += 1
            cv2.imwrite(
                str(OUT_ROOT / "images" / split / f"{fname}.jpg"),
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 93]
            )
            (OUT_ROOT / "labels" / split / f"{fname}.txt").write_text(
                "\n".join(annotations)
            )

            if i % 200 == 0:
                print(f"  [{split}] {i}/{n_split}")

    print(f"\nDataset v3 terminé : {idx} images dans {OUT_ROOT}")
    _write_yaml(OUT_ROOT)


def _write_yaml(out_root: Path) -> None:
    """Génère le dataset.yaml pour l'entraînement YOLO."""
    names = [
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

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    yaml = f"""path: {out_root.resolve()}
train: images/train
val:   images/val

nc: {len(names)}
names:
{names_yaml}
"""
    (out_root / "dataset.yaml").write_text(yaml)
    print(f"  dataset.yaml écrit ({len(names)} classes)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
