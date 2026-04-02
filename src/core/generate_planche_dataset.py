#!/usr/bin/env python3
"""
generate_planche_dataset.py
Pipeline to generate YOLO training data from CEN website.

Steps:
  1. Query API for each of the 303 class names → get glyph crops + planche refs
  2. Discover codex path (codexId → directory)
  3. Download glyph crop (from theme)
  4. Download planche image (from cote → codif)
  5. Template matching → bounding box in planche
  6. Save YOLO annotations

Usage:
  python src/core/generate_planche_dataset.py --out data/planche_dataset
"""
import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
API_URL   = "https://cen2.sup-infor.com/Systeme/getCEN.json"
IMG_BASE  = "https://cen.sup-infor.com"
COND      = json.dumps({
    "mode": "start", "type": "element", "glyphe": "lecture",
    "element": "designation", "multiple": [], "elements": False
})

# Known codexId → directory mapping (discovered empirically)
CODEX_DIRS = {
    "1":  "/Home/2/CEN/Codex/t_010/",
    "2":  "/Home/2/CEN/Codex/t_026/",
    "3":  "/Home/2/CEN/Codex/t_027/",
    "4":  "/Home/2/CEN/Codex/t_028/",
    "5":  "/Home/2/CEN/Codex/t_030/",
    "6":  "/Home/2/CEN/Codex/t_032/",
    "7":  "/Home/2/CEN/Codex/t_033/",
    "8":  "/Home/2/CEN/Codex/t_035/",
    "9":  "/Home/2/CEN/Codex/t_072/",
    "10": "/Home/2/CEN/Codex/t_073/",
    "11": "/Home/2/CEN/Codex/t_075/",
    "12": "/Home/2/CEN/Codex/t_082/",
    "13": "/Home/2/CEN/Codex/t_108/",
    "14": "/Home/2/CEN/Codex/t_115/",
    "15": "/Home/2/CEN/Codex/t_116/",
    "16": "/Home/2/CEN/Codex/t_374/",
    "17": "/Home/2/CEN/Codex/t_376/",
    "18": "/Home/2/CEN/Codex/t_385a/",
    "42": "/Home/2/CEN/Codex/t_391/",
    "43": "/Home/2/CEN/Codex/t_392a/",
    "44": "/Home/2/CEN/Codex/t_393b/",
    "45": "/Home/2/CEN/Codex/t_394/",
    "46": "/Home/2/CEN/Codex/t_395/",
    "47": "/Home/2/CEN/Codex/t_396/",
    "48": "/Home/2/CEN/Codex/t_393a/",
    "49": "/Home/2/CEN/Codex/t_397/",
    "50": "/Home/2/CEN/Codex/t_398/",
}

# 387 codices: folio ranges → sub-directory
# Discovered by testing URLs; key = folio prefix (int), value = directory
CODEX_387_RANGES = [
    (0,    "/Home/2/CEN/Codex/t_387_01/"),
    (420,  "/Home/2/CEN/Codex/t_387_02/"),
    (450,  "/Home/2/CEN/Codex/t_387_03/"),
    (505,  "/Home/2/CEN/Codex/t_387_04/"),
    (540,  "/Home/2/CEN/Codex/t_387_05/"),
    (555,  "/Home/2/CEN/Codex/t_387_06/"),
    (600,  "/Home/2/CEN/Codex/t_387_07/"),
    (640,  "/Home/2/CEN/Codex/t_387_08/"),
    (660,  "/Home/2/CEN/Codex/t_387_09/"),
    (700,  "/Home/2/CEN/Codex/t_387_10/"),
    (750,  "/Home/2/CEN/Codex/t_387_11/"),
    (780,  "/Home/2/CEN/Codex/t_387_12/"),
    (810,  "/Home/2/CEN/Codex/t_387_13/"),
    (835,  "/Home/2/CEN/Codex/t_387_14/"),
    (860,  "/Home/2/CEN/Codex/t_387_15/"),
    (880,  "/Home/2/CEN/Codex/t_387_16/"),
    (900,  "/Home/2/CEN/Codex/t_387_17/"),
    (9999, "/Home/2/CEN/Codex/t_387_18/"),
]

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_387_dir(cote: str) -> str:
    """Infer t_387_XX directory from folio number in cote."""
    import re
    m = re.match(r"387_(\d+)", cote)
    if not m:
        return "/Home/2/CEN/Codex/t_387_01/"
    folio = int(m.group(1))
    chosen = CODEX_387_RANGES[0][1]
    for start, path in CODEX_387_RANGES:
        if folio >= start:
            chosen = path
    return chosen


def get_directory(codex_id: str, cote: str):
    if codex_id in CODEX_DIRS:
        return CODEX_DIRS[codex_id]
    if cote.startswith("387_"):
        return get_387_dir(cote)
    return None


def fetch_url(url: str, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read()
        except Exception:
            time.sleep(1)
    return None


def api_query(element_name: str) -> list:
    payload = json.dumps({
        "mode": "tlachia", "list": False, "codex": "",
        "word": element_name, "cond": COND
    }).encode()
    req = urllib.request.Request(
        API_URL, data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
            return [e for e in data.get("elements", []) if e["element"] == element_name]
    except Exception as e:
        print(f"  API error for {element_name}: {e}")
        return []


def cote_to_planche_name(cote: str) -> str:
    """Extract planche codif name from cote (remove last segment)."""
    # Try underscore split first
    idx = cote.rfind("_")
    if idx == -1:
        idx = cote.rfind(".")
    if idx == -1:
        return cote + "$c"
    return cote[:idx].lower() + "$c"


def template_match(planche_bgr: np.ndarray, crop_bgr: np.ndarray,
                   threshold=0.55):
    """
    Find crop in planche using multi-scale template matching.
    Returns (x1, y1, x2, y2) in pixels or None.
    """
    ph, pw = planche_bgr.shape[:2]
    ch, cw = crop_bgr.shape[:2]

    if ch > ph or cw > pw:
        return None

    planche_gray = cv2.cvtColor(planche_bgr, cv2.COLOR_BGR2GRAY)
    crop_gray    = cv2.cvtColor(crop_bgr,    cv2.COLOR_BGR2GRAY)

    best_val, best_loc, best_scale = -1, None, 1.0

    for scale in [1.0, 0.9, 0.8, 0.7, 1.1, 1.2, 1.3]:
        sw = max(8, int(cw * scale))
        sh = max(8, int(ch * scale))
        if sw > pw or sh > ph:
            continue
        resized = cv2.resize(crop_gray, (sw, sh))
        result  = cv2.matchTemplate(planche_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(result)
        if val > best_val:
            best_val, best_loc, best_scale = val, loc, scale

    if best_val < threshold or best_loc is None:
        return None

    sw = int(cw * best_scale)
    sh = int(ch * best_scale)
    x1, y1 = best_loc
    x2, y2 = x1 + sw, y1 + sh
    return (x1, y1, x2, y2)


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


# ── Main ──────────────────────────────────────────────────────────────────────

def main(out_dir: Path, max_per_class: int, match_threshold: float):
    images_dir = out_dir / "images" / "train"
    labels_dir = out_dir / "labels" / "train"
    crops_dir  = out_dir / "_crops_cache"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "matched": 0, "skipped": 0}
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    for class_name in CLASS_NAMES:
        class_idx = class_to_idx[class_name]
        print(f"\n[{class_idx:3d}] {class_name}")

        elements = api_query(class_name)
        if not elements:
            print("  → no results")
            continue

        matched_count = 0
        for elem in elements:
            if matched_count >= max_per_class:
                break

            codex_id = elem["codexId"]
            cote     = elem["cote"]
            theme    = elem["theme"].replace(".", "_").lower()
            stats["total"] += 1

            directory = get_directory(str(codex_id), cote)
            if not directory:
                print(f"  ✗ unknown codexId={codex_id}")
                stats["skipped"] += 1
                continue

            # ── Download crop ────────────────────────────────────────────────
            crop_url  = IMG_BASE + directory + cote.lower() + ".jpg.limit.800x800.jpg"
            crop_data = fetch_url(crop_url)
            if not crop_data or len(crop_data) < 200:
                print(f"  ✗ crop not found: {crop_url}")
                stats["skipped"] += 1
                continue

            crop_arr = np.frombuffer(crop_data, np.uint8)
            crop_bgr = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
            if crop_bgr is None:
                stats["skipped"] += 1
                continue

            # ── Download planche ─────────────────────────────────────────────
            planche_base = cote[:cote.rfind("_")] if "_" in cote else cote
            planche_url  = IMG_BASE + directory + planche_base.lower() + ".jpg.limit.800x800.jpg"
            planche_data = fetch_url(planche_url)
            if not planche_data or len(planche_data) < 200:
                print(f"  ✗ planche not found: {planche_url}")
                stats["skipped"] += 1
                continue

            planche_arr = np.frombuffer(planche_data, np.uint8)
            planche_bgr = cv2.imdecode(planche_arr, cv2.IMREAD_COLOR)
            if planche_bgr is None:
                stats["skipped"] += 1
                continue

            ph, pw = planche_bgr.shape[:2]

            # ── Template matching ─────────────────────────────────────────────
            bbox = template_match(planche_bgr, crop_bgr, threshold=match_threshold)
            if bbox is None:
                print(f"  ✗ no match  cote={cote}")
                stats["skipped"] += 1
                continue

            x1, y1, x2, y2 = bbox

            # ── Save planche image ────────────────────────────────────────────
            img_name   = f"{class_name}_{cote.lower().replace('/','_')}.jpg"
            img_path   = images_dir / img_name
            label_path = labels_dir / img_name.replace(".jpg", ".txt")

            # Append annotation if planche already saved (multiple glyphs per planche)
            if not img_path.exists():
                cv2.imwrite(str(img_path), planche_bgr)

            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, pw, ph)
            with open(label_path, "a") as f:
                f.write(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            print(f"  ✓ {cote}  conf={bbox[0] if isinstance(bbox[0], float) else '?'}  "
                  f"bbox=({x1},{y1},{x2},{y2})")
            stats["matched"] += 1
            matched_count += 1
            time.sleep(0.05)  # polite scraping

    # ── Write dataset.yaml ───────────────────────────────────────────────────
    yaml_path = out_dir / "dataset.yaml"
    names_str = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir.resolve()}\n")
        f.write("train: images/train\nval: images/train\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\nnames:\n{names_str}\n")

    print(f"\n{'='*50}")
    print(f"Total queries : {stats['total']}")
    print(f"Matched       : {stats['matched']}")
    print(f"Skipped       : {stats['skipped']}")
    print(f"Dataset saved : {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",       default="data/planche_dataset",
                        help="Output directory")
    parser.add_argument("--max",       type=int, default=20,
                        help="Max annotations per class (default 20)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Template match confidence threshold (default 0.55)")
    args = parser.parse_args()
    main(Path(args.out), args.max, args.threshold)
