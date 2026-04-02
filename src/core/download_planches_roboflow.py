#!/usr/bin/env python3
"""
download_planches_roboflow.py
Télécharge ~80 planches uniques depuis le CEN pour annotation manuelle sur Roboflow.
Choisit des planches variées (plusieurs codex, plusieurs classes).

Usage:
    python src/core/download_planches_roboflow.py --out data/planches_roboflow
"""
import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

API_URL  = "https://cen2.sup-infor.com/Systeme/getCEN.json"
IMG_BASE = "https://cen.sup-infor.com"

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

COND = json.dumps({
    "mode": "start", "type": "element", "glyphe": "lecture",
    "element": "designation", "multiple": [], "elements": False
})

# Sample classes spread across the 303 to get diverse planches
SAMPLE_CLASSES = [
    "atl", "tetl", "calli", "tepetl", "xochitl", "cuauhtli", "cohuatl",
    "tlalli", "ocelotl", "quetzalli", "acatl", "tecpatl", "tochtli",
    "mazatl", "ehecatl", "tonatiuh", "metztli", "citlalin", "olin",
    "maitl_1", "icpalli", "petlatl", "macuahuitl", "chimalli", "tilmatli",
    "comitl", "caxitl", "mecatl", "ayatl", "chalchihuitl", "tletl",
    "omitl_1", "yollotl", "cueitl", "huipilli", "metl", "nopalli",
    "nochtli", "chilli", "etl", "tamalli", "octli", "tlaxcalli",
    "amatl", "teocuitlatl", "tepoztli", "malacatl", "huehuetl",
    "chapolin", "zolin", "miztli", "coyotl", "itzcuintli", "ozomatli",
    "totolin", "michin", "cuetzpalin", "cipactli", "xolotl", "papalotl",
]


def fetch_url(url, retries=3):
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read()
        except Exception:
            time.sleep(1)
    return None


def api_query(element_name):
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
        print(f"  API error: {e}")
        return []


def main(out_dir: Path, max_planches: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_planches = set()   # avoid duplicates
    downloaded = 0
    failed = 0

    for class_name in SAMPLE_CLASSES:
        if downloaded >= max_planches:
            break

        print(f"\n[{downloaded}/{max_planches}] Querying: {class_name}")
        elements = api_query(class_name)
        if not elements:
            continue

        for elem in elements:
            if downloaded >= max_planches:
                break

            codex_id = str(elem["codexId"])
            cote     = elem["cote"]

            directory = CODEX_DIRS.get(codex_id)
            if not directory:
                continue

            # Build planche URL
            planche_base = cote[:cote.rfind("_")] if "_" in cote else cote
            planche_key  = directory + planche_base.lower()

            if planche_key in seen_planches:
                continue
            seen_planches.add(planche_key)

            planche_url = IMG_BASE + directory + planche_base.lower() + ".jpg.limit.800x800.jpg"
            data = fetch_url(planche_url)

            if not data or len(data) < 500:
                failed += 1
                continue

            # Save with a clean filename: codex_planche.jpg
            fname = f"{directory.strip('/').split('/')[-1]}_{planche_base.lower().replace('/','_')}.jpg"
            fpath = out_dir / fname
            fpath.write_bytes(data)

            downloaded += 1
            print(f"  ✓ [{downloaded}] {fname}  ({len(data)//1024}KB)")
            time.sleep(0.1)

    print(f"\n{'='*50}")
    print(f"✅ Downloaded : {downloaded} planches")
    print(f"✗  Failed     : {failed}")
    print(f"📁 Saved to   : {out_dir}")
    print(f"\nUpload the '{out_dir}' folder to Roboflow to start annotating.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/planches_roboflow",
                        help="Output folder")
    parser.add_argument("--max", type=int, default=80,
                        help="Max number of planches to download (default 80)")
    args = parser.parse_args()
    main(Path(args.out), args.max)
