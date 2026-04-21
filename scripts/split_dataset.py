#!/usr/bin/env python3
"""
split_dataset.py — Split images + labels en train/val pour YOLO.
Crée la structure :
  data/
    images/train/  images/val/
    labels/train/  labels/val/
Usage:
  python split_dataset.py \
    --images <dossier_images> \
    --labels <dossier_labels> \
    --output <dossier_data> \
    --val-ratio 0.2 \
    --seed 42
"""
import argparse
import random
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    copy: bool = True,
):
    """
    Split les images et labels en train/val.
    Args:
        images_dir: Dossier source des images.
        labels_dir: Dossier source des labels .txt.
        output_dir: Dossier de sortie (data/).
        val_ratio: Proportion de validation (0.0 à 1.0).
        seed: Graine aléatoire pour la reproductibilité.
        copy: Si True, copie les fichiers. Si False, crée des symlinks.
    """
    # Créer les dossiers de sortie
    dirs = {
        "train_images": output_dir / "images" / "train",
        "val_images": output_dir / "images" / "val",
        "train_labels": output_dir / "labels" / "train",
        "val_labels": output_dir / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    # Lister les images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions
    )
    if not image_files:
        print(f"❌ Aucune image trouvée dans {images_dir}")
        sys.exit(1)
    # Vérifier que chaque image a un label
    paired = []
    missing_labels = 0
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            paired.append((img_path, label_path))
        else:
            missing_labels += 1
    if missing_labels > 0:
        print(f"⚠️  {missing_labels} images n'ont pas de label correspondant (ignorées)")
    print(f"📁 {len(paired)} paires image/label trouvées")
    # Shuffle et split
    random.seed(seed)
    random.shuffle(paired)
    val_count = int(len(paired) * val_ratio)
    val_set = paired[:val_count]
    train_set = paired[val_count:]
    print(f"📊 Split : {len(train_set)} train / {len(val_set)} val")
    # Fonction de transfert
    transfer_fn = shutil.copy2 if copy else os.symlink
    def transfer_files(file_pairs, img_dest, lbl_dest, desc):
        for img_path, lbl_path in tqdm(file_pairs, desc=desc):
            transfer_fn(str(img_path), str(img_dest / img_path.name))
            transfer_fn(str(lbl_path), str(lbl_dest / lbl_path.name))
    # Transférer
    transfer_files(
        train_set, dirs["train_images"], dirs["train_labels"], "Train"
    )
    transfer_files(
        val_set, dirs["val_images"], dirs["val_labels"], "Val"
    )
    # Résumé
    print("\n" + "=" * 50)
    print("✅ SPLIT TERMINÉ")
    print("=" * 50)
    print(f"  Train : {len(train_set)} images → {dirs['train_images']}")
    print(f"  Val   : {len(val_set)} images → {dirs['val_images']}")
    print(f"  Labels train → {dirs['train_labels']}")
    print(f"  Labels val   → {dirs['val_labels']}")
    return len(train_set), len(val_set)
def main():
    parser = argparse.ArgumentParser(
        description="Split images + labels en train/val pour YOLO"
    )
    parser.add_argument(
        "--images", type=Path, required=True, help="Dossier source des images"
    )
    parser.add_argument(
        "--labels", type=Path, required=True, help="Dossier source des labels .txt"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Dossier de sortie (data/)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proportion de validation (défaut: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Graine aléatoire (défaut: 42)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Utiliser des symlinks au lieu de copies (économise l'espace disque)",
    )
    args = parser.parse_args()
    split_dataset(
        args.images,
        args.labels,
        args.output,
        args.val_ratio,
        args.seed,
        copy=not args.symlink,
    )
if __name__ == "__main__":
    main()
