#!/usr/bin/env python3
"""
convert_masks.py — Convertit des masques PNG en labels YOLO segmentation (.txt)
Chaque masque PNG contient les classes encodées par la valeur des pixels :
  - pixel == 0  → fond (ignoré)
  - pixel == k  → classe (k - 1)   (car YOLO commence à 0)
Le script extrait les contours de chaque classe présente dans le masque,
normalise les coordonnées, et écrit un fichier .txt au format YOLO segmentation :
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
Usage:
  python convert_masks.py --images <dossier_images> --masks <dossier_masques> --output <dossier_labels>
Options:
  --images     Dossier contenant les images originales (pour les dimensions)
  --masks      Dossier contenant les masques PNG
  --output     Dossier de sortie pour les labels .txt
  --min-area   Aire minimale (en pixels) pour retenir un contour (défaut: 100)
  --epsilon    Facteur d'approximation des contours (défaut: 0.001)
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
def extract_polygons_from_mask(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    min_area: float = 100,
    epsilon_factor: float = 0.001,
) -> list[tuple[int, list[float]]]:
    """
    Extrait les polygones normalisés depuis un masque multi-classes.
    Args:
        mask: Masque numpy (H, W) avec les valeurs de classe (0 = fond).
        img_h: Hauteur de l'image originale.
        img_w: Largeur de l'image originale.
        min_area: Aire minimale pour conserver un contour.
        epsilon_factor: Facteur pour cv2.approxPolyDP (plus petit = plus fidèle).
    Returns:
        Liste de tuples (class_id, [x1, y1, x2, y2, ...]) normalisés [0, 1].
    """
    polygons = []
    unique_classes = np.unique(mask)
    for class_val in unique_classes:
        if class_val == 0:
            # 0 = fond, on l'ignore
            continue
        # Masque binaire pour cette classe
        binary = (mask == class_val).astype(np.uint8) * 255
        # Trouver les contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            # Approximation du contour pour réduire le nombre de points
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Il faut au minimum 3 points pour un polygone valide
            if len(approx) < 3:
                continue
            # Normaliser les coordonnées
            coords = []
            for point in approx:
                x = point[0][0] / img_w
                y = point[0][1] / img_h
                # Clamp entre 0 et 1
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                coords.extend([x, y])
            # class_id = class_val - 1 (YOLO est 0-indexed, et 0 du masque = fond)
            class_id = int(class_val) - 1
            polygons.append((class_id, coords))
    return polygons
def write_yolo_label(filepath: Path, polygons: list[tuple[int, list[float]]]):
    """Écrit les polygones au format YOLO segmentation dans un fichier .txt."""
    with open(filepath, "w") as f:
        for class_id, coords in polygons:
            coords_str = " ".join(f"{c:.6f}" for c in coords)
            f.write(f"{class_id} {coords_str}\n")
def process_dataset(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    min_area: float = 100,
    epsilon_factor: float = 0.001,
):
    """
    Traite l'ensemble du dataset : convertit les masques PNG en labels YOLO.
    Les fichiers masques doivent avoir le même nom de base que les images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Trouver toutes les images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions
    )
    if not image_files:
        print(f"❌ Aucune image trouvée dans {images_dir}")
        sys.exit(1)
    print(f"📁 {len(image_files)} images trouvées dans {images_dir}")
    print(f"📝 Labels de sortie → {output_dir}")
    stats = {"total": 0, "success": 0, "skipped": 0, "errors": 0}
    class_counts = {}
    for img_path in tqdm(image_files, desc="Conversion"):
        stats["total"] += 1
        # Chercher le masque correspondant (même nom, peut avoir une extension diff)
        mask_path = None
        for ext in [".png", ".PNG"]:
            candidate = masks_dir / f"{img_path.stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            tqdm.write(f"⚠️  Masque non trouvé pour {img_path.name}")
            stats["skipped"] += 1
            continue
        try:
            # Lire le masque en niveaux de gris (ou channel unique)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Essayer en couleur puis convertir
                mask_color = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
                if mask_color is not None:
                    # Utiliser le canal rouge comme identifiant de classe
                    mask = mask_color[:, :, 2]
                else:
                    tqdm.write(f"❌ Impossible de lire le masque {mask_path.name}")
                    stats["errors"] += 1
                    continue
            # Lire l'image pour ses dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                tqdm.write(f"❌ Impossible de lire l'image {img_path.name}")
                stats["errors"] += 1
                continue
            img_h, img_w = img.shape[:2]
            # Si le masque n'a pas la même taille, le redimensionner
            if mask.shape[:2] != (img_h, img_w):
                mask = cv2.resize(
                    mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                )
            # Extraire les polygones
            polygons = extract_polygons_from_mask(
                mask, img_h, img_w, min_area, epsilon_factor
            )
            # Compter les classes
            for class_id, _ in polygons:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            # Écrire le label
            label_path = output_dir / f"{img_path.stem}.txt"
            write_yolo_label(label_path, polygons)
            # Même si le masque est tout noir (aucun objet), on crée un fichier vide
            # car YOLO en a besoin comme image de fond
            stats["success"] += 1
        except Exception as e:
            tqdm.write(f"❌ Erreur pour {img_path.name}: {e}")
            stats["errors"] += 1
    # Rapport final
    print("\n" + "=" * 50)
    print("📊 RAPPORT DE CONVERSION")
    print("=" * 50)
    print(f"  Total images     : {stats['total']}")
    print(f"  ✅ Converties     : {stats['success']}")
    print(f"  ⚠️  Ignorées      : {stats['skipped']}")
    print(f"  ❌ Erreurs        : {stats['errors']}")
    print(f"  📦 Classes uniques: {len(class_counts)}")
    if class_counts:
        print("\n  Top 10 classes (par nb d'instances) :")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls_id, count in sorted_classes[:10]:
            print(f"    Classe {cls_id:>3d} : {count} instances")
    return stats
def main():
    parser = argparse.ArgumentParser(
        description="Convertit les masques PNG en labels YOLO segmentation"
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Dossier contenant les images originales",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="Dossier contenant les masques PNG",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Dossier de sortie pour les labels .txt",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=100,
        help="Aire minimale (pixels²) pour retenir un contour (défaut: 100)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Facteur d'approximation des contours (défaut: 0.001)",
    )
    args = parser.parse_args()
    if not args.images.is_dir():
        print(f"❌ Le dossier images n'existe pas : {args.images}")
        sys.exit(1)
    if not args.masks.is_dir():
        print(f"❌ Le dossier masques n'existe pas : {args.masks}")
        sys.exit(1)
    process_dataset(args.images, args.masks, args.output, args.min_area, args.epsilon)
if __name__ == "__main__":
    main()