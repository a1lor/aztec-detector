import cv2
import os
import shutil
import yaml

input_dir = "data/aug_rot"
output_masks = "data/masks"
output_images = "data/images_raw"  # On copie les images ici pour l'entraînement

os.makedirs(output_masks, exist_ok=True)
os.makedirs(output_images, exist_ok=True)

# 1. Charger les noms de classes depuis dataset.yaml pour connaître leur ID (de 0 à 302)
with open("configs/dataset.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Créer un dictionnaire : {"0015-cacahuatl": 0, "0015-teocomitl": 1, ...}
class_name_to_id = {v: k for k, v in config["names"].items()}

images_trouvees = 0

# 2. Parcourir TOUS les sous-dossiers
for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            images_trouvees += 1
            img_path = os.path.join(root, filename)

            # 3. Trouver la classe (on regarde le nom du dossier parent ou le nom du fichier)
            dossier_parent = os.path.basename(root)
            class_id = class_name_to_id.get(dossier_parent)

            # Si le dossier parent n'est pas le nom de la classe, on vérifie si le nom de fichier commence par la classe
            if class_id is None:
                for cls_name, c_id in class_name_to_id.items():
                    if cls_name in filename or cls_name in img_path:
                        class_id = c_id
                        break

            if class_id is None:
                print(f"⚠️ Impossible de trouver la classe pour {img_path}")
                continue

            # 4. Générer le masque
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Seuillage pour séparer l'objet du fond (fond = 0, objet = 255)
            _, mask_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # IMPORTANT : Remplacer 255 par (class_id + 1) pour que convert_masks.py comprenne !
            # (Le fond reste à 0)
            mask_classes = (mask_binary > 0).astype('uint8') * (class_id + 1)

            # 5. Sauvegarder le masque et copier l'image originale
            # On utilise le même nom pour qu'ils soient reliés
            # On génère un nom unique au cas où plusieurs fichiers auraient le même nom
            base_name = f"{class_id}_{filename}"

            cv2.imwrite(os.path.join(output_masks, base_name), mask_classes)
            shutil.copy(img_path, os.path.join(output_images, base_name))

print(f"✅ Terminé ! {images_trouvees} images traitées et copiées dans data/images_raw, masques dans data/masks.")

