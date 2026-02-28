import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --- CONFIGURATION ---
# Chemin corrigé (sans l'espace)
dataset_dir = '/Users/davidlitvak/Desktop/Aivancity/3_Annee/Projet Aztec/data/processed /aug_rot'
target_count = 150  # Objectif strict


# --- FONCTIONS DE TRANSFORMATION (Bruit & Pixels) ---

def add_noise(image):
    """Ajoute du 'grain' à l'image"""
    # Conversion en tableau de nombres
    img_array = np.array(image)

    # Création du bruit (écart-type de 15 pour que ce soit visible mais pas destructeur)
    noise = np.random.normal(loc=0, scale=15, size=img_array.shape)

    # Addition image + bruit
    noisy_img = img_array + noise

    # On s'assure que les valeurs de pixel restent entre 0 et 255
    noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')

    return Image.fromarray(noisy_img)


def change_brightness(image):
    """Change la luminosité (plus clair ou plus sombre)"""
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.6, 1.4)  # Entre 60% et 140%
    return enhancer.enhance(factor)


def change_contrast(image):
    """Change le contraste (couleurs vives ou ternes)"""
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(0.6, 1.4)
    return enhancer.enhance(factor)


def apply_blur(image):
    """Applique un léger flou"""
    # Rayon aléatoire entre 0.5 et 1.5 pixels
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))


def apply_random_augmentation(image):
    """Choisit une transformation au hasard"""
    methods = [add_noise, change_brightness, change_contrast, apply_blur]
    chosen_method = random.choice(methods)
    return chosen_method(image)


# --- EXECUTION PRINCIPALE ---

if not os.path.exists(dataset_dir):
    print(f"ERREUR : Le dossier n'existe pas : {dataset_dir}")
    exit()

print(f"--- Démarrage de l'équilibrage à {target_count} images par dossier ---")

for subfolder in os.listdir(dataset_dir):
    subfolder_path = os.path.join(dataset_dir, subfolder)

    # On vérifie que c'est un vrai dossier
    if os.path.isdir(subfolder_path):
        # Extensions valides
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(valid_ext)]
        num_images = len(images)

        # Si le dossier est vide, on ne peut rien faire (pas d'image source)
        if num_images == 0:
            print(f"[{subfolder}] : 0 image -> Ignoré (impossible de créer du bruit sur du vide).")
            continue

        # CAS 1 : Il manque des images -> On ajoute du bruit/pixels
        if num_images < target_count:
            needed = target_count - num_images
            print(f"[{subfolder}] : {num_images} images -> Ajout de {needed} images (Bruit/Pixels)...")

            for i in range(needed):
                try:
                    # 1. Choisir une image source existante au hasard
                    src_name = random.choice(images)
                    src_path = os.path.join(subfolder_path, src_name)

                    with Image.open(src_path) as img:
                        # Convertir en RGB pour éviter les erreurs (ex: PNG transparent)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # 2. Appliquer une transformation
                        new_img = apply_random_augmentation(img)

                        # 3. Sauvegarder
                        # Nom unique : aug_noise_3_nomOriginal.jpg
                        new_name = f"aug_pixel_{i}_{src_name}"
                        save_path = os.path.join(subfolder_path, new_name)
                        new_img.save(save_path)

                except Exception as e:
                    print(f"Erreur sur {src_name} : {e}")

        # CAS 2 : Trop d'images -> On supprime
        elif num_images > target_count:
            to_remove = num_images - target_count
            print(f"[{subfolder}] : {num_images} images -> Suppression de {to_remove} images aléatoires...")

            files_to_delete = random.sample(images, to_remove)
            for f in files_to_delete:
                try:
                    os.remove(os.path.join(subfolder_path, f))
                except OSError as e:
                    print(f"Erreur suppression : {e}")

        # CAS 3 : Parfait
        else:
            print(f"[{subfolder}] : Déjà 150 images. OK.")

print("--- Terminé ! Tous les dossiers contiennent maintenant 150 images. ---")