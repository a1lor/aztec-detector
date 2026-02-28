import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --- CONFIGURATION ---
# Le dossier contenant TOUS vos sous-dossiers de classes (ex: 0022-molcaxitl, etc.)
# C'est le dossier parent où vous avez copié vos fichiers précédemment.
root_dir = "/Users/davidlitvak/Desktop/Aivancity/3_Annee/PythonProject/images/0020-tziuhtli"

target_count = 150  # Objectif d'images par dossier


# --- FONCTIONS D'AUGMENTATION ---

def add_noise(image):
    """Ajoute du bruit Gaussien (aspect granuleux)"""
    img_array = np.array(image)
    # Génération du bruit
    mean = 0
    std = 15  # Intensité du bruit
    noise = np.random.normal(mean, std, img_array.shape).astype('uint8')
    # Ajout du bruit
    noisy_img = img_array + noise
    # On s'assure que les valeurs restent entre 0 et 255
    noisy_img = np.clip(noisy_img, 0, 255)
    return Image.fromarray(noisy_img.astype('uint8'))


def change_brightness(image):
    """Change la luminosité (plus clair ou plus sombre)"""
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.7, 1.3)  # Entre 70% et 130% de luminosité
    return enhancer.enhance(factor)


def change_contrast(image):
    """Change le contraste"""
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)


def apply_blur(image):
    """Applique un léger flou"""
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))


def random_augment(image):
    """Choisit une transformation au hasard parmi la liste"""
    augmentations = [add_noise, change_brightness, change_contrast, apply_blur]
    chosen_func = random.choice(augmentations)
    return chosen_func(image)


# --- EXECUTION PRINCIPALE ---

print(f"--- Début de l'équilibrage vers {target_count} images par classe ---")

# On liste tous les dossiers présents dans le root_dir
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

for folder in folders:
    folder_path = os.path.join(root_dir, folder)

    # Liste des images valides (extensions courantes)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    current_count = len(images)
    needed = target_count - current_count

    if needed > 0:
        print(f"Traitement de {folder} : {current_count} images -> Ajout de {needed} images...")

        # Si le dossier est vide, on ne peut rien faire
        if current_count == 0:
            print(f"[SKIP] Dossier vide : {folder}")
            continue

        # Boucle de génération
        for i in range(needed):
            # 1. Choisir une image source au hasard
            src_filename = random.choice(images)
            src_path = os.path.join(folder_path, src_filename)

            try:
                with Image.open(src_path) as img:
                    # Convertir en RGB pour éviter les soucis de transparence/niveaux de gris
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # 2. Appliquer une augmentation
                    aug_img = random_augment(img)

                    # 3. Sauvegarder avec un nom unique
                    # Exemple de nom : aug_noise_3_original.jpg
                    new_filename = f"aug_{i}_{src_filename}"
                    save_path = os.path.join(folder_path, new_filename)

                    aug_img.save(save_path)
            except Exception as e:
                print(f"Erreur sur {src_filename}: {e}")

    elif needed < 0:
        print(f"{folder} a déjà {current_count} images (plus que l'objectif). On ne touche pas.")
    else:
        print(f"{folder} est déjà complet (115 images).")

print("--- Terminé ! ---")