import os

# --- CONFIGURATION ---
# J'ai corrigé l'espace en trop ici : "processed/aug_rot" au lieu de "processed /aug_rot"
dataset_dir = '/Users/davidlitvak/Desktop/Aivancity/3_Annee/Projet Aztec/data/processed /aug_rot'

target_count = 150

# --- VERIFICATION ---
if not os.path.exists(dataset_dir):
    print(f"ERREUR : Le dossier est introuvable : {dataset_dir}")
    print("Vérifiez qu'il n'y a pas d'espace en trop dans le nom des dossiers.")
    exit()

print(f"--- Vérification du dataset : Objectif {target_count} images par classe ---\n")

all_good = True
total_folders = 0
error_folders = 0

# On récupère la liste des dossiers triée par ordre alphabétique
folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])

for folder in folders:
    folder_path = os.path.join(dataset_dir, folder)

    # On compte les images (extensions valides uniquement, pour éviter les fichiers cachés)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
    count = len(images)
    total_folders += 1

    if count == target_count:
        # Tout est OK
        print(f"✅ {folder} : {count}")
    else:
        # Il y a un problème
        diff = count - target_count
        if diff < 0:
            print(f"❌ {folder} : {count} (Manque {abs(diff)} images)")
        else:
            print(f"⚠️ {folder} : {count} (Trop de {diff} images)")
        all_good = False
        error_folders += 1

print("\n" + "-" * 40)
if all_good:
    print(f"🎉 PARFAIT ! Les {total_folders} dossiers contiennent exactement {target_count} images.")
else:
    print(f"🚫 ATTENTION : {error_folders} dossiers sur {total_folders} ne sont pas corrects.")
print("-" * 40)