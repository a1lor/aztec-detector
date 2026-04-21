import os

# Chemin vers le dossier contenant tes classes (dossiers)
folder_path = r"C:\Users\Pablo\pythonProject\CodexClinic\data\aug_rot"  # <-- modifie si besoin

# Liste tous les dossiers dans le dossier principal
class_names = [name for name in os.listdir(folder_path)
               if os.path.isdir(os.path.join(folder_path, name))]

# Trie la liste pour correspondre à ton ordre (si nécessaire)
class_names.sort()

# Exemple de "placeholders" à remplacer
placeholders = [f"classe_{i:03d}" for i in range(1, len(class_names)+1)]

# Création d'un dictionnaire de correspondance : placeholder -> nom réel
mapping = dict(zip(placeholders, class_names))

# Exemple : liste à remplacer (ici, on simule la liste générique)
old_list = placeholders.copy()  # ou ta liste existante avec classe_001 etc.

# Remplacement automatique
new_list = [mapping.get(item, item) for item in old_list]

# Affiche le résultat
for i, name in enumerate(new_list):
    print(f"{i}: {name}")

# Optionnel : sauvegarde dans un fichier
with open("classe_remplacees.txt", "w") as f:
    for name in new_list:
        f.write(name + "\n")

print(f"\n{len(new_list)} classes remplacées et sauvegardées dans 'classe_remplacees.txt'")
