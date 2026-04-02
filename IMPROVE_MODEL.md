# Guide — Amélioration du modèle AztecVision

## Contexte

Le modèle actuel détecte **303 glyphes aztèques** sur des crops isolés (mAP50 = 0.766).
Objectif : le faire détecter sur des **planches entières** (pages de manuscrits).

---

## Prérequis

**Environnement recommandé : Google Colab (GPU T4 gratuit) ou Kaggle (30h GPU/semaine)**

```bash
pip install ultralytics opencv-python-headless
```

---

## Étape 1 — Cloner le repo et récupérer le modèle

```bash
git clone https://github.com/aryanchristopher/AztecVision.git
cd AztecVision
```

Télécharger le modèle de base depuis le repo ou demander à David de t'envoyer `best_fixed.pt` (19MB).
Le placer **à la racine du projet** :
```
AztecVision/
└── best_fixed.pt   ← ici
```

---

## Étape 2 — Récupérer le dataset original de crops

Le dataset complet (36 360 images annotées) est trop lourd pour GitHub.
Demander à David de t'envoyer `data/full_yolo_dataset.zip` ou le partager via Google Drive.

Structure attendue :
```
data/full_yolo_dataset/
├── images/train/   ← crops PNG des glyphes
└── labels/train/   ← annotations YOLO (.txt)
```

---

## Étape 3 — Générer le dataset synthétique de planches

```bash
python src/core/synthetic_compositing.py \
    --crops  data/full_yolo_dataset/images/train \
    --labels data/full_yolo_dataset/labels/train \
    --out    data/synthetic_planches \
    --n      2000
```

Ce script génère 2000 images de planches synthétiques :
- Glyphes collés sur fonds parchemin
- Annotations YOLO parfaites (pas de template matching)
- Split auto train/val (80/20)

Durée : ~5 min sur CPU, ~1 min sur GPU.

---

## Étape 4 — Lancer le fine-tuning

```bash
python src/core/finetune_planches.py
```

Paramètres clés :
| Paramètre | Valeur | Pourquoi |
|-----------|--------|----------|
| epochs | 15 | Évite le catastrophic forgetting |
| lr0 | 0.00005 | Très bas — ajuster sans écraser |
| freeze | 15 | Backbone gelé, seule la tête s'adapte |

Durée estimée :
- GPU T4 (Colab) : ~20 min
- CPU : ~2h

---

## Étape 5 — Ce que tu nous envoies

Une fois l'entraînement terminé, envoie **un seul fichier zip** :

```bash
zip aztec_finetuned_results.zip \
    models/runs/detect/finetune_planches/weights/best.pt \
    models/runs/detect/finetune_planches/results.csv \
    models/runs/detect/finetune_planches/confusion_matrix.png \
    models/runs/detect/finetune_planches/results.png
```

Taille estimée : **~25 MB**

---

## Structure du repo

```
aztec-detector/
├── best_fixed.pt                          ← modèle de base (à télécharger séparément)
├── src/core/
│   ├── synthetic_compositing.py           ← ÉTAPE 3 : générer le dataset
│   ├── finetune_planches.py               ← ÉTAPE 4 : fine-tuner le modèle
│   ├── generate_planche_dataset.py        ← (ancienne approche, pas nécessaire)
│   └── download_planches_roboflow.py      ← télécharger planches pour annotation manuelle
├── data/
│   └── synthetic_planches/               ← généré par synthetic_compositing.py
└── models/runs/detect/
    └── finetune_planches/weights/best.pt  ← résultat final à nous envoyer
```

---

## FAQ

**Q : Le modèle plante avec "CUDA not available" ?**
Colab free tier : Runtime → Change runtime type → GPU (T4)

**Q : Combien de temps ça prend ?**
~20 min sur Colab T4, ~2h sur CPU.

**Q : Le loss monte après quelques epochs ?**
Normal si `patience=5` → le training s'arrête automatiquement au meilleur epoch.

**Q : Je veux générer plus d'images synthétiques ?**
Changer `--n 2000` en `--n 5000` dans l'étape 3. Plus d'images = meilleur modèle.
