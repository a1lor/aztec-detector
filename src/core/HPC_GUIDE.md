# Guide HPC — AztecVision

## Étape 0 — Identifier vos ressources (obligatoire)

```bash
# Sur le serveur, lancer EN PREMIER :
python server_diagnostic.py
```

Copiez la sortie complète. Elle contient tout ce qu'il faut savoir :
GPUs disponibles, VRAM, partitions SLURM, scheduler utilisé.

---

## Commandes essentielles SLURM

```bash
sinfo -o '%P %G %C %m'   # Voir toutes les partitions et leurs GPUs
squeue -u $USER           # Voir vos jobs en cours
sbatch train_yolo.slurm   # Soumettre l'entraînement
scancel <JOB_ID>          # Annuler un job
sacct -j <JOB_ID>         # Voir les stats d'un job terminé
```

---

## Adapter batch size selon la VRAM du GPU

| GPU              | VRAM  | batch | workers | Durée estimée |
|------------------|-------|-------|---------|---------------|
| A100             | 80 GB | 16    | 8       | ~25 min       |
| A100             | 40 GB | 12    | 8       | ~30 min       |
| V100             | 32 GB | 10    | 6       | ~40 min       |
| RTX 3090 / 4090  | 24 GB | 8     | 6       | ~50 min       |
| V100             | 16 GB | 6     | 4       | ~60 min       |
| T4 / P100        | 16 GB | 4     | 4       | ~90 min       |
| RTX 3080         |  8 GB | 2     | 2       | ~3h           |

Modifier `--batch` dans `train_yolo.slurm` selon le GPU obtenu.

---

## Si vous avez plusieurs GPUs

Modifier dans `train_yolo.slurm` :
```bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
```

Et dans `finetune_planches_v3.py`, changer `device` :
```python
device="0,1"   # 2 GPUs
device="0,1,2,3"  # 4 GPUs
```

YOLOv8 gère le multi-GPU nativement via DDP (Distributed Data Parallel).
Avec 2 GPUs, le batch effectif double et le temps est divisé par ~1.8.

---

## Workflow complet recommandé

```bash
# 1. Diagnostic (interactif, ~30 sec)
python server_diagnostic.py

# 2. Générer le dataset (job CPU, ~10-30 min)
sbatch generate_dataset.slurm

# 3. Vérifier que le dataset est prêt
squeue -u $USER
ls data/synthetic_v3/images/train/ | wc -l   # doit afficher ~3200

# 4. Lancer l'entraînement (job GPU, 25-90 min selon GPU)
sbatch train_yolo.slurm

# 5. Suivre en temps réel
tail -f logs/train_<JOB_ID>.out

# 6. Récupérer les résultats
ls runs/finetune/planches_v3_hpc/weights/
```

---

## Transfert des fichiers vers/depuis le serveur

```bash
# Envoyer le modèle de base et les scripts
scp best_fixed.pt user@serveur.univ.fr:~/aztec-detector/
scp *.py *.slurm user@serveur.univ.fr:~/aztec-detector/

# Envoyer le dataset (si non dispo sur le serveur)
rsync -avz --progress data/full_yolo_dataset/ \
      user@serveur.univ.fr:~/aztec-detector/data/full_yolo_dataset/

# Récupérer les résultats après entraînement
scp user@serveur.univ.fr:~/aztec-detector/aztec_finetuned_hpc_*.zip .
```

---

## Si votre cluster n'utilise pas SLURM

| Scheduler | Commande de soumission |
|-----------|----------------------|
| SLURM     | `sbatch train_yolo.slurm` |
| PBS/Torque| `qsub train_yolo.pbs` |
| SGE       | `qsub -cwd train_yolo.sge` |
| LSF       | `bsub < train_yolo.lsf` |

Demandez à votre admin : `echo $SCHEDULER` ou `which sbatch qsub bsub`.

---

## Problèmes fréquents

**CUDA not available dans le job**
```bash
module load cuda/11.8   # Ajouter en tête du script .slurm
```

**Out of memory (OOM)**
Réduire `--batch` de moitié dans `train_yolo.slurm`.

**Job killed après X heures**
Augmenter `--time=12:00:00` ou activer `save_period=5` (déjà fait dans v3).

**Partition gpu introuvable**
```bash
sinfo -o '%P %G' | grep -i gpu   # Trouver le vrai nom de la partition GPU
```
