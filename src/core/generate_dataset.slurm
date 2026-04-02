#!/bin/bash
# =============================================================================
# generate_dataset.slurm
# Génération du dataset synthétique en parallèle sur CPU
#
# La génération des 4000 images est divisée en N chunks parallèles.
# Sur HPC avec 32 CPUs, ce qui prendrait 30min en local → ~4min.
#
# Usage :
#   sbatch generate_dataset.slurm
# =============================================================================

#SBATCH --job-name=aztec_datagen
#SBATCH --partition=cpu              # Partition CPU (pas besoin de GPU ici)
#SBATCH --cpus-per-task=16          # Beaucoup de CPUs pour la parallélisation
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/datagen_%j.out
#SBATCH --error=logs/datagen_%j.err

mkdir -p logs

echo "Génération dataset synthétique v3"
echo "CPUs : $SLURM_CPUS_PER_TASK"
echo "Début : $(date)"

# Activer l'environnement
source $HOME/aztec_env/bin/activate 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true

# Génération avec tous les CPUs disponibles
python3 synthetic_compositing_v3.py \
    --crops  data/full_yolo_dataset/images/train \
    --labels data/full_yolo_dataset/labels/train \
    --out    data/synthetic_v3 \
    --n      4000 \
    --canvas 1280

echo "Fin : $(date)"
echo "Dataset prêt dans data/synthetic_v3/"
du -sh data/synthetic_v3/
