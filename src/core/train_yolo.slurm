#!/bin/bash
# =============================================================================
# train_yolo.slurm
# Script SLURM pour entraîner YOLOv8 sur cluster universitaire HPC
#
# Usage :
#   sbatch train_yolo.slurm
#
# Adapter les lignes #SBATCH selon votre cluster.
# Si vous ne connaissez pas les noms de partitions : sinfo -o '%P %G'
# =============================================================================

# ── Ressources demandées ─────────────────────────────────────────────────────
#SBATCH --job-name=aztec_yolo
#SBATCH --partition=gpu              # À adapter : sinfo pour voir les partitions
#SBATCH --gres=gpu:1                 # 1 GPU (mettre gpu:2 si plusieurs dispo)
#SBATCH --cpus-per-task=8            # CPU pour le dataloader (4 par GPU minimum)
#SBATCH --mem=32G                    # RAM système (dataset en mémoire)
#SBATCH --time=08:00:00             # Durée max (8h, ajuster selon quota)
#SBATCH --output=logs/train_%j.out  # Log standard
#SBATCH --error=logs/train_%j.err   # Log erreurs
#SBATCH --mail-type=END,FAIL        # Notification email fin/erreur
#SBATCH --mail-user=votre@email.com # Votre email

# ── Optionnel : plusieurs GPUs ───────────────────────────────────────────────
# #SBATCH --gres=gpu:2
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=64G

# ── Optionnel : partition spécifique ────────────────────────────────────────
# #SBATCH --partition=gpu-v100
# #SBATCH --partition=gpu-a100
# #SBATCH --constraint=v100          # Forcer un type de GPU

# ── Début du job ─────────────────────────────────────────────────────────────
mkdir -p logs

echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "GPUs      : $SLURM_GPUS_ON_NODE"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Démarrage : $(date)"
echo "========================================"

# ── Modules HPC (adapter selon votre cluster) ────────────────────────────────
# Décommenter les lignes correspondant à votre système :

# --- CUDA (obligatoire) ---
module load cuda/11.8         # ou cuda/12.1 selon dispo
# module load cuda/12.1

# --- Python / Conda ---
# module load python/3.10
# module load anaconda/2023
# source activate aztec_env

# --- PyTorch (si module dispo) ---
# module load pytorch/2.0

# ── Environnement Python ─────────────────────────────────────────────────────
# Option A : virtualenv (recommandé sur HPC)
if [ -d "$HOME/aztec_env" ]; then
    source $HOME/aztec_env/bin/activate
    echo "Environnement : $HOME/aztec_env"
else
    echo "Création de l'environnement virtuel..."
    python3 -m venv $HOME/aztec_env
    source $HOME/aztec_env/bin/activate
    pip install --quiet ultralytics opencv-python-headless scipy
fi

# ── Vérification GPU ─────────────────────────────────────────────────────────
echo ""
echo "GPUs disponibles :"
nvidia-smi --query-gpu=name,memory.total,memory.free \
           --format=csv,noheader,nounits
echo ""

python3 -c "
import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA OK  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'GPU {i}   : {p.name}  {p.total_memory//1e9:.0f}GB')
"

# ── Lancer l'entraînement ────────────────────────────────────────────────────
echo ""
echo "Lancement de l'entraînement YOLO..."
echo ""

python3 finetune_planches_v3.py \
    --data  data/synthetic_v3/dataset.yaml \
    --model best_fixed.pt \
    --epochs 20 \
    --imgsz 1280 \
    --batch 8 \
    --freeze 8 \
    --project runs/finetune \
    --name planches_v3_hpc

STATUS=$?

# ── Fin du job ───────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Fin : $(date)"
echo "Status : $STATUS"
echo "========================================"

if [ $STATUS -eq 0 ]; then
    echo "Entraînement terminé avec succès."
    echo "Résultats dans : runs/finetune/planches_v3_hpc/"

    # Packaging automatique des résultats
    zip -j aztec_finetuned_hpc_$SLURM_JOB_ID.zip \
        runs/finetune/planches_v3_hpc/weights/best.pt \
        runs/finetune/planches_v3_hpc/results.csv \
        runs/finetune/planches_v3_hpc/results.png \
        runs/finetune/planches_v3_hpc/confusion_matrix.png \
        logs/train_$SLURM_JOB_ID.out

    echo "Fichier zip : aztec_finetuned_hpc_$SLURM_JOB_ID.zip"
else
    echo "ERREUR : Vérifiez logs/train_$SLURM_JOB_ID.err"
fi
