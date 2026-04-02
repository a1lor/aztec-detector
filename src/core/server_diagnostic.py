"""
server_diagnostic.py
====================
Lance ce script EN PREMIER sur le serveur pour connaître
exactement les ressources disponibles.

Usage :
    python server_diagnostic.py
"""

import subprocess
import sys
import platform
import os
from pathlib import Path


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except:
        return "N/A"


def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


# ── CPU ──────────────────────────────────────────────
section("CPU")
print(f"OS          : {platform.system()} {platform.release()}")
print(f"Python      : {sys.version.split()[0]}")
n_cpu = os.cpu_count()
print(f"CPU cores   : {n_cpu}")

# RAM
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if "MemTotal" in line:
                ram_kb = int(line.split()[1])
                print(f"RAM totale  : {ram_kb // 1024 // 1024} GB")
                break
except:
    print("RAM totale  : N/A (non-Linux)")

# ── GPU ──────────────────────────────────────────────
section("GPU")
try:
    import torch
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"GPUs détectés : {n_gpu}")
        for i in range(n_gpu):
            name  = torch.cuda.get_device_name(i)
            vram  = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i} : {name}  —  {vram:.1f} GB VRAM")
        print(f"\nCUDA version  : {torch.version.cuda}")
        print(f"PyTorch       : {torch.__version__}")

        # Test rapide bande passante GPU
        import time
        x = torch.zeros(1000, 1000).cuda()
        t0 = time.time()
        for _ in range(100):
            x = x * 2 + 1
        torch.cuda.synchronize()
        print(f"GPU compute   : OK ({(time.time()-t0)*10:.1f} ms/op)")
    else:
        print("Aucun GPU CUDA détecté.")
        print("→ Vérifiez le module CUDA sur le cluster :")
        print("  module avail | grep cuda")
except ImportError:
    print("PyTorch non installé.")
    print("→ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

# ── STOCKAGE ─────────────────────────────────────────
section("Stockage")
try:
    stat = os.statvfs(".")
    free_gb  = stat.f_bavail * stat.f_frsize / 1e9
    total_gb = stat.f_blocks * stat.f_frsize / 1e9
    print(f"Disque courant : {free_gb:.0f} GB libres / {total_gb:.0f} GB total")
except:
    pass

# Taille estimée du dataset
data_paths = [
    Path("data/full_yolo_dataset"),
    Path("data/synthetic_v3"),
    Path("data/synthetic_planches"),
]
for p in data_paths:
    if p.exists():
        result = run(f"du -sh {p}")
        print(f"  {p} : {result.split()[0] if result != 'N/A' else 'N/A'}")

# ── SCHEDULER (SLURM / PBS) ───────────────────────────
section("Job Scheduler")
slurm = run("sinfo --version")
pbs   = run("qstat --version")
sge   = run("qsub --help | head -1")

if slurm != "N/A":
    print(f"SLURM détecté : {slurm}")
    print("\nPartitions disponibles :")
    print(run("sinfo -o '%P %G %C %m' --noheader | head -20"))
    print("\nGPUs disponibles :")
    print(run("sinfo -o '%P %G' --noheader | grep -i gpu | head -10"))
elif pbs != "N/A":
    print(f"PBS/Torque détecté : {pbs}")
    print(run("qstat -q | head -20"))
else:
    print("Aucun scheduler détecté (machine interactive ou autre).")
    print("→ Si HPC : vérifiez avec 'squeue', 'qstat', ou 'bjobs'")

# ── RÉSEAU / TRANSFERT ────────────────────────────────
section("Réseau")
print(f"Hostname    : {run('hostname')}")
print(f"IP locale   : {run('hostname -I | awk {print $1}')}")

# ── ENVIRONNEMENT ─────────────────────────────────────
section("Environnement Python")
pkgs = ["ultralytics", "torch", "torchvision", "cv2", "numpy", "scipy"]
for pkg in pkgs:
    try:
        mod = __import__(pkg if pkg != "cv2" else "cv2")
        ver = getattr(mod, "__version__", "installé")
        print(f"  {pkg:<15} : {ver}")
    except ImportError:
        print(f"  {pkg:<15} : NON INSTALLÉ")

# ── RECOMMANDATIONS IMMÉDIATES ────────────────────────
section("Recommandations immédiates")

try:
    import torch
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        print("→ Pas de GPU accessible. Sur HPC, chargez un module GPU :")
        print("     module load cuda/11.8")
        print("     ou soumettez un job avec #SBATCH --gres=gpu:1")
    elif n_gpu == 1:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 40:
            print(f"→ GPU {torch.cuda.get_device_name(0)} ({vram:.0f}GB) : excellent")
            print("   Utilisez batch=16, imgsz=1280, workers=8")
        elif vram >= 16:
            print(f"→ GPU {torch.cuda.get_device_name(0)} ({vram:.0f}GB) : bon")
            print("   Utilisez batch=8, imgsz=1280, workers=6")
        elif vram >= 8:
            print(f"→ GPU {torch.cuda.get_device_name(0)} ({vram:.0f}GB) : correct")
            print("   Utilisez batch=4, imgsz=1280, workers=4")
        else:
            print(f"→ GPU {torch.cuda.get_device_name(0)} ({vram:.0f}GB) : limité")
            print("   Utilisez batch=2, imgsz=640, workers=2")
    elif n_gpu > 1:
        print(f"→ {n_gpu} GPUs disponibles : entraînement multi-GPU possible !")
        print("   Ajoutez device='0,1' dans model.train()")
        total_vram = sum(
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(n_gpu)
        )
        print(f"   VRAM totale : {total_vram:.0f} GB")
        print(f"   Utilisez batch={n_gpu * 8}, workers={n_gpu * 4}")
except:
    pass

print("\n" + "="*50)
print("Copiez-collez la sortie complète pour obtenir")
print("une configuration optimale personnalisée.")
print("="*50 + "\n")
