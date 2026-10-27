#!/bin/bash
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=av24-campaign
#SBATCH --output=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/research/logs/%x-%j.out
#SBATCH --error=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/research/logs/%x-%j.err
set -euo pipefail
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/research
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$PWD/runtime/xdg" TORCH_HOME="$PWD/runtime/torch" MPLCONFIGDIR="$PWD/runtime/matplotlib" NUMBA_CACHE_DIR="$PWD/runtime/numba" TMPDIR="$PWD/runtime/tmp" TORCH_EXTENSIONS_DIR="$PWD/runtime/extensions" TRITON_CACHE_DIR="$PWD/runtime/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun --cpu-bind=none /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python -B -u scripts/node.py "${1:-0}"
