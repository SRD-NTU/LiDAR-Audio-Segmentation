#!/bin/bash
#SBATCH --job-name=echo_lidar_adapt
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/cache/tmp"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export PIP_CACHE_DIR="$PWD/cache/pip"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
nvidia-smi
PYTHON=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
for method in echosegnet jaeger said sound3dvdet; do
    "$PYTHON" scripts/train_passive.py --method "$method" --device cuda --workers 0 \
        --batch-size 1 --context-points 256 --query-points 128 --smoke
done
"$PYTHON" scripts/train_passive.py --method echosegnet --device cuda --workers 2 \
    --batch-size 4 --context-points 2048 --query-points 1024 --epochs 6 \
    --run-dir "runs/echosegnet_lidar_full_${SLURM_JOB_ID}"
