#!/bin/bash
#SBATCH --job-name=activeview-auditfix1-eval
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
PY=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
RANK=${1:?rank required}
WORKERS=${2:-2}
exec 9>"runtime/auditfix1-eval-${RANK}.lock"
flock -n 9
"$PY" -B -u scripts/evaluate.py --split test --rank "$RANK" --workers "$WORKERS" --protocol config/frozen_evaluation_auditfix1.json
