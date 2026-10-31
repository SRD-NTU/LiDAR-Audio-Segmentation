#!/bin/bash
#SBATCH --job-name=activeview-neo-multiview
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3-Neo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
PY=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
RANK=${1:?scene shard rank required}
WORKERS=${2:-2}
exec 9>"runtime/neo1-multiview-${RANK}.lock"
if ! flock -n 9; then
  # An opportunistic worker in the older allocation yields after its next saved view.
  touch "runtime/neo1-multiview-bootstrap-${RANK}.stop"
  flock 9
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -B tests/test_multiview.py
"$PY" -B -u scripts/evaluate_multiview.py --split test --rank "$RANK" --workers "$WORKERS"
