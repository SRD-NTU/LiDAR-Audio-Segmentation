#!/bin/bash
#SBATCH --job-name=activeview-v3-pose-recovery
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
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3-Neo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
PY=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
RANK=${1:?rank or two-view required}
if [[ "$RANK" == two-view ]]; then
  exec 9>runtime/two-view-recovery.lock
  flock -n 9
  "$PY" -B -u scripts/recover_pose_validation.py --two-view
else
  exec 9>"runtime/multiview-test-${RANK}.lock"
  flock -n 9
  "$PY" -B -u scripts/recover_pose_validation.py --rank "$RANK" --workers 2
fi
