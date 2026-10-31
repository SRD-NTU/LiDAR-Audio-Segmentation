#!/bin/bash
# Run as an overlapping Slurm step inside the existing GPU allocation.
# A separately queued job takes over using the same lock and saved observations.
set -euo pipefail
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3-Neo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
exec 9>runtime/neo1-multiview-0.lock
flock -n 9 || { echo 'Dedicated shard 0 already owns the work'; exit 0; }
PY=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
"$PY" -B -u scripts/evaluate_multiview.py --split test --rank 0 --workers 2 --stop-file runtime/neo1-multiview-bootstrap-0.stop
