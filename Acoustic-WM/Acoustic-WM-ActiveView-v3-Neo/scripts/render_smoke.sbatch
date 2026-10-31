#!/bin/bash
#SBATCH --job-name=activeview-v3-smoke
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3-Neo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
/projects/soumyaratna/LiDAR_Segmentation/.redun/dataset-utility/utility/environments/audiolidar/bin/python -B -u scripts/render_smoke.py
