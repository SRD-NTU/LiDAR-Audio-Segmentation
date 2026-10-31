#!/bin/bash
#SBATCH --job-name=activeview-v3-render
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3-Neo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
render_branch() {
 /projects/soumyaratna/LiDAR_Segmentation/.redun/dataset-utility/utility/environments/audiolidar/bin/python -B -u scripts/render_training.py --split train --rank "$1"
 /projects/soumyaratna/LiDAR_Segmentation/.redun/dataset-utility/utility/environments/audiolidar/bin/python -B -u scripts/render_training.py --split val --rank "$1"
}
pids=()
for rank in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="$((rank % 2))" render_branch "$rank" &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
