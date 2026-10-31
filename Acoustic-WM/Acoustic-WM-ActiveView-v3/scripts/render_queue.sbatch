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
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
pids=()
IFS=',' read -r -a gpu_devices <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU visibility is required}"
for ((rank=0; rank<${QUEUE_WORKERS:-4}; rank++)); do
 CUDA_VISIBLE_DEVICES="${gpu_devices[$((rank % ${#gpu_devices[@]}))]}" /projects/soumyaratna/LiDAR_Segmentation/.redun/dataset-utility/utility/environments/audiolidar/bin/python -B -u scripts/render_queue.py --worker "$((rank + ${QUEUE_OFFSET:-0}))" &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
