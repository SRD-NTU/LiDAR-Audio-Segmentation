#!/bin/bash
#SBATCH --job-name=adapted_methods
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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
PYTHON=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
nvidia-smi

# Echo training is in its predecessor job; this job evaluates that checkpoint.
ECHO="runs/echosegnet_lidar_full_228748/best.pt"
if [[ -f "$ECHO" ]]; then
    "$PYTHON" scripts/run_baseline.py --method echosegnet --mode passive \
        --variant adapted --split test --checkpoint "$ECHO" \
        --device cuda --output-dir "results/echosegnet_adapted_test_${SLURM_JOB_ID}"
fi

for method in jaeger said sound3dvdet; do
    run="runs/${method}_full_${SLURM_JOB_ID}"
    "$PYTHON" scripts/train_passive.py --method "$method" --device cuda \
        --workers 2 --batch-size 2 --context-points 2048 --query-points 1024 \
        --epochs 6 --run-dir "$run"
    "$PYTHON" scripts/run_baseline.py --method "$method" --mode passive \
        --variant adapted --split test --checkpoint "$run/best.pt" \
        --device cuda --output-dir "results/${method}_adapted_test_${SLURM_JOB_ID}"
done

run="runs/magnet_full_${SLURM_JOB_ID}"
"$PYTHON" scripts/train_magnet.py --device cuda --workers 2 \
    --batch-size 4 --context-points 2048 --epochs 10 --run-dir "$run"
if [[ -f "$ECHO" ]]; then
    "$PYTHON" scripts/run_baseline.py --method magnet --mode active \
        --variant adapted --split test --checkpoint "$run/best.pt" \
        --backend-method echosegnet --backend-checkpoint "$ECHO" \
        --device cuda --output-dir "results/magnet_conditional_menu_test_${SLURM_JOB_ID}"
fi
