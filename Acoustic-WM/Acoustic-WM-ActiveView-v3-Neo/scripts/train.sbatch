#!/bin/bash
#SBATCH --job-name=activeview-v3-train
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
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3-Neo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
$PY -B -u scripts/extract.py --split train --workers 1
$PY -B -u scripts/extract.py --split val --workers 1
$PY -B -u tests/test_contracts.py
$PY -B -u scripts/train.py --stage evidence
$PY -B -u scripts/train.py --stage world --seed 1337
$PY -B -u scripts/train.py --stage world --seed 2027
$PY -B -u scripts/train.py --stage direct
$PY -B -u scripts/freeze_protocol.py
$PY -B -u scripts/evaluate.py --split test
$PY -B -u scripts/report.py --split test
