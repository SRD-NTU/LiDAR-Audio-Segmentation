#!/bin/bash
#SBATCH --job-name=activeview-auditfix1-train
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
"$PY" -B tests/test_contracts.py
"$PY" -B tests/test_multiview.py
"$PY" -B tests/test_auditfix.py
"$PY" -B -u scripts/train.py --stage evidence --revision auditfix1
"$PY" -B -u scripts/train.py --stage world --seed 1337 --revision auditfix1
"$PY" -B -u scripts/train.py --stage world --seed 2027 --revision auditfix1
"$PY" -B -u scripts/train.py --stage direct --revision auditfix1
"$PY" -B -u scripts/freeze_protocol.py --revision auditfix1
"$PY" -B -u scripts/check_two_view_resume.py
"$PY" -B -u scripts/smoke_multiview.py --revision neo1 --parent-revision auditfix1
"$PY" -B -u scripts/diagnose_world.py --split val --revision auditfix1
"$PY" -B -u scripts/freeze_multiview.py --revision neo1 --parent-revision auditfix1
