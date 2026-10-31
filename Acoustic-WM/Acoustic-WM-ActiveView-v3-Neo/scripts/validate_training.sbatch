#!/bin/bash
#SBATCH --job-name=neo-validate
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
"$PY" -B tests/test_contracts.py
"$PY" -B tests/test_multiview.py
"$PY" -B tests/test_auditfix.py
"$PY" -B tests/test_neo_memory.py
"$PY" -B scripts/check_neo_real.py
"$PY" -B -u scripts/check_two_view_resume.py
"$PY" -B -u scripts/smoke_multiview.py --revision neo1 --parent-revision auditfix1
"$PY" -B -u scripts/diagnose_world.py --split val --revision auditfix1
