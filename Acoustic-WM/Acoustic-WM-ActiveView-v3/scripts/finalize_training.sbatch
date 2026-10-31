#!/bin/bash
#SBATCH --job-name=v3-freeze
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PY=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
"$PY" -B -u scripts/freeze_protocol.py --revision auditfix1
"$PY" -B -u scripts/freeze_multiview.py --revision auditfix1
