#!/bin/bash
#SBATCH --job-name=activeview-auditfix1-report
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=wanglin_2026_08
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/Acoustic-WM-ActiveView-v3-Neo
PY=/projects/soumyaratna/LiDAR_Segmentation/Acoustic-WM/.venv/bin/python
"$PY" -B scripts/report.py --split test --protocol config/frozen_evaluation_auditfix1.json
