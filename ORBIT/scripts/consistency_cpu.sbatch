#!/bin/bash
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=wanglin_2026_08
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=orbit-v2-cache
#SBATCH --output=ORBIT/logs/%x-%j.out
#SBATCH --error=ORBIT/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
srun bash ORBIT/run.sh full -u "$@"
