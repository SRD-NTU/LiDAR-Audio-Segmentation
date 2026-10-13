#!/bin/bash -l
#SBATCH --job-name=orbit
#SBATCH --output=ORBIT/logs/%x-%j.out
#SBATCH --error=ORBIT/logs/%x-%j.err
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/ORBIT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
srun "${PYTHON:-python}" -u "$@"
