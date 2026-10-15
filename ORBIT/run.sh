#!/usr/bin/env bash
set -euo pipefail
orbit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$orbit_dir")"
tier="${1:-full}"
case "$tier" in easy|medium|hard|full) ;; *) echo 'Usage: ORBIT/run.sh {easy|medium|hard|full} [python arguments]' >&2; exit 2 ;; esac
if [[ $# -gt 0 ]]; then shift; fi
if [[ -z "${AUDIO_LIDAR_DATA_ROOT:-}" ]]; then
    export AUDIO_LIDAR_DATA_ROOT="$project_dir/dataset/v2-tetra4-4channel-$tier"
    if [[ ! -d "$AUDIO_LIDAR_DATA_ROOT" && -f "$(dirname -- "$project_dir")/LiSeg-Dataset/metadata/test.json" ]]; then
        export AUDIO_LIDAR_DATA_ROOT="$(dirname -- "$project_dir")/LiSeg-Dataset"
    elif [[ ! -d "$AUDIO_LIDAR_DATA_ROOT" && -f "$project_dir/dataset-sample/metadata/val.json" ]]; then
        export AUDIO_LIDAR_DATA_ROOT="$project_dir/dataset-sample"
    fi
fi
export PYTHONPATH="$orbit_dir:$orbit_dir/runtime${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$project_dir"
if [[ -n "${ORBIT_PYTHON:-}" ]]; then
    python_bin="$ORBIT_PYTHON"
elif [[ -x "$orbit_dir/.venv/bin/python" ]]; then
    python_bin="$orbit_dir/.venv/bin/python"
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
    python_bin="$project_dir/.venv/bin/python"
else
    python_bin="python3"
fi
exec "$python_bin" "$@"
