#!/usr/bin/env bash
# Run from any directory: bash ORBIT/scripts/setup_env.sh [--cpu]
# Default: Linux/NVIDIA CUDA 12.8 PyTorch + the tested sparse extensions.
# --cpu installs only the core/testing environment; raw Utonia needs CUDA.
set -euo pipefail
orbit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---cuda}"
case "$mode" in --cuda|--cpu) ;; *) echo 'Usage: setup_env.sh [--cuda|--cpu]' >&2; exit 2 ;; esac
if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif command -v python3.10 >/dev/null 2>&1; then
    python_bin=python3.10
elif command -v python3.11 >/dev/null 2>&1; then
    python_bin=python3.11
elif command -v python >/dev/null 2>&1 && python -c 'import sys; raise SystemExit(not ((3,10) <= sys.version_info[:2] < (3,12)))'; then
    python_bin=python
else
    echo 'Python 3.10 or 3.11 is required; set PYTHON to its executable path' >&2
    exit 2
fi
"$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,12), "Use Python 3.10 or 3.11"'
"$python_bin" -m venv "$orbit_dir/.venv"
py="$orbit_dir/.venv/bin/python"
"$py" -m pip install --upgrade pip setuptools wheel
torch_index='https://download.pytorch.org/whl/cu128'
if [[ "$mode" == '--cpu' ]]; then torch_index='https://download.pytorch.org/whl/cpu'; fi
"$py" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$torch_index"
"$py" -m pip install -r "$orbit_dir/requirements.txt"
if [[ "$mode" == '--cuda' ]]; then
    "$py" -m pip install spconv-cu126==2.3.8
    "$py" -m pip install 'torch-scatter==2.1.2+pt28cu128' --only-binary=:all: \
        --find-links 'https://data.pyg.org/whl/torch-2.8.0+cu128.html'
fi
"$py" -m pip install --no-deps -e "$orbit_dir"
echo 'Installed. From the project root, run: bash ORBIT/run.sh full ORBIT/scripts/run_sample.py --help'
