#!/usr/bin/env bash
# Install the tested ORBIT environment from any working directory.
set -euo pipefail
environment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$environment_dir/../ORBIT/scripts/setup_env.sh" "$@"
