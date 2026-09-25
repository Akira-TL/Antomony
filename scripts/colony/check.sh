#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run pytest -q tests/colony "$@" 2>&1 | tee logs/colony-check.log
