#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
uv run python scripts/diagnostics/colony_behavior.py 2>&1 | tee logs/colony-behavior-regression.log
