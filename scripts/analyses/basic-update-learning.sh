#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
uv run --no-sync --offline python scripts/analyses/basic_update_learning.py \
  --config .research/analysis/basic-update-learning/A001/config.json "$@" 2>&1 | tee logs/basic-update-learning-analysis.log
