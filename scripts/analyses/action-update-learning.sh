#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
printf 'started=%s\n' "$(date -u --iso-8601=ns)"
timeout --kill-after=10s 120s uv run --no-sync --offline python scripts/analyses/action_update_learning.py \
  --config .research/analysis/action-update-learning/A001/config.json "$@" \
  2>&1 | tee logs/action-update-learning-analysis.log
printf 'completed=%s\n' "$(date -u --iso-8601=ns)"
