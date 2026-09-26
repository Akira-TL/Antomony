#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
printf 'started=%s\n' "$(date -u --iso-8601=ns)"
timeout --kill-after=10s 120s uv run --no-sync --offline python scripts/analyses/credit_history.py \
  --config .research/analysis/credit-history/A001/config.json "$@" \
  2>&1 | tee logs/credit-history-analysis.log
printf 'completed=%s\n' "$(date -u --iso-8601=ns)"
