#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs/action-update-learning
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
printf 'run=%s started=%s\n' "$RUN_ID" "$(date -u --iso-8601=ns)"
timeout --kill-after=10s 600s uv run --no-sync --offline python -m mathhackson.training.foraging.update_curriculum \
  --plan .research/protocols/action-update-learning.json \
  --output "logs/action-update-learning/$RUN_ID" "$@" \
  2>&1 | tee "logs/action-update-learning/$RUN_ID.log"
printf 'run=%s completed=%s\n' "$RUN_ID" "$(date -u --iso-8601=ns)"
