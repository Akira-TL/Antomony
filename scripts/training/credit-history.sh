#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
printf 'run=%s started=%s\n' "$RUN_ID" "$(date -u --iso-8601=ns)"
for VARIANT in four sixteen; do
  timeout --kill-after=10s 300s uv run --no-sync --offline python -m mathhackson.training.foraging.candidate_probe \
    --plan ".research/protocols/credit-history-$VARIANT.json" \
    --output "logs/credit-history/$RUN_ID/$VARIANT" "$@" \
    2>&1 | tee "logs/credit-history-$RUN_ID-$VARIANT.log"
done
printf 'run=%s completed=%s\n' "$RUN_ID" "$(date -u --iso-8601=ns)"
