#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
for WINDOW in 16 64; do
  uv run --no-sync --offline python -m mathhackson.training.foraging.candidate_probe \
    --plan ".research/protocols/feedback-window-$WINDOW.json" \
    --output "logs/feedback-window/$RUN_ID/window-$WINDOW" "$@" 2>&1 | tee "logs/feedback-window-$RUN_ID-$WINDOW.log"
done
