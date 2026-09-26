#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
for ARM in zero random coherent; do
  uv run --no-sync --offline python -m mathhackson.training.foraging.candidate_probe \
    --plan ".research/protocols/restoration-$ARM.json" \
    --output "logs/restoration-control/$RUN_ID/$ARM" "$@" 2>&1 | tee "logs/restoration-control-$RUN_ID-$ARM.log"
done
