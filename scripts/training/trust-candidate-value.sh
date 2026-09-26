#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.foraging.candidate_probe \
  --plan .research/protocols/trust-candidate-value.json \
  --output "logs/trust-candidate-value/$RUN_ID" "$@" 2>&1 | tee "logs/trust-candidate-value-$RUN_ID.log"
