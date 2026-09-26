#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs/memory-update-learning
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.foraging.update_curriculum \
  --plan .research/protocols/memory-update-learning.json \
  --output "logs/memory-update-learning/$RUN_ID" "$@" \
  2>&1 | tee "logs/memory-update-learning/$RUN_ID.log"
