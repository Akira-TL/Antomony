#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs/feedback-plasticity
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.foraging.plastic_course.run \
  --output "logs/feedback-plasticity/$RUN_ID" "$@" \
  2>&1 | tee "logs/feedback-plasticity/$RUN_ID.log"
