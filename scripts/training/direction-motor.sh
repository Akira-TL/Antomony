#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync python scripts/training/direction_motor.py \
  --output "logs/direction-motor/$RUN_ID" "$@" 2>&1 | tee "logs/direction-motor-$RUN_ID.log"
