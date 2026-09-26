#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/logs/external-models/runtime${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python scripts/training/direction_gate_control.py \
  --output "logs/direction-gate-control/$RUN_ID" "$@" 2>&1 | tee "logs/direction-gate-control-$RUN_ID.log"
