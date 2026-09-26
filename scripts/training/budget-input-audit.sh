#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python scripts/training/budget_input_audit.py \
  --output "logs/budget-input-audit/$RUN_ID" "$@" 2>&1 | tee "logs/budget-input-audit-$RUN_ID.log"
