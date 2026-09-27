#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs/analysis
uv run --no-sync --offline python scripts/analyses/plastic_projection.py "$@" \
  2>&1 | tee "logs/analysis/plastic-projection-$(date -u +%Y%m%dT%H%M%S).log"
