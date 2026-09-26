#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
uv run --no-sync --offline python scripts/analyses/restoration_control.py \
  --config .research/analysis/restoration-control/A001/config.json "$@" 2>&1 | tee logs/restoration-control-analysis.log
