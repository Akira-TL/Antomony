#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
uv run --no-sync --offline python scripts/analyses/trust_candidate_value.py \
  --config .research/analysis/trust-candidate-value/A001/config.json \
  2>&1 | tee logs/trust-candidate-value-analysis.log
