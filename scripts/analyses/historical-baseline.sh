#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
for VARIANT in off on; do
  uv run --no-sync scripts/acceptance/check_revival.py \
    "logs/historical-baseline/20260926T151701-2/$VARIANT" \
    2>&1 | tee "logs/analysis/historical-baseline-$VARIANT-integrity-A001.log"
done
uv run --no-sync scripts/analyses/historical_baseline.py \
  --config .research/analysis/historical-baseline/A001/config.json \
  2>&1 | tee logs/analysis/historical-baseline-A001.log
