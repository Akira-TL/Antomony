#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
date -u --iso-8601=ns > logs/analysis/window-cadence-A001-times.log
for VARIANT in sixteen four; do
  uv run --no-sync scripts/acceptance/check_revival.py \
    "logs/window-cadence/20260926T165559-3/$VARIANT" \
    2>&1 | tee "logs/analysis/window-cadence-$VARIANT-integrity-A001.log"
done
uv run --no-sync scripts/analyses/window_cadence.py \
  --config .research/analysis/window-cadence/A001/config.json \
  2>&1 | tee logs/analysis/window-cadence-A001.log
date -u --iso-8601=ns >> logs/analysis/window-cadence-A001-times.log
