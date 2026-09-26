#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
for VARIANT in window early; do
  uv run --no-sync scripts/acceptance/check_revival.py \
    "logs/feedback-trigger/20260926T163643-3/$VARIANT" \
    2>&1 | tee "logs/analysis/feedback-trigger-$VARIANT-integrity-A001.log"
done
uv run --no-sync scripts/analyses/feedback_trigger.py \
  --config .research/analysis/feedback-trigger/A001/config.json \
  2>&1 | tee logs/analysis/feedback-trigger-A001.log
