#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
for VARIANT in legacy survival; do
  uv run --no-sync scripts/acceptance/check_revival.py \
    "logs/survival-feedback/20260926T161127-3/$VARIANT" \
    2>&1 | tee "logs/analysis/survival-feedback-$VARIANT-integrity-A001.log"
done
uv run --no-sync scripts/analyses/survival_feedback.py \
  --config .research/analysis/survival-feedback/A001/config.json \
  2>&1 | tee logs/analysis/survival-feedback-A001.log
