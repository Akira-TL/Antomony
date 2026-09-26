#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
uv run --no-sync scripts/analyses/survival_timing.py \
  --config .research/analysis/survival-timing/A001/config.json \
  2>&1 | tee logs/analysis/survival-timing-A001.log
