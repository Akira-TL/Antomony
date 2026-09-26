#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
timeout --signal=TERM --kill-after=5s 300s uv run --no-sync scripts/analyses/update_constraints.py \
  --config .research/analysis/update-constraints/A001/config.json \
  2>&1 | tee logs/analysis/update-constraints-A001.log
