#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
uv run --no-sync scripts/analyses/injury_candidate.py \
  --config .research/analysis/injury-candidate/A001/config.json \
  2>&1 | tee logs/analysis/injury-candidate-A001.log
