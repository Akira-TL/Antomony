#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
date -u --iso-8601=ns > logs/analysis/candidate-steering-A001-times.log
uv run --no-sync scripts/analyses/candidate_steering.py \
  --config .research/analysis/candidate-steering/A001/config.json \
  2>&1 | tee logs/analysis/candidate-steering-A001.log
date -u --iso-8601=ns >> logs/analysis/candidate-steering-A001-times.log
