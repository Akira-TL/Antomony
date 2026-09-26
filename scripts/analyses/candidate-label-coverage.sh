#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
timeout --signal=KILL 90 env UV_CACHE_DIR=/tmp/mathhackson-uv-cache uv run --no-sync \
  scripts/analyses/candidate_label_coverage.py --config .research/analysis/candidate-label-coverage/A001/config.json \
  2>&1 | tee logs/analysis/candidate-label-coverage-A001.log
