#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
uv run --no-sync scripts/analyses/continuous_adaptation.py --config ".research/analysis/continuous-adaptation/A001/config.json" 2>&1 | tee logs/analysis/continuous-adaptation-A001.log
