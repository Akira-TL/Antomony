#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
for VARIANT in on off; do
  uv run --no-sync scripts/acceptance/check_revival.py \
    "logs/return-reward-ablation/20260926T142153-2/$VARIANT" \
    2>&1 | tee "logs/analysis/return-reward-$VARIANT-integrity-A002.log"
done
uv run --no-sync scripts/analyses/return_reward_ablation.py \
  --config .research/analysis/return-reward-ablation/A002/config.json \
  2>&1 | tee logs/analysis/return-reward-ablation-A002.log
