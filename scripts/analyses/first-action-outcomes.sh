#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase="${1:?sample or audit}"
case "$phase" in
  sample) limit=240; attempt=A001 ;;
  audit) limit=58; attempt=A002 ;;
  *) exit 2 ;;
esac
mkdir -p logs/analysis
timeout --signal=KILL "$limit" env UV_CACHE_DIR=/tmp/mathhackson-uv-cache uv run --no-sync \
  scripts/analyses/first_action_outcomes.py "$phase" --config ".research/analysis/first-action-outcomes/$attempt/config.json" \
  2>&1 | tee "logs/analysis/first-action-outcomes-${phase}-${attempt}.log"
