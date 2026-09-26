#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/analysis
attempt="${1:-A002}"
if [[ "$attempt" != A001 && "$attempt" != A002 ]]; then
  printf '%s\n' '仅允许已登记执行配置' >&2
  exit 2
fi
budget=300
if [[ "$attempt" == A002 ]]; then budget=260; fi
timeout --signal=TERM --kill-after=5s "${budget}s" uv run --no-sync scripts/analyses/update_constraints.py \
  --config ".research/analysis/update-constraints/$attempt/config.json" \
  2>&1 | tee "logs/analysis/update-constraints-$attempt.log"
