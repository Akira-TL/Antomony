#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
uv run --no-sync --offline python scripts/analyses/feedback_window.py \
  --config .research/analysis/feedback-window/A001/config.json "$@" 2>&1 | tee logs/feedback-window-analysis.log
