#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
uv run --no-sync --offline python scripts/analyses/exploration_course.py \
  --config .research/analysis/exploration-course/A001/config.json "$@" \
  2>&1 | tee logs/exploration-course-analysis.log
