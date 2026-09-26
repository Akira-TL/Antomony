#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/exploration-course
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.comparison.exploration_course \
  --config .research/protocols/exploration-course.json \
  --output "logs/exploration-course/$RUN_ID" "$@" \
  2>&1 | tee "logs/exploration-course/$RUN_ID.log"
