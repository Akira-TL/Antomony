#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
OUTPUT="logs/four-frame/$RUN_ID"
mkdir -p "$OUTPUT"
uv run --no-sync --offline python -m mathhackson.training.comparison.continuous \
  --config .research/protocols/four-frame-development.json \
  --output "$OUTPUT/run" --workers 4 "$@" \
  2>&1 | tee "$OUTPUT/run.log"
printf '完成：%s\n' "$OUTPUT"
