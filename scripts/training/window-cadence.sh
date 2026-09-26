#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
OUTPUT="logs/window-cadence/$RUN_ID"
mkdir -p "$OUTPUT"
for VARIANT in sixteen four; do
  CONFIG=.research/protocols/window-cadence-16.json
  if [[ "$VARIANT" == four ]]; then
    CONFIG=.research/protocols/four-frame-development.json
  fi
  uv run --no-sync --offline python -m mathhackson.training.comparison.continuous \
    --config "$CONFIG" --output "$OUTPUT/$VARIANT" --workers 4 "$@" \
    2>&1 | tee "$OUTPUT/$VARIANT.log"
done
printf '完成：%s\n' "$OUTPUT"
