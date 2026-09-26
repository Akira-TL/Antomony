#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
OUTPUT="logs/survival-feedback/$RUN_ID"
mkdir -p "$OUTPUT"
for VARIANT in legacy survival; do
  uv run --no-sync --offline python -m mathhackson.training.comparison.continuous \
    --config ".research/protocols/survival-feedback-$VARIANT.json" \
    --output "$OUTPUT/$VARIANT" --workers 4 "$@" \
    2>&1 | tee "$OUTPUT/$VARIANT.log"
done
printf '完成：%s\n' "$OUTPUT"
