#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
OUTPUT="logs/return-reward-ablation/$RUN_ID"
mkdir -p "$OUTPUT"
for VARIANT in on off; do
  uv run --no-sync --offline python -m mathhackson.training.comparison.continuous \
    --config ".research/protocols/return-reward-$VARIANT.json" \
    --output "$OUTPUT/$VARIANT" --workers 8 "$@" \
    2>&1 | tee "$OUTPUT/$VARIANT.log"
done
printf '完成：%s\n' "$OUTPUT"
