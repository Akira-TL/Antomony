#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/matched-foundation
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.comparison.matched_foundation \
  --config .research/protocols/matched-foundation.json \
  --output "logs/matched-foundation/$RUN_ID" "$@" \
  2>&1 | tee "logs/matched-foundation/$RUN_ID.log"
