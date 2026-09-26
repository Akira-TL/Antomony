#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/memory-check
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.comparison.memory_check \
  --output "logs/memory-check/$RUN_ID" "$@" \
  2>&1 | tee "logs/memory-check/$RUN_ID.log"
