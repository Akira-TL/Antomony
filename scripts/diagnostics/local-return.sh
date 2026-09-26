#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python scripts/diagnostics/local_return.py \
  2>&1 | tee "logs/local-return-$RUN_ID.jsonl"
