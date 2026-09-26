#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/continuous-adaptation
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.comparison.continuous \
  --config .research/protocols/continuous-adaptation.json \
  --output "logs/continuous-adaptation/$RUN_ID" "$@" \
  2>&1 | tee "logs/continuous-adaptation/$RUN_ID.log"
