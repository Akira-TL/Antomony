#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/foundation-qualification
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.comparison.qualification \
  --config .research/protocols/foundation-qualification.json \
  --output "logs/foundation-qualification/$RUN_ID" "$@" \
  2>&1 | tee "logs/foundation-qualification/$RUN_ID.log"
