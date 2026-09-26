#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/injury-candidate
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python -m mathhackson.training.comparison.injury_probe \
  --config .research/protocols/injury-candidate.json \
  --output "logs/injury-candidate/$RUN_ID" \
  2>&1 | tee "logs/injury-candidate/$RUN_ID.log"
