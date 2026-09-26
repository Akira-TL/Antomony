#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/acceptance-revival
uv run --no-sync --offline python -m mathhackson.training.comparison.continuous \
    --config docs/engineering/reviving-acceptance.json \
    --output logs/acceptance-revival/20260926-v1 --workers 8 \
    2>&1 | tee -a logs/acceptance-revival/20260926-v1.log
