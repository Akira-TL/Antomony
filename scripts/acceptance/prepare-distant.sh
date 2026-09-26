#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/acceptance-far
uv run --no-sync --offline python -m mathhackson.training.comparison.continuous \
    --config docs/engineering/distant-acceptance.json \
    --output logs/acceptance-far/20260926-v1 \
    2>&1 | tee -a logs/acceptance-far/20260926-v1.log
