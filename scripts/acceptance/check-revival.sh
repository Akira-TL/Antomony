#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/acceptance-revival
uv run --no-sync --offline python scripts/acceptance/check_revival.py \
    logs/acceptance-revival/20260926-v1 \
    2>&1 | tee logs/acceptance-revival/verification.log
