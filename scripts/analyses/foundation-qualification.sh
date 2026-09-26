#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
uv run --no-sync --offline python scripts/analyses/foundation_qualification.py \
  --config .research/analysis/foundation-qualification/A001/config.json "$@" \
  2>&1 | tee logs/foundation-qualification-analysis.log
