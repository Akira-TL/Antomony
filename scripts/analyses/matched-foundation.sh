#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
uv run --no-sync --offline python scripts/analyses/matched_foundation.py \
  --config .research/analysis/matched-foundation/A001/config.json "$@" \
  2>&1 | tee logs/matched-foundation-analysis.log
