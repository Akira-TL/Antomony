#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
{
    uv run python -m unittest discover -s tests -v
    bash scripts/setup-skills.sh --check
} 2>&1 | tee logs/check.log
