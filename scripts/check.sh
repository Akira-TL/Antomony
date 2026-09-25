#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
{
    uv run --locked python -m pytest tests
    bash scripts/setup-skills.sh --check
} 2>&1 | tee logs/check.log
