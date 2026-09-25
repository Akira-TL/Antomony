#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run python scripts/colony/smoke.py 2>&1 | tee logs/colony-smoke.log
