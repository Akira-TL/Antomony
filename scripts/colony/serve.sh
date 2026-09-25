#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec uv run python -m uvicorn mathhackson.colony.server:app --host 127.0.0.1 --port "${PORT:-8765}" --no-access-log
