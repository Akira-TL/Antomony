#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec uv run --no-sync python -m uvicorn mathhackson.training.server:app --host 127.0.0.1 --port "${PORT:-8766}" --no-access-log
