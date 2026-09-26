#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec uv run --no-sync uvicorn mathhackson.interactive.server:app --host 127.0.0.1 --port "${PORT:-8775}"
