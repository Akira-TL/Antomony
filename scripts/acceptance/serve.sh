#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run --no-sync uvicorn mathhackson.training.comparison.acceptance:app --host 127.0.0.1 --port "${PORT:-8774}"
