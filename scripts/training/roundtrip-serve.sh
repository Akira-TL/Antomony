#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec uv run --offline uvicorn mathhackson.training.roundtrip_server:app --host 127.0.0.1 --port "${PORT:-8772}"
