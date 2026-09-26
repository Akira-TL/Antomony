#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
uv pip install --target logs/external-models/runtime --no-deps \
  --requirements scripts/training/pretrained-runtime.txt "$@" 2>&1 | tee logs/pretrained-setup.log
