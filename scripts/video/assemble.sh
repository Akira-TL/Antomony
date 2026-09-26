#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/video-v2
uv run --no-project --python 3.12 communication/video/production/assemble.py "$@" 2>&1 | tee logs/video-v2/assemble.log
