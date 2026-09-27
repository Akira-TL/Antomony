#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/video-v2
uv run --no-project --python 3.12 --with numpy communication/video/production/retime_audio.py 2>&1 | tee logs/video-v2/retime.log
