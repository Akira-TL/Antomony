#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/video-v2
uv run --no-project --python 3.12 --with edge-tts==7.2.8 communication/video/production/audio.py "$@" 2>&1 | tee logs/video-v2/audio.log
