#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/video-v2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-project --python 3.12 --with 'manimgl @ git+https://github.com/3b1b/manim.git@fafa083a4fb274bba9cabde0b6e2f50ba6da0622' --with av communication/video/production/check_artwork.py 2>&1 | tee logs/video-v2/artwork-check.log
