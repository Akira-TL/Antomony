#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --with huggingface-hub==0.34.4 --with safetensors==0.4.5 --with mup==1.0.0 \
  python scripts/training/pretrained_probe.py "$@" 2>&1 | tee "logs/pretrained-probe-$RUN_ID.log"
