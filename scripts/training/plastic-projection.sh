#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ "$#" -ne 1 ]]; then
    printf '%s\n' '用法：bash scripts/training/plastic-projection.sh <冻结提交>' >&2
    exit 2
fi
RUN="logs/plastic-projection/$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$RUN"
printf '原始记录：%s\n' "$RUN"
for MODE in unit bounded; do
    timeout --signal=KILL 440s uv run --no-sync --offline python -m mathhackson.training.foraging.plastic_course.run \
      --output "$RUN/$MODE" --plan ".research/protocols/plastic-projection-$MODE.json" --freeze-commit "$1" \
      2>&1 | tee "$RUN/$MODE.log"
done
printf '原始记录：%s\n' "$RUN"
