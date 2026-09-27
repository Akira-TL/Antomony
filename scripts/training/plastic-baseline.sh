#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ "$#" -ne 1 ]]; then
    printf '%s\n' '用法：bash scripts/training/plastic-baseline.sh <冻结提交>' >&2
    exit 2
fi
if [[ "$(date -u +%Y%m%d%H%M%S)" > "20260927004300" ]]; then
    printf '%s\n' '超过预定最晚启动时点，本批未实施' >&2
    exit 2
fi
RUN="logs/plastic-baseline/$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$RUN"
printf '原始记录：%s\n' "$RUN"
for MODE in history paired; do
    timeout --signal=KILL 380s uv run --no-sync --offline python -m mathhackson.training.foraging.plastic_course.run \
      --output "$RUN/$MODE" --plan ".research/protocols/plastic-baseline-$MODE.json" --freeze-commit "$1" \
      2>&1 | tee "$RUN/$MODE.log"
done
printf '原始记录：%s\n' "$RUN"
