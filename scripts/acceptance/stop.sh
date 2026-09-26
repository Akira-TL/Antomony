#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8774}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || { echo '端口无效'; exit 1; }
PIDFILE="logs/acceptance-$PORT.pid"
[[ -f "$PIDFILE" ]] || exit 0
pid="$(< "$PIDFILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    [[ "$(readlink "/proc/$pid/cwd")" == "$ROOT" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'mathhackson.training.comparison.acceptance:app' || { echo 'PID不属于本验收服务，拒绝停止'; exit 1; }
    kill "$pid"
fi
rm -- "$PIDFILE"
