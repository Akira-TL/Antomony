#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8775}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || { echo '端口无效'; exit 1; }
PIDFILE="logs/interactive-$PORT.pid"
[[ -f "$PIDFILE" ]] || exit 0
pid="$(< "$PIDFILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    [[ "$(readlink "/proc/$pid/cwd")" == "$ROOT" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'mathhackson.interactive.server:app' || { echo 'PID不属于本验收服务，拒绝停止'; exit 1; }
    kill "$pid"
    for _ in $(seq 1 40); do
        kill -0 "$pid" 2>/dev/null || break
        sleep .25
    done
    if kill -0 "$pid" 2>/dev/null; then echo '仍在保存记录，未强制结束'; exit 1; fi
fi
rm -- "$PIDFILE"
