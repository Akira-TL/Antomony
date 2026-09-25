#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8766}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || { echo '端口无效。' >&2; exit 1; }
PIDFILE="logs/training-server-$PORT.pid"
[[ -f "$PIDFILE" ]] || { echo '没有该训练服务的进程记录。'; exit 0; }
pid="$(cat "$PIDFILE")"
[[ "$pid" =~ ^[0-9]+$ ]] || { echo '无效进程记录。' >&2; exit 1; }
if ! kill -0 "$pid" 2>/dev/null; then rm -- "$PIDFILE"; exit 0; fi
if [[ "$(readlink "/proc/$pid/cwd")" != "$ROOT" ]] || ! tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'mathhackson.training.server:app'; then
    echo '进程归属不符，拒绝停止。' >&2
    exit 1
fi
kill -TERM "$pid"
for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then rm -- "$PIDFILE"; echo '训练服务已停止。'; exit 0; fi
    sleep .2
done
echo '已发送停止信号，尚未退出，未强制终止。' >&2
exit 1
