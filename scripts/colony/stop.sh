#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PIDFILE="logs/colony-server.pid"
[[ -f "$PIDFILE" ]] || { echo '没有本项目记录的服务进程。'; exit 0; }
pid="$(cat "$PIDFILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then echo '无效进程记录，未终止任何进程。' >&2; exit 1; fi
if ! kill -0 "$pid" 2>/dev/null; then rm -- "$PIDFILE"; echo '服务已停止。'; exit 0; fi
if [[ "$(readlink "/proc/$pid/cwd")" != "$ROOT" ]] || ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'mathhackson.colony.server:app'; then
    echo '进程归属不符，拒绝终止。' >&2
    exit 1
fi
kill -TERM "$pid"
for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then rm -- "$PIDFILE"; echo '本项目服务已停止。'; exit 0; fi
    sleep .2
done
echo '已发送停止信号；进程尚未退出，未强制终止。' >&2
exit 1
