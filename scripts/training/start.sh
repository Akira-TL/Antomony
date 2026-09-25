#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8766}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || { echo '端口必须在 1024 至 65535 之间。' >&2; exit 1; }
mkdir -p logs
PIDFILE="logs/training-server-$PORT.pid"
LOGFILE="logs/training-server-$PORT.log"
if [[ -f "$PIDFILE" ]]; then
    pid="$(cat "$PIDFILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && [[ "$(readlink "/proc/$pid/cwd")" == "$ROOT" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'mathhackson.training.server:app'; then
        printf '训练服务已启动：http://localhost:%s/training.html\n' "$PORT"
        exit 0
    fi
fi
if ss -ltnH "sport = :$PORT" | rg -q .; then
    printf '端口 %s 已占用；请使用 PORT 指定空闲端口，现有服务未受影响。\n' "$PORT" >&2
    exit 1
fi
[[ -f web/dist/training.html ]] || { echo '请先执行 npm --prefix web run build。' >&2; exit 1; }
export PORT
nohup bash scripts/training/serve.sh > "$LOGFILE" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PIDFILE"
for _ in $(seq 1 40); do
    if curl -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null | rg -q '^true$'; then
        printf '单蚁训练：http://localhost:%s/training.html\n日志：%s/%s\n' "$PORT" "$ROOT" "$LOGFILE"
        exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then tail -30 "$LOGFILE" >&2; exit 1; fi
    sleep .5
done
printf '训练服务尚未就绪，请查看 %s。\n' "$LOGFILE" >&2
exit 1
