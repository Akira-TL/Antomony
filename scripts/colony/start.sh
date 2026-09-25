#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
PORT="${PORT:-8765}"
PIDFILE="logs/colony-server.pid"
if [[ -f "$PIDFILE" ]]; then
    pid="$(cat "$PIDFILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && [[ "$(readlink "/proc/$pid/cwd")" == "$ROOT" ]]; then
        printf '服务已启动：http://localhost:%s\n' "$PORT"
        exit 0
    fi
fi
if ss -ltnH "sport = :$PORT" | grep -q .; then
    printf '端口 %s 已被占用；未停止或修改现有服务。\n' "$PORT" >&2
    exit 1
fi
if [[ ! -f web/dist/index.html ]]; then
    npm --prefix web ci
    npm --prefix web run build
fi
export PORT
nohup bash scripts/colony/serve.sh > logs/colony-server.log 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PIDFILE"
for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null | grep -q '"ready":true'; then
        printf '独立神经蚁群已启动：http://localhost:%s\n日志：%s/logs/colony-server.log\n' "$PORT" "$ROOT"
        exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        tail -30 logs/colony-server.log >&2
        exit 1
    fi
    sleep .5
done
printf '服务尚未通过健康检查，请查看 logs/colony-server.log。\n' >&2
exit 1
