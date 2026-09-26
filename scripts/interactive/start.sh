#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8775}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || { echo '端口无效'; exit 1; }
mkdir -p logs
PIDFILE="logs/interactive-$PORT.pid"
if [[ -f "$PIDFILE" ]]; then
    pid="$(< "$PIDFILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && [[ "$(readlink "/proc/$pid/cwd")" == "$ROOT" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'mathhackson.interactive.server:app'; then
        printf '交互验收：http://localhost:%s/interactive.html\n' "$PORT"
        exit 0
    fi
fi
if ss -ltnH "sport = :$PORT" | rg -q .; then
    echo "端口 $PORT 已占用，未修改现有服务。"; exit 1
fi
[[ -f web/dist/interactive.html ]] || { echo '请先执行 npm --prefix web run build'; exit 1; }
if [[ "${1:-}" == '--managed' ]]; then
    systemd-run --user --collect --unit "mathhackson-interactive-$PORT" \
        --working-directory "$ROOT" --setenv "PORT=$PORT" --setenv "PATH=$PATH" \
        --setenv "UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/mathhackson-uv-cache}" \
        --property "StandardOutput=append:$ROOT/logs/interactive-$PORT.log" \
        --property "StandardError=append:$ROOT/logs/interactive-$PORT.log" \
        /bin/bash "$ROOT/scripts/interactive/serve.sh"
    pid="$(systemctl --user show "mathhackson-interactive-$PORT" -p MainPID --value)"
else
    PORT="$PORT" nohup bash scripts/interactive/serve.sh > "logs/interactive-$PORT.log" 2>&1 < /dev/null &
    pid=$!
fi
printf '%s\n' "$pid" > "$PIDFILE"
for _ in $(seq 1 40); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null | rg -q '^true$'; then
        printf '交互验收：http://localhost:%s/interactive.html\n' "$PORT"; exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then tail -20 "logs/interactive-$PORT.log"; exit 1; fi
    sleep .5
done
echo "服务未就绪，请查看 logs/interactive-$PORT.log"; exit 1
