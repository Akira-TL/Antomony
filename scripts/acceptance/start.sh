#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8774}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || { echo '端口无效'; exit 1; }
mkdir -p logs
PIDFILE="logs/acceptance-$PORT.pid"
if [[ -f "$PIDFILE" ]]; then
    pid="$(< "$PIDFILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && [[ "$(readlink "/proc/$pid/cwd")" == "$ROOT" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'mathhackson.training.comparison.acceptance:app'; then
        printf '只读验收：http://localhost:%s/acceptance.html\n' "$PORT"
        exit 0
    fi
fi
if ss -ltnH "sport = :$PORT" | rg -q .; then
    echo "端口 $PORT 已占用，未修改现有服务。"; exit 1
fi
[[ -f web/dist/acceptance.html ]] || { echo '请先执行 npm --prefix web run build'; exit 1; }
if [[ "${1:-}" == '--managed' ]]; then
    systemd-run --user --collect --unit "mathhackson-acceptance-$PORT" \
        --working-directory "$ROOT" --setenv "PORT=$PORT" --setenv "PATH=$PATH" \
        --setenv "UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/mathhackson-uv-cache}" \
        --property "StandardOutput=append:$ROOT/logs/acceptance-$PORT.log" \
        --property "StandardError=append:$ROOT/logs/acceptance-$PORT.log" \
        /bin/bash "$ROOT/scripts/acceptance/serve.sh"
    pid="$(systemctl --user show "mathhackson-acceptance-$PORT" -p MainPID --value)"
else
    PORT="$PORT" nohup bash scripts/acceptance/serve.sh > "logs/acceptance-$PORT.log" 2>&1 < /dev/null &
    pid=$!
fi
printf '%s\n' "$pid" > "$PIDFILE"
for _ in $(seq 1 40); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null | rg -q '^true$'; then
        printf '只读验收：http://localhost:%s/acceptance.html\n' "$PORT"; exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then tail -20 "logs/acceptance-$PORT.log"; exit 1; fi
    sleep .5
done
echo "服务未就绪，请查看 logs/acceptance-$PORT.log"; exit 1
