#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8772}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || { echo '端口无效。' >&2; exit 1; }
[[ -z "${ROUNDTRIP_DEMO_DIR:-}" || -d "$ROUNDTRIP_DEMO_DIR" ]] || { echo '演示快照目录不存在。' >&2; exit 1; }
[[ -z "${ROUNDTRIP_DEMO_SOURCE:-}" || -f "$ROUNDTRIP_DEMO_SOURCE" ]] || { echo '局部气味检查点不存在。' >&2; exit 1; }
mkdir -p logs
PIDFILE="logs/roundtrip-server-$PORT.pid"
LOGFILE="logs/roundtrip-server-$PORT.log"
if [[ -f "$PIDFILE" ]]; then
    pid="$(< "$PIDFILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && [[ "$(readlink "/proc/$pid/cwd")" == "$ROOT" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'mathhackson.training.roundtrip_server:app'; then
        printf '往返验收：http://localhost:%s/roundtrip.html\n' "$PORT"
        exit 0
    fi
fi
if ss -ltnH "sport = :$PORT" | rg -q .; then
    printf '端口 %s 已占用；现有服务未受影响。\n' "$PORT" >&2
    exit 1
fi
[[ -f web/dist/roundtrip.html ]] || { echo '请先执行 npm --prefix web run build。' >&2; exit 1; }
if [[ "${1:-}" == '--managed' ]]; then
    service_env=(--setenv "PORT=$PORT" --setenv "PATH=$PATH")
    if [[ -n "${ROUNDTRIP_DEMO_DIR:-}" ]]; then
        service_env+=(--setenv "ROUNDTRIP_DEMO_DIR=$ROUNDTRIP_DEMO_DIR")
    fi
    if [[ -n "${ROUNDTRIP_DEMO_SOURCE:-}" ]]; then
        service_env+=(--setenv "ROUNDTRIP_DEMO_SOURCE=$ROUNDTRIP_DEMO_SOURCE")
    fi
    systemd-run --user --collect --unit "mathhackson-roundtrip-$PORT" \
        --working-directory "$ROOT" "${service_env[@]}" \
        --property "StandardOutput=append:$ROOT/$LOGFILE" \
        --property "StandardError=append:$ROOT/$LOGFILE" \
        /bin/bash "$ROOT/scripts/training/roundtrip-serve.sh"
    pid="$(systemctl --user show "mathhackson-roundtrip-$PORT" -p MainPID --value)"
else
    PORT="$PORT" nohup bash scripts/training/roundtrip-serve.sh > "$LOGFILE" 2>&1 < /dev/null &
    pid=$!
fi
printf '%s\n' "$pid" > "$PIDFILE"
for _ in $(seq 1 40); do
    if curl -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null | rg -q '^true$'; then
        printf '往返验收：http://localhost:%s/roundtrip.html\n日志：%s/%s\n' "$PORT" "$ROOT" "$LOGFILE"
        exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then tail -30 "$LOGFILE" >&2; exit 1; fi
    sleep .5
done
printf '往返服务尚未就绪，请查看 %s。\n' "$LOGFILE" >&2
exit 1
