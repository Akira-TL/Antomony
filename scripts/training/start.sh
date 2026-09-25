#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8766}"
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || { echo '端口必须在 1024 至 65535 之间。' >&2; exit 1; }
[[ "${MOTOR_CONTINUE:-0}" == 0 || "${MOTOR_CONTINUE:-0}" == 1 ]] || { echo 'MOTOR_CONTINUE 只能为 0 或 1。' >&2; exit 1; }
[[ -z "${MOTOR_CHECKPOINT:-}" || -f "$MOTOR_CHECKPOINT" ]] || { echo '动作快照不存在。' >&2; exit 1; }
[[ "${MOTOR_CONTINUE:-0}" != 1 || -n "${MOTOR_CHECKPOINT:-}" ]] || { echo '继续动作训练必须指定 MOTOR_CHECKPOINT。' >&2; exit 1; }
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
if [[ "${1:-}" == '--managed' ]]; then
    service_env=(--setenv "PORT=$PORT" --setenv "PATH=$PATH")
    if [[ -n "${MOTOR_CHECKPOINT:-}" ]]; then
        service_env+=(--setenv "MOTOR_CHECKPOINT=$MOTOR_CHECKPOINT")
        service_env+=(--setenv "MOTOR_CONTINUE=${MOTOR_CONTINUE:-0}")
    fi
    systemd-run --user --collect --unit "mathhackson-training-$PORT" \
        --working-directory "$ROOT" "${service_env[@]}" \
        --property "StandardOutput=append:$ROOT/$LOGFILE" \
        --property "StandardError=append:$ROOT/$LOGFILE" \
        /bin/bash "$ROOT/scripts/training/serve.sh"
    pid="$(systemctl --user show "mathhackson-training-$PORT" -p MainPID --value)"
else
    nohup bash scripts/training/serve.sh > "$LOGFILE" 2>&1 < /dev/null &
    pid=$!
fi
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
