#!/bin/bash
# qwen-web — Gradio 面板启停控制 (监听 Tailscale IP:7860)
PY=${QDIT_WEB_PYTHON:-python3}
APP=${QDIT_WEB_APP:-"$(dirname "${BASH_SOURCE[0]}")/../src/qwen21dit/web.py"}
STATE_DIR=${QDIT_WEB_STATE_DIR:-"${XDG_STATE_HOME:-$HOME/.local/state}/qwen21dit"}
PIDF="$STATE_DIR/web.pid"
LOG="$STATE_DIR/web.log"

case "$1" in
  start)
    mkdir -p "$STATE_DIR" || exit 1
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "已在运行 pid $(cat "$PIDF")"; exit 0
    fi
    export PYTHONUNBUFFERED=1
    nohup "$PY" "$APP" > "$LOG" 2>&1 &
    echo $! > "$PIDF"
    sleep 4
    if kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "started pid $(cat "$PIDF")"; tail -4 "$LOG"
    else
      echo "启动失败，日志尾部："; tail -25 "$LOG"
    fi ;;
  stop)
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null && rm -f "$PIDF" && echo "stopped" || echo "未在运行" ;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "running pid $(cat "$PIDF")"; ss -tlnp 2>/dev/null | grep 7860 || true
    else echo "not running"; fi ;;
  log) tail -f "$LOG" ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  *) echo "用法: qwen-web {start|stop|restart|status|log}" ;;
esac
