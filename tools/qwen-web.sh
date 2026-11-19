#!/bin/bash
# qwen-web — 82 上 gradio 出图服务的启停控制 (监听 tailscale IP:7860)
PY=$HOME/venvs/qwen-web/bin/python
APP=$HOME/build/qwen21/web/app.py
PIDF=$HOME/build/qwen21/web/web.pid
LOG=$HOME/build/qwen21/web/web.log

case "$1" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "已在运行 pid $(cat $PIDF)"; exit 0
    fi
    export PYTHONUNBUFFERED=1
    nohup "$PY" "$APP" > "$LOG" 2>&1 &
    echo $! > "$PIDF"
    sleep 4
    if kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "started pid $(cat $PIDF)"; tail -4 "$LOG"
    else
      echo "启动失败，日志尾部："; tail -25 "$LOG"
    fi ;;
  stop)
    [ -f "$PIDF" ] && kill "$(cat $PIDF)" 2>/dev/null && rm -f "$PIDF" && echo "stopped" || echo "未在运行" ;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "running pid $(cat $PIDF)"; ss -tlnp 2>/dev/null | grep 7860 || true
    else echo "not running"; fi ;;
  log) tail -f "$LOG" ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  *) echo "用法: qwen-web {start|stop|restart|status|log}" ;;
esac
