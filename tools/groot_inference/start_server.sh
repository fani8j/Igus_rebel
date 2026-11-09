#!/usr/bin/env bash
# Start/stop/inspect the GR00T N1.7 policy server for the rebel front-lid checkpoint.
# Runs as a transient systemd --user unit (survives the calling shell / Claude session).
#
#   start_server.sh start   [--port N] [--host H] [--checkpoint DIR]
#   start_server.sh stop
#   start_server.sh restart [...]
#   start_server.sh status        (unit state + GPU memory of the server process)
#   start_server.sh ping    [--port N]   (exit 0 if the server answers "ping")
#   start_server.sh logs          (tail of the server log)
#
# No ROS involved. Everything bulky lives on the external drive via ~/groot.
set -euo pipefail

GROOT_HOME="${GROOT_HOME:-$HOME/groot}"
REPO="$GROOT_HOME/Isaac-GR00T"
PY="$REPO/.venv/bin/python"                       # use the venv directly, NOT `uv run` (uv re-installs flash-attn on every run)
UNIT="${GROOT_UNIT:-groot-server}"
CHECKPOINT="${GROOT_CHECKPOINT:-$GROOT_HOME/checkpoints/rebel_open_frontlid_real_12ep/checkpoint-2000}"
HOST="${GROOT_HOST:-127.0.0.1}"
PORT="${GROOT_PORT:-5555}"
EMBODIMENT="${GROOT_EMBODIMENT:-new_embodiment}"
LOG="$GROOT_HOME/logs/server.log"
READY_TIMEOUT_S="${GROOT_READY_TIMEOUT_S:-1500}"   # loading 12.5 GB from the USB drive can take ~11 min

cmd="${1:-status}"; shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

ping_server() {   # raw ZeroMQ REQ + msgpack, same protocol as gr00t PolicyClient / the friend's Gr00tClient
  "$PY" - "$HOST" "$PORT" <<'PY'
import sys, zmq, msgpack
host, port = sys.argv[1], sys.argv[2]
ctx = zmq.Context(); s = ctx.socket(zmq.REQ)
s.setsockopt(zmq.RCVTIMEO, 5000); s.setsockopt(zmq.SNDTIMEO, 5000); s.setsockopt(zmq.LINGER, 0)
s.connect(f"tcp://{host}:{port}")
try:
    s.send(msgpack.packb({"endpoint": "ping"}))
    r = msgpack.unpackb(s.recv(), raw=False)
    print("ping:", r); sys.exit(0 if r.get("status") == "ok" else 1)
except Exception as e:
    print("ping failed:", e); sys.exit(1)
PY
}

is_active() { systemctl --user is-active --quiet "$UNIT"; }

case "$cmd" in
  start)
    if is_active; then echo "$UNIT already running"; exit 0; fi
    [ -d "$CHECKPOINT" ] || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 1; }
    [ -x "$PY" ] || { echo "venv python missing: $PY" >&2; exit 1; }
    mkdir -p "$GROOT_HOME/logs" "$GROOT_HOME/tmp"
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    # Backbone nvidia/Cosmos-Reason2-2B is pre-downloaded into HF_HOME (token in HF_HOME/token). Do NOT set
    # HF_HUB_OFFLINE=1: transformers 4.57.3 tokenizer loading then raises OfflineModeIsEnabled (model_info call).
    systemd-run --user --unit="$UNIT" --collect --description="GR00T N1.7 policy server (rebel front lid)" \
      --property=StandardOutput=append:"$LOG" --property=StandardError=append:"$LOG" \
      --setenv=HF_HOME="$GROOT_HOME/hf_home" \
      --setenv=TMPDIR="$GROOT_HOME/tmp" --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
      --setenv=TOKENIZERS_PARALLELISM=false \
      --working-directory="$REPO" \
      "$PY" gr00t/eval/run_gr00t_server.py \
        --model-path "$CHECKPOINT" --embodiment-tag "$EMBODIMENT" --host "$HOST" --port "$PORT"
    echo "started $UNIT (log: $LOG); waiting for ping (model load takes ~1-3 min) ..."
    t0=$(date +%s)
    until ping_server >/dev/null 2>&1; do
      is_active || { echo "server died during startup, see $LOG" >&2; tail -30 "$LOG" >&2; exit 1; }
      [ $(( $(date +%s) - t0 )) -lt "$READY_TIMEOUT_S" ] || { echo "timeout waiting for server" >&2; exit 1; }
      sleep 3
    done
    ping_server
    ;;
  stop)
    systemctl --user stop "$UNIT" 2>/dev/null && echo "stopped $UNIT" || echo "$UNIT not running"
    ;;
  restart)
    "$0" stop; "$0" start --port "$PORT" --host "$HOST" --checkpoint "$CHECKPOINT"
    ;;
  status)
    if is_active; then
      pid=$(systemctl --user show -p MainPID --value "$UNIT")
      echo "$UNIT: active (MainPID $pid), tcp://$HOST:$PORT"
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep "^$pid," || true
      ping_server || true
    else
      echo "$UNIT: not running"; exit 3
    fi
    ;;
  ping) ping_server;;
  logs) tail -n 60 "$LOG";;
  *) echo "usage: $0 start|stop|restart|status|ping|logs [--port N] [--host H] [--checkpoint DIR]" >&2; exit 2;;
esac
