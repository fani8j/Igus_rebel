#!/usr/bin/env bash
# Start/stop/inspect the TurboVLA policy server (default: the bundle's LEFT-arm mocap10k checkpoint; profile right19: RIGHT arm).
# Runs as a transient systemd --user unit (survives the calling shell / Claude session).
#
#   start_turbovla_server.sh start   [--profile NAME] [--port N] [--host H] [--bundle DIR]
#                                    [--checkpoint FILE --language TEXT --expect-sha256 HEX]
#   start_turbovla_server.sh stop
#   start_turbovla_server.sh restart [same args as start]
#   start_turbovla_server.sh status        (unit state + GPU memory of the server process)
#   start_turbovla_server.sh ping    [--port N]   (exit 0 if the server answers "ping")
#   start_turbovla_server.sh info    [--port N]   (print get_modality_config: arm_channels, language, checkpoint sha256)
#   start_turbovla_server.sh logs
#
# Profiles live next to this script in profiles/NAME.profile (KEY=value: CHECKPOINT, LANGUAGE, EXPECT_SHA256; @TV_HOME@ expands).
# Default profile "bundle" = the bundle checkpoint (MANIFEST pin, language 'Open the box flaps.').  Any checkpoint outside the bundle
# needs both a language and an --expect-sha256 (the server refuses to start otherwise).  Explicit flags override the profile.
# One server at a time: the unit name (turbovla-server) and the port (5556) are shared by all profiles.
#
# No ROS involved. Everything bulky lives on the external drive via ~/turbovla.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TV_HOME="${TURBOVLA_HOME:-$HOME/turbovla}"
BUNDLE="${TURBOVLA_BUNDLE:-$TV_HOME/compal-turbovla-handoff-v1}"
PY="$BUNDLE/.venv/bin/python"
DEPS="${TURBOVLA_DEPS:-$TV_HOME/server_deps}"     # pyzmq/msgpack/msgpack-numpy/pyarrow/pandas, kept out of the bundle venv
UNIT="${TURBOVLA_UNIT:-turbovla-server}"
HOST="${TURBOVLA_HOST:-127.0.0.1}"
PORT="${TURBOVLA_PORT:-5556}"
LOG="$TV_HOME/logs/server.log"
READY_TIMEOUT_S="${TURBOVLA_READY_TIMEOUT_S:-600}"
PROFILE="${TURBOVLA_PROFILE:-bundle}"
CHECKPOINT="" LANGUAGE_TEXT="${TURBOVLA_LANGUAGE:-}" EXPECT_SHA256="${TURBOVLA_EXPECT_SHA256:-}"

cmd="${1:-status}"; shift || true
ORIG_ARGS=("$@")
EXPLICIT_CKPT="" EXPLICIT_LANG="" EXPLICIT_SHA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --bundle) BUNDLE="$2"; PY="$BUNDLE/.venv/bin/python"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --checkpoint) EXPLICIT_CKPT="$2"; shift 2;;
    --language) EXPLICIT_LANG="$2"; shift 2;;
    --expect-sha256) EXPLICIT_SHA="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

load_profile() {   # parse (never source) profiles/NAME.profile: KEY=value lines, '#' comments
  case "$PROFILE" in *[!A-Za-z0-9_-]*|"") echo "bad profile name: $PROFILE" >&2; exit 2;; esac
  local f="$HERE/profiles/$PROFILE.profile" key value
  [ -f "$f" ] || { echo "no such profile: $f (have: $(ls "$HERE/profiles" 2>/dev/null | sed 's/\.profile$//' | tr '\n' ' '))" >&2; exit 2; }
  while IFS='=' read -r key value; do
    case "$key" in ''|\#*) continue;; esac
    value="${value//@TV_HOME@/$TV_HOME}"
    case "$key" in
      CHECKPOINT) CHECKPOINT="$value";;
      LANGUAGE) LANGUAGE_TEXT="$value";;
      EXPECT_SHA256) EXPECT_SHA256="$value";;
      *) echo "profile $f: unknown key $key" >&2; exit 2;;
    esac
  done < "$f"
  [ -n "$EXPLICIT_CKPT" ] && CHECKPOINT="$EXPLICIT_CKPT"
  [ -n "$EXPLICIT_LANG" ] && LANGUAGE_TEXT="$EXPLICIT_LANG"
  [ -n "$EXPLICIT_SHA" ] && EXPECT_SHA256="$EXPLICIT_SHA"
  SERVER_ARGS=()
  [ -n "$CHECKPOINT" ] && SERVER_ARGS+=(--checkpoint "$CHECKPOINT")
  [ -n "$LANGUAGE_TEXT" ] && SERVER_ARGS+=(--language "$LANGUAGE_TEXT")
  [ -n "$EXPECT_SHA256" ] && SERVER_ARGS+=(--expect-sha256 "$EXPECT_SHA256")
  # a non-bundle checkpoint must be fully pinned: fail here with a clear message (the server enforces it again)
  if [ -n "$CHECKPOINT" ] && { [ -z "$LANGUAGE_TEXT" ] || [ -z "$EXPECT_SHA256" ]; }; then
    echo "checkpoint $CHECKPOINT needs both a language and an expect-sha256 (profile or --language / --expect-sha256)" >&2; exit 2
  fi
}

ping_server() {   # raw ZeroMQ REQ + msgpack, same protocol as the GR00T PolicyClient
  PYTHONPATH="$DEPS" "$PY" - "$HOST" "$PORT" <<'PY'
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

info_server() {   # get_modality_config of the running server ($1 = brief: one line)
  PYTHONPATH="$DEPS" "$PY" - "$HOST" "$PORT" "${1:-full}" <<'PY'
import sys, zmq, msgpack, json
host, port, brief = sys.argv[1], sys.argv[2], sys.argv[3] == "brief"
ctx = zmq.Context(); s = ctx.socket(zmq.REQ)
s.setsockopt(zmq.RCVTIMEO, 5000); s.setsockopt(zmq.SNDTIMEO, 5000); s.setsockopt(zmq.LINGER, 0)
s.connect(f"tcp://{host}:{port}")
try:
    s.send(msgpack.packb({"endpoint": "get_modality_config"}))
    r = msgpack.unpackb(s.recv(), raw=False)
    print(f"arm={r.get('arm')} arm_channels={r.get('arm_channels')} language={r.get('language')!r} checkpoint_sha256={r.get('checkpoint_sha256')}"
          if brief else json.dumps(r, indent=1))
except Exception as e:
    print("get_modality_config failed:", e); sys.exit(1)
PY
}

is_active() { systemctl --user is-active --quiet "$UNIT"; }

case "$cmd" in
  start)
    if is_active; then echo "$UNIT already running (stop it first to switch profile)"; exit 0; fi
    load_profile
    [ -x "$PY" ] || { echo "bundle venv python missing (run bundle.py setup): $PY" >&2; exit 1; }
    [ -d "$DEPS" ] || { echo "server deps missing: $DEPS (see README.md)" >&2; exit 1; }
    mkdir -p "$TV_HOME/logs" "$TV_HOME/tmp"
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    # Offline model assets + bundle source paths mirror bundle.py environment(); the server script also sets them.
    systemd-run --user --unit="$UNIT" --collect --description="TurboVLA policy server (Compal, profile $PROFILE)" \
      --property=StandardOutput=append:"$LOG" --property=StandardError=append:"$LOG" \
      --setenv=PYTHONPATH="$DEPS" \
      --setenv=TMPDIR="$TV_HOME/tmp" --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
      --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 --setenv=WANDB_MODE=disabled \
      --setenv=TOKENIZERS_PARALLELISM=false --setenv=OMP_NUM_THREADS=4 \
      --setenv=TURBOVLA_BUNDLE="$BUNDLE" \
      --working-directory="$BUNDLE/source" \
      nice -n 10 "$PY" "$HERE/turbovla_server.py" --bundle "$BUNDLE" --host "$HOST" --port "$PORT" "${SERVER_ARGS[@]}"
    echo "started $UNIT profile=$PROFILE (log: $LOG); waiting for ping (model load takes ~1-3 min) ..."
    t0=$(date +%s)
    until ping_server >/dev/null 2>&1; do
      is_active || { echo "server died during startup, see $LOG" >&2; tail -30 "$LOG" >&2; exit 1; }
      [ $(( $(date +%s) - t0 )) -lt "$READY_TIMEOUT_S" ] || { echo "timeout waiting for server" >&2; exit 1; }
      sleep 3
    done
    ping_server
    info_server brief
    ;;
  stop)
    systemctl --user stop "$UNIT" 2>/dev/null && echo "stopped $UNIT" || echo "$UNIT not running"
    ;;
  restart)
    "$0" stop; "$0" start "${ORIG_ARGS[@]}"
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
  info) info_server;;
  logs) tail -n 60 "$LOG";;
  *) echo "usage: $0 start|stop|restart|status|ping|info|logs [--profile NAME] [--port N] [--host H] [--bundle DIR] [--checkpoint F --language T --expect-sha256 H]" >&2; exit 2;;
esac
