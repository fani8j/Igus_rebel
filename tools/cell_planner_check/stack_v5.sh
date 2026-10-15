#!/usr/bin/env bash
# v5 copy of stack_v4.sh: ledger in scan_results/planner_check_v5_20261007/, nice 19, domain 187 localhost only.  The process-group id is recorded from INSIDE the new session
# (setsid bash -c 'echo $$ > pidfile; exec ...'), plus the boot time (uptime -s) next to it; `stop` only acts on a group whose recorded boot time equals the current one and
# whose leader is still our launch (never kills by name / pattern).
#   ./stack_v5.sh start <build_dir> <pidfile> ;  ./stack_v5.sh stop <pidfile> ;  ./stack_v5.sh status <pidfile>
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEDGER=/home/farhan/igus_rebel_ros2/scan_results/planner_check_v5_20261007/process_ledger.txt
export V5_DOMAIN=${V5_DOMAIN:-187}; case $V5_DOMAIN in 187|188) ;; *) echo 'V5_DOMAIN must be 187 or 188'; exit 2;; esac
export ROS_DOMAIN_ID=$V5_DOMAIN ROS_LOCALHOST_ONLY=1
cmd="${1:-}"; shift || true
alive() { [ -f "$1" ] && [ "$(cat "$1.boot" 2>/dev/null)" = "$(uptime -s)" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
case "$cmd" in
  start)
    build="$1"; shift
    pidfile="$1"; shift
    if alive "$pidfile"; then echo "already running (pgid $(cat "$pidfile"))"; exit 1; fi
    mkdir -p "$(dirname "$pidfile")"; mkdir -p "$(dirname "$LEDGER")"
    uptime -s > "$pidfile.boot"
    log="${pidfile%.pgid}.log"
    setsid bash -c 'echo $$ > "$1"; shift; exec nice -n 19 bash -c "source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash; export ROS_DOMAIN_ID=$V5_DOMAIN ROS_LOCALHOST_ONLY=1; exec ros2 launch \"\$@\"" _ "$@"' _ "$pidfile" "$HERE/cell_planning.launch.py" "build_dir:=$build" "$@" > "$log" 2>&1 < /dev/null &
    for i in $(seq 1 20); do [ -s "$pidfile" ] && break; sleep 0.25; done
    echo "$(date +%F_%T) START domain=$V5_DOMAIN pgid=$(cat "$pidfile") boot=$(cat "$pidfile.boot") build=$build" >> "$LEDGER"
    echo "started pgid $(cat "$pidfile")  (log $log)"
    ;;
  stop)
    pidfile="$1"
    if alive "$pidfile"; then
      pg="$(cat "$pidfile")"
      kill -INT -- "-$pg" 2>/dev/null; for i in $(seq 1 20); do kill -0 "$pg" 2>/dev/null || break; sleep 0.5; done
      kill -TERM -- "-$pg" 2>/dev/null; sleep 1
      kill -KILL -- "-$pg" 2>/dev/null
      rm -f "$pidfile" "$pidfile.boot"; echo "stopped pgid $pg"; echo "$(date +%F_%T) STOP  pgid=$pg members_left=$(ps -eo pgid= | awk -v g="$pg" '$1==g' | wc -l)" >> "$LEDGER"
    else echo "no live pidfile of this boot (stale files are NOT acted on)"; fi
    ;;
  status)
    pidfile="$1"
    if alive "$pidfile"; then echo "running pgid $(cat "$pidfile")"; ps -o pid,pgid,ni,etime,cmd -g "$(cat "$pidfile")" | cut -c1-150; else echo "not running"; fi
    ;;
  *) echo "usage: $0 start <build_dir> <pidfile> | stop <pidfile> | status <pidfile>"; exit 2;;
esac
