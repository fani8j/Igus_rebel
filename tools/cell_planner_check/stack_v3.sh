#!/usr/bin/env bash
# v3 copy of stack.sh: identical logic, but the process ledger goes to scan_results/planner_check_v3_20261004/ and nice is 19.
#   ./stack_v3.sh start <build_dir> <pidfile> ;  ./stack_v3.sh stop <pidfile> ;  ./stack_v3.sh status <pidfile>
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEDGER=/home/farhan/igus_rebel_ros2/scan_results/planner_check_v3_20261004/process_ledger.txt
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
cmd="${1:-}"; shift || true
case "$cmd" in
  start)
    build="$1"; shift
    pidfile="$1"; shift
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "already running (pgid $(cat "$pidfile"))"; exit 1; fi
    source /opt/ros/humble/setup.bash
    source /home/farhan/igus_rebel_ros2/install/setup.bash
    export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
    mkdir -p "$(dirname "$pidfile")"
    setsid nice -n 19 ros2 launch "$HERE/cell_planning.launch.py" "build_dir:=$build" "$@" > "${pidfile%.pgid}.log" 2>&1 &
    pid=$!
    echo "$pid" > "$pidfile"
    echo "$(date +%F_%T) START pgid=$pid build=$build" >> "$LEDGER"
    echo "started pgid $pid  (log ${pidfile%.pgid}.log)"
    ;;
  stop)
    pidfile="$1"
    if [ -f "$pidfile" ]; then
      pg="$(cat "$pidfile")"
      if kill -0 "$pg" 2>/dev/null; then
        kill -INT -- "-$pg" 2>/dev/null; for i in $(seq 1 20); do kill -0 "$pg" 2>/dev/null || break; sleep 0.5; done
        kill -TERM -- "-$pg" 2>/dev/null; sleep 1
        kill -KILL -- "-$pg" 2>/dev/null
      fi
      rm -f "$pidfile"; echo "stopped pgid $pg"; echo "$(date +%F_%T) STOP  pgid=$pg members_left=$(ps -eo pgid= | awk -v g="$pg" '$1==g' | wc -l)" >> "$LEDGER"
    else echo "no pidfile"; fi
    ;;
  status)
    pidfile="$1"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "running pgid $(cat "$pidfile")"; ps -o pid,pgid,ni,etime,cmd -g "$(cat "$pidfile")" | cut -c1-150; else echo "not running"; fi
    ;;
  *) echo "usage: $0 start <build_dir> <pidfile> | stop <pidfile> | status <pidfile>"; exit 2;;
esac
