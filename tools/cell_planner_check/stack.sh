#!/usr/bin/env bash
# Start / stop the planning-only stack on ROS domain 187 (localhost only).  Records the process group in <pidfile>; stop kills ONLY that group.
#   ./stack.sh start <build_dir> [pidfile] [extra launch args...]
#   ./stack.sh stop  [pidfile]
#   ./stack.sh status [pidfile]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
cmd="${1:-}"; shift || true
case "$cmd" in
  start)
    build="$1"; shift
    pidfile="${1:-$HERE/build/.stack.pgid}"; [ $# -gt 0 ] && shift
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "already running (pgid $(cat "$pidfile"))"; exit 1; fi
    source /opt/ros/humble/setup.bash
    source /home/farhan/igus_rebel_ros2/install/setup.bash
    export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
    logdir="$(dirname "$pidfile")"; mkdir -p "$logdir"
    setsid nice -n 10 ros2 launch "$HERE/cell_planning.launch.py" "build_dir:=$build" "$@" > "${pidfile%.pgid}.log" 2>&1 &
    pid=$!
    echo "$pid" > "$pidfile"       # setsid makes the child a session/group leader: pgid == pid
    echo "$(date +%F_%T) START pgid=$pid build=$build" >> /home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003/process_ledger.txt
    echo "started pgid $pid  (log ${pidfile%.pgid}.log)"
    ;;
  stop)
    pidfile="${1:-$HERE/build/.stack.pgid}"
    if [ -f "$pidfile" ]; then
      pg="$(cat "$pidfile")"
      if kill -0 "$pg" 2>/dev/null; then
        kill -INT -- "-$pg" 2>/dev/null; for i in $(seq 1 20); do kill -0 "$pg" 2>/dev/null || break; sleep 0.5; done
        kill -TERM -- "-$pg" 2>/dev/null; sleep 1
        kill -KILL -- "-$pg" 2>/dev/null
      fi
      rm -f "$pidfile"; echo "stopped pgid $pg"; echo "$(date +%F_%T) STOP  pgid=$pg members_left=$(ps -eo pgid= | awk -v g="$pg" '$1==g' | wc -l)" >> /home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003/process_ledger.txt
    else echo "no pidfile"; fi
    ;;
  status)
    pidfile="${1:-$HERE/build/.stack.pgid}"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "running pgid $(cat "$pidfile")"; ps -o pid,pgid,ni,etime,cmd -g "$(cat "$pidfile")" | cut -c1-150; else echo "not running"; fi
    ;;
  *) echo "usage: $0 start <build_dir> [pidfile] | stop [pidfile] | status [pidfile]"; exit 2;;
esac
