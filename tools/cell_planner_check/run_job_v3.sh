#!/usr/bin/env bash
# run a v3 job (cellcheck_v3.py) against the stack already up on domain 187; records the python PID in <out>/run.pid
#   ./run_job_v3.sh <job.json> <build_dir> <out_dir> [extra args]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
job="$1"; build="$2"; out="$3"; shift 3
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
mkdir -p "$out"
cd "$HERE"
setsid nice -n 19 python3 -u "$HERE/cellcheck_v3.py" --job "$job" --build "$build" --out "$out" "$@" > "$out/run.log" 2>&1 &
echo $! > "$out/run.pid"; echo "started pid $(cat "$out/run.pid")  log $out/run.log"
