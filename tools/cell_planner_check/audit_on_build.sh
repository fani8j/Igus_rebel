#!/usr/bin/env bash
# start the stack on an existing build, run path_audit.py for one job, stop.   ./audit_on_build.sh <build_name> <job.json> <out.json>
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="$1"; job="$2"; out="$3"
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
cd "$HERE"
./stack.sh start $PWD/build/${name} $PWD/build/${name}.pgid; sleep 25
nice -n 10 python3 path_audit.py --job "$job" --build build/${name} --out "$out" > "${out%.json}.log" 2>&1
./stack.sh stop $PWD/build/${name}.pgid
echo "$(date +%H:%M:%S) audit $name finished"
