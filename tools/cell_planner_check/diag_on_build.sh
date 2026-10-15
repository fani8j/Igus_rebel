#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
name="$1"; shift
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
./stack.sh start $PWD/build/${name} $PWD/build/${name}.pgid; sleep 25
nice -n 10 python3 diag_look.py --build build/${name} "$@"
./stack.sh stop $PWD/build/${name}.pgid
