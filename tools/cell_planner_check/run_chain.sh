#!/usr/bin/env bash
# chain: wait for the running main job, run the extra yaws + IK audit on the SAME stack, then stop it and run the queue.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003
cd "$HERE"
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
while kill -0 "$1" 2>/dev/null; do sleep 3; done
echo "$(date +%H:%M:%S) main job finished"
python3 summarize.py $R/h000n.0/result_h000n.0_v1.json --out $R/h000n.0/summary.md > /dev/null
nice -n 10 python3 -u cellcheck.py --job jobs/h000n.0_v1_yaw180_345.json --build build/h000n.0_s0355 --out $R/h000n.0_yaw180_345 > $R/h000n.0_yaw180_345.log 2>&1
echo "$(date +%H:%M:%S) yaw180_345 done"
python3 summarize.py $R/h000n.0_yaw180_345/result_*.json --out $R/h000n.0_yaw180_345/summary.md > /dev/null
nice -n 10 python3 ik_audit.py --job jobs/h000n.0_v1.json --build build/h000n.0_s0355 --out $R/ik_audit_h000n.0.json > $R/ik_audit_h000n.0.log 2>&1
echo "$(date +%H:%M:%S) ik audit done"
./run_all.sh "" $PWD/build/stackA.pgid
