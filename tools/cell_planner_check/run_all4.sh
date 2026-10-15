#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003
cd "$HERE"
until grep -q "queue3 finished" $R/run_all3.log; do sleep 5; done
run() { echo "$(date +%H:%M:%S) ==== $1"; ./run_on_build.sh "$@"; }
run h000n.0_proxy $R/h000n.0_proxy_final "jobs/h000n.0_v1.json"
echo "$(date +%H:%M:%S) queue4 finished"
