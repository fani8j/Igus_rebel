#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003
cd "$HERE"
until grep -q "queue2 finished" $R/run_all2.log; do sleep 5; done
run() { echo "$(date +%H:%M:%S) ==== $1"; ./run_on_build.sh "$@"; }
run v2_E5_s046 $R/v2_E5_final "jobs/v2_E5_nominal.json jobs/v2_E5_offsets.json jobs/v2_E5_small_single2.json"
run baseline_s046 $R/baseline_final "jobs/baseline_v1.json"
run baseline_s046_proxy $R/baseline_proxy_final "jobs/baseline_v1.json"
run v2_E5_s046_proxy $R/v2_E5_proxy_final "jobs/v2_E5_nominal.json jobs/v2_E5_offsets.json"
run h000n.0_s0355 $R/h000n.0_final "jobs/h000n.0_v1.json jobs/h000n.0_v1_yaw180_345.json"
echo "$(date +%H:%M:%S) queue3 finished"
