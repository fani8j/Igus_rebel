#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003
cd "$HERE"
until grep -q "rebuild-done" $R/rebuild_v1.log; do sleep 5; done
run() { echo "$(date +%H:%M:%S) ==== $1"; ./final_on_build.sh "$@"; }
run h000n.0_s0355 $R/final/h000n.0_real "jobs/h000n.0_v1.json jobs/h000n.0_v1_yaw180_345.json"
run h000n.0_proxy $R/final/h000n.0_proxy "jobs/h000n.0_v1.json"
run baseline_s046 $R/final/baseline_real "jobs/baseline_v1.json"
run baseline_s046_proxy $R/final/baseline_proxy "jobs/baseline_v1.json"
run v2_E5_s046 $R/final/v2_E5_real "jobs/v2_E5_nominal.json jobs/v2_E5_offsets.json jobs/v2_E5_small_single2.json"
run v2_E5_s046_proxy $R/final/v2_E5_proxy "jobs/v2_E5_nominal.json jobs/v2_E5_offsets.json"
echo "$(date +%H:%M:%S) final queue finished"
