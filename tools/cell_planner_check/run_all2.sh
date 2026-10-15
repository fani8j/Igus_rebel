#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003
cd "$HERE"
run() { echo "$(date +%H:%M:%S) ==== $1"; ./experiment_multi.sh "$@"; }
run v2_E5_s046 designs/v2_E5.json 0.46 $R/v2_E5 "jobs/v2_E5_nominal.json jobs/v2_E5_offsets.json jobs/v2_E5_small_single.json" --fkcases jobs/v2_E5_nominal_fkcases.json
run v2_E5_s046_proxy designs/v2_E5.json 0.46 $R/v2_E5_proxy "jobs/v2_E5_nominal.json jobs/v2_E5_offsets.json jobs/v2_E5_small_single.json" --no-matrix --fkcases jobs/v2_E5_nominal_fkcases.json -- --geometry proxy --structure study
run h000n.0_proxy designs/h000n.0.json 0.355 $R/h000n.0_proxy "jobs/h000n.0_v1.json" --no-matrix --fkcases jobs/h000n.0_v1_fkcases.json -- --geometry proxy --structure study
echo "$(date +%H:%M:%S) queue2 finished"
