#!/usr/bin/env bash
# Sequential queue of the planner-check experiments (domain 187, one stack at a time).  Records its own PID in <out>/run_all.pid.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003
cd "$HERE"
# wait for a running job (PID given as $1) and stop the manually started stack (pgid file given as $2)
if [ -n "$1" ]; then while kill -0 "$1" 2>/dev/null; do sleep 3; done; fi
if [ -n "$2" ]; then ./stack.sh stop "$2"; fi
run() { echo "$(date +%H:%M:%S) ==== $1"; ./experiment.sh "$@" ; }
run h000n.0_proxy designs/h000n.0.json 0.355 jobs/h000n.0_v1.json $R/h000n.0_proxy --no-matrix --fkcases jobs/h000n.0_v1_fkcases.json -- --geometry proxy --structure study
run baseline_s046 designs/baseline.json 0.46 jobs/baseline_v1.json $R/baseline --fkcases jobs/baseline_v1_fkcases.json
run baseline_s046_proxy designs/baseline.json 0.46 jobs/baseline_v1.json $R/baseline_proxy --no-matrix --fkcases jobs/baseline_v1_fkcases.json -- --geometry proxy --structure study
run v2_D00_s046 designs/v2_D00.json 0.46 jobs/v2_D00_v2.json $R/v2_D00 --fkcases jobs/v2_D00_v2_fkcases.json
run v2_D00_s046_proxy designs/v2_D00.json 0.46 jobs/v2_D00_v2.json $R/v2_D00_proxy --no-matrix --fkcases jobs/v2_D00_v2_fkcases.json -- --geometry proxy --structure study
echo "$(date +%H:%M:%S) queue finished"
