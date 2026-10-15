#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003
cd "$HERE"
until grep -q "queue4 finished" $R/run_all4.log; do sleep 5; done
mkdir -p $R/audit
./audit_on_build.sh h000n.0_s0355 jobs/h000n.0_v1.json $R/audit/audit_h000n.0_real.json
./audit_on_build.sh h000n.0_proxy jobs/h000n.0_v1.json $R/audit/audit_h000n.0_proxy.json
./audit_on_build.sh baseline_s046 jobs/baseline_v1.json $R/audit/audit_baseline_real.json
./audit_on_build.sh baseline_s046_proxy jobs/baseline_v1.json $R/audit/audit_baseline_proxy.json
./audit_on_build.sh v2_E5_s046 jobs/v2_E5_nominal.json $R/audit/audit_v2_E5_nominal_real.json
./audit_on_build.sh v2_E5_s046_proxy jobs/v2_E5_nominal.json $R/audit/audit_v2_E5_nominal_proxy.json
./audit_on_build.sh v2_E5_s046 jobs/v2_E5_offsets.json $R/audit/audit_v2_E5_offsets_real.json
./audit_on_build.sh v2_E5_s046_proxy jobs/v2_E5_offsets.json $R/audit/audit_v2_E5_offsets_proxy.json
echo "$(date +%H:%M:%S) queue5 finished"
