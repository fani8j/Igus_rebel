#!/bin/bash
# v5 quick re-check of the recommended design after a small change.  See README_quick.md.
#   ./quick_recheck.sh CHANGE.json TAG [NW=20]        (env: SKIP_MOVEIT=1, QR_ALL=1 evaluate all offline points, QR_FAST=1 skips verify_skid)
# Only domains 187 (table) and 188 (stand), ROS_LOCALHOST_ONLY=1, planning only.  All python under nice -n 19.  CAD rebuild runs in parallel with the offline region check.
CH=$(readlink -f "$1"); TAG=$2; NW=${3:-20}
T=/home/farhan/igus_rebel_ros2/tools/cell_planner_check
J=/home/farhan/igus_rebel_ros2/scan_results/planner_check_v5_20261007
W=$J/quick/$TAG
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
cd $T
nice -n 19 python3 quick_recheck.py prep "$CH" "$TAG" || exit 1
t0=$(date +%s)
( nice -n 19 python3 quick_recheck.py cad "$TAG" > $W/cad_stdout.log 2>&1 ) &
cadpid=$!
eval "$(python3 - <<PY
import json
ch=json.load(open('$W/change.json'))
m={'blade_exposed_mm':'V5_BLADE_EXP','skid_exposure_mm':'V5_EXPOSURE','pad_r_mm':'V5_PAD_R','pad_off_mm':'V5_PAD_OFF'}
for k,v in m.items():
    if k in ch: print('export %s=%s' % (v, ch[k]))
PY
)"
nice -n 19 python3 quick_recheck.py offline "$TAG" "$NW" > $W/offline_stdout.log 2>&1
wait $cadpid
if [ "$SKIP_MOVEIT" != "1" ] && ! python3 -c "import json,sys;sys.exit(0 if json.load(open('$W/change.json')).get('fixture') else 1)"; then
  nice -n 19 python3 quick_recheck.py moveitprep "$TAG" || exit 1
  export V5_J=$W
  nice -n 19 ./moveit_run_v5.sh table 187 > $W/run_table.log 2>&1 &
  p1=$!
  nice -n 19 ./moveit_run_v5.sh stand 188 > $W/run_stand.log 2>&1 &
  p2=$!
  wait $p1 $p2
fi
python3 quick_recheck.py report "$TAG" | tee $W/report.txt
echo "total $(( $(date +%s) - t0 )) s"
