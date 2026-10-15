#!/bin/bash
# v6 MoveIt prep (no ROS processes): design json (exact CAD collision meshes), cell builds (table + stand, scene-camera post), jobs from the v6 study plans (cell_v6_20261007/plans_<label>_*.json).
# CAD collision dir: default = unchanged CAD v3 (geometry identical in v6; no v3b needed).  New file; v5 scripts untouched.
T=/home/farhan/igus_rebel_ros2/tools/cell_planner_check
S=/home/farhan/igus_rebel_ros2/scan_results/cell_v6_20261007
J=/home/farhan/igus_rebel_ros2/scan_results/planner_check_v6_20261007
CAD=${CAD:-/home/farhan/igus_rebel_ros2/cad/dual_arm_tooling_v3}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $J/build $J/jobs $J/runs
cd $T
name=$(python3 -c "import json;print(json.load(open('$S/recommended.json'))['combos'][0]['label'])")
s=$(python3 -c "import json;print(json.load(open('$S/recommended.json'))['combos'][0]['layout']['s'])")
if [ "$SKIP_BUILD" != "1" ]; then
python3 - <<PY || exit 1
import json, sys
sys.path.insert(0, '$T')
import make_design_v5 as MD
c = json.load(open('$S/recommended.json'))['combos'][0]
d = MD.design_from_entry(c, '$CAD/collision'); d['name'] = 'v6_' + c['label']
json.dump(d, open('$J/design_v6.json', 'w'), indent=1); print('wrote $J/design_v6.json')
PY
for st in table stand; do
  python3 gen_cell_v5.py --design $J/design_v6.json --s $s --out $J/build/${name}_$st --structure $st --scene-camera || exit 1
done
fi
for st in "" "stand_"; do for kind in BIG SMALL; do
  [ -f $S/plans_${name}_${st}${kind}.json ] && python3 build_job_v5.py --plans $S/plans_${name}_${st}${kind}.json --design v6_$name --kind ${st}${kind} --out $J/jobs/${name}_${st}${kind}.json
done; done
echo "$name" > $J/label_rank1.txt
