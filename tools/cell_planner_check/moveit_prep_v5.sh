#!/bin/bash
# v5 MoveIt prep (no ROS processes): design json, cell builds (table + stand, scene-camera post, EXACT CAD v3 collision meshes), jobs from the study plans.  RANKS="1" (default).
T=/home/farhan/igus_rebel_ros2/tools/cell_planner_check
S=/home/farhan/igus_rebel_ros2/scan_results/cell_v5_20261007
J=/home/farhan/igus_rebel_ros2/scan_results/planner_check_v5_20261007
CAD=${CAD:-/home/farhan/igus_rebel_ros2/cad/dual_arm_tooling_v3}
RANKS=${RANKS:-"1"}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $J/build $J/jobs $J/runs
cd $T
for rank in $RANKS; do
  name=$(python3 -c "import json;print([c for c in json.load(open('$S/recommended.json'))['combos'] if c['rank']==$rank][0]['label'])")
  s=$(python3 -c "import json;print([c for c in json.load(open('$S/recommended.json'))['combos'] if c['rank']==$rank][0]['layout']['s'])")
  if [ "$rank" = "1" ]; then cdir=$CAD/collision; else cdir=$CAD/design_B/collision; fi
  python3 make_design_v5.py $S/recommended.json $rank $cdir || exit 1
  python3 gen_cell_v5.py --design designs/v5_$name.json --s $s --out $J/build/${name}_table --structure table --scene-camera || exit 1
  python3 gen_cell_v5.py --design designs/v5_$name.json --s $s --out $J/build/${name}_stand --structure stand --scene-camera || exit 1
  for st in "" "stand_"; do for fx in "" "sh_"; do for kind in BIG SMALL; do
    [ -f $S/plans_${name}_${st}${fx}${kind}.json ] && python3 build_job_v5.py --plans $S/plans_${name}_${st}${fx}${kind}.json --design v5_$name --kind ${st}${fx}${kind} --out $J/jobs/${name}_${st}${fx}${kind}.json
  done; done; done
  echo "$name" > $J/label_rank$rank.txt
done
