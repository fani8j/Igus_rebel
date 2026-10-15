#!/bin/bash
# MoveIt validation of the v4 finalists with the EXACT CAD v2 collision meshes (domain 187, planning only, table + real stand, scene camera).
# usage: moveit_stage_v4.sh  (reads recommended.json ranks listed in RANKS, CAD dirs in CADDIR_<rank>).  Stack PGID recorded per run; stopped at the end of each structure.
T=/home/farhan/igus_rebel_ros2/tools/cell_planner_check
S=/home/farhan/igus_rebel_ros2/scan_results/cell_v4_20261006
J=/home/farhan/igus_rebel_ros2/scan_results/planner_check_v4_20261006
CAD=/home/farhan/igus_rebel_ros2/cad/dual_arm_tooling_v2
RANKS=${RANKS:-"1 2"}
export PYTHONDONTWRITEBYTECODE=1 ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1
mkdir -p $J/build $J/jobs $J/runs
cd $T
for rank in $RANKS; do
  name=$(python3 -c "import json;print([c for c in json.load(open('$S/recommended.json'))['combos'] if c['rank']==$rank][0]['label'])")
  s=$(python3 -c "import json;print([c for c in json.load(open('$S/recommended.json'))['combos'] if c['rank']==$rank][0]['layout']['s'])")
  [ -f $J/done_$name ] && { echo "skip $name"; continue; }
  if [ "$rank" = "1" ]; then cdir=$CAD/collision; else cdir=$CAD/design_B/collision; fi
  python3 make_design_v4.py $S/recommended.json $rank $cdir
  python3 gen_cell_v4.py --design designs/v4_$name.json --s $s --out $J/build/${name}_table --structure table --scene-camera
  python3 gen_cell_v4.py --design designs/v4_$name.json --s $s --out $J/build/${name}_stand --structure stand --scene-camera
  # table structure: plans of the table runs ; stand structure: plans of the stand-mode runs (+ the table plans replayed on the stand, informational)
  for st in "" "stand_"; do for fx in "" "fx_"; do for kind in BIG SMALL; do
    [ -f $S/plans_${name}_${st}${fx}${kind}.json ] && python3 build_job_v3.py --plans $S/plans_${name}_${st}${fx}${kind}.json --design v4_$name --kind ${st}${fx}${kind} --out $J/jobs/${name}_${st}${fx}${kind}.json
  done; done; done
  for struct in table stand; do
    ./stack_v4.sh start $J/build/${name}_$struct $J/build/stack_${name}_$struct.pgid; sleep 40
    if ! grep -q "You can start planning now" $J/build/stack_${name}_$struct.log; then sleep 30; fi
    if ! grep -q "You can start planning now" $J/build/stack_${name}_$struct.log || grep -q "process has died" $J/build/stack_${name}_$struct.log; then echo "ABORT: move_group not ready/alive for $name $struct (see $J/build/stack_${name}_$struct.log)"; ./stack_v4.sh stop $J/build/stack_${name}_$struct.pgid; continue; fi
    if [ "$struct" = "table" ]; then joblist=$(ls $J/jobs/${name}_*.json | grep -v "_stand_"); else joblist=$(ls $J/jobs/${name}_*.json); fi
    for job in $joblist; do
      jn=$(basename $job .json)
      export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1; source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash; export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
      python3 -u cellcheck_v3.py --job $job --build $J/build/${name}_$struct --out $J/runs/${jn}_$struct > $J/runs_${jn}_$struct.log 2>&1
    done
    ./stack_v4.sh stop $J/build/stack_${name}_$struct.pgid
  done
  touch $J/done_$name
done
