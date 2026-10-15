#!/bin/bash
# MoveIt validation of the v3 finalists (domain 187, planning only, table structure). Stack PGID recorded in <planner_check_v3>/build/stack_<name>.pgid ; stopped at the end of each design.
# python workers inside: stack 2 + client 1.
T=/home/farhan/igus_rebel_ros2/tools/cell_planner_check
S=/home/farhan/igus_rebel_ros2/scan_results/cell_v3_20261004
J=/home/farhan/igus_rebel_ros2/scan_results/planner_check_v3_20261004
export PYTHONDONTWRITEBYTECODE=1 ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
cd $T
for name in $(python3 -c "import json;print(' '.join(c['label'] for c in json.load(open('$S/recommended.json'))['combos']))"); do
  [ -f $J/done_$name ] && { echo "skip $name"; continue; }
  s=$(python3 -c "import json;print([c for c in json.load(open('$S/recommended.json'))['combos'] if c['label']=='$name'][0]['layout']['s'])")
  rank=$(python3 -c "import json;print([c for c in json.load(open('$S/recommended.json'))['combos'] if c['label']=='$name'][0]['rank'])")
  python3 make_design_v3.py $S/recommended.json $rank
  python3 gen_cell_v3.py --design designs/v3_$name.json --s $s --out $J/build/${name}_table --structure table --scene-camera
  python3 gen_cell_v3.py --design designs/v3_$name.json --s $s --out $J/build/${name}_stand --structure stand --scene-camera
  for kind in BIG SMALL; do python3 build_job_v3.py --plans $S/plans_${name}_$kind.json --design v3_$name --kind $kind --out $J/jobs/${name}_$kind.json; done
  for struct in table stand; do
    ./stack_v3.sh start $J/build/${name}_$struct $J/build/stack_${name}_$struct.pgid; sleep 30
    for kind in BIG SMALL; do
      export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1; source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash; export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
      python3 -u cellcheck_v3.py --job $J/jobs/${name}_$kind.json --build $J/build/${name}_$struct --out $J/runs/${name}_${struct}_$kind > $J/runs_${name}_${struct}_$kind.log 2>&1
    done
    ./stack_v3.sh stop $J/build/stack_${name}_$struct.pgid
  done
  touch $J/done_$name
done
