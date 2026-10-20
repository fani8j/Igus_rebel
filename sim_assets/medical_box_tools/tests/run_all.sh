#!/bin/bash
# Runs all Isaac Sim 5.1 headless tests of the medical-box tools; logs to tests/results/.
# Safe settings: ROS_DOMAIN_ID=170, ROS_LOCALHOST_ONLY=1, nice 19, no ROS 2 bridge extension (the python files force the env as well).
# QUICK=1 skips the two cabinet grasp runs (about 25 min each on a busy machine).
set -u
export ROS_DOMAIN_ID=170 ROS_LOCALHOST_ONLY=1
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$HERE/results"
mkdir -p "$R"
PY=/home/farhan/isaac-sim/python.sh
cd /tmp
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tee "$R/gpu_before.txt"
for f in left_suction_tool right_gripper_tool right_gripper_tool_independent shelf_handoff_scene pick_test_scene shelf_cartons rebel_left_suction rebel_right_gripper dual_arm_rig_medical \
         d200_dual_arm_medical d200_rig_test_scene cabinet_grasp_test_scene cabinet_robot_test_scene; do
  echo "== load $f"
  nice -n 19 "$PY" "$HERE/test_load.py" "$f.usda" > "$R/test_load_$f.log" 2>&1
  echo "exit $?"
done
for t in calibrate_surface_gripper test_left_suction test_right_gripper test_assemblies test_d200; do
  echo "== $t"
  nice -n 19 "$PY" "$HERE/$t.py" > "$R/$t.log" 2>&1
  echo "exit $?"
done
if [ -z "${QUICK:-}" ]; then
  for mode in topdown front; do
    echo "== cabinet grasp $mode"
    nice -n 19 "$PY" "$HERE/test_cabinet_grasp.py" --mode $mode --targets all --variants vac35,squeeze,lat2,tilt3 --shots > "$R/test_cabinet_grasp_$mode.log" 2>&1
    echo "exit $?"
  done
  python3 "$HERE/summarize_cabinet.py" > /dev/null
  # full-robot replay (needs data/plans/*.json from: python3 tools/parking_study.py ...; python3 tools/export_plan.py ...; PyKDL system python)
  if ls "$HERE/../data/plans/front_"*.json > /dev/null 2>&1; then
    echo "== cabinet grasp ROBOT front"
    nice -n 19 "$PY" "$HERE/test_cabinet_grasp_robot.py" --mode front --targets all > "$R/test_cabinet_grasp_robot_front.log" 2>&1
    echo "exit $?"
    python3 "$HERE/summarize_robot.py" > /dev/null
  fi
fi
python3 "$HERE/summarize_log.py" "$R"/test_*.log "$R"/calibrate_surface_gripper.log > "$R/warnings_summary.txt"
cat "$R/warnings_summary.txt"
