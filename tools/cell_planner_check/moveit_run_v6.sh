#!/bin/bash
# v5 MoveIt run for ONE structure on ONE domain (planning only, no execution):  ./moveit_run_v5.sh table 187   |   ./moveit_run_v5.sh stand 188     (RANKS="1"; JOBS filter via env JOBSEL regex)
# table: jobs <name>_{BIG,SMALL,sh_BIG,sh_SMALL};  stand: jobs <name>_stand_{BIG,SMALL,sh_BIG,sh_SMALL}.  Never touches domain 11.
struct=$1; dom=$2
case $dom in 187|188) ;; *) echo "domain must be 187 or 188"; exit 2;; esac
T=/home/farhan/igus_rebel_ros2/tools/cell_planner_check
J=${V5_J:-/home/farhan/igus_rebel_ros2/scan_results/planner_check_v6_20261007}      # V5_J: work dir (quick_recheck.sh) with build/ jobs/ runs/ label_rank1.txt
RANKS=${RANKS:-"1"}
export PYTHONDONTWRITEBYTECODE=1 V5_DOMAIN=$dom ROS_DOMAIN_ID=$dom ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1
cd $T
for rank in $RANKS; do
  name=$(cat $J/label_rank$rank.txt)
  pid=$J/build/stack_${name}_${struct}.pgid
  ./stack_v6.sh start $J/build/${name}_$struct $pid; sleep 40
  if ! grep -q "You can start planning now" $J/build/stack_${name}_$struct.log; then sleep 40; fi
  if ! grep -q "You can start planning now" $J/build/stack_${name}_$struct.log || grep -q "process has died" $J/build/stack_${name}_$struct.log; then echo "ABORT: move_group not ready for $name $struct"; ./stack_v6.sh stop $pid; exit 1; fi
  if [ "$struct" = "table" ]; then joblist=$(ls $J/jobs/${name}_*.json | grep -v "_stand_"); else joblist=$(ls $J/jobs/${name}_*stand_*.json); fi
  for job in $joblist; do
    jn=$(basename $job .json)
    [ -n "$JOBSEL" ] && ! echo $jn | grep -Eq "$JOBSEL" && continue
    [ -f $J/runs/${jn}_$struct/done ] && continue
    source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash; export ROS_DOMAIN_ID=$dom ROS_LOCALHOST_ONLY=1
    python3 -u cellcheck_v5.py --job $job --build $J/build/${name}_$struct --out $J/runs/${jn}_$struct > $J/runs_${jn}_$struct.log 2>&1 && touch $J/runs/${jn}_$struct/done
  done
  ./stack_v6.sh stop $pid
done
echo "$(date +%T) moveit_run $struct $dom finished"
