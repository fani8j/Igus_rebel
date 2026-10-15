#!/usr/bin/env bash
# like experiment.sh but runs several jobs against ONE stack.
#   ./experiment_multi.sh <name> <design.json> <s> <out_dir> "<job1.json job2.json ...>" [--no-matrix] [--fkcases cases.json] [-- <gen_cell args>] [-- <cellcheck args>]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="$1"; design="$2"; s="$3"; out="$4"; jobs="$5"; shift 5
matrix=1; fk=""; gen_args=(); chk_args=(); mode=opts
while [ $# -gt 0 ]; do
  case "$1" in
    --no-matrix) matrix=0;;
    --fkcases) fk="$2"; shift;;
    --) if [ "$mode" = opts ]; then mode=gen; elif [ "$mode" = gen ]; then mode=chk; fi;;
    *) if [ "$mode" = gen ]; then gen_args+=("$1"); elif [ "$mode" = chk ]; then chk_args+=("$1"); fi;;
  esac; shift
done
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
cd "$HERE"; mkdir -p "$out" build
log() { echo "$(date +%H:%M:%S) $*"; }
if [ "$matrix" = 1 ]; then
  nice -n 10 python3 gen_cell.py --design "$design" --s "$s" --out build/${name}_pre "${gen_args[@]}" || exit 1
  ./stack.sh start $PWD/build/${name}_pre $PWD/build/${name}_pre.pgid; sleep 25
  nice -n 10 python3 sample_collisions.py --build build/${name}_pre --n ${SAMPLES:-20000} --out build/${name}_pre/collision_matrix.json || { ./stack.sh stop $PWD/build/${name}_pre.pgid; exit 1; }
  ./stack.sh stop $PWD/build/${name}_pre.pgid; sleep 2
  nice -n 10 python3 gen_cell.py --design "$design" --s "$s" --out build/${name} --matrix build/${name}_pre/collision_matrix.json "${gen_args[@]}" || exit 1
else
  nice -n 10 python3 gen_cell.py --design "$design" --s "$s" --out build/${name} "${gen_args[@]}" || exit 1
fi
./stack.sh start $PWD/build/${name} $PWD/build/${name}.pgid; sleep 25
log "stack up"; ros2 node list | tr '\n' ' '; echo
nice -n 10 python3 sanity_check.py --build build/${name} ${fk:+--cases $fk} --out "$out/sanity_${name}.json" > "$out/sanity.log" 2>&1
log "sanity done"
for job in $jobs; do
  jn=$(basename "$job" .json)
  log "job $jn"
  nice -n 10 python3 -u cellcheck.py --job "$job" --build build/${name} --out "$out" "${chk_args[@]}" > "$out/run_${jn}.log" 2>&1
  res=$(ls "$out"/result_*.json | grep -- "result_" | xargs -n1 basename | sed 's/^result_//; s/\.json$//' | grep -x "$(python3 -c "import json,sys;print(json.load(open('$job'))['name'])")" | head -1)
  [ -n "$res" ] && nice -n 10 python3 summarize.py "$out/result_${res}.json" --out "$out/summary_${res}.md" --title "${name} / ${res}" > /dev/null
done
./stack.sh stop $PWD/build/${name}.pgid
log "experiment $name finished"
