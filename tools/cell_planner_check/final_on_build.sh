#!/usr/bin/env bash
# stack on an EXISTING build; sanity check, run jobs (cellcheck + summary), audit the study paths of the jobs, stop.
#   ./final_on_build.sh <build_name> <out_dir> "<job1 job2 ...>" [extra cellcheck args]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="$1"; out="$2"; jobs="$3"; shift 3
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
cd "$HERE"; mkdir -p "$out"
./stack.sh start $PWD/build/${name} $PWD/build/${name}.pgid; sleep 25
echo "$(date +%H:%M:%S) stack up: $(ros2 node list | tr '\n' ' ')" > "$out/stack_nodes.txt"
first=1
for job in $jobs; do
  jn=$(python3 -c "import json;print(json.load(open('$job'))['name'])")
  if [ $first = 1 ]; then
    fk="${job%.json}_fkcases.json"
    nice -n 10 python3 sanity_check.py --build build/${name} $( [ -f "$fk" ] && echo --cases $fk ) --out "$out/sanity_${name}.json" > "$out/sanity.log" 2>&1
    first=0
  fi
  echo "$(date +%H:%M:%S) job $jn"
  nice -n 10 python3 -u cellcheck.py --job "$job" --build build/${name} --out "$out" "$@" > "$out/run_${jn}.log" 2>&1
  nice -n 10 python3 summarize.py "$out/result_${jn}.json" --out "$out/summary_${jn}.md" --title "${name} / ${jn}" > /dev/null
  nice -n 10 python3 path_audit.py --job "$job" --build build/${name} --out "$out/audit_${jn}.json" > "$out/audit_${jn}.log" 2>&1
done
./stack.sh stop $PWD/build/${name}.pgid
echo "$(date +%H:%M:%S) final_on_build $name finished"
