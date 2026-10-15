#!/usr/bin/env bash
# start the stack on an EXISTING build dir, run jobs, summarise, stop.   ./run_on_build.sh <build_name> <out_dir> "<job1 job2 ...>" [cellcheck args]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="$1"; out="$2"; jobs="$3"; shift 3
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash; source /home/farhan/igus_rebel_ros2/install/setup.bash
export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
cd "$HERE"; mkdir -p "$out"
./stack.sh start $PWD/build/${name} $PWD/build/${name}.pgid; sleep 25
echo "$(date +%H:%M:%S) stack up: $(ros2 node list | tr '\n' ' ')"
for job in $jobs; do
  jn=$(python3 -c "import json;print(json.load(open('$job'))['name'])")
  echo "$(date +%H:%M:%S) job $jn"
  nice -n 10 python3 -u cellcheck.py --job "$job" --build build/${name} --out "$out" "$@" > "$out/run_${jn}.log" 2>&1
  nice -n 10 python3 summarize.py "$out/result_${jn}.json" --out "$out/summary_${jn}.md" --title "${name} / ${jn}" > /dev/null
done
./stack.sh stop $PWD/build/${name}.pgid
echo "$(date +%H:%M:%S) run_on_build $name finished"
