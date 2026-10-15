#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
R=/home/farhan/igus_rebel_ros2/scan_results/planner_check_20261003
nice -n 10 python3 -u build_job.py v2 --design E5 --boxes BIG370 --yaws 0:180:15 --name v2_E5_nominal --out jobs/v2_E5_nominal.json
nice -n 10 python3 -u build_job.py v2 --design E5 --boxes BIG370 --yaws 0:180:15 --offsets 30:0,-30:0,0:30,0:-30 --name v2_E5_offsets --out jobs/v2_E5_offsets.json
nice -n 10 python3 -u build_job.py v2 --design E5 --boxes SMALL --yaws 0:180:15 --mode single --name v2_E5_small_single --out jobs/v2_E5_small_single.json
echo builds-done
