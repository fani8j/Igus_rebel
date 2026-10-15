#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
nice -n 10 python3 -u build_job.py v1 --design h000n.0 --layout 0.0,0.40,0.355 --boxes BIG370,SMALL --yaws 0:180:15 --out jobs/h000n.0_v1.json
nice -n 10 python3 -u build_job.py v1 --design h000n.0 --layout 0.0,0.40,0.355 --boxes BIG370 --yaws 180:360:15 --name h000n.0_v1_yaw180_345 --out jobs/h000n.0_v1_yaw180_345.json
nice -n 10 python3 -u build_job.py v1 --design baseline --layout 0.0,0.35,0.46 --boxes BIG370,SMALL --yaws 0:180:15 --name baseline_v1 --out jobs/baseline_v1.json
echo rebuild-done
