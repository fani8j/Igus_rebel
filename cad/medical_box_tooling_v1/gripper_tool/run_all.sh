#!/bin/bash
# rebuild everything (about 1 minute); venv python from the shared params
set -e
cd "$(dirname "$0")"
PY="/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="/media/farhan/My Passport/igus_nvidia/scratch_gripper_tool/tmp"
mkdir -p "$TMPDIR"
nice -n 19 "$PY" build.py
nice -n 19 "$PY" verify.py
nice -n 19 "$PY" render.py
