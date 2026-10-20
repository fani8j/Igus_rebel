#!/bin/bash
# rebuild everything (about 3 minutes: build -> verify -> render); venv python of the shared params
set -e
cd "$(dirname "$0")"
PY="/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python"
export PYTHONDONTWRITEBYTECODE=1
nice -n 19 "$PY" build.py
nice -n 19 "$PY" verify.py
nice -n 19 "$PY" render.py
