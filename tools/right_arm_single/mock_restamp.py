#!/usr/bin/env python3
"""MOCK ONLY: copy a historical snapshot with a fresh acquisition_stamp_ns and a fresh mtime.

The executor refuses snapshots whose file mtime OR acquisition_stamp_ns is older than 240 s.
Never use the result to justify physical motion (it is marked "mock_restamped").
usage: mock_restamp.py SRC DST
"""
import json
import sys
import time

with open(sys.argv[1], encoding="utf-8") as handle:
    doc = json.load(handle)
doc["acquisition_stamp_ns"] = time.time_ns()
doc["mock_restamped"] = True
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
print(f"wrote {sys.argv[2]} (restamped, mock only)")
