#!/usr/bin/env python3
"""Lists the Kit [Warning]/[Error] lines that appear after the first '### ' marker of a test log (Kit start-up deprecation warnings come before it),
grouped by section marker, with shutdown noise separated. Usage: summarize_log.py log [log ...]"""
import re
import sys
from collections import Counter

SHUTDOWN = ('Could not find category', 'USD stage detach not called', 'outstanding SimStageWithHistory', 'contexts were leaked',
            'Recursive unloadAllPlugins', 'was deleted while being used by a shape in a tensor view', 'Simulation view object is invalidated')
for path in sys.argv[1:]:
    sec, seen, shut = None, Counter(), 0
    started = False
    out = []
    for line in open(path, errors='replace'):
        if line.startswith('### '):
            started, sec = True, line.strip()
            continue
        if not started or ('[Warning]' not in line and '[Error]' not in line):
            continue
        if any(s in line for s in SHUTDOWN):
            shut += 1
            continue
        msg = re.sub(r'^\S+ \[[\d,]+ms\] ', '', line.strip())
        msg = re.sub(r'/(?:[\w.\-]+/)+', lambda m: '/.../', msg) if len(msg) > 400 else msg
        seen[(sec, msg[:300])] += 1
    print(f'--- {path}: {sum(seen.values())} warning/error lines after the first marker ({len(seen)} distinct), {shut} shutdown-noise lines ignored')
    for (s, m), n in seen.items():
        print(f'  [{s}] x{n}: {m}')
