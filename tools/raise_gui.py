#!/usr/bin/env /usr/bin/python3
"""Raise buried GUI windows (RViz, TCP calibrator) above the fullscreen stack.

GNOME on this workstation creates the windows but never surfaces them:
RViz spawns 10x10 unmapped, the Tk calibrator maps but stays buried under
the terminal/Helium. Run this after any manual `ros2 launch ...` whose
window does not appear.

Usage:
  raise_gui.py                  # raise RViz + TCP windows
  raise_gui.py --watch          # keep watching, raise newcomers for 60s
"""
import sys
import time

from Xlib import X, display

TARGETS = ("rviz", "tcp", "rebel", "tk")


def raise_matching(verbose=True):
    d = display.Display(":0")
    root = d.screen().root
    raised = []
    try:
        windows = root.query_tree().children
    except Exception as exc:
        print(f"query failed: {exc}")
        return raised
    for w in windows:
        try:
            cls = w.get_wm_class() or ("", "")
            name = w.get_wm_name() or ""
        except Exception:
            continue
        hay = f"{cls[0]} {cls[1]} {name}".lower()
        if any(t in hay for t in TARGETS):
            try:
                w.configure(stackmode=X.Above)
                w.map()
                raised.append(name or cls[1] or hex(w.id))
            except Exception as exc:
                print(f"raise {name} failed: {exc}")
    d.sync()
    if verbose:
        print("raised:", raised if raised else "none matched")
    return raised


if __name__ == "__main__":
    if "--watch" in sys.argv[1:]:
        deadline = time.time() + 60
        seen = set()
        while time.time() < deadline:
            for name in raise_matching(verbose=False):
                if name not in seen:
                    seen.add(name)
                    print(f"raised: {name}", flush=True)
            time.sleep(2)
        print("watch done, raised:", sorted(seen) if seen else "none")
    else:
        raise_matching()
