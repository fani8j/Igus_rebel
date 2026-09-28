"""Make the pure-Python core importable (it lives outside this colcon workspace).

Set ``SIDE_TAPE_CORE`` to the directory that contains the ``precision_box_detector``
package; defaults to the rgbd_captures checkout. Import this module before any
``precision_box_detector`` import.
"""

import os
import sys

DEFAULT_CORE = "/home/farhan/Downloads/rgbd_captures/precision_box_detector"
DEFAULT_CONFIG = os.path.expanduser("~/igus_rebel_ros2/src/side_tape_ros/config/knife_mount.yaml")

CORE = os.environ.get("SIDE_TAPE_CORE", DEFAULT_CORE)
if not os.path.isdir(os.path.join(CORE, "precision_box_detector")):
    raise ImportError(f"side_tape core not found under {CORE!r}; set SIDE_TAPE_CORE")
if CORE not in sys.path:
    sys.path.insert(0, CORE)
