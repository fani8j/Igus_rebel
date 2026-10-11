"""MarkerArray of the detected seam / box corners from a box_h_cut_snapshot/v1 dict (frame = snapshot frame_id)."""

from __future__ import annotations

from typing import Any

CORNERS = ("back_left", "back_right", "front_right", "front_left")
SEAM_POINTS = {"seam_start": (0.1, 0.9, 0.1), "seam_end": (0.9, 0.1, 0.1), "seam_center": (1.0, 0.85, 0.0)}
OUTLINE = ("back_left", "back_right", "front_right", "front_left", "back_left")


def _pt(d):
    from geometry_msgs.msg import Point
    return Point(x=float(d["x"]), y=float(d["y"]), z=float(d["z"]))


def _get(snapshot: dict, name: str):
    v = snapshot.get(name)
    if isinstance(v, dict) and all(k in v for k in "xyz"):
        return v
    return None


def clear_markers():
    from visualization_msgs.msg import Marker, MarkerArray
    arr = MarkerArray()
    arr.markers.append(Marker(action=Marker.DELETEALL))
    return arr


def snapshot_markers(snapshot: dict[str, Any], stamp=None, ns: str = "box_seam"):
    """MarkerArray for the snapshot (spheres + labels for seam start/end/center and the four top corners, the
    seam line and the top-face outline). Missing points are skipped."""
    from builtin_interfaces.msg import Duration
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    frame = snapshot.get("frame_id") or "base_link"
    arr = MarkerArray()
    arr.markers.append(Marker(action=Marker.DELETEALL))
    next_id = [0]

    def base(mtype, name):
        m = Marker()
        m.header.frame_id = frame
        if stamp is not None:
            m.header.stamp = stamp
        m.ns = f"{ns}/{name}"
        m.id = next_id[0]
        next_id[0] += 1
        m.type = mtype
        m.action = Marker.ADD
        m.lifetime = Duration(sec=0, nanosec=0)
        m.pose.orientation.w = 1.0
        return m

    def sphere(name, p, rgb, size):
        m = base(Marker.SPHERE, name)
        m.pose.position = _pt(p)
        m.scale.x = m.scale.y = m.scale.z = size
        m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
        arr.markers.append(m)
        t = base(Marker.TEXT_VIEW_FACING, name + "_label")
        t.pose.position = _pt(p)
        t.pose.position.z += 0.02
        t.scale.z = 0.012
        t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        t.text = name
        arr.markers.append(t)

    def line(name, pts, rgb, width):
        m = base(Marker.LINE_STRIP, name)
        m.scale.x = width
        m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
        m.points = [_pt(p) for p in pts]
        arr.markers.append(m)

    seam = [_get(snapshot, n) for n in ("seam_start", "seam_end")]
    for name, rgb in SEAM_POINTS.items():
        p = _get(snapshot, name)
        if p:
            sphere(name, p, rgb, 0.012)
    if all(seam):
        line("seam_line", seam, (0.1, 0.9, 0.1), 0.004)
    for name in CORNERS:
        p = _get(snapshot, name)
        if p:
            sphere(name, p, (0.2, 0.5, 1.0), 0.010)
    outline = [_get(snapshot, n) for n in OUTLINE]
    if all(outline):
        line("top_outline", outline, (0.9, 0.9, 0.9), 0.003)
    return arr
