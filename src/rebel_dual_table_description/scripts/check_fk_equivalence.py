#!/usr/bin/env python3
"""FK equivalence of the dual-table arms against the single-arm model.

For N random joint configurations (inside the joint limits) the pose of EVERY link of the single-arm
model (arm links, igus_tool0, xeg32_tool_tip, knife_c_holder / handpiece / wrist camera, ...) relative
to the arm base in rebel_dual_table.urdf.xacro (arm_left_*, arm_right_*) is compared with the same pose
in igus_rebel_xeg32_wrapper.urdf.xacro (base_link -> ...). The collision and visual geometry of every
link (shape, size, mesh path, origin) must also be identical, EXCEPT the collision geometry of the links in
make_arm_macro.DUAL_COLLISION_OVERRIDES (dual-only convex hulls of the real link meshes): those must use exactly
the override meshes (identity origin, no scale), and their visuals must still equal the single arm's. The only link
the dual model adds is <arm>knife_c_blade (dual-only blade collision box, see rebel_tool.macro.xacro): it must exist
for both arms, hang rigidly on igus_tool0, and its collision box must contain the exposed blade (CAD constants below,
NOT read from the xacro) and the segment seat-face -> TCP (xeg32_tool_tip, taken from the model) with >= 1 mm margin
on every side. Pure numpy: needs no ROS runtime except
`xacro` and the installed packages.

    ros2 run rebel_dual_table_description check_fk_equivalence.py [--n 10] [--tol 1e-9]
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from ament_index_python.packages import get_package_share_directory

# Frames compared: every link of the single-arm model except "world" (see run()).
# Frames that must be present, so a tool-less model cannot pass vacuously:
REQUIRED_FRAMES = ["link6", "igus_tool0", "knife_c_tcp_point", "xeg32_tool_tip", "knife_c_holder",
                   "knife_c_handpiece", "knife_c_wrist_camera"]
JOINTS = [f"joint{i}" for i in range(1, 7)]

# Blade CAD constants (igus_tool0 frame, metres; scan_results/knife_c_handpiece_fix_20261012/README.md):
# body rear-face centre, knife axis, blade/seat axis y (CAD stub axis), seat front face = blade start, blade point.
BLADE_REAR = np.array([0.04745, -0.09200, -0.05511])
BLADE_AXIS = np.array([-0.5, 0.0, np.sqrt(3.0) / 2.0])
BLADE_Y = -0.092262
BLADE_SEAT_FRONT = 0.1708          # 160.6 body+collar + 10.2 seat
BLADE_POINT = 0.1852               # + 14.4 exposed blade (caliper 2026-10-12: 185.2 mm)
BLADE_HALF_WIDTH = 0.002           # BDC-200P blade is 4 mm wide
BLADE_HALF_THICK = 0.0005          # modelled 1.0 mm (real 0.38 mm)
BLADE_MARGIN = 0.001               # required margin on every side
BLADE_MAX_BOX = 0.030              # sanity: no box dimension above 30 mm
BLADE_LINK = "knife_c_blade"


def blade_cad_points():
    """(seat face centre, [slab corners + blade point], in igus_tool0, metres) from the CAD constants."""
    s0 = BLADE_REAR + BLADE_SEAT_FRONT * BLADE_AXIS
    s0 = np.array([s0[0], BLADE_Y, s0[2]])
    kx = np.array([np.sqrt(3.0) / 2.0, 0.0, 0.5])
    ky = np.array([0.0, 1.0, 0.0])
    pts = []
    for z in (0.0, BLADE_POINT - BLADE_SEAT_FRONT):
        for x in (-BLADE_HALF_THICK, BLADE_HALF_THICK):
            for y in (-BLADE_HALF_WIDTH, BLADE_HALF_WIDTH):
                pts.append(s0 + x * kx + y * ky + z * BLADE_AXIS)
    return s0, pts


def blade_box(root, arm):
    """(origin xyz, size) of the single collision box of <arm>knife_c_blade, or raises."""
    link = [ln for ln in root.findall("link") if ln.get("name") == arm + BLADE_LINK]
    if len(link) != 1:
        raise ValueError(f"{arm}{BLADE_LINK} missing")
    cols = link[0].findall("collision")
    if len(cols) != 1 or cols[0].find("geometry/box") is None:
        raise ValueError(f"{arm}{BLADE_LINK} must have exactly one box collision")
    org = cols[0].find("origin")
    xyz = np.array([float(v) for v in (org.get("xyz", "0 0 0") if org is not None else "0 0 0").split()])
    rpy = [float(v) for v in (org.get("rpy", "0 0 0") if org is not None else "0 0 0").split()]
    if any(rpy):
        raise ValueError("blade box collision origin must have identity rotation")
    return xyz, np.array([float(v) for v in cols[0].find("geometry/box").get("size").split()])


def blade_margin_ok(box_xyz, box_size, T_tool0_link, pts_tool0):
    """Min over points of the distance to the nearest box face (positive = inside), in metres."""
    Ti = np.linalg.inv(T_tool0_link)
    worst = np.inf
    for p in pts_tool0:
        q = (Ti @ np.append(p, 1.0))[:3] - box_xyz
        worst = min(worst, float((box_size / 2.0 - np.abs(q)).min()))
    return worst


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(3)
    x, y, z = axis / n
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def geometry_signature(root, prefix):
    """{link name without prefix: [(kind, origin xyz, origin rpy, geometry tag+attrs) ...]} for collision and visual
    (materials are ignored on purpose: their names are prefixed)."""
    out = {}
    for ln in root.findall("link"):
        name = ln.get("name")
        if prefix and not name.startswith(prefix):
            continue
        sig = []
        for kind in ("collision", "visual"):
            for g in ln.findall(kind):
                org = g.find("origin")
                xyz = tuple(float(v) for v in (org.get("xyz", "0 0 0") if org is not None else "0 0 0").split())
                rpy = tuple(float(v) for v in (org.get("rpy", "0 0 0") if org is not None else "0 0 0").split())
                geo = g.find("geometry")[0]
                sig.append((kind, xyz, rpy, geo.tag, tuple(sorted(geo.attrib.items()))))
        out[name[len(prefix):]] = sig
    return out


def dual_collision_overrides():
    """The dual-only collision override table of make_arm_macro.py (same directory, source or installed)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_arm_macro.py")
    spec = importlib.util.spec_from_file_location("make_arm_macro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DUAL_COLLISION_OVERRIDES


def expected_dual_signature(sig_single, overrides):
    """Single-arm geometry signature with the collision entries of the overridden links replaced by exactly the
    override meshes (identity origin, no scale attribute); everything else (all visuals, all other links) unchanged."""
    exp = {k: list(v) for k, v in sig_single.items()}
    for link, meshes in overrides.items():
        if link not in exp:
            raise KeyError(f"override table names {link}, which is not a link of the single-arm model")
        visuals = [e for e in sig_single[link] if e[0] == "visual"]
        cols = [("collision", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "mesh", (("filename", m),)) for m in meshes]
        exp[link] = cols + visuals
    return exp


def xacro_to_urdf(path, args):
    out = subprocess.run(["xacro", path, *args], check=True, capture_output=True, text=True)
    return ET.fromstring(out.stdout)


class Model:
    def __init__(self, root):
        self.joints = {}
        self.parent_joint = {}  # child link -> joint name
        for j in root.findall("joint"):
            name = j.get("name")
            org = j.find("origin")
            xyz = [float(v) for v in (org.get("xyz", "0 0 0") if org is not None else "0 0 0").split()]
            rpy = [float(v) for v in (org.get("rpy", "0 0 0") if org is not None else "0 0 0").split()]
            ax = j.find("axis")
            lim = j.find("limit")
            self.joints[name] = dict(
                type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=np.array(xyz), R=rpy_matrix(*rpy),
                axis=[float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1],
                lower=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                upper=float(lim.get("upper")) if lim is not None and lim.get("upper") else None)
            self.parent_joint[j.find("child").get("link")] = name

    def limits(self, joint):
        j = self.joints[joint]
        return j["lower"], j["upper"]

    def fk(self, base, tip, q):
        """Pose of `tip` in `base` for joint values q {joint name: value}."""
        chain = []
        link = tip
        while link != base:
            if link not in self.parent_joint:
                raise ValueError(f"{tip} is not below {base}")
            jn = self.parent_joint[link]
            chain.append(jn)
            link = self.joints[jn]["parent"]
        T = np.eye(4)
        for jn in reversed(chain):
            j = self.joints[jn]
            Tj = np.eye(4)
            Tj[:3, :3], Tj[:3, 3] = j["R"], j["xyz"]
            if j["type"] in ("revolute", "continuous"):
                Tq = np.eye(4)
                Tq[:3, :3] = axis_angle(j["axis"], q[jn])
                Tj = Tj @ Tq
            elif j["type"] == "prismatic":
                Tq = np.eye(4)
                Tq[:3, 3] = np.asarray(j["axis"], float) / np.linalg.norm(j["axis"]) * q.get(jn, 0.0)
                Tj = Tj @ Tq
            T = T @ Tj
        return T


def run(n=10, tol=1e-9, seed=1, verbose=True):
    single_root = xacro_to_urdf(
        get_package_share_directory("igus_rebel_description") + "/urdf/igus_rebel_xeg32_wrapper.urdf.xacro",
        ["hardware_protocol:=mock_hardware"])
    dual_root = xacro_to_urdf(
        get_package_share_directory("rebel_dual_table_description") + "/urdf/rebel_dual_table.urdf.xacro",
        ["hardware_protocol:=mock_hardware"])
    single, dual = Model(single_root), Model(dual_root)
    links = [ln.get("name") for ln in single_root.findall("link") if ln.get("name") != "world"]
    for f in REQUIRED_FRAMES:
        if f not in links:
            print(f"single-arm model has no link {f}", file=sys.stderr)
            return False, float("inf"), 0
    sig_single = geometry_signature(single_root, "")
    sig_single.pop("world", None)
    overrides = dual_collision_overrides()
    for arm in ("arm_left_", "arm_right_"):
        # the dual-only blade link is checked separately below; it is the only link that may exist only in the dual model
        sigd = geometry_signature(dual_root, arm)
        if BLADE_LINK not in sigd:
            print(f"MISMATCH: {arm}{BLADE_LINK} missing", file=sys.stderr)
            return False, float("inf"), 0
    sig_expected = expected_dual_signature(sig_single, overrides)
    for arm in ("arm_left_", "arm_right_"):
        sig = geometry_signature(dual_root, arm)
        sig.pop(BLADE_LINK, None)
        if sig != sig_expected:
            diff = sorted(k for k in set(sig) | set(sig_expected) if sig.get(k) != sig_expected.get(k))
            print(f"MISMATCH collision/visual geometry of {arm}: links {diff} (expected the single-arm geometry, "
                  f"except the collision of {sorted(overrides)} which must be the override meshes)", file=sys.stderr)
            return False, float("inf"), 0
    share = get_package_share_directory("rebel_dual_table_description")
    for link, meshes in overrides.items():
        for m in meshes:
            prefix = "package://rebel_dual_table_description/"
            if not m.startswith(prefix) or not os.path.isfile(os.path.join(share, m[len(prefix):])):
                print(f"override mesh {m} of {link} is not installed in the description package", file=sys.stderr)
                return False, float("inf"), 0
    s0, cad_pts = blade_cad_points()
    for arm in ("arm_left_", "arm_right_"):
        try:
            bxyz, bsize = blade_box(dual_root, arm)
        except ValueError as e:
            print(f"MISMATCH: {e}", file=sys.stderr)
            return False, float("inf"), 0
        if arm + BLADE_LINK not in dual.parent_joint or dual.joints[dual.parent_joint[arm + BLADE_LINK]]["type"] != "fixed" \
                or dual.joints[dual.parent_joint[arm + BLADE_LINK]]["parent"] != arm + "igus_tool0":
            print(f"MISMATCH: {arm}{BLADE_LINK} is not fixed to {arm}igus_tool0", file=sys.stderr)
            return False, float("inf"), 0
        T_link = dual.fk(arm + "igus_tool0", arm + BLADE_LINK, {})
        tcp = dual.fk(arm + "igus_tool0", arm + "xeg32_tool_tip", {})[:3, 3]
        seat_face = T_link[:3, 3]
        if np.abs(seat_face - s0).max() > 1e-6:
            print(f"MISMATCH: {arm}{BLADE_LINK} frame is not at the CAD seat face centre ({seat_face} vs {s0})", file=sys.stderr)
            return False, float("inf"), 0
        pts = cad_pts + [seat_face, tcp] + [seat_face + t * (tcp - seat_face) for t in np.linspace(0, 1, 11)]
        margin = blade_margin_ok(bxyz, bsize, T_link, pts)
        if margin < BLADE_MARGIN - 1e-9:
            print(f"MISMATCH: {arm}{BLADE_LINK} box leaves only {margin * 1000:.3f} mm margin around the blade/TCP segment "
                  f"(need {BLADE_MARGIN * 1000:.1f} mm)", file=sys.stderr)
            return False, float("inf"), 0
        if bsize.max() > BLADE_MAX_BOX:
            print(f"MISMATCH: {arm}{BLADE_LINK} box is implausibly large {bsize}", file=sys.stderr)
            return False, float("inf"), 0
        if verbose:
            print(f"{arm}{BLADE_LINK}: box {np.round(bsize * 1000, 2).tolist()} mm, min margin {margin * 1000:.3f} mm")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for k in range(n):
        qs = {}
        for jn in JOINTS:
            lo, hi = single.limits(jn)
            qs[jn] = float(rng.uniform(lo, hi))
        for arm in ("arm_left_", "arm_right_"):
            qd = {arm + jn: v for jn, v in qs.items()}
            for f in links:
                Ts = single.fk("base_link", f, qs)
                Td = dual.fk(arm + "base_link", arm + f, qd)
                err = float(np.abs(Ts - Td).max())
                worst = max(worst, err)
                if err > tol:
                    print(f"MISMATCH config {k} {arm}{f}: max abs diff {err:.3e}", file=sys.stderr)
                    return False, worst, 0
        if verbose:
            Ts = single.fk("base_link", "xeg32_tool_tip", qs)
            print(f"config {k}: q={np.round(list(qs.values()), 3).tolist()} tip xyz={np.round(Ts[:3, 3], 6).tolist()}")
    return True, worst, len(links)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    res = run(a.n, a.tol, a.seed)
    ok, worst = res[0], res[1]
    nlinks = res[2]
    print(f"{'PASS' if ok else 'FAIL'}: {a.n} random configs x 2 arms x {nlinks} links, worst |dT| = {worst:.3e} "
          f"(tol {a.tol:g}); visual geometry identical per link, collision identical except the "
          f"{len(dual_collision_overrides())} overridden links; dual-only {BLADE_LINK} box contains blade + TCP segment")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
