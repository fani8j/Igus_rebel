#!/usr/bin/env python3
"""Compare the URDF forward kinematics with the igus ReBeL controller's own Cartesian pose.

The controller reports POSCARTROBOT (x y z a b c) in every CRI STATUS. The igus_rebel driver
publishes it, unconverted, with the joint angles of the same STATUS on <driver>/robot_cart_pose.

  capture   record still poses (joints + raw POSCARTROBOT) to a CSV; subscribes only.
  analyze   compare the CSV with the URDF forward kinematics (position offsets, rigid-base fit,
            Euler convention search).

analyze needs only numpy; capture needs a sourced ROS 2 environment with igus_rebel_msgs.
"""
import argparse
import csv
import itertools
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

CSV_COLUMNS = (
    ["stamp_ns"] + [f"j{i}_rad" for i in range(1, 7)] + ["x", "y", "z", "a", "b", "c"] +
    ["mode", "kinstate", "n_samples", "joint_std_deg_max"])
RAW_COLUMNS = ["x", "y", "z", "a", "b", "c"]

DEFAULT_NODE = "/igus_rebel"          # Igus::Rebel::kDefaultNodeName (single-arm launch)
STILL_JOINT_STD_DEG = 0.02            # reject a pose if any joint's std over the window exceeds it
STILL_CART_STD = 0.05                 # same for the raw cart values (mm / deg); also catches wraps
WINDOW_S = 1.0
MIN_WINDOW_SAMPLES = 10
MIN_POSES = 6
MIN_SPREAD_DEG = 20.0


# --------------------------------------------------------------------------------------------
# Rotation helpers
# --------------------------------------------------------------------------------------------
def rot_axis(axis, angle):
    """Elementary rotation matrix about 'x', 'y' or 'z' (angle in rad)."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(axis)


def rot_vec(axis, angle):
    """Rodrigues rotation about an arbitrary axis."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle) < 1e-15:
        return np.eye(3)
    k = axis / n
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def rpy_matrix(rpy):
    """URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_axis("z", rpy[2]) @ rot_axis("y", rpy[1]) @ rot_axis("x", rpy[0])


def euler_to_matrix(order, intrinsic, angles):
    """Rotation matrix from three angles (rad) given in the order of `order` (e.g. 'zyx').

    extrinsic: rotations about the FIXED axes in the given order: R = R3 R2 R1.
    intrinsic: rotations about the MOVING axes in the given order: R = R1 R2 R3.
    """
    mats = [rot_axis(ax, ang) for ax, ang in zip(order, angles)]
    if intrinsic:
        return mats[0] @ mats[1] @ mats[2]
    return mats[2] @ mats[1] @ mats[0]


def euler_conventions():
    """All 12 axis orders (6 Tait-Bryan, 6 proper Euler) x intrinsic/extrinsic = 24 conventions."""
    orders = [o for o in itertools.product("xyz", repeat=3) if o[0] != o[1] and o[1] != o[2]]
    assert len(orders) == 12
    return [("".join(o), intr) for o in orders for intr in (False, True)]


# With BOTH a free base rotation Rb and a free tool rotation Rt, several conventions give exactly
# the same residual, so they cannot be told apart from poses alone:
#  - Tait-Bryan orders (all three axes distinct) that are cyclic relabellings x->y->z->x of each
#    other (xyz == yzx == zxy; xzy == zyx == yxz), same intrinsic/extrinsic flag: relabelling is
#    a proper rotation P, R' = P R P^T, absorbed by Rb' = P Rb, Rt' = Rt P^T.
#  - Proper Euler orders (first == last axis): the middle axis only differs by a constant
#    rotation about the first axis (Ry(b) = Rz(90) Rx(b) Rz(-90)), so zyz == zxz == xyx == ...
#    all six are equivalent for the same flag.
# Intrinsic abc is NOT equivalent to extrinsic abc (that would need the angles reversed).
_CYCLIC_EVEN = ("xyz", "yzx", "zxy")


def convention_class_key(order, intrinsic):
    if order[0] == order[2]:
        return ("proper", intrinsic)
    return ("tait-bryan-even" if order in _CYCLIC_EVEN else "tait-bryan-odd", intrinsic)


def convention_classes():
    """The 24 conventions grouped into 6 classes that are indistinguishable with free Rb, Rt."""
    classes = {}
    for order, intrinsic in euler_conventions():
        classes.setdefault(convention_class_key(order, intrinsic), []).append((order, intrinsic))
    return list(classes.values())


def convention_name(order, intrinsic):
    return f"{'intrinsic' if intrinsic else 'extrinsic'} {order}"


def rotation_angle(r):
    """Angle (rad) of a rotation matrix."""
    return math.acos(max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0)))


def rotvec_of(r):
    """Axis-angle vector (rad) of a rotation matrix."""
    angle = rotation_angle(r)
    if angle < 1e-12:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:
        w, v = np.linalg.eigh((r + r.T) / 2 - np.eye(3) * math.cos(angle))
        axis = v[:, np.argmax(w)]
    else:
        axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        axis = axis / np.linalg.norm(axis)
    return axis * angle


def kabsch_rotation(h):
    """Rotation Q maximising tr(Q^T H) ... i.e. argmin sum||A - Q B||_F with H = sum A B^T."""
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


# --------------------------------------------------------------------------------------------
# URDF forward kinematics
# --------------------------------------------------------------------------------------------
class Urdf:
    def __init__(self, path_or_text):
        if os.path.exists(path_or_text):
            root = ET.parse(path_or_text).getroot()
        else:
            root = ET.fromstring(path_or_text)
        self.joints = {}      # child link -> joint dict
        for j in root.findall("joint"):     # top level only: <ros2_control> also has <joint> tags
            origin = j.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None
                                               else "0 0 0").split()])
            rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None
                                               else "0 0 0").split()])
            axis_el = j.find("axis")
            axis = np.array([float(v) for v in (axis_el.get("xyz", "1 0 0") if axis_el is not None
                                                else "1 0 0").split()])
            self.joints[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "xyz": xyz, "rpy": rpy, "axis": axis,
                "mimic": j.find("mimic") is not None,
            }

    def chain(self, base_link, tip_link):
        """Joints from base_link to tip_link, base first."""
        out = []
        link = tip_link
        while link != base_link:
            if link not in self.joints:
                raise ValueError(f"no kinematic path from '{base_link}' to '{tip_link}' "
                                 f"(stopped at '{link}')")
            joint = self.joints[link]
            out.append(joint)
            link = joint["parent"]
        out.reverse()
        return out

    def arm_joints(self, base_link, tip_link):
        """The actuated (revolute / continuous) joints of the chain, base first."""
        joints = [j for j in self.chain(base_link, tip_link)
                  if j["type"] in ("revolute", "continuous") and not j["mimic"]]
        for j in self.chain(base_link, tip_link):
            if j["type"] not in ("revolute", "continuous", "fixed", "floating", "planar"):
                raise ValueError(f"unsupported joint type {j['type']} ({j['name']})")
        return joints

    def fk(self, base_link, tip_link, q):
        """4x4 transform base_link -> tip_link for the 6 joint values q (rad, chain order)."""
        t = np.eye(4)
        qi = 0
        for j in self.chain(base_link, tip_link):
            origin = np.eye(4)
            origin[:3, :3] = rpy_matrix(j["rpy"])
            origin[:3, 3] = j["xyz"]
            t = t @ origin
            if j["type"] in ("revolute", "continuous") and not j["mimic"]:
                m = np.eye(4)
                m[:3, :3] = rot_vec(j["axis"], q[qi])
                t = t @ m
                qi += 1
        return t


def default_links(prefix):
    return prefix + "base_link", prefix + "link6"


def resolve_link(name, prefix):
    if prefix and not name.startswith(prefix):
        return prefix + name
    return name


# --------------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no poses")
    q = np.array([[float(r[f"j{i}_rad"]) for i in range(1, 7)] for r in rows])
    raw = np.array([[float(r[c]) for c in RAW_COLUMNS] for r in rows])
    return q, raw, rows


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------
def fit_translation_and_tool(p_ours, r_ours, p_igus):
    """p_igus = p_ours + R t + b (all metres). Returns b, t, residuals (N,3), singular values."""
    n = len(p_ours)
    a = np.zeros((3 * n, 6))
    y = np.zeros(3 * n)
    for i in range(n):
        a[3 * i:3 * i + 3, 0:3] = np.eye(3)
        a[3 * i:3 * i + 3, 3:6] = r_ours[i]
        y[3 * i:3 * i + 3] = p_igus[i] - p_ours[i]
    sol, _, _, sv = np.linalg.lstsq(a, y, rcond=None)
    res = (y - a @ sol).reshape(n, 3)
    return sol[0:3], sol[3:6], res, sv


def rigid_fit(src, dst):
    """Rb, b minimising sum|dst - (Rb src + b)|^2."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    h = (dst - cd).T @ (src - cs)
    rb = kabsch_rotation(h)
    return rb, cd - rb @ cs


def fit_base_rotation_and_tool(p_ours, r_ours, p_igus, t0):
    """p_igus = Rb (p_ours + R t) + b; Levenberg-Marquardt on t with Kabsch inside."""
    def solve(t):
        q = p_ours + np.einsum("nij,j->ni", r_ours, t)
        rb, b = rigid_fit(q, p_igus)
        res = p_igus - (q @ rb.T + b)
        return rb, b, res

    t = np.array(t0, dtype=float)
    lam = 1e-3
    _, _, res = solve(t)
    cost = float(np.sum(res ** 2))
    for _ in range(100):
        r0 = res.ravel()
        jac = np.zeros((r0.size, 3))
        h = 1e-7
        for k in range(3):
            dt = np.zeros(3)
            dt[k] = h
            jac[:, k] = (solve(t + dt)[2].ravel() - r0) / h
        jtj, jtr = jac.T @ jac, jac.T @ r0
        improved = False
        for _ in range(12):
            try:
                step = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), -jtr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            _, _, res_new = solve(t + step)
            cost_new = float(np.sum(res_new ** 2))
            if cost_new < cost:
                t, res, cost, lam, improved = t + step, res_new, cost_new, lam / 3, True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-12:
            break
    rb, b, res = solve(t)
    return rb, b, t, res


def euler_matrices(angles_rad, order, intrinsic):
    return np.array([euler_to_matrix(order, intrinsic, a) for a in angles_rad])


def fit_rotations(r_ours, r_igus, inits):
    """R_igus = Rb R_ours Rt by alternating Procrustes from several starts. Returns Rb, Rt, rms."""
    best = None
    for rt in inits:
        prev = None
        rb = np.eye(3)
        for _ in range(100):
            # Rb = argmin sum |A - Rb (R Rt)|: H = sum A (R Rt)^T
            h = np.einsum("nij,nkj->ik", r_igus, r_ours @ rt)
            rb = kabsch_rotation(h)
            # Rt = argmin sum |Rb^T A - R Rt|: H = sum R^T (Rb^T A)
            h = np.einsum("nji,njk->ik", r_ours, rb.T @ r_igus)
            rt = kabsch_rotation(h)
            cost = float(np.sum((r_igus - rb @ r_ours @ rt) ** 2))
            if prev is not None and abs(prev - cost) < 1e-13:
                break
            prev = cost
        res = [rotation_angle(a.T @ rb @ r @ rt) for a, r in zip(r_igus, r_ours)]
        rms = math.sqrt(sum(x * x for x in res) / len(res))
        if best is None or rms < best[2]:
            best = (rb, rt, rms, res)
    return best


def cube_rotations():
    """The 24 proper rotations that map the coordinate axes onto themselves."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if np.linalg.det(m) > 0:
                out.append(m)
    return out


def analyze(urdf_path, csv_path, base_link="base_link", tip_link="link6", prefix="",
            pos_units="mm", angle_units="deg"):
    """Returns a dict with all results and a list of warnings."""
    base_link, tip_link = resolve_link(base_link, prefix), resolve_link(tip_link, prefix)
    q, raw, rows = read_csv(csv_path)
    urdf = Urdf(urdf_path)
    arm = urdf.arm_joints(base_link, tip_link)
    if len(arm) != 6:
        raise ValueError(f"expected 6 revolute joints between {base_link} and {tip_link}, "
                         f"found {len(arm)}: {[j['name'] for j in arm]}")
    n = len(q)
    warnings = []

    t_ours = [urdf.fk(base_link, tip_link, qi) for qi in q]
    p_ours = np.array([t[:3, 3] for t in t_ours])
    r_ours = np.array([t[:3, :3] for t in t_ours])

    scale = {"mm": 1e-3, "m": 1.0}[pos_units]
    p_igus = raw[:, 0:3] * scale
    ang_scale = {"deg": math.pi / 180.0, "rad": 1.0}[angle_units]
    angles = raw[:, 3:6] * ang_scale

    # ---- flags ----
    if n < MIN_POSES:
        warnings.append(f"only {n} poses (< {MIN_POSES}): fits are underdetermined or unreliable")
    if pos_units == "mm" and float(np.max(np.abs(raw[:, 0:3]))) < 5.0:
        warnings.append("position values are all < 5: they look like METRES, not mm "
                        "(rerun with --pos-units m)")
    if pos_units == "m" and float(np.max(np.abs(raw[:, 0:3]))) > 10.0:
        warnings.append("position values exceed 10 with --pos-units m: they look like mm")
    if angle_units == "deg" and float(np.max(np.abs(raw[:, 3:6]))) <= 2 * math.pi + 0.01:
        warnings.append("angle values are all within +-2*pi: they look like RADIANS, not degrees "
                        "(rerun with --angle-units rad)")
    spread = np.degrees(q.max(axis=0) - q.min(axis=0))
    for i in range(1, 6):
        if spread[i] < MIN_SPREAD_DEG:
            warnings.append(f"joint {i + 1} spans only {spread[i]:.1f} deg across the poses "
                            f"(< {MIN_SPREAD_DEG:.0f}): poses not spread enough")
    modes = sorted({r.get("mode", "") for r in rows})
    if any(m not in ("", "joint", "cartbase", "carttool", "platform", "fsm") for m in modes):
        warnings.append(f"unexpected STATUS modes in CSV: {modes}")

    result = {"n": n, "base_link": base_link, "tip_link": tip_link, "warnings": warnings,
              "spread_deg": spread, "joint_names": [j["name"] for j in arm]}

    # ---- (a) raw difference ----
    diff = p_igus - p_ours
    result["raw_diff_mm"] = diff * 1e3
    result["raw_diff_norm_mm"] = np.linalg.norm(diff, axis=1) * 1e3
    result["p_ours_mm"] = p_ours * 1e3

    # ---- (b) constant base offset + tool offset ----
    b, t, res, sv = fit_translation_and_tool(p_ours, r_ours, p_igus)
    result["b_fit"] = {"b_mm": b * 1e3, "t_mm": t * 1e3, "res_mm": res * 1e3,
                       "res_norm_mm": np.linalg.norm(res, axis=1) * 1e3,
                       "rms_mm": math.sqrt(float(np.mean(np.sum(res ** 2, axis=1)))) * 1e3,
                       "singular_values": sv}
    if sv[-1] < 1e-3 * sv[0]:
        warnings.append("fit (b) is ill-conditioned (tool orientation barely changes across the "
                        "poses): the base and tool offsets are not separable")

    # ---- (c) rigid base + tool offset ----
    if n >= 4:
        rb, b2, t2, res2 = fit_base_rotation_and_tool(p_ours, r_ours, p_igus, t)
        result["c_fit"] = {"Rb": rb, "rotvec_deg": np.degrees(rotvec_of(rb)), "b_mm": b2 * 1e3,
                           "t_mm": t2 * 1e3, "res_mm": res2 * 1e3,
                           "res_norm_mm": np.linalg.norm(res2, axis=1) * 1e3,
                           "rms_mm": math.sqrt(float(np.mean(np.sum(res2 ** 2, axis=1)))) * 1e3}
    else:
        result["c_fit"] = None
        warnings.append("fit (c) needs >= 4 poses")

    # ---- (d) orientation: every Euler convention ----
    inits = cube_rotations()
    rng = np.random.default_rng(1)
    for _ in range(3):
        v = rng.normal(size=3)
        inits.append(rot_vec(v, rng.uniform(0, math.pi)))
    conv_results = []
    for members in convention_classes():
        member_list = []
        for order, intrinsic in members:
            r_igus = euler_matrices(angles, order, intrinsic)
            rb, rt, rms, res_rot = fit_rotations(r_ours, r_igus, inits)
            member_list.append({"order": order, "intrinsic": intrinsic,
                                "name": convention_name(order, intrinsic),
                                "Rb": rb, "Rt": rt, "rms_deg": math.degrees(rms),
                                "res_deg": np.degrees(res_rot),
                                "rb_angle_deg": math.degrees(rotation_angle(rb))})
        # Same residual for every member (exact equivalence): the member whose base rotation is
        # closest to the identity is the most plausible reading.
        member_list.sort(key=lambda m: (round(m["rms_deg"], 3), m["rb_angle_deg"]))
        top = member_list[0]
        conv_results.append({"members": member_list, "rms_deg": top["rms_deg"],
                             "res_deg": top["res_deg"], "name": " == ".join(
                                 m["name"] for m in member_list),
                             **{k: top[k] for k in ("order", "intrinsic", "Rb", "Rt")}})
    conv_results.sort(key=lambda c: c["rms_deg"])
    result["conventions"] = conv_results
    best, second = conv_results[0], conv_results[1]
    result["orientation_best"] = best
    result["orientation_margin_deg"] = second["rms_deg"] - best["rms_deg"]
    result["orientation_runner_up"] = second
    if n < 6:
        warnings.append("orientation convention search is not meaningful with < 6 poses")
    if second["rms_deg"] < 2.0 * best["rms_deg"] + 0.05:
        warnings.append("no Euler convention is clearly best (runner-up residual is similar): "
                        "check the units/frame, or capture more spread poses")
    return result


def fmt_vec(v, unit="mm"):
    return "[" + ", ".join(f"{x:9.3f}" for x in v) + f"] {unit}"


def verdict_label(rms_mm):
    if rms_mm <= 1.0:
        return "consistent"
    if rms_mm <= 3.0:
        return "marginal"
    return "MISMATCH"


def print_report(res, out=None):
    out = out or sys.stdout
    w = lambda s="": print(s, file=out)  # noqa: E731
    w(f"Poses: {res['n']}   chain: {res['base_link']} -> {res['tip_link']}   "
      f"joints: {', '.join(res['joint_names'])}")
    w(f"Joint spread per joint [deg]: " + ", ".join(f"{s:.1f}" for s in res["spread_deg"]))
    w()
    w("(a) raw position difference (igus - ours), assuming mm")
    for i in range(res["n"]):
        w(f"  pose {i:2d}: {fmt_vec(res['raw_diff_mm'][i])}  |d| = {res['raw_diff_norm_mm'][i]:8.3f} mm")
    w()
    bf = res["b_fit"]
    w("(b) fit p_igus = p_ours + R_ours t + b   (b: base offset in base frame, "
      "t: tool offset in tip frame)")
    w(f"  b = {fmt_vec(bf['b_mm'])}")
    w(f"  t = {fmt_vec(bf['t_mm'])}")
    for i in range(res["n"]):
        w(f"  pose {i:2d}: residual {bf['res_norm_mm'][i]:8.3f} mm")
    w(f"  RMS = {bf['rms_mm']:.3f} mm")
    w()
    cf = res["c_fit"]
    if cf is not None:
        w("(c) fit p_igus = Rb (p_ours + R_ours t) + b   (rigid base transform + tool offset)")
        w(f"  Rb rotation vector = {fmt_vec(cf['rotvec_deg'], 'deg')}")
        w(f"  b = {fmt_vec(cf['b_mm'])}")
        w(f"  t = {fmt_vec(cf['t_mm'])}")
        for i in range(res["n"]):
            w(f"  pose {i:2d}: residual {cf['res_norm_mm'][i]:8.3f} mm")
        w(f"  RMS = {cf['rms_mm']:.3f} mm")
        w()
    w("(d) orientation: R_igus(a,b,c) = Rb R_ours Rt for each Euler convention (degrees)")
    w("    With Rb and Rt both free, cyclic relabellings of Tait-Bryan orders and all proper-Euler")
    w("    orders of the same flavour fit identically: they are grouped into 6 classes; the member")
    w("    with the smallest base rotation is the most plausible one.")
    for c in res["conventions"]:
        names = c["name"] if len(c["name"]) < 70 else f"{c['members'][0]['name']} (+{len(c['members']) - 1} equivalent)"
        w(f"  {names:<72} angular RMS = {c['rms_deg']:8.4f} deg")
    best, second = res["orientation_best"], res["orientation_runner_up"]
    ratio = second["rms_deg"] / best["rms_deg"] if best["rms_deg"] > 1e-9 else float("inf")
    pick = best["members"][0]
    w(f"  best class: {best['name']}  (RMS {best['rms_deg']:.4f} deg)")
    w(f"  runner-up : {second['name']}  (RMS {second['rms_deg']:.4f} deg); best beats it by "
      f"{res['orientation_margin_deg']:.4f} deg ({ratio:.1f}x)")
    w(f"  most plausible member: {pick['name']}: Rb rotation vector = "
      f"{fmt_vec(np.degrees(rotvec_of(pick['Rb'])), 'deg')}")
    w(f"  Rt rotation vector (tool/flange frame) = {fmt_vec(np.degrees(rotvec_of(pick['Rt'])), 'deg')}")
    w()
    if res["warnings"]:
        w("FLAGS")
        for m in res["warnings"]:
            w(f"  - {m}")
        w()
    line = (f"VERDICT: RMS position residual after (b) base+tool offset = {bf['rms_mm']:.3f} mm "
            f"[{verdict_label(bf['rms_mm'])}]")
    if cf is not None:
        line += (f"; after (c) rigid base+tool = {cf['rms_mm']:.3f} mm "
                 f"[{verdict_label(cf['rms_mm'])}]")
    line += (f"; orientation best {best['members'][0]['name']} "
             f"(1 of {len(best['members'])} equivalent) {best['rms_deg']:.3f} deg, "
             f"{res['orientation_margin_deg']:.3f} deg ahead of runner-up")
    w(line)
    w("  (labels are heuristic: <=1 mm consistent, <=3 mm marginal, else mismatch; "
      "judge from the residual pattern, not only the RMS)")


# --------------------------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------------------------
def evaluate_window(samples):
    """samples: list of dicts {stamp_ns, q (6 rad), raw (6), kinstate, mode}.

    Returns (row_dict, None) for an accepted pose or (None, reason).
    """
    if len(samples) < MIN_WINDOW_SAMPLES:
        return None, f"only {len(samples)} samples in the window (need >= {MIN_WINDOW_SAMPLES})"
    kin = sorted({s["kinstate"] for s in samples})
    if kin != [0]:
        return None, f"kinstate not zero: {kin}"
    q = np.array([s["q"] for s in samples])
    raw = np.array([s["raw"] for s in samples])
    q_std_deg = np.degrees(q.std(axis=0))
    if q_std_deg.max() > STILL_JOINT_STD_DEG:
        return None, (f"robot not still: joint std {q_std_deg.max():.4f} deg "
                      f"(limit {STILL_JOINT_STD_DEG})")
    raw_std = raw.std(axis=0)
    if raw_std.max() > STILL_CART_STD:
        return None, f"POSCARTROBOT not stable: std {raw_std.max():.4f} (limit {STILL_CART_STD})"
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(raw))):
        return None, "non-finite values"
    mean_q, mean_raw = q.mean(axis=0), raw.mean(axis=0)
    row = {"stamp_ns": int(np.median([s["stamp_ns"] for s in samples])),
           "mode": samples[-1]["mode"], "kinstate": 0, "n_samples": len(samples),
           "joint_std_deg_max": f"{q_std_deg.max():.5f}"}
    for i in range(6):
        row[f"j{i + 1}_rad"] = f"{mean_q[i]:.9f}"
    for c, v in zip(RAW_COLUMNS, mean_raw):
        row[c] = f"{v:.6f}"
    return row, None


def append_row(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            wr.writeheader()
        wr.writerow(row)


def run_capture(args):
    import threading
    import time

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from igus_rebel_msgs.msg import RobotCartPose

    rclpy.init()
    node = Node("igus_fk_capture")
    topic = args.node.rstrip("/") + "/robot_cart_pose"
    lock = threading.Lock()
    buf = []

    def on_msg(msg):
        with lock:
            buf.append({"stamp_ns": msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec,
                        "q": list(msg.joint_position_rad), "raw": list(msg.pos_cart_robot_raw),
                        "kinstate": int(msg.kinstate), "mode": msg.mode,
                        "t_recv": time.monotonic()})
            del buf[:-2000]

    node.create_subscription(RobotCartPose, topic, on_msg, qos_profile_sensor_data)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    print(f"Subscribed to {topic} (subscribe only). Output: {args.out}")
    accepted = 0
    attempts = 0
    try:
        t0 = time.monotonic()
        while True:
            with lock:
                have = bool(buf) and time.monotonic() - buf[-1]["t_recv"] < 1.0
            if have:
                break
            if time.monotonic() - t0 > args.wait:
                print(f"No messages on {topic} after {args.wait:.0f} s "
                      f"(domain {os.environ.get('ROS_DOMAIN_ID')}); is the driver running?")
                return 2
            time.sleep(0.1)
        while args.auto is None or accepted < args.auto:
            if args.auto is None:
                try:
                    cmd = input(f"[{accepted} saved] Hold the arm still, then press Enter "
                                f"to capture (q + Enter to finish): ")
                except EOFError:
                    break
                if cmd.strip().lower() in ("q", "quit", "exit"):
                    break
            elif attempts:
                time.sleep(args.auto_interval)
            attempts += 1
            if args.auto is not None and attempts > args.auto + 10:
                print("Too many rejected attempts; giving up.")
                return 3
            start = time.monotonic()
            time.sleep(WINDOW_S)
            with lock:
                window = [s for s in buf if s["t_recv"] >= start]
            row, reason = evaluate_window(window)
            if row is None:
                print(f"  rejected: {reason}")
                continue
            append_row(args.out, row)
            accepted += 1
            q_deg = [float(row[f"j{i}_rad"]) * 180 / math.pi for i in range(1, 7)]
            print("  saved pose %d: joints [deg] %s; raw %s (n=%d, joint std %s deg)" % (
                accepted, ", ".join(f"{v:.2f}" for v in q_deg),
                ", ".join(row[c] for c in RAW_COLUMNS), row["n_samples"],
                row["joint_std_deg_max"]))
    finally:
        stop.set()
        spinner.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
    print(f"{accepted} pose(s) appended to {args.out}")
    return 0 if accepted else 1


# --------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="record still poses (subscribe only)")
    c.add_argument("--node", default=DEFAULT_NODE,
                   help=f"driver node name (default {DEFAULT_NODE}); topic is <node>/robot_cart_pose")
    c.add_argument("--out", default="fk_capture.csv", help="CSV to append to")
    c.add_argument("--auto", type=int, default=None, metavar="N",
                   help="non-interactive: capture N poses without waiting for Enter")
    c.add_argument("--auto-interval", type=float, default=2.0,
                   help="seconds between automatic captures (default 2)")
    c.add_argument("--wait", type=float, default=15.0,
                   help="seconds to wait for the first message (default 15)")
    a = sub.add_parser("analyze", help="compare a capture CSV with the URDF kinematics")
    a.add_argument("csv")
    a.add_argument("--urdf", required=True, help="expanded URDF file (xacro output)")
    a.add_argument("--prefix", default="", help="dual-arm link prefix, e.g. arm_left_")
    a.add_argument("--base-link", default="base_link")
    a.add_argument("--tip-link", default="link6", help="link6 (flange) or xeg32_tool_tip")
    a.add_argument("--pos-units", choices=["mm", "m"], default="mm")
    a.add_argument("--angle-units", choices=["deg", "rad"], default="deg")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "capture":
        return run_capture(args)
    res = analyze(args.urdf, args.csv, args.base_link, args.tip_link, args.prefix,
                  args.pos_units, args.angle_units)
    print_report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
