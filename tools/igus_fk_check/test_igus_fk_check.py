"""Tests for igus_fk_check: synthetic captures from the real ReBeL URDF with injected offsets.

Run: cd tools/igus_fk_check && python3 -m pytest -p no:anyio test_igus_fk_check.py
"""
import csv
import io
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import igus_fk_check as fk

HERE = os.path.dirname(os.path.abspath(__file__))
DESC_PKG = os.path.abspath(os.path.join(HERE, "..", "..", "src", "igus_rebel_description"))
XACRO_FILE = os.path.join(DESC_PKG, "urdf", "igus_rebel_xeg32_wrapper.urdf.xacro")

pytestmark = pytest.mark.skipif(
    shutil.which("xacro") is None or not os.path.exists(XACRO_FILE),
    reason="needs xacro (source /opt/ros/humble/setup.bash) and igus_rebel_description")


def expand_urdf(tmp_dir):
    """Expand the single-arm xacro with the xeg32 wrapper, no ros2_control, no install needed.

    xacro's $(find igus_rebel_description) goes through the ament index, so a throw-away prefix
    with a symlink to the source package is enough.
    """
    prefix = os.path.join(tmp_dir, "prefix")
    os.makedirs(os.path.join(prefix, "share", "ament_index", "resource_index", "packages"))
    open(os.path.join(prefix, "share", "ament_index", "resource_index", "packages",
                      "igus_rebel_description"), "w").close()
    os.symlink(DESC_PKG, os.path.join(prefix, "share", "igus_rebel_description"))
    env = dict(os.environ)
    env["AMENT_PREFIX_PATH"] = prefix + ":" + env.get("AMENT_PREFIX_PATH", "")
    out = subprocess.run(
        ["xacro", XACRO_FILE, "hardware_protocol:=mock_hardware", "include_ros2_control:=false"],
        check=True, capture_output=True, text=True, env=env).stdout
    path = os.path.join(tmp_dir, "robot.urdf")
    with open(path, "w") as f:
        f.write(out)
    return path


@pytest.fixture(scope="module")
def urdf_path(tmp_path_factory):
    return expand_urdf(str(tmp_path_factory.mktemp("urdf")))


def sample_joints(urdf, n, seed):
    rng = np.random.default_rng(seed)
    arm = urdf.arm_joints("base_link", "link6")
    assert len(arm) == 6
    # Spread over +-60 % of a conservative range (the ReBeL joint limits are about +-160 deg for
    # j1, j2..j5 smaller): +-70 deg on j2/j3/j5, +-120 deg elsewhere.
    span = np.radians([150, 60, 60, 120, 70, 150])
    return rng.uniform(-span, span, size=(n, 6))


EXTRINSIC_ZYX = ("zyx", False)
INTRINSIC_ZYZ = ("zyz", True)


def make_capture(urdf, q, tip="link6", base_off_mm=(12.3, -4.5, 7.8), tool_off_mm=(3.0, -5.0, 87.0),
                 base_rot=(0.0, 0.0, 0.0), conv=EXTRINSIC_ZYX, tool_rot_deg=(0.0, 0.0, 0.0),
                 pos_noise_mm=0.1, ang_noise_deg=0.02, seed=3, urdf_for_truth=None):
    """Synthetic 'controller' pose: Rb (p + R t) + b, orientation Rb R Rt as Euler (deg)."""
    rng = np.random.default_rng(seed)
    truth = urdf_for_truth or urdf
    rb = fk.rot_vec([0, 0, 1], math.radians(base_rot[2])) @ \
        fk.rot_vec([0, 1, 0], math.radians(base_rot[1])) @ \
        fk.rot_vec([1, 0, 0], math.radians(base_rot[0]))
    rt = fk.rot_vec([0, 0, 1], math.radians(tool_rot_deg[2])) @ \
        fk.rot_vec([0, 1, 0], math.radians(tool_rot_deg[1])) @ \
        fk.rot_vec([1, 0, 0], math.radians(tool_rot_deg[0]))
    rows = []
    for qi in q:
        t = truth.fk("base_link", tip, qi)
        p, r = t[:3, 3], t[:3, :3]
        p_igus = rb @ (p + r @ (np.array(tool_off_mm) * 1e-3)) + np.array(base_off_mm) * 1e-3
        p_igus_mm = p_igus * 1e3 + rng.normal(scale=pos_noise_mm, size=3)
        r_igus = rb @ r @ rt
        ang = euler_from_matrix(r_igus, conv[0], conv[1])
        ang_deg = np.degrees(ang) + rng.normal(scale=ang_noise_deg, size=3)
        rows.append((qi, p_igus_mm, ang_deg))
    return rows


def euler_from_matrix(r, order, intrinsic):
    """Numerically invert euler_to_matrix (3 unknowns, Gauss-Newton from several starts)."""
    best = None
    rng = np.random.default_rng(0)
    for start in range(40):
        a = rng.uniform(-math.pi, math.pi, 3) if start else np.zeros(3)
        for _ in range(60):
            f = (fk.euler_to_matrix(order, intrinsic, a) - r).ravel()
            jac = np.zeros((9, 3))
            for k in range(3):
                d = np.zeros(3)
                d[k] = 1e-7
                jac[:, k] = ((fk.euler_to_matrix(order, intrinsic, a + d) - r).ravel() - f) / 1e-7
            step = np.linalg.lstsq(jac, -f, rcond=None)[0]
            a = a + step
            if np.linalg.norm(step) < 1e-13:
                break
        err = np.linalg.norm(fk.euler_to_matrix(order, intrinsic, a) - r)
        if best is None or err < best[0]:
            best = (err, a)
        if err < 1e-9:
            break
    assert best[0] < 1e-6, f"euler inversion failed ({best[0]})"
    return best[1]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fk.CSV_COLUMNS)
        w.writeheader()
        for i, (q, p_mm, ang_deg) in enumerate(rows):
            row = {"stamp_ns": i, "mode": "joint", "kinstate": 0, "n_samples": 50,
                   "joint_std_deg_max": "0.001"}
            for k in range(6):
                row[f"j{k + 1}_rad"] = f"{q[k]:.9f}"
            for c, v in zip(fk.RAW_COLUMNS, list(p_mm) + list(ang_deg)):
                row[c] = f"{v:.6f}"
            w.writerow(row)


# --------------------------------------------------------------------------------------------
def test_chain_and_zero_pose_height(urdf_path):
    urdf = fk.Urdf(urdf_path)
    arm = urdf.arm_joints("base_link", "link6")
    assert [j["name"] for j in arm] == [f"joint{i}" for i in range(1, 7)]
    # The same chain reaches the kinematic TCP; link6 -> tool tip is fixed joints only.
    assert len(urdf.arm_joints("base_link", "xeg32_tool_tip")) == 6
    t = urdf.fk("base_link", "link6", np.zeros(6))
    # Upright at zero: x=y~0 (small offsets from the motor mounts), z = sum of the link heights.
    assert abs(t[0, 3]) < 0.01 and abs(t[1, 3]) < 0.01
    assert t[2, 3] == pytest.approx(0.87, abs=0.001)  # sum of the link heights
    # joint1 rotates about the vertical axis: tip stays at the same height.
    t1 = urdf.fk("base_link", "link6", np.array([0.7, 0, 0, 0, 0, 0]))
    assert t1[2, 3] == pytest.approx(t[2, 3], abs=1e-9)


def test_equivalence_classes_are_really_equivalent():
    classes = fk.convention_classes()
    assert sorted(len(c) for c in classes) == [3, 3, 3, 3, 6, 6]
    assert len({m for c in classes for m in c}) == 24
    # Same orientation set up to constant Rb, Rt: two members of a class fit the same data with
    # the same residual; members of different classes do not.
    rng = np.random.default_rng(0)
    r_ours = np.array([fk.rot_vec(rng.normal(size=3), rng.uniform(0, 3)) for _ in range(10)])
    angles = rng.uniform(-1, 1, size=(10, 3))
    inits = fk.cube_rotations()
    rms = {}
    for members in classes:
        vals = []
        for order, intr in members:
            r_igus = fk.euler_matrices(angles, order, intr)
            vals.append(fk.fit_rotations(r_ours, r_igus, inits)[2])
        rms[tuple(members)] = vals
    for vals in rms.values():
        assert max(vals) - min(vals) < 1e-6 * (1 + max(vals)) or max(vals) < 1e-6


def test_euler_conventions_are_24_distinct_matrices():
    convs = fk.euler_conventions()
    assert len(convs) == 24
    angles = (0.3, -0.5, 0.9)
    mats = [fk.euler_to_matrix(o, i, angles) for o, i in convs]
    for m in mats:
        assert np.allclose(m @ m.T, np.eye(3)) and np.linalg.det(m) == pytest.approx(1.0)
    # intrinsic xyz == extrinsic zyx with the angles reversed (sanity of the two definitions)
    assert np.allclose(fk.euler_to_matrix("xyz", True, angles),
                       fk.euler_to_matrix("zyx", False, angles[::-1]))
    # distinct for these angles
    keys = {tuple(np.round(m.ravel(), 6)) for m in mats}
    assert len(keys) == 24


def test_recovers_base_tool_offsets_and_convention(urdf_path, tmp_path):
    urdf = fk.Urdf(urdf_path)
    q = sample_joints(urdf, 10, seed=1)
    rows = make_capture(urdf, q, base_off_mm=(12.3, -4.5, 7.8), tool_off_mm=(3.0, -5.0, 87.0),
                        base_rot=(0.0, 0.0, 0.0), conv=EXTRINSIC_ZYX,
                        tool_rot_deg=(0, 0, 90), pos_noise_mm=0.1, ang_noise_deg=0.02)
    csv_path = str(tmp_path / "cap.csv")
    write_csv(csv_path, rows)
    res = fk.analyze(urdf_path, csv_path)

    # (a) raw difference is large (offsets), not noise
    assert res["raw_diff_norm_mm"].mean() > 50
    # (b) offsets recovered to ~noise, RMS ~ noise
    b = res["b_fit"]
    assert b["rms_mm"] < 0.3
    assert np.allclose(b["b_mm"], [12.3, -4.5, 7.8], atol=0.5)
    assert np.allclose(b["t_mm"], [3.0, -5.0, 87.0], atol=0.5)
    assert res["c_fit"]["rms_mm"] < 0.3
    # (d) the injected convention wins clearly
    best = res["orientation_best"]
    assert (best["order"], best["intrinsic"]) == EXTRINSIC_ZYX   # smallest-base-rotation member
    assert best["rms_deg"] < 0.1
    assert len(best["members"]) == 3
    assert res["orientation_margin_deg"] > 1.0
    assert not [w for w in res["warnings"] if "spans only" in w or "only" in w and "poses" in w]


def test_recovers_rigid_base_rotation_and_intrinsic_convention(urdf_path, tmp_path):
    urdf = fk.Urdf(urdf_path)
    q = sample_joints(urdf, 12, seed=2)
    rows = make_capture(urdf, q, base_off_mm=(-30.0, 55.0, 12.0), tool_off_mm=(-4.0, 6.0, 120.0),
                        base_rot=(0.4, -0.3, 1.5), conv=INTRINSIC_ZYZ,
                        tool_rot_deg=(10, -20, 30), pos_noise_mm=0.1, ang_noise_deg=0.02, seed=5)
    csv_path = str(tmp_path / "cap.csv")
    write_csv(csv_path, rows)
    res = fk.analyze(urdf_path, csv_path)
    # A constant translation cannot absorb a rotated base: (b) is visibly worse, (c) absorbs it.
    assert res["b_fit"]["rms_mm"] > 2.0
    c = res["c_fit"]
    assert c["rms_mm"] < 0.3
    assert np.allclose(c["t_mm"], [-4.0, 6.0, 120.0], atol=1.0)
    rb_true = fk.rot_vec([0, 0, 1], math.radians(1.5)) @ fk.rot_vec([0, 1, 0], math.radians(-0.3)) \
        @ fk.rot_vec([1, 0, 0], math.radians(0.4))
    assert fk.rotation_angle(c["Rb"].T @ rb_true) < math.radians(0.05)
    best = res["orientation_best"]
    # All proper-Euler orders fit equally well (free Rb, Rt) but with large base rotations; the
    # reported member is the one with the small base rotation = the injected one.
    assert len(best["members"]) == 6 and all(m["intrinsic"] for m in best["members"])
    assert all(m["order"][0] == m["order"][2] for m in best["members"])
    assert (best["order"], best["intrinsic"]) == INTRINSIC_ZYZ
    assert fk.rotation_angle(best["Rb"].T @ rb_true) < math.radians(0.1)
    assert best["rms_deg"] < 0.1
    assert res["orientation_margin_deg"] > 1.0


def test_tool_tip_link_and_cli_report(urdf_path, tmp_path, capsys):
    urdf = fk.Urdf(urdf_path)
    q = sample_joints(urdf, 8, seed=4)
    rows = make_capture(urdf, q, tip="xeg32_tool_tip", tool_off_mm=(0.5, -0.5, 1.0),
                        base_off_mm=(0, 0, 0), conv=EXTRINSIC_ZYX, seed=7)
    csv_path = str(tmp_path / "cap.csv")
    write_csv(csv_path, rows)
    rc = fk.main(["analyze", csv_path, "--urdf", urdf_path, "--tip-link", "xeg32_tool_tip"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VERDICT" in out and "RMS position residual after (b)" in out
    assert "(a) raw position difference" in out and "(d) orientation" in out
    assert "most plausible member: extrinsic zyx" in out


def test_wrong_link_length_gives_large_residual(urdf_path, tmp_path):
    urdf_good = fk.Urdf(urdf_path)
    with open(urdf_path) as f:
        text = f.read()
    # Lengthen the upper arm (J_motor3 z 0.24152 -> 0.26152 = +20 mm) in OUR model only.
    assert "0.24152" in text
    bad_path = str(tmp_path / "bad.urdf")
    with open(bad_path, "w") as f:
        f.write(text.replace("0.24152", "0.26152"))
    q = sample_joints(urdf_good, 10, seed=6)
    rows = make_capture(urdf_good, q, pos_noise_mm=0.1)
    csv_path = str(tmp_path / "cap.csv")
    write_csv(csv_path, rows)
    good = fk.analyze(urdf_path, csv_path)
    bad = fk.analyze(bad_path, csv_path)
    assert good["b_fit"]["rms_mm"] < 0.3
    assert bad["b_fit"]["rms_mm"] > 3.0
    assert bad["b_fit"]["rms_mm"] > 10 * good["b_fit"]["rms_mm"]
    assert bad["c_fit"]["rms_mm"] > 3.0
    buf = io.StringIO()
    fk.print_report(bad, buf)
    assert "MISMATCH" in buf.getvalue()


def test_flags_units_few_poses_and_spread(urdf_path, tmp_path):
    urdf = fk.Urdf(urdf_path)
    # Metres + radians, 3 poses, tiny spread on joint 3.
    q = sample_joints(urdf, 3, seed=8)
    q[:, 2] = np.radians([10.0, 12.0, 14.0])
    rows = make_capture(urdf, q, base_off_mm=(0, 0, 0), tool_off_mm=(0, 0, 0))
    rows = [(qi, p / 1000.0, np.radians(a) * 0.1) for qi, p, a in rows]
    csv_path = str(tmp_path / "cap.csv")
    write_csv(csv_path, rows)
    res = fk.analyze(urdf_path, csv_path)
    joined = " | ".join(res["warnings"])
    assert "METRES" in joined
    assert "RADIANS" in joined
    assert "only 3 poses" in joined
    assert "joint 3 spans only" in joined
    # joint 1 is exempt from the spread rule
    q[:, 0] = 0.0
    write_csv(csv_path, [(qi, p, a) for qi, (_, p, a) in zip(q, rows)])
    res = fk.analyze(urdf_path, csv_path)
    assert not any("joint 1 spans" in w for w in res["warnings"])


def test_dual_arm_prefix_resolution():
    assert fk.resolve_link("base_link", "arm_left_") == "arm_left_base_link"
    assert fk.resolve_link("arm_left_link6", "arm_left_") == "arm_left_link6"
    assert fk.resolve_link("link6", "") == "link6"
    assert fk.default_links("arm_right_") == ("arm_right_base_link", "arm_right_link6")


def test_dual_arm_prefixed_urdf(urdf_path, tmp_path):
    """The same single-arm URDF with every link and joint name prefixed gives the same FK."""
    import re
    with open(urdf_path) as f:
        text = f.read()
    names = set(re.findall(r'<link name="([^"]+)"', text))
    pref = text
    for n in sorted(names, key=len, reverse=True):
        pref = re.sub(rf'(name|link)="{re.escape(n)}"', lambda m, n=n: f'{m.group(1)}="arm_left_{n}"',
                      pref)
    path = str(tmp_path / "prefixed.urdf")
    with open(path, "w") as f:
        f.write(pref)
    plain, prefixed = fk.Urdf(urdf_path), fk.Urdf(path)
    q = np.array([0.3, -0.4, 0.5, 0.2, -0.6, 0.1])
    a = plain.fk("base_link", "link6", q)
    b = prefixed.fk("arm_left_base_link", "arm_left_link6", q)
    assert np.allclose(a, b)
    # via analyze() with --prefix
    rows = make_capture(plain, sample_joints(plain, 8, seed=9))
    csv_path = str(tmp_path / "cap.csv")
    write_csv(csv_path, rows)
    res = fk.analyze(path, csv_path, prefix="arm_left_")
    assert res["base_link"] == "arm_left_base_link" and res["tip_link"] == "arm_left_link6"
    assert res["b_fit"]["rms_mm"] < 0.3


# ---- capture window logic (no ROS needed) ----
def _samples(n=50, jitter_deg=0.0, kin=0):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        q = np.radians([10, 20, -30, 40, 50, -60]) + np.radians(rng.normal(scale=jitter_deg, size=6))
        out.append({"stamp_ns": 1000 + i, "q": list(q), "raw": [300, -45, 410, 12, -170, 33],
                    "kinstate": kin, "mode": "joint"})
    return out


def test_capture_window_accepts_still_pose_and_averages():
    row, reason = fk.evaluate_window(_samples(jitter_deg=0.002))
    assert reason is None
    assert float(row["j1_rad"]) == pytest.approx(math.radians(10), abs=1e-4)
    assert float(row["x"]) == pytest.approx(300.0)
    assert row["mode"] == "joint" and row["n_samples"] == 50


def test_capture_window_rejects_moving_bad_kinstate_and_too_few():
    row, reason = fk.evaluate_window(_samples(jitter_deg=0.1))
    assert row is None and "not still" in reason
    row, reason = fk.evaluate_window(_samples(kin=99))
    assert row is None and "kinstate" in reason
    row, reason = fk.evaluate_window(_samples(n=3))
    assert row is None and "samples" in reason


def test_append_row_writes_header_once(tmp_path):
    row, _ = fk.evaluate_window(_samples())
    path = str(tmp_path / "x.csv")
    fk.append_row(path, row)
    fk.append_row(path, row)
    with open(path) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("stamp_ns,")
    q, raw, _ = fk.read_csv(path)
    assert q.shape == (2, 6) and raw.shape == (2, 6)
