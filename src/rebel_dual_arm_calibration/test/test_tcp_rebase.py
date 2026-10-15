"""TCP robustness of the calibration suite: touches carry link6 -> TCP, ``solve --tcp-file`` re-bases them exactly,
mixed TCPs are refused, the FK of the joint-offset path follows the TCP in use, and the TCP-consistency check only
counts points with tilt diversity (the pivot observability measure).

The re-basing tests use a small hand-written 6-joint URDF (fast, no real ReBeL URDF needed); one CLI test uses the
real dual-arm URDF and is skipped without it."""
import json
import os

import numpy as np
import pytest
import yaml

from rebel_dual_arm_calibration import base_check, session, solve_cli, solver
from rebel_dual_arm_calibration import synthetic as S
from rebel_dual_arm_calibration.base_check import GateOptions, tcp_consistency, tilt_observability
from rebel_dual_arm_calibration.kinematics import ArmKinematics, Urdf
from rebel_dual_arm_calibration.solver import SolverError, Touch
from rebel_dual_arm_calibration.transforms import inv_T, rpy_to_R, xyz_rpy_to_T

R_OLD = np.array([0.005937, 0.094121, 0.130699])       # igus_tool0 -> xeg32_tool_tip, as in the wrapper
R_NEW = np.array([0.0112, 0.0891, 0.1368])

URDF_TMPL = """<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link"/><link name="link1"/><link name="link2"/><link name="link3"/><link name="link4"/>
  <link name="link5"/><link name="link6"/><link name="igus_tool0"/><link name="xeg32_tool_tip"/>
  <joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="joint2" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="0.02 0 0.1" rpy="0 0.1 0"/><axis xyz="0 1 0"/></joint>
  <joint name="joint3" type="revolute"><parent link="link2"/><child link="link3"/><origin xyz="0 0 0.3" rpy="0 0 0"/><axis xyz="0 1 0"/></joint>
  <joint name="joint4" type="revolute"><parent link="link3"/><child link="link4"/><origin xyz="0 0 0.25" rpy="0 0 0.05"/><axis xyz="0 0 1"/></joint>
  <joint name="joint5" type="revolute"><parent link="link4"/><child link="link5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/></joint>
  <joint name="joint6" type="revolute"><parent link="link5"/><child link="link6"/><origin xyz="0 0 0.08" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="link6_to_igus_tool0" type="fixed"><parent link="link6"/><child link="igus_tool0"/><origin xyz="0 0 0.0527" rpy="0.02 -0.01 3.141592653589793"/></joint>
  <joint name="igus_tool0_to_xeg32_tool_tip" type="fixed"><parent link="igus_tool0"/><child link="xeg32_tool_tip"/><origin xyz="%s" rpy="0.1 -0.2 0.3"/></joint>
</robot>
"""


def _urdf(r=R_OLD):
    return URDF_TMPL % " ".join(repr(float(v)) for v in r)


def _kin(r=R_OLD):
    return ArmKinematics.from_urdf(Urdf.parse(_urdf(r)), "base_link", "link6", "xeg32_tool_tip")


def _touches(kin, n_points=4, per_point=3, seed=0, with_stored=False, pid0=1, arm="left"):
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n_points):
        for _ in range(per_point):
            q = rng.uniform(-1.2, 1.2, 6)
            T = kin.chain_tcp.fk(q)
            T6 = kin.chain6.fk(q)
            out.append(Touch(arm=arm, point_id=pid0 + p, p_board=[0.01 * p, 0.0, 0.0], T_base_tcp=T,
                             joints={n: float(v) for n, v in zip(kin.chain6.joint_names, q)}, tcp_frame="xeg32_tool_tip",
                             T_base_link6=T6 if with_stored else None,
                             T_link6_tcp=np.linalg.inv(T6) @ T if with_stored else None))
    return out


def _session(path, touches, urdf_text=None, arm="left"):
    d = str(path)
    session.init_session(d, {"parameters": {"active_arms": [arm], f"{arm}_base_frame": "base_link",
                                            f"{arm}_tip_frame": "link6", f"{arm}_tcp_frame": "xeg32_tool_tip"}})
    for t in touches:
        session.append_touch(d, t, {})
    if urdf_text is not None:
        session.save_urdf_once(d, urdf_text)
    return d


def _tcp_yaml(path, r=R_NEW, arm="left", verdict="ok", **extra):
    d = {"arm": arm, "verdict": verdict, "r_tool0_m": [float(v) for v in r], **extra}
    with open(str(path), "w") as f:
        yaml.safe_dump(d, f)
    return str(path)


def _rebase(dirs, tcp_path, arm="left"):
    data = session.load_sessions_full(dirs)
    files = {arm: solver.load_tcp_file(tcp_path, arm)} if tcp_path else {}
    rep, kin = solver.prepare_touch_tcp(data.touches, data.kin, files)
    return data, rep, kin


# ---- recording format --------------------------------------------------------------------------------------
def test_touch_json_new_keys_are_additive():
    kin = _kin()
    t = _touches(kin, 1, 1, with_stored=True)[0]
    d = t.to_json_dict()
    assert "T_base_link6" in d and "T_link6_tcp" in d
    back = Touch.from_json_dict(json.loads(json.dumps(d)))
    assert np.allclose(back.T_link6_tcp, t.T_link6_tcp) and np.allclose(back.T_base_link6, t.T_base_link6)
    assert back.tcp_source == "recorded"
    legacy = _touches(kin, 1, 1, with_stored=False)[0].to_json_dict()
    assert "T_base_link6" not in legacy and "T_link6_tcp" not in legacy
    old = Touch.from_json_dict(legacy)
    assert old.T_link6_tcp is None and old.T_base_link6 is None and old.tcp_source == ""


# ---- re-basing: exactness ------------------------------------------------------------------------------------
@pytest.mark.parametrize("stored", [False, True])
def test_rebased_touches_equal_touches_recorded_directly_with_the_new_tcp(tmp_path, stored):
    kin_old, kin_new = _kin(R_OLD), _kin(R_NEW)
    old = _touches(kin_old, with_stored=stored, seed=5)
    direct = _touches(kin_new, with_stored=stored, seed=5)         # same joints, recorded with the new TCP
    d = _session(tmp_path / "s", old, _urdf(R_OLD))
    data, rep, kin_used = _rebase([d], _tcp_yaml(tmp_path / "tcp.yaml"))
    assert len(data.touches) == len(direct)
    for t, ref in zip(data.touches, direct):
        assert np.max(np.abs(t.T_base_tcp - ref.T_base_tcp)) < 1e-9
        X_new = np.linalg.inv(kin_new.chain6.fk(t.q)) @ ref.T_base_tcp
        assert np.max(np.abs(t.T_link6_tcp - X_new)) < 1e-9
        assert t.tcp_source == "rebased"
    e = rep["left"]
    assert e["rebased"] and e["n_touches"] == len(old) and e["verdict"] == "ok"
    assert np.allclose(e["r_tool0_tip_m"], R_OLD, atol=1e-9) and np.allclose(e["r_tool0_tip_m_new"], R_NEW)
    # the shift of a touch is |R_base_tool0 (r_new - r_old)| = |r_new - r_old| for a pure rotation
    assert abs(e["shift_mm"]["mean"] - np.linalg.norm(R_NEW - R_OLD) * 1e3) < 1e-6
    assert len(e["shift_mm"]["per_touch"]) == len(old)
    # the kept parts: link6 -> igus_tool0 and every rotation
    T1, T2 = kin_old.tool_split()
    assert np.allclose(np.linalg.inv(T1) @ data.touches[0].T_link6_tcp, np.block(
        [[T2[:3, :3], R_NEW[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]), atol=1e-12)
    # the input file is untouched on disk
    assert session.load_touches(d)[0].T_link6_tcp is None or stored


def test_fk_of_the_offset_path_uses_the_new_tool_transform(tmp_path):
    kin_old = _kin(R_OLD)
    d = _session(tmp_path / "s", _touches(kin_old, seed=2), _urdf(R_OLD))
    data, rep, kin_used = _rebase([d], _tcp_yaml(tmp_path / "tcp.yaml"))
    assert rep["left"]["fk_tcp_patched"]
    q = np.array([t.q for t in data.touches])
    Trec = np.array([t.T_base_tcp for t in data.touches])
    # patched FK reproduces the re-based poses: A = T_rec inv(FK) is the identity (exactly consistent data)
    blk = solver._TouchBlock(data.touches, kin_used, True)
    assert np.max(np.abs(blk.A["left"] - np.eye(4))) < 1e-9
    assert np.max(np.abs(kin_used["left"].chain_tcp.fk(q) - Trec)) < 1e-9
    assert np.allclose(blk.tip_base({"left": np.zeros(6)}), Trec[:, :3, 3], atol=1e-9)
    # with the UNPATCHED chain the anchor A would absorb a joint-dependent TCP change: that is the bug being fixed
    blk_bad = solver._TouchBlock(data.touches, data.kin, True)
    assert np.max(np.abs(blk_bad.A["left"] - np.eye(4))) > 1e-3
    # a joint offset moves the tip by the NEW lever arm: FK(q + dq) differs from FK(q) in the new chain
    off = np.zeros(6)
    off[4] = 0.01
    moved = blk.tip_base({"left": off}) - blk.tip_base({"left": np.zeros(6)})
    ref = kin_used["left"].chain_tcp.fk(q + off)[:, :3, 3] - kin_used["left"].chain_tcp.fk(q)[:, :3, 3]
    assert np.allclose(moved, ref, atol=1e-12)
    # the original kinematics object is not mutated
    assert np.allclose(data.kin["left"].T_link6_tcp(), kin_old.T_link6_tcp())


def test_without_tcp_file_nothing_changes_and_the_tcp_is_reported(tmp_path):
    kin = _kin()
    ts = _touches(kin, seed=3)
    d = _session(tmp_path / "s", ts, _urdf())
    data, rep, kin_used = _rebase([d], None)
    for t, ref in zip(data.touches, ts):
        assert np.array_equal(t.T_base_tcp, ref.T_base_tcp) and t.tcp_source == "session_urdf"
    e = rep["left"]
    assert not e["rebased"] and np.allclose(e["r_tool0_tip_m"], R_OLD, atol=1e-9) and "fk_tcp_patched" not in e
    assert kin_used["left"] is data.kin["left"]
    text = "\n".join(solver.format_tcp_report(rep))
    assert "TCP left: as recorded" in text and "5.937" in text and "94.121" in text


# ---- refusals ----------------------------------------------------------------------------------------------------
def test_refusal_when_the_old_tcp_is_unknown(tmp_path):
    ts = _touches(_kin(), seed=4)
    d = _session(tmp_path / "nourdf", ts)                      # no URDF, no stored link6 -> TCP
    data = session.load_sessions_full([d])
    assert all(t.T_link6_tcp is None for t in data.touches)
    solver.prepare_touch_tcp(data.touches, data.kin, {})       # without --tcp-file: fine (nothing to check)
    with pytest.raises(SolverError, match="neither stored in touches.jsonl nor available"):
        solver.prepare_touch_tcp(data.touches, data.kin, {"left": solver.load_tcp_file(_tcp_yaml(tmp_path / "t.yaml"), "left")})
    # the stored transform is enough to know X_old, but the igus_tool0 split needs the URDF chain
    stored = _touches(_kin(), seed=4, with_stored=True)
    d2 = _session(tmp_path / "stored_nourdf", stored)
    data2 = session.load_sessions_full([d2])
    with pytest.raises(SolverError, match="cannot re-base"):
        solver.prepare_touch_tcp(data2.touches, data2.kin, {"left": solver.load_tcp_file(str(tmp_path / "t.yaml"), "left")})
    # the CLI reports it with exit 2
    assert solve_cli.main([d, "--out", str(tmp_path / "o.yaml"), "--tcp-file", f"left={tmp_path / 't.yaml'}"]) == 2


def test_mixed_tcps_of_one_arm_are_refused_unless_rebased(tmp_path, capsys):
    k_old, k_new = _kin(R_OLD), _kin(R_NEW)
    s1 = _session(tmp_path / "s1", _touches(k_old, 2, 3, seed=1, pid0=1), _urdf(R_OLD))
    s2 = _session(tmp_path / "s2", _touches(k_new, 2, 3, seed=2, pid0=3), _urdf(R_NEW))
    data = session.load_sessions_full([s1, s2])        # the sessions merge: only the TCP offset differs
    with pytest.raises(SolverError, match="recorded with different TCPs"):
        solver.prepare_touch_tcp(data.touches, data.kin, {})
    assert solve_cli.main([s1, s2, "--out", str(tmp_path / "o.yaml")]) == 2
    err = capsys.readouterr().err
    assert "different TCPs" in err and "--tcp-file left=" in err
    # re-basing ALL touches of the arm onto one TCP resolves the mix; every touch ends up with the same X_new
    tcp = _tcp_yaml(tmp_path / "tcp.yaml", R_NEW)
    rep, kin_used = solver.prepare_touch_tcp(data.touches, data.kin, {"left": solver.load_tcp_file(tcp, "left")})
    Xs = [t.T_link6_tcp for t in data.touches]
    assert max(np.max(np.abs(X - Xs[0])) for X in Xs) < 1e-12
    shifts = [p["shift_mm"] for p in rep["left"]["shift_mm"]["per_touch"]]
    assert max(shifts[6:]) < 1e-6 and min(shifts[:6]) > 1.0          # session 2 was already recorded with R_NEW
    for t in data.touches:                                           # and all agree with direct recording
        assert np.max(np.abs(t.T_base_tcp - k_new.chain_tcp.fk(t.q))) < 1e-9
    # the arm's chain geometry (joints) must still match between sessions
    other = _urdf(R_OLD).replace('xyz="0.02 0 0.1"', 'xyz="0.03 0 0.1"')
    s3 = _session(tmp_path / "s3", _touches(k_old, 1, 3, seed=9, pid0=7), other)
    with pytest.raises(session.SessionMismatchError, match="URDF kinematics"):
        session.load_sessions_full([s1, s3])


def test_urdf_saved_before_a_tcp_change_follows_the_stored_tcp(tmp_path):
    k_old, k_new = _kin(R_OLD), _kin(R_NEW)
    # touches recorded after the TCP change (stored link6 -> TCP = new), but the session URDF was saved before it
    d = _session(tmp_path / "s", _touches(k_new, seed=6, with_stored=True), _urdf(R_OLD))
    data, rep, kin_used = _rebase([d], None)
    e = rep["left"]
    assert e["fk_tcp_patched"] and any("differs from the session URDF" in w for w in e["warnings"])
    assert np.allclose(e["r_tool0_tip_m"], R_NEW, atol=1e-9)
    assert np.max(np.abs(solver._TouchBlock(data.touches, kin_used, True).A["left"] - np.eye(4))) < 1e-9


def test_tcp_file_validation(tmp_path):
    ok = _tcp_yaml(tmp_path / "ok.yaml")
    f = solver.load_tcp_file(ok, "left")
    assert np.allclose(f["r_tool0_m"], R_NEW) and f["verdict"] == "ok"
    bad = {
        "len": dict(r_tool0_m=[0.1, 0.2]), "nan": dict(r_tool0_m=[0.1, float("nan"), 0.0]),
        "mm": dict(r_tool0_m=[10.0, 90.0, 130.0]), "arm": dict(arm="right"), "verdict": dict(verdict="FAILED"),
        "str": dict(r_tool0_m="abc"),
    }
    for name, over in bad.items():
        d = {"arm": "left", "verdict": "ok", "r_tool0_m": [0.01, 0.09, 0.13]}
        d.update(over)
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(SolverError, match="--tcp-file left"):
            solver.load_tcp_file(str(p), "left")
    (tmp_path / "list.yaml").write_text("- 1\n")
    with pytest.raises(SolverError):
        solver.load_tcp_file(str(tmp_path / "list.yaml"), "left")
    with pytest.raises(SolverError):
        solver.load_tcp_file(str(tmp_path / "missing.yaml"), "left")
    # extra fields are ignored; the file may have no arm / verdict keys
    (tmp_path / "min.yaml").write_text("r_tool0_m: [0.0, 0.1, 0.1]\nsomething: [1, 2]\n")
    assert solver.load_tcp_file(str(tmp_path / "min.yaml"), "right")["verdict"] is None
    assert solver.parse_tcp_file_args(["left=a.yaml,right=b.yaml"]) == {"left": "a.yaml", "right": "b.yaml"}
    assert solver.parse_tcp_file_args(["left=a.yaml", "right=/x,y/b.yaml"]) == {"left": "a.yaml", "right": "/x,y/b.yaml"}
    for spec in ("a.yaml", "middle=a.yaml", "left="):
        with pytest.raises(SolverError):
            solver.parse_tcp_file_args([spec])
    with pytest.raises(SolverError):
        solver.parse_tcp_file_args(["left=a.yaml", "left=b.yaml"])


# ---- CLI end to end (real dual-arm URDF) -----------------------------------------------------------------------
@pytest.mark.skipif(S.rebel_urdf_path() is None, reason="igus_rebel2.urdf not found")
def test_cli_rebase_matches_a_session_recorded_with_the_new_tcp(tmp_path, capsys):
    from test_loop_closure import PRE, World, _write_session
    w = World(0.0, seed=3)
    old_xyz = "0.005937 0.094121 0.130699"
    new_xyz = " ".join(repr(float(v)) for v in R_NEW)
    urdf_old = S.dual_urdf_text(PRE)
    urdf_new = urdf_old.replace(f'xyz="{old_xyz}"', f'xyz="{new_xyz}"')
    assert urdf_new != urdf_old
    kin_new = {a: ArmKinematics.from_urdf(Urdf.parse(urdf_new), f"{p}base_link", f"{p}link6", f"{p}xeg32_tool_tip")
               for a, p in zip(("left", "right"), PRE)}
    # noise-free touches: the same joints, once recorded with the old and once with the new TCP
    w.touches = S.simulate_touches(w.scene, w.table, w.kin, noise_m=0.0, seed=33)
    direct = []
    for t in w.touches:
        d = Touch.from_json_dict(t.to_json_dict())
        d.T_base_tcp = kin_new[t.arm].chain_tcp.fk(kin_new[t.arm].q_from_joints(t.joints))
        direct.append(d)
    sold = _write_session(tmp_path / "old", w, None, w.tables, urdf_text=urdf_old)
    saved = w.touches
    w.touches = direct
    snew = _write_session(tmp_path / "new", w, None, w.tables, urdf_text=urdf_new)
    w.touches = saved
    tcps = ",".join(f"{a}={_tcp_yaml(tmp_path / (a + '.yaml'), R_NEW, a)}" for a in ("left", "right"))
    out_a, out_b = str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")
    common = ["--joint-offsets", "--no-validate"]
    assert solve_cli.main([sold, "--out", out_a, "--tcp-file", tcps] + common) == 0
    text = capsys.readouterr().out
    assert "TCP left: RE-BASED with" in text and "TCP right: RE-BASED with" in text and "touches shifted by" in text
    assert solve_cli.main([snew, "--out", out_b] + common) == 0
    text_b = capsys.readouterr().out
    assert "TCP left: as recorded" in text_b and "TCP left: RE-BASED" not in text_b
    ya, yb = (yaml.safe_load(open(p))["dual_arm_calibration"] for p in (out_a, out_b))
    assert np.allclose(ya["arm_right_base"]["xyz"], yb["arm_right_base"]["xyz"], atol=1e-6)
    assert np.allclose(ya["arm_right_base"]["rpy"], yb["arm_right_base"]["rpy"], atol=1e-6)
    ra = json.load(open(str(tmp_path / "a_report.json")))
    assert ra["tcp"]["left"]["rebased"] and ra["tcp"]["left"]["shift_mm"]["max"] > 1.0
    assert ra["tcp"]["left"]["verdict"] == "ok" and ra["tcp"]["right"]["n_touches"] == 4
    assert "re-based onto TCP" in ya["notes"]
    rb = json.load(open(str(tmp_path / "b_report.json")))
    assert not rb["tcp"]["left"]["rebased"]


# ---- TCP-consistency: tilt diversity ---------------------------------------------------------------------------------
def _rot_axis(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, dtype=float)).as_matrix()


def _point_touches(rotations, r_err, c=(0.4, 0.1, 0.02), arm="left", pid=1):
    """Touches of ONE point: p_i = c + R_i r_err is the recorded tip (r_err: TCP error in the tool frame)."""
    out = []
    for R in rotations:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(c) + R @ np.asarray(r_err)
        out.append(Touch(arm=arm, point_id=pid, p_board=np.zeros(3), T_base_tcp=T))
    return out


BASE = _rot_axis([1, 0, 0], 180.0)          # tool pointing down


def test_pure_roll_is_excluded_even_though_the_total_angle_is_large():
    r_err = [0.0, 0.0, 0.006]                                    # 6 mm TCP error along the tool z axis ...
    roll = [BASE @ _rot_axis([0, 0, 1], a) for a in (0, 40, 80, 120)]       # ... invisible for a roll about z
    ts = _point_touches(roll, r_err)
    assert max(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))) for a in roll for b in roll) > 100
    obs = tilt_observability(roll)
    assert obs["sigma_min_norm"] < 1e-9 and obs["rank"] == 2
    assert np.allclose(np.abs(obs["weakest_direction_tool"]), [0, 0, 1], atol=1e-9)
    c = tcp_consistency(ts)
    a = c["arms"]["left"]
    assert a["n_points"] == 0 and a["spread_rms_mm"] is None and not a["skipped_similar_orientation"]
    assert len(a["insufficient_tilt"]) == 1
    ins = a["insufficient_tilt"][0]
    assert ins["point"] == 1 and ins["message"] == "insufficient tilt diversity for a TCP check at point 1"
    assert np.allclose(np.abs(ins["weakest_direction_tool"]), [0, 0, 1], atol=1e-9)
    assert any("insufficient tilt diversity for a TCP check at point 1" in w for w in c["warnings"])
    # the old criterion would have reported a spread of ~0 mm and called the TCP fine
    P = np.array([t.T_base_tcp[:3, 3] for t in ts])
    assert np.max(np.linalg.norm(P - P.mean(axis=0), axis=1)) < 1e-9
    # a roll about the tool x axis hides an x error, not a z error
    rollx = [BASE @ _rot_axis([1, 0, 0], a) for a in (0, 40, 80, 120)]
    ox = tilt_observability(rollx)
    assert np.allclose(np.abs(ox["weakest_direction_tool"]), [1, 0, 0], atol=1e-9)
    assert tcp_consistency(_point_touches(rollx, [0.006, 0, 0]))["arms"]["left"]["n_points"] == 0


def test_two_axis_tilts_are_included_and_see_the_tcp_error():
    tilts = [BASE, BASE @ _rot_axis([1, 0, 0], 40), BASE @ _rot_axis([0, 1, 0], 40),
             BASE @ _rot_axis([1, 1, 0], -40), BASE @ _rot_axis([0, 1, 0], -35)]
    good = tcp_consistency(_point_touches(tilts, [0.0, 0.0, 0.0]))["arms"]["left"]
    assert good["n_points"] == 1 and good["spread_rms_mm"] < 1e-9 and not good["insufficient_tilt"]
    assert good["weakest_sigma_min_norm"] >= 0.2 and good["weakest_point"] == 1
    bad = tcp_consistency(_point_touches(tilts, [0.0, 0.0, 0.006]))["arms"]["left"]
    assert bad["n_points"] == 1 and bad["spread_rms_mm"] > 1.5
    assert len(bad["per_point"][0]["weakest_direction_tool"]) == 3
    assert tcp_consistency(_point_touches(tilts, [0.0, 0.0, 0.006]))["warnings"]


def test_two_touches_and_small_threshold_behaviour():
    # two touches never see the rotation axis between them; a point mixes with others: only qualifying points count
    two = [BASE, BASE @ _rot_axis([1, 0, 0], 60)]
    c = tcp_consistency(_point_touches(two, [0, 0, 0.006]))["arms"]["left"]
    assert c["n_points"] == 0 and c["insufficient_tilt"][0]["rank"] == 2
    tilts = [BASE, BASE @ _rot_axis([1, 0, 0], 40), BASE @ _rot_axis([0, 1, 0], 40), BASE @ _rot_axis([1, 1, 0], -40)]
    both = (_point_touches(tilts, [0, 0, 0.004], pid=1) + _point_touches([BASE @ _rot_axis([0, 0, 1], a) for a in (0, 50, 100)],
                                                                         [0, 0, 0.004], pid=2))
    c = tcp_consistency(both)["arms"]["left"]
    assert c["n_points"] == 1 and c["per_point"][0]["point"] == 1 and c["insufficient_tilt"][0]["point"] == 2
    # the threshold is an option
    lax = tcp_consistency(both, GateOptions(min_tilt_sigma=0.0))["arms"]["left"]
    assert lax["n_points"] == 1 and lax["insufficient_tilt"][0]["point"] == 2     # rank 3 is still required

