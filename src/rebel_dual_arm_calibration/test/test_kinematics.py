"""FK from the URDF: independent check against PyKDL, and consistency of the synthetic recorded poses."""
import numpy as np
import pytest

from rebel_dual_arm_calibration import synthetic as S
from rebel_dual_arm_calibration.kinematics import ArmKinematics, KinematicsError, Urdf

pytestmark = pytest.mark.skipif(S.rebel_urdf_path() is None, reason="igus_rebel2.urdf not found")


def _kdl_chain(urdf_text, base, tip):
    """Independent FK: PyKDL chain built from urdf_parser_py (kdl_parser convention)."""
    import PyKDL as kdl
    from urdf_parser_py.urdf import URDF
    robot = URDF.from_xml_string(urdf_text.encode("utf-8"))
    child_to_joint = {j.child: j for j in robot.joints}
    path, cur = [], tip
    while cur != base:
        path.append(child_to_joint[cur])
        cur = child_to_joint[cur].parent
    path.reverse()
    chain = kdl.Chain()
    names = []
    for j in path:
        o = j.origin
        xyz, rpy = (o.xyz if o else [0, 0, 0]), (o.rpy if o else [0, 0, 0])
        F = kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))
        if j.type == "fixed":
            chain.addSegment(kdl.Segment(j.child, kdl.Joint(kdl.Joint.Fixed), F))
        else:
            ax = F.M * kdl.Vector(*j.axis)
            chain.addSegment(kdl.Segment(j.child, kdl.Joint(j.name, F.p, ax, kdl.Joint.RotAxis), F))
            names.append(j.name)
    return chain, names


def _kdl_fk(chain, q):
    import PyKDL as kdl
    fk = kdl.ChainFkSolverPos_recursive(chain)
    qq = kdl.JntArray(len(q))
    for i, v in enumerate(q):
        qq[i] = float(v)
    f = kdl.Frame()
    assert fk.JntToCart(qq, f) >= 0
    T = np.eye(4)
    for i in range(3):
        for j in range(3):
            T[i, j] = f.M[i, j]
        T[i, 3] = f.p[i]
    return T


def test_fk_matches_kdl_to_1e9():
    pytest.importorskip("PyKDL")
    pytest.importorskip("urdf_parser_py")
    text = S.rebel_urdf_text(True)
    kin = ArmKinematics.from_urdf(Urdf.parse(text), "base_link", "link6", "xeg32_tool_tip")
    rng = np.random.default_rng(0)
    for base, tip, ch in (("base_link", "link6", kin.chain6), ("base_link", "xeg32_tool_tip", kin.chain_tcp)):
        kc, names = _kdl_chain(text, base, tip)
        assert names == ch.joint_names
        q = rng.uniform(-2.5, 2.5, (40, 6))
        T = ch.fk(q)
        for k in range(len(q)):
            assert np.max(np.abs(T[k] - _kdl_fk(kc, q[k]))) < 1e-9


def test_recorded_link6_pose_equals_fk_of_recorded_joints():
    """Synthetic samples built from the real URDF: T_base_link6 == FK(joints) to 1e-9, for both arms."""
    from rebel_dual_arm_calibration.targets import load_targets_all, default_targets_path
    arms, table = load_targets_all(default_targets_path())
    kin = S.default_kinematics()
    sc = S.make_kin_scene(seed=0, offset_deg=0.4)
    ss = S.simulate_kin_samples(sc, arms, kin, n_per_arm=8, seed=3)
    touches = S.simulate_touches(sc, table, kin, seed=4)
    for s in ss:
        q = kin[s.arm].q_from_joints(s.joints)
        assert np.max(np.abs(kin[s.arm].chain6.fk(q) - s.T_base_link6)) < 1e-9
    for t in touches:
        q = kin[t.arm].q_from_joints(t.joints)
        noise = np.max(np.abs(kin[t.arm].chain_tcp.fk(q)[:3, 3] - t.T_base_tcp[:3, 3]))
        assert noise < 2e-3          # the recorded tip carries the 0.3 mm touch noise
        assert np.max(np.abs(kin[t.arm].chain_tcp.fk(q)[:3, :3] - t.T_base_tcp[:3, :3])) < 1e-9


def test_prefixed_dual_arm_names_give_same_fk():
    k0 = S.default_kinematics()["left"]
    k1 = S.default_kinematics(("arm_left_", "arm_right_"))["right"]
    assert k1.chain6.joint_names == [f"arm_right_joint{i}" for i in range(1, 7)]
    q = np.random.default_rng(1).uniform(-2, 2, (5, 6))
    assert np.allclose(k0.chain6.fk(q), k1.chain6.fk(q), atol=1e-12)
    assert k1.tcp_frame == "arm_right_xeg32_tool_tip" and k1.same_geometry(k0)
    with pytest.raises(KinematicsError):
        k1.q_from_joints({"joint1": 0.0})


def test_parser_errors():
    with pytest.raises(KinematicsError):
        Urdf.parse("<not xml")
    with pytest.raises(KinematicsError):
        Urdf.parse("<foo/>")
    u = Urdf.parse(S.rebel_urdf_text(False))
    with pytest.raises(KinematicsError):
        u.chain("link6", "base_link")          # not a descendant
    with pytest.raises(KinematicsError):
        u.chain("base_link", "nope")
