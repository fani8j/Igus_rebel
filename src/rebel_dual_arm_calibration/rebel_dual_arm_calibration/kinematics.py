"""Forward kinematics from a URDF (numpy only, no ROS, no urdf_parser_py).

A minimal parser for the expanded URDF that ``robot_state_publisher`` serves (``/robot_description``):
links, ``fixed`` / ``revolute`` / ``continuous`` joints with origin (xyz, rpy) and axis.  ``Chain`` is the
ordered joint list between two links with a batched forward kinematics ``fk(q)`` (q: (N, n_revolute)),
used by the calibration solver to model joint zero offsets: ``T_base_link6(q + offset)``.

Joint values are given in chain order of the *revolute* joints (``Chain.joint_names``).  The recorder stores
joint states by name; ``ArmKinematics.q_from_joints`` orders them.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .transforms import rpy_to_R


class KinematicsError(ValueError):
    pass


@dataclass
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _floats(text: Optional[str], default) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=float)
    vals = [float(v) for v in text.split()]
    if len(vals) != 3:
        raise KinematicsError(f"expected 3 numbers, got '{text}'")
    return np.array(vals, dtype=float)


class Urdf:
    """Joint tree of a URDF document."""

    def __init__(self, joints: Sequence[UrdfJoint], links: Sequence[str]):
        self.links = list(links)
        self.joints: Dict[str, UrdfJoint] = {j.name: j for j in joints}
        self.parent_joint: Dict[str, UrdfJoint] = {j.child: j for j in joints}

    @staticmethod
    def parse(text) -> "Urdf":
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            raise KinematicsError(f"URDF is not valid XML: {e}") from e
        if root.tag != "robot":
            raise KinematicsError(f"URDF root element is '{root.tag}', expected 'robot'")
        links = [l.get("name") for l in root.findall("link")]
        joints = []
        for j in root.findall("joint"):
            typ = j.get("type", "")
            org = j.find("origin")
            axis = _floats(j.find("axis").get("xyz") if j.find("axis") is not None else None, [1, 0, 0])
            n = float(np.linalg.norm(axis))
            if typ in ("revolute", "continuous"):
                if n < 1e-9:
                    raise KinematicsError(f"joint '{j.get('name')}' has a zero axis")
                axis = axis / n
            joints.append(UrdfJoint(
                name=j.get("name"), type=typ, parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=_floats(org.get("xyz") if org is not None else None, [0, 0, 0]),
                rpy=_floats(org.get("rpy") if org is not None else None, [0, 0, 0]), axis=axis))
        return Urdf(joints, links)

    @staticmethod
    def load(path: str) -> "Urdf":
        with open(path, "rb") as f:
            return Urdf.parse(f.read())

    def has_link(self, name: str) -> bool:
        return name in self.links

    def chain(self, base: str, tip: str) -> "Chain":
        """Joints from ``base`` down to ``tip`` (``tip`` must be a descendant of ``base``)."""
        for name in (base, tip):
            if not self.has_link(name):
                raise KinematicsError(f"link '{name}' not in the URDF")
        path: List[UrdfJoint] = []
        cur = tip
        while cur != base:
            j = self.parent_joint.get(cur)
            if j is None:
                raise KinematicsError(f"'{tip}' is not a descendant of '{base}' in the URDF")
            path.append(j)
            cur = j.parent
        path.reverse()
        for j in path:
            if j.type not in ("fixed", "revolute", "continuous"):
                raise KinematicsError(f"joint '{j.name}' has unsupported type '{j.type}'")
        return Chain(base, tip, path)


def _origin_T(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = xyz
    return T


def _axis_rotations(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """(N,4,4) homogeneous rotations about a unit axis (Rodrigues)."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    outer = np.outer(axis, axis)
    R = (c[:, None, None] * np.eye(3) + s[:, None, None] * K + (1 - c)[:, None, None] * outer)
    T = np.zeros((len(angle), 4, 4))
    T[:, :3, :3] = R
    T[:, 3, 3] = 1.0
    return T


class Chain:
    """Serial chain base -> tip with a batched forward kinematics."""

    def __init__(self, base: str, tip: str, joints: Sequence[UrdfJoint]):
        self.base, self.tip = base, tip
        self.joints = list(joints)
        self.T0 = np.array([_origin_T(j.xyz, j.rpy) for j in self.joints]).reshape(-1, 4, 4)
        self.axes = np.array([j.axis for j in self.joints]).reshape(-1, 3)
        self.revolute = [i for i, j in enumerate(self.joints) if j.type in ("revolute", "continuous")]
        self.joint_names = [self.joints[i].name for i in self.revolute]

    @property
    def n_dof(self) -> int:
        return len(self.revolute)

    def fk(self, q) -> np.ndarray:
        """T_base_tip for joint values q (N, n_dof) or (n_dof,); returns (N,4,4) (or (4,4) for 1-D q)."""
        q = np.asarray(q, dtype=float)
        single = q.ndim == 1
        q = q.reshape(-1, self.n_dof)
        T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        k = 0
        for i in range(len(self.joints)):
            T = T @ self.T0[i]
            if self.joints[i].type in ("revolute", "continuous"):
                T = T @ _axis_rotations(self.axes[i], q[:, k])
                k += 1
        return T[0] if single else T

    def fixed_transform(self) -> np.ndarray:
        """Product of the origins of an all-fixed chain (e.g. link6 -> TCP): T_base_tip, 4x4."""
        if self.revolute:
            raise KinematicsError(f"chain {self.base} -> {self.tip} has revolute joints")
        T = np.eye(4)
        for T0 in self.T0:
            T = T @ T0
        return T

    def with_tail_transform(self, n_prefix: int, X: np.ndarray) -> "Chain":
        """Copy whose fixed joints after the first ``n_prefix`` joints are replaced by the single transform ``X``
        (identity for all but the last tail joint).  Only the numeric origins (``T0``) are patched; the
        ``UrdfJoint`` records keep their original xyz / rpy."""
        if not 0 <= n_prefix < len(self.joints):
            raise KinematicsError("chain has no tail to replace")
        if any(j.type != "fixed" for j in self.joints[n_prefix:]):
            raise KinematicsError("the tail of the chain contains non-fixed joints")
        c = Chain(self.base, self.tip, self.joints)
        c.T0 = self.T0.copy()
        c.T0[n_prefix:] = np.eye(4)
        c.T0[-1] = np.asarray(X, dtype=float).reshape(4, 4)
        return c

    def same_geometry(self, other: "Chain", tol: float = 1e-9, ignore_last: bool = False) -> bool:
        """Same joint types / origins / axes.  ``ignore_last`` skips the origin of the last joint (the
        igus_tool0 -> TCP offset, which is calibrated and may legitimately differ between sessions)."""
        if not (self.T0.shape == other.T0.shape and [j.type for j in self.joints] == [j.type for j in other.joints]
                and np.allclose(self.axes, other.axes, atol=tol)):
            return False
        a, b = (self.T0[:-1], other.T0[:-1]) if ignore_last else (self.T0, other.T0)
        return bool(np.allclose(a, b, atol=tol))


@dataclass
class ArmKinematics:
    """FK of one arm: base -> link6 (``chain6``) and, if the URDF has it, base -> TCP frame (``chain_tcp``)."""
    base_frame: str
    link6_frame: str
    chain6: Chain
    tcp_frame: str = ""
    chain_tcp: Optional[Chain] = None
    chain_link6_tcp: Optional[Chain] = None     # link6 -> TCP (fixed joints only; e.g. link6 -> igus_tool0 -> tip)

    @staticmethod
    def from_urdf(urdf: Urdf, base: str, link6: str, tcp: str = "") -> "ArmKinematics":
        c6 = urdf.chain(base, link6)
        ctcp = None
        c_l6_tcp = None
        if tcp and urdf.has_link(tcp):
            ctcp = urdf.chain(base, tcp)
            if ctcp.joint_names[: c6.n_dof] != c6.joint_names:
                raise KinematicsError(f"TCP chain {tcp} does not pass through the {link6} joints")
            try:
                c = urdf.chain(link6, tcp)
                if not c.revolute and c.joints:
                    c_l6_tcp = c
            except KinematicsError:
                c_l6_tcp = None
        return ArmKinematics(base, link6, c6, tcp if ctcp is not None else "", ctcp, c_l6_tcp)

    def T_link6_tcp(self) -> Optional[np.ndarray]:
        """The link6 -> TCP transform of the URDF, or None if the URDF has no fixed chain link6 -> TCP."""
        return None if self.chain_link6_tcp is None else self.chain_link6_tcp.fixed_transform()

    def tool_split(self) -> Optional[tuple]:
        """(T_link6_tool0, T_tool0_tcp) of the URDF chain link6 -> igus_tool0 -> TCP, or None if the chain does not
        have that shape (>= 2 fixed joints, the last one starting at a link whose name ends with ``igus_tool0``)."""
        c = self.chain_link6_tcp
        if c is None or len(c.joints) < 2 or not c.joints[-1].parent.endswith("igus_tool0"):
            return None
        T1 = np.eye(4)
        for T0 in c.T0[:-1]:
            T1 = T1 @ T0
        return T1, c.T0[-1].copy()

    def with_tcp_transform(self, X_link6_tcp: np.ndarray) -> "ArmKinematics":
        """Copy whose TCP chain (``chain_tcp`` and ``chain_link6_tcp``) uses ``X_link6_tcp`` as link6 -> TCP."""
        if self.chain_tcp is None or self.chain_link6_tcp is None:
            raise KinematicsError(f"arm has no TCP chain (tcp frame '{self.tcp_frame}')")
        n6 = len(self.chain6.joints)
        if [j.name for j in self.chain_tcp.joints[:n6]] != [j.name for j in self.chain6.joints]:
            raise KinematicsError("TCP chain does not start with the link6 chain")
        return ArmKinematics(self.base_frame, self.link6_frame, self.chain6, self.tcp_frame,
                             self.chain_tcp.with_tail_transform(n6, X_link6_tcp),
                             self.chain_link6_tcp.with_tail_transform(0, X_link6_tcp))

    def q_from_joints(self, joints: Dict[str, float]) -> np.ndarray:
        missing = [n for n in self.chain6.joint_names if n not in joints]
        if missing:
            raise KinematicsError(f"joint values missing for {missing}")
        return np.array([joints[n] for n in self.chain6.joint_names], dtype=float)

    def same_geometry(self, other: "ArmKinematics", tol: float = 1e-9, ignore_tcp_offset: bool = False) -> bool:
        """Same kinematics.  ``ignore_tcp_offset`` leaves out the last TCP joint (igus_tool0 -> TCP): the TCP is
        calibrated and may differ between sessions (touches carry their own link6 -> TCP transform)."""
        ok = self.chain6.same_geometry(other.chain6, tol)
        if self.chain_tcp is not None and other.chain_tcp is not None:
            ok = ok and self.chain_tcp.same_geometry(other.chain_tcp, tol, ignore_last=ignore_tcp_offset)
        return ok
