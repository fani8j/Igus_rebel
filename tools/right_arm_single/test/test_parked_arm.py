import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parked_arm as pa  # noqa: E402
import right_snapshot as rs  # noqa: E402

URDF = """<robot name="t">
 <link name="a"><collision><origin xyz="0 0 0.5"/><geometry><box size="0.2 0.4 1.0"/></geometry></collision></link>
 <link name="b"><collision><origin xyz="0 0 0" rpy="1.5707963267948966 0 0"/><geometry><cylinder radius="0.1" length="0.6"/></geometry></collision></link>
 <link name="c"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
 <joint name="j1" type="revolute"><parent link="a"/><child link="b"/><origin xyz="1 0 0"/></joint>
 <joint name="j2" type="fixed"><parent link="b"/><child link="c"/><origin xyz="0 0 2"/></joint>
</robot>"""


def test_aabb_primitives():
    low, high, n = pa.collision_aabb(URDF, lambda f: f)
    assert n == 3
    # box x [-0.1,0.1] y [-0.2,0.2] z [0,1]; cylinder at x=1, axis along -y: x [0.9,1.1], y +-0.3, z +-0.1
    # sphere at (1, 0, 2) (only the collision origin of b is rotated, not the link frame)
    assert np.allclose(low, [-0.1, -0.3, -0.1])
    assert np.allclose(high, [1.1, 0.3, 2.05])


def test_stl_binary_and_ascii(tmp_path):
    import struct
    tri = [(0, 0, 0), (1, 0, 0), (0, 2, 3)]
    binary = b"\0" * 80 + struct.pack("<I", 1) + struct.pack("<12fH", 0, 0, 1, *[v for p in tri for v in p], 0)
    (tmp_path / "b.stl").write_bytes(binary)
    ascii_ = "solid x\nfacet normal 0 0 1\nouter loop\n" + "".join(f"vertex {p[0]} {p[1]} {p[2]}\n" for p in tri) + \
        "endloop\nendfacet\nendsolid\n"
    (tmp_path / "a.stl").write_text(ascii_)
    for name in ("a.stl", "b.stl"):
        v = pa.read_stl_vertices(tmp_path / name)
        assert v.shape == (3, 3) and np.allclose(v.max(axis=0), [1, 2, 3])


def test_parked_box_frame():
    cal = rs.load_calibration(rs.DEFAULT_CALIBRATION)
    dims, pos, quat = pa.parked_box([-0.1, -0.1, 0.0], [0.1, 0.1, 1.0], 0.05, cal)
    assert np.allclose(dims, [0.3, 0.3, 1.1])
    # centre (0,0,0.5) in the left frame -> B^-1 * centre
    b = rs.right_base_pose(cal)
    back = b[:3, :3] @ pos + b[:3, 3]
    assert np.allclose(back, [0, 0, 0.5], atol=1e-12)
