import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from box_seam_detection.ros import cloud
from box_seam_detection.ros.fake_camera import load_capture, pcd_to_depth, read_pcd_xyzrgb

CAPTURES = Path("/home/farhan/Igus_rebel/outputs/rgbd_captures/captures")


def _saved_manifest():
    for m in sorted(CAPTURES.glob("frame_000001_20261008_*.json"), reverse=True) if CAPTURES.is_dir() else []:
        d = json.loads(m.read_text())
        if Path(d["pointcloud"]["path"]).exists() and Path(d["image"]["path"]).exists():
            return m
    return None


def _msg(arr, enc, frame="f"):
    h, w = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1
    step = w * ch * arr.itemsize
    hd = SimpleNamespace(stamp=SimpleNamespace(sec=5, nanosec=6), frame_id=frame)
    return SimpleNamespace(height=h, width=w, encoding=enc, is_bigendian=0, step=step, data=arr.tobytes(), header=hd)


def test_image_decode_encodings():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert (cloud.rgb_from_color_msg(_msg(rgb, "rgb8")) == rgb).all()
    assert (cloud.rgb_from_color_msg(_msg(rgb, "bgr8")) == rgb[:, :, ::-1]).all()
    d16 = np.array([[0, 1000, 65535]], dtype="<u2")
    assert (cloud.image_array(_msg(d16, "16UC1")) == d16).all()
    d32 = np.array([[0.5, np.nan]], dtype="<f4")
    out = cloud.image_array(_msg(d32, "32FC1"))
    assert out[0, 0] == 0.5 and np.isnan(out[0, 1])
    with pytest.raises(cloud.CaptureError):
        cloud.image_array(_msg(rgb, "yuv422"))
    with pytest.raises(cloud.CaptureError):
        cloud.rgb_from_color_msg(_msg(d16, "16UC1"))


def test_backproject_synthetic_formula_and_order():
    depth = np.zeros((3, 4), dtype=np.uint16)
    depth[0, 2], depth[2, 1] = 1000, 2000
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[0, 2] = (1, 2, 3)
    k = [100.0, 0, 1.0, 0, 50.0, 2.0, 0, 0, 1]
    p = cloud.backproject_depth(depth, rgb, k, 0.001)
    assert p.shape == (2, 4) and p.dtype == np.float32
    np.testing.assert_allclose(p[0, :3], [(2 - 1) * 1.0 / 100, (0 - 2) * 1.0 / 50, 1.0], rtol=1e-6)
    np.testing.assert_allclose(p[1, :3], [0.0, 0.0, 2.0], atol=1e-7)  # row-major: (0,2) first
    assert p[0, 3:].view(np.uint32)[0] == (1 << 16 | 2 << 8 | 3)
    # float encodings: NaN / 0 / negative are invalid
    df = np.array([[np.nan, 0.0, -1.0, 0.5]], dtype=np.float32)
    q = cloud.backproject_depth(df, np.zeros((1, 4, 3), np.uint8), k, 1.0)
    assert q.shape[0] == 1 and q[0, 2] == np.float32(0.5)


def test_depth_roundtrip_matches_saved_capture():
    mpath = _saved_manifest()
    if mpath is None:
        pytest.skip("no saved capture with pcd + png on this machine")
    rgb, depth, m = load_capture(mpath)
    k = m["camera_info"]["k"]
    xyz0, rgb0 = read_pcd_xyzrgb(m["pointcloud"]["path"])
    pts = cloud.backproject_depth(depth, rgb, k, 0.001)
    # every original point lies on its own pixel: same count and row-major order
    assert pts.shape[0] == xyz0.shape[0]
    # position error: depth quantisation (0.5 mm) propagated to x/y (|x/z| < ~1) plus float32
    err = np.abs(pts[:, :3] - xyz0)
    assert err[:, 2].max() < 6e-4, err[:, 2].max()
    assert err.max() < 1.2e-3, err.max()
    # colours: the PNG pixel packed exactly like the driver cloud's rgb
    assert (pts[:, 3].view(np.uint32) == rgb0 & 0x00FFFFFF).mean() > 0.999


def test_manifest_schema_equals_saved_capture(tmp_path):
    mpath = _saved_manifest()
    if mpath is None:
        pytest.skip("no saved capture")
    saved = json.loads(mpath.read_text())
    ci = saved["camera_info"]
    hd = lambda n: SimpleNamespace(stamp=SimpleNamespace(sec=n // 10**9, nanosec=n % 10**9), frame_id="camera_color_optical_frame")  # noqa: E731
    color = SimpleNamespace(width=ci["width"], height=ci["height"], encoding="rgb8", header=hd(10**18))
    depth = SimpleNamespace(encoding="16UC1", header=hd(10**18 + 1000))
    info = SimpleNamespace(distortion_model=ci["distortion_model"], d=ci["d"], r=ci["r"], p=ci["p"], header=hd(10**18))
    topics = {"rgb": "/camera/color/image_raw", "depth": "/camera/depth/image_raw"}
    ext = {"parent_frame": "camera_link", "child_frame": "camera_color_optical_frame"}
    man = cloud.build_manifest("cid", "/x.png", "/x.pcd", color, depth, info, ci["k"], 123, topics, ext)
    assert man["schema"] == saved["schema"]
    assert set(saved) <= set(man) and set(man) - set(saved) == {"driver_extrinsic"}
    for key in ("image", "camera_info"):
        assert set(man[key]) == set(saved[key]), key
    assert set(saved["pointcloud"]) <= set(man["pointcloud"])
    assert set(man["pointcloud"]) - set(saved["pointcloud"]) == {"source"}
    assert set(saved["topics"]) <= set(man["topics"])
    assert set(man["image"]["header"]) == set(saved["image"]["header"])
    assert man["pointcloud"]["stored_height"] == 1 and man["pointcloud"]["height"] == 1
    assert man["sync_delta_ns"] == 1000
    path = cloud.write_manifest(str(tmp_path / "m.json"), man)
    assert json.loads(Path(path).read_text()) == man


def test_pcd_written_in_recorder_format(tmp_path):
    pts = np.array([[1, 2, 3, 0], [4, 5, 6, 0]], dtype=np.float32)
    path = cloud.write_pcd(str(tmp_path / "a.pcd"), pts)
    head = Path(path).read_bytes().split(b"DATA binary\n")[0].decode()
    assert "FIELDS x y z rgb" in head and "WIDTH 2" in head and "HEIGHT 1" in head and "POINTS 2" in head
    xyz, _ = read_pcd_xyzrgb(path)
    np.testing.assert_array_equal(xyz, pts[:, :3])


def test_registration_validation():
    hd = lambda: SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=1), frame_id="f")  # noqa: E731
    col = SimpleNamespace(width=64, height=48, encoding="rgb8", header=hd())
    dep = SimpleNamespace(width=64, height=48, encoding="16UC1", header=hd())
    k = [60.0, 0, 31.5, 0, 61.0, 23.5, 0, 0, 1]
    info = SimpleNamespace(width=64, height=48, k=k, header=hd())
    assert cloud.validate_registration(col, dep, info)[0] == 0.001
    dep2 = SimpleNamespace(width=128, height=96, encoding="16UC1", header=hd())
    with pytest.raises(cloud.CaptureError, match="not aligned"):
        cloud.validate_registration(col, dep2, info)
    with pytest.raises(cloud.CaptureError, match="depth encoding"):
        cloud.validate_registration(col, SimpleNamespace(width=64, height=48, encoding="8UC1", header=hd()), info)
    other = SimpleNamespace(width=64, height=48, encoding="16UC1",
                            header=SimpleNamespace(stamp=None, frame_id="g"))
    with pytest.raises(cloud.CaptureError, match="frame_id"):
        cloud.validate_registration(col, other, info)
    # uniform scale of K is accepted, non-uniform refused
    info2 = SimpleNamespace(width=32, height=24, k=k, header=hd())
    sc, k2, w, h = cloud.validate_registration(col, dep, info2)
    assert (w, h) == (64, 48) and k2[0] == 120.0 and k2[5] == 47.0
    info3 = SimpleNamespace(width=32, height=48, k=k, header=hd())
    with pytest.raises(cloud.CaptureError, match="uniform"):
        cloud.validate_registration(col, dep, info3)
