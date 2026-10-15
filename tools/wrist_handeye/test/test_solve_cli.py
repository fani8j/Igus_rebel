"""Offline `solve <run_dir>` round trip on synthetic data: files written, result close to the truth."""
import json
import os
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import test_handeye_math as t  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_solve_cli(tmp_path):
    B = t.board_flat()
    smp = t.synth_samples(B, t.X_TRUE, noise=0.2, outlier=False)
    recs = [{"pose_index": i, "stamp_ns": i, "T_base_link6": s.T_base_link6.tolist(), "obj": s.obj.tolist(), "pix": s.pix.tolist(),
             "n_points": len(s.obj), "image": "x%d.png" % i} for i, s in enumerate(smp)]
    (tmp_path / "samples.json").write_text(json.dumps({"samples": recs}))
    (tmp_path / "camera_info.json").write_text(json.dumps({"K": t.K.tolist(), "D": t.D.tolist(), "width": 1280, "height": 800,
                                                           "frame_id": "wrist_camera_color_optical_frame"}))
    (tmp_path / "board.json").write_text(json.dumps({"T_base_board_scene": B.tolist()}))
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "wrist_handeye.py"), "solve", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    res = yaml.safe_load((tmp_path / "result.yaml").read_text())
    assert res["schema"] == "wrist_handeye/v1" and res["parent"] == "link6" and res["child"] == "wrist_camera_color_optical_frame"
    assert res["status"] == "PASS"
    from handeye_math import T_from_xyz_rpy, pose_diff
    mm, dg = pose_diff(T_from_xyz_rpy(*res["xyz"], *res["rpy"]), t.X_TRUE)
    assert mm < 1.0 and dg < 0.1
    assert "<joint" in (tmp_path / "urdf_snippet.xml").read_text()
