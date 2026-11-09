import os

import cv2
import numpy as np

import synth
from side_cut_vision import replay
from side_cut_vision.coloronly import detect_color_only, draw_color_only


def test_color_only_on_synthetic_left_view_finds_edge_and_seam_in_pixels():
    cam = synth.load_camera()
    model = synth.load_arm("left")
    T, tip, _ = synth.approach_pose(model)
    color, _, truth = synth.render(cam, T, model, synth.SceneSpec(d_across_mm=0.0, d_along_mm=0.0,
                                                                   d_normal_mm=0.0, yaw_deg=0.0))
    res = detect_color_only(color, "horizontal")
    assert res.ok and res.edge_p0 is not None and res.seams
    # the true seam: crossing S projected into the image (the seam runs perpendicular to the edge)
    Tinv = np.linalg.inv(T)
    uv = cam.project((Tinv[:3, :3] @ model.S + Tinv[:3, 3])[None, :])[0]
    best = res.seams[0]
    xs = best.p_top[0] + (uv[1] - best.p_top[1]) / (best.p_edge[1] - best.p_top[1] + 1e-9) * (best.p_edge[0] - best.p_top[0])
    assert abs(xs - uv[0]) < 25, (xs, uv)
    img = draw_color_only(color, res, "synthetic")
    assert img.shape == color.shape


def test_color_only_bad_images_do_not_raise(tmp_path):
    assert not detect_color_only(np.zeros((10, 10), np.uint8)).ok
    flat = np.full((800, 1280, 3), 128, np.uint8)
    res = detect_color_only(flat)
    assert not res.ok and res.reasons
    draw_color_only(flat, res)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), flat[:, :, ::-1])
    out = replay.replay_color_only(str(p), str(tmp_path / "o"))
    assert out and not out[0]["ok"]


def test_color_only_cli_on_video(tmp_path):
    cam = synth.load_camera()
    model = synth.load_arm("left")
    T, tip, _ = synth.approach_pose(model)
    color, _, _ = synth.render(cam, T, model, synth.SceneSpec())
    vp = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (1280, 800))
    for _ in range(10):
        vw.write(color[:, :, ::-1].copy())
    vw.release()
    rc = replay.main(["color-only", vp, "--at-s", "0.2", "1.0", "--out-dir", str(tmp_path / "o"),
                      "--orientation", "horizontal"])
    assert rc == 0
    assert len([f for f in os.listdir(tmp_path / "o") if f.endswith(".png")]) == 2
