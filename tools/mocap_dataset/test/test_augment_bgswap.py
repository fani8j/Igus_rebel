"""Tests for augment_bgswap.py: compositing math, determinism, mask-preserving pixels, wrist confidence, and a tiny end to end
encode of one synthetic episode (PyAV + libx264 only; no GPU, no SAM3).

    cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_augment_bgswap.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import augment_bgswap as ab  # noqa: E402

H, W = ab.H, ab.W


def synth_frame(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def synth_labels(n=5):
    lab = np.zeros((n, H, W), np.uint8)
    for i in range(n):
        lab[i, 100:300, 50 + 3 * i:200 + 3 * i] = 1          # "robot"
        lab[i, 320:440, 300:500] = 2                          # "box"
    return lab


# ------------------------------------------------------------------------------------------------ compositing math
def test_soft_alpha_properties():
    fg = synth_labels(1)[0] > 0
    a = ab.soft_alpha(fg, grow_px=2, feather_px=5)
    assert a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1
    assert np.all(a[fg] == 1.0)                               # mask pixels fully foreground
    d = np.array(__import__("cv2").distanceTransform((~fg).astype(np.uint8), 2, 3))
    assert np.all(a[d > 2 + 5 + 1] == 0.0)                    # nothing beyond the guard + feather
    assert np.any((a > 0) & (a < 1))                          # a band exists
    # monotone non-increasing with distance from the mask (sampled along a row crossing the right edge of the robot box)
    row = a[200, 200:230]
    assert np.all(np.diff(row) <= 1e-6)


def test_composite_identity_and_replacement():
    f = synth_frame(1)
    bg = np.full((H, W, 3), 17.0, np.float32)
    assert np.array_equal(ab.composite(f, np.ones((H, W), np.float32), bg), f)
    assert np.all(ab.composite(f, np.zeros((H, W), np.float32), bg) == 17)
    half = ab.composite(f, np.full((H, W), 0.5, np.float32), bg)
    assert np.abs(half.astype(int) - np.rint(0.5 * f + 0.5 * 17)).max() <= 1


@pytest.mark.parametrize("cam", ["head"])
def test_mask_pixels_byte_identical_without_jitter(cam):
    app = ab.sample_appearance(123, 1)
    tex = ab.make_texture(app)
    f = synth_frame(2)
    lab = synth_labels()
    fg = ab.union_fg(lab, 2)
    out = ab.augment_frame(f, fg, tex, app, cam, ab.render_distractors(app, (lab > 0).mean(0)), jitter=False)
    assert out.shape == f.shape and out.dtype == np.uint8
    assert np.array_equal(out[fg], f[fg])                     # robot + box untouched
    far = ab.soft_alpha(fg, 2, 4) == 0
    assert far.mean() > 0.5
    assert not np.array_equal(out[far], f[far])               # the background really changed


def test_union_fg_covers_neighbours():
    lab = synth_labels(5)
    u = ab.union_fg(lab, 2)
    assert np.all(u[lab[1] > 0]) and np.all(u[lab[3] > 0]) and np.all(u[lab[2] > 0])
    assert ab.union_fg(lab, 0).sum() >= (lab[0] > 0).sum()


def test_jitter_is_global_and_bounded():
    app = ab.sample_appearance(5, 1)
    f = synth_frame(3)
    j = ab.apply_jitter(f, app)
    assert j.shape == f.shape and j.dtype == np.uint8
    assert np.abs(j.astype(int) - f.astype(int)).mean() < 30       # mild
    ident = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}
    assert np.abs(ab.global_jitter(f, ident) - f).max() < 1e-3


def test_recolor_identity_and_range():
    f = synth_frame(4)
    r = ab.recolor(f, 0.0, [1, 1, 1], 1.0)
    assert np.abs(r - f).max() < 1.0
    r2 = ab.recolor(f, 40.0, [1.1, 0.9, 1.0], 0.9)
    assert r2.min() >= 0 and r2.max() <= 255 and np.abs(r2 - f).mean() > 5


# ------------------------------------------------------------------------------------------------ determinism
def test_appearance_deterministic_and_seed_sensitive():
    s1 = ab.seed_for(1, "mocap_right_2_ep001", 1)
    assert s1 == ab.seed_for(1, "mocap_right_2_ep001", 1)
    assert s1 != ab.seed_for(1, "mocap_right_2_ep001", 2)
    assert s1 != ab.seed_for(1, "mocap_right_2_ep002", 1)
    assert s1 != ab.seed_for(2, "mocap_right_2_ep001", 1)
    assert ab.sample_appearance(s1, 1) == ab.sample_appearance(s1, 1)
    assert ab.sample_appearance(s1, 1) != ab.sample_appearance(s1 + 1, 1)


@pytest.mark.parametrize("kind", ab.TEX_KINDS)
def test_textures_deterministic_in_range(kind):
    app = ab.sample_appearance(77, 1)
    app.tex_kind = kind
    t1, t2 = ab.make_texture(app), ab.make_texture(app)
    assert np.array_equal(t1, t2)
    assert t1.shape == (H, W, 3) and t1.dtype == np.float32 and t1.min() >= 0 and t1.max() <= 255
    assert t1.std() > 0.5 or kind == "plain" and t1.std() >= 0


def test_all_texture_kinds_get_drawn():
    kinds = {ab.sample_appearance(s, 1).tex_kind for s in range(200)}
    assert kinds == set(ab.TEX_KINDS)


def test_frame_output_deterministic():
    app = ab.sample_appearance(9, 2)
    tex = ab.make_texture(app)
    f, lab = synth_frame(5), synth_labels()
    fg = ab.union_fg(lab, 1)
    d = ab.render_distractors(app, (lab > 0).mean(0))
    o1 = ab.augment_frame(f, fg, tex, app, "head", d)
    o2 = ab.augment_frame(f, fg, ab.make_texture(app), app, "head", ab.render_distractors(app, (lab > 0).mean(0)))
    assert np.array_equal(o1, o2)


def test_same_appearance_every_frame_no_flicker():
    """The background layer of a static scene is identical in every frame (one appearance per episode)."""
    app = ab.sample_appearance(11, 1)
    tex = ab.make_texture(app)
    f = synth_frame(6)
    lab = synth_labels()
    outs = [ab.augment_frame(f, ab.union_fg(lab, i), tex, app, "head", None, jitter=True) for i in range(5)]
    far = (ab.soft_alpha(np.logical_or.reduce([lab[i] > 0 for i in range(5)]), 2, 4) == 0)
    for o in outs[1:]:
        assert np.array_equal(o[far], outs[0][far])


def test_distractors_avoid_busy_pixels():
    app = ab.sample_appearance(21, 1)
    app.distractors = [{"shape": "rect", "u": 0.5, "v": 0.5, "size": 40.0, "aspect": 1.0, "rot": 0.0, "color": [250.0, 10.0, 10.0], "shadow": 0.3}]
    busy = np.ones((H, W), np.float32)
    busy[300:400, 100:200] = 0.0                              # only free region
    rgb, alpha, shade = ab.render_distractors(app, busy)
    ys, xs = np.nonzero(alpha > 0.5)
    assert len(ys) > 100 and ys.min() >= 280 and ys.max() <= 420 and xs.min() >= 80 and xs.max() <= 220
    assert shade.min() < 1.0 and shade.max() <= 1.0


def test_distractor_count_range():
    counts = {len(ab.sample_appearance(s, 1).distractors) for s in range(100)}
    assert counts == {0, 1, 2}


# ------------------------------------------------------------------------------------------------ wrist photometric
def test_wrist_photo_ranges_and_determinism():
    for s in range(100):
        wp = ab.sample_appearance(s, 1).wrist_photo
        assert 0.8 <= wp["brightness"] <= 1.2 and 0.8 <= wp["contrast"] <= 1.2 and 0.8 <= wp["saturation"] <= 1.2
        assert abs(wp["hue_deg"]) <= 0.03 * 360 + 1e-9 and 0.85 <= wp["gamma"] <= 1.15
    assert ab.sample_appearance(5, 1).wrist_photo == ab.sample_appearance(5, 1).wrist_photo


def test_wrist_photometric_identity_and_geometry():
    f = synth_frame(8)
    ident = {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "hue_deg": 0.0, "gamma": 1.0}
    assert np.abs(ab.wrist_photometric(f, ident).astype(int) - f.astype(int)).max() <= 1
    wp = ab.sample_appearance(3, 1).wrist_photo
    o = ab.wrist_photometric(f, wp)
    assert o.shape == f.shape and o.dtype == np.uint8
    assert np.array_equal(ab.wrist_photometric(f, wp), o)                    # same params -> same output
    # purely per-pixel: a pixel's output depends only on that pixel (no texture, no mask, no geometry change)
    g = f.copy()
    g[:, 100:] = 7
    assert np.array_equal(ab.wrist_photometric(g, wp)[:, :100], o[:, :100])


# ------------------------------------------------------------------------------------------------ encode one episode
def test_run_episode_cam_end_to_end(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("lerobot")
    n = 8
    src = tmp_path / "src.mp4"
    with av.open(str(src), "w") as c:
        st = c.add_stream("libx264", rate=30)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "16", "g": "2"}
        for i in range(n):
            img = np.full((H, W, 3), 60 + 10 * i, np.uint8)
            img[100:300, 100:300] = (200, 50, 50)
            for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    lab = np.zeros((n, H, W), np.uint8)
    lab[:, 100:300, 100:300] = 1
    np.savez_compressed(tmp_path / "labels.npz", labels=lab)
    apps = [ab.asdict(ab.sample_appearance(s, k + 1)) for k, s in enumerate((1, 2))]
    task = {"id": (0, "observation.images.head"), "cam": "head", "length": n, "fps": 30.0, "src": str(src), "from_ts": 0.0,
            "mask_path": str(tmp_path / "labels.npz"), "appearances": apps, "dst": [str(tmp_path / f"o{k}.mp4") for k in range(2)],
            "crf": 16, "gop": 2, "grow_px": 2, "feather_px": 4, "jitter": False, "static_robot_area": 0.0}
    r = ab.run_episode_cam(task)
    assert [x["frames"] for x in r["results"]] == [n, n] and r["unchanged_frames"] == []
    for k in range(2):
        with av.open(str(tmp_path / f"o{k}.mp4")) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
        assert len(frames) == n and frames[0].shape == (H, W, 3)
        assert np.abs(frames[3][150:250, 150:250].astype(int) - np.array([200, 50, 50])).mean() < 6   # robot survives the codec
        assert np.abs(frames[3][400:, 400:].astype(int) - np.array([60 + 30] * 3)).mean() > 5         # background changed
    assert r["results"][0]["stats"]["mean"].shape == (3, 1, 1)


def test_wrist_episode_has_no_texture(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("lerobot")
    n = 6
    src = tmp_path / "w.mp4"
    base = np.dstack([np.tile(np.linspace(20, 230, W, dtype=np.uint8), (H, 1)), np.tile(np.linspace(200, 30, H, dtype=np.uint8)[:, None], (1, W)),
                      np.full((H, W), 90, np.uint8)])
    with av.open(str(src), "w") as c:
        st = c.add_stream("libx264", rate=30)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "10", "g": "2"}
        for i in range(n):
            for p in st.encode(av.VideoFrame.from_ndarray(base, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    app = ab.sample_appearance(4, 1)
    task = {"id": (0, "observation.images.wrist_right"), "cam": "wrist_right", "length": n, "fps": 30.0, "src": str(src), "from_ts": 0.0,
            "mask_path": "unused", "appearances": [ab.asdict(app)], "dst": [str(tmp_path / "w1.mp4")], "crf": 16, "gop": 2,
            "grow_px": 2, "feather_px": 4, "jitter": True, "static_robot_area": 0.0}
    r = ab.run_episode_cam(task)
    assert r["unchanged_frames"] == [] and r["results"][0]["frames"] == n
    with av.open(str(tmp_path / "w1.mp4")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    with av.open(str(src)) as c:
        ref = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    expect = ab.wrist_photometric(ref[2], app.wrist_photo)
    assert np.abs(frames[2].astype(int) - expect.astype(int)).mean() < 6
