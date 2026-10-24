# Provenance: ported from rgbd_captures/tests/test_corner_names.py (HEAD 6df5994); imports made package-relative.
"""Corner naming rule long_side_back_v1 (scripts/autolabel_v2/corner_names.py)
and its use in the autolabeler record renamer, the evaluator and the review
server. Synthetic geometry only."""

import itertools
import json
import math
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

from box_seam_detection.core import corner_names as cn  # noqa: E402
from box_seam_detection.core.fold_edge import _project  # noqa: E402

C = ("back_left", "back_right", "front_right", "front_left")
K = np.array([[690.0, 0, 640.0], [0, 690.0, 360.0], [0, 0, 1]])
DIST = [0.00705, -0.0492, 0.00026, 0.0000918, 0.0332]


def scene(yaw_deg, ox=0.0, pitch_deg=40.0, depth=0.55, long_m=0.112,
          short_m=0.100):
    """Box top on a table seen by a camera pitched down. Returns (true
    named xyz dict, plane dict)."""
    p = math.radians(pitch_deg)
    r = np.array([1.0, 0, 0])
    f = np.array([0.0, -math.sin(p), math.cos(p)])   # horizontal forward
    up = np.array([0.0, -math.cos(p), -math.sin(p)])  # plane normal
    P0 = np.array([0.0, 0.0, depth]) + ox * r
    y = math.radians(yaw_deg)
    e1 = math.cos(y) * r + math.sin(y) * f   # long axis
    e2 = -math.sin(y) * r + math.cos(y) * f  # short axis
    pts = {}
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            pts[(s1, s2)] = P0 + s1 * e1 * long_m / 2 + s2 * e2 * short_m / 2
    plane = {"normal": up.tolist(), "offset_m": float(-up @ P0)}
    return pts, plane


def project(X, dist=None):
    from box_seam_detection.core.fold_edge import Camera
    return _project(np.atleast_2d(X), Camera(K, dist))[0]


def expected_names(pts, dist=None):
    """Independent brute force from the TRUE 3-D rectangle: back = long side
    (constant s2) whose midpoint is farther; left = smaller image x."""
    mids = {s2: 0.5 * (pts[(-1, s2)] + pts[(1, s2)]) for s2 in (-1, 1)}
    sb = max((-1, 1), key=lambda s: np.linalg.norm(mids[s]))
    sf = -sb
    ends = sorted((-1, 1), key=lambda s1: project(pts[(s1, sb)], dist)[0])
    bl, br = ends
    return {"back_left": pts[(bl, sb)], "back_right": pts[(br, sb)],
            "front_right": pts[(br, sf)], "front_left": pts[(bl, sf)]}, \
        abs(np.linalg.norm(mids[sb]) - np.linalg.norm(mids[sf]))


@pytest.mark.parametrize("dist", [None, DIST])
def test_rule_sweep_rotations_any_input_order(dist):
    checked = 0
    for yaw in range(0, 360, 15):
        for ox in (-0.08, 0.0, 0.08):
            pts, plane = scene(yaw, ox)
            exp3, dd = expected_names(pts, dist)
            exp_px = {k: project(v, dist) for k, v in exp3.items()}
            b = exp_px["back_right"] - exp_px["back_left"]
            if dd * 1000 < 6.0 or abs(b[0]) < math.tan(
                    math.radians(12)) * abs(b[1]):
                continue  # ambiguous / near-vertical: dedicated tests
            px = [project(v, dist) for v in pts.values()]
            for perm in itertools.permutations(range(4)):
                named, info = cn.canonical_corner_names(
                    [px[i] for i in perm], plane, K, dist)
                assert info["method"] == "plane_3d"
                for k in C:
                    assert named[k] == pytest.approx(exp_px[k], abs=1e-6), \
                        (yaw, ox, perm, k)
            assert info["long_side_mm"] == pytest.approx(112.0, abs=0.1)
            assert info["short_side_mm"] == pytest.approx(100.0, abs=0.1)
            assert info["back_mid_dist_m"] > info["front_mid_dist_m"]
            checked += 1
    assert checked > 40
    # 45 deg is in the sweep (yaw 45, 225 ...)


def test_yaw45_explicit():
    pts, plane = scene(45.0, 0.03)
    exp3, _ = expected_names(pts)
    named, info = cn.canonical_corner_names(
        {k: project(v).tolist() for k, v in zip(
            ("front_left", "back_left", "back_right", "front_right"),
            exp3.values())}, plane, K)
    for k in C:
        assert named[k] == pytest.approx(project(exp3[k]).tolist())
    assert info["renamed"] is True and info["orientation"] == "cw"


def test_near_vertical_back_edge_clockwise_tie_rule():
    # long sides run away from the camera; the right one (x = +0.10 m) is
    # farther. Its image is within 10 deg of vertical -> back_left is the
    # end making the ring clockwise = its FAR (upper) end.
    pts, plane = scene(90.0, ox=0.05, pitch_deg=60.0, depth=0.5)
    px = {k: project(v) for k, v in pts.items()}
    far_right = [k for k in pts if pts[k][0] > 0.09]
    assert len(far_right) == 2
    top, bottom = sorted(far_right, key=lambda k: px[k][1])
    named, info = cn.canonical_corner_names(list(px.values()), plane, K)
    assert "near_vertical_back_edge" in info["flags"]
    assert named["back_left"] == pytest.approx(px[top].tolist())
    assert named["back_right"] == pytest.approx(px[bottom].tolist())
    # fronts adjacent to their backs (same long-axis end)
    fl = [k for k in pts if k[0] == top[0] and k != top][0]
    fr = [k for k in pts if k[0] == bottom[0] and k != bottom][0]
    assert named["front_left"] == pytest.approx(px[fl].tolist())
    assert named["front_right"] == pytest.approx(px[fr].tolist())
    ring = np.array([named[k] for k in C])
    area = 0.5 * sum(ring[i][0] * ring[(i + 1) % 4][1]
                     - ring[(i + 1) % 4][0] * ring[i][1] for i in range(4))
    assert area > 0  # clockwise on screen (y down)


def test_ambiguous_square_and_image_fallback():
    pts, plane = scene(10.0, 0.0, long_m=0.101, short_m=0.100)
    px = [project(v) for v in pts.values()]
    _, info = cn.canonical_corner_names(px, plane, K)
    assert "ambiguous_long_side" in info["flags"] and info["ambiguous"]
    pts, plane = scene(0.0, 0.0)
    px = [project(v) for v in pts.values()]
    named, info = cn.canonical_corner_names(px, None, None)
    assert info["method"] == "image_fallback"
    assert "no_plane_image_fallback" in info["flags"]
    assert info["long_side_mm"] is None
    # yaw 0: back is the upper long side, left = smaller x
    assert named["back_left"][1] < named["front_left"][1]
    assert named["back_left"][0] < named["back_right"][0]


def _record_from(names_px, names_xyz, plane):
    """Autolabel-like record whose stored names are ``names_px`` keys."""
    flags = {k: {"flags": [f"f_{i}"], "shift_px": float(i)}
             for i, k in enumerate(C)}
    per_edge = {e: {"method": f"m_{e}", "side_face": f"s_{e}"}
                for e in ("back", "right", "front", "left")}
    return {
        "schema": "box_autolabel/v5", "capture_id": "cap",
        "image_size": [1280, 720],
        "provenance": {
            "attempts": [{"status": "refine_rejected", "coarse": [[0, 0]] * 4},
                         {"status": "accepted",
                          "coarse": [[names_px[k][0] + 1, names_px[k][1]]
                                     for k in C]}],
            "corner_fold": {"corners_px_before": {
                k: [names_px[k][0] + 2, names_px[k][1]] for k in C},
                "per_edge": per_edge}},
        "top_face": {"corners_px": {k: list(names_px[k]) for k in C},
                     "corners_xyz_camera_m": {k: list(names_xyz[k])
                                              for k in C},
                     "plane": plane},
        "seam": {"status": "ok", "axis": "lr",
                 "endpoints_px": {"seam_left": [1.0, 2.0],
                                  "seam_right": [3.0, 4.0]},
                 "estimate": {"axis": "lr",
                              "candidates": [{"axis": "lr"}, {"axis": "bf"}],
                              "estimate": {"axis": "lr", "gap_support":
                                           {"lr": 0.9, "bf": 0.1},
                                           "gap": {"prior_rect": 5}}}},
        "checks": {"corner_flags": {"corners": flags, "any_flag": True},
                   "reasons": ["corner_flags(back_left,front_right)",
                               "corner_back_left(f_0)",
                               "corner_fold_moved(back,left)"]},
    }


def test_rename_record_permutes_consistently_and_swaps_axis():
    pts, plane = scene(20.0, 0.02)
    exp3, _ = expected_names(pts)
    true_px = {k: project(v).tolist() for k, v in exp3.items()}
    true_xyz = {k: v.tolist() for k, v in exp3.items()}
    # stored names rotated one step: stored back_left = true front_left ...
    rot = {"back_left": "front_left", "back_right": "back_left",
           "front_right": "back_right", "front_left": "front_right"}
    old_px = {k: true_px[rot[k]] for k in C}
    old_xyz = {k: true_xyz[rot[k]] for k in C}
    rec = _record_from(old_px, old_xyz, plane)
    man = {"camera_info": {"k": K.ravel().tolist(), "d": [0.0] * 5,
                           "distortion_model": "plumb_bob"}}
    out, info = cn.rename_record(rec, man, source="src")
    assert info["renamed"] and info["axis_swapped"]
    tf = out["top_face"]
    for k in C:
        assert tf["corners_px"][k] == pytest.approx(true_px[k])
        assert tf["corners_xyz_camera_m"][k] == pytest.approx(true_xyz[k])
    # per-corner data follows its point
    old_by_pt = {tuple(old_px[k]): k for k in C}
    for k in C:
        ok = old_by_pt[tuple(tf["corners_px"][k])]
        assert out["checks"]["corner_flags"]["corners"][k] == \
            rec["checks"]["corner_flags"]["corners"][ok]
        assert out["provenance"]["corner_fold"]["corners_px_before"][k] == \
            rec["provenance"]["corner_fold"]["corners_px_before"][ok]
    coarse = out["provenance"]["attempts"][1]["coarse"]
    for i, k in enumerate(C):
        assert coarse[i] == [tf["corners_px"][k][0] + 1,
                             tf["corners_px"][k][1]]
    assert out["provenance"]["attempts"][0]["coarse"] == [[0, 0]] * 4
    # per-edge data follows the physical edge (same two corners)
    edge_new = info["edge_mapping_new_from_old"]
    for e, oe in edge_new.items():
        assert out["provenance"]["corner_fold"]["per_edge"][e] == \
            rec["provenance"]["corner_fold"]["per_edge"][oe]
        a, b = cn.EDGE_CORNERS[e]
        m = info["mapping_new_from_old"]
        assert {m[a], m[b]} == set(cn.EDGE_CORNERS[oe])
    # seam: axis label swapped everywhere, endpoints untouched, rect frame kept
    s = out["seam"]
    assert s["axis"] == "bf" and s["estimate"]["axis"] == "bf"
    assert [c["axis"] for c in s["estimate"]["candidates"]] == ["bf", "lr"]
    assert s["estimate"]["estimate"]["gap_support"] == {"bf": 0.9, "lr": 0.1}
    assert s["estimate"]["estimate"]["gap"] == {"prior_rect": 5}
    assert s["endpoints_px"] == rec["seam"]["endpoints_px"]
    # reasons renamed with the same mapping
    o2n = {v: k for k, v in info["mapping_new_from_old"].items()}
    r = out["checks"]["reasons"]
    assert r[1] == f"corner_{o2n['back_left']}(f_0)"
    assert r[0] == "corner_flags(" + ",".join(
        sorted([o2n["back_left"], o2n["front_right"]])) + ")"
    assert r[2].startswith("corner_fold_moved(")
    assert out["provenance"]["corner_names"]["rule"] == cn.RULE
    # idempotent
    out2, info2 = cn.rename_record(out, man)
    assert not info2["renamed"]
    assert out2["top_face"] == out["top_face"] and out2["seam"] == out["seam"]


def test_rename_record_no_corners():
    rec = {"top_face": {"corners_px": None}, "seam": {"axis": None}}
    out, info = cn.rename_record(rec, None)
    assert info["renamed"] is False and info["reason"] == "no_corners"
    assert out["top_face"] == rec["top_face"]


def test_pipeline_apply_corner_names_respects_config():
    from box_seam_detection.core.config import default_config
    from box_seam_detection.core.pipeline import apply_corner_names
    pts, plane = scene(20.0, 0.02)
    exp3, _ = expected_names(pts)
    true_px = {k: project(v).tolist() for k, v in exp3.items()}
    rot = {"back_left": "front_left", "back_right": "back_left",
           "front_right": "back_right", "front_left": "front_right"}
    rec = _record_from({k: true_px[rot[k]] for k in C},
                       {k: exp3[rot[k]].tolist() for k in C}, plane)
    man = {"camera_info": {"k": K.ravel().tolist(), "d": []}}
    cfg = default_config()
    assert cfg["corner_names"]["enabled"] is True
    debug = {}
    out = apply_corner_names(rec, man, cfg, debug)
    assert out["top_face"]["corners_px"]["back_left"] == pytest.approx(
        true_px["back_left"])
    assert debug["estimate"]["axis"] == "bf"
    cfg["corner_names"]["enabled"] = False
    assert apply_corner_names(rec, man, cfg) is rec


# ------------------------------------------------------------ evaluator

EVAL = REPO / "scripts" / "eval" / "evaluate.py"


def _eval_env(tmp_path, human_names_rotated):
    cid = "frame_000001_20260917_061638_077995"
    pts, plane = scene(20.0, 0.02)
    exp3, _ = expected_names(pts)
    true_px = {k: project(v).tolist() for k, v in exp3.items()}
    (tmp_path / "captures").mkdir()
    (tmp_path / "captures" / f"{cid}.json").write_text(json.dumps({
        "camera_info": {"k": K.ravel().tolist(), "d": [0.0] * 5,
                        "distortion_model": "plumb_bob"},
        "image": {"width": 1280, "height": 720}}))
    ap = tmp_path / "auto"
    ap.mkdir()
    (ap / f"{cid}.json").write_text(json.dumps({
        "capture_id": cid, "image_size": [1280, 720],
        "top_face": {"corners_px": true_px, "plane": plane},
        "seam": {"status": "invisible"}, "checks": {}, "confidence": 0.8}))
    rot = {"back_left": "front_left", "back_right": "back_left",
           "front_right": "back_right", "front_left": "front_right"}
    hum = {k: [true_px[rot[k] if human_names_rotated else k][0] + 0.3,
               true_px[rot[k] if human_names_rotated else k][1]]
           for k in C}
    lp = tmp_path / "labels.json"
    lp.write_text(json.dumps({"schema": "box_autolabel_review/v5",
                              "records": {cid: {
                                  "decision": "fix", "corners_px": hum,
                                  "seam_status": "invisible",
                                  "seam_endpoints_px": None,
                                  "pixel_convention": "center",
                                  "blind": True}}}))
    ids = tmp_path / "ids.txt"
    ids.write_text(cid + "\n")
    return lp, ap, ids


def _run(lp, ap, ids, out, data_root, *extra):
    r = subprocess.run(
        [sys.executable, str(EVAL), "--reviews", str(lp), "--blind-labels",
         str(lp), "--ids-file", str(ids), "--pred", f"autolabel:{ap}",
         "--out", str(out), "--data-root", str(data_root),
         "--shown-autolabel-root", "", *extra],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s = json.loads((out / "summary.json").read_text())
    return s["predictors"][f"autolabel:{ap.name}"]






# ------------------------------------------------------------ review server

def _load_srv():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "serve_review_cn", str(REPO / "review" / "serve_review.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _review_env(tmp_path, with_manifest=True, blind=False):
    srv_mod = _load_srv()
    pts, plane = scene(20.0, 0.02)
    exp3, _ = expected_names(pts)
    true_px = {k: [round(float(v), 1) for v in project(p)]
               for k, p in exp3.items()}
    root = tmp_path / "rgb"
    root.mkdir()
    adir = tmp_path / "auto"
    adir.mkdir()
    caps = tmp_path / "captures"
    caps.mkdir()
    cid = "cap_a"
    (adir / f"{cid}.json").write_text(json.dumps({
        "schema": "box_autolabel/v5", "capture_id": cid,
        "image": str(root / "a.png"), "image_size": [1280, 720],
        "top_face": {"corners_px": true_px, "plane": plane},
        "seam": {"status": "invisible"}, "checks": {}, "confidence": 0.5}))
    if with_manifest:
        (caps / f"{cid}.json").write_text(json.dumps({"camera_info": {
            "k": K.ravel().tolist(), "d": [0.0] * 5,
            "distortion_model": "plumb_bob"}}))
    srv = srv_mod.build_server(str(adir), str(tmp_path / "rev.json"), "t", 0,
                               data_root=str(root), blind=blind)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    return srv, base, cid, true_px


def _post(base, path, obj):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())






_NODE = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const ops = [];
function mkCtx(tag) {
  return new Proxy({}, {get(t, k) { if (k in t) return t[k];
    return (...a) => { ops.push([tag, String(k), a]); }; },
    set(t, k, v) { t[k] = v; return true; }});
}
const els = {};
function el(id) {
  if (!els[id]) els[id] = {id, style: {}, textContent: "", innerHTML: "", value: "",
    title: "", width: 0, height: 0, dataset: {}, listeners: {},
    classList: {toggle() {}}, appendChild() {},
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    getBoundingClientRect() { return {left: 0, top: 0, width: 800, height: 600}; },
    getContext() { return mkCtx(id); }};
  return els[id];
}
const docListeners = {}, winListeners = {};
const posts = [];
const ITEMS = JSON.parse(process.argv[3]);
const NAMED = {back_left: [30, 20], back_right: [80, 20], front_right: [80, 60], front_left: [30, 60]};
const sandbox = {
  console, Math, JSON, Object, Array, Number, String, Promise, Proxy, parseInt, isFinite,
  setTimeout, encodeURIComponent,
  window: {devicePixelRatio: 1, location: {search: ""},
           addEventListener(t, f) { (winListeners[t] = winListeners[t] || []).push(f); }},
  document: {getElementById: el, createElement: () => el("_n" + Math.random()),
             querySelectorAll: () => [], activeElement: null,
             addEventListener(t, f) { (docListeners[t] = docListeners[t] || []).push(f); }},
  alert: (m) => { ops.push(["alert", m]); },
  fetch: async (url, opt) => {
    if (opt && opt.method === "POST") {
      const b = JSON.parse(opt.body); posts.push(b);
      const rec = Object.assign({}, b, {corners_px: NAMED, corner_naming: "long_side_back_v1",
                                         pixel_convention: "center", blind: true});
      return {ok: true, json: async () => ({ok: true, record: rec,
        corner_naming: {method: "plane_3d", renamed: true, flags: []}})};
    }
    return {ok: true, json: async () => ({items: ITEMS, blind: true})};
  },
  Image: function () { const self = this; this.naturalWidth = 0; this.naturalHeight = 0;
    Object.defineProperty(this, "src", {set(v) { self._src = v;
      self.naturalWidth = 100; self.naturalHeight = 80;
      setTimeout(() => self.onload && self.onload(), 0); }, get() { return self._src; }}); },
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(src + `
globalThis.__T = {natToCanvas, draw, select, get edit() { return edit; },
  set placeMode(v) { placeMode = v; }};`, sandbox);
const T = sandbox.__T;
function mouse(type, x, y) {
  const c = T.natToCanvas(x, y);
  const ev = {clientX: c[0], clientY: c[1], button: 0, shiftKey: false, preventDefault() {}};
  (els.cv.listeners[type] || []).forEach(f => f(ev));
}
setTimeout(async () => {
  const out = {};
  T.placeMode = "corner";
  // bowtie click order: FR, BL, FL, BR
  mouse("mousedown", 80, 60); mouse("mousedown", 30, 20);
  out.hint2 = els.placehint.textContent;
  mouse("mousedown", 30, 60); mouse("mousedown", 80, 20);
  out.pending = T.edit.namesPending;
  out.ring = ["back_left", "back_right", "front_right", "front_left"].map(k => T.edit.corners[k]);
  ops.length = 0; T.draw();
  out.labels = ops.filter(o => o[0] === "cv" && o[1] === "fillText").map(o => o[2][0]);
  (docListeners.keydown || []).forEach(f => f({key: "f", shiftKey: false}));
  await new Promise(r => setTimeout(r, 5));
  out.post = posts[0];
  out.namemsg = els.namemsg.textContent;
  T.select(0);  // back to the saved item: names from the server record
  out.afterPending = T.edit.namesPending;
  out.afterBL = T.edit.corners.back_left;
  ops.length = 0; T.draw();
  out.afterLabels = ops.filter(o => o[0] === "cv" && o[1] === "fillText").map(o => o[2][0]);
  out.help = els.helpbody.innerHTML;
  out.alerts = ops.filter(o => o[0] === "alert");
  console.log(JSON.stringify(out));
}, 20);
"""




def test_run_rename_only_mode(tmp_path):
    from box_seam_detection.core import run as runmod
    cid = "frame_999999_20990101_000000_000000"  # no manifest -> image rule
    src = tmp_path / "src"
    src.mkdir()
    q = {"back_left": [100.0, 300.0], "back_right": [100.0, 100.0],
         "front_right": [400.0, 100.0], "front_left": [400.0, 300.0]}
    rec = _record_from(q, {k: [float(i), 0.0, 1.0] for i, k in enumerate(C)},
                       None)
    rec["capture_id"] = cid
    (src / f"{cid}.json").write_text(json.dumps(rec))
    (src / "summary.json").write_text("{}")
    rc = runmod.main(["--rename-corners-from", str(src), "--run-name", "v",
                      "--out-root", str(tmp_path), "--no-overlay"])
    assert rc == 0
    s = json.loads((tmp_path / "v" / "summary.json").read_text())
    assert s["n_captures"] == 1 and s["n_geometry_mismatch"] == 0
    assert s["n_renamed"] == 1 and s["methods"] == {"image_fallback": 1}
    out = json.loads((tmp_path / "v" / f"{cid}.json").read_text())
    # image rule: long side = the 300 px sides (top y=100 is "farther")
    assert out["top_face"]["corners_px"]["back_left"] == [100.0, 100.0]
    assert out["top_face"]["corners_px"]["back_right"] == [400.0, 100.0]
    assert out["provenance"]["corner_names"]["from"] == str(src)
    assert out["seam"]["axis"] == "bf"
    with pytest.raises(SystemExit):
        runmod.main(["--rename-corners-from", str(src), "--run-name", "v",
                     "--out-root", str(tmp_path), "--no-overlay"])


def test_crossed_named_quad_is_not_straightened():
    from box_seam_detection.core.corner_names import canonical_corner_names
    crossed = {"back_left": [0, 0], "back_right": [100, 100],
               "front_right": [100, 0], "front_left": [0, 100]}
    named, info = canonical_corner_names(crossed)
    assert named == {k: [float(v[0]), float(v[1])] for k, v in crossed.items()}
    assert info["renamed"] is False and "input_not_simple" in info["flags"]
    # arbitrary click order (review tool) is still named
    named2, info2 = canonical_corner_names(crossed, strict_ring=False)
    assert "input_not_simple" not in info2["flags"]
