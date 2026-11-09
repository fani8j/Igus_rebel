#!/usr/bin/env python3
"""Rerun viewer for a LeRobot mocap episode (v3.0 or v2.1; a ReBeL arm opening carton flaps; ``--arm`` picks the arm).

Shows, on two timelines (``time`` = dataset seconds, ``frame`` = frame index):
  * an animated 3D model of the real dual-arm cell (rebel_dual_table.urdf.xacro, expanded with xacro)
    driven by the joints of one arm (``--arm``, default: the arm that moves; the other arm stays at its zero pose), plus its
    TCP (<arm>_xeg32_tool_tip) as a moving marker and its whole-episode path as a line strip;
  * the camera videos (head, wrist_left, wrist_right, knife; those the dataset has) as rr.AssetVideo + rr.VideoFrameReference;
  * time-series plots of every non-zero state / action channel.

Usage (run from anywhere, not from inside the dataset directory):

    D=~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN
    # open the viewer on one episode dataset
    python3 tools/mocap_dataset/rerun_view.py $D/mocap_left_13
    # parent directory: pick the episode dataset with --episode-dir
    python3 tools/mocap_dataset/rerun_view.py $D --episode-dir mocap_left_new_3
    # write a file instead of spawning a viewer, then open it later with `rerun out.rrd`
    python3 tools/mocap_dataset/rerun_view.py $D/mocap_left_13 --save /tmp/left_13.rrd
    # options: --source state|action|both (plots; both = state drives the robot), --no-video,
    #          --no-robot, --episode N, --video-frames (decode with PyAV and log JPEG frames), --arm arm_right
    # the right-arm real_right output (6-D, one dataset with many episodes):
    python3 tools/mocap_dataset/rerun_view.py /media/farhan/.../real_right_clean_v1 --episode 5 --save /tmp/real_right_5.rrd

Needs rerun-sdk 0.26, pyarrow, numpy, and for the 3D part xacro + the colcon workspace
(``/opt/ros/humble`` and ``<repo>/install``; they are sourced in a subshell, nothing needs to be
sourced beforehand).  Mesh ``package://`` URIs are resolved by rerun's URDF loader through
AMENT_PREFIX_PATH, which this script fills in from that subshell.

Implementation notes
  * URDF: rerun's built-in loader (``rr.log_file_from_path``) logs the cell once, statically, as an
    entity tree ``/cell/<robot>/<root link>/<joint>/<child link>/...``.  Static data always wins
    over temporal data, so the loader's own joint transforms (the fixed origins) cannot be
    animated.  The origin of a revolute joint lives on the *joint* entity and the *link* entity
    below it carries no transform, so the angle q is logged on the link entity as
    a quaternion for (axis, q) (``Transform3D(quaternion=...)``): T_parent * T_origin(joint) * R(axis, q).
    No FK is written here.  The TCP path uses the existing numpy URDF FK in
    ``rebel_dual_arm_calibration.kinematics``.
  * Meshes: rerun's loader does not render Collada, so the ReBeL link ``.dae`` files are converted to
    ``.glb`` with trimesh (cached in ~/.cache/igus_rerun_meshes) and the expanded URDF is rewritten
    to point at them before loading.
  * Video: the left-arm mp4s are AV1, the real_right ones H.264.  The path used is AssetVideo (no Python decode); the viewer
    needs its native decoder (rerun-cli 0.26.2 reports ``Video features: av1``; H.264 needs the system ffmpeg).
    ``--video-frames`` is the fallback that decodes with PyAV and logs JPEG frames.  The .rrd embeds the whole mp4 files
    (one file holds several episodes), so it is large.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from lerobot_episode import Episode, load_episode  # noqa: E402

XACRO = REPO / "src" / "rebel_dual_table_description" / "urdf" / "rebel_dual_table.urdf.xacro"
ROS_SETUP = "/opt/ros/humble/setup.bash"
WS_SETUP = REPO / "install" / "setup.bash"
CELL_PREFIX = "cell"            # entity_path_prefix of the URDF loader
ARM = "arm_left_"
TCP_LINK = ARM + "xeg32_tool_tip"   # kinematic knife TCP (link6 -> igus_tool0 -> knife_c_tcp_point -> tool tip)
ROOT_LINK = "world"
VIDEO_ORDER = ["head", "wrist_left", "wrist_right", "knife"]


# --------------------------------------------------------------------------------------- URDF side
def _import_kinematics():
    try:
        from rebel_dual_arm_calibration import kinematics
    except ImportError:
        sys.path.insert(0, str(REPO / "src" / "rebel_dual_arm_calibration"))
        from rebel_dual_arm_calibration import kinematics
    return kinematics


def expand_cell_urdf(xacro_file: Path = XACRO, ros_setup: str = ROS_SETUP, ws_setup: Path = WS_SETUP) -> tuple[str, str]:
    """Run xacro on the cell description (default calibration file).  Returns (urdf text, AMENT_PREFIX_PATH)."""
    env = dict(os.environ, ROS_DOMAIN_ID="209", ROS_LOCALHOST_ONLY="1")
    src = f"source {shlex.quote(ros_setup)} >/dev/null && source {shlex.quote(str(ws_setup))} >/dev/null"
    try:
        urdf = subprocess.run(["bash", "-c", f"{src} && xacro {shlex.quote(str(xacro_file))}"], env=env,
                              check=True, capture_output=True, text=True).stdout
        ament = subprocess.run(["bash", "-c", f'{src} && printf %s "$AMENT_PREFIX_PATH"'], env=env,
                               check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"xacro failed (is ROS Humble at {ros_setup} and the workspace built at {ws_setup}?):\n{e.stderr}")
    return urdf, ament


MESH_CACHE = Path("~/.cache/igus_rerun_meshes").expanduser()
_DAE_RE = re.compile(r'filename="(package://([^/"]+)/([^"]+?\.dae))"', re.IGNORECASE)


def _resolve_package_file(pkg: str, rel: str, ament_prefix_path: str) -> Path:
    for prefix in ament_prefix_path.split(os.pathsep):
        f = Path(prefix) / "share" / pkg / rel
        if prefix and f.is_file():
            return f.resolve()
    raise FileNotFoundError(f"package://{pkg}/{rel} not found under AMENT_PREFIX_PATH")


def dae_to_glb(dae: Path, cache_dir: Path = MESH_CACHE) -> Path:
    """Convert a Collada mesh to .glb (rerun's URDF loader cannot render .dae).  Cached by path + mtime + size.

    The ReBeL .dae files are <unit meter="1"> and Z_UP, which is the URDF frame; trimesh writes the vertex
    values unchanged (no Y-up rotation) and rerun's glTF loader applies no axis conversion either."""
    dae = Path(dae).resolve()
    st = dae.stat()
    key = hashlib.sha1(f"{dae}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:12]
    out = cache_dir / f"{dae.stem}-{key}.glb"
    if not out.is_file():
        import trimesh
        scene = trimesh.load(str(dae), force="scene")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(scene.export(file_type="glb"))
        tmp.replace(out)
    return out


def rewrite_dae_meshes(urdf_text: str, ament_prefix_path: str, cache_dir: Path = MESH_CACHE) -> str:
    """Replace every ``package://pkg/x.dae`` mesh filename by the absolute path of its cached .glb."""
    def sub(m):
        return f'filename="{dae_to_glb(_resolve_package_file(m.group(2), m.group(3), ament_prefix_path), cache_dir)}"'
    return _DAE_RE.sub(sub, urdf_text)


def entity_paths(urdf, robot_name: str, prefix: str = CELL_PREFIX, root: str = ROOT_LINK) -> dict[str, str]:
    """Entity path rerun's URDF loader gives every link and joint: .../<root link>/<joint>/<child link>/..."""
    paths = {root: f"/{prefix}/{robot_name}/{root}" if prefix else f"/{robot_name}/{root}"}
    children: dict[str, list] = {}
    for j in urdf.joints.values():
        children.setdefault(j.parent, []).append(j)
    stack = [root]
    while stack:
        link = stack.pop()
        for j in children.get(link, []):
            paths[j.name] = f"{paths[link]}/{j.name}"
            paths[j.child] = f"{paths[j.name]}/{j.child}"
            stack.append(j.child)
    return paths


def joint_entities(urdf, paths: dict[str, str], arm: str = ARM) -> list[tuple[str, str, np.ndarray]]:
    """[(joint name, entity that carries its angle, unit axis)] for ``<arm>joint1..6``, in order.

    The angle goes on the child *link* entity (the joint entity holds the fixed origin, which is static)."""
    out = []
    for i in range(1, 7):
        j = urdf.joints[f"{arm}joint{i}"]
        if j.type not in ("revolute", "continuous"):
            raise ValueError(f"{j.name} is {j.type}")
        out.append((j.name, paths[j.child], np.asarray(j.axis, dtype=float)))
    return out


def joint_quaternions(angles: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """(N,) angles [rad] about a unit axis -> (N,4) xyzw quaternions (rerun's order): the joint-to-transform mapping."""
    half = np.asarray(angles, dtype=float) / 2.0
    return np.column_stack([np.outer(np.sin(half), axis), np.cos(half)])


def collision_entities(urdf_text: str, paths: dict[str, str]) -> list[str]:
    """The loader also logs <collision> meshes as ``<link>/collision_<i>``; hide them in the 3D view."""
    root = ET.fromstring(urdf_text)
    out = []
    for link in root.findall("link"):
        for i in range(len(link.findall("collision"))):
            if link.get("name") in paths:
                out.append(f"{paths[link.get('name')]}/collision_{i}")
    return out


def joint_limit_report(urdf_text: str, q: np.ndarray, arm: str = ARM) -> list[str]:
    root = ET.fromstring(urdf_text)
    msgs = []
    for k in range(6):
        j = root.find(f".//joint[@name='{arm}joint{k + 1}']")
        lim = None if j is None else j.find("limit")
        if lim is None or lim.get("lower") is None:
            continue
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        if q[:, k].min() < lo - 1e-6 or q[:, k].max() > hi + 1e-6:
            msgs.append(f"{arm}joint{k + 1}: data {q[:, k].min():+.2f}..{q[:, k].max():+.2f} rad outside URDF limits {lo:+.2f}..{hi:+.2f}")
    return msgs


# ------------------------------------------------------------------------------------ dataset side
def find_episode_root(path: Path, episode_dir: str | None) -> Path:
    path = Path(path).expanduser().resolve()
    if (path / "meta" / "info.json").is_file():
        return path
    subs = sorted(p for p in path.iterdir() if (p / "meta" / "info.json").is_file()) if path.is_dir() else []
    if not subs:
        raise SystemExit(f"{path}: not a LeRobot dataset and has no dataset sub-directories")
    names = [p.name for p in subs]
    if episode_dir is None:
        raise SystemExit(f"{path} holds {len(subs)} datasets; choose one with --episode-dir NAME. Available: {', '.join(names)}")
    if episode_dir not in names:
        raise SystemExit(f"--episode-dir {episode_dir!r} not found. Available: {', '.join(names)}")
    return path / episode_dir


def plot_channels(ep: Episode, source: str) -> tuple[list[str], list[str]]:
    """(channel names to plot, notes about skipped all-zero channels)."""
    srcs = {"state": [ep.state], "action": [ep.action], "both": [ep.state, ep.action]}[source]
    keep, skipped = [], []
    for k, name in enumerate(ep.names):
        (keep if any(np.any(np.abs(s[:, k]) > 1e-9) for s in srcs) else skipped).append(name)
    return keep, skipped


# ------------------------------------------------------------------------------------------ logging
def log_plots(rr, ep: Episode, channels: list[str], source: str, frames: np.ndarray) -> None:
    idx = [rr.TimeColumn("time", duration=ep.timestamp), rr.TimeColumn("frame", sequence=frames)]
    styles = {"state": ([31, 119, 180], "state"), "action": ([255, 127, 14], "action")}
    for name in channels:
        k = ep.names.index(name)
        for src in (["state", "action"] if source == "both" else [source]):
            color, label = styles[src]
            path = f"plots/{name}/{src}"
            rr.log(path, rr.SeriesLines(colors=[color], names=[label], widths=[1.5]), static=True)
            rr.send_columns(path, indexes=idx, columns=rr.Scalars.columns(scalars=(ep.state if src == "state" else ep.action)[:, k]))


def log_videos_asset(rr, ep: Episode, frames: np.ndarray) -> list[str]:
    names = []
    for short in VIDEO_ORDER:
        key = f"observation.images.{short}"
        if key not in ep.videos:
            continue
        mp4, from_ts = ep.videos[key]
        rr.log(f"video_assets/{short}", rr.AssetVideo(path=str(mp4)), static=True)
        for i, t in enumerate(ep.timestamp):
            rr.set_time("time", duration=float(t))
            rr.set_time("frame", sequence=int(frames[i]))
            rr.log(f"video/{short}", rr.VideoFrameReference(seconds=float(from_ts + t), video_reference=f"video_assets/{short}"))
        names.append(short)
    return names


def log_videos_frames(rr, ep: Episode, frames: np.ndarray) -> list[str]:
    import av
    names = []
    for short in VIDEO_ORDER:
        key = f"observation.images.{short}"
        if key not in ep.videos:
            continue
        mp4, from_ts = ep.videos[key]
        with av.open(str(mp4)) as c:
            for fr in c.decode(video=0):
                t = float(fr.pts * fr.time_base) - from_ts
                if t < -1e-6 or t > ep.timestamp[-1] + 1e-6:
                    continue
                rr.set_time("time", duration=t)
                rr.set_time("frame", sequence=int(round(t * ep.fps)))
                rr.log(f"video/{short}", rr.Image(fr.to_ndarray(format="rgb24")).compress(jpeg_quality=85))
        names.append(short)
    return names


def log_robot(rr, ep: Episode, source: str, frames: np.ndarray, arm: str = "arm_left") -> dict:
    """Load the cell URDF, animate one arm (``arm`` = 'arm_left' / 'arm_right'), draw its TCP path.  Returns TCP stats."""
    kin = _import_kinematics()
    urdf_text, ament = expand_cell_urdf()
    os.environ["AMENT_PREFIX_PATH"] = ament           # read by rerun's loader to resolve package:// URIs
    robot_name = ET.fromstring(urdf_text).get("name")
    urdf = kin.Urdf.parse(urdf_text)
    paths = entity_paths(urdf, robot_name)
    prefix = f"{arm}_"
    tcp_link = prefix + "xeg32_tool_tip"
    q = ep.arm(arm, "action" if source == "action" else "state")
    for m in joint_limit_report(urdf_text, q, prefix):
        print("note: " + m)

    with tempfile.TemporaryDirectory(prefix="cell_urdf_") as tmp:
        f = Path(tmp) / "rebel_dual_table.urdf"
        f.write_text(rewrite_dae_meshes(urdf_text, ament))
        rr.log_file_from_path(str(f), entity_path_prefix=CELL_PREFIX, static=True)

    joints = joint_entities(urdf, paths, prefix)
    for k, (name, ent, axis) in enumerate(joints):
        rr.send_columns(ent, indexes=[rr.TimeColumn("time", duration=ep.timestamp), rr.TimeColumn("frame", sequence=frames)],
                        columns=rr.Transform3D.columns(quaternion=joint_quaternions(q[:, k], axis)))

    chain = urdf.chain(ROOT_LINK, tcp_link)
    if chain.joint_names != [n for n, _, _ in joints]:
        raise SystemExit(f"unexpected chain joints {chain.joint_names}")
    tcp = chain.fk(q)[:, :3, 3]
    rr.log("tcp/path", rr.LineStrips3D([tcp], colors=[[255, 215, 0]], radii=[0.0015]), static=True)
    rr.send_columns("tcp/tip", indexes=[rr.TimeColumn("time", duration=ep.timestamp), rr.TimeColumn("frame", sequence=frames)],
                    columns=rr.Points3D.columns(positions=tcp))
    rr.log("tcp/tip", rr.Points3D.from_fields(radii=[0.007], colors=[[255, 40, 40]]), static=True)
    table_top = -0.001      # rebel_dual_table.urdf.xacro: table top surface in the world (= left base) frame
    return {"min": tcp.min(0), "max": tcp.max(0), "table_top": table_top, "tcp_link": tcp_link,
            "collisions": collision_entities(urdf_text, paths)}


def build_blueprint(rr, rrb, robot: bool, videos: list[str], channels: list[str], collisions: list[str]):
    top = []
    if robot:
        contents = [f"/{CELL_PREFIX}/**", "/tcp/**"] + [f"- {c}" for c in collisions]
        top.append(rrb.Spatial3DView(origin="/", contents=contents, name="Cell (world = left base)",
                                     background=[30, 30, 36]))
    top += [rrb.Spatial2DView(origin=f"/video/{v}", name=v) for v in videos]
    plots = [rrb.TimeSeriesView(origin=f"/plots/{c}", name=c) for c in channels]
    shares = ([2] if robot else []) + [1] * len(videos)
    rows, row_shares = [], []
    if top:
        rows.append(rrb.Horizontal(*top, column_shares=shares)), row_shares.append(3)
    if plots:
        rows.append(rrb.Grid(*plots, grid_columns=4 if len(plots) > 6 else 3)), row_shares.append(2)
    root = rrb.Vertical(*rows, row_shares=row_shares) if len(rows) > 1 else rows[0]
    return rrb.Blueprint(root, rrb.TimePanel(timeline="time"), rrb.SelectionPanel(state="collapsed"),
                         rrb.BlueprintPanel(state="collapsed"))


def view_episode(root: Path, episode: int = 0, source: str = "both", save: str | None = None,
                 no_video: bool = False, no_robot: bool = False, video_frames: bool = False, arm: str | None = None) -> dict:
    import rerun as rr
    import rerun.blueprint as rrb
    ep = load_episode(root, episode)
    arm = arm or ep.active_arm()
    frames = np.arange(len(ep.timestamp))
    channels, skipped = plot_channels(ep, source)
    print(f"{ep.root.name} ep {ep.episode_index}: '{ep.task}', {len(frames)} frames @ {ep.fps:g} fps, {ep.timestamp[-1]:.1f} s, arm {arm}")
    if skipped:
        print(f"note: skipped all-zero channels (not plotted): {', '.join(skipped)}")

    rr.init(f"mocap_{ep.root.name}_ep{ep.episode_index}", spawn=False)
    if save:
        rr.save(save)
    else:
        rr.spawn()
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    info = {"channels": channels, "skipped": skipped, "videos": [], "tcp": None}
    robot = {}
    if not no_robot:
        robot = log_robot(rr, ep, source, frames, arm)
        info["tcp"] = robot
        lo, hi = robot["min"], robot["max"]
        print(f"TCP ({robot['tcp_link']}) in world frame [m]: x {lo[0]:+.3f}..{hi[0]:+.3f}  y {lo[1]:+.3f}..{hi[1]:+.3f}  "
              f"z {lo[2]:+.3f}..{hi[2]:+.3f}  (table top z = {robot['table_top']:+.3f})")
    log_plots(rr, ep, channels, source, frames)
    if not no_video:
        info["videos"] = (log_videos_frames if video_frames else log_videos_asset)(rr, ep, frames)
    rr.send_blueprint(build_blueprint(rr, rrb, not no_robot, info["videos"], channels, robot.get("collisions", [])))
    if save:
        rr.disconnect()
        print(f"saved {save}")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="one episode dataset dir (has meta/info.json) or the parent dir holding several")
    ap.add_argument("--episode-dir", help="sub-directory name when DATASET is a parent dir")
    ap.add_argument("--episode", type=int, default=0, help="episode_index inside the dataset (default 0)")
    ap.add_argument("--source", choices=["state", "action", "both"], default="both",
                    help="series to plot; the robot follows 'action' for --source action, else 'state'")
    ap.add_argument("--arm", help="arm to animate (arm_left / arm_right; default: the arm that moves in the episode)")
    ap.add_argument("--save", metavar="OUT.rrd", help="write an .rrd instead of spawning the viewer")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-robot", action="store_true", help="skip the 3D cell model and TCP trail")
    ap.add_argument("--video-frames", action="store_true", help="fallback: decode video with PyAV and log JPEG frames")
    a = ap.parse_args(argv)
    view_episode(find_episode_root(a.dataset, a.episode_dir), a.episode, a.source, a.save,
                 a.no_video, a.no_robot, a.video_frames, a.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
