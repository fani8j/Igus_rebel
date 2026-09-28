#!/usr/bin/env python3
"""Capture an MTC plan and render mock H-cut bag/plan as two-view offline videos.

Use after sourcing the ROS 2 underlay and workspace overlay. No live robot is used
by the render mode. The capture/wait modes are invoked by mock_hcut_demo.sh.
"""

import argparse
import json
import math
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro"
ARM = [f"joint{i}" for i in range(1, 7)]
LIGHT = np.array([0.5, 0.5, math.sqrt(0.5)])
ROBOT_COLOR = (0.7, 0.7, 0.7)
GRIPPER_COLOR = (0.3, 0.3, 0.3)
TOOL_COLOR = (1.0, 0.6, 0.0)


def duration(seconds):
    return seconds.sec + seconds.nanosec * 1e-9


def trajectory_from_solution(solution, stages):
    segments = []
    joint_names = None
    for part in solution.sub_trajectory:
        traj = part.trajectory.joint_trajectory
        if not traj.points:
            continue  # scene edits carry no robot motion
        names = list(traj.joint_names)
        if not all(name in names for name in ARM):
            continue
        if joint_names is None:
            joint_names = names
        if set(names) != set(joint_names):
            raise ValueError("MTC segments use inconsistent joint sets")
        indices = [names.index(name) for name in joint_names]
        segments.append({
            "stage_name": stages.get(part.info.stage_id, f"stage {part.info.stage_id}"),
            "trajectory": [
                {"positions": [float(point.positions[i]) for i in indices],
                 "time_from_start": duration(point.time_from_start)}
                for point in traj.points
            ],
        })
    if not segments:
        raise ValueError("MTC Solution contains no arm trajectories")
    return {"joint_names": joint_names, "trajectories": segments,
            "task_id": solution.task_id}


def capture_plan(out, solution_topic, description_topic):
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from moveit_task_constructor_msgs.msg import Solution, TaskDescription

    rclpy.init()
    node = rclpy.create_node("mock_hcut_plan_capture")
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    descriptions = {}
    result = []

    def description(msg):
        descriptions[msg.task_id] = {stage.id: stage.name for stage in msg.stages}

    def solution(msg):
        try:
            data = trajectory_from_solution(msg, descriptions.get(msg.task_id, {}))
            temp = out / "planned_trajectory.json.tmp"
            temp.write_text(json.dumps(data, indent=2))
            temp.replace(out / "planned_trajectory.json")
            result.append(True)
        except ValueError as exc:
            node.get_logger().error(str(exc))

    if description_topic:
        node.create_subscription(TaskDescription, description_topic, description, qos)
    # Volatile subscription ignores latched solutions from previous MTC tasks;
    # only the solution published after this capture starts belongs to this run.
    live_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE)
    node.create_subscription(Solution, solution_topic, solution, live_qos)
    (out / "plan_capture.ready").touch()
    try:
        deadline = time.monotonic() + 240
        while not result and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not result:
            raise TimeoutError("no executable MTC solution received")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def wait_stationary(out):
    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("mock_hcut_stationary_watch")
    prev = None
    moving = False
    stopped_since = None
    done = False

    def joint_state(msg):
        nonlocal prev, moving, stopped_since, done
        if not all(j in msg.name for j in ARM):
            return
        positions = np.array([msg.position[msg.name.index(j)] for j in ARM])
        stamp = time.monotonic()
        if prev is not None:
            dt = stamp - prev[0]
            speed = np.max(np.abs(positions - prev[1])) / dt if dt > 0 else 0
            if speed >= 0.01:
                moving = True
                stopped_since = None
            elif moving:
                if stopped_since is None:
                    stopped_since = stamp
                if stamp - stopped_since >= 3:
                    done = True
        prev = (stamp, positions)

    node.create_subscription(JointState, "/joint_states", joint_state, 10)
    (out / "stationary.ready").touch()
    try:
        deadline = time.monotonic() + 180
        while not done and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        (out / "stationary_result.txt").write_text(
            "stationary after motion\n" if done else "timeout before stationary\n")
        if not done:
            raise TimeoutError("no 3-second stationary interval after motion within 180s")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def read_bag(directory):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(directory / "bag"), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                            output_serialization_format="cdr"))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    states, boxes, stages = [], {}, {}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic not in types:
            continue
        if topic not in ("/joint_states", "/collision_object",
                         "/box_motion/camera_pole_collision", "/monitored_planning_scene") \
                and topic != "/description":
            continue
        msg = deserialize_message(raw, types[topic])
        if topic == "/joint_states" and all(j in msg.name for j in ARM):
            states.append((stamp * 1e-9, [msg.position[msg.name.index(j)] for j in ARM]))
        elif topic in ("/collision_object", "/box_motion/camera_pole_collision"):
            add_box(boxes, msg)
        elif topic == "/monitored_planning_scene":
            for obj in msg.world.collision_objects:
                add_box(boxes, obj)
        elif topic == "/description":
            stages[msg.task_id] = {stage.id: stage.name for stage in msg.stages}
    return states, boxes, stages


def add_box(boxes, obj):
    if obj.id not in ("detected_carton", "camera_pole"):
        return
    if obj.operation == obj.REMOVE:
        boxes.pop(obj.id, None)
    elif obj.primitives and obj.primitive_poses:
        # The wrapper URDF defines world -> base_link as an identity fixed joint.
        if obj.header.frame_id.lstrip("/") not in ("base_link", "world"):
            raise ValueError(f"Unsupported collision frame: {obj.header.frame_id}")
        primitive, pose = obj.primitives[0], obj.primitive_poses[0]
        if primitive.type == primitive.BOX:
            # PlanningScene may migrate the primitive pose into obj.pose.
            boxes[obj.id] = (np.array(primitive.dimensions),
                             pose_matrix(obj.pose) @ pose_matrix(pose))


def fallback_carton(out):
    snapshot = json.loads((out / "replay_snapshot.json").read_text())
    corners = np.array([[snapshot[key][axis] for axis in "xyz"] for key in
                        ("front_left", "front_right", "back_left", "back_right")])
    front = corners[1] - corners[0]
    back = corners[2] - corners[0]
    normal = np.cross(front, back)
    normal /= np.linalg.norm(normal)
    x = front / np.linalg.norm(front)
    y = np.cross(normal, x)
    center = corners.mean(axis=0) - normal * 0.1
    return (np.array([np.linalg.norm(front), np.linalg.norm(back), 0.2]),
            center, np.column_stack((x, y, normal)))


def rotation_rpy(rpy):
    roll, pitch, yaw = rpy
    cx, sx, cy, sy, cz, sz = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                     [-sy, cy*sx, cy*cx]])


def transform(xyz=None, rotation=None):
    t = np.eye(4)
    t[:3, 3] = np.zeros(3) if xyz is None else xyz
    t[:3, :3] = np.eye(3) if rotation is None else rotation
    return t


def origin(element):
    if element is None:
        return np.eye(4)
    return transform(np.fromstring(element.get("xyz", "0 0 0"), sep=" "),
                     rotation_rpy(np.fromstring(element.get("rpy", "0 0 0"), sep=" ")))


def quaternion_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = x*x + y*y + z*z + w*w
    if norm < 1e-12:
        return np.eye(3)
    s = 2 / norm
    return np.array([[1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
                     [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
                     [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]])


def pose_matrix(pose):
    return transform(np.array([pose.position.x, pose.position.y, pose.position.z]),
                     quaternion_matrix(pose.orientation))


def triangles_box(size):
    return trimesh.creation.box(extents=size).triangles


def load_robot():
    xml = subprocess.check_output(["xacro", str(URDF), "hardware_protocol:=mock_hardware",
                                   "include_ros2_control:=false"], text=True)
    root = ET.fromstring(xml)
    children = {}
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        axis = np.fromstring(j.find("axis").get("xyz"), sep=" ") if j.find("axis") is not None else np.array([1., 0, 0])
        children.setdefault(parent, []).append((child, j.get("name"), j.get("type"), origin(j.find("origin")), axis))
    visuals = []
    mesh_cache = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh = geometry.find("mesh")
            if mesh is not None:
                path = mesh.get("filename").replace("package://igus_rebel_description/",
                    str(ROOT / "src/igus_rebel_description") + "/")
                if path not in mesh_cache:
                    model = trimesh.load(path, force="mesh", process=False)
                    # CAD imports contain thousands of isolated tiny patches;
                    # their simplifier cannot reach the requested face budget.
                    # Hull the arm housings for a connected shaded silhouette;
                    # keep detailed gripper/tool surfaces for the carton close-up.
                    if not link_name.startswith("xeg32_"):
                        model = model.convex_hull
                    elif len(model.faces) > 500:
                        model = model.simplify_quadric_decimation(face_count=500)
                    mesh_cache[path] = model.triangles
                tris = mesh_cache[path]
                scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                if not np.all(scale == 1):
                    tris = tris * scale
            elif geometry.find("box") is not None:
                tris = triangles_box(np.fromstring(geometry.find("box").get("size"), sep=" "))
            else:
                continue
            color = ROBOT_COLOR if not link_name.startswith("xeg32_") else GRIPPER_COLOR
            if link_name in ("xeg32_knife_handpiece", "xeg32_knife_holder"):
                color = TOOL_COLOR
            normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            visuals.append((link_name, origin(visual.find("origin")), tris, normals, color))
    return children, visuals


def fk(children, positions):
    poses = {"world": np.eye(4)}
    queue = ["world"]
    while queue:
        parent = queue.pop()
        for child, name, kind, joint_origin, axis in children.get(parent, []):
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                angle = positions.get(name, 0)
                axis = axis / np.linalg.norm(axis)
                cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                motion[:3, :3] = np.eye(3) + math.sin(angle)*cross + (1-math.cos(angle))*(cross @ cross)
            elif kind == "prismatic":
                motion[:3, 3] = axis * positions.get(name, 0)
            poses[child] = poses[parent] @ joint_origin @ motion
            queue.append(child)
    if "xeg32_tool_tip" not in poses:
        raise ValueError("URDF has no xeg32_tool_tip transform")
    return poses


def box_triangles(spec):
    size, pose = spec
    return triangles_box(size) @ pose[:3, :3].T + pose[:3, 3]


def plan_samples(plan):
    names = plan["joint_names"]
    if not all(j in names for j in ARM):
        raise ValueError("planned trajectory lacks the six arm joints")
    samples = []
    elapsed = 0.0
    for stage in plan["trajectories"]:
        points = stage["trajectory"]
        for point in points:
            samples.append((elapsed + point["time_from_start"],
                            [point["positions"][names.index(j)] for j in ARM], stage["stage_name"]))
        if points:
            elapsed += points[-1]["time_from_start"]
    if not samples:
        raise ValueError("no planned positions")
    return samples


def sample_frames(samples, fps, max_seconds):
    start, end = samples[0][0], samples[-1][0]
    count = min(max(2, math.ceil((end-start)*fps)+1), max(2, int(max_seconds*fps)))
    indices = np.searchsorted([s[0] for s in samples], np.linspace(start, end, count), side="right") - 1
    return [samples[max(0, i)] for i in indices]


def shaded_faces(normals, color, alpha, rotation=None):
    direction = normals if rotation is None else normals @ rotation.T
    light = np.maximum(0.3, direction @ LIGHT)
    return np.column_stack((light[:, None] * color, np.full(len(normals), alpha)))


def execution_stages(states, planned):
    # The mock controller follows the same waypoints at a different clock rate.
    # Joint-space arc length aligns recorded motion to MTC stages; idle samples
    # keep the preceding stage instead of advancing on wall-clock time.
    plan_joints = np.array([p for _, p, _ in planned])
    actual_joints = np.array([p for _, p in states])
    plan_distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(plan_joints, axis=0), axis=1))]
    actual_distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(actual_joints, axis=0), axis=1))]
    stage_indices = np.searchsorted(plan_distance, actual_distance, side="right") - 1
    return [(t, p, planned[min(int(i), len(planned)-1)][2])
            for (t, p), i in zip(states, stage_indices)]


def render_video(kind, samples, boxes, children, visuals, output, fps, max_seconds):
    frames = sample_frames(samples, fps, max_seconds)
    poses = [fk(children, dict(zip(ARM, state[1]))) for state in frames]
    path = np.array([pose["xeg32_tool_tip"][:3, 3] for pose in poses])
    fixed = []
    for name, color, alpha in (("detected_carton", (0.6, 0.4, 0.2), 0.7),
                               ("camera_pole", (0.2, 0.4, 1.0), 0.3)):
        triangles = box_triangles(boxes[name])
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        fixed.append((name, triangles, shaded_faces(normals, color, alpha)))

    home = fk(children, dict.fromkeys(ARM, 0.0))
    bounds = [path, *[tri.reshape(-1, 3) for _, tri, _ in fixed]]
    for link, visual_origin, triangles, _, _ in visuals:
        matrix = home[link] @ visual_origin
        bounds.append(triangles.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3])
    extent = np.vstack(bounds)
    center = (extent.max(axis=0) + extent.min(axis=0)) / 2
    radius = max((extent.max(axis=0) - extent.min(axis=0)) / 2) + 0.1
    carton_center = boxes["detected_carton"][1][:3, 3]

    fig = plt.figure(figsize=(13, 6), dpi=100)
    views = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
    for ax, elev, azim, focus, half_width, label in (
            (views[0], 30, 120, center, radius, "Isometric | workspace"),
            (views[1], 25, -55, carton_center, 0.22, "Close-up | carton")):
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        ax.set(xlim=(focus[0]-half_width, focus[0]+half_width),
               ylim=(focus[1]-half_width, focus[1]+half_width),
               zlim=(focus[2]-half_width, focus[2]+half_width),
               xlabel="X / m", ylabel="Y / m", zlabel="Z / m")
        ax.set_box_aspect((1, 1, 1), zoom=1.14)
        ax.set_title(label, fontsize=11)
        ax.tick_params(labelsize=8, pad=0)
        for name, tri, colors in fixed:
            ax.add_collection3d(Poly3DCollection(tri, facecolors=colors, edgecolors="none"))
            if name == "detected_carton" or ax is views[0]:
                label_pos = boxes[name][1][:3, 3].copy()
                label_pos[2] = min(label_pos[2] + boxes[name][0][2]/2 + 0.02,
                                   focus[2] + half_width - 0.02)
                ax.text(*label_pos, "carton" if name == "detected_carton" else "camera_pole",
                        color="#513415" if name == "detected_carton" else "#1944aa",
                        fontsize=10, fontweight="bold")
        for axis, color in enumerate(("red", "green", "blue")):
            end = np.zeros(3)
            end[axis] = 0.08
            ax.plot([0, end[0]], [0, end[1]], [0, end[2]], color=color, linewidth=2.4)
            if ax is views[0]:
                ax.text(*end, "XYZ"[axis], color=color, fontsize=9)

    collections = [[ax.add_collection3d(Poly3DCollection([], edgecolors="none"))
                    for _ in visuals] for ax in views]
    traces = [ax.plot([], [], [], color="green", linewidth=2.2)[0] for ax in views]
    # Matplotlib's 3D circular depth-shaded markers read as spheres. Scale each
    # diameter to 0.02 m at the focus plane of its orthographic camera.
    markers = [ax.scatter([], [], [], color="red", depthshade=True,
                          s=(fig.dpi * 3.8 * 0.02 / (2 * half_width) * 72 / fig.dpi)**2)
               for ax, half_width in ((views[0], radius), (views[1], 0.22))]
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.87, wspace=0.01)
    width, height = fig.canvas.get_width_height()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
           "-i", "-", "-an", "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(output / f"{kind}.mp4")]
    started = time.monotonic()
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            for index, (frame, pose) in enumerate(zip(frames, poses)):
                t, _, stage = frame
                title = "Planned (MoveIt)" if kind == "plan" else "Mock execution (joint_states)"
                fig.suptitle(f"{title} | Elapsed: {t - samples[0][0]:.1f}s | Stage: {stage}", fontsize=12)
                transformed = []
                for link, visual_origin, triangles, normals, color in visuals:
                    matrix = pose[link] @ visual_origin
                    transformed.append((triangles @ matrix[:3, :3].T + matrix[:3, 3],
                                        shaded_faces(normals, color, 0.9, matrix[:3, :3])))
                for collection, trace, marker in zip(collections, traces, markers):
                    for coll, (verts, colors) in zip(collection, transformed):
                        coll.set_verts(verts)
                        coll.set_facecolor(colors)
                    trace.set_data(path[:index+1, 0], path[:index+1, 1])
                    trace.set_3d_properties(path[:index+1, 2])
                    marker._offsets3d = (path[index:index+1, 0], path[index:index+1, 1], path[index:index+1, 2])
                fig.canvas.draw()
                encoder.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
            fig.savefig(output / f"{kind}_final.png")
        if encoder.returncode:
            raise RuntimeError(f"ffmpeg failed: {encoder.returncode}")
    finally:
        plt.close(fig)
    print(f"{kind}: {len(frames)} frames, source {samples[-1][0]-samples[0][0]:.1f} s, render {time.monotonic()-started:.1f} s", flush=True)



def render(out, output, fps, max_seconds):
    plan = json.loads((out / "planned_trajectory.json").read_text())
    states, boxes, descriptions = read_bag(out)
    if "detected_carton" not in boxes:
        print("Carton absent in bag; using replay snapshot geometry")
        size, center, rotation = fallback_carton(out)
        boxes["detected_carton"] = (size, transform(center, rotation))
    if "camera_pole" not in boxes:
        raise ValueError("bag has no camera_pole collision object")
    if not states:
        raise ValueError("bag has no six-joint states")
    start_file = out / "execution_start.txt"
    if not start_file.exists():
        raise ValueError("execution_start.txt absent: no executed motion to render")
    start = float(start_file.read_text())
    planned = plan_samples(plan)
    executed_states = [(t, p) for t, p in states if t >= start]
    executed = execution_stages(executed_states, planned)
    if len(executed) < 2:
        raise ValueError("bag has no execution joint-state sequence")
    print(f"Bag: {len(states)} joint states; boxes: {', '.join(sorted(boxes))}; "
          f"MTC stage descriptions: {sum(map(len, descriptions.values()))}")
    children, visuals = load_robot()
    output.mkdir(parents=True, exist_ok=True)
    render_video("plan", planned, boxes, children, visuals, output, fps, max_seconds)
    render_video("execution", executed, boxes, children, visuals, output, fps, max_seconds)
    for name in ("plan", "execution"):
        subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames,duration,width,height",
                        "-of", "default=noprint_wrappers=1", str(output / f"{name}.mp4")], check=True)
        print(f"Final frame: {output / (name + '_final.png')} ({(output / (name + '_final.png')).stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--capture-plan", type=Path, metavar="DIRECTORY")
    parser.add_argument("--solution-topic")
    parser.add_argument("--description-topic", default="")
    parser.add_argument("--wait-stationary", type=Path, metavar="DIRECTORY")
    args = parser.parse_args()
    if args.capture_plan:
        capture_plan(args.capture_plan, args.solution_topic, args.description_topic)
    elif args.wait_stationary:
        wait_stationary(args.wait_stationary)
    else:
        if not args.directory or args.fps <= 0 or args.max_seconds <= 0:
            parser.error("directory, positive --fps and --max-seconds required")
        render(args.directory, args.output_dir or args.directory, args.fps, args.max_seconds)


if __name__ == "__main__":
    main()
