"""ROS side of the wrist-camera (eye-in-hand) panel: wrist image / camera_info, TF lookups at an image stamp, the
planning-scene objects of the safety scene and a static TF for live checking.

Nothing here commands a robot.  The only things this module publishes are (a) planning-scene diffs (ADD / REMOVE of the four
``calib_*`` boxes, via /apply_planning_scene) and (b) one optional static TF ``link6 -> wrist_camera_link`` (the camera driver
owns wrist_camera_link -> ... -> optical frame; the optical frame never gets a second parent).
Blocking calls (``apply``, ``scene_boxes``) must be made from a worker thread, never from the Tornado loop.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..safety_scene import BoxSpec, OUR_IDS, SceneRefused, box_from_object
from ..transforms import R_to_quat

SERVICE_TIMEOUT_S = 5.0
FRAME_KEEP = 24
LOG = logging.getLogger("calibration_ui.wrist_ros")


class WristRos:
    def __init__(self, node, wrist_image_topic: str, wrist_info_topic: str, bridge, base_frame: str = "base_link",
                 tip_frame: str = "link6", scene_optical_frame: str = "camera_color_optical_frame",
                 child_frame: str = "wrist_camera_link", optical_frame: str = "wrist_camera_color_optical_frame"):
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from rclpy.duration import Duration
        from tf2_ros import Buffer, TransformListener
        from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
        self.node, self.bridge = node, bridge
        self.base_frame, self.tip_frame, self.scene_optical_frame = base_frame, tip_frame, scene_optical_frame
        self.child_frame, self.optical_frame = child_frame, optical_frame
        self._lock = threading.Lock()
        self._frames: "collections.deque" = collections.deque(maxlen=FRAME_KEEP)    # (stamp_s, rx_mono, Image)
        self._info: Optional[CameraInfo] = None
        self._info_rx: Optional[float] = None
        self._seq = 0
        cb = ReentrantCallbackGroup()
        node.create_subscription(Image, wrist_image_topic, self._on_image, qos_profile_sensor_data, callback_group=cb)
        node.create_subscription(CameraInfo, wrist_info_topic, self._on_info, qos_profile_sensor_data, callback_group=cb)
        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, node, spin_thread=False)
        self._cli_get = node.create_client(GetPlanningScene, "/get_planning_scene", callback_group=cb)
        self._cli_apply = node.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=cb)
        self._static = StaticTransformBroadcaster(node)
        self._applied: set = set()

    # ---- subscriptions ----------------------------------------------------------------------------
    def _on_image(self, msg) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lag = self.node.get_clock().now().nanoseconds * 1e-9 - stamp if stamp > 0 else None
        with self._lock:
            self._seq += 1
            self._frames.append((stamp, time.monotonic(), msg, self._seq, lag))

    def _on_info(self, msg) -> None:
        with self._lock:
            self._info, self._info_rx = msg, time.monotonic()

    def wrist_health(self) -> dict:
        with self._lock:
            rx = self._frames[-1][1] if self._frames else None
            info_rx = self._info_rx
        now = time.monotonic()
        return {"image_age_s": None if rx is None else now - rx, "camera_info": info_rx is not None and now - info_rx < 2.0}

    def wrist_info(self) -> Optional[dict]:
        with self._lock:
            i = self._info
        if i is None or not any(i.k):
            return None
        return {"K": np.array(i.k, dtype=float).reshape(3, 3), "dist": np.array(i.d, dtype=float),
                "size": (int(i.width), int(i.height)), "frame_id": str(i.header.frame_id)}

    def wrist_frames(self, since_mono: float = 0.0) -> List[dict]:
        """Newest-last list of frames received after ``since_mono`` (time.monotonic): {stamp, rx, seq, bgr}."""
        from ..recorder_node import image_msg_to_bgr
        with self._lock:
            items = [f for f in self._frames if f[1] >= since_mono]
        out = []
        for stamp, rx, msg, seq, lag in items:
            try:
                bgr = image_msg_to_bgr(msg)
            except ValueError as e:
                LOG.warning("wrist image not decodable: %s", e)
                continue
            out.append({"stamp": stamp if stamp > 0 else rx, "rx": rx, "seq": seq, "bgr": bgr,
                        "frame_id": str(msg.header.frame_id), "lag_s": lag})
        return out

    def latest_wrist_frame(self) -> Optional[dict]:
        fr = self.wrist_frames(0.0)
        return fr[-1] if fr else None

    # ---- scene camera (static) ------------------------------------------------------------------
    def scene_frame(self):
        """(bgr, stamp_s, rx_mono) of the newest scene-camera frame or None."""
        item = self.bridge.latest_frame()
        if item is None:
            return None
        bgr, seq, rx = item
        return bgr, seq, rx

    def scene_info(self) -> Optional[dict]:
        i = getattr(self.bridge, "latest_info", lambda: None)()
        if i is None or not any(i.k):
            return None
        return {"K": np.array(i.k, dtype=float).reshape(3, 3), "dist": np.array(i.d, dtype=float),
                "size": (int(i.width), int(i.height)), "frame_id": str(i.header.frame_id)}

    # ---- TF ------------------------------------------------------------------------------------------
    def _lookup(self, parent: str, child: str, stamp_s: Optional[float]):
        from rclpy.time import Time
        try:
            t = Time() if stamp_s is None else Time(seconds=int(stamp_s), nanoseconds=int((stamp_s % 1.0) * 1e9))
            tf = self._tf_buffer.lookup_transform(parent, child, t)
        except Exception as e:  # noqa: BLE001 - tf2 raises several types
            return None, f"TF {parent}->{child} unavailable: {e}"
        return _tf_T(tf), ""

    def tf_base_link6(self, stamp_s: Optional[float]):
        """(T_base_link6 at the stamp or None, reason).  stamp None = latest."""
        return self._lookup(self.base_frame, self.tip_frame, stamp_s)

    def tf_age_s(self) -> Optional[float]:
        """Age of the newest base->link6 TF (seconds), None if unknown."""
        try:
            from rclpy.time import Time
            tf = self._tf_buffer.lookup_transform(self.base_frame, self.tip_frame, Time())
        except Exception:  # noqa: BLE001
            return None
        st = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        return None if st <= 0 else self.node.get_clock().now().nanoseconds * 1e-9 - st

    def tf_camera_link_optical(self, optical_frame: Optional[str] = None):
        """(T_wrist_camera_link_optical or None, reason): the camera driver's own transform from wrist_camera_link to the optical
        frame the images are in (identity on the real camera)."""
        return self._lookup(self.child_frame, optical_frame or self.optical_frame, None)

    def tf_base_scene_optical(self):
        return self._lookup(self.base_frame, self.scene_optical_frame, None)

    def now_s(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    # ---- planning scene -----------------------------------------------------------------------------
    def _call(self, client, request, timeout_s: float = SERVICE_TIMEOUT_S):
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=1.0):
            raise SceneRefused(f"{client.srv_name} is not available (is move_group running?)")
        done = threading.Event()
        fut = client.call_async(request)
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout_s):
            try:
                client.remove_pending_request(fut)
            except Exception:  # noqa: BLE001
                pass
            raise SceneRefused(f"{client.srv_name} did not answer within {timeout_s:.0f} s")
        res = fut.result()
        if res is None:
            raise SceneRefused(f"{client.srv_name} failed")
        return res

    def scene_objects(self) -> Dict[str, object]:
        """{id: CollisionObject} of the whole world in move_group's planning scene."""
        from moveit_msgs.msg import PlanningSceneComponents
        from moveit_msgs.srv import GetPlanningScene
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.WORLD_OBJECT_NAMES
                                     | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        res = self._call(self._cli_get, req)
        return {o.id: o for o in res.scene.world.collision_objects}

    def scene_boxes(self, names: Sequence[str]) -> Dict[str, Optional[BoxSpec]]:
        objs = self.scene_objects()
        out: Dict[str, Optional[BoxSpec]] = {}
        for n in names:
            out[n] = box_from_object(objs[n]) if n in objs else None
        return out

    def scene_has(self, ids: Sequence[str]) -> Tuple[bool, List[str]]:
        have = set(self.scene_objects())
        missing = [i for i in ids if i not in have]
        return (not missing), missing

    def apply_boxes(self, specs: Sequence[BoxSpec]) -> None:
        """ADD (replace) the boxes; idempotent."""
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from moveit_msgs.srv import ApplyPlanningScene
        from shape_msgs.msg import SolidPrimitive
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.robot_state.is_diff = True
        for s in specs:
            co = CollisionObject()
            co.header.frame_id = s.frame
            co.id = s.id
            co.operation = CollisionObject.ADD
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(v) for v in s.size]
            co.primitives = [prim]
            p = Pose()
            co.primitive_poses = [p]
            p.position.x, p.position.y, p.position.z = (float(v) for v in s.T[:3, 3])
            q = R_to_quat(s.T[:3, :3])
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
            req.scene.world.collision_objects.append(co)
        res = self._call(self._cli_apply, req)
        if not res.success:
            raise SceneRefused("MoveIt rejected the safety scene (/apply_planning_scene success=false)")
        self._applied.update(s.id for s in specs)

    def remove_boxes(self, ids: Sequence[str] = OUR_IDS) -> None:
        from moveit_msgs.msg import CollisionObject
        from moveit_msgs.srv import ApplyPlanningScene
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.robot_state.is_diff = True
        for i in ids:
            co = CollisionObject()
            co.header.frame_id = self.base_frame
            co.id = i
            co.operation = CollisionObject.REMOVE
            req.scene.world.collision_objects.append(co)
        res = self._call(self._cli_apply, req)
        if not res.success:
            raise SceneRefused("MoveIt rejected the removal (/apply_planning_scene success=false)")
        self._applied.difference_update(ids)

    # ---- static TF (live check only) ---------------------------------------------------------------------
    def publish_static_tf(self, T: np.ndarray) -> None:
        from geometry_msgs.msg import TransformStamped
        ts = TransformStamped()
        ts.header.stamp = self.node.get_clock().now().to_msg()
        ts.header.frame_id = self.tip_frame
        ts.child_frame_id = self.child_frame
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = (float(v) for v in T[:3, 3])
        q = R_to_quat(T[:3, :3])
        r = ts.transform.rotation
        r.x, r.y, r.z, r.w = (float(v) for v in q)
        self._static.sendTransform(ts)


def _tf_T(tf) -> np.ndarray:
    from ..recorder_node import tf_msg_to_T
    return tf_msg_to_T(tf)
