"""ROS side of the calibration UI: subscriptions, recorder service clients and a cached recorder status.

Everything here is thread-safe and blocking calls (``trigger`` / ``set_parameters``) must be run from a worker thread,
never from the Tornado loop.  The bridge never publishes anything: robot motion is the jog module's job alone.
"""
import json
import math
import threading
import time
from typing import Dict, Optional, Tuple

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

STATUS_PERIOD_S = 1.0
STATUS_STALE_S = 4.0       # the recorder counts as "available" while a status_json answer is at most this old
IMAGE_OK_AGE_S = 2.0
TRIGGERS = ("capture", "capture_touch", "capture_intrinsics")


def _encode(value) -> ParameterValue:
    if isinstance(value, bool):
        return ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)
    if isinstance(value, int):
        return ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value)
    if isinstance(value, float):
        return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)
    if isinstance(value, str):
        return ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
    raise ValueError(f"unsupported parameter value {value!r}")


class RosBridge:
    def __init__(self, node, image_topic: str, camera_info_topic: str, recorder: str = "dual_arm_calibration_recorder"):
        self.node = node
        self.recorder = recorder.strip("/")
        self._lock = threading.Lock()
        self._image: Optional[Image] = None
        self._image_rx: Optional[float] = None
        self._image_seq = 0
        self._decoded: Tuple[int, object] = (-1, None)
        self._info_rx: Optional[float] = None
        self._info: Optional[CameraInfo] = None
        self._status: Optional[dict] = None
        self._status_rx: Optional[float] = None
        self._status_pending: Optional[float] = None
        group = ReentrantCallbackGroup()
        node.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data, callback_group=group)
        node.create_subscription(CameraInfo, camera_info_topic, self._on_info, qos_profile_sensor_data,
                                 callback_group=group)
        self._triggers = {n: node.create_client(Trigger, f"/{self.recorder}/{n}", callback_group=group)
                          for n in TRIGGERS}
        self._status_client = node.create_client(Trigger, f"/{self.recorder}/status_json", callback_group=group)
        self._params_client = node.create_client(SetParameters, f"/{self.recorder}/set_parameters",
                                                 callback_group=group)
        node.create_timer(STATUS_PERIOD_S, self._poll_status, callback_group=group)

    # ---- subscriptions ---------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        with self._lock:
            self._image, self._image_rx = msg, time.monotonic()
            self._image_seq += 1

    def _on_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._info_rx = time.monotonic()
            self._info = msg

    def latest_info(self) -> Optional[CameraInfo]:
        """The newest camera_info message (scene camera) or None."""
        with self._lock:
            return self._info

    def latest_frame(self):
        """(bgr ndarray, seq, rx_monotonic) of the newest image, or None.  Decoded on demand, cached per seq."""
        with self._lock:
            msg, rx, seq = self._image, self._image_rx, self._image_seq
            cached = self._decoded
        if msg is None:
            return None
        if cached[0] != seq:
            from ..recorder_node import image_msg_to_bgr
            try:
                bgr = image_msg_to_bgr(msg)
            except ValueError as e:
                self.node.get_logger().warn(f"image not decodable: {e}", throttle_duration_sec=10.0)
                return None
            with self._lock:
                self._decoded = (seq, bgr)
            cached = (seq, bgr)
        return cached[1], seq, rx

    # ---- recorder status (cached, polled) ----------------------------------------------------
    def _poll_status(self) -> None:
        now = time.monotonic()
        if self._status_pending is not None and now - self._status_pending < 5.0:
            return
        if not self._status_client.service_is_ready():
            return
        self._status_pending = now
        fut = self._status_client.call_async(Trigger.Request())
        fut.add_done_callback(self._on_status)

    def _on_status(self, fut) -> None:
        self._status_pending = None
        try:
            r = fut.result()
            if r is not None and r.success:
                status = json.loads(r.message)
                with self._lock:
                    self._status, self._status_rx = status, time.monotonic()
        except Exception as e:  # noqa: BLE001 - a bad answer just leaves the old status to age out
            self.node.get_logger().warn(f"recorder status_json: {e}", throttle_duration_sec=10.0)

    def recorder_status(self) -> Optional[dict]:
        with self._lock:
            if self._status is None or time.monotonic() - self._status_rx > STATUS_STALE_S:
                return None
            return self._status

    def health(self) -> dict:
        now = time.monotonic()
        with self._lock:
            age = None if self._image_rx is None else now - self._image_rx
            size = None if self._image is None else [int(self._image.width), int(self._image.height)]
            info_ok = self._info_rx is not None and now - self._info_rx < IMAGE_OK_AGE_S
            rec_ok = self._status_rx is not None and now - self._status_rx <= STATUS_STALE_S
        return {"image": {"ok": age is not None and age < IMAGE_OK_AGE_S, "age_s": age, "size": size},
                "camera_info": bool(info_ok), "recorder": bool(rec_ok)}

    # ---- blocking service calls (worker thread) ------------------------------------------------
    def _call(self, client, request, timeout_s: float):
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=1.0):
            return None, "recorder service not available"
        done = threading.Event()
        fut = client.call_async(request)
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout_s):
            try:
                client.remove_pending_request(fut)
            except Exception:  # noqa: BLE001
                pass
            return None, f"recorder did not answer within {timeout_s:.0f} s"
        try:
            return fut.result(), ""
        except Exception as e:  # noqa: BLE001
            return None, f"service call failed: {e}"

    def trigger(self, name: str, timeout_s: float = 15.0) -> Tuple[bool, str]:
        if name not in self._triggers:
            raise ValueError(f"unknown recorder service '{name}'")
        resp, err = self._call(self._triggers[name], Trigger.Request(), timeout_s)
        if resp is None:
            return False, err
        return bool(resp.success), str(resp.message)

    def set_parameters(self, values: Dict[str, object], timeout_s: float = 5.0) -> Tuple[bool, str]:
        req = SetParameters.Request()
        for k, v in values.items():
            if isinstance(v, float) and not math.isfinite(v):
                return False, f"{k}: not a finite number"
            req.parameters.append(Parameter(name=k, value=_encode(v)))
        resp, err = self._call(self._params_client, req, timeout_s)
        if resp is None:
            return False, err
        bad = [f"{p.name}: {r.reason}" for p, r in zip(req.parameters, resp.results) if not r.successful]
        return (False, "; ".join(bad)) if bad else (True, "parameters set: " + ", ".join(values))
