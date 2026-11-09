"""ROS domain planning for the UI (no ROS imports): the coordinator is talked to in the UI node's own domain, the camera topics
may be read from another domain (read-only), and the detector subprocess runs in the camera domain unless told otherwise."""
import contextlib
import os
from typing import Callable, NamedTuple, Optional

MAX_DOMAIN_ID = 232          # Fast DDS: higher ids do not work (port numbers too high)


class DomainPlan(NamedTuple):
    camera_domain: Optional[int]          # None: the cameras are in the UI node's own domain
    separate_camera_context: bool         # a second rclpy context in camera_domain is needed
    detect_domain: Optional[int]          # None: the detector inherits the environment


def check_domain_param(name: str, value) -> int:
    """-1 (unset) or 0..232; anything else is a configuration error."""
    if isinstance(value, bool) or not isinstance(value, int) or not (value == -1 or 0 <= value <= MAX_DOMAIN_ID):
        raise ValueError(f"{name} must be -1 (not set) or 0..{MAX_DOMAIN_ID}, got {value!r}")
    return value


def check_localhost_param(name: str, value) -> Optional[int]:
    """-1 (inherit) -> None, 0 or 1 -> that value; anything else is a configuration error."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 0, 1):
        raise ValueError(f"{name} must be -1 (inherit), 0 or 1, got {value!r}")
    return None if value == -1 else value


def plan_domains(node_domain: int, camera_domain_id: int, detect_domain_id: int) -> DomainPlan:
    cam = check_domain_param("camera_domain_id", camera_domain_id)
    det = check_domain_param("detect_domain_id", detect_domain_id)
    camera = cam if cam >= 0 else None
    detect = det if det >= 0 else camera              # default: the detector shares the camera domain
    return DomainPlan(camera, camera is not None and camera != node_domain, detect)


def select_camera_source(plan: DomainPlan, main_source, make_camera_source: Callable[[int], object]):
    """The object the CameraHub subscribes through: the main bridge, or (separate domain) one in the camera context.  The
    factory is only called when a separate context is really needed."""
    return make_camera_source(plan.camera_domain) if plan.separate_camera_context else main_source


def startup_message(plan: DomainPlan, node_domain: int) -> str:
    if plan.separate_camera_context:
        return f"cameras read from ROS domain {plan.camera_domain} (read-only); coordinator in ROS domain {node_domain}"
    return f"cameras read from the UI's own ROS domain {node_domain}"


@contextlib.contextmanager
def localhost_only_env(value):
    """Temporarily set ROS_LOCALHOST_ONLY (None: leave it alone).  Only used at start-up, before any other thread runs."""
    if value is None:
        yield
        return
    old = os.environ.get("ROS_LOCALHOST_ONLY")
    os.environ["ROS_LOCALHOST_ONLY"] = str(value)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("ROS_LOCALHOST_ONLY", None)
        else:
            os.environ["ROS_LOCALHOST_ONLY"] = old
