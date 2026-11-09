"""ROS domain planning for the probe (no ROS imports; same rules as dual_cut_ui/domains.py): the robot topics (TF, joint states,
coordinator status) are read in the node's own domain, the camera topics may be read from another domain (read-only)."""
from typing import Callable, NamedTuple, Optional

MAX_DOMAIN_ID = 232          # Fast DDS: higher ids do not work (port numbers too high)


class DomainPlan(NamedTuple):
    camera_domain: Optional[int]          # None: the cameras are in the node's own domain
    separate_camera_context: bool         # a second rclpy context in camera_domain is needed
    detect_domain: Optional[int]          # None: the detector inherits the environment


def check_domain_param(name: str, value) -> int:
    """-1 (unset) or 0..232; anything else is a configuration error."""
    if isinstance(value, bool) or not isinstance(value, int) or not (value == -1 or 0 <= value <= MAX_DOMAIN_ID):
        raise ValueError(f"{name} must be -1 (not set) or 0..{MAX_DOMAIN_ID}, got {value!r}")
    return value


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
        return f"cameras read from ROS domain {plan.camera_domain} (read-only); robot topics in ROS domain {node_domain}"
    return f"cameras and robot topics in ROS domain {node_domain}"
