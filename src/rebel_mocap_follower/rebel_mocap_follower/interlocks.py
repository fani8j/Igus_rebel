"""ROS-free interlock decisions of the dual-arm mocap follower.

Each function returns None (no objection) or a reason string; the node feeds a reason into the
follower core's `blocked` input, which holds that arm. Times are seconds on a monotonic clock.
"""
import math
from typing import Iterable, Optional, Sequence, Tuple

# dual_cut_msgs/msg/CoordinatorStatus
STATE_WAITING_BARRIER = 3
STATE_EXECUTING = 4
STATE_NAMES = {0: "IDLE", 1: "PLANNING", 2: "PLANNED", 3: "WAITING_BARRIER",
               4: "EXECUTING", 5: "STOPPED"}
CONTROLLER_ANSWER_MAX_AGE_S = 3.0


def coordinator_gate(state: Optional[int], age_s: float, timeout_s: float) -> Optional[str]:
    """Hold while the coordinator runs a phase, or when its status is missing/stale."""
    if state is None or not math.isfinite(age_s) or age_s > timeout_s:
        shown = "never" if state is None or not math.isfinite(age_s) else f"{age_s:.1f} s ago"
        return (f"coordinator status silent (last {shown}, limit {timeout_s:.1f} s): "
                "coordinator not running or not reachable")
    if state in (STATE_WAITING_BARRIER, STATE_EXECUTING):
        return f"coordinator is {STATE_NAMES.get(state, state)}: it owns the arms"
    return None


def controller_gate(controller: str, state: Optional[str], age_s: float,
                    max_age_s: float = CONTROLLER_ANSWER_MAX_AGE_S) -> Optional[str]:
    """Hold unless the trajectory controller was reported 'active' recently."""
    if state is None or not math.isfinite(age_s) or age_s > max_age_s:
        return (f"controller {controller} state unknown (no list_controllers answer for "
                f"{'ever' if state is None else f'{age_s:.1f} s'}, limit {max_age_s:.0f} s)")
    if state != "active":
        return f"controller {controller} is '{state}', not active"
    return None


def exclusivity_gate(topic: str, publishers: Iterable[Tuple[str, str]],
                     own_node_prefix: str, allowed: Sequence[str] = (),
                     required_own: str = "") -> Optional[str]:
    """Hold if anyone but our own Servo publishes on the JTC command topic.

    publishers: (node_namespace, node_name) pairs; own_node_prefix: namespace of our Servo
    (e.g. /mocap/arm_left/servo). A publisher belongs to us when its full node path is under it.
    allowed: full node paths that may publish anyway (the dual coordinator publishes its trajectories on this
    topic; it is gated separately by coordinator_gate).
    required_own: when set (servo_relay:=true: the relay node path), exactly ONE own publisher must exist and it must be
    that node: a second relay, or a Servo that is still wired to the JTC topic, would otherwise be accepted as "own"
    just because it lives under the Servo namespace. Zero own publishers also holds (the relay is not running).
    """
    own = own_node_prefix.rstrip("/")
    others = []
    own_pubs = []
    for ns, name in publishers:
        full = f"{ns.rstrip('/')}/{name}"
        if full == own or full.startswith(own + "/"):
            own_pubs.append(full)
            continue
        if full in allowed:
            continue
        others.append(full)
    if others:
        return f"{topic} also has publisher(s) {', '.join(sorted(others))}: another program commands this arm"
    if required_own and own_pubs != [required_own]:
        if not own_pubs:
            return f"{topic} has no publisher {required_own}: the Servo relay is not running"
        return (f"{topic} must have exactly one own publisher ({required_own}) but has "
                f"{len(own_pubs)}: {', '.join(sorted(own_pubs))}")
    return None


def first_reason(reasons: Sequence[Optional[str]]) -> Optional[str]:
    for reason in reasons:
        if reason:
            return reason
    return None
