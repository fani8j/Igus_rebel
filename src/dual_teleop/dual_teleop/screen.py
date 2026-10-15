"""Text screen of the dual_teleop status (pure function, tested)."""


def render(st):
    """Text screen for one status dict (pure function, tested)."""
    lines = []
    touch = st.get("touch") or {}
    if touch.get("on"):
        left = f", auto-off in {touch['remaining_s']:.0f} s" if touch.get("remaining_s") is not None else ""
        lines.append(f"!!! TOUCH MODE {str(touch.get('arm')).upper()}: knife may contact carton/table "
                     f"(speed x{touch.get('speed_scale')}{left}) !!!")
    elif touch.get("retreat"):
        lines.append(f"~~~ RETREAT {str(touch.get('retreat_arm')).upper()}: touch mode is off but the tool end is still unchecked "
                     f"against carton/table until it is 80 mm away (or 60 s); jog AWAY ~~~")
    if st.get("prompt"):
        lines.insert(0, f">>> {st['prompt']}")
    banner = {"ACTIVE": "TELEOP ACTIVE", "PAUSED": "TELEOP PAUSED (press 'e' to enable)",
              "BLOCKED": "TELEOP BLOCKED"}.get(st.get("state"), str(st.get("state")))
    lines.append(f"== dual_teleop: {banner} ==")
    lines.append(f"selected arm : {st.get('selected', '?').upper()}   (l = left, r = right, e = enable/disable, t = touch mode, q = quit)")
    lines.append(f"enabled      : {st.get('enabled')}    speed scale {st.get('speed_scale')}")
    lines.append(f"coordinator  : {st.get('coordinator')}")
    ctrl = st.get("controllers", {})
    lines.append("controllers  : " + ", ".join(f"{a}={'active' if v else 'NOT ACTIVE'}" for a, v in ctrl.items()))
    for arm, sv in st.get("servo", {}).items():
        state = {True: "running", False: "stopped"}.get(sv.get("running"), "unknown")
        lines.append(f"servo {arm:5s}  : {state:11s} status {sv.get('status')}")
    for reason in st.get("reasons", []):
        lines.append(f"  BLOCK: {reason}")
    for warn in st.get("warnings", []):
        lines.append(f"  WARNING: {warn}")
    if touch.get("last_off_reason") and not touch.get("on"):
        lines.append(f"touch mode   : OFF - {touch['last_off_reason']}")
    if st.get("last_drop"):
        lines.append(f"last dropped : {st['last_drop']}")
    c = st.get("counts", {})
    lines.append(f"commands     : forwarded {c.get('forwarded', 0)}, dropped {c.get('dropped', 0)}")
    return "\n".join(lines)
