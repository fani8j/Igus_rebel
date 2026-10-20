"""Small helpers to command the two medical-pick tools in Isaac Sim 5.1 (import inside Isaac's python, see README).

Suction (Surface Gripper, extension isaacsim.robot.surface_gripper must be enabled):
    iface = suction_interface(); suction_close(iface, SG); suction_open(iface, SG)
    set_vacuum_level(stage, SG, 50.0)      # ONE parameter: design vacuum in kPa -> coaxial/shear force limits of every attachment point
Gripper (two position-drive fingers; q = 0 closed 8 mm pad gap, 0.024 open 56 mm):
    set_finger_targets_usd(stage, ROOT, finger_target(gap_m=0.056))  # or CLOSE_TARGET
"""
GRIP_OPEN_Q = 0.024
GRIP_CLOSE_Q = -0.010          # past the closed stop: the drive force saturates at its 24.3 N limit on any carton
GRIP_GAP_CLOSED = 0.008


def suction_interface():
    import isaacsim.robot.surface_gripper._surface_gripper as sgm
    iface = sgm.acquire_surface_gripper_interface()
    iface.set_write_to_usd(True)
    return iface


def suction_close(iface, sg_path):
    return iface.close_gripper(sg_path)


def suction_open(iface, sg_path):
    return iface.open_gripper(sg_path)


def suction_status(iface, sg_path):
    return str(iface.get_gripper_status(sg_path)).split('.')[-1]


def set_vacuum_level(stage, sg_path, kpa, area_m2=None, mu=None, calibrate=True):
    """coaxial limit = kpa * 1e3 * pad area; shear limit = mu * coaxial (both per attachment point). Call before the timeline starts
    (or use GripperView.set_surface_gripper_properties at runtime)."""
    p = stage.GetPrimAtPath(sg_path)
    area = area_m2 if area_m2 is not None else p.GetCustomDataByKey('pad_area_m2')
    mu = mu if mu is not None else p.GetCustomDataByKey('shear_mu')
    co = abs(kpa) * 1e3 * area
    fc = p.GetCustomDataByKey('cal_factor_coaxial') if calibrate else None
    fs = p.GetCustomDataByKey('cal_factor_shear') if calibrate else None
    fc, fs = (1.0 if fc is None else fc), (1.0 if fs is None else fs)
    p.GetAttribute('isaac:coaxialForceLimit').Set(float(co * fc))
    p.GetAttribute('isaac:shearForceLimit').Set(float(mu * co * fs))
    p.SetCustomDataByKey('vacuum_kpa', float(abs(kpa)))
    return co, mu * co      # PHYSICAL limits per attachment point (the authored attributes are these times the calibration factors)


def finger_target(gap_m=None, close=False):
    """Drive target q for both fingers: close=True -> past the stop (force-limited grip), else the target for the given pad gap."""
    if close:
        return GRIP_CLOSE_Q
    return min(max((gap_m - GRIP_GAP_CLOSED) / 2.0, 0.0), GRIP_OPEN_Q)


def set_finger_targets_usd(stage, root, q_left, q_right=None):
    q_right = q_left if q_right is None else q_right
    for side, q in (('left', q_left), ('right', q_right)):
        stage.GetPrimAtPath(f'{root}/joints/finger_{side}_joint').GetAttribute('drive:linear:physics:targetPosition').Set(float(q))


# ---- finger speed: NOT a joint velocity limit (physxJoint:maxJointVelocity halves the drive force in Isaac Sim 5.1) but a target ramp
FINGER_SPEED_MOVE = 0.040      # m/s per finger: LEHF20 open/close speed 5..80 mm/s for both fingers
FINGER_SPEED_PUSH = 0.010      # m/s per finger: LEHF20 pushing speed 5..20 mm/s for both fingers
Q_PUSH_START = 0.012           # m: ramp fast down to here (pad gap 32 mm), then push slowly (contact for 12..16 mm cartons is at q = 0.002..0.004)


def finger_target_profile(q_from, q_to, dt, push=False):
    """List of per-step targets moving the finger drive target from q_from to q_to at the catalog speeds (fast to Q_PUSH_START, then pushing speed when push)."""
    out, q = [], q_from
    if q_to >= q_from:                                   # opening
        while q < q_to:
            q = min(q + FINGER_SPEED_MOVE * dt, q_to)
            out.append(q)
        return out
    while q > max(q_to, Q_PUSH_START if push else q_to):
        q = max(q - FINGER_SPEED_MOVE * dt, q_to, Q_PUSH_START if push else q_to)
        out.append(q)
    while push and q > q_to:
        q = max(q - FINGER_SPEED_PUSH * dt, q_to)
        out.append(q)
    return out


# ---- Surface Gripper calibration (Isaac Sim 5.1, measured with tests/calibrate_surface_gripper.py, see README)
# * shear: the extension's shearForceLimit is exact for ONE attachment point; three rigid points share the load unevenly (group total 1.21 x limit along tool x,
#   1.43 x along tool y) -> authored shear limit = mu x coaxial x cal_factor_shear (2.2): 3 pads detach at 7.7 N (x) / 9.1 N (y) vs 3 x 2.9 = 8.7 N physical (+-15 %)
# * pull-off (coaxial): the extension detaches at 2.3x...4.8x the configured coaxial limit (non-linear, also depends on the shear limit, floor ~23-31 N for 3 pads), so NO
#   single factor can reach the physical 3 x 5.8 N. The physical pull-off limit is enforced in software by SuctionLimitMonitor from the pad buffer joint efforts.
def physical_limits(stage, sg_path):
    """(coaxial N per point, shear N per point) of the physical pad at the configured vacuum (without calibration factors)."""
    p = stage.GetPrimAtPath(sg_path)
    co = p.GetCustomDataByKey('vacuum_kpa') * 1e3 * p.GetCustomDataByKey('pad_area_m2')
    return co, p.GetCustomDataByKey('shear_mu') * co


class SuctionLimitMonitor:
    """Opens the gripper when the pull-off force of any pad exceeds the physical coaxial limit (vacuum x pad area).
    The pull-off tension of pad i is the measured joint effort of its buffer joint (negative effort = carton pulled away from the pad, verified: sum of the three = applied force; the monitor uses tension = -effort).
    Call step() once per physics step, or attach() to run it from the PhysX step events.   art: isaacsim.core.prims.Articulation of the tool (or its robot)."""

    def __init__(self, art, iface, sg_path, coaxial_limit_N, pad_dofs=None):
        self.art, self.iface, self.sg, self.limit = art, iface, sg_path, float(coaxial_limit_N)
        names = list(art.dof_names)
        self.idx = pad_dofs or [names.index(f'pad_{i}_buffer_joint') for i in (1, 2, 3)]
        self.tripped_at = None          # (max pad tension N, sum N) when it opened the gripper
        self.max_tension = 0.0
        self._sub = None

    def step(self):
        if suction_status(self.iface, self.sg) != 'Closed':
            return
        e = self.art.get_measured_joint_efforts()[0]
        t = [-float(e[i]) for i in self.idx]
        self.max_tension = max(self.max_tension, max(t))
        if max(t) > self.limit:
            self.tripped_at = (max(t), sum(t))
            self.iface.open_gripper(self.sg)

    def attach(self):
        import omni.physx
        self._sub = omni.physx.get_physx_interface().subscribe_physics_step_events(lambda dt: self.step())
        return self

    def detach(self):
        self._sub = None
