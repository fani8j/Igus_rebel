#!/usr/bin/env python3
"""MoveIt - medicine grasp: plan-only verification of the stack.  ADAPTED from rebel_dual_arm_stand_moveit_config/scripts/verify_plan_only.py (same plan-only discipline).
NOTHING is executed: only /get_planning_scene, /apply_planning_scene (scene objects + a scoped allowed-collision entry), /check_state_validity, /compute_ik and
/plan_kinematic_path are called (no /execute_trajectory, no action goals, no controller commands).

  export ROS_DOMAIN_ID=171 ROS_LOCALHOST_ONLY=1
  python3 verify_plan_only.py --out result.json [--w-origin X Y Z --w-yaw-deg A]      # the shelf pose under test (default: config/medicine_shelf_scene.yaml)
  python3 verify_plan_only.py --search --out search.json                              # IK-only reachability grid over shelf placements (no collision checking)

Checks (frame W = shelf frame of common/handoff.py, see config/medicine_shelf_scene.yaml; target = box 3 `carton_2`):
  1 scene: shelf, plinth and 5 cartons present; `home` valid; AMR box present as a robot link.
  2 arm_left : home -> pre-grasp (suction_tcp 50 mm in front of the target spine, tool z along +Y_w = into the spine; OMPL pose goal), then a straight Pilz LIN
               from the pre-grasp to contact (compressed TCP on the spine = pads pressed 5 mm), with the suction tool <-> target carton pair allowed in the
               collision matrix ONLY for that LIN (the same scoped contact allowance idea as the cutting config); a second LIN to 3 mm before contact needs no allowance.
  3 arm_right: target carton pulled out 100 mm (common/params.PULL_OUT); home -> gripper_tcp above the pulled-out carton (stop face 4 mm above the top edge + `lift`
               above the final pose): lift 0.10 m (fingers clear of the carton), then lift 0 (the handoff pose itself, gripper open) - OMPL pose goals.
  4 right_gripper: open (56 mm) -> closed (8 mm), joint-space.
  5 AMR: every waypoint of every returned plan is checked with /check_state_validity; none may touch d200_base_link.  Negative control: a state that puts the
               left TCP inside the AMR box must be reported in collision with d200_base_link and cannot be planned to.
Failures are reported, never worked around.  No limit is loosened.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (AllowedCollisionEntry, AllowedCollisionMatrix, BoundingVolume, CollisionObject, Constraints, JointConstraint, MotionPlanRequest,
                             OrientationConstraint, PlanningScene, PositionConstraint, RobotState)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan, GetPlanningScene, GetPositionFK, GetPositionIK, GetStateValidity
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.join(HERE, '..', 'lib', 'rebel_medicine_grasp_moveit_config')):
    sys.path.insert(0, _d)
import add_medicine_shelf as shelf  # noqa: E402

ERR = {1: 'SUCCESS', -1: 'PLANNING_FAILED', -2: 'INVALID_MOTION_PLAN', -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE', -4: 'CONTROL_FAILED',
       -6: 'TIMED_OUT', -7: 'PREEMPTED', -10: 'START_STATE_IN_COLLISION', -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS', -12: 'GOAL_IN_COLLISION',
       -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS', -14: 'GOAL_CONSTRAINTS_VIOLATED', -15: 'INVALID_GROUP_NAME', -16: 'INVALID_GOAL_CONSTRAINTS',
       -17: 'INVALID_ROBOT_STATE', -18: 'INVALID_LINK_NAME', -19: 'INVALID_OBJECT_NAME', -21: 'FRAME_TRANSFORM_FAILURE', -22: 'COLLISION_CHECKING_UNAVAILABLE',
       -31: 'NO_IK_SOLUTION'}

# ---- tool / handoff constants: cad/medical_box_tooling_v1/common/{params,handoff}.py (m), re-checked by test/check_tool_frames.py against the URDF ----
PULL_OUT = 0.100
GRIPPER_STOP_GAP = 0.004          # stop face above the carton top edge at the final pose (G_NOM)
GRIPPER_TCP_Z, GRIPPER_STOP_Z = 0.1932, 0.1675
GRIPPER_PAD_S_CENTRE = 0.0425     # pad centre behind the (pulled) spine
SUCTION_PRESS = 0.005             # pads compressed 5 mm = TCP z 159 -> 154 mm
PRE_GRASP_DIST = 0.050
LEFT_TCP, RIGHT_TCP = 'arm_left_suction_tcp', 'arm_right_gripper_tcp'
R_SUCTION = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float)     # tool x = +Z_w, y = +X_w, z = +Y_w (into the spine)
R_GRIPPER = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(float)   # tool z = -Z_w (down), x = +X_w (stroke across the thickness)
RZ180 = np.diag([-1.0, -1.0, 1.0])
FINGER = 'arm_right_finger_left_joint'
HOME_FINGER_OPEN, FINGER_CLOSED = 0.024, 0.0


def quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2
    q = [0.0] * 4
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


class Frame:
    """W -> world (pure yaw + translation)"""
    def __init__(self, origin, yaw_deg):
        self.O = np.array(origin, float)
        self.R = shelf.Rz(math.radians(yaw_deg))

    def pose(self, p_w, R_w):
        return (self.O + self.R @ np.asarray(p_w, float)), self.R @ R_w


class Client(Node):
    def __init__(self):
        super().__init__('verify_plan_only')
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.plan = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.valid = self.create_client(GetStateValidity, '/check_state_validity')
        self.scene = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.apply = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.fk = self.create_client(GetPositionFK, '/compute_fk')
        self.js = None
        self.create_subscription(JointState, '/joint_states', self._js, 10)

    def _js(self, m):
        self.js = m

    def wait(self, t=90.0):
        for c in (self.ik, self.plan, self.valid, self.scene, self.apply, self.fk):
            if not c.wait_for_service(timeout_sec=t):
                raise RuntimeError('service %s not available' % c.srv_name)
        t0 = time.time()
        while self.js is None and time.time() - t0 < 20:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.js is None:
            raise RuntimeError('no /joint_states')

    def call(self, client, req, timeout=60.0):
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def home_state(self):
        rs = RobotState()
        rs.joint_state = self.js
        return rs

    @staticmethod
    def state_with(base, names, values):
        rs = RobotState()
        js = JointState()
        js.name = list(base.joint_state.name)
        js.position = list(base.joint_state.position)
        for n, v in zip(names, values):
            js.position[js.name.index(n)] = float(v)
        rs.joint_state = js
        return rs

    def apply_scene(self, scene):
        r = self.call(self.apply, ApplyPlanningScene.Request(scene=scene))
        return bool(r and r.success)


def check_valid(c, rs, group=''):
    r = c.call(c.valid, GetStateValidity.Request(robot_state=rs, group_name=group))
    return dict(valid=bool(r.valid), contacts=sorted({(x.contact_body_1, x.contact_body_2) for x in r.contacts})[:8])


def joint_goal(names, vals, tol=1e-3):
    cs = Constraints()
    for n, v in zip(names, vals):
        cs.joint_constraints.append(JointConstraint(joint_name=n, position=float(v), tolerance_above=tol, tolerance_below=tol, weight=1.0))
    return cs


def pose_goal(link, xyz, quat, ptol=1e-3, rtol=0.02):
    cs = Constraints()
    pc = PositionConstraint()
    pc.header.frame_id = 'world'
    pc.link_name = link
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)
    pose.orientation.w = 1.0
    pc.constraint_region = BoundingVolume(primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[ptol])], primitive_poses=[pose])
    pc.weight = 1.0
    oc = OrientationConstraint()
    oc.header.frame_id = 'world'
    oc.link_name = link
    oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = (float(v) for v in quat)
    oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = rtol
    oc.weight = 1.0
    cs.position_constraints.append(pc)
    cs.orientation_constraints.append(oc)
    return cs


def state_from_plan(start, traj):
    """robot state at the end of a returned plan (start state with the planned joints replaced)"""
    jt = traj.joint_trajectory
    return Client.state_with(start, jt.joint_names, jt.points[-1].positions)


def do_plan(c, group, start, goal, pipeline='ompl', planner='RRTConnectkConfigDefault', t=10.0, vel=0.2, check_amr=True, retries=3):
    req = MotionPlanRequest()
    req.group_name = group
    req.start_state = start
    req.goal_constraints = [goal]
    req.pipeline_id = pipeline
    req.planner_id = planner
    req.num_planning_attempts = 5 if pipeline == 'ompl' else 1
    req.allowed_planning_time = t
    req.max_velocity_scaling_factor = vel
    req.max_acceleration_scaling_factor = vel
    t0 = time.time()
    attempts = 0
    while True:
        # OMPL paths are re-sampled by the time parameterization; a re-sampled state can very rarely clip a thin shelf board / carton and move_group's path validation then
        # answers INVALID_MOTION_PLAN. Such a plan is never used: ask again (up to `retries` extra times), and report how many calls were needed.
        r = c.call(c.plan, GetMotionPlan.Request(motion_plan_request=req), timeout=t + 60)
        res = r.motion_plan_response
        attempts += 1
        if res.error_code.val != -2 or attempts > retries:
            break
    out = dict(attempts=attempts, pipeline=pipeline, planner=planner, error_code=int(res.error_code.val), error=ERR.get(res.error_code.val, str(res.error_code.val)),
               wall_s=round(time.time() - t0, 3))
    pts = res.trajectory.joint_trajectory.points
    if res.error_code.val == 1 and pts:
        out['n_points'] = len(pts)
        out['duration_s'] = round(pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9, 3)
        out['final'] = {n: round(v, 4) for n, v in zip(res.trajectory.joint_trajectory.joint_names, pts[-1].positions)}
        out['_traj'] = res.trajectory
        if check_amr:
            out['amr_check'] = amr_check(c, start, res.trajectory)
    return out


def amr_check(c, start, traj, every=1):
    """every waypoint (full robot state) through /check_state_validity; count states with ANY contact involving the AMR box (d200_base_link) and with any contact"""
    jt = traj.joint_trajectory
    n_amr = n_any = 0
    n = 0
    for i, p in enumerate(jt.points):
        if i % every and i != len(jt.points) - 1:
            continue
        rs = Client.state_with(start, jt.joint_names, p.positions)
        v = check_valid(c, rs)
        n += 1
        if not v['valid']:
            n_any += 1
        if any('d200_base_link' in x for pair in v['contacts'] for x in pair):
            n_amr += 1
    return dict(waypoints_checked=n, states_in_collision=n_any, states_touching_amr=n_amr)


def tcp_path(c, start, traj, link):
    """TCP positions (world) of every waypoint via /compute_fk, and the straightness: max distance of the points to the segment first -> last, plus the end orientation drift"""
    jt = traj.joint_trajectory
    pts = []
    for p in jt.points:
        req = GetPositionFK.Request()
        req.header.frame_id = 'world'
        req.fk_link_names = [link]
        req.robot_state = Client.state_with(start, jt.joint_names, p.positions)
        r = c.call(c.fk, req)
        pp = r.pose_stamped[0].pose
        pts.append([pp.position.x, pp.position.y, pp.position.z])
    P = np.array(pts)
    d = P[-1] - P[0]
    L = float(np.linalg.norm(d))
    u = d / L if L > 0 else d
    dev = np.linalg.norm((P - P[0]) - np.outer((P - P[0]) @ u, u), axis=1)
    return dict(length_m=round(L, 4), max_deviation_from_line_m=float(dev.max()), start=[round(v, 4) for v in P[0]], end=[round(v, 4) for v in P[-1]])


def do_ik(c, group, link, xyz, quat, seed, avoid=True, timeout=1.0):
    req = GetPositionIK.Request()
    ik = req.ik_request
    ik.group_name = group
    ik.robot_state = seed
    ik.avoid_collisions = avoid
    ik.ik_link_name = link
    ik.timeout.sec = int(timeout)
    ik.timeout.nanosec = int((timeout - int(timeout)) * 1e9)
    ps = PoseStamped()
    ps.header.frame_id = 'world'
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in xyz)
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = (float(v) for v in quat)
    ik.pose_stamped = ps
    r = c.call(c.ik, req)
    out = dict(error_code=int(r.error_code.val), error=ERR.get(r.error_code.val, str(r.error_code.val)))
    if r.error_code.val == 1:
        js = r.solution.joint_state
        out['solution'] = {n: round(v, 5) for n, v in zip(js.name, js.position) if n.startswith(group + '_joint')}
        out['_state'] = r.solution
    return out


# ------------------------------------------------------------------ target poses
def targets(F, spec, pull_out=0.0):
    """dict of world poses (xyz, R) for the left suction TCP and the right gripper TCP, in the W frame of the shelf"""
    c = spec['cartons']
    b = c['boxes_mm'][c['sequence'][c['target_index']]]
    h = b['h'] * 1e-3
    zc = h / 2
    out = {}
    for roll, Rr in (('roll0', np.eye(3)), ('roll180', RZ180)):
        Rl = R_SUCTION @ Rr
        for name, y in (('pre_grasp', -PRE_GRASP_DIST), ('contact_minus3mm', -0.003), ('contact_pressed', SUCTION_PRESS)):
            # y: TCP (uncompressed lip) position along +Y_w relative to the spine; pressed = lip 5 mm past the spine (== compressed TCP on the spine, pads 5 mm in)
            out['left_%s_%s' % (name, roll)] = F.pose([0.0, y, zc], Rl)
        Rg = R_GRIPPER @ Rr
        ytcp = -pull_out + GRIPPER_PAD_S_CENTRE if pull_out else -PULL_OUT + GRIPPER_PAD_S_CENTRE
        ztcp = h + GRIPPER_STOP_GAP - (GRIPPER_TCP_Z - GRIPPER_STOP_Z)
        for name, lift in (('handoff_above_100mm', 0.10), ('handoff_final', 0.0)):
            out['right_%s_%s' % (name, roll)] = F.pose([0.0, ytcp, ztcp + lift], Rg)
    return out


def scene_diff_objects(path, w_frame, pull_out, add=True):
    frame, objs = shelf.load_objects(path, pull_out, w_frame)
    sc = shelf.build_scene(frame, objs)
    if not add:
        for co in sc.world.collision_objects:
            co.operation = CollisionObject.REMOVE
        sc.object_colors = []
    return sc, objs


def acm_diff(pairs, allow):
    """PlanningScene diff that sets entries of the allowed collision matrix (names must exist in the scene's ACM, so each name is added to both axes)"""
    acm = AllowedCollisionMatrix()
    names = sorted({n for p in pairs for n in p})
    acm.entry_names = names
    acm.entry_values = [AllowedCollisionEntry(enabled=[False] * len(names)) for _ in names]
    for a, b in pairs:
        i, j = names.index(a), names.index(b)
        acm.entry_values[i].enabled[j] = allow
        acm.entry_values[j].enabled[i] = allow
    return PlanningScene(is_diff=True, allowed_collision_matrix=acm)


def current_acm(c):
    req = GetPlanningScene.Request()
    req.components.components = req.components.ALLOWED_COLLISION_MATRIX
    return c.call(c.scene, req).scene.allowed_collision_matrix


def set_allowed(c, pairs, allow):
    """read-modify-write of the live ACM: the diff must carry a full matrix that contains the names"""
    acm = current_acm(c)
    for a, b in pairs:
        for n in (a, b):
            if n not in acm.entry_names:
                acm.entry_names.append(n)
                for e in acm.entry_values:
                    e.enabled.append(False)
                acm.entry_values.append(AllowedCollisionEntry(enabled=[False] * len(acm.entry_names)))
    for a, b in pairs:
        i, j = acm.entry_names.index(a), acm.entry_names.index(b)
        acm.entry_values[i].enabled[j] = allow
        acm.entry_values[j].enabled[i] = allow
    return c.apply_scene(PlanningScene(is_diff=True, allowed_collision_matrix=acm))


# ------------------------------------------------------------------ search
def ik_multi(c, group, link, p, q, home, seeds, timeout=0.4):
    """collision-aware IK from several seeds; True if any seed gives a solution"""
    for s in seeds:
        if do_ik(c, group, link, p, q, s, avoid=True, timeout=timeout)['error'] == 'SUCCESS':
            return True
    return False


def search(c, a, spec_path):
    """collision-aware IK over a grid of shelf placements; the scene (shelf, cartons) is re-applied for every candidate, packed for the left arm, target pulled out for the right"""
    spec = yaml.safe_load(open(spec_path))
    home = c.home_state()
    seeds_l = [home, c.state_with(home, ['arm_left_joint%d' % i for i in range(1, 7)], [-0.6, 0.9, 1.2, 0.0, 0.0, 0.0]),
               c.state_with(home, ['arm_left_joint%d' % i for i in range(1, 7)], [-0.3, 0.7, 1.0, 0.0, 0.3, 0.0])]
    seeds_r = [home, c.state_with(home, ['arm_right_joint%d' % i for i in range(1, 7)], [0.6, 0.4, 1.5, 0.0, 1.2, 0.0]),
               c.state_with(home, ['arm_right_joint%d' % i for i in range(1, 7)], [0.3, 0.5, 1.4, 0.0, 1.2, 0.0])]
    ids = ['medicine_shelf', 'shelf_plinth'] + ['carton_%d' % i for i in range(len(spec['cartons']['sequence']))]
    res = []
    for x, y, z in itertools.product([float(v) for v in a.search_x.split(',')], [float(v) for v in a.search_y.split(',')], [float(v) for v in a.search_z.split(',')]):
        wf = dict(origin=[x, y, z], yaw_deg=a.w_yaw_deg)
        F = Frame(wf['origin'], wf['yaw_deg'])
        T = targets(F, spec)
        row = dict(origin=[x, y, z], left={}, right={})
        for pull, key in ((0.0, 'left'), (PULL_OUT, 'right')):
            rm = PlanningScene(is_diff=True)
            rm.world.collision_objects = [CollisionObject(id=i, operation=CollisionObject.REMOVE) for i in ids]
            c.apply_scene(rm)
            sc, _ = scene_diff_objects(spec_path, wf, pull)
            c.apply_scene(sc)
            for roll in ('roll0', 'roll180'):
                if key == 'left':
                    ok = [ik_multi(c, 'arm_left', LEFT_TCP, *(lambda pq: (pq[0], quat_from_R(pq[1])))(T['left_%s_%s' % (n, roll)]), home, seeds_l) for n in ('pre_grasp', 'contact_minus3mm')]
                else:
                    ok = [ik_multi(c, 'arm_right', RIGHT_TCP, *(lambda pq: (pq[0], quat_from_R(pq[1])))(T['right_%s_%s' % (n, roll)]), home, seeds_r) for n in ('handoff_above_100mm', 'handoff_final')]
                row[key][roll] = ok
        row['left_ok'] = any(all(v) for v in row['left'].values())
        row['right_ok'] = any(all(v) for v in row['right'].values())
        res.append(row)
        print(row['origin'], 'left', row['left_ok'], 'right', row['right_ok'], flush=True)
    return res


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith('_')}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--scene-file', default=shelf.default_scene_file())
    ap.add_argument('--w-origin', type=float, nargs=3, default=None, help='shelf frame W origin in world (= d200_base_footprint); default: the yaml')
    ap.add_argument('--w-yaw-deg', type=float, default=None)
    ap.add_argument('--search', action='store_true')
    ap.add_argument('--search-x', default='0.70,0.75,0.80')
    ap.add_argument('--search-y', default='0.0,0.1,0.2')
    ap.add_argument('--search-z', default='1.25,1.31,1.37')
    ap.add_argument('--wait', type=float, default=90.0)
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.scene_file))
    wf = dict(spec['w_frame'])
    if a.w_origin:
        wf['origin'] = a.w_origin
    if a.w_yaw_deg is not None:
        wf['yaw_deg'] = a.w_yaw_deg
    a.w_yaw_deg = wf['yaw_deg']
    rclpy.init()
    c = Client()
    c.wait(a.wait)

    if a.search:
        r = search(c, a, a.scene_file)
        json.dump(r, open(a.out, 'w'), indent=1)
        rclpy.shutdown()
        return 0

    res = dict(schema='rebel_medicine_grasp/verify_plan_only/v1', time=time.strftime('%Y-%m-%d %H:%M:%S'), nothing_executed=True, w_frame=wf)
    F = Frame(wf['origin'], wf['yaw_deg'])
    ids = ['medicine_shelf', 'shelf_plinth'] + ['carton_%d' % i for i in range(len(spec['cartons']['sequence']))]

    def put_scene(pull):
        rm = PlanningScene(is_diff=True)
        rm.world.collision_objects = [CollisionObject(id=i, operation=CollisionObject.REMOVE) for i in ids]
        c.apply_scene(rm)
        sc, _ = scene_diff_objects(a.scene_file, wf, pull)
        return c.apply_scene(sc)

    res['scene_applied_packed'] = put_scene(0.0)
    sc = c.call(c.scene, GetPlanningScene.Request()).scene
    res['scene_objects'] = sorted(o.id for o in sc.world.collision_objects)
    res['scene_has_all'] = all(i in res['scene_objects'] for i in ids)
    home = c.home_state()
    res['home_joint_state'] = {n: round(v, 6) for n, v in zip(home.joint_state.name, home.joint_state.position)}
    res['home_validity'] = check_valid(c, home)
    T = targets(F, spec)
    res['targets_world'] = {k: dict(xyz=[round(float(v), 4) for v in p], quat_xyzw=[round(v, 4) for v in quat_from_R(R)]) for k, (p, R) in T.items()}

    # ---- arm_left: pre-grasp and LIN approach
    left = []
    names_l = ['arm_left_joint%d' % i for i in range(1, 7)]
    for roll in ('roll0', 'roll180'):
        e = dict(roll=roll)
        p_pre, R_pre = T['left_pre_grasp_' + roll]
        e['ik_pre_grasp'] = do_ik(c, 'arm_left', LEFT_TCP, p_pre, quat_from_R(R_pre), home)
        e['plan_home_to_pre_grasp'] = do_plan(c, 'arm_left', home, pose_goal(LEFT_TCP, p_pre, quat_from_R(R_pre)))
        if e['plan_home_to_pre_grasp']['error_code'] == 1:
            s1 = state_from_plan(home, e['plan_home_to_pre_grasp']['_traj'])
            for name in ('contact_minus3mm',):
                p, R = T['left_%s_%s' % (name, roll)]
                e['lin_pre_grasp_to_' + name] = do_plan(c, 'arm_left', s1, pose_goal(LEFT_TCP, p, quat_from_R(R), 1e-4, 0.01), pipeline='pilz_industrial_motion_planner',
                                                        planner='LIN', t=5.0)
            # pressed contact: allow the suction tool <-> target carton for this one motion only
            p, R = T['left_contact_pressed_' + roll]
            pair = [('arm_left_suction_tool', 'carton_2')]
            e['contact_allowance_set'] = set_allowed(c, pair, True)
            e['lin_pre_grasp_to_contact_pressed'] = do_plan(c, 'arm_left', s1, pose_goal(LEFT_TCP, p, quat_from_R(R), 1e-4, 0.01), pipeline='pilz_industrial_motion_planner',
                                                            planner='LIN', t=5.0)
            e['contact_allowance_removed'] = set_allowed(c, pair, False)
            lin = e['lin_pre_grasp_to_contact_pressed']
            if lin['error_code'] == 1:
                lin['tcp_path'] = tcp_path(c, s1, lin['_traj'], LEFT_TCP)
                # straightness: the TCP path is checked by the caller (check_plan_results_fk.py); here the joint-space end point and duration
                e['lin_without_allowance'] = do_plan(c, 'arm_left', s1, pose_goal(LEFT_TCP, p, quat_from_R(R), 1e-4, 0.01), pipeline='pilz_industrial_motion_planner',
                                                     planner='LIN', t=5.0, check_amr=False)
        left.append(e)
    res['arm_left'] = left

    # ---- arm_right: pulled-out target carton
    res['scene_applied_pulled'] = put_scene(PULL_OUT)
    right = []
    for roll in ('roll0', 'roll180'):
        e = dict(roll=roll)
        for name in ('handoff_above_100mm', 'handoff_final'):
            p, R = T['right_%s_%s' % (name, roll)]
            e['ik_' + name] = do_ik(c, 'arm_right', RIGHT_TCP, p, quat_from_R(R), home)
            e['plan_home_to_' + name] = do_plan(c, 'arm_right', home, pose_goal(RIGHT_TCP, p, quat_from_R(R)))
        right.append(e)
    res['arm_right'] = right

    # ---- gripper
    g_open = c.state_with(home, [FINGER], [HOME_FINGER_OPEN])
    res['right_gripper'] = dict(
        open_validity=check_valid(c, g_open, 'right_gripper'),
        open_to_closed=do_plan(c, 'right_gripper', g_open, joint_goal([FINGER], [FINGER_CLOSED]), check_amr=False),
        closed_to_open=do_plan(c, 'right_gripper', c.state_with(home, [FINGER], [FINGER_CLOSED]), joint_goal([FINGER], [HOME_FINGER_OPEN]), check_amr=False),
        pilz_ptp_open_to_closed=do_plan(c, 'right_gripper', g_open, joint_goal([FINGER], [FINGER_CLOSED]), pipeline='pilz_industrial_motion_planner', planner='PTP',
                                        check_amr=False))

    # ---- AMR checks.  (a) Control that the AMR collision box is live: a temporary probe box overlapping the AMR deck must be reported as a contact with d200_base_link
    # by /check_state_validity at `home`, and must be gone again afterwards.  (b) The robot cannot reach the AMR deck anyway (lowest left-TCP height over random states with the
    # lift at its lower limit is computed offline by test/check_amr_clearance.py); (c) every waypoint of every plan above was checked for contacts with d200_base_link.
    res['put_scene_packed_again'] = put_scene(0.0)
    probe = CollisionObject(id='amr_probe', operation=CollisionObject.ADD)
    probe.header.frame_id = 'world'
    probe.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.10, 0.10, 0.10]))
    pp = Pose()
    pp.position.x, pp.position.y, pp.position.z = 0.40, 0.0, 0.20
    pp.orientation.w = 1.0
    probe.primitive_poses.append(pp)
    c.apply_scene(PlanningScene(is_diff=True, world=__import__('moveit_msgs.msg', fromlist=['PlanningSceneWorld']).PlanningSceneWorld(collision_objects=[probe])))
    v = check_valid(c, home)
    c.apply_scene(PlanningScene(is_diff=True, world=__import__('moveit_msgs.msg', fromlist=['PlanningSceneWorld']).PlanningSceneWorld(
        collision_objects=[CollisionObject(id='amr_probe', operation=CollisionObject.REMOVE)])))
    res['amr_control'] = dict(probe_overlapping_amr_deck=v, probe_contacts_amr=any('d200_base_link' in x for pair in v['contacts'] for x in pair),
                              probe_removed_home_valid=check_valid(c, home)['valid'])

    res = strip(res)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
