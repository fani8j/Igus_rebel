#!/usr/bin/env python3
"""Plan-only smoke test of the dual-arm MoveIt stack.  Written new for this package.  NOTHING is executed: only /get_planning_scene,
/check_state_validity, /compute_ik and /plan_kinematic_path are called (no /execute_trajectory, no action goals, no controller commands).

  export ROS_DOMAIN_ID=<the stack's unused id> ROS_LOCALHOST_ONLY=1
  python3 verify_plan_only.py --out result.json

Checks: scene objects present; `home` valid (self + scene collision); per arm: joint-space plan from home to a reachable joint goal; for a set of
roll angles about the world Z axis, the knife_tip pose `--above` m over the carton top centre with the tool z axis pointing DOWN: /compute_ik
(collision aware) and a pose-goal plan; gantry joint-space plan.  Failures are reported, never worked around.
"""
import argparse
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (BoundingVolume, Constraints, JointConstraint, MotionPlanRequest, OrientationConstraint, PositionConstraint,
                             RobotState)
from moveit_msgs.srv import GetMotionPlan, GetPlanningScene, GetPositionIK, GetStateValidity
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

ERR = {1: 'SUCCESS', -1: 'PLANNING_FAILED', -2: 'INVALID_MOTION_PLAN', -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE', -4: 'CONTROL_FAILED',
       -5: 'UNABLE_TO_AQUIRE_SENSOR_DATA', -6: 'TIMED_OUT', -7: 'PREEMPTED', -10: 'START_STATE_IN_COLLISION', -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
       -12: 'GOAL_IN_COLLISION', -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS', -14: 'GOAL_CONSTRAINTS_VIOLATED', -15: 'INVALID_GROUP_NAME',
       -16: 'INVALID_GOAL_CONSTRAINTS', -17: 'INVALID_ROBOT_STATE', -18: 'INVALID_LINK_NAME', -19: 'INVALID_OBJECT_NAME', -21: 'FRAME_TRANSFORM_FAILURE',
       -22: 'COLLISION_CHECKING_UNAVAILABLE', -23: 'ROBOT_STATE_STALE', -24: 'SENSOR_INFO_STALE', -25: 'COMMUNICATION_FAILURE', -26: 'CRASH',
       -27: 'ABORT', -31: 'NO_IK_SOLUTION'}


def quat_z_down(roll_about_world_z):
    """tool z -> world -Z, then a rotation about the world Z axis: q = Rz(gamma) * Rx(pi); returns xyzw"""
    g = roll_about_world_z / 2.0
    # Rz(g) * Rx(pi): Rx(pi) = (1,0,0,0) xyzw
    return [math.cos(g), math.sin(g), 0.0, 0.0]


class Client(Node):
    def __init__(self):
        super().__init__('verify_plan_only')
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.plan = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.valid = self.create_client(GetStateValidity, '/check_state_validity')
        self.scene = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.js = None
        self.create_subscription(JointState, '/joint_states', self._js, 10)

    def _js(self, m):
        self.js = m

    def wait(self, t=90.0):
        for c in (self.ik, self.plan, self.valid, self.scene):
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

    def state_with(self, base, names, values):
        rs = RobotState()
        js = JointState()
        js.name = list(base.joint_state.name)
        js.position = list(base.joint_state.position)
        for n, v in zip(names, values):
            js.position[js.name.index(n)] = float(v)
        rs.joint_state = js
        return rs


def check_valid(c, rs, group=''):
    r = c.call(c.valid, GetStateValidity.Request(robot_state=rs, group_name=group))
    return dict(valid=bool(r.valid), contacts=[(x.contact_body_1, x.contact_body_2) for x in r.contacts][:6])


def joint_goal(names, vals, tol=1e-3):
    cs = Constraints()
    for n, v in zip(names, vals):
        cs.joint_constraints.append(JointConstraint(joint_name=n, position=float(v), tolerance_above=tol, tolerance_below=tol, weight=1.0))
    return cs


def pose_goal(link, frame, xyz, quat, ptol=1e-3, rtol=0.02):
    cs = Constraints()
    pc = PositionConstraint()
    pc.header.frame_id = frame
    pc.link_name = link
    sp = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[ptol])
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.w = 1.0
    pc.constraint_region = BoundingVolume(primitives=[sp], primitive_poses=[pose])
    pc.weight = 1.0
    oc = OrientationConstraint()
    oc.header.frame_id = frame
    oc.link_name = link
    oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = quat
    oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = rtol
    oc.weight = 1.0
    cs.position_constraints.append(pc)
    cs.orientation_constraints.append(oc)
    return cs


def do_plan(c, group, start, goal, pipeline='ompl', planner='RRTConnectkConfigDefault', t=10.0):
    req = MotionPlanRequest()
    req.group_name = group
    req.start_state = start
    req.goal_constraints = [goal]
    req.pipeline_id = pipeline
    req.planner_id = planner
    req.num_planning_attempts = 5
    req.allowed_planning_time = t
    req.max_velocity_scaling_factor = 0.2
    req.max_acceleration_scaling_factor = 0.2
    t0 = time.time()
    r = c.call(c.plan, GetMotionPlan.Request(motion_plan_request=req), timeout=t + 30)
    res = r.motion_plan_response
    out = dict(error_code=int(res.error_code.val), error=ERR.get(res.error_code.val, str(res.error_code.val)), wall_s=round(time.time() - t0, 3))
    pts = res.trajectory.joint_trajectory.points
    if res.error_code.val == 1 and pts:
        out['n_points'] = len(pts)
        out['duration_s'] = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9
        out['final'] = {n: round(v, 4) for n, v in zip(res.trajectory.joint_trajectory.joint_names, pts[-1].positions)}
    return out


def do_ik(c, group, link, frame, xyz, quat, seed, timeout=1.0):
    req = GetPositionIK.Request()
    ik = req.ik_request
    ik.group_name = group
    ik.robot_state = seed
    ik.avoid_collisions = True
    ik.ik_link_name = link
    ik.timeout.sec = int(timeout)
    ik.timeout.nanosec = int((timeout - int(timeout)) * 1e9)
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = quat
    ik.pose_stamped = ps
    r = c.call(c.ik, req)
    out = dict(error_code=int(r.error_code.val), error=ERR.get(r.error_code.val, str(r.error_code.val)))
    if r.error_code.val == 1:
        js = r.solution.joint_state
        out['solution'] = {n: round(v, 4) for n, v in zip(js.name, js.position) if n.startswith(group + '_joint')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--above', type=float, default=0.10, help='knife_tip height above the carton top centre (m)')
    ap.add_argument('--rolls-deg', default='0,90,180,270', help='tool roll values about world Z for the pose goals')
    ap.add_argument('--wait', type=float, default=90.0)
    a = ap.parse_args()
    rclpy.init()
    c = Client()
    c.wait(a.wait)
    res = dict(schema='rebel_dual_arm/verify_plan_only/v1', time=time.strftime('%Y-%m-%d %H:%M:%S'), nothing_executed=True)

    sc = c.call(c.scene, GetPlanningScene.Request()).scene
    # MoveIt stores the first shape's pose in the object pose (primitive_poses[0] is then identity), so compose both
    def obj_pos(o):
        p, q = o.pose.position, o.primitive_poses[0].position
        return [round(v, 4) for v in (p.x + q.x, p.y + q.y, p.z + q.z)]
    objs = {o.id: dict(pos=obj_pos(o), size=[round(v, 4) for v in o.primitives[0].dimensions],
                       quat_xyzw=[round(v, 4) for v in (o.pose.orientation.x, o.pose.orientation.y, o.pose.orientation.z, o.pose.orientation.w)])
            for o in sc.world.collision_objects if o.primitives}
    res['scene_objects'] = objs
    home = c.home_state()
    res['home_joint_state'] = {n: round(v, 6) for n, v in zip(home.joint_state.name, home.joint_state.position)}
    res['home_validity'] = check_valid(c, home)

    carton = objs.get('carton')
    top_centre = None
    if carton:
        top_centre = [carton['pos'][0], carton['pos'][1], carton['pos'][2] + carton['size'][2] / 2]
        res['carton_top_centre_world'] = top_centre
    jgoals = {'arm_left': [-0.3, 0.5, 0.6, 0.0, 0.4, 0.0], 'arm_right': [0.3, 0.5, 0.6, 0.0, 0.4, 0.0]}
    res['arms'] = {}
    for arm in ('arm_left', 'arm_right'):
        r = {}
        names = [arm + '_joint%d' % i for i in range(1, 7)]
        gs = c.state_with(home, names, jgoals[arm])
        r['joint_goal'] = dict(goal=jgoals[arm], goal_validity=check_valid(c, gs, arm))
        r['joint_goal']['plan'] = do_plan(c, arm, home, joint_goal(names, jgoals[arm]))
        r['pose_goals'] = []
        if top_centre:
            tgt = [top_centre[0], top_centre[1], top_centre[2] + a.above]
            for roll in [float(x) for x in a.rolls_deg.split(',')]:
                q = quat_z_down(math.radians(roll))
                e = dict(roll_deg=roll, target_xyz=[round(v, 4) for v in tgt], target_quat_xyzw=[round(v, 4) for v in q])
                e['ik'] = do_ik(c, arm, arm + '_knife_tip', 'world', tgt, q, home)
                e['plan_pose_goal'] = do_plan(c, arm, home, pose_goal(arm + '_knife_tip', 'world', tgt, q))
                if e['ik']['error_code'] == 1 and e['plan_pose_goal']['error_code'] != 1:
                    names_ik = list(e['ik']['solution'].keys())
                    e['plan_to_ik_solution'] = do_plan(c, arm, home, joint_goal(names_ik, list(e['ik']['solution'].values()), 2e-3))
                r['pose_goals'].append(e)
        res['arms'][arm] = r
    gn = ['lift_joint', 'slider_left_joint', 'slider_right_joint']
    gv = [-0.3, -0.04, 0.04]
    res['gantry'] = dict(goal=dict(zip(gn, gv)), goal_validity=check_valid(c, c.state_with(home, gn, gv), 'gantry'), plan=do_plan(c, 'gantry', home, joint_goal(gn, gv)))
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
