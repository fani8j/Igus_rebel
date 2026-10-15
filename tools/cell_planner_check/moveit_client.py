"""Thin synchronous rclpy client for the standard move_group services (planning only; nothing is executed).

Services used: /compute_ik /compute_fk /compute_cartesian_path /plan_kinematic_path /check_state_validity /apply_planning_scene /get_planning_scene.
Always run with ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1 exported in the same shell (see README / report).
"""
import math, os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import (RobotState, Constraints, JointConstraint, MotionPlanRequest, PlanningScene, CollisionObject,
                             AllowedCollisionMatrix, AllowedCollisionEntry, PlanningSceneComponents, MoveItErrorCodes, WorkspaceParameters)
from moveit_msgs.srv import (GetPositionIK, GetPositionFK, GetCartesianPath, GetMotionPlan, GetStateValidity, ApplyPlanningScene, GetPlanningScene)

import cellgeom as G

ERR = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}


def err_name(code):
    return ERR.get(code, str(code))


def pose_msg(T=None, pose=None):
    p = Pose()
    if T is not None:
        pose = G.T_to_pose(T)
    p.position.x, p.position.y, p.position.z = pose['xyz']
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = pose['quat']
    return p


def msg_to_T(p):
    return G.T_from(G.quat_to_R([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]), [p.position.x, p.position.y, p.position.z])


class MoveItClient(Node):
    def __init__(self, meta, name='cell_planner_check', timeout=60.0):
        super().__init__(name)
        self.meta = meta
        self.arms = list(meta['arms'].keys())            # ['arm1', 'arm2']
        self.joints = {a: meta['arms'][a]['joints'] for a in self.arms}
        self.all_joints = sum([self.joints[a] for a in self.arms], [])
        self.timeout = timeout
        self.cli = dict(
            ik=self.create_client(GetPositionIK, '/compute_ik'),
            fk=self.create_client(GetPositionFK, '/compute_fk'),
            cart=self.create_client(GetCartesianPath, '/compute_cartesian_path'),
            plan=self.create_client(GetMotionPlan, '/plan_kinematic_path'),
            valid=self.create_client(GetStateValidity, '/check_state_validity'),
            apply=self.create_client(ApplyPlanningScene, '/apply_planning_scene'),
            getscene=self.create_client(GetPlanningScene, '/get_planning_scene'))
        for k, c in self.cli.items():
            if not c.wait_for_service(timeout_sec=30.0):
                raise RuntimeError('service %s not available (is the stack up on ROS_DOMAIN_ID=%s?)' % (k, os.environ.get('ROS_DOMAIN_ID')))
        self.ncalls = {k: 0 for k in self.cli}
        self.tcalls = {k: 0.0 for k in self.cli}

    # ---------------------------------------------------------------- plumbing
    def call(self, key, req, timeout=None):
        t0 = time.time()
        fut = self.cli[key].call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout or self.timeout)
        self.ncalls[key] += 1
        self.tcalls[key] += time.time() - t0
        if not fut.done():
            raise TimeoutError('service %s timed out' % key)
        return fut.result()

    # ---------------------------------------------------------------- states
    def state(self, q):
        """q: dict arm -> 6 joint values (rad) -> moveit_msgs/RobotState with all 12 joints"""
        js = JointState()
        js.name = list(self.all_joints)
        js.position = [float(v) for a in self.arms for v in q[a]]
        rs = RobotState()
        rs.joint_state = js
        return rs

    def state_to_q(self, rs):
        d = dict(zip(rs.joint_state.name, rs.joint_state.position))
        return {a: [float(d[j]) for j in self.joints[a]] for a in self.arms}

    # ---------------------------------------------------------------- FK / IK
    def fk(self, q, link):
        req = GetPositionFK.Request()
        req.header.frame_id = 'world'
        req.fk_link_names = [link]
        req.robot_state = self.state(q)
        r = self.call('fk', req)
        if r.error_code.val != 1 or not r.pose_stamped:
            raise RuntimeError('FK failed %s' % err_name(r.error_code.val))
        return msg_to_T(r.pose_stamped[0].pose)

    def ik(self, arm, T, q_seed, avoid_collisions=True, timeout_s=0.5, link=None):
        """returns (q_arm list or None, error_name).  q_seed: dict of all arms (the other arm stays as given)."""
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = arm
        ik.robot_state = self.state(q_seed)
        ik.avoid_collisions = bool(avoid_collisions)
        ik.ik_link_name = link or self.meta['arms'][arm]['tip']
        ik.pose_stamped = PoseStamped()
        ik.pose_stamped.header.frame_id = 'world'
        ik.pose_stamped.pose = pose_msg(T)
        ik.timeout = Duration(seconds=timeout_s).to_msg()
        r = self.call('ik', req, timeout=timeout_s + 20)
        if r.error_code.val != 1:
            return None, err_name(r.error_code.val)
        q = self.state_to_q(r.solution)
        return q[arm], 'SUCCESS'

    # ---------------------------------------------------------------- validity
    def validity(self, q, group='both', want_contacts=True):
        req = GetStateValidity.Request()
        req.robot_state = self.state(q)
        req.group_name = group
        r = self.call('valid', req)
        contacts = sorted({tuple(sorted((c.contact_body_1, c.contact_body_2))) for c in r.contacts}) if want_contacts else []
        return bool(r.valid), contacts

    # ---------------------------------------------------------------- scene
    def get_acm(self):
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        r = self.call('getscene', req)
        return r.scene.allowed_collision_matrix

    def get_scene_objects(self):
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY | PlanningSceneComponents.WORLD_OBJECT_NAMES
        r = self.call('getscene', req)
        return r.scene.world.collision_objects

    def box_object(self, name, size, centre, quat):
        co = CollisionObject()
        co.header.frame_id = 'world'
        co.id = name
        co.operation = CollisionObject.ADD
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.BOX
        sp.dimensions = [float(v) for v in size]
        co.primitives.append(sp)
        co.primitive_poses.append(pose_msg(pose={'xyz': list(map(float, centre)), 'quat': list(map(float, quat))}))
        return co

    def set_scene(self, boxes, remove=()):
        """boxes: list of dict(name,size,centre,quat).  Removes `remove` ids and any previously added ones listed, then adds boxes.  Planning-scene diff."""
        sc = PlanningScene()
        sc.is_diff = True
        existing = {o.id for o in self.get_scene_objects()}
        for nm in remove:
            if nm not in existing:
                continue
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.id = nm
            co.operation = CollisionObject.REMOVE
            sc.world.collision_objects.append(co)
        for b in boxes:
            sc.world.collision_objects.append(self.box_object(b['name'], b['size'], b['centre'], b['quat']))
        req = ApplyPlanningScene.Request()
        req.scene = sc
        r = self.call('apply', req)
        return r.success

    def clear_world(self, names):
        return self.set_scene([], remove=names)

    def set_acm_allow(self, pairs, allowed=True):
        """allow / disallow collisions between named bodies (e.g. armN_knife_tip vs the carton object) through /apply_planning_scene.
        Reads the current ACM, adds missing names, sets the entries and writes the whole matrix back (diff)."""
        acm = self.get_acm()
        names = list(acm.entry_names)
        n0 = len(names)
        for a, b in pairs:
            for nm in (a, b):
                if nm not in names:
                    names.append(nm)
        n = len(names)
        mat = [[False] * n for _ in range(n)]
        for i in range(n0):
            for j in range(n0):
                mat[i][j] = bool(acm.entry_values[i].enabled[j])
        for a, b in pairs:
            i, j = names.index(a), names.index(b)
            mat[i][j] = mat[j][i] = bool(allowed)
        new = AllowedCollisionMatrix()
        new.entry_names = names
        for i in range(n):
            e = AllowedCollisionEntry()
            e.enabled = mat[i]
            new.entry_values.append(e)
        new.default_entry_names = list(acm.default_entry_names)
        new.default_entry_values = list(acm.default_entry_values)
        sc = PlanningScene()
        sc.is_diff = True
        sc.allowed_collision_matrix = new
        req = ApplyPlanningScene.Request()
        req.scene = sc
        return self.call('apply', req).success

    # ---------------------------------------------------------------- planning
    def plan(self, arm, q_start, q_goal_arm, vel, acc, planner='RRTConnectkConfigDefault', pipeline='ompl', planning_time=5.0, attempts=1):
        """joint-space goal for `arm` (the other arm stays at its state in q_start).  returns dict(ok, error, time, traj, duration)"""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request
        mp.group_name = arm
        mp.pipeline_id = pipeline
        mp.planner_id = planner
        mp.num_planning_attempts = attempts
        mp.allowed_planning_time = float(planning_time)
        mp.max_velocity_scaling_factor = float(vel)
        mp.max_acceleration_scaling_factor = float(acc)
        mp.start_state = self.state(q_start)
        c = Constraints()
        for j, v in zip(self.joints[arm], q_goal_arm):
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = float(v)
            jc.tolerance_above = 1e-4
            jc.tolerance_below = 1e-4
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        mp.goal_constraints.append(c)
        r = self.call('plan', req, timeout=planning_time + 30)
        res = r.motion_plan_response
        out = dict(ok=res.error_code.val == 1, error=err_name(res.error_code.val), planning_time=float(res.planning_time))
        if out['ok']:
            out.update(self.traj_summary(res.trajectory.joint_trajectory, arm))
        return out

    def cartesian(self, arm, q_start, waypoints_T, vel, acc, max_step=0.005, avoid_collisions=True, jump_rev=0.0, link=None):
        """Cartesian path of the arm TCP through waypoints (4x4 world poses).  returns dict(fraction, ok, traj, duration, max_joint_step)"""
        req = GetCartesianPath.Request()
        req.header.frame_id = 'world'
        req.start_state = self.state(q_start)
        req.group_name = arm
        req.link_name = link or self.meta['arms'][arm]['tip']
        req.waypoints = [pose_msg(T) for T in waypoints_T]
        req.max_step = float(max_step)
        req.jump_threshold = 0.0
        req.revolute_jump_threshold = float(jump_rev)
        req.prismatic_jump_threshold = 0.0
        req.avoid_collisions = bool(avoid_collisions)
        req.max_velocity_scaling_factor = float(vel)
        req.max_acceleration_scaling_factor = float(acc)
        r = self.call('cart', req, timeout=120)
        out = dict(fraction=float(r.fraction), error=err_name(r.error_code.val))
        out['ok'] = (r.error_code.val == 1 and r.fraction >= 1.0 - 1e-9)
        jt = r.solution.joint_trajectory
        if len(jt.points):
            out.update(self.traj_summary(jt, arm))
        return out

    def traj_summary(self, jt, arm):
        names = list(jt.joint_names)
        idx = [names.index(j) for j in self.joints[arm]]
        P = np.array([[p.positions[i] for i in idx] for p in jt.points])
        T = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points])
        step = np.abs(np.diff(P, axis=0)).max() if len(P) > 1 else 0.0
        return dict(q=P.tolist(), t=T.tolist(), duration=float(T[-1]) if len(T) else 0.0, npoints=len(P), max_joint_step=float(step))


def init():
    if not rclpy.ok():
        rclpy.init()
