"""MoveIt plan-only confirmation of a parking result (rebel_medicine_grasp_moveit_config, mock hardware, isolated domain, nothing executed).
The real cabinet (axis-aligned part boxes + carton blocks, target carton left out = it is the grasped object) is put into the planning scene at the parking pose; lift / sliders from the
parking result; then the whole sequence is planned: free-space transits with OMPL (joint goal = my IK branch, collision checked by MoveIt) and every straight segment with
/compute_cartesian_path (fraction must be 1.0, avoid_collisions=True).
  source /opt/ros/humble/setup.bash; source <ws>/install/setup.bash; export ROS_DOMAIN_ID=172 ROS_LOCALHOST_ONLY=1
  python3 tools/moveit_confirm.py --parking tests/results/amr_parking_front.json --mode front --targets T3,T8 --out tests/results/moveit_confirm_front.json"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene, RobotState, Constraints, JointConstraint, MotionPlanRequest, WorkspaceParameters
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetMotionPlan
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as Rot
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parking as pk, parking_study as ps, robot_kin as rk

TIP = {'left': 'arm_left_suction_tcp_compressed', 'right': 'arm_right_gripper_tcp'}


def pose_msg(T):
    q = Rot.from_matrix(T[:3, :3]).as_quat()
    p = Pose()
    p.position.x, p.position.y, p.position.z = [float(v) for v in T[:3, 3]]
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = [float(v) for v in q]
    return p


class C(Node):
    def __init__(self):
        super().__init__('moveit_confirm')
        self.scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.cart = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.plan = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        for c in (self.scene, self.cart, self.plan):
            assert c.wait_for_service(timeout_sec=60.0), c.srv_name

    def call(self, cli, req, timeout=120.0):
        f = cli.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=timeout)
        return f.result()


def state_msg(names, vals):
    rs = RobotState()
    rs.joint_state = JointState(name=list(names), position=[float(v) for v in vals])
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parking', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--targets', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rclpy.init()
    node = C()
    R, cab = rk.Robot(), rk.Cabinet()
    pr = {r['target']: r for r in json.load(open(a.parking)) if r['mode'] == a.mode and r.get('feasible')}
    allres = {}
    for tn in a.targets.split(','):
        shelf, box, which = ps.TARGETS[tn]
        tg = cab.pick(shelf, box, which)
        par = pr[tn]
        pull = ps.PULL_BY_MODE_SHELF.get((a.mode, shelf), rk.PULL_OUT)
        lift = par['lift']
        W = rk.World(cab, shelf, tg, par['best_d'], par['lateral_offset_m'])
        # planning scene: clear, then add the cabinet
        co = []
        for nm, (c, half) in W.static.items():
            o = CollisionObject()
            o.header.frame_id = 'world'
            o.id = nm.replace(':', '_')
            sp = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(2 * h) for h in half])
            o.primitives = [sp]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = W.R, W.to_world(c)
            o.primitive_poses = [pose_msg(T)]
            o.operation = CollisionObject.ADD
            co.append(o)
        ps_ = PlanningScene(is_diff=True)
        ps_.world.collision_objects = co
        r_ = node.call(node.scene, ApplyPlanningScene.Request(scene=ps_))
        assert r_ and r_.success, 'apply_planning_scene failed'
        ev, seqs = ps.make_eval(R, cab, tg, shelf, a.mode, pull, par['best_d'], par['lateral_offset_m'], tuple(lift) if isinstance(lift, list) else lift, par['slider_left'], par['slider_right'])
        res = ev.evaluate(seqs, record=True)
        assert res['ok']
        liftv = lift if isinstance(lift, list) else [lift, lift]
        names = ['lift_joint', 'slider_left_joint', 'slider_right_joint'] + [f'arm_left_joint{k}' for k in range(1, 7)] + [f'arm_right_joint{k}' for k in range(1, 7)] + ['arm_right_finger_left_joint']
        rows = []
        qL, qR = np.zeros(6), np.zeros(6)
        fin = 0.024
        for arm, seq in (('left', res['left_seq']), ('right', res['right_seq'])):
            lf = liftv[0] if arm == 'left' else liftv[1]
            qarm = np.zeros(6)
            segs = {s.name: s for s in seq}
            for tr in res[arm]['traj']:
                sg = segs[tr['name']]
                other = qR if arm == 'left' else qL
                vals = [lf, par['slider_left'], par['slider_right']] + list(qarm if arm == 'left' else qL) + list(qarm if arm == 'right' else qR) + [fin]
                if arm == 'left':
                    vals = [lf, par['slider_left'], par['slider_right']] + list(qarm) + list(qR) + [fin]
                else:
                    vals = [lf, par['slider_left'], par['slider_right']] + list(qL) + list(qarm) + [tr['fingers']]
                t0 = time.time()
                row = dict(arm=arm, segment=sg.name)
                if sg.kind == 'joint':
                    req = GetMotionPlan.Request()
                    mp = req.motion_plan_request
                    mp.group_name = 'arm_' + arm
                    mp.start_state = state_msg(names, vals)
                    mp.num_planning_attempts = 5
                    mp.allowed_planning_time = 20.0
                    mp.max_velocity_scaling_factor = 0.5
                    mp.max_acceleration_scaling_factor = 0.5
                    gc = Constraints()
                    for k, v in enumerate(tr['q'][-1]):
                        gc.joint_constraints.append(JointConstraint(joint_name=f'arm_{arm}_joint{k + 1}', position=float(v), tolerance_above=1e-3, tolerance_below=1e-3, weight=1.0))
                    mp.goal_constraints = [gc]
                    resp = node.call(node.plan, req, 60.0)
                    ok = bool(resp and resp.motion_plan_response.error_code.val == 1)
                    row.update(kind='OMPL joint goal', ok=ok, error=(resp.motion_plan_response.error_code.val if resp else None),
                               points=(len(resp.motion_plan_response.trajectory.joint_trajectory.points) if ok else 0))
                else:
                    req = GetCartesianPath.Request()
                    req.header.frame_id = 'world'
                    req.start_state = state_msg(names, vals)
                    req.group_name = 'arm_' + arm
                    req.link_name = TIP[arm]
                    req.waypoints = [pose_msg(sg.T1)] if np.linalg.norm(sg.T1[:3, 3] - sg.T0[:3, 3]) > 1e-6 else []
                    req.max_step = 0.005
                    req.jump_threshold = 0.0
                    req.avoid_collisions = True
                    if req.waypoints:
                        resp = node.call(node.cart, req, 60.0)
                        row.update(kind='Cartesian', fraction=float(resp.fraction), ok=bool(resp.fraction > 0.999), error=int(resp.error_code.val),
                                   points=len(resp.solution.joint_trajectory.points))
                    else:
                        row.update(kind='hold', ok=True)
                row['plan_time_s'] = round(time.time() - t0, 2)
                print(tn, row, flush=True)
                rows.append(row)
                qarm = np.asarray(tr['q'])[-1]
                if arm == 'left':
                    qL = qarm
                else:
                    qR = qarm
                fin = tr['fingers'] if arm == 'right' else fin
            if arm == 'left':
                qL = np.zeros(6) if res['stage_variant'] is not None and seqs[res['stage_variant'][0]]['left_final'] == 'home' else qL
        allres[tn] = dict(parking={k: par[k] for k in ('best_d', 'lateral_offset_m', 'lift', 'slider_left', 'slider_right')}, segments=rows, all_ok=all(r['ok'] for r in rows))
        print(tn, 'ALL OK' if allres[tn]['all_ok'] else 'SOME FAILED', flush=True)
        rm = PlanningScene(is_diff=True)
        for o in co:
            o2 = CollisionObject()
            o2.id = o.id
            o2.operation = CollisionObject.REMOVE
            rm.world.collision_objects.append(o2)
        node.call(node.scene, ApplyPlanningScene.Request(scene=rm))
    json.dump(allres, open(a.out, 'w'), indent=1)
    rclpy.shutdown()


main()
