#!/usr/bin/env python3
"""MOCK ONLY - drives the MOCK single-arm stack through calibration poses and calls the recorder.  NOT FOR REAL ROBOTS.

It stands in for the operator moving the arm (RViz) and pressing "capture".  It sends joint trajectories to the
mock JTC action, waits until the arm is still and calls the recorder's services, exactly like the operator does:

    # ~30 varied, visible, collision-checked poses, one capture per pose
    ros2 run rebel_dual_arm_calibration mock_pose_sequence.py poses --arm left  --n 30 --log left_poses.jsonl
    # touch the table-board points with the TCP (xeg32_tool_tip) at the hidden-truth positions, 3 wrist orientations
    ros2 run rebel_dual_arm_calibration mock_pose_sequence.py touch --arm left  --points 1,2,3,4 --orientations 3
    # ... and the free stickers of config/mock_truth.yaml (fit 101-104, check 111-112; ids >= 100 are stickers)
    ros2 run rebel_dual_arm_calibration mock_pose_sequence.py touch --arm left  --points 101,102,103,104,111,112

``--arm`` tells which ground-truth base the running single-arm stack represents (left: world = base_link, right: the
hidden truth of arm_right_base).  Touch points are computed from the hidden table-board pose in config/mock_truth.yaml
and expressed in that arm's base frame; ``--touch-offset-mm dx,dy,dz`` (world frame) displaces the touched positions
(negative test: an arm that touches the wrong place).  Touch parameters are set with the real CLI
(``ros2 param set`` / ``ros2 service call``, as the README tells the operator) unless ``--no-cli`` is given.
Needs: mock rebel.launch.py (JTC), move_group (check_state_validity, optional), mock_scene_camera.py, the recorder.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetStateValidity
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
from scipy.optimize import least_squares

from rebel_dual_arm_calibration.kinematics import ArmKinematics, Urdf
from rebel_dual_arm_calibration.mock_common import arm_board_visibility, load_truth
from rebel_dual_arm_calibration.targets import default_targets_path, load_targets_all
from rebel_dual_arm_calibration.transforms import inv_T

JOINTS = [f"joint{i}" for i in range(1, 7)]
LO = np.array([-3.10, -1.37, -1.37, -3.10, -1.64, -3.10])
HI = np.array([3.10, 2.42, 2.42, 3.10, 1.64, 3.10])
HOME = np.array([0.0, 0.8, 1.1, 0.0, 0.7, 0.0])


class Driver(Node):
    def __init__(self, arm: str, recorder: str, controller: str, truth_file: str):
        super().__init__("mock_pose_sequence")
        self.arm = arm
        self.recorder = "/" + recorder.strip("/")
        self.truth = load_truth(truth_file or None)
        self.specs, self.table = load_targets_all(default_targets_path())
        self.q: Optional[np.ndarray] = None
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)
        self.traj = ActionClient(self, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        self.cli_capture = self.create_client(Trigger, f"{self.recorder}/capture")
        self.cli_status = self.create_client(Trigger, f"{self.recorder}/status")
        self.cli_valid = self.create_client(GetStateValidity, "/check_state_validity")
        self.kin = self._load_kinematics()

    def _on_js(self, msg: JointState):
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in JOINTS):
            self.q = np.array([d[j] for j in JOINTS])

    def _load_kinematics(self) -> ArmKinematics:
        cli = self.create_client(GetParameters, "/robot_state_publisher/get_parameters")
        if not cli.wait_for_service(20.0):
            raise RuntimeError("robot_state_publisher not found (is the mock single-arm stack running?)")
        req = GetParameters.Request()
        req.names = ["robot_description"]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        urdf = Urdf.parse(fut.result().values[0].string_value)
        return ArmKinematics.from_urdf(urdf, "base_link", "link6", "xeg32_tool_tip")

    # ---- helpers --------------------------------------------------------------------------------------------
    def spin_for(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_joints(self):
        end = time.monotonic() + 15
        while self.q is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.q is None:
            raise RuntimeError("no /joint_states")

    def call(self, client, req=None, timeout=15.0):
        if not client.wait_for_service(10.0):
            raise RuntimeError(f"service {client.srv_name} not available")
        fut = client.call_async(req if req is not None else Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def valid(self, q: np.ndarray) -> bool:
        if not self.cli_valid.service_is_ready():
            return True        # move_group not up: no collision information, accept
        req = GetStateValidity.Request()
        req.group_name = "igus_rebel_arm"
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = [float(v) for v in q]
        res = self.call(self.cli_valid, req)
        return bool(res is not None and res.valid)

    def move(self, q_goal: np.ndarray, min_s: float = 2.5, mid_probe=None) -> float:
        """Joint-space move with a 2-point trajectory; returns the duration.  mid_probe() is called mid-way."""
        self.wait_joints()
        q0 = self.q.copy()
        dur = max(min_s, float(np.max(np.abs(q_goal - q0))) / 0.35 + 1.0)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        p1 = JointTrajectoryPoint()
        p1.positions = [float(v) for v in q_goal]
        p1.velocities = [0.0] * 6
        p1.time_from_start.sec = int(dur)
        p1.time_from_start.nanosec = int((dur % 1) * 1e9)
        goal.trajectory.points = [p1]
        if not self.traj.wait_for_server(10.0):
            raise RuntimeError("follow_joint_trajectory action server not found")
        fut = self.traj.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            raise RuntimeError("trajectory goal rejected")
        res_fut = gh.get_result_async()
        t_start = time.monotonic()
        probed = False
        while not res_fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if mid_probe is not None and not probed and time.monotonic() - t_start > 0.5 * dur:
                probed = True
                mid_probe()
        result = res_fut.result()
        if result.result.error_code != 0:
            raise RuntimeError(f"trajectory failed: {result.result.error_code} {result.result.error_string}")
        return dur

    def capture(self) -> tuple:
        res = self.call(self.cli_capture)
        return bool(res.success), res.message

    def T_base_tcp(self, q) -> np.ndarray:
        return self.kin.chain_tcp.fk(np.asarray(q, float))

    def T_base_link6(self, q) -> np.ndarray:
        return self.kin.chain6.fk(np.asarray(q, float))

    # ---- pose generation ----------------------------------------------------------------------------------
    def candidates(self, n_cand: int, seed: int, view_deg: float, margin_px: float):
        rng = np.random.default_rng(seed)
        lo = np.array([-1.1, 0.25, 0.25, -2.6, -1.35, -3.0])
        hi = np.array([1.1, 1.35, 1.9, 2.6, 1.35, 3.0])
        Q = rng.uniform(lo, hi, (n_cand, 6))
        T6 = self.kin.chain6.fk(Q)
        Ttcp = self.kin.chain_tcp.fk(Q)
        Tb = self.truth.T_w_base(self.arm)
        Tm = self.truth.T_l6_target[self.arm]
        spec = self.specs[self.arm]
        out = []
        for i in range(n_cand):
            p6, ptcp = T6[i][:3, 3], Ttcp[i][:3, 3]
            if not (0.12 < p6[0] < 0.60 and abs(p6[1]) < 0.40 and 0.14 < p6[2] < 0.65 and ptcp[2] > 0.05
                    and ptcp[0] > 0.12):
                continue
            ok, _, ang = arm_board_visibility(self.truth, spec, Tb @ T6[i] @ Tm, margin_px=margin_px,
                                              max_view_deg=view_deg)
            if ok:
                out.append((Q[i], T6[i], ang))
        return out

    def select_poses(self, n: int, seed: int, view_deg: float = 55.0, margin_px: float = 100.0):
        cands = self.candidates(20000, seed, view_deg, margin_px)
        if len(cands) < n:
            raise RuntimeError(f"only {len(cands)} visible candidate poses")
        Qs = np.array([c[0] for c in cands])
        feat = np.hstack([(Qs - LO) / (HI - LO) * 1.5,
                          np.array([c[1][:3, 3] for c in cands]) / 0.35,
                          np.array([c[1][:3, 2] for c in cands]) * 1.0,
                          np.array([c[1][:3, 0] for c in cands]) * 1.0])
        rng = np.random.default_rng(seed + 1)
        chosen: List[int] = []
        d = np.full(len(cands), np.inf)
        nxt = int(rng.integers(len(cands)))
        while len(chosen) < n:
            if nxt in chosen:
                break
            if self.valid(Qs[nxt]):
                chosen.append(nxt)
                d = np.minimum(d, np.linalg.norm(feat - feat[nxt], axis=1))
            else:
                d[nxt] = -1.0
            if len(chosen) < n:
                nxt = int(np.argmax(d))
        # order: greedy nearest-neighbour tour in joint space starting near the current pose
        order, rest = [], list(chosen)
        cur = self.q if self.q is not None else HOME
        while rest:
            j = min(rest, key=lambda k: np.linalg.norm(Qs[k] - cur))
            order.append(j)
            rest.remove(j)
            cur = Qs[j]
        return [(Qs[k], cands[k][2]) for k in order]

    # ---- IK for touches ---------------------------------------------------------------------------------
    def ik(self, p_base: np.ndarray, z_des: np.ndarray, seed: np.ndarray, tries: int = 120, rng=None):
        """Joint values that put the TCP at p_base with its z axis (tool axis) along z_des; the roll about the tool
        axis is free.  Returns the solution nearest to ``seed`` among the converged random restarts."""
        rng = rng or np.random.default_rng(0)

        def res(q):
            T = self.kin.chain_tcp.fk(q)
            return np.concatenate([(T[:3, 3] - p_base) * 1000.0, 300.0 * np.cross(T[:3, 2], z_des)])

        best = None
        n_ok = 0
        for t in range(tries):
            q0 = np.clip(seed + (0 if t == 0 else rng.normal(0, 0.4 + 0.02 * t, 6)), LO + 1e-3, HI - 1e-3)
            sol = least_squares(res, q0, bounds=(LO, HI), xtol=1e-12, ftol=1e-12, gtol=1e-12)
            T = self.kin.chain_tcp.fk(sol.x)
            err_mm = np.linalg.norm(T[:3, 3] - p_base) * 1000
            err_deg = np.degrees(np.arccos(np.clip(T[:3, 2] @ z_des, -1, 1)))
            if err_mm < 0.01 and err_deg < 0.05:
                n_ok += 1
                cost = np.linalg.norm(sol.x - seed)
                if best is None or cost < best[0]:
                    best = (cost, sol.x)
                if n_ok >= 6:
                    break
        return None if best is None else best[1]


def tool_axis_down(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Tool z axis pointing down, tilted by ``tilt`` towards ``azimuth`` (about the base z axis, from +x)."""
    t, a = np.radians([tilt_deg, azimuth_deg])
    return np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), -math.cos(t)])


# ---------------------------------------------------------------------------------------------------------------
def cmd_poses(drv: Driver, args) -> int:
    drv.wait_joints()
    for c in (drv.cli_capture,):
        if not c.wait_for_service(20.0):
            raise RuntimeError("recorder services not found (record.launch.py running?)")
    print(drv.call(drv.cli_status).message)
    poses = drv.select_poses(args.n, args.seed)
    print(f"{len(poses)} candidate poses selected (collision-checked, board visible)")
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    ok_n, probes = 0, []
    reasons: Dict[str, int] = {}
    for i, (q, ang) in enumerate(poses):
        probe_mid, probe_end = [], []

        def mid():
            probe_mid.append(drv.capture())

        dur = drv.move(q, mid_probe=mid if i < args.probe_first else None)
        if i < args.probe_first:
            probe_end.append(drv.capture())           # immediately at the goal: must be refused (not still long enough)
        drv.spin_for(args.settle_s)
        success, msg = False, ""
        for attempt in range(args.retries + 1):
            success, msg = drv.capture()
            retry = ("not stationary" in msg or "predates" in msg or "stale" in msg or "newest image" in msg
                     or "no earlier image" in msg or "still moving" in msg)
            if success and not msg.startswith("NO ARM SAMPLE"):
                break
            if retry:
                drv.spin_for(0.8)          # the arm is still settling (2 mrad stationary tolerance) or the board shakes
            else:
                break
        rec = {"i": i, "q": [round(float(v), 5) for v in q], "view_deg": round(ang, 1), "move_s": round(dur, 2),
               "success": success, "message": msg, "probe_mid_motion": probe_mid, "probe_at_goal": probe_end}
        print(f"[{i + 1}/{len(poses)}] {'OK ' if success else 'REJ'} {msg}")
        if probe_mid or probe_end:
            print(f"      probes: mid-motion {probe_mid}  at-goal {probe_end}")
        ok_n += bool(success)
        if not success:
            key = msg.split(":", 1)[-1].strip()[:60]
            reasons[key] = reasons.get(key, 0) + 1
        if log:
            log.write(json.dumps(rec) + "\n")
            log.flush()
    print(f"\ncaptures accepted {ok_n}/{len(poses)}; rejection reasons: {reasons}")
    print(drv.call(drv.cli_status).message)
    return 0


def run_cli(cmd: str, retries: int = 2) -> str:
    """Run a ros2 CLI command (the operator's way); a CLI that hangs in discovery is retried (a repeated touch is harmless)."""
    env = dict(os.environ)
    for attempt in range(retries + 1):
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=45, env=env,
                                 executable="/bin/bash")
            return (out.stdout + out.stderr).strip()
        except subprocess.TimeoutExpired:
            print(f"   (CLI timed out, attempt {attempt + 1}: {cmd})", flush=True)
    return "CLI timed out"


def cmd_touch(drv: Driver, args) -> int:
    drv.wait_joints()
    truth, table = drv.truth, drv.table
    off = np.array([float(v) for v in args.touch_offset_mm.split(",")]) * 1e-3 if args.touch_offset_mm else np.zeros(3)
    T_base_w = inv_T(truth.T_w_base(drv.arm))
    ids = [int(v) for v in args.points.split(",")]
    # (tilt, azimuth) of the tool axis from the vertical; consecutive touches of one point differ by >= 25 deg
    orients = [(0, 0), (30, 90), (30, 270), (30, 180), (30, 0), (45, 135)][: args.orientations]
    if args.repeat_same > 1:        # the same orientation several times (repeatability evidence), lifted in between
        orients = [orients[0]] * args.repeat_same
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng(args.seed + 1000)
    n_ok = 0
    for pid in ids:
        if pid in table.touch_points:
            x, y = table.touch_points[pid]
            p_w = (truth.T_w_board @ np.array([x, y, 0.0, 1.0]))[:3] + off
        elif pid in truth.stickers:
            p_w = truth.stickers[pid] + off
        else:
            print(f"point {pid}: neither a table-board touch point nor a sticker of the truth file, skipped")
            continue
        p_b = (T_base_w @ np.append(p_w, 1.0))[:3]
        for (tilt, az) in orients:
            p_touch = p_b + (noise_rng.normal(0, args.touch_noise_mm * 1e-3, 3) if args.touch_noise_mm > 0 else 0.0)
            z_des = tool_axis_down(tilt, az)
            seed = drv.q.copy() if np.all(drv.q != 0) else HOME
            q_up = drv.ik(p_touch + 0.05 * (-z_des), z_des, seed, rng=rng)   # 50 mm above the point along the tool axis
            q_dn = None if q_up is None else drv.ik(p_touch, z_des, q_up, rng=rng)
            if q_dn is None or not (drv.valid(q_up) and drv.valid(q_dn)):
                print(f"point {pid} tilt {tilt} az {az}: no reachable collision-free IK solution, skipped")
                continue
            drv.move(q_up)
            drv.move(q_dn, min_s=2.0)
            drv.spin_for(args.settle_s)
            T = drv.T_base_tcp(q_dn)
            print(f"point {pid} orientation (tilt {tilt}, az {az}): tcp target {np.round(p_touch, 4)} "
                  f"reached {np.round(T[:3, 3], 4)}")
            if args.no_cli:
                msg = "(no-cli) skipped"
            else:
                r1 = run_cli(f"ros2 param set {drv.recorder} touch_point_id {pid}")
                for attempt in range(4):
                    r2 = run_cli(f"ros2 service call {drv.recorder}/capture_touch std_srvs/srv/Trigger")
                    if "success=True" in r2 or not any(k in r2 for k in ("still moving", "not stationary", "no earlier image",
                                                                         "predates")):
                        break
                    drv.spin_for(1.5)        # still settling / the two-frame check saw motion: wait and try again
                msg = r1 + " | " + r2.replace("\n", " ")
            print("   " + msg)
            ok = "success=True" in msg or "success: True" in msg
            n_ok += ok
            if log:
                log.write(json.dumps({"point": pid, "tilt": tilt, "az": az, "target_base": np.asarray(p_touch).tolist(),
                                      "response": msg}) + "\n")
                log.flush()
            drv.move(q_up)
    print(f"\ntouches recorded {n_ok}")
    drv.move(HOME)
    print(drv.call(drv.cli_status).message)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MOCK ONLY pose / touch driver for the calibration recorder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("poses", "touch"):
        s = sub.add_parser(name)
        s.add_argument("--arm", choices=["left", "right"], required=True,
                       help="which ground-truth base the running single-arm stack represents")
        s.add_argument("--recorder", default="dual_arm_calibration_recorder")
        s.add_argument("--controller", default="rebel_arm_trajectory_controller")
        s.add_argument("--truth-file", default="")
        s.add_argument("--log", default="")
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--settle-s", type=float, default=0.8, help="wait after the move before capturing")
        if name == "poses":
            s.add_argument("--n", type=int, default=30)
            s.add_argument("--retries", type=int, default=4)
            s.add_argument("--probe-first", type=int, default=3,
                           help="for the first N poses also call capture mid-motion and right at the goal "
                                "(expected to be refused: proves the stationary check)")
        else:
            s.add_argument("--points", default="1,2,3,4")
            s.add_argument("--orientations", type=int, default=3)
            s.add_argument("--repeat-same", type=int, default=1,
                           help="touch each point this many times at the SAME wrist orientation (RP evidence)")
            s.add_argument("--touch-offset-mm", default="", help="world-frame displacement dx,dy,dz of the touched points")
            s.add_argument("--touch-noise-mm", type=float, default=0.0,
                           help="1-sigma per axis of a random contact error added to EVERY touch (the operator's hand; "
                                "the mock arm itself is exact), seeded by --seed")
            s.add_argument("--no-cli", action="store_true", help="do not call the ros2 CLI (print only)")
    args = ap.parse_args(argv)
    rclpy.init()
    drv = Driver(args.arm, args.recorder, args.controller, args.truth_file)
    try:
        return cmd_poses(drv, args) if args.cmd == "poses" else cmd_touch(drv, args)
    finally:
        drv.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
