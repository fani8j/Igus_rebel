#!/usr/bin/env python3
"""Open-loop jog speed check: how fast does one joint really move per jog percent?

Uses the driver's direct joint-jog mode (~/joint_jog_enable + ~/joint_jog_cmd; the trajectory
controller must be INACTIVE), so no feedback controller is involved: the driver sends
jog % = jog_velocity_scale * commanded deg/s and the robot's own speed is measured from
~/robot_joint_states over the constant-speed part of each run.

For each percent the joint is jogged by about --travel-deg, alternating + / - so the arm ends
near where it started. A position guard sends zero early if the joint leaves
start +- --max-excursion-deg. Ctrl+C sends zero and leaves the jog mode; the driver's own 0.15 s
command watchdog zeroes the jog if this script dies.

    python3 tools/jog_speed_check.py --joint joint1 --percents 10 20 30 40 --real --approved
"""

import argparse
import json
import math
import os
import re
import sys
import time

import rclpy
from control_msgs.msg import JointJog
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

AMAX_DEG_S2 = math.degrees(0.50)   # robot jog acceleration limit (10-26 fit)
RAMP_MARGIN_S = 0.25               # delay + lag margin after the ramp before the cruise window
PUBLISH_HZ = 50.0
MAX_PERCENT = 40.0
MAX_TRAVEL_DEG = 45.0
MAX_EXCURSION_DEG = 35.0
RETURN_PERCENT = 10.0
# Travel per run by jog percent: long enough for a >= 0.5 s constant-speed window. With the
# alternating direction 10/20/30/40 % go 0 -> +20 -> 0 -> +30 -> -15, then a 10 % return to start.
TRAVEL_BY_PERCENT = ((10, 20.0), (20, 20.0), (30, 30.0), (40, 45.0))


def cruise_velocity(samples, t_start, t_end):
    """Least-squares slope [deg/s] of (t, pos_deg) samples inside [t_start, t_end]; None if < 5."""
    pts = [(t, p) for t, p in samples if t_start <= t <= t_end]
    if len(pts) < 5:
        return None, len(pts)
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    mp = sum(p for _, p in pts) / n
    stt = sum((t - mt) ** 2 for t, _ in pts)
    if stt <= 0.0:
        return None, n
    return sum((t - mt) * (p - mp) for t, p in pts) / stt, n


SWEEP_MAX_PERCENT = 100.0
SWEEP_MAX_LIMIT_DEG = 90.0
GUARD_ACCEL_DEG_S2 = math.degrees(0.8)    # predictive guard braking: measured 1.45-1.75 rad/s^2 on a zero jog (10-31), half of it


def stopping_distance_deg(v_deg_s, accel_deg_s2=GUARD_ACCEL_DEG_S2):
    return v_deg_s * v_deg_s / (2.0 * accel_deg_s2)


def guard_trips(pos_deg, vel_deg_s, limit_deg, accel_deg_s2=GUARD_ACCEL_DEG_S2):
    """True if the joint, decelerating from vel at accel, would end beyond +-limit."""
    end = pos_deg + math.copysign(stopping_distance_deg(vel_deg_s, accel_deg_s2), vel_deg_s)
    return abs(end) > limit_deg


def recent_velocity(samples, window_s=0.08):
    if len(samples) < 3:
        return 0.0
    t1, p1 = samples[-1]
    for t0, p0 in reversed(samples):
        if t1 - t0 >= window_s:
            return (p1 - p0) / (t1 - t0)
    return 0.0


def plan_runs(percents, scale, travel_deg):
    """[(sign, percent, cmd_deg_s, duration_s, cruise_window_s)] with alternating direction."""
    runs = []
    for k, pct in enumerate(percents):
        v = pct / scale
        travel = next((t for p, t in TRAVEL_BY_PERCENT if pct <= p), MAX_TRAVEL_DEG)
        duration = travel / v
        ramp = v / AMAX_DEG_S2
        runs.append((1 if k % 2 == 0 else -1, pct, v, duration, duration - ramp - RAMP_MARGIN_S))
    return runs


class JogCheck(Node):
    def __init__(self, driver, joint):
        super().__init__('jog_speed_check')
        self.joint = joint
        self.samples = []          # (t_stamp_s, pos_deg) from robot_joint_states
        self.latest = None
        self.create_subscription(JointState, f'/{driver}/robot_joint_states', self._on_js,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(JointJog, f'/{driver}/joint_jog_cmd', 2)
        self.enable = self.create_client(SetBool, f'/{driver}/joint_jog_enable')
        self.names = None

    def _on_js(self, msg):
        if self.names is None:
            self.names = list(msg.name)
        if self.joint not in msg.name:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = math.degrees(msg.position[msg.name.index(self.joint)])
        self.samples.append((t, p))
        self.latest = (t, p)

    def spin_for(self, s):
        end = time.monotonic() + s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.005)

    def set_mode(self, on):
        if not self.enable.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('joint_jog_enable service not available')
        fut = self.enable.call_async(SetBool.Request(data=on))
        end = time.monotonic() + 5.0
        while not fut.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
        res = fut.result()
        if res is None or not res.success:
            raise RuntimeError('joint_jog_enable %s refused: %s' % (on, res.message if res else 'timeout'))
        return res.message

    def send(self, vel_rad_s):
        m = JointJog()
        m.header.stamp = self.get_clock().now().to_msg()
        m.joint_names = list(self.names)
        m.velocities = [vel_rad_s if n == self.joint else 0.0 for n in self.names]
        self.pub.publish(m)


def robot_description_info():
    """Reads robot_ip, jog_velocity_scale and joint_jog_max_percent from robot_state_publisher."""
    import subprocess
    out = subprocess.run(['ros2', 'param', 'get', '/robot_state_publisher', 'robot_description'],
                         capture_output=True, text=True, timeout=20).stdout
    def grab(name):
        m = re.search(r'name="%s"\s*>\s*([^<\s]+)\s*<' % name, out)
        return m.group(1) if m else None
    return grab('robot_ip'), grab('jog_velocity_scale'), grab('joint_jog_max_percent')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--joint', default='joint1')
    ap.add_argument('--percents', type=float, nargs='+', default=[10, 20, 30, 40])
    ap.add_argument('--driver', default='igus_rebel')
    ap.add_argument('--real', action='store_true', help='robot_ip must be 192.168.3.11')
    ap.add_argument('--fake', action='store_true', help='robot_ip must be 127.0.0.1')
    ap.add_argument('--approved', action='store_true', help='operator approved this motion in chat')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sweep', type=float, default=None,
                    help='single open-loop run at this jog %% (up to 100), no return; needs --travel-deg, --sign, --limit-deg')
    ap.add_argument('--travel-deg', type=float, default=None, help='--sweep: planned travel (open loop, by time)')
    ap.add_argument('--sign', type=int, choices=(-1, 1), default=1, help='--sweep: direction')
    ap.add_argument('--guard-ends-run', action='store_true',
                    help='--sweep: allow a planned end beyond --limit-deg (the predictive guard ends the run)')
    ap.add_argument('--limit-deg', type=float, default=None,
                    help='--sweep: absolute joint bound; zero is sent early if position + stopping distance would pass it')
    a = ap.parse_args()
    if a.sweep is not None:
        return sweep_main(a)

    if os.environ.get('ROS_DOMAIN_ID') in (None, '', '11') or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        sys.exit('REFUSED: set ROS_DOMAIN_ID (not 11) and ROS_LOCALHOST_ONLY=1 explicitly')
    if a.real == a.fake:
        sys.exit('REFUSED: exactly one of --real / --fake')
    if a.real and not a.approved:
        sys.exit('REFUSED: --real needs --approved')
    if any(not 0 < p <= MAX_PERCENT for p in a.percents):
        sys.exit('REFUSED: percents must be in (0, %g]' % MAX_PERCENT)
    ip, scale_s, maxpct_s = robot_description_info()
    want_ip = '192.168.3.11' if a.real else '127.0.0.1'
    if ip != want_ip:
        sys.exit('REFUSED: robot_ip %s != %s' % (ip, want_ip))
    scale = float(scale_s) if scale_s else 2.0
    maxpct = float(maxpct_s) if maxpct_s else 10.0
    if max(a.percents) > maxpct:
        sys.exit('REFUSED: driver joint_jog_max_percent is %g < %g (relaunch with joint_jog_max_percent:=%g)'
                 % (maxpct, max(a.percents), max(a.percents)))
    runs = plan_runs(a.percents, scale, None)
    pos = 0.0
    for sign, pct, v, dur, cw in runs:
        pos += sign * v * dur
        if abs(pos) > MAX_EXCURSION_DEG - 2.0:
            sys.exit('REFUSED: planned excursion %.1f deg exceeds %g deg (reorder the percents)' % (pos, MAX_EXCURSION_DEG))
    print('scale %.4f, joint_jog_max_percent %g, joint %s' % (scale, maxpct, a.joint))
    for sign, pct, v, dur, cw in runs:
        print('  %+d x %5.1f %%  = %6.2f deg/s for %.2f s (cruise window %.2f s)' % (sign, pct, v, dur, cw))

    rclpy.init()
    node = JogCheck(a.driver, a.joint)
    results = []
    entered = False
    try:
        node.spin_for(1.0)
        if node.latest is None:
            raise RuntimeError('no robot_joint_states')
        start = node.latest[1]
        print('start %s = %.3f deg' % (a.joint, start))
        print(node.set_mode(True))
        entered = True
        node.spin_for(0.3)
        for sign, pct, v, dur, cw in runs:
            vel = sign * math.radians(v)
            t0 = node.get_clock().now().nanoseconds * 1e-9
            guard = False
            period = 1.0 / PUBLISH_HZ
            next_t = time.monotonic()
            end = next_t + dur
            while time.monotonic() < end:
                if node.latest and abs(node.latest[1] - start) > MAX_EXCURSION_DEG:
                    guard = True
                    break
                node.send(vel)
                next_t += period
                while time.monotonic() < next_t:
                    rclpy.spin_once(node, timeout_sec=0.002)
            t1 = node.get_clock().now().nanoseconds * 1e-9
            stop_end = time.monotonic() + 1.8
            while time.monotonic() < stop_end:
                node.send(0.0)
                node.spin_for(period)
            ramp = v / AMAX_DEG_S2
            meas, n = cruise_velocity(node.samples, t0 + ramp + RAMP_MARGIN_S, t1)
            r = dict(sign=sign, percent=pct, cmd_deg_s=v, duration_s=t1 - t0, n=n, guard=guard,
                     measured_deg_s=None if meas is None else abs(meas))
            if meas is not None:
                r['ratio_measured_over_cmd'] = abs(meas) / v
                r['deg_s_per_percent'] = abs(meas) / pct
                r['scale_for_exact_speed'] = pct / abs(meas)
                print('  %+d %5.1f %%: cmd %6.2f deg/s  measured %6.2f deg/s  ratio %.3f  -> scale %.3f  (%d samples)%s'
                      % (sign, pct, v, abs(meas), abs(meas) / v, pct / abs(meas), n, '  GUARD' if guard else ''))
            else:
                print('  %+d %5.1f %%: no cruise window (%d samples)' % (sign, pct, n))
            results.append(r)
            if guard:
                print('position guard hit: stopping the series')
                break
        # Return to the start at 10 %, open loop by time, then report the residual.
        off = node.latest[1] - start
        if abs(off) > 0.5:
            v = RETURN_PERCENT / scale
            vel = -math.copysign(math.radians(v), off)
            end = time.monotonic() + abs(off) / v
            print('return: %.2f deg at %g %%' % (-off, RETURN_PERCENT))
            while time.monotonic() < end:
                node.send(vel)
                node.spin_for(1.0 / PUBLISH_HZ)
            stop_end = time.monotonic() + 1.0
            while time.monotonic() < stop_end:
                node.send(0.0)
                node.spin_for(1.0 / PUBLISH_HZ)
    except KeyboardInterrupt:
        print('Ctrl+C: zero and leave')
    except RuntimeError as exc:
        print('ERROR: %s' % exc)
    finally:
        for _ in range(10):
            try:
                node.send(0.0)
            except Exception:
                pass
            node.spin_for(0.02)
        if entered:
            try:
                print(node.set_mode(False))
            except RuntimeError as exc:
                print('WARNING: %s' % exc)
        node.spin_for(0.5)
        final = node.latest[1] if node.latest else None
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, 'jog_speed_%s_%s.json' % (a.joint, time.strftime('%H%M%S')))
        with open(path, 'w') as f:
            json.dump(dict(args=vars(a), scale=scale, max_percent=maxpct, results=results,
                           final_deg=final, samples=node.samples), f)
        print('final %s = %s deg; saved %s' % (a.joint, '%.3f' % final if final is not None else '?', path))
        node.destroy_node()
        rclpy.shutdown()


def sweep_main(a):
    if os.environ.get('ROS_DOMAIN_ID') in (None, '', '11') or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        sys.exit('REFUSED: set ROS_DOMAIN_ID (not 11) and ROS_LOCALHOST_ONLY=1 explicitly')
    if a.real == a.fake or (a.real and not a.approved):
        sys.exit('REFUSED: exactly one of --real / --fake, and --real needs --approved')
    if not 0 < a.sweep <= SWEEP_MAX_PERCENT or a.travel_deg is None or not 0 < a.travel_deg <= 2 * SWEEP_MAX_LIMIT_DEG \
            or a.limit_deg is None or not 0 < a.limit_deg <= SWEEP_MAX_LIMIT_DEG:
        sys.exit('REFUSED: --sweep in (0, 100], --travel-deg in (0, 180], --limit-deg in (0, 90] required')
    ip, scale_s, maxpct_s = robot_description_info()
    if ip != ('192.168.3.11' if a.real else '127.0.0.1'):
        sys.exit('REFUSED: robot_ip %s' % ip)
    scale = float(scale_s) if scale_s else 2.0
    maxpct = float(maxpct_s) if maxpct_s else 10.0
    if a.sweep > maxpct:
        sys.exit('REFUSED: driver joint_jog_max_percent %g < %g' % (maxpct, a.sweep))
    v = a.sweep / scale
    dur = a.travel_deg / v
    rclpy.init()
    node = JogCheck(a.driver, a.joint)
    entered = False
    result = dict(percent=a.sweep, cmd_deg_s=v, planned_duration_s=dur)
    try:
        node.spin_for(1.0)
        if node.latest is None:
            raise RuntimeError('no robot_joint_states')
        start = node.latest[1]
        end_planned = start + a.sign * a.travel_deg
        if abs(start) > a.limit_deg or (abs(end_planned) > a.limit_deg and not a.guard_ends_run):
            raise RuntimeError('start %.1f / planned end %.1f outside +-%g' % (start, end_planned, a.limit_deg))
        print('sweep %g %% = %.2f deg/s, %s%.1f deg from %.2f to ~%.1f (%.2f s), guard +-%g deg'
              % (a.sweep, v, '+' if a.sign > 0 else '-', a.travel_deg, start, end_planned, dur, a.limit_deg))
        print(node.set_mode(True))
        entered = True
        node.spin_for(0.3)
        vel = a.sign * math.radians(v)
        t0 = node.get_clock().now().nanoseconds * 1e-9
        period = 1.0 / PUBLISH_HZ
        next_t = time.monotonic()
        end = next_t + dur
        guard = False
        while time.monotonic() < end:
            if node.latest and guard_trips(node.latest[1], recent_velocity(node.samples), a.limit_deg):
                guard = True
                break
            node.send(vel)
            next_t += period
            while time.monotonic() < next_t:
                rclpy.spin_once(node, timeout_sec=0.002)
        t1 = node.get_clock().now().nanoseconds * 1e-9
        stop_end = time.monotonic() + 3.5
        while time.monotonic() < stop_end:
            node.send(0.0)
            node.spin_for(period)
        w = [(t, p) for t, p in node.samples if t0 <= t <= t1 + 3.0]
        vs = []
        for i in range(len(w)):
            j = next((k for k in range(i, len(w)) if w[k][0] - w[i][0] >= 0.2), None)
            if j is not None:
                vs.append((w[i][0] - t0, abs(w[j][1] - w[i][1]) / (w[j][0] - w[i][0])))
        vmax = max((x for _, x in vs), default=0.0)
        # plateau: samples within 3 % of the max, 0.2 s windows
        plateau = [x for _, x in vs if x >= 0.97 * vmax]
        result.update(guard=guard, commanded_s=t1 - t0, peak_deg_s=vmax,
                      plateau_deg_s=(sum(plateau) / len(plateau)) if plateau else None,
                      plateau_s=len(plateau) * 0.028, start_deg=start)
        print('commanded %.2f s%s; peak (0.2 s windows) %.2f deg/s; plateau mean %.2f deg/s over ~%.2f s; '
              'commanded %.2f deg/s -> ratio %.3f'
              % (t1 - t0, ' (GUARD stopped early)' if guard else '', vmax, result['plateau_deg_s'] or 0,
                 result['plateau_s'], v, (result['plateau_deg_s'] or 0) / v))
    except KeyboardInterrupt:
        print('Ctrl+C: zero and leave')
    except RuntimeError as exc:
        print('ERROR: %s' % exc)
    finally:
        for _ in range(10):
            try:
                node.send(0.0)
            except Exception:
                pass
            node.spin_for(0.02)
        if entered:
            try:
                print(node.set_mode(False))
            except RuntimeError as exc:
                print('WARNING: %s' % exc)
        node.spin_for(0.5)
        final = node.latest[1] if node.latest else None
        result['final_deg'] = final
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, 'sweep_%s_%g_%s.json' % (a.joint, a.sweep, time.strftime('%H%M%S')))
        with open(path, 'w') as f:
            json.dump(dict(args=vars(a), scale=scale, result=result, samples=node.samples), f)
        print('final %s = %s deg; saved %s' % (a.joint, '%.2f' % final if final is not None else '?', path))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
