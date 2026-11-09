#!/usr/bin/env bash
# run_mock_test.sh -- MOCK-ONLY end-to-end test of groot_ros_rollout.py against stub_policy_server.py.
#
# Starts (as systemd --user units, so only these can be stopped): the mock driver
#   --suite left  (default)  single arm:  ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
#   --suite right            dual table:  ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=mock_hardware
# and fake cameras publishing the two compressed topics at 1920x1080 / 1280x720 (left: /arm_left/gripper_camera, right:
# /arm_right/gripper_camera), on an isolated ROS domain (ROS_LOCALHOST_ONLY=1).  Then runs the scenarios of the suite (python driver
# below; names tvr_* are the right-arm TurboVLA suite, all others the left/single-arm suite), prints PASS/FAIL with evidence, and stops
# exactly the units it started.  It never touches any other domain, and refuses to run if the domain is not empty
# or the hardware is not mock_components/GenericSystem.
#
#   ./run_mock_test.sh [--suite left|right] [--model right19|mocap31|real3nov] [--domain 197] [--port 5597] [--only name1,name2] [--out DIR] [--real-port 5556] [--udp-only]
#
# --model M (right suite only, default right19): which RIGHT-arm TurboVLA checkpoint the tvr_* scenarios test: right19 = rollout_config_turbovla_right.yaml
#   (right19-10k, recorder 1152x648), mocap31 = rollout_config_turbovla_mocap31.yaml (mocap31-10k step 6000, recorder 640x480),
#   real3nov = rollout_config_turbovla_real3nov.yaml (real3nov-10k step 10000, recorder 1152x648, language "open the next closed flap ...").  Selects the config, the pinned
#   sha256, the language, the normalization.json for the training_range check, the stub identity and what the real server must announce.
#
# --udp-only: run every process of the test (mock stack, fake cameras, driver, rollout) with a Fast DDS profile without the shared-memory
#   transport (FASTRTPS_DEFAULT_PROFILES_FILE).  On this machine the SHM transport intermittently delivers NO camera frames to a freshly started
#   subscriber (a bare 20-line rclpy subscriber got nothing in 10 s in 14 of 25 launches; UDP only: 0 of 15), which shows up as
#   "no inputs within 10 s ... head False" (exit 4) in random scenarios.  Default off (the real stack uses the default transports).
#
# --real-port N (right suite): also run the tvr_real_* scenarios against a REAL TurboVLA server of the selected --model already listening on N
#   (start it with tools/turbovla_inference/start_turbovla_server.sh start --profile right19|mocap31|real3nov; this script never starts or stops it).
#   The images are fake, so the model outputs are meaningless: these scenarios test plumbing, latency and that the gates hold.
#
# Needs ~/venvs/gr00t_client (see README.md) and the built workspace (~/igus_rebel_ros2/install).
# (no set -u: the ROS setup scripts use unset variables)
DOMAIN=${GROOT_MOCK_DOMAIN:-197}
PORT=${GROOT_MOCK_PORT:-5597}
ONLY=""
OUT=""
SUITE=left
REAL_PORT=""
UDP_ONLY=0
MODEL=right19
while [ $# -gt 0 ]; do
  case "$1" in
    --udp-only) UDP_ONLY=1; shift;;
    --suite) SUITE=$2; shift 2;;
    --real-port) REAL_PORT=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --domain) DOMAIN=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --only) ONLY=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    -h|--help) sed -n 2,30p "$0"; exit 0;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
case "$SUITE" in left|right) ;; *) echo "--suite must be left or right"; exit 2;; esac
case "$MODEL" in right19|mocap31|real3nov) ;; *) echo "--model must be right19, mocap31 or real3nov"; exit 2;; esac
if [ "$SUITE" = left ] && [ "$MODEL" != right19 ]; then echo "--model applies to --suite right only"; exit 2; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYV="$HOME/venvs/gr00t_client/bin/python"
WS="$HOME/igus_rebel_ros2"
[ -x "$PYV" ] || { echo "missing $PYV: create the venv first (README.md)"; exit 2; }
if [ "$DOMAIN" -gt 232 ] 2>/dev/null; then echo "domain $DOMAIN > 232 does not work here"; exit 2; fi
if [ "$DOMAIN" = "11" ]; then echo "domain 11 is the real robot: refusing"; exit 2; fi
export ROS_DOMAIN_ID=$DOMAIN ROS_LOCALHOST_ONLY=1
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-/tmp/groot_mock_test_$STAMP}
WORK=$(mktemp -d /tmp/groot_mock_work_XXXXXX)
mkdir -p "$OUT"
DDS_ENV=""
if [ "$UDP_ONLY" = 1 ]; then
  cat > "$WORK/udp_only.xml" <<'XML'
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor><transport_id>udp</transport_id><type>UDPv4</type></transport_descriptor>
  </transport_descriptors>
  <participant profile_name="udp_only" is_default_profile="true">
    <rtps><userTransports><transport_id>udp</transport_id></userTransports><useBuiltinTransports>false</useBuiltinTransports></rtps>
  </participant>
</profiles>
XML
  export FASTRTPS_DEFAULT_PROFILES_FILE="$WORK/udp_only.xml"
  DDS_ENV="export FASTRTPS_DEFAULT_PROFILES_FILE=$WORK/udp_only.xml"
  echo "[mock] --udp-only: Fast DDS shared memory transport disabled for this test"
fi
U_STACK="groot-mock-stack-$DOMAIN"
U_CAMS="groot-mock-cams-$DOMAIN"
STARTED_UNITS=()
DAEMON_WAS_UP=0
ros2 daemon status >/dev/null 2>&1 && DAEMON_WAS_UP=1

cleanup() {
  for u in "${STARTED_UNITS[@]}"; do
    [ -n "$u" ] && systemctl --user stop "$u" >/dev/null 2>&1
  done
  # the driver stops the processes it spawned itself (stub, rollout) by PID before exiting
  if [ "$DAEMON_WAS_UP" = 0 ]; then ros2 daemon stop >/dev/null 2>&1; fi
  rm -rf "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ------------------------------------------------------------------------------------------------ fake cameras (mock only)
cat > "$WORK/fake_cams.py" <<'PYEOF'
#!/usr/bin/env python3
"""MOCK ONLY: publishes the two training-camera topics as CompressedImage (JPEG) at their real resolutions."""
import sys, numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

side = sys.argv[2] if len(sys.argv) > 2 else "left"   # which arm's gripper camera topic
CAMS = [("/camera/color/image_raw/compressed", 1920, 1080, "camera_color_optical_frame"),
        (f"/arm_{side}/gripper_camera/image_raw/compressed", 1280, 720, "gripper_camera")]
hz = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

class Fake(Node):
    def __init__(self):
        super().__init__("fake_cams_mock")
        self.pubs, self.frames, self.n = [], [], 0
        for i, (topic, w, h, fid) in enumerate(CAMS):
            self.pubs.append((self.create_publisher(CompressedImage, topic, qos_profile_sensor_data), fid))
            fr = []
            yy, xx = np.mgrid[0:h, 0:w]
            base = np.stack([(xx * 255 // w), (yy * 255 // h), np.full((h, w), 60 + 90 * i)], -1).astype(np.uint8)
            for k in range(30):
                img = base.copy()
                x0 = int((w - 200) * k / 29)
                cv2.rectangle(img, (x0, h // 3), (x0 + 200, h // 3 + 150), (255, 255, 255), -1)
                cv2.putText(img, "MOCK %s %d" % ("HEAD" if i == 0 else "GRIP", k), (40, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                fr.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
            self.frames.append(fr)
        self.create_timer(1.0 / hz, self.tick)
        self.get_logger().info("publishing %s at %.1f Hz" % ([c[0] for c in CAMS], hz))

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        for (pub, fid), fr in zip(self.pubs, self.frames):
            m = CompressedImage(); m.header.stamp = stamp; m.header.frame_id = fid
            m.format = "bgr8; jpeg compressed bgr8"; m.data = fr[self.n % 30]
            pub.publish(m)
        self.n += 1

rclpy.init(); rclpy.spin(Fake())
PYEOF

if [ "$SUITE" = right ]; then
  # mock_move_joint:=false: the robot-side MoveJoint mock is not needed here (and needs a newer igus_rebel install than some checkouts have)
  LAUNCH="rebel_dual_table_description dual_table.launch.py hardware_protocol:=mock_hardware mock_move_joint:=false"
else
  LAUNCH="igus_rebel rebel.launch.py hardware_protocol:=mock_hardware"
fi
cat > "$WORK/stack.sh" <<EOF
#!/usr/bin/env bash
export ROS_DOMAIN_ID=$DOMAIN ROS_LOCALHOST_ONLY=1
$DDS_ENV
source /opt/ros/humble/setup.bash; source $WS/install/setup.bash
exec nice -n 19 ros2 launch $LAUNCH
EOF
cat > "$WORK/cams.sh" <<EOF
#!/usr/bin/env bash
export ROS_DOMAIN_ID=$DOMAIN ROS_LOCALHOST_ONLY=1
$DDS_ENV
source /opt/ros/humble/setup.bash
exec nice -n 19 python3 $WORK/fake_cams.py 15 $SUITE
EOF
chmod +x "$WORK/stack.sh" "$WORK/cams.sh"

# ------------------------------------------------------------------------------------------------ driver (scenarios)
cat > "$WORK/driver.py" <<'PYEOF'
#!/usr/bin/env python3
import ast, collections, functools, io, json, os, signal, subprocess, sys, tempfile, threading, time, traceback, typing
from pathlib import Path
import numpy as np

TOOL, OUT, PORT = Path(os.environ["G_TOOLDIR"]), Path(os.environ["G_OUT"]), int(os.environ["G_PORT"])
PYV, ONLY, UNIT_CAMS = os.environ["G_PYV"], [s for s in os.environ.get("G_ONLY", "").split(",") if s], os.environ["G_UNIT_CAMS"]
assert os.environ["ROS_LOCALHOST_ONLY"] == "1" and os.environ["ROS_DOMAIN_ID"] not in ("", "11")
sys.path.insert(0, str(TOOL))
import groot_ros_rollout as G
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatusArray
from control_msgs.action import FollowJointTrajectory as FJT
from rclpy.action import ActionClient
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint

SUITE, REAL_PORT = os.environ.get("G_SUITE", "left"), os.environ.get("G_REAL_PORT", "")
MODEL = os.environ.get("G_MODEL", "right19")   # right suite: which right-arm checkpoint (right19 | mocap31 | real3nov)
START = [0.3, 0.7, 1.2, 0.0, 1.5, 0.0]
TARGET = [0.45, 0.85, 1.35, 0.15, 1.4, 0.15]
JN = [f"joint{i}" for i in range(1, 7)]
ACTION = "/rebel_arm_trajectory_controller/follow_joint_trajectory"
if SUITE == "right":   # dual-table mock (rebel_dual_table_description): right arm, inside the training range of both right checkpoints
    START = [0.2, 1.2, 0.6, 0.0, -0.6, 0.0]
    TARGET = [0.35, 1.35, 0.75, 0.15, -0.5, 0.15]
    JN = [f"arm_right_joint{i}" for i in range(1, 7)]
    ACTION = "/arm_right_trajectory_controller/follow_joint_trajectory"
RESULTS = []


def check(scn, name, ok, evidence):
    RESULTS.append({"scenario": scn, "check": name, "ok": bool(ok), "evidence": evidence})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {evidence}", flush=True)
    return ok


class Bench(Node):
    def __init__(self):
        super().__init__("groot_mock_bench")
        self.lock = threading.Lock()
        self.trace = []
        self.goal_ids, self.last_status = set(), []
        self.create_subscription(JointState, "/joint_states", self.on_js, qos_profile_sensor_data)
        self.create_subscription(GoalStatusArray, ACTION + "/_action/status", self.on_status,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        from sensor_msgs.msg import CompressedImage
        self.cam_t = 0.0
        self.create_subscription(CompressedImage, "/camera/color/image_raw/compressed", lambda m: setattr(self, "cam_t", time.monotonic()), qos_profile_sensor_data)
        self.ac = ActionClient(self, FJT, ACTION)
        self.ex = MultiThreadedExecutor(3)
        self.ex.add_node(self)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    def on_js(self, m):
        d = dict(zip(m.name, m.position))
        with self.lock:
            self.trace.append((time.monotonic(), np.array([d[j] for j in JN]), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9))

    def on_status(self, m):
        with self.lock:
            st = [(bytes(s.goal_info.goal_id.uuid).hex(), s.status) for s in m.status_list]
            self.last_status = st
            self.goal_ids.update(g for g, _ in st)

    def q(self):
        with self.lock:
            return self.trace[-1][1].copy()

    def window(self, t0, t1):
        with self.lock:
            tr = [x for x in self.trace if t0 <= x[0] <= t1]
        return np.array([x[0] for x in tr]), (np.array([x[1] for x in tr]) if tr else np.zeros((0, 6)))

    def peak(self, t0, t1, win=0.2):
        """peak per-joint speed over `win`-second windows using the controller's own message stamps (immune to receive jitter)"""
        with self.lock:
            tr = [x for x in self.trace if t0 <= x[0] <= t1]
        return G.peak_speed([x[2] for x in tr], [x[1] for x in tr], win) if len(tr) > 2 else 0.0

    def ptp(self, t0, t1):
        ts, Q = self.window(t0, t1)
        return float(np.ptp(Q, axis=0).max()) if len(ts) > 1 else float("nan")

    def goals_seen(self):
        with self.lock:
            return set(self.goal_ids)

    def active_goals(self):
        with self.lock:
            return [g for g, s in self.last_status if s in (1, 2, 3)]  # accepted / executing / canceling

    def goto(self, q, T=6.0):
        self.ac.wait_for_server(5)
        g = FJT.Goal(); g.trajectory.joint_names = JN
        p = JointTrajectoryPoint(); p.positions = [float(v) for v in q]; p.time_from_start = Duration(seconds=T).to_msg()
        g.trajectory.points = [p]
        f = self.ac.send_goal_async(g)
        while not f.done(): time.sleep(0.01)
        r = f.result().get_result_async()
        while not r.done(): time.sleep(0.02)
        time.sleep(0.3)
        return r.result().result.error_code


class Stub:
    def __init__(self, mode, extra=(), backend="groot"):
        self.log = OUT / f"stub_{backend}_{mode}_{int(time.time() * 1000) % 100000}.log"
        self.proc = subprocess.Popen([PYV, str(TOOL / "stub_policy_server.py"), "--backend", backend, "--mode", mode, "--port", str(PORT),
                                      "--log", str(self.log) + ".jsonl", *extra],
                                     stdout=open(self.log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        c = G.Gr00tClient("127.0.0.1", PORT, 3.0)
        for _ in range(40):
            try:
                c.call("ping", timeout_s=0.5)
                break
            except G.ServerError:
                time.sleep(0.2)
        else:
            raise RuntimeError("stub did not start")
        c.close()

    def records(self):
        p = Path(str(self.log) + ".jsonl")
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Roll:
    """One rollout subprocess in its own session (we signal exactly this PID)."""

    def __init__(self, name, argv, stdin=False):
        self.name, self.dir = name, OUT / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.dir / "console.log", "w")
        self.t0 = time.monotonic()
        self.proc = subprocess.Popen([PYV, str(TOOL / "groot_ros_rollout.py"), "--run-dir", str(self.dir), "--port", str(PORT), *argv],
                                     stdin=subprocess.PIPE if stdin else subprocess.DEVNULL, stdout=self.logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def wait(self, timeout):
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            print(f"  !! {self.name}: timed out after {timeout}s, killed (pid {self.proc.pid})")
        self.t1 = time.monotonic()
        return self.proc.returncode

    def sig(self, s):
        os.kill(self.proc.pid, s)

    def recs(self, typ="query"):
        p = self.dir / "queries.jsonl"
        out = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        return [r for r in out if r.get("type") == typ]

    def end(self):
        e = self.recs("end")
        return e[-1] if e else {}

    def console(self):
        return (self.dir / "console.log").read_text()


def failed_gates(rec):
    return sorted(g["name"] for g in rec.get("gates", []) if not g["ok"])


MOCKYES = ["--execute", "--mock-yes"]
B = None
SCN = {}


def scenario(f):
    SCN[f.__name__] = f
    return f


# ======================================================================================================== unit-level
@scenario
def unit_gates():
    s = "unit_gates"
    cfg = G.load_config(TOOL / "rollout_config.yaml")
    g, tr = cfg["gates"], cfg["training_range"]
    lim = np.array([[-3.1067, 3.1067], [-1.3788, 2.4260], [-1.3788, 2.4260], [-3.1067, 3.1067], [-1.6406, 1.6406], [-3.1067, 3.1067]])
    q = np.array(START)
    good = np.array(TARGET) * 0 + q + np.linspace(0, 0.08, 17)[1:, None]
    ev = lambda ch, qq=q, **kw: G.evaluate_gates(ch, qq, lim, {**g, **kw}, tr["min"], tr["max"], 8)
    bad = lambda r: sorted(x["name"] for x in r if not x["ok"])
    check(s, "good chunk passes every gate", bad(ev(good)) == [], str(bad(ev(good))))
    c = good.copy(); c[3, 2] = np.nan
    check(s, "NaN -> finite", bad(ev(c)) == ["finite"], str(bad(ev(c))))
    c = good.copy(); c[:, 0] += 0.06
    check(s, "first step 0.06 off -> first_step", "first_step" in bad(ev(c)), str(bad(ev(c))))
    c = good.copy(); c[9:, 1] += 0.15
    check(s, "0.15 jump at step 9 -> step_jump (full-chunk scope)", "step_jump" in bad(ev(c)), str(bad(ev(c))))
    check(s, "same jump beyond the horizon passes with validate_full_chunk=false", "step_jump" not in bad(ev(c, validate_full_chunk=False)),
          str(bad(ev(c, validate_full_chunk=False))))
    # URDF alone: state near the URDF limit but the training range is widened so only the URDF gate can fire
    qn = np.array([0.3, 0.7, 1.2, 0.0, 1.60, 0.0]); c = qn + np.zeros((16, 6)); c[:, 4] += np.linspace(0.0, 0.03, 16)
    r = ev(c, qn, training_margin_rad=1.0, first_step_max_rad=0.5)
    check(s, "joint5 into URDF inset (1.64-0.05) -> urdf_limits only", bad(r) == ["urdf_limits"], str(bad(r)))
    # training range alone
    qn = np.array([0.7, 0.7, 1.2, 0.0, 0.3, 0.0]); c = qn + np.zeros((16, 6)); c[:, 0] += np.linspace(0.0, 0.1, 16)
    r = ev(c, qn, state_in_training_range=False)
    check(s, "joint1 0.8 > 0.6765+0.1 -> training_range only", bad(r) == ["training_range"], str(bad(r)))
    c = q + np.linspace(0, 0.64, 17)[1:, None]
    check(s, "displacement 0.32 at step 8 > 0.3 over the horizon -> chunk_displacement", "chunk_displacement" in bad(ev(c, max_step_rad=1.0, first_step_max_rad=1.0)),
          str(bad(ev(c, max_step_rad=1.0, first_step_max_rad=1.0))))
    qz = np.zeros(6)
    check(s, "state at all-zeros (joint3 < 0.26-0.1) -> state_in_training_range", "state_in_training_range" in bad(ev(good - q, qz)), str(bad(ev(good - q, qz))))
    for shape, ok in [((16, 6), True), ((1, 16, 6), True), ((1, 1, 16, 6), True), ((15, 6), False), ((16, 5), False), ((2, 16, 6), False), ((96,), False)]:
        a, prob = G.normalize_chunk(np.zeros(shape, np.float32), 16, 6)
        check(s, f"normalize_chunk{shape}", (a is not None) == ok, str(prob))
    a, prob = G.normalize_chunk(np.array([["a"] * 6] * 16), 16, 6)
    check(s, "normalize_chunk string dtype refused", a is None, str(prob))
    # timing: never compressed, never above vmax
    qs = q + np.cumsum(np.full((8, 6), 0.0053), 0)
    t = G.trajectory_times(q, qs, 1 / 15, 0.1)
    check(s, "slow chunk keeps nominal 1/15 s timing", np.allclose(np.diff(np.r_[0, t]), 1 / 15), f"duration {t[-1]:.4f} s")
    qs = q + np.cumsum(np.full((8, 6), 0.03), 0)
    t = G.trajectory_times(q, qs, 1 / 15, 0.1)
    seg = np.abs(np.diff(np.vstack([q, qs]), axis=0)).max(1) / np.diff(np.r_[0, t])
    check(s, "fast chunk (0.03 rad/step = 0.45 rad/s) stretched to 0.1 rad/s", np.isclose(seg.max(), 0.1) and t[-1] > 8 / 15, f"max seg speed {seg.max():.4f}, duration {t[-1]:.3f} s vs nominal {8 / 15:.3f}")
    # hardware classification
    M, R = {"class_type": "mock_components/GenericSystem"}, {"class_type": "Igus::Rebel"}
    for comps, exp in [(None, False), ([], False), ([M], True), ([M, M], True), ([M, R], False), ([R], False)]:
        check(s, f"hardware_is_mock({[c['class_type'] for c in comps] if comps is not None else None})", G.hardware_is_mock(comps) == exp, str(G.hardware_is_mock(comps)))
    # freshness
    snap = {"q": np.zeros(6), "q_age": 0.25, "ros_now": 0.0, "img": {"head": {"age": 0.1, "stamp": 0.0}, "wrist_left": {"age": 0.7, "stamp": 0.0}}}
    r = G.Rollout(cfg, G.parse_args([])).check_freshness(snap)
    check(s, "freshness reports joint (0.25>0.2) and wrist_left (0.7>0.5) but not head", len(r) == 2 and "joint_states" in r[0] and "wrist_left" in r[1], str(r))
    # config errors
    rc = G.main(["--set", "gates.nope=1"])
    check(s, "unknown --set key -> exit 2", rc == 2, f"rc={rc}")
    rc = G.main(["--horizon", "17"])
    check(s, "horizon 17 > chunk_len -> exit 2", rc == 2, f"rc={rc}")


@scenario
def unit_preprocess():
    s = "unit_preprocess"
    import cv2
    from PIL import Image as PILImage
    cfg = G.load_config(TOOL / "rollout_config.yaml")
    check(s, "default config: preprocess.mode recorder_pil, video_size [640, 480]", cfg["preprocess"]["mode"] == "recorder_pil" and cfg["model"]["video_size"] == [640, 480],
          f"{cfg['preprocess']['mode']} {cfg['model']['video_size']}")
    rng = np.random.default_rng(1)

    def make(w, h):
        yy, xx = np.mgrid[0:h, 0:w]
        tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32), (0, 0), 4) * 60  # natural-ish low-frequency texture
        img = np.stack([xx * 200 // w, yy * 200 // h, 100 + 0 * xx], -1).astype(np.float32) + 20 + tex
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.circle(img, (w * 9 // 20, h * 10 // 21), h // 5, (20, 200, 240), -1)  # BGR: orange disc
        cv2.putText(img, "FLAP 123", (w // 6, h * 3 // 4), cv2.FONT_HERSHEY_SIMPLEX, h / 180, (255, 255, 255), max(2, h // 90))
        return img

    def recorder_reference(jpg_bytes, width=640, height=480):
        """Independent re-implementation of ~/robot_remote/recording/record_dataset.py image_msg_to_rgb(min_size=(w, h)) + resize_rgb
        (written out here from that code, not imported from the client)."""
        img = PILImage.open(io.BytesIO(bytes(jpg_bytes)))
        img.draft("RGB", (width, height))
        rgb = np.asarray(img.convert("RGB"))
        if rgb.shape[1] == width and rgb.shape[0] == height:
            return rgb
        return np.asarray(PILImage.fromarray(rgb).resize((width, height), PILImage.BILINEAR))

    for (w, h) in ((1920, 1080), (1280, 720)):  # the two live camera sizes
        img = make(w, h)
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()  # what the camera driver publishes
        mine, full = G.preprocess_image(np.frombuffer(jpg, np.uint8), cfg["preprocess"], cfg["model"]["video_size"])
        ref = recorder_reference(jpg)
        check(s, f"{w}x{h}: output (480,640,3) uint8 RGB, reported source size {w}x{h}", mine.shape == (480, 640, 3) and mine.dtype == np.uint8 and list(full) == [w, h],
              f"{mine.shape} {mine.dtype} full={full}")
        check(s, f"{w}x{h}: identical to the independent recorder re-implementation (PIL draft + convert + BILINEAR)", np.array_equal(mine, ref),
              f"max abs diff {np.abs(mine.astype(int) - ref.astype(int)).max()}")
        # draft really is part of the recipe: without it PIL's BILINEAR (antialiased) result differs slightly (only for the 1080p source)
        nodraft = np.asarray(PILImage.fromarray(np.asarray(PILImage.open(io.BytesIO(jpg)).convert("RGB"))).resize((640, 480), PILImage.BILINEAR))
        d = np.abs(mine.astype(int) - nodraft.astype(int)).mean()
        check(s, f"{w}x{h}: draft step {'changes the result slightly' if w == 1920 else 'is a no-op (needs 1/2 scale >= 480 rows)'} (mean abs diff vs no draft {d:.3f})",
              (0 < d < 3) if w == 1920 else d == 0, f"{d:.3f}")
        if w == 1920:
            # aspect distortion kept as in training: the orange disc (circle in the source) becomes an ellipse x/y = (640/1920)/(480/1080) = 0.75
            m = (mine[:, :, 0] > 200) & (mine[:, :, 1] > 150) & (mine[:, :, 2] < 80)
            ys, xs = np.where(m)
            ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) if len(xs) else float("nan")
            check(s, "aspect not preserved (disc bbox width/height 0.75 in the model image)", abs(ratio - 0.75) < 0.05, f"{ratio:.3f} ({len(xs)} px)")
            # legacy OpenCV mode still works and is selectable (different recipe: close but not identical)
            leg_pp = {**cfg["preprocess"], "mode": "opencv_legacy"}
            leg, lfull = G.preprocess_image(np.frombuffer(jpg, np.uint8), leg_pp, cfg["model"]["video_size"])
            dl = np.abs(leg.astype(int) - mine.astype(int)).mean()
            check(s, "preprocess.mode opencv_legacy selectable: (480,640,3), close to the recorder path but not identical", leg.shape == (480, 640, 3) and list(lfull) == [w, h] and 0 < dl < 8, f"mean abs diff {dl:.3f}")
            noRT, _ = G.preprocess_image(np.frombuffer(jpg, np.uint8), {**leg_pp, "jpeg_roundtrip": False}, cfg["model"]["video_size"])
            check(s, "legacy jpeg_roundtrip=false option still works", 0 < np.abs(noRT.astype(int) - leg.astype(int)).mean() < 3, f"{np.abs(noRT.astype(int) - leg.astype(int)).mean():.3f}")
    try:
        G.preprocess_image(np.frombuffer(jpg, np.uint8), {**cfg["preprocess"], "mode": "nope"}, [640, 480])
        ok = False
    except ValueError as e:
        ok = "nope" in str(e)
    check(s, "unknown preprocess.mode -> ValueError", ok, "")
    try:
        G.preprocess_image(np.frombuffer(b"not an image", np.uint8), cfg["preprocess"], [640, 480])
        ok = False
    except ValueError:
        ok = True
    check(s, "undecodable bytes -> ValueError (client reports input_error, no crash)", ok, "")


@scenario
def unit_wire():
    s = "unit_wire"
    cfg = G.load_config(TOOL / "rollout_config.yaml")
    src = Path(os.path.expanduser("~/groot/Isaac-GR00T/gr00t/policy/server_client.py"))
    if not src.exists():
        check(s, "Isaac-GR00T server_client.py present", False, f"{src} missing: wire test against the real MsgSerializer skipped")
        return
    import msgpack, msgpack_numpy as mnp, zmq
    text = src.read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "MsgSerializer")
    ns = dict(functools=functools, io=io, json=json, msgpack=msgpack, mnp=mnp, np=np, Any=typing.Any,
              ModalityConfig=type("ModalityConfig", (), {}), to_json_serializable=lambda x: x)
    exec(ast.get_source_segment(text, cls), ns)
    Ser = ns["MsgSerializer"]
    got = {}

    def server():
        ctx = zmq.Context(); sock = ctx.socket(zmq.REP); sock.bind(f"tcp://127.0.0.1:{PORT + 1}")
        sock.RCVTIMEO = 8000
        for _ in range(3):
            try:
                req = Ser.from_bytes(sock.recv())
            except zmq.Again:
                break
            got[req["endpoint"]] = req
            if req["endpoint"] == "get_action":
                obs = req["data"]["observation"]
                a = np.tile(obs["state"]["single_arm"][0, 0], (16, 1))[None].astype(np.float32)
                sock.send(Ser.to_bytes((({"single_arm": a}), {"info": 1})))
            else:
                sock.send(Ser.to_bytes({"status": "ok"}))
        sock.close(linger=0); ctx.term()

    th = threading.Thread(target=server, daemon=True); th.start()
    c = G.Gr00tClient("127.0.0.1", PORT + 1, 5.0)
    c.call("ping", timeout_s=4); c.call("reset", {"options": None}, timeout_s=4)
    v = {k: np.zeros((480, 640, 3), np.uint8) for k in ("head", "wrist_left")}
    obs = G.build_observation(v, np.array(START), cfg["instruction"], cfg["model"])
    act = c.get_action(obs)
    th.join(5); c.close()
    o = got["get_action"]["data"]["observation"]
    check(s, "real MsgSerializer decodes client request (video uint8 (1,1,480,640,3), state float32 (1,1,6), language [[str]])",
          o["video"]["head"].shape == (1, 1, 480, 640, 3) and o["video"]["head"].dtype == np.uint8 and o["state"]["single_arm"].dtype == np.float32
          and o["language"]["annotation.human.task_description"] == [[cfg["instruction"]]] and got["get_action"]["data"]["options"] is None,
          f"{o['video']['head'].shape} {o['video']['head'].dtype} {o['state']['single_arm'].dtype}")
    check(s, "client decodes real MsgSerializer reply (action dict with (1,16,6) float32)",
          act["single_arm"].shape == (1, 16, 6) and act["single_arm"].dtype == np.float32 and np.allclose(act["single_arm"][0, 0], START, atol=1e-6),
          f"{act['single_arm'].shape} {act['single_arm'].dtype}")


@scenario
def unit_confirm_nonmock():
    """confirm_execute() with a FAKE non-mock hardware list: --mock-yes must not bypass the typed `yes`."""
    s = "unit_confirm_nonmock"
    code = r'''
import os, sys
sys.path.insert(0, %r)
import groot_ros_rollout as G
cfg = G.load_config(%r)
a = G.parse_args(["--execute", "--mock-yes"])
r = G.Rollout(cfg, a)
class FakeIO:
    def list_hardware(self, timeout=5.0): return [{"name": "rebel_arm_controller", "type": "system", "class_type": "Igus::Rebel", "state": "active"}]
    def robot_description(self, t=5.0, s=None): return None
r.io = FakeIO()
print("RESULT", r.confirm_execute())
''' % (str(TOOL), str(TOOL / "rollout_config.yaml"))
    for text, exp in [("no\n", False), ("YES\n", False), ("\n", False), ("yes\n", True), ("", False)]:
        p = subprocess.run([PYV, "-c", code], input=text.encode(), capture_output=True, timeout=30)
        out = p.stdout.decode()
        res = "RESULT True" in out
        check(s, f"real hardware (fake Igus::Rebel) + --mock-yes + stdin {text!r} -> {exp}", res == exp and "--mock-yes ignored" in out,
              ("prompt shown, " if "type  yes" in out else "NO PROMPT, ") + f"result={res}")
    # unknown hardware (service missing) fails closed
    code2 = code.replace('return [{"name": "rebel_arm_controller", "type": "system", "class_type": "Igus::Rebel", "state": "active"}]', "return None")
    p = subprocess.run([PYV, "-c", code2], input=b"yes\n", capture_output=True, timeout=30)
    check(s, "hardware type unreadable -> execute refused even with `yes`", b"RESULT False" in p.stdout, p.stdout.decode().strip().splitlines()[-2:].__str__())


# ====================================================================================================== integration
@scenario
def stub_startup_race():
    s = "stub_startup_race"
    B.goto(START, 6)
    res = []
    for i in range(12):
        stub = Stub("good")
        r = Roll(f"{s}_{i}", ["--max-queries", "1"])
        res.append(r.wait(40))
        stub.stop()
    check(s, "12 x (start stub, immediately run observe) all exit 0", res == [0] * 12, f"exit codes {res}")


@scenario
def observe_good():
    s = "observe_good"
    B.goto(START, 6)
    stub = Stub("good")
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s, ["--max-queries", "4", "--set", "observe.period_s=0.3", "--save-images"])
        rc = r.wait(60)
        t1 = time.monotonic()
        q = r.recs()
        check(s, "exit 0, 4 queries, all PASS", rc == 0 and len(q) == 4 and all(x["verdict"] == "pass" for x in q), f"rc={rc} verdicts={[x['verdict'] for x in q]}")
        check(s, "arm not moved (observe never commands)", B.ptp(t0, t1) < 1e-4, f"joint ptp over the run {B.ptp(t0, t1):.2e} rad")
        check(s, "no trajectory goal was created", B.goals_seen() == g0, f"new goals: {len(B.goals_seen() - g0)}")
        check(s, "no hold goal either; none of the records says sent", not any(x["sent"] for x in q), "sent flags " + str([x["sent"] for x in q]))
        check(s, "stub got the exact observation layout (it validates shapes/dtypes; 4 get_action answered)", len(stub.records()) == 4 and stub.records()[0]["instruction"] == G.load_config(TOOL / "rollout_config.yaml")["instruction"], f"{len(stub.records())} requests")
        imgs = sorted(p.name for p in (r.dir / "images").glob("*.jpg"))
        check(s, "model input images saved (2 per query)", len(imgs) == 8, f"{len(imgs)} files")
        q0 = q[0]
        check(s, "record has time/state/chunk/gates/latency", all(k in q0 for k in ("t_iso", "state", "chunk", "gates", "latency_ms")) and len(q0["chunk"]) == 16,
              f"latency {q0['latency_ms']} ms, chunk {len(q0['chunk'])}x{len(q0['chunk'][0])}, gates {[g['name'] for g in q0['gates']]}")
    finally:
        stub.stop()


@scenario
def execute_at_zero_pose():
    s = "execute_at_zero_pose"
    B.goto([0, 0, 0, 0, 0, 0], 4)
    stub = Stub("good")
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s, MOCKYES + ["--max-steps", "16"])
        rc = r.wait(60)
        q = r.recs()
        check(s, "refused: exit 3, state_in_training_range failed", rc == 3 and q and "state_in_training_range" in failed_gates(q[0]), f"rc={rc} failed={failed_gates(q[0]) if q else None}")
        check(s, "arm stayed at zeros", B.ptp(t0, time.monotonic()) < 1e-4 and np.abs(B.q()).max() < 1e-4, f"|q|max={np.abs(B.q()).max():.1e}")
        check(s, "no chunk goal sent (record sent=false)", q and q[0]["sent"] is False, "")
    finally:
        stub.stop()
        B.goto(START, 6)


@scenario
def execute_good():
    s = "execute_good"
    B.goto(START, 6)
    stub = Stub("good", ["--speed", "0.08"])
    try:
        t0 = time.monotonic()
        r = Roll(s, MOCKYES + ["--max-steps", "40"])
        rc = r.wait(120)
        t1 = time.monotonic()
        q = r.recs()
        check(s, "exit 0 after 5 chunks (40 steps)", rc == 0 and len(q) == 5 and all(x["verdict"] == "pass" and x.get("exec", {}).get("error_code") == 0 for x in q),
              f"rc={rc} n={len(q)} {[x['verdict'] for x in q]}")
        pk = B.peak(t0, t1)
        check(s, "velocity cap: independent measurement from /joint_states <= 0.105 rad/s (0.2 s windows, message stamps)", pk <= 0.105, f"peak {pk:.4f} rad/s (commanded 0.08 rad/s)")
        qf = B.q()
        err = np.abs(qf - np.array(TARGET)).max()
        check(s, "arm moved along the chunks to the stub target", err < 0.005, f"final {np.round(qf, 4).tolist()} target {TARGET}, max err {err:.4f}")
        d = np.abs(qf - np.array(START)).max()
        check(s, "net displacement equals the sum of executed chunks", d > 0.1, f"moved {d:.3f} rad from the start pose")
        ex = q[0]["exec"]
        check(s, "timing nominal (no stretch needed at 0.08 rad/s)", abs(ex["stretch"] - 1.0) < 1e-6, f"duration {ex['duration_s']:.3f}s nominal {ex['nominal_s']:.3f}s tracking err {ex['max_tracking_err']:.4f}")
        check(s, "trajectory path actually followed: executed chunks consecutive (each starts where the last ended)",
              all(np.abs(np.array(q[i + 1]["state"]) - np.array(q[i]["exec"]["state_after"])).max() < 0.003 for i in range(len(q) - 1)), "")
    finally:
        stub.stop()


@scenario
def execute_fast_stretch():
    s = "execute_fast_stretch"
    B.goto(START, 6)
    stub = Stub("fast", ["--fast-speed", "0.3", "--target", "0.7,1.1,1.5,0.4,1.2,0.4"])  # far target: every step is the full 0.02 rad
    try:
        t0 = time.monotonic()
        r = Roll(s, MOCKYES + ["--max-steps", "16"])
        rc = r.wait(120)
        t1 = time.monotonic()
        q = r.recs()
        ex = q[0]["exec"] if q else {}
        check(s, "exit 0, chunks passed the gates (0.02 rad/step)", rc == 0 and len(q) == 2 and all(x["verdict"] == "pass" for x in q), f"rc={rc} {[x['verdict'] for x in q]}")
        check(s, "stub asked for 0.3 rad/s, client stretched timing ~3x", abs(ex.get("stretch", 0) - 3.0) < 0.1, f"stretch {ex.get('stretch', 0):.3f}, duration {ex.get('duration_s', 0):.2f}s vs nominal {ex.get('nominal_s', 0):.2f}s")
        pk = B.peak(t0, t1)
        check(s, "measured peak speed <= 0.105 rad/s (cap 0.1, 0.2 s windows, message stamps)", pk <= 0.105, f"peak {pk:.4f} rad/s")
        check(s, "wall time of a goal matches the stretched duration", abs(ex.get("wall_s", 0) - ex.get("duration_s", 0)) < 0.15, f"wall {ex.get('wall_s', 0):.3f}s planned {ex.get('duration_s', 0):.3f}s")
    finally:
        stub.stop()


def bad_mode(s, mode, expect_gates, extra_stub=(), start=START):
    B.goto(start, 6)
    stub = Stub(mode, extra_stub)
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(f"{s}_{mode}", MOCKYES + ["--max-steps", "40"])
        rc = r.wait(60)
        t1 = time.monotonic()
        q = r.recs()
        fg = failed_gates(q[0]) if q else None
        check(f"{s}", f"{mode}: exit 3 and refused", rc == 3 and len(q) == 1 and q[0]["verdict"] == "refuse" and q[0]["sent"] is False, f"rc={rc} verdicts={[x['verdict'] for x in q]}")
        check(f"{s}", f"{mode}: expected gates failed {expect_gates}", all(e in (fg or []) for e in expect_gates), f"failed gates {fg}")
        check(f"{s}", f"{mode}: arm held still", B.ptp(t0, t1) < 1e-4, f"joint ptp {B.ptp(t0, t1):.2e} rad over the run")
        e = r.end()
        check(f"{s}", f"{mode}: stopped with hold (cancel+hold sequence ran)", (e.get("stop") or {}).get("held") is True, f"stop info {e.get('stop')}")
        check(f"{s}", f"{mode}: only a hold goal exists, no chunk goal", len(B.goals_seen() - g0) == 1, f"new goals {len(B.goals_seen() - g0)} (the hold)")
    finally:
        stub.stop()


@scenario
def refuse_jump():
    bad_mode("refuse_jump", "jump", ["step_jump"])


@scenario
def refuse_nan():
    bad_mode("refuse_nan", "nan", ["finite"])


@scenario
def refuse_limit():
    bad_mode("refuse_limit", "limit", ["urdf_limits", "training_range"])


@scenario
def refuse_relative():
    bad_mode("refuse_relative", "relative", ["first_step"])


@scenario
def refuse_badshape():
    bad_mode("refuse_badshape", "badshape", ["shape"])


@scenario
def refuse_slow():
    s = "refuse_slow"
    B.goto(START, 6)
    stub = Stub("slow", ["--slow-s", "8"])
    try:
        t0 = time.monotonic()
        r = Roll(s, MOCKYES + ["--max-steps", "16", "--set", "server.timeout_s=2.0", "--set", "server.first_query_timeout_s=2.0"])
        rc = r.wait(60)
        t1 = time.monotonic()
        q = r.recs()
        check(s, "server timeout -> exit 4, verdict server_error", rc == 4 and q and q[0]["verdict"] == "server_error", f"rc={rc} {[x['verdict'] for x in q]} problems={q[0]['problems'] if q else None}")
        check(s, "gave up after 2 x 2 s (one re-send), not 8", t1 - r.t0 < 9 and q and q[0].get("server_retries") is None, f"run took {t1 - r.t0:.1f}s incl. ~2 s startup")
        check(s, "arm held still", B.ptp(t0, t1) < 1e-4, f"ptp {B.ptp(t0, t1):.2e}")
    finally:
        stub.stop()


@scenario
def retry_after_lost_reply():
    """first get_action is answered too late (stub sleeps 2 s, timeout 1.5 s): the client re-sends on a fresh socket and carries on"""
    s = "retry_after_lost_reply"
    B.goto(START, 6)
    stub = Stub("slow", ["--slow-s", "2", "--slow-first", "1"])
    try:
        t0 = time.monotonic()
        r = Roll(s, MOCKYES + ["--max-steps", "16", "--set", "server.timeout_s=1.5", "--set", "server.first_query_timeout_s=1.5"])
        rc = r.wait(60)
        q = r.recs()
        check(s, "exit 0, both chunks executed, first query needed exactly 1 re-send", rc == 0 and len(q) == 2 and all(x["verdict"] == "pass" for x in q) and q[0].get("server_retries") == 1 and q[1].get("server_retries") == 0,
              f"rc={rc} retries={[x.get('server_retries') for x in q]} verdicts={[x['verdict'] for x in q]}")
        check(s, "arm moved", B.ptp(t0, time.monotonic()) > 0.01, f"ptp {B.ptp(t0, time.monotonic()):.3f}")
    finally:
        stub.stop()


@scenario
def modality_mismatch():
    """the server announces other video keys (wrist_right instead of wrist_left): execute refuses before any query, observe warns"""
    s = "modality_mismatch"
    B.goto(START, 6)
    # parse/compare helpers
    cfg = G.load_config(TOOL / "rollout_config.yaml")
    info = G.parse_modality_reply({"video": {"__ModalityConfig__": True, "as_json": json.dumps({"delta_indices": [0], "modality_keys": ["head", "wrist_right"]})},
                                   "action": {"as_json": {"delta_indices": list(range(16)), "modality_keys": ["single_arm"]}}})
    check(s, "parse_modality_reply understands the Isaac-GR00T encoding (json str and dict)", info == {"video": {"keys": ["head", "wrist_right"], "n": 1}, "action": {"keys": ["single_arm"], "n": 16}}, str(info))
    check(s, "compare_modality flags wrist_right", len(G.compare_modality(info, cfg)) == 1 and "wrist" in G.compare_modality(info, cfg)[0], str(G.compare_modality(info, cfg)))
    check(s, "parse_modality_reply({}) -> None (real server may return {})", G.parse_modality_reply({}) is None, "")
    stub = Stub("good", ["--video-keys", "head,wrist_right"])
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s + "_execute", MOCKYES + ["--max-steps", "16"])
        rc = r.wait(60)
        check(s, "execute: exit 4 'server modality mismatch', no query made, arm still, no chunk goal", rc == 4 and "modality mismatch" in r.end().get("reason", "") and not r.recs() and B.ptp(t0, time.monotonic()) < 1e-4 and len(B.goals_seen() - g0) <= 1,
              f"rc={rc} {r.end().get('reason')}")
        r = Roll(s + "_observe", ["--max-queries", "1"])
        rc = r.wait(60)
        check(s, "observe: warns loudly but still runs (exit 0)", rc == 0 and "MODALITY MISMATCH" in r.console() and len(r.recs()) == 1, f"rc={rc}")
    finally:
        stub.stop()
    stub = Stub("good", ["--action-len", "8"])
    try:
        r = Roll(s + "_horizon", MOCKYES + ["--max-steps", "16"])
        rc = r.wait(60)
        check(s, "execute: server action horizon 8 != client 16 -> refused", rc == 4 and "action horizon" in r.end().get("reason", ""), f"rc={rc} {r.end().get('reason')}")
    finally:
        stub.stop()
    stub = Stub("good")
    try:
        r = Roll(s + "_ok", MOCKYES + ["--max-steps", "8"])
        rc = r.wait(60)
        check(s, "matching keys: log shows the server modality and the run works", rc == 0 and "server modality: video ['head', 'wrist_left']" in r.console(), f"rc={rc}")
    finally:
        stub.stop()


@scenario
def refuse_server_error():
    s = "refuse_server_error"
    B.goto(START, 6)
    stub = Stub("good", ["--expect-instruction", "something else"])
    try:
        t0 = time.monotonic()
        r = Roll(s, MOCKYES + ["--max-steps", "16"])
        rc = r.wait(60)
        q = r.recs()
        check(s, "server rejects observation -> {'error'} reply -> exit 4, arm still", rc == 4 and q and q[0]["verdict"] == "server_error" and B.ptp(t0, time.monotonic()) < 1e-4,
              f"rc={rc} problems={q[0]['problems'] if q else None}")
    finally:
        stub.stop()


def cams_unit(on):
    active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT_CAMS]).returncode == 0
    if on and not active:  # transient units disappear when stopped: start a fresh one
        subprocess.run(["systemd-run", "--user", f"--unit={UNIT_CAMS}", "--collect", os.environ["G_CAMS_SH"]], check=True, capture_output=True)
    elif not on and active:
        subprocess.run(["systemctl", "--user", "stop", UNIT_CAMS], check=True)


def cams(on, timeout=60):
    cams_unit(on)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:  # wait until the topic really flows / has really stopped
        age = time.monotonic() - B.cam_t if B.cam_t else 1e9
        if (on and age < 0.3) or (not on and age > 0.8):
            return
        time.sleep(0.1)
    raise RuntimeError(f"cams({on}) did not settle")


def wait_for(pred, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


@scenario
def stale_camera_execute():
    s = "stale_camera_execute"
    B.goto(START, 6)
    stub = Stub("good", ["--speed", "0.08", "--target", "0.6,1.0,1.5,0.3,1.3,0.3"])
    try:
        r = Roll(s, MOCKYES + ["--max-steps", "400"])
        wait_for(lambda: len(r.recs()) >= 3, 30)
        n_before = len(r.recs())
        t_stop = time.monotonic()
        cams_unit(False)  # the age gate must catch it while the run is live
        rc = r.wait(60)
        q = r.recs()
        last = q[-1]
        check(s, "ran fine before the camera stopped", n_before >= 3 and all(x["verdict"] == "pass" for x in q[:-1]), f"{n_before} chunks passed before the stop")
        check(s, "after the camera stopped: verdict stale, exit 4, names the stale topic",
              rc == 4 and last["verdict"] == "stale" and any("image" in p and ("/camera/color" in p or "gripper" in p) for p in last["problems"]), f"rc={rc} {last['problems']}")
        check(s, "refused within ~1.5 s of the stop (age gate 0.5 s)", r.t1 - t_stop < 2.5, f"{r.t1 - t_stop:.2f} s from camera stop to process end")
        time.sleep(1.5)
        check(s, "arm holds still after the refusal", B.ptp(r.t1 + 0.5, time.monotonic()) < 1e-4, f"ptp {B.ptp(r.t1 + 0.5, time.monotonic()):.2e}")
        e = r.end()
        check(s, "hold sequence ran", (e.get("stop") or {}).get("held") is True, str(e.get("stop")))
    finally:
        stub.stop()
        cams(True)


@scenario
def stale_camera_observe():
    s = "stale_camera_observe"
    stub = Stub("good")
    try:
        r = Roll(s, ["--set", "observe.period_s=0.2", "--max-time", "120"])
        wait_for(lambda: len(r.recs()) >= 2, 30)
        cams_unit(False)
        got_stale = wait_for(lambda: any(x["verdict"] == "stale" for x in r.recs()), 10)
        cams(True)
        n = len(r.recs())
        got_back = wait_for(lambda: any(x["verdict"] == "pass" for x in r.recs()[n:]), 20)
        r.sig(signal.SIGINT)
        rc = r.wait(20)
        v = [x["verdict"] for x in r.recs()]
        check(s, "observe: pass -> stale -> pass again when the camera comes back", got_stale and got_back,
              f"verdict sequence (compressed): {[k for i, k in enumerate(v) if i == 0 or k != v[i - 1]]}")
        stale = next((x for x in r.recs() if x["verdict"] == "stale"), {})
        check(s, "stale record names the topic and the age", any("s old" in p for p in stale.get("problems", [])), str(stale.get("problems")))
        check(s, "observe never commanded anything (exit 130 from our SIGINT, no goals in records)", rc == 130 and not any(x["sent"] for x in r.recs()), f"rc={rc}")
    finally:
        stub.stop()
        cams(True)
    # no camera at startup
    cams(False)
    r = Roll(s + "_start", ["--max-queries", "1", "--set", "ros.wait_for_inputs_s=2"])
    rc = r.wait(40)
    check(s, "no camera at start -> exit 4 'no inputs'", rc == 4 and "no inputs" in r.end().get("reason", ""), f"rc={rc} {r.end().get('reason')}")
    cams(True)


def interrupt_scenario(s, how):
    B.goto(START, 6)
    stub = Stub("fast", ["--fast-speed", "0.2", "--target", "0.5,0.9,1.4,0.3,1.45,0.3"])
    try:
        r = Roll(s, MOCKYES + ["--max-steps", "160", "--set", "execution.horizon=16"], stdin=(how == "enter"))
        t_wait = time.monotonic()
        while time.monotonic() - t_wait < 30:  # wait until the arm really moves
            if np.abs(B.q() - np.array(START)).max() > 0.01:
                break
            time.sleep(0.02)
        time.sleep(0.3)
        q_at_sig = B.q()
        g_active = len(B.active_goals())
        t_sig = time.monotonic()
        if how == "sigint":
            r.sig(signal.SIGINT)
        elif how == "sigint2":
            r.sig(signal.SIGINT); time.sleep(0.15); r.sig(signal.SIGINT); time.sleep(0.15); r.sig(signal.SIGINT)
        elif how == "enter":
            r.proc.stdin.write(b"\n"); r.proc.stdin.flush()
        rc = r.wait(30)
        t_end = time.monotonic()
        time.sleep(1.5)
        qe = B.q()
        moved_after = float(np.abs(qe - q_at_sig).max())
        check(s, "a chunk goal was running when the stop came", g_active >= 1, f"{g_active} active goal(s), arm {np.abs(q_at_sig - np.array(START)).max():.3f} rad from start")
        check(s, "process ended with exit 130 within 6 s", rc == 130 and t_end - t_sig < 6, f"rc={rc}, {t_end - t_sig:.2f} s after the stop request; reason={r.end().get('reason')!r}")
        check(s, "cancel confirmed and hold goal finished", (r.end().get("stop") or {}).get("cancelled") is True and (r.end().get("stop") or {}).get("held") is True, str(r.end().get("stop")))
        # the goal would have run ~2 s more: arm must have stopped within a few cm
        check(s, "arm stopped almost immediately (0.2 rad/s plan: <= 0.1 rad after the request)", moved_after < 0.1, f"moved {moved_after:.4f} rad after the stop request, planned goal 0.21 rad")
        check(s, "arm stationary after the exit", B.ptp(t_end + 0.2, time.monotonic()) < 1e-4, f"ptp {B.ptp(t_end + 0.2, time.monotonic()):.2e}")
        check(s, "no goal left active on the controller", len(B.active_goals()) == 0, f"active: {B.active_goals()}")
    finally:
        stub.stop()


@scenario
def stop_ctrl_c():
    interrupt_scenario("stop_ctrl_c", "sigint")


@scenario
def stop_ctrl_c_repeated():
    interrupt_scenario("stop_ctrl_c_repeated", "sigint2")


@scenario
def stop_enter_key():
    interrupt_scenario("stop_enter_key", "enter")


@scenario
def stop_on_tracking_error():
    """in-flight monitor: absurdly tight tracking tolerance -> StopRun inside the wait loop -> cancel + hold"""
    s = "stop_on_tracking_error"
    B.goto(START, 6)
    stub = Stub("fast", ["--fast-speed", "0.2", "--target", "0.5,0.9,1.4,0.3,1.45,0.3"])
    try:
        t0 = time.monotonic()
        r = Roll(s, MOCKYES + ["--max-steps", "160", "--set", "execution.horizon=16", "--set", "execution.tracking_error_rad=0.00001"])
        rc = r.wait(40)
        t1 = time.monotonic()
        time.sleep(1.0)
        check(s, "exit 5 with reason tracking error", rc == 5 and "tracking error" in r.end().get("reason", ""), f"rc={rc} {r.end().get('reason')}")
        check(s, "goal cancelled, hold ran, arm still afterwards, no active goal", (r.end().get("stop") or {}).get("held") is True and B.ptp(t1 + 0.3, time.monotonic()) < 1e-4 and not B.active_goals(), str(r.end().get("stop")))
    finally:
        stub.stop()


@scenario
def stop_server_dies():
    s = "stop_server_dies"
    B.goto(START, 6)
    stub = Stub("good", ["--target", "0.6,1.0,1.5,0.3,1.3,0.3"])
    try:
        r = Roll(s, MOCKYES + ["--max-steps", "400", "--set", "server.timeout_s=1.5", "--set", "server.first_query_timeout_s=1.5"])
        time.sleep(2.5)
        stub.proc.terminate()  # our own stub PID
        stub.proc.wait(5)
        rc = r.wait(40)
        time.sleep(1.0)
        check(s, "server gone mid-run -> exit 4 (timeout), hold, still", rc == 4 and "no reply" in " ".join(r.recs()[-1]["problems"]) and B.ptp(r.t1 + 0.3, time.monotonic()) < 1e-4 and not B.active_goals(),
              f"rc={rc} {r.recs()[-1]['problems']}")
    finally:
        stub.stop()


@scenario
def confirmation_prompt():
    s = "confirmation_prompt"
    B.goto(START, 6)
    stub = Stub("good")
    try:
        for label, text, exp_rc, moves in [("no", b"no\n", 2, False), ("EOF/no stdin", b"", 2, False), ("yes", b"yes\n", 0, True)]:
            g0, t0 = B.goals_seen(), time.monotonic()
            r = Roll(f"{s}_{label.split('/')[0]}", ["--execute", "--max-steps", "8"], stdin=True)
            time.sleep(4.0)
            if text:
                r.proc.stdin.write(text); r.proc.stdin.flush()
            r.proc.stdin.close()
            rc = r.wait(40)
            con = r.console()
            moved = B.ptp(t0, time.monotonic()) > 1e-3
            check(s, f"typed {label!r} (no --mock-yes): exit {exp_rc}, {'moves' if moves else 'no motion, no goal'}",
                  rc == exp_rc and moved == moves and ("type  yes" in con) and "mock_components/GenericSystem" in con and (moves or B.goals_seen() == g0),
                  f"rc={rc} moved={moved} new_goals={len(B.goals_seen() - g0)} prompt_shown={'type  yes' in con} shows_hw={'mock_components/GenericSystem' in con}")
        B.goto(START, 6)
        r = Roll(s + "_no_execute_flag", ["--mock-yes", "--max-queries", "1"])
        rc = r.wait(30)
        check(s, "--mock-yes without --execute is still observe mode", rc == 0 and r.recs() and r.recs()[0]["mode"] == "observe" and r.recs()[0]["sent"] is False, f"rc={rc}")
    finally:
        stub.stop()


# ====================================================================================================== TurboVLA backend
TVCFG = TOOL / "rollout_config_turbovla.yaml"
TV = ["--config", str(TVCFG)]
TVX = TV + MOCKYES


def ref_ensemble(chunks, k=0.01, H=30):
    """Independent re-implementation of the semantics of ensemble_recorded_chunks for ONE episode (written out here, not imported):
    at frame i the chunk issued at frame s (i-H < s <= i) contributes its element i-s; contributors ordered oldest first,
    weights exp(-k*j), j = 0, 1, ... normalised."""
    out = []
    for i in range(len(chunks)):
        cand = [np.asarray(chunks[s])[i - s] for s in range(max(0, i - H + 1), i + 1)]
        w = np.exp(-k * np.arange(len(cand)))
        out.append((np.array(cand) * (w / w.sum())[:, None]).sum(axis=0))
    return np.array(out)


def tv_stub(mode, extra=()):
    return Stub(mode, extra, backend="turbovla")


@scenario
def tv_unit_ensemble():
    s = "tv_unit_ensemble"
    rng = np.random.default_rng(7)
    chunks = rng.normal(size=(120, 30, 14))
    ens = G.ACTTemporalEnsembler(30, 14, 0.01)
    mine = np.array([ens.step(c, step=i) for i, c in enumerate(chunks)])
    ref = ref_ensemble(chunks)
    check(s, "client ACTTemporalEnsembler == independent reference of ensemble_recorded_chunks (120 steps, horizon 30, k=0.01)",
          np.allclose(mine, ref, atol=1e-12, rtol=0), f"max abs diff {np.abs(mine - ref).max():.2e}")
    # two episodes with a reset in between
    ens.reset()
    mine2 = []
    for ep in range(2):
        ens.reset()
        mine2.append(np.array([ens.step(c, step=i) for i, c in enumerate(chunks[ep * 60:(ep + 1) * 60])]))
    ref2 = [ref_ensemble(chunks[ep * 60:(ep + 1) * 60]) for ep in range(2)]
    check(s, "reset() restarts the episode (2 x 60 steps equal the per-episode reference)",
          all(np.allclose(a, b, atol=1e-12, rtol=0) for a, b in zip(mine2, ref2)), "")
    # the bundle's own code, when it is on this machine
    bund = Path(os.path.expanduser("~/turbovla/compal-turbovla-handoff-v1/source/experiments/compal/act_temporal.py"))
    if bund.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("bundle_act_temporal", bund)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        eps = np.r_[np.zeros(70, int), np.ones(50, int)]
        bundled = mod.ensemble_recorded_chunks(chunks, eps, decay=0.01)
        e2 = G.ACTTemporalEnsembler(30, 14, 0.01)
        out = []
        for i, c in enumerate(chunks):
            if i == 0 or eps[i] != eps[i - 1]:
                e2.reset(); st = 0
            out.append(e2.step(c, step=st)); st += 1
        check(s, "identical to the TurboVLA bundle's ensemble_recorded_chunks (2 episodes)", np.array_equal(np.array(out), bundled) or np.allclose(np.array(out), bundled, atol=1e-13, rtol=0),
              f"max abs diff {np.abs(np.array(out) - bundled).max():.2e}")
    else:
        check(s, "bundle act_temporal.py present (comparison against the original code)", True, f"{bund} not found: skipped")
    # hand-computed: chunk s element j = 10*s + j  ->  step 2 candidates 2, 11, 20 (oldest first)
    ch = [np.full((30, 14), 10.0 * k) + np.arange(30)[:, None] for k in range(3)]
    e3 = G.ACTTemporalEnsembler(30, 14, 0.01)
    outs = [e3.step(c, step=i) for i, c in enumerate(ch)]
    w = np.exp(-0.01 * np.arange(3))
    hand = (2 * w[0] + 11 * w[1] + 20 * w[2]) / w.sum()
    check(s, "hand-computed step 2 = (2*w0 + 11*w1 + 20*w2)/sum(w), w = exp(-0.01*[0,1,2]) (oldest first, element t-s)", np.allclose(outs[2], hand, atol=1e-12) and np.allclose(outs[0], 0.0),
          f"{outs[2][0]:.6f} vs {hand:.6f}; step 0 = {outs[0][0]:.3f}")
    e4 = G.ACTTemporalEnsembler(30, 14, 5.0)
    o = [e4.step(c, step=i) for i, c in enumerate(ch)]
    check(s, "large decay -> dominated by the OLDEST contributor (ACT convention): step 2 ~ 2, not the newest 20", abs(o[2][0] - 2.0) < 0.2, f"{o[2][0]:.4f}")
    e5 = G.ACTTemporalEnsembler(30, 14, 0.0)
    o = [e5.step(c, step=i) for i, c in enumerate(ch)]
    check(s, "decay 0 -> plain mean of the contributors (2, 11, 20 -> 11)", np.allclose(o[2], 11.0), f"{o[2][0]:.4f}")
    e6 = G.ACTTemporalEnsembler(30, 14, 0.01)
    sizes = []
    for i in range(40):
        e6.step(chunks[i], step=i); sizes.append(e6.history_size)
    check(s, "history grows 1..30 then stays at the horizon (40 steps)", sizes[:30] == list(range(1, 31)) and set(sizes[30:]) == {30}, f"{sizes[:3]}..{sizes[28:32]}..{sizes[-1]}")
    # causality: later chunks do not change earlier outputs
    chunks_b = chunks.copy(); chunks_b[50:] += 5.0
    eb = G.ACTTemporalEnsembler(30, 14, 0.01)
    ob = np.array([eb.step(c, step=i) for i, c in enumerate(chunks_b)])
    check(s, "causal: changing chunks from step 50 on leaves steps 0..49 unchanged", np.array_equal(ob[:50], mine[:50]) and not np.allclose(ob[50:], mine[50:]), "")
    bad = []
    e7 = G.ACTTemporalEnsembler(30, 14, 0.01)
    e7.step(chunks[0], step=0)
    for label, fn in [("skipped step", lambda: e7.step(chunks[1], step=2)), ("repeated step", lambda: e7.step(chunks[1], step=0)),
                      ("NaN chunk", lambda: e7.step(np.where(np.arange(30)[:, None] == 3, np.nan, chunks[1]), step=1)),
                      ("wrong shape", lambda: e7.step(chunks[1][:, :13], step=1)), ("float step", lambda: e7.step(chunks[1], step=1.0))]:
        try:
            fn(); bad.append((label, "accepted"))
        except ValueError:
            pass
    check(s, "rejects skipped / repeated step, NaN, wrong shape (ValueError, state unchanged)", not bad and e7.history_size == 1 and e7.next_step == 1, str(bad))
    for kw in ({"horizon": 0}, {"action_dim": 0}, {"decay": -1}, {"decay": float("nan")}):
        try:
            G.ACTTemporalEnsembler(**{"horizon": 30, "action_dim": 14, "decay": 0.01, **kw}); ok = False
        except ValueError:
            ok = True
        check(s, f"constructor rejects {kw}", ok, "")


@scenario
def tv_unit_preprocess():
    s = "tv_unit_preprocess"
    import cv2
    import PIL
    from PIL import Image as PILImage
    cfg = G.load_config(TVCFG)
    m, pp = cfg["model"], cfg["preprocess"]
    check(s, "turbovla config: recorder_pil, 640x480 intermediate, 224x224 output", pp["mode"] == "recorder_pil" and m["recorder_size"] == [640, 480] and m["image_size"] == [224, 224], f"{pp['mode']} {m['recorder_size']} {m['image_size']}")
    rng = np.random.default_rng(2)

    def make(w, h):
        yy, xx = np.mgrid[0:h, 0:w]
        tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32), (0, 0), 4) * 60
        img = np.clip(np.stack([xx * 200 // w, yy * 200 // h, 100 + 0 * xx], -1).astype(np.float32) + 20 + tex, 0, 255).astype(np.uint8)
        cv2.circle(img, (w * 9 // 20, h * 10 // 21), h // 5, (20, 200, 240), -1)
        cv2.putText(img, "FLAP 123", (w // 6, h * 3 // 4), cv2.FONT_HERSHEY_SIMPLEX, h / 180, (255, 255, 255), max(2, h // 90))
        return img

    def reference(jpg):
        """Independent: recorder image_msg_to_rgb(min_size=(640,480)) + resize_rgb, then the TurboVLA data pipeline's per-frame
        Image.fromarray(rgb).resize((224,224)) with NO resample argument (Pillow default)."""
        img = PILImage.open(io.BytesIO(bytes(jpg)))
        img.draft("RGB", (640, 480))
        rgb = np.asarray(img.convert("RGB"))
        if rgb.shape[:2] != (480, 640):
            rgb = np.asarray(PILImage.fromarray(rgb).resize((640, 480), PILImage.BILINEAR))
        return rgb, np.asarray(PILImage.fromarray(rgb).resize((224, 224)))

    for (w, h) in ((1920, 1080), (1280, 720)):
        jpg = cv2.imencode(".jpg", make(w, h), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        mine, full = G.turbovla_rgb(np.frombuffer(jpg, np.uint8), m, pp)
        rgb640, ref = reference(jpg)
        check(s, f"{w}x{h}: output (224,224,3) uint8, reported source size {w}x{h}", mine.shape == (224, 224, 3) and mine.dtype == np.uint8 and list(full) == [w, h], f"{mine.shape} {mine.dtype} {full}")
        check(s, f"{w}x{h}: identical to the independent PIL chain (draft, convert, BILINEAR 640x480, default-resample 224)", np.array_equal(mine, ref), f"max abs diff {np.abs(mine.astype(int) - ref.astype(int)).max()}")
        bic = np.asarray(PILImage.fromarray(rgb640).resize((224, 224), PILImage.BICUBIC))
        check(s, f"{w}x{h}: Pillow {PIL.__version__} default resample == BICUBIC", np.array_equal(mine, bic), "")
        for name, flt in (("BILINEAR", PILImage.BILINEAR), ("NEAREST", PILImage.NEAREST), ("LANCZOS", PILImage.LANCZOS)):
            alt = np.asarray(PILImage.fromarray(rgb640).resize((224, 224), flt))
            if w == 1920:
                check(s, f"resample choice matters: {name} differs from the default (mean abs diff {np.abs(alt.astype(int) - mine.astype(int)).mean():.2f})", not np.array_equal(alt, mine), "")
        if w == 1920:
            # Pillow 12.3 (bundle) vs this Pillow: bit-identical resize?  (needs the bundle's cached PIL + python3.12; no numpy needed there)
            import glob, shutil, tempfile
            cands = glob.glob(os.path.expanduser("~/turbovla/compal-turbovla-handoff-v1/.cache/uv/archive-v0/*/PIL/_imaging.cpython-312*.so"))
            py12 = shutil.which("python3.12") or os.path.expanduser("~/.local/bin/python3.12")
            if cands and os.path.exists(py12):
                site = str(Path(cands[0]).parent.parent)
                d = Path(tempfile.mkdtemp(prefix="tv_pil12_"))
                (d / "in.raw").write_bytes(rgb640.tobytes())
                (d / "r.py").write_text("import sys\nsys.path.insert(0, %r)\nfrom PIL import Image\nimport PIL\n"
                                        "im = Image.frombytes('RGB', (640, 480), open(%r, 'rb').read())\n"
                                        "open(%r, 'wb').write(im.resize((224, 224)).tobytes())\nprint(PIL.__version__)\n" % (site, str(d / "in.raw"), str(d / "out.raw")))
                p = subprocess.run([py12, "-I", str(d / "r.py")], capture_output=True, timeout=60)
                out12 = np.frombuffer((d / "out.raw").read_bytes(), np.uint8).reshape(224, 224, 3) if (d / "out.raw").exists() else None
                check(s, f"bundle Pillow {p.stdout.decode().strip()} (python3.12) default resize == this client's Pillow {PIL.__version__} output, bit for bit",
                      out12 is not None and np.array_equal(out12, mine), f"rc={p.returncode} " + (f"max abs diff {np.abs(out12.astype(int) - mine.astype(int)).max()}" if out12 is not None else p.stderr.decode()[-200:]))
                shutil.rmtree(d, ignore_errors=True)
            else:
                check(s, "bundle Pillow 12 available for a cross-version check", True, "bundle PIL / python3.12 not found: skipped")
    try:
        G.turbovla_rgb(np.frombuffer(jpg, np.uint8), m, {"mode": "opencv_legacy"}); ok = False
    except ValueError:
        ok = True
    check(s, "preprocess.mode other than recorder_pil -> ValueError", ok, "")
    try:
        G.turbovla_rgb(np.frombuffer(b"not an image", np.uint8), m, pp); ok = False
    except ValueError:
        ok = True
    check(s, "undecodable bytes -> ValueError", ok, "")
    img = np.zeros((224, 224, 3), np.uint8)
    obs = G.build_observation_turbovla({"head": img, "wrist_left": img}, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 7, G.tv_arm_channels(m))
    check(s, "observation: state float64 (14,) = [q1..6, 0 x 8] (left gripper, right arm, right gripper exactly 0), step int, keys exact",
          set(obs) == {"state", "head", "wrist_left", "step"} and obs["state"].shape == (14,) and obs["state"].dtype == np.float64
          and np.allclose(obs["state"][:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) and not obs["state"][6:].any() and obs["step"] == 7 and type(obs["step"]) is int,
          f"{obs['state'].tolist()}")


@scenario
def tv_unit_gates():
    s = "tv_unit_gates"
    cfg = G.load_config(TVCFG)
    g, tr = cfg["gates"], cfg["training_range"]
    lim = np.array([[-3.1067, 3.1067], [-1.3788, 2.4260], [-1.3788, 2.4260], [-3.1067, 3.1067], [-1.6406, 1.6406], [-3.1067, 3.1067]])
    q = np.array(START)

    def mk(q0, per_row=0.002):
        c = np.full((30, 14), 0.7)
        c[:, :6] = q0 + per_row * np.arange(1, 31)[:, None]
        return c
    good = mk(q)
    ev = lambda ch, qq=q, **kw: G.evaluate_gates_turbovla(ch, qq, lim, {**g, **kw}, tr["min"], tr["max"], G.tv_arm_channels(cfg["model"]))
    bad = lambda r: sorted(x["name"] for x in r if not x["ok"])
    check(s, "good chunk (channels 6..13 = 0.7, ignored) passes every raw gate", bad(ev(good)) == [], str(bad(ev(good))))
    check(s, "gate names", sorted(x["name"] for x in ev(good)) == sorted(["state_in_training_range", "finite", "first_step", "row_step", "urdf_limits", "training_range"]), str([x["name"] for x in ev(good)]))
    c = good.copy(); c[3, 2] = np.nan
    check(s, "NaN in a used channel -> finite only", bad(ev(c)) == ["finite"], str(bad(ev(c))))
    c = good.copy(); c[3, 9] = np.nan
    check(s, "NaN in an ignored channel (9) still -> finite (the whole 30x14 chunk must be finite)", bad(ev(c)) == ["finite"], str(bad(ev(c))))
    c = good.copy(); c[:, 8] = 1e3
    check(s, "huge FINITE values in ignored channels pass (only channels 0..5 are used)", bad(ev(c)) == [], str(bad(ev(c))))
    c = good.copy(); c[:, 0] += 0.06
    check(s, "row 0 0.06 off the state -> first_step", "first_step" in bad(ev(c)), str(bad(ev(c))))
    c = good.copy(); c[12:, 1] += 0.2
    check(s, "0.2 jump at row 12 -> row_step", "row_step" in bad(ev(c)), str(bad(ev(c))))
    check(s, "same jump with check_all_rows=false passes (row 0 only is judged)", bad(ev(c, check_all_rows=False)) == [], str(bad(ev(c, check_all_rows=False))))
    qn = np.array([0.3, 0.7, 1.2, 0.0, 1.50, 0.0]); c = mk(qn, 0.0); c[:, 4] += np.linspace(0.0, 0.1, 30)
    r = ev(c, qn, training_margin_rad=1.0)
    check(s, "joint5 up to 1.60 > 1.6406-0.05 -> urdf_limits only", bad(r) == ["urdf_limits"], str(bad(r)))
    check(s, "... and with check_all_rows=false the late rows are not looked at -> pass", bad(ev(c, qn, training_margin_rad=1.0, check_all_rows=False)) == [], str(bad(ev(c, qn, training_margin_rad=1.0, check_all_rows=False))))
    qn = np.array([0.7, 0.7, 1.2, 0.0, 0.3, 0.0]); c = mk(qn, 0.0); c[:, 0] += np.linspace(0.0, 0.1, 30)
    r = ev(c, qn, state_in_training_range=False)
    check(s, "joint1 0.8 > 0.68114+0.1 -> training_range only", bad(r) == ["training_range"], str(bad(r)))
    check(s, "state at zeros (joint3 < 0.25993-0.1) -> state_in_training_range", "state_in_training_range" in bad(ev(mk(np.zeros(6)), np.zeros(6))), str(bad(ev(mk(np.zeros(6)), np.zeros(6)))))
    check(s, "mocap start pose [0,0,1.0472,0,0.8727,0] is inside the training range +- margin", bad(ev(mk(np.array([0, 0, 1.0472, 0, 0.8727, 0.0])), np.array([0, 0, 1.0472, 0, 0.8727, 0.0]))) == [], "")
    # target gates
    tg = lambda t, qq=q, **kw: G.evaluate_target_gates(t, qq, lim, {**g, **kw}, tr["min"], tr["max"])
    check(s, "target 0.01 from the state passes all four target gates", bad(tg(q + 0.01)) == [] and len(tg(q + 0.01)) == 4, str(bad(tg(q + 0.01))))
    check(s, "target 0.06 from the state -> target_step", bad(tg(q + 0.06)) == ["target_step"], str(bad(tg(q + 0.06))))
    t = q.copy(); t[2] = np.nan
    check(s, "NaN target -> target_finite", bad(tg(t)) == ["target_finite"], str(bad(tg(t))))
    qn = np.array([0.3, 0.7, 1.2, 0.0, 1.60, 0.0])
    check(s, "target joint5 = 1.62 -> target_urdf_limits (only)", bad(tg(qn + np.array([0, 0, 0, 0, 0.02, 0]), qn, training_margin_rad=1.0)) == ["target_urdf_limits"], str(bad(tg(qn + np.array([0, 0, 0, 0, 0.02, 0]), qn, training_margin_rad=1.0))))
    # shape handling
    for shape, ok in [((30, 14), True), ((1, 30, 14), False), ((30, 13), False), ((29, 14), False), ((420,), False)]:
        a, prob = G.tv_normalize_chunk(np.zeros(shape, np.float32), 30, 14)
        check(s, f"tv_normalize_chunk{shape}", (a is not None) == ok, str(prob))
    a, prob = G.tv_normalize_chunk(np.array([["a"] * 14] * 30), 30, 14)
    check(s, "tv_normalize_chunk string dtype refused", a is None, str(prob))
    # timing
    t = G.trajectory_times(q, [q + 0.0005], 1 / 30, 0.1)
    check(s, "tiny step keeps the nominal 1/30 s", abs(t[0] - 1 / 30) < 1e-12, f"{t[0]:.5f}")
    t = G.trajectory_times(q, [q + 0.03], 1 / 30, 0.1)
    check(s, "0.03 rad step stretched to 0.3 s (= 0.1 rad/s cap)", abs(t[0] - 0.3) < 1e-9, f"{t[0]:.5f}")
    # modality comparison
    ok_reply = {"backend": "turbovla", "cameras": ["head", "knife", "wrist_left"], "client_cameras": ["head", "wrist_left"], "image_size": [224, 224],
                "state_dim": 14, "action_dim": 14, "horizon": 30, "control_rate_hz": 30, "checkpoint_sha256": "ab" * 32, "language": "Open the box flaps."}
    cmp = lambda rep, c=cfg: G.compare_modality_turbovla(rep, c)[0]
    check(s, "matching modality reply -> no problems (also control_rate_hz 30.0)", cmp(ok_reply) == [] and cmp({**ok_reply, "control_rate_hz": 30.0}) == [], str(cmp(ok_reply)))
    for label, rep in [("GR00T-style reply (no backend)", {"video": {"as_json": "{}"}}), ("horizon 16", {**ok_reply, "horizon": 16}),
                       ("missing client camera", {**ok_reply, "client_cameras": ["head"]}), ("server cameras without knife", {**ok_reply, "cameras": ["head", "wrist_left"]}),
                       ("image_size 480x640", {**ok_reply, "image_size": [480, 640]}), ("state_dim 12", {**ok_reply, "state_dim": 12}),
                       ("rate 15", {**ok_reply, "control_rate_hz": 15}), ("other language", {**ok_reply, "language": "Close the box."}),
                       ("backend groot", {**ok_reply, "backend": "groot"})]:
        check(s, f"modality mismatch detected: {label}", len(cmp(rep)) >= 1, str(cmp(rep))[:150])
    check(s, "empty / non-dict reply -> not understood (info None: warn only)", G.compare_modality_turbovla({}, cfg)[1] is None and G.compare_modality_turbovla(None, cfg)[1] is None, "")
    cfg2 = {**cfg, "server": {**cfg["server"], "expect_checkpoint_sha256": "cd" * 32}}
    check(s, "expect_checkpoint_sha256 mismatch detected", len(cmp(ok_reply, cfg2)) == 1 and "checkpoint_sha256" in cmp(ok_reply, cfg2)[0], str(cmp(ok_reply, cfg2)))
    # config handling
    rc = G.main(["--config", str(TVCFG), "--horizon", "4"])
    check(s, "--horizon with the turbovla config -> exit 2", rc == 2, f"rc={rc}")
    rc = G.main(["--set", "policy.backend=turbovla"])
    check(s, "turbovla backend on the GR00T config (sections missing) -> exit 2", rc == 2, f"rc={rc}")
    rc = G.main(["--config", str(TVCFG), "--set", "policy.backend=nope"])
    check(s, "unknown backend -> exit 2", rc == 2, f"rc={rc}")
    rc = G.main(["--config", str(TVCFG), "--set", "ensemble.horizon=16"])
    check(s, "ensemble.horizon != chunk_len -> exit 2", rc == 2, f"rc={rc}")
    rc = G.main(["--config", str(TVCFG), "--set", "execution.max_joint_vel=0.9"])
    check(s, "max_joint_vel 0.9 -> exit 2", rc == 2, f"rc={rc}")


@scenario
def tv_unit_wire():
    s = "tv_unit_wire"
    src = Path(os.path.expanduser("~/groot/Isaac-GR00T/gr00t/policy/server_client.py"))
    if not src.exists():
        check(s, "Isaac-GR00T server_client.py present", True, f"{src} missing: wire test against the real MsgSerializer skipped")
        return
    import msgpack, msgpack_numpy as mnp, zmq
    text = src.read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "MsgSerializer")
    ns = dict(functools=functools, io=io, json=json, msgpack=msgpack, mnp=mnp, np=np, Any=typing.Any,
              ModalityConfig=type("ModalityConfig", (), {}), to_json_serializable=lambda x: x)
    exec(ast.get_source_segment(text, cls), ns)
    Ser = ns["MsgSerializer"]
    got = {}

    def server():
        ctx = zmq.Context(); sock = ctx.socket(zmq.REP); sock.bind(f"tcp://127.0.0.1:{PORT + 1}")
        sock.RCVTIMEO = 8000
        for _ in range(2):
            try:
                req = Ser.from_bytes(sock.recv())
            except zmq.Again:
                break
            got[req["endpoint"]] = req
            if req["endpoint"] == "get_action":
                a = np.tile(req["data"]["observation"]["state"], (30, 1)).astype(np.float32)
                sock.send(Ser.to_bytes([{"actions": a}, {"inference_ms": 1.5, "step": req["data"]["observation"]["step"]}]))
            else:
                sock.send(Ser.to_bytes({"status": "ok"}))
        sock.close(linger=0); ctx.term()
    th = threading.Thread(target=server, daemon=True); th.start()
    c = G.Gr00tClient("127.0.0.1", PORT + 1, 5.0, label="TurboVLA")
    c.call("reset", None, timeout_s=4)
    img = np.full((224, 224, 3), 7, np.uint8)
    act, info = c.get_action_full(G.build_observation_turbovla({"head": img, "wrist_left": img}, START, 3, slice(0, 6)), timeout_s=4)
    th.join(5); c.close()
    o = got["get_action"]["data"]["observation"]
    check(s, "real MsgSerializer decodes the request (state float64 (14,), images uint8 (224,224,3), step 3, options None); reset carries no data",
          o["state"].shape == (14,) and o["state"].dtype == np.float64 and o["head"].shape == (224, 224, 3) and o["head"].dtype == np.uint8 and o["step"] == 3
          and got["get_action"]["data"]["options"] is None and "data" not in got["reset"], f"{o['state'].dtype} {o['head'].dtype} reset={got['reset']}")
    check(s, "client decodes the real MsgSerializer reply [{'actions': (30,14) float32}, {'inference_ms', 'step'}]",
          act["actions"].shape == (30, 14) and act["actions"].dtype == np.float32 and np.allclose(act["actions"][5, :6], START, atol=1e-6) and info == {"inference_ms": 1.5, "step": 3}, f"{act['actions'].shape} {info}")


@scenario
def tv_stub_contract():
    """the stub's observation validation is exact (a client bug must not slip through the mock tests)"""
    s = "tv_stub_contract"
    B.goto(START, 6)
    stub = tv_stub("good")
    try:
        c = G.Gr00tClient("127.0.0.1", PORT, 5.0, label="TurboVLA")
        img = np.zeros((224, 224, 3), np.uint8)
        good = lambda step=0: G.build_observation_turbovla({"head": img, "wrist_left": img}, START, step, slice(0, 6))
        def attempt(obs):
            try:
                c.get_action_full(obs, timeout_s=3); return None
            except G.ServerError as e:
                return str(e)
        c.call("reset", None)
        check(s, "valid observation accepted, reply is a (30,14) float32 'actions' chunk + info with step",
              (lambda r: r[0]["actions"].shape == (30, 14) and r[0]["actions"].dtype == np.float32 and r[1]["step"] == 0 and r[1]["inference_ms"] >= 0)(c.get_action_full(good(0), timeout_s=3)), "")
        o = good(1); o["state"][9] = 0.01
        check(s, "right-arm state channel != 0 rejected", "must be exactly 0" in (attempt(o) or ""), "")
        o = good(1); o["state"][6] = 0.5
        check(s, "left gripper channel != 0 rejected", "must be exactly 0" in (attempt(o) or ""), "")
        o = good(1); o["extra"] = 1
        check(s, "extra key rejected", "keys" in (attempt(o) or ""), "")
        o = good(1); del o["step"]
        check(s, "missing step rejected", "keys" in (attempt(o) or ""), "")
        o = good(1); o["head"] = np.zeros((480, 640, 3), np.uint8)
        check(s, "640x480 image rejected (must be 224x224)", "shape" in (attempt(o) or ""), "")
        o = good(1); o["wrist_left"] = img.astype(np.float32)
        check(s, "float image rejected (must be uint8)", "uint8" in (attempt(o) or ""), "")
        o = good(1); o["state"] = o["state"][:12]
        check(s, "12-D state rejected", "state shape" in (attempt(o) or ""), "")
        o = good(1); o["state"][2] = np.nan
        check(s, "NaN state rejected", "not finite" in (attempt(o) or ""), "")
        check(s, "step jump 0 -> 5 rejected", "must repeat or advance" in (attempt(good(5)) or ""), "")
        check(s, "a repeat of the last accepted step (0) and then an advance by 1 are accepted", attempt(good(0)) is None and attempt(good(1)) is None, "")
        c.call("reset", None)
        check(s, "after reset the sequence restarts at 0 (step 1 rejected, 0 accepted)", attempt(good(1)) is not None and attempt(good(0)) is None, "")
        c.close()
    finally:
        stub.stop()


@scenario
def tv_observe():
    s = "tv_observe"
    B.goto(START, 6)
    stub = tv_stub("good")
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s, TV + ["--max-queries", "8", "--set", "observe.period_s=0.15", "--save-images", "--set", "logging.log_full_chunk=true"])
        rc = r.wait(60)
        t1 = time.monotonic()
        q = r.recs()
        check(s, "exit 0, 8 queries, all PASS", rc == 0 and len(q) == 8 and all(x["verdict"] == "pass" for x in q), f"rc={rc} verdicts={[x['verdict'] for x in q]}")
        check(s, "arm not moved (observe never commands), no goal created, nothing sent", B.ptp(t0, t1) < 1e-4 and B.goals_seen() == g0 and not any(x["sent"] for x in q),
              f"ptp {B.ptp(t0, t1):.2e} rad, new goals {len(B.goals_seen() - g0)}")
        steps = [x["step"] for x in stub.records()]
        check(s, "stub received 8 observations with consecutive steps 0..7 (it validated keys/dtypes/shapes/zeros)", steps == list(range(8)), str(steps))
        check(s, "ensembler ran on the static state: steps 0..7, history sizes 1..8", [x["ensemble"]["step"] for x in q] == list(range(8)) and [x["ensemble"]["history_size"] for x in q] == list(range(1, 9)),
              f"hist {[x['ensemble']['history_size'] for x in q]}")
        chunks = [np.array(x["chunk"]) for x in q]
        ref = ref_ensemble(chunks)[:, :6]
        got = np.array([x["ensemble"]["target"] for x in q])
        check(s, "logged ensembled targets == independent reference ensemble of the logged raw chunks", np.allclose(got, ref, atol=1e-9, rtol=0), f"max abs diff {np.abs(got - ref).max():.2e}")
        check(s, "observe logs what it WOULD command (would_command == ensembled target)", all(np.allclose(x["would_command"], x["ensemble"]["target"]) for x in q), "")
        import cv2
        imgs = sorted((r.dir / "images").glob("*.jpg"))
        shapes = {cv2.imread(str(p)).shape for p in imgs}
        check(s, "model input images saved: 2 per query, 224x224", len(imgs) == 16 and shapes == {(224, 224, 3)}, f"{len(imgs)} files {shapes}")
        q0 = q[0]
        need = ("t_iso", "state", "gates", "latency_ms", "server_info", "chunk_summary", "ensemble", "cycle_ms", "image_age_s", "image_size_wh", "step")
        check(s, "record has time/state/gates/latency/server_info/chunk_summary/ensemble/cycle", all(k in q0 for k in need) and q0["image_size_wh"] == {"head": [1920, 1080], "wrist_left": [1280, 720]},
              f"latency {q0['latency_ms']} ms server {q0['server_info']}, gates {[g['name'] for g in q0['gates']]}")
        check(s, "console: modality line shows backend turbovla", "server modality: backend turbovla" in r.console(), "")
    finally:
        stub.stop()
    # default logging: summary only
    stub = tv_stub("good")
    try:
        r = Roll(s + "_summary", TV + ["--max-queries", "2", "--set", "observe.period_s=0.1"])
        rc = r.wait(60); q = r.recs()
        cs = q[0]["chunk_summary"] if q else {}
        check(s, "default logging: no full chunk, chunk_summary with row0/row_last/min/max/ignored_channels_absmax=0.7", rc == 0 and "chunk" not in q[0] and {"row0", "row_last", "min", "max", "row0_minus_state_max", "max_row_step", "ignored_channels_absmax"} <= set(cs) and abs(cs["ignored_channels_absmax"] - 0.7) < 1e-6,
              f"rc={rc} keys {sorted(cs)}")
    finally:
        stub.stop()


@scenario
def tv_check_mode():
    s = "tv_check_mode"
    r = Roll(s, TV + ["--check"])
    rc = r.wait(40)
    con = r.console()
    check(s, "--check with the turbovla config: exit 0, shows recorder->224 preprocessing, both cameras decoded, mock hardware, no server needed",
          rc == 0 and "[640, 480] (recorder) then [224, 224]" in con and "camera head" in con and "size [1920, 1080]" in con and "size [1280, 720]" in con and "hardware_is_mock=True" in con, f"rc={rc}")


@scenario
def tv_check_all_rows_option():
    s = "tv_check_all_rows_option"
    B.goto(START, 6)
    stub = tv_stub("limit")
    try:
        r1 = Roll(s + "_all", TV + ["--max-queries", "1"]); rc1 = r1.wait(40)
        r2 = Roll(s + "_row0", TV + ["--max-queries", "1", "--set", "gates.check_all_rows=false"]); rc2 = r2.wait(40)
        a, b = r1.recs(), r2.recs()
        check(s, "limit stub, check_all_rows=true: raw chunk refused (urdf_limits, training_range)", rc1 == 0 and a and a[0]["verdict"] == "refuse" and {"urdf_limits", "training_range"} <= set(failed_gates(a[0])), f"rc={rc1} failed {failed_gates(a[0]) if a else None}")
        check(s, "limit stub, check_all_rows=false: only row 0 is judged -> pass", rc2 == 0 and b and b[0]["verdict"] == "pass", f"rc={rc2} {[x['verdict'] for x in b]}")
    finally:
        stub.stop()


@scenario
def tv_execute_good():
    s = "tv_execute_good"
    B.goto(START, 6)
    stub = tv_stub("good", ["--speed", "0.08"])
    try:
        g0 = B.goals_seen()
        t0 = time.monotonic()
        r = Roll(s, TVX + ["--max-steps", "60", "--set", "logging.log_full_chunk=true"])
        rc = r.wait(180)
        t1 = time.monotonic()
        q = r.recs()
        ok_all = len(q) == 60 and all(x["verdict"] == "pass" and x.get("exec", {}).get("error_code") == 0 for x in q)
        check(s, "exit 0 after 60 steps, every step passed and its goal succeeded", rc == 0 and ok_all, f"rc={rc} n={len(q)} {sorted(set(x['verdict'] for x in q))}")
        check(s, "exactly one trajectory goal per step (60), no hold at a normal end", len(B.goals_seen() - g0) == 60, f"new goals {len(B.goals_seen() - g0)}")
        pk = B.peak(t0, t1)
        check(s, "velocity cap: independent measurement from /joint_states <= 0.105 rad/s (0.2 s windows, message stamps)", pk <= 0.105, f"peak {pk:.4f} rad/s (stub 0.08 rad/s, cap 0.1)")
        qf = B.q()
        err = np.abs(qf[:3] - np.array(TARGET)[:3]).max()
        check(s, "arm followed the policy toward the stub target (joints 1-3 within 0.03 rad after 60 steps)", err < 0.03, f"final {np.round(qf, 4).tolist()} target {TARGET}, max err j1-3 {err:.4f}")
        steps = [x["step"] for x in stub.records()]
        check(s, "server saw consecutive steps 0..59", steps == list(range(60)), f"{steps[:3]}..{steps[-3:]}")
        h = [x["ensemble"]["history_size"] for x in q]
        check(s, "ensemble history grows 1..30 and stays at 30", h[:30] == list(range(1, 31)) and set(h[30:]) == {30}, f"{h[:3]}..{h[28:32]}..{h[-1]}")
        ref = ref_ensemble([np.array(x["chunk"]) for x in q])[:, :6]
        got = np.array([x["ensemble"]["target"] for x in q])
        check(s, "commanded targets == independent reference ensemble of the raw chunks (60 steps, ensemble wraps past 30)", np.allclose(got, ref, atol=1e-9, rtol=0), f"max abs diff {np.abs(got - ref).max():.2e}")
        d = [x["ensemble"]["target_minus_state_max"] for x in q]
        check(s, "every |target - state| <= 0.05 rad (gate) and the arm reached each target", max(d) <= 0.05 and max(x["exec"]["residual_to_last_point"] for x in q) < 0.01, f"max |target-state| {max(d):.4f}, max residual {max(x['exec']['residual_to_last_point'] for x in q):.5f}")
        check(s, "command peak velocity <= 0.1 rad/s (+1e-6) on every step", max(x["exec"]["cmd_peak_vel"] for x in q) <= 0.1 + 1e-6, f"max {max(x['exec']['cmd_peak_vel'] for x in q):.4f}")
        cyc = np.array([x["cycle_ms"] for x in q[1:]])
        sl = np.array([x["server_info"]["inference_ms"] for x in q])
        chk = np.array([x["latency_ms"] for x in q])
        check(s, "step-synchronous loop rate measured (stub, mock arm; informational)", True,
              f"cycle median {np.median(cyc):.0f} ms (p95 {np.percentile(cyc, 95):.0f}) = {1000 / np.median(cyc):.1f} Hz; get_action round trip median {np.median(chk):.1f} ms (server {np.median(sl):.2f}); goal wall median {np.median([x['exec']['wall_s'] for x in q]) * 1000:.0f} ms; run {t1 - r.t0:.1f} s")
    finally:
        stub.stop()


@scenario
def tv_execute_static():
    s = "tv_execute_static"
    B.goto(START, 6)
    stub = tv_stub("static")
    try:
        t0 = time.monotonic()
        r = Roll(s, TVX + ["--max-steps", "15"])
        rc = r.wait(60); t1 = time.monotonic(); q = r.recs()
        check(s, "static policy: 15 steps pass, goals succeed, arm does not move", rc == 0 and len(q) == 15 and all(x["verdict"] == "pass" for x in q) and B.ptp(t0, t1) < 1e-4, f"rc={rc} n={len(q)} ptp {B.ptp(t0, t1):.2e}")
        check(s, "static: each goal is the nominal 1/30 s", all(abs(x["exec"]["duration_s"] - 1 / 30) < 1e-9 for x in q), "")
    finally:
        stub.stop()


@scenario
def tv_execute_fast_stretch():
    s = "tv_execute_fast_stretch"
    B.goto(START, 6)
    stub = tv_stub("fast", ["--fast-speed", "0.3", "--target", "0.7,1.1,1.5,0.4,1.2,0.4"])
    try:
        t0 = time.monotonic()
        r = Roll(s, TVX + ["--max-steps", "20"])
        rc = r.wait(120); t1 = time.monotonic(); q = r.recs()
        st = np.array([x["exec"]["stretch"] for x in q])
        check(s, "exit 0, 20 steps passed (0.01 rad per row)", rc == 0 and len(q) == 20 and all(x["verdict"] == "pass" for x in q), f"rc={rc} n={len(q)}")
        check(s, "stub asked for 0.3 rad/s: client stretched every goal ~3x (1/30 s -> 0.1 s)", abs(np.median(st) - 3.0) < 0.1 and st.min() > 2.5, f"stretch median {np.median(st):.3f} min {st.min():.3f}")
        pk = B.peak(t0, t1)
        check(s, "measured peak speed <= 0.105 rad/s (cap 0.1)", pk <= 0.105, f"peak {pk:.4f} rad/s")
        check(s, "goal wall time matches the stretched duration", abs(np.median([x["exec"]["wall_s"] - x["exec"]["duration_s"] for x in q])) < 0.12, f"median (wall - planned) {np.median([x['exec']['wall_s'] - x['exec']['duration_s'] for x in q]) * 1000:.0f} ms")
    finally:
        stub.stop()


def tv_bad_mode(s, mode, expect_gates, extra_stub=(), start=START, extra_args=()):
    B.goto(start, 6)
    stub = tv_stub(mode, extra_stub)
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(f"{s}_{mode}", TVX + ["--max-steps", "40", *extra_args])
        rc = r.wait(60); t1 = time.monotonic(); q = r.recs()
        fg = failed_gates(q[0]) if q else None
        check(s, f"{mode}: exit 3 and refused at the first query, nothing sent", rc == 3 and len(q) == 1 and q[0]["verdict"] == "refuse" and q[0]["sent"] is False, f"rc={rc} verdicts={[x['verdict'] for x in q]}")
        check(s, f"{mode}: expected gates failed {expect_gates}", all(e in (fg or []) for e in expect_gates), f"failed gates {fg}")
        check(s, f"{mode}: arm held still", B.ptp(t0, t1) < 1e-4, f"joint ptp {B.ptp(t0, t1):.2e} rad over the run")
        e = r.end()
        check(s, f"{mode}: cancel + hold sequence ran, only the hold goal exists", (e.get("stop") or {}).get("held") is True and len(B.goals_seen() - g0) == 1, f"stop {e.get('stop')} new goals {len(B.goals_seen() - g0)}")
        return fg
    finally:
        stub.stop()


@scenario
def tv_refuse_jump():
    tv_bad_mode("tv_refuse_jump", "jump", ["row_step"])


@scenario
def tv_refuse_nan():
    tv_bad_mode("tv_refuse_nan", "nan", ["finite"])


@scenario
def tv_refuse_limit():
    tv_bad_mode("tv_refuse_limit", "limit", ["urdf_limits", "training_range"])


@scenario
def tv_refuse_relative():
    tv_bad_mode("tv_refuse_relative", "relative", ["first_step"])


@scenario
def tv_refuse_badshape():
    fg = tv_bad_mode("tv_refuse_badshape", "badshape", ["shape"])


@scenario
def tv_refuse_target_step():
    """raw chunk is fine, the ENSEMBLED target exceeds the per-step limit (tightened to 0.0005 rad): only target_step fires"""
    fg = tv_bad_mode("tv_refuse_target_step", "good", ["target_step"], extra_args=["--set", "gates.max_target_step_rad=0.0005"])
    check("tv_refuse_target_step", "only target_step failed (raw gates all passed)", fg == ["target_step"], str(fg))


@scenario
def tv_refuse_first_step():
    fg = tv_bad_mode("tv_refuse_first_step", "good", ["first_step"], extra_args=["--set", "gates.first_step_max_rad=0.0005"])
    check("tv_refuse_first_step", "only first_step failed", fg == ["first_step"], str(fg))


@scenario
def tv_refuse_state_range():
    s = "tv_refuse_state_range"
    B.goto([0, 0, 0, 0, 0, 0], 4)
    stub = tv_stub("good")
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s, TVX + ["--max-steps", "16"])
        rc = r.wait(60); q = r.recs()
        check(s, "refused: exit 3, state_in_training_range failed, nothing sent", rc == 3 and q and "state_in_training_range" in failed_gates(q[0]) and q[0]["sent"] is False, f"rc={rc} failed={failed_gates(q[0]) if q else None}")
        check(s, "arm stayed at zeros", B.ptp(t0, time.monotonic()) < 1e-4 and np.abs(B.q()).max() < 1e-4, f"|q|max={np.abs(B.q()).max():.1e}")
    finally:
        stub.stop()
        B.goto(START, 6)


@scenario
def tv_refuse_slow():
    s = "tv_refuse_slow"
    B.goto(START, 6)
    stub = tv_stub("slow", ["--slow-s", "8"])
    try:
        t0 = time.monotonic()
        r = Roll(s, TVX + ["--max-steps", "16", "--set", "server.timeout_s=2.0", "--set", "server.first_query_timeout_s=2.0"])
        rc = r.wait(60); t1 = time.monotonic(); q = r.recs()
        check(s, "server timeout -> exit 4, verdict server_error, message names TurboVLA", rc == 4 and q and q[0]["verdict"] == "server_error" and "TurboVLA" in q[0]["problems"][0], f"rc={rc} problems={q[0]['problems'] if q else None}")
        check(s, "gave up after 2 x 2 s (one re-send), arm held still", t1 - r.t0 < 9 and B.ptp(t0, t1) < 1e-4, f"run took {t1 - r.t0:.1f} s, ptp {B.ptp(t0, t1):.2e}")
    finally:
        stub.stop()


@scenario
def tv_server_error_midrun():
    s = "tv_server_error_midrun"
    B.goto(START, 6)
    stub = tv_stub("good", ["--error-at", "6"])
    try:
        g0 = B.goals_seen()
        r = Roll(s, TVX + ["--max-steps", "100"])
        rc = r.wait(60); t1 = time.monotonic(); q = r.recs()
        time.sleep(1.0)
        check(s, "5 steps executed, the 6th answered {'error'}: exit 4, verdict server_error", rc == 4 and len(q) == 6 and [x["verdict"] for x in q[:5]] == ["pass"] * 5 and q[5]["verdict"] == "server_error" and "stub error injected" in q[5]["problems"][0],
              f"rc={rc} verdicts={[x['verdict'] for x in q]}")
        check(s, "hold ran, arm still afterwards, no goal left active (5 chunk goals + 1 hold)", (r.end().get("stop") or {}).get("held") is True and B.ptp(t1 + 0.3, time.monotonic()) < 1e-4 and not B.active_goals() and len(B.goals_seen() - g0) == 6,
              f"stop {r.end().get('stop')} goals {len(B.goals_seen() - g0)}")
    finally:
        stub.stop()


@scenario
def tv_stale_camera():
    s = "tv_stale_camera"
    B.goto(START, 6)
    stub = tv_stub("good", ["--speed", "0.08", "--target", "0.6,1.0,1.5,0.3,1.3,0.3"])
    try:
        r = Roll(s, TVX + ["--max-steps", "400"])
        wait_for(lambda: len(r.recs()) >= 5, 40)
        n_before = len(r.recs())
        t_stop = time.monotonic()
        cams_unit(False)
        rc = r.wait(60)
        q = r.recs(); last = q[-1]
        check(s, "ran fine before the camera stopped", n_before >= 5 and all(x["verdict"] == "pass" for x in q[:-1]), f"{n_before} steps passed before the stop")
        check(s, "after the camera stopped: verdict stale, exit 4, names the stale image topic",
              rc == 4 and last["verdict"] == "stale" and any("image" in p and ("/camera/color" in p or "gripper" in p) for p in last["problems"]), f"rc={rc} {last['problems']}")
        check(s, "refused within ~2.5 s of the stop (age gate 0.5 s)", r.t1 - t_stop < 2.5, f"{r.t1 - t_stop:.2f} s from camera stop to process end")
        time.sleep(1.5)
        check(s, "arm holds still after the refusal, hold sequence ran", B.ptp(r.t1 + 0.5, time.monotonic()) < 1e-4 and (r.end().get("stop") or {}).get("held") is True, f"ptp {B.ptp(r.t1 + 0.5, time.monotonic()):.2e} stop {r.end().get('stop')}")
    finally:
        stub.stop()
        cams(True)


@scenario
def tv_stale_camera_observe():
    s = "tv_stale_camera_observe"
    stub = tv_stub("good")
    try:
        r = Roll(s, TV + ["--set", "observe.period_s=0.15", "--max-time", "120"])
        wait_for(lambda: len(r.recs()) >= 3, 30)
        cams_unit(False)
        got_stale = wait_for(lambda: any(x["verdict"] == "stale" for x in r.recs()), 10)
        cams(True)
        n = len(r.recs())
        got_back = wait_for(lambda: any(x["verdict"] == "pass" for x in r.recs()[n:]), 20)
        r.sig(signal.SIGINT)
        rc = r.wait(20)
        v = [x["verdict"] for x in r.recs()]
        check(s, "observe: pass -> stale -> pass again when the camera comes back (ensemble continues with consecutive steps)", got_stale and got_back and rc == 130,
              f"rc={rc} verdict sequence: {[k for i, k in enumerate(v) if i == 0 or k != v[i - 1]]}")
        steps = [x["step"] for x in stub.records()]
        check(s, "stale queries never reached the server: the steps it saw are consecutive", steps == list(range(len(steps))), f"{len(steps)} steps, last {steps[-1] if steps else None}")
    finally:
        stub.stop()
        cams(True)


def tv_interrupt(s, how):
    B.goto(START, 6)
    # 0.04 rad per row at a 0.05 rad/s cap -> every goal lasts ~0.8 s, so a goal is always running when the stop comes
    stub = tv_stub("fast", ["--fast-speed", "1.2", "--target", "0.65,1.05,1.5,0.35,1.3,0.35"])
    try:
        r = Roll(s, TVX + ["--max-steps", "200", "--set", "execution.max_joint_vel=0.05"], stdin=(how == "enter"))
        t_wait = time.monotonic()
        while time.monotonic() - t_wait < 30:
            if np.abs(B.q() - np.array(START)).max() > 0.01:
                break
            time.sleep(0.02)
        time.sleep(0.3)
        q_at_sig = B.q()
        g_active = len(B.active_goals())
        t_sig = time.monotonic()
        if how == "sigint":
            r.sig(signal.SIGINT)
        elif how == "sigint2":
            r.sig(signal.SIGINT); time.sleep(0.15); r.sig(signal.SIGINT); time.sleep(0.15); r.sig(signal.SIGINT)
        elif how == "enter":
            r.proc.stdin.write(b"\n"); r.proc.stdin.flush()
        rc = r.wait(30)
        t_end = time.monotonic()
        time.sleep(1.5)
        moved_after = float(np.abs(B.q() - q_at_sig).max())
        check(s, "a step goal was running when the stop came", g_active >= 1, f"{g_active} active goal(s), arm {np.abs(q_at_sig - np.array(START)).max():.3f} rad from start")
        check(s, "process ended with exit 130 within 6 s", rc == 130 and t_end - t_sig < 6, f"rc={rc}, {t_end - t_sig:.2f} s after the stop request; reason={r.end().get('reason')!r}")
        check(s, "cancel confirmed and hold goal finished", (r.end().get("stop") or {}).get("cancelled") is True and (r.end().get("stop") or {}).get("held") is True, str(r.end().get("stop")))
        check(s, "arm stopped almost immediately (<= 0.03 rad after the request; the goal was 0.04 rad)", moved_after < 0.03, f"moved {moved_after:.4f} rad after the stop request")
        check(s, "arm stationary after the exit, no goal left active", B.ptp(t_end + 0.2, time.monotonic()) < 1e-4 and len(B.active_goals()) == 0, f"ptp {B.ptp(t_end + 0.2, time.monotonic()):.2e} active {B.active_goals()}")
    finally:
        stub.stop()


@scenario
def tv_stop_ctrl_c():
    tv_interrupt("tv_stop_ctrl_c", "sigint")


@scenario
def tv_stop_ctrl_c_repeated():
    tv_interrupt("tv_stop_ctrl_c_repeated", "sigint2")


@scenario
def tv_stop_enter_key():
    tv_interrupt("tv_stop_enter_key", "enter")


@scenario
def tv_modality_mismatch():
    s = "tv_modality_mismatch"
    B.goto(START, 6)
    # (a) a GR00T-protocol server behind the turbovla config
    stub = Stub("good")  # backend groot
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s + "_groot_execute", TVX + ["--max-steps", "16"])
        rc = r.wait(60)
        check(s, "GR00T server on the turbovla config, execute: exit 4 'server modality mismatch' (backend), no query, arm still, no chunk goal",
              rc == 4 and "modality mismatch" in r.end().get("reason", "") and "backend" in r.end().get("reason", "") and not r.recs() and B.ptp(t0, time.monotonic()) < 1e-4 and len(B.goals_seen() - g0) <= 1,
              f"rc={rc} {r.end().get('reason')}")
        r = Roll(s + "_groot_observe", TV + ["--max-queries", "1"])
        rc = r.wait(60)
        q = r.recs()
        check(s, "... observe: loud MODALITY MISMATCH warning, still runs; the GR00T stub rejects the turbovla observation (server_error), no crash", rc == 0 and "MODALITY MISMATCH" in r.console() and len(q) == 1 and q[0]["verdict"] == "server_error", f"rc={rc} {[x['verdict'] for x in q]}")
    finally:
        stub.stop()
    for label, extra, word in [("horizon 16", ["--tv-horizon", "16"], "horizon"), ("cameras without knife", ["--tv-cameras", "head,wrist_left"], "cameras"),
                               ("other language", ["--tv-language", "Close the box."], "language")]:
        stub = tv_stub("good", extra)
        try:
            r = Roll(s + "_" + word, TVX + ["--max-steps", "16"])
            rc = r.wait(60)
            check(s, f"turbovla server announcing {label}: execute refused (exit 4, names {word}), no query", rc == 4 and word in r.end().get("reason", "") and not r.recs(), f"rc={rc} {r.end().get('reason')}")
        finally:
            stub.stop()
    stub = tv_stub("good")
    try:
        r = Roll(s + "_sha", TVX + ["--max-steps", "16", "--set", "server.expect_checkpoint_sha256=" + "ab" * 32])
        rc = r.wait(60)
        check(s, "expect_checkpoint_sha256 set and different -> execute refused", rc == 4 and "checkpoint_sha256" in r.end().get("reason", ""), f"rc={rc} {r.end().get('reason')}")
        # (e) the turbovla server behind the GR00T config
        r = Roll(s + "_on_groot_cfg", MOCKYES + ["--max-steps", "16"])
        rc = r.wait(60)
        check(s, "turbovla server on the GR00T config, execute: exit 4 'server modality mismatch' (backend), no query", rc == 4 and "modality mismatch" in r.end().get("reason", "") and "backend" in r.end().get("reason", "") and not r.recs(), f"rc={rc} {r.end().get('reason')}")
        r = Roll(s + "_ok", TVX + ["--max-steps", "8"])
        rc = r.wait(60)
        check(s, "matching server: run works, the modality line is logged with the checkpoint sha", rc == 0 and "server modality: backend turbovla" in r.console() and "checkpoint_sha256=stub" in r.console(), f"rc={rc}")
    finally:
        stub.stop()


@scenario
def tv_confirmation_prompt():
    s = "tv_confirmation_prompt"
    B.goto(START, 6)
    stub = tv_stub("good")
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s + "_no", TV + ["--execute", "--max-steps", "8"], stdin=True)
        time.sleep(4.0)
        r.proc.stdin.write(b"no\n"); r.proc.stdin.flush(); r.proc.stdin.close()
        rc = r.wait(40); con = r.console()
        check(s, "typed 'no' (no --mock-yes): exit 2, prompt + hardware + TurboVLA stop-and-go summary shown, no motion, no goal",
              rc == 2 and "type  yes" in con and "mock_components/GenericSystem" in con and "TurboVLA step loop (stop-and-go)" in con and B.ptp(t0, time.monotonic()) < 1e-4 and B.goals_seen() == g0,
              f"rc={rc} new_goals={len(B.goals_seen() - g0)}")
        r = Roll(s + "_yes", TV + ["--execute", "--max-steps", "5"], stdin=True)
        time.sleep(4.0)
        r.proc.stdin.write(b"yes\n"); r.proc.stdin.flush(); r.proc.stdin.close()
        rc = r.wait(60)
        check(s, "typed 'yes': exit 0, 5 steps executed", rc == 0 and len(r.recs()) == 5 and all(x["sent"] for x in r.recs()), f"rc={rc} n={len(r.recs())}")
        B.goto(START, 6)
        r = Roll(s + "_no_execute_flag", TV + ["--mock-yes", "--max-queries", "1"])
        rc = r.wait(30)
        check(s, "--mock-yes without --execute is still observe mode", rc == 0 and r.recs() and r.recs()[0]["mode"] == "observe" and r.recs()[0]["sent"] is False, f"rc={rc}")
    finally:
        stub.stop()
    code = r'''
import os, sys
sys.path.insert(0, %r)
import groot_ros_rollout as G
cfg = G.load_config(%r)
r = G.Rollout(cfg, G.parse_args(["--execute", "--mock-yes"]))
class FakeIO:
    def list_hardware(self, timeout=5.0): return [{"name": "rebel_arm_controller", "type": "system", "class_type": "Igus::Rebel", "state": "active"}]
    def robot_description(self, t=5.0, s=None): return None
r.io = FakeIO()
print("RESULT", r.confirm_execute())
''' % (str(TOOL), str(TVCFG))
    for text, exp in [("no\n", False), ("yes\n", True)]:
        p = subprocess.run([PYV, "-c", code], input=text.encode(), capture_output=True, timeout=30)
        out = p.stdout.decode()
        check(s, f"turbovla config, real hardware (fake Igus::Rebel) + --mock-yes + stdin {text!r} -> {exp} (--mock-yes ignored, prompt shown)", ("RESULT True" in out) == exp and "--mock-yes ignored" in out and "type  yes" in out, "")


# ====================================================================================================== TurboVLA backend, RIGHT arm
# Suite --suite right (dual-table mock, arm_right).  The stub emulates the selected right server (--model: right19-10k | mocap31-10k | real3nov-10k; arm_channels [7, 13], its language and
# sha); tvr_real_* talk to the REAL server of that model (--real-port) with fake images: plumbing, latency, gates only.
# --model selects the checkpoint under test (the server announces its sha256; the config pins it):
RIGHT_MODELS = {
    "right19": dict(cfg="rollout_config_turbovla_right.yaml", ckpt="right19-10k", profile="right19", rec=[1152, 648],
                    sha="02f163bef316fcc1894ec3b614a8d9cb7640a7d1771c48bc8f23607fa5598068",
                    norm="~/turbovla/right19/runs/right19-10k/normalization.json",
                    start=[-0.34907, 0.40684, 1.11841, 0.29322, -0.96185, 0.15254],   # first recorded state of the held-out episode real_0005 (in distribution)
                    start_desc="real_0005 first recorded state", lang="open the front flap of the cardboard box", diff=None),
    "mocap31": dict(cfg="rollout_config_turbovla_mocap31.yaml", ckpt="mocap31-10k", profile="mocap31", rec=[640, 480],
                    sha="82eab1f745064fbabbdea12f5a8d14f54a5e5195e5f03ac61e713b0e574b3ec6",
                    norm="~/turbovla/mocap31/runs/mocap31-10k/normalization.json",
                    start=[0.0, 0.0, 1.0472, 0.0, 0.87266, 0.0],   # first recorded state of EVERY mocap31 episode (home pose 0/0/60/0/50/0 deg; train + validation parquet)
                    start_desc="mocap31 episode start (home) pose", lang="open the front flap of the cardboard box",
                    diff=["model.recorder_size", "server.expect_checkpoint_sha256", "training_range.max", "training_range.min"]),
    "real3nov": dict(cfg="rollout_config_turbovla_real3nov.yaml", ckpt="real3nov-10k", profile="real3nov", rec=[1152, 648],
                     sha="53d1f024930295be142cba3aad14550b6f7e1568fead7653b42b24a8b86a83d1",
                     norm="~/turbovla/real3nov/real3nov/runs/real3nov-10k/normalization.json",
                     start=[-0.34907, 0.40684, 1.11841, 0.29322, -0.96185, 0.15254],   # first recorded state of the front-flap episodes of the 3 Nov set (= the 29 Oct start)
                     start_desc="real3nov front-flap episode start", lang="open the next closed flap of the cardboard box",
                     diff=["instruction", "model.language", "server.expect_checkpoint_sha256", "training_range.max", "training_range.min"]),
}
RM = RIGHT_MODELS[MODEL]
TVRCFG = TOOL / RM["cfg"]
TVR = ["--config", str(TVRCFG)]
TVRX = TVR + MOCKYES
RIGHT_LANG = RM["lang"]
RIGHT_SHA = RM["sha"]
RIGHT_REC = RM["rec"]
RIGHT_ARM = slice(7, 13)
REAL_START = RM["start"]


def tvr_stub(mode, extra=(), match=True):
    """stub emulating the selected right-arm server (--model); `match=False` keeps the stub's own default language/sha (mismatch tests)"""
    base = ["--tv-arm-channels", "7,13", "--target", ",".join(str(v) for v in TARGET)]
    if match:
        base += ["--tv-language", RIGHT_LANG, "--tv-sha", RIGHT_SHA]
    return Stub(mode, [*base, *extra], backend="turbovla")


@scenario
def tvr_unit_config():
    s = "tvr_unit_config"
    left, right = G.load_config(TOOL / "rollout_config_turbovla.yaml"), G.load_config(TVRCFG)
    m = right["model"]
    check(s, f"{MODEL} config ({RM['cfg']}): arm_channels [7, 13] (left config [0, 6]), recorder_size {RIGHT_REC}, 224 output, language, pinned sha256",
          m["arm_channels"] == [7, 13] and left["model"]["arm_channels"] == [0, 6] and m["recorder_size"] == RIGHT_REC and m["image_size"] == [224, 224]
          and m["language"] == RIGHT_LANG and right["server"]["expect_checkpoint_sha256"] == RIGHT_SHA, f"{m['arm_channels']} {m['recorder_size']} {right['server']['expect_checkpoint_sha256'][:12]}")
    check(s, "right config: joints arm_right_joint1..6, action /arm_right_trajectory_controller/follow_joint_trajectory, joint_states /joint_states",
          right["ros"]["joints"] == [f"arm_right_joint{i}" for i in range(1, 7)] and right["ros"]["action"] == "/arm_right_trajectory_controller/follow_joint_trajectory"
          and right["ros"]["joint_states_topic"] == "/joint_states", f"{right['ros']['action']}")
    check(s, "right config: cameras head /camera/color/image_raw/compressed 1920x1080 + wrist_left slot = /arm_right/gripper_camera/image_raw/compressed 1280x720",
          right["cameras"] == {"head": {"topic": "/camera/color/image_raw/compressed", "expected_size": [1920, 1080]},
                               "wrist_left": {"topic": "/arm_right/gripper_camera/image_raw/compressed", "expected_size": [1280, 720]}}, str(right["cameras"]))
    raised = ("first_step_max_rad", "max_target_step_rad")   # the only two gates raised for this model (operator, 2026-10-30)
    rg = {k: v for k, v in right["gates"].items() if k not in raised}; lg = {k: v for k, v in left["gates"].items() if k not in raised}
    check(s, f"gates identical to the left config except {raised}", rg == lg, "" if rg == lg else f"{rg} vs {lg}")
    for sect in ("execution", "freshness", "ensemble", "observe", "preprocess", "logging"):
        check(s, f"NO limit loosened: section {sect!r} identical to the left config", right[sect] == left[sect], "" if right[sect] == left[sect] else f"{right[sect]} vs {left[sect]}")
    check(s, "server section identical to the left config except the pinned sha", {**right["server"], "expect_checkpoint_sha256": None} == left["server"], "")
    check(s, "model section identical to the left config except recorder_size, language, arm_channels",
          {k: v for k, v in right["model"].items() if k not in ("recorder_size", "language", "arm_channels")} == {k: v for k, v in left["model"].items() if k not in ("recorder_size", "language", "arm_channels")}, "")
    check(s, "right gates: first_step_max_rad 0.15, max_target_step_rad 0.10 (raised), max_joint_vel 0.1, row_step_max_rad 0.1 (unchanged)",
          right["gates"]["first_step_max_rad"] == 0.15 and right["execution"]["max_joint_vel"] == 0.1 and right["gates"]["max_target_step_rad"] == 0.10 and right["gates"]["row_step_max_rad"] == 0.1, "")
    norm = Path(os.path.expanduser(RM["norm"]))
    if norm.exists():
        a = json.loads(norm.read_text())["action"]
        tr = right["training_range"]
        d = max(np.abs(np.array(tr["min"]) - np.array(a["min"][7:13])).max(), np.abs(np.array(tr["max"]) - np.array(a["max"][7:13])).max())
        check(s, "training_range == normalization.json action min/max channels 7..12 (to the 5 decimals written)", d < 1.0e-5, f"max diff {d:.2e}")
    else:
        check(s, f"{MODEL} normalization.json present for the training_range cross-check", True, f"{norm} not found: skipped")
    if MODEL != "right19":   # the mocap31 / real3nov configs are copies of the right19 config: ONLY the keys in RIGHT_MODELS[MODEL]['diff'] may differ
        base = G.load_config(TOOL / RIGHT_MODELS["right19"]["cfg"])
        def flat(d, pre=""):
            o = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    o.update(flat(v, pre + k + "."))
                else:
                    o[pre + k] = v
            return o
        fb, fm = flat(base), flat(right)
        diff = sorted(k for k in set(fb) | set(fm) if k != "_path" and fb.get(k) != fm.get(k))   # (_path = where load_config read the file from)
        allowed = RM["diff"]
        check(s, f"{MODEL} config differs from the right19 config ONLY in {allowed}", diff == allowed, f"differing keys {diff}")
    for label, val in [("missing (null)", "null"), ("[0, 5]", "[0, 5]"), ("[7, 12]", "[7, 12]"), ("[9, 15] beyond 14 channels", "[9, 15]"), ("[7.0, 13.0] floats", "[7.0, 13.0]"), ("scalar 7", "7")]:
        rc = G.main(["--config", str(TVRCFG), "--set", f"model.arm_channels={val}"])
        check(s, f"model.arm_channels {label} -> config error exit 2", rc == 2, f"rc={rc}")
    d = Path(tempfile.mkdtemp(prefix="tvr_cfg_"))
    txt = TVRCFG.read_text()
    (d / "no_key.yaml").write_text("\n".join(l for l in txt.splitlines() if "arm_channels: [7, 13]" not in l.split("#")[0]) + "\n")
    rc = G.main(["--config", str(d / "no_key.yaml")])
    check(s, "config WITHOUT model.arm_channels (required key) -> exit 2", rc == 2, f"rc={rc}")


@scenario
def tvr_unit_functions():
    s = "tvr_unit_functions"
    cfg = G.load_config(TVRCFG)
    arm = G.tv_arm_channels(cfg["model"])
    g, tr = cfg["gates"], cfg["training_range"]
    lim = np.array([[-3.1067, 3.1067], [-1.3788, 2.4260], [-1.3788, 2.4260], [-3.1067, 3.1067], [-1.6406, 1.6406], [-3.1067, 3.1067]])
    q = np.array(START)
    img = np.zeros((224, 224, 3), np.uint8)
    obs = G.build_observation_turbovla({"head": img, "wrist_left": img}, q, 4, arm)
    st = obs["state"]
    check(s, "observation: state (14,) float64 with the right joints in channels 7..12, ALL other channels (left arm, both grippers) exactly 0",
          st.shape == (14,) and st.dtype == np.float64 and np.allclose(st[7:13], q) and not st[:7].any() and st[13] == 0 and obs["step"] == 4, f"{st.tolist()}")
    obs_l = G.build_observation_turbovla({"head": img, "wrist_left": img}, q, 4, slice(0, 6))
    check(s, "same call with the left slice puts the joints in channels 0..5 (left behaviour unchanged)", np.allclose(obs_l["state"][:6], q) and not obs_l["state"][6:].any(), "")

    def mk(q0, per_row=0.002):
        c = np.full((30, 14), 0.7)
        c[:, 7:13] = q0 + per_row * np.arange(1, 31)[:, None]
        return c
    good = mk(q)
    ev = lambda ch, qq=q, **kw: G.evaluate_gates_turbovla(ch, qq, lim, {**g, **kw}, tr["min"], tr["max"], arm)
    bad = lambda r: sorted(x["name"] for x in r if not x["ok"])
    check(s, "good right chunk (channels outside 7..12 = 0.7, ignored) passes every raw gate", bad(ev(good)) == [], str(bad(ev(good))))
    c = good.copy(); c[:, :7] = 1e3; c[:, 13] = -1e3
    check(s, "huge FINITE values in the left-arm / gripper channels are ignored", bad(ev(c)) == [], str(bad(ev(c))))
    c = good.copy(); c[3, 2] = np.nan
    check(s, "NaN in an ignored channel (left joint3) still -> finite (the whole 30x14 chunk must be finite)", bad(ev(c)) == ["finite"], str(bad(ev(c))))
    c = good.copy(); c[:, 7] += 0.14
    check(s, "channel 7 (right joint1) 0.14 off the state passes first_step (raised right limit 0.15)", "first_step" not in bad(ev(c)), str(bad(ev(c))))
    c = good.copy(); c[:, 7] += 0.16
    check(s, "channel 7 (right joint1) 0.16 off the state -> first_step (gate limit 0.15)", "first_step" in bad(ev(c)), str(bad(ev(c))))
    c = good.copy(); c[:, 0] += 0.5
    check(s, "the SAME offset on channel 0 (left joint1) is not judged", bad(ev(c)) == [], str(bad(ev(c))))
    c = good.copy(); c[12:, 8] += 0.2
    check(s, "0.2 jump at row 12 on channel 8 -> row_step", "row_step" in bad(ev(c)), str(bad(ev(c))))
    c = good.copy(); c[:, 11] = np.linspace(-0.6, -1.9, 30) + 0.0  # joint5 below its training range (-1.637 - 0.1)
    qq = np.array(START); qq[4] = -0.6
    check(s, f"right joint5 down to -1.9 (< {tr['min'][4]:.3f}-0.1) -> training_range", "training_range" in bad(ev(c, qq, first_step_max_rad=9.0, row_step_max_rad=9.0)), str(bad(ev(c, qq, first_step_max_rad=9.0, row_step_max_rad=9.0))))
    check(s, "state at all zeros (out of the training range of the selected model) -> state_in_training_range", "state_in_training_range" in bad(ev(mk(np.zeros(6)), np.zeros(6))), "")
    check(s, f"{RM['start_desc']} is inside the training range +- margin", bad(ev(mk(np.array(REAL_START)), np.array(REAL_START))) == [], "")
    cs = G.chunk_summary(good, q, arm)
    check(s, "chunk_summary: row0 = right joints, ignored_channels_absmax = 0.7 (other 8 channels)", np.allclose(cs["row0"], good[0, 7:13]) and abs(cs["ignored_channels_absmax"] - 0.7) < 1e-12, f"{cs['ignored_channels_absmax']}")
    ok_reply = {"backend": "turbovla", "cameras": ["head", "knife", "wrist_left"], "client_cameras": ["head", "wrist_left"], "image_size": [224, 224],
                "state_dim": 14, "action_dim": 14, "horizon": 30, "control_rate_hz": 30, "checkpoint_sha256": RIGHT_SHA, "language": RIGHT_LANG, "arm_channels": [7, 13]}
    cmp = lambda rep, c=cfg: G.compare_modality_turbovla(rep, c)[0]
    check(s, "right server reply (arm_channels [7, 13], language, sha) vs right config: no problems", cmp(ok_reply) == [], str(cmp(ok_reply)))
    check(s, "arm_channels [0, 6] announced to the right config -> mismatch naming arm_channels", any("arm_channels" in p for p in cmp({**ok_reply, "arm_channels": [0, 6]})), str(cmp({**ok_reply, "arm_channels": [0, 6]})))
    r_noann = {k: v for k, v in ok_reply.items() if k != "arm_channels"}
    check(s, "reply WITHOUT arm_channels (pre-right-arm server) vs right config -> mismatch (treated as [0, 6])", any("arm_channels" in p and "not announced" in p for p in cmp(r_noann)), str(cmp(r_noann)))
    lcfg = G.load_config(TOOL / "rollout_config_turbovla.yaml")
    lreply = {**ok_reply, "arm_channels": [0, 6], "language": "Open the box flaps.", "checkpoint_sha256": "ab" * 32}
    check(s, "reply WITHOUT arm_channels vs LEFT config -> no arm problem (backward compatible)", not any("arm_channels" in p for p in cmp({k: v for k, v in lreply.items() if k != "arm_channels"}, lcfg)), "")
    check(s, "reply [0, 6] vs LEFT config -> no problem at all", cmp(lreply, lcfg) == [], str(cmp(lreply, lcfg)))
    check(s, "right server reply [7, 13] vs LEFT config -> mismatch", any("arm_channels" in p for p in cmp({**lreply, "arm_channels": [7, 13]}, lcfg)), "")
    check(s, "non-list arm_channels in the reply -> mismatch", any("arm_channels" in p for p in cmp({**ok_reply, "arm_channels": "right"})), "")
    check(s, "wrong language / wrong sha for the right config -> two problems", len(cmp({**ok_reply, "language": "Open the box flaps.", "checkpoint_sha256": "ab" * 32})) == 2, "")
    check(s, "tv_arm_channels rejects malformed values", all(_raises(lambda v=v: G.tv_arm_channels({"arm_channels": v, "state_dim": 14})) for v in (None, [0, 5], [7, 12], [9, 15], [-1, 5], "x", [7.0, 13.0], [True, 7])), "")


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


@scenario
def tvr_unit_preprocess():
    s = "tvr_unit_preprocess"
    import cv2
    from PIL import Image as PILImage
    cfg = G.load_config(TVRCFG)
    m, pp = cfg["model"], cfg["preprocess"]
    RW, RH = RIGHT_REC
    rng = np.random.default_rng(3)

    def make(w, h):
        yy, xx = np.mgrid[0:h, 0:w]
        tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32), (0, 0), 4) * 60
        img = np.clip(np.stack([xx * 200 // w, yy * 200 // h, 100 + 0 * xx], -1).astype(np.float32) + 20 + tex, 0, 255).astype(np.uint8)
        cv2.circle(img, (w * 9 // 20, h * 10 // 21), h // 5, (20, 200, 240), -1)
        cv2.putText(img, "FLAP 123", (w // 6, h * 3 // 4), cv2.FONT_HERSHEY_SIMPLEX, h / 180, (255, 255, 255), max(2, h // 90))
        return img

    def reference(jpg):
        """independent chain: PIL open, draft to the recorder size, convert, BILINEAR to the recorder size if needed, then the default-resample 224"""
        img = PILImage.open(io.BytesIO(bytes(jpg)))
        img.draft("RGB", (RW, RH))
        drafted = img.size
        rgb = np.asarray(img.convert("RGB"))
        if rgb.shape[:2] != (RH, RW):
            rgb = np.asarray(PILImage.fromarray(rgb).resize((RW, RH), PILImage.BILINEAR))
        return drafted, rgb, np.asarray(PILImage.fromarray(rgb).resize((224, 224)))

    for (w, h) in ((1920, 1080), (1280, 720)):
        jpg = cv2.imencode(".jpg", make(w, h), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        mine, full = G.turbovla_rgb(np.frombuffer(jpg, np.uint8), m, pp)
        d0, _, ref = reference(jpg)
        if (RW, RH) == (1152, 648):
            check(s, f"{w}x{h}: JPEG draft to 1152x648 leaves the full size ({d0[0]}x{d0[1]}): it never decodes below the target", tuple(d0) == (w, h), f"{d0}")
        else:
            check(s, f"{w}x{h}: JPEG draft to {RW}x{RH} ({d0[0]}x{d0[1]}) never decodes below the target", d0[0] >= RW and d0[1] >= RH, f"{d0}")
        check(s, f"{w}x{h}: output (224,224,3) uint8, reported source size {w}x{h}", mine.shape == (224, 224, 3) and mine.dtype == np.uint8 and list(full) == [w, h], f"{mine.shape} {full}")
        check(s, f"{w}x{h}: identical to the independent chain (BILINEAR to {RW}x{RH}, then default resample to 224)", np.array_equal(mine, ref), f"max abs diff {np.abs(mine.astype(int) - ref.astype(int)).max()}")
        rgb1152, _ = G.recorder_rgb(np.frombuffer(jpg, np.uint8), m["recorder_size"])
        check(s, f"{w}x{h}: recorder_rgb with recorder_size [{RW}, {RH}] returns ({RH}, {RW}, 3)", rgb1152.shape == (RH, RW, 3), f"{rgb1152.shape}")
        left, _ = G.turbovla_rgb(np.frombuffer(jpg, np.uint8), G.load_config(TOOL / "rollout_config_turbovla.yaml")["model"], pp)
        if (RW, RH) == (640, 480):
            check(s, f"{w}x{h}: identical to the left-arm 640x480 path (same recorder size)", np.array_equal(left, mine), "")
        else:
            check(s, f"{w}x{h}: differs from the left-arm 640x480 path (informational: mean abs diff {np.abs(left.astype(int) - mine.astype(int)).mean():.2f})", not np.array_equal(left, mine), "")
        if w == 1920 and (RW, RH) == (1152, 648):  # aspect kept: the orange disc stays a circle in the 1152x648 intermediate (it was an ellipse 0.75 in the 640x480 path)
            ms = (rgb1152[:, :, 0] > 200) & (rgb1152[:, :, 1] > 150) & (rgb1152[:, :, 2] < 80)
            ys, xs = np.where(ms)
            ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) if len(xs) else float("nan")
            check(s, "16:9 aspect preserved in the 1152x648 intermediate (disc bbox width/height 1.0)", abs(ratio - 1.0) < 0.05, f"{ratio:.3f}")


def tvr_roll(name, argv, **kw):
    return Roll(name, TVR + argv, **kw)


@scenario
def tvr_check_mode():
    s = "tvr_check_mode"
    r = tvr_roll(s, ["--check"])
    rc = r.wait(40)
    con = r.console()
    check(s, f"--check, {MODEL} config: exit 0, recorder {RIGHT_REC} then [224, 224], both cameras decoded at the live sizes, arm_channels [7, 13], right joints, mock hardware",
          rc == 0 and f"{RIGHT_REC} (recorder) then [224, 224]" in con and "camera head" in con and "size [1920, 1080]" in con and "size [1280, 720]" in con
          and "model.arm_channels [7, 13]" in con and "arm_right_joint1" in con and "/arm_right/gripper_camera/image_raw/compressed" in con and "hardware_is_mock=True" in con
          and "action server /arm_right_trajectory_controller/follow_joint_trajectory: available" in con, f"rc={rc}")


@scenario
def tvr_observe_stub():
    s = "tvr_observe_stub"
    B.goto(START, 6)
    stub = tvr_stub("good")
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = tvr_roll(s, ["--max-queries", "6", "--set", "observe.period_s=0.1", "--save-images", "--set", "logging.log_full_chunk=true"])
        rc = r.wait(60); t1 = time.monotonic(); q = r.recs()
        con = r.console()
        check(s, "modality match: arm_channels=[7, 13], language and sha shown, no MODALITY MISMATCH", "arm_channels=[7, 13]" in con and RIGHT_LANG in con and RIGHT_SHA[:16] in con and "MODALITY MISMATCH" not in con, "")
        check(s, "exit 0, 6 queries, all PASS", rc == 0 and len(q) == 6 and all(x["verdict"] == "pass" for x in q), f"rc={rc} {[x['verdict'] for x in q]}")
        check(s, "arm not moved, nothing sent, no goal", B.ptp(t0, t1) < 1e-4 and B.goals_seen() == g0 and not any(x["sent"] for x in q), f"ptp {B.ptp(t0, t1):.2e}")
        recs = stub.records()
        check(s, "stub (exact validation: right joints in 7..12, every other state channel exactly 0) saw steps 0..5", [x["step"] for x in recs] == list(range(6)), str([x["step"] for x in recs]))
        st = np.array([x["state"] for x in recs])
        qs = np.array([x["state"] for x in q])
        check(s, "state the stub received = the measured right joints in channels 7..12 (== logged state), channels 0..6 and 13 exactly 0", np.allclose(st[:, 7:13], qs, atol=1e-9) and not st[:, :7].any() and not st[:, 13].any(), f"{st[0].round(3).tolist()}")
        check(s, "measured joints are the START pose of the right arm", np.allclose(qs[0], START, atol=0.02), f"{qs[0].round(3).tolist()}")
        chunks = [np.array(x["chunk"]) for x in q]
        ref = ref_ensemble(chunks)[:, 7:13]
        got = np.array([x["ensemble"]["target"] for x in q])
        check(s, "logged ensembled targets == independent reference ensemble of the logged raw chunks, taken from channels 7..12", np.allclose(got, ref, atol=1e-9, rtol=0), f"max abs diff {np.abs(got - ref).max():.2e}")
        import cv2
        imgs = sorted((r.dir / "images").glob("*.jpg"))
        check(s, "model input images saved: 2 per query, 224x224", len(imgs) == 12 and {cv2.imread(str(p)).shape for p in imgs} == {(224, 224, 3)}, f"{len(imgs)}")
        check(s, "records carry the live camera sizes (1920x1080 head, 1280x720 right gripper) and chunk_summary", q[0]["image_size_wh"] == {"head": [1920, 1080], "wrist_left": [1280, 720]} and abs(q[0]["chunk_summary"]["ignored_channels_absmax"] - 0.7) < 1e-6, "")
    finally:
        stub.stop()


@scenario
def tvr_execute_good():
    s = "tvr_execute_good"
    B.goto(START, 6)
    stub = tvr_stub("good", ["--speed", "0.08"])
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s, TVRX + ["--max-steps", "40", "--set", "logging.log_full_chunk=true"])
        rc = r.wait(120); t1 = time.monotonic(); q = r.recs()
        check(s, "exit 0 after 40 steps, every step passed and its goal succeeded", rc == 0 and len(q) == 40 and all(x["verdict"] == "pass" and x.get("exec", {}).get("error_code") == 0 for x in q), f"rc={rc} n={len(q)} {sorted(set(x['verdict'] for x in q))}")
        check(s, "console: confirm block names the right action server, joints and model channels [7, 13]", "/arm_right_trajectory_controller/follow_joint_trajectory" in r.console() and "model channels [7, 13]" in r.console() and "arm_right_joint1" in r.console(), "")
        check(s, "exactly one trajectory goal per step (40) on the RIGHT controller", len(B.goals_seen() - g0) == 40, f"new goals {len(B.goals_seen() - g0)}")
        pk = B.peak(t0, t1)
        check(s, "velocity cap: independent measurement from /joint_states <= 0.105 rad/s", pk <= 0.105, f"peak {pk:.4f} rad/s")
        qf = B.q()
        moved = qf - np.array(START)
        check(s, "right arm moved toward the stub target (joint 1-3 delta has the target's sign and is > 0.05 rad)", np.all(np.sign(moved[:3]) == np.sign(np.array(TARGET)[:3] - np.array(START)[:3])) and np.abs(moved[:3]).min() > 0.05, f"final {qf.round(4).tolist()}")
        steps = [x["step"] for x in stub.records()]
        check(s, "server saw consecutive steps 0..39", steps == list(range(40)), "")
        ref = ref_ensemble([np.array(x["chunk"]) for x in q])[:, 7:13]
        got = np.array([x["ensemble"]["target"] for x in q])
        check(s, "commanded targets == independent reference ensemble of the raw chunks (channels 7..12)", np.allclose(got, ref, atol=1e-9, rtol=0), f"max abs diff {np.abs(got - ref).max():.2e}")
        check(s, "every |target - state| <= 0.05 (gate) and command peak velocity <= 0.1 rad/s", max(x["ensemble"]["target_minus_state_max"] for x in q) <= 0.05 and max(x["exec"]["cmd_peak_vel"] for x in q) <= 0.1 + 1e-6, "")
        cyc = np.array([x["cycle_ms"] for x in q[1:]])
        check(s, "step-synchronous loop rate (stub, mock; informational)", True, f"cycle median {np.median(cyc):.0f} ms ({1000 / np.median(cyc):.1f} Hz), get_action round trip median {np.median([x['latency_ms'] for x in q]):.1f} ms")
    finally:
        stub.stop()


@scenario
def tvr_refuse_first_step():
    """chunk offset 0.20 rad from the state on right joint1: first_step (0.15) fires; nothing is sent"""
    s = "tvr_refuse_first_step"
    B.goto(START, 6)
    stub = tvr_stub("good", ["--first-step-offset", "0.20"])
    try:
        g0, t0 = B.goals_seen(), time.monotonic()
        r = Roll(s, TVRX + ["--max-steps", "40"])
        rc = r.wait(60); t1 = time.monotonic(); q = r.recs()
        fg = failed_gates(q[0]) if q else None
        meta = r.recs("meta")
        check(s, "config in effect: first_step_max_rad 0.15, max_target_step_rad 0.10", meta and meta[0]["config"]["gates"]["first_step_max_rad"] == 0.15
              and meta[0]["config"]["gates"]["max_target_step_rad"] == 0.10, "")
        check(s, "exit 3, refused at the first query, nothing sent", rc == 3 and len(q) == 1 and q[0]["verdict"] == "refuse" and q[0]["sent"] is False, f"rc={rc} verdicts={[x['verdict'] for x in q]}")
        check(s, "first_step failed (detail names the 0.20 offset)", fg is not None and "first_step" in fg and "0.20" in [g for g in q[0]["gates"] if g["name"] == "first_step"][0]["detail"], f"failed {fg}")
        check(s, "arm held still, cancel + hold ran, only the hold goal exists", B.ptp(t0, t1) < 1e-4 and (r.end().get("stop") or {}).get("held") is True and len(B.goals_seen() - g0) == 1, f"stop {r.end().get('stop')} goals {len(B.goals_seen() - g0)}")
    finally:
        stub.stop()
    # boundaries: 0.09 passes both; 0.12 passes first_step (0.15) but not target_step (0.10); 0.16 fails first_step (a chunk refused by a
    # raw gate is not ensembled, so target_step is not reported for it)
    for off, exp, want in ((0.09, "pass", set()), (0.12, "refuse", {"target_step"}), (0.16, "refuse", {"first_step"})):
        B.goto(START, 6)
        stub = tvr_stub("static", ["--first-step-offset", str(off)])
        try:
            r = Roll(f"{s}_{off}", TVR + ["--max-queries", "1"])
            rc = r.wait(40); q = r.recs()
            check(s, f"observe with offset {off} rad: first query {exp}, failed gates {sorted(want)}", rc == 0 and q and q[0]["verdict"] == exp and set(failed_gates(q[0]) or []) == want,
                  f"rc={rc} {[x['verdict'] for x in q]} failed {failed_gates(q[0]) if q else None}")
        finally:
            stub.stop()


@scenario
def tvr_refuse_gates():
    """the other gate families also hold on the right-arm channels (jump, NaN, limit)"""
    s = "tvr_refuse_gates"
    # (the stub's "limit" mode ramps joint5 by +0.03 rad/row: from the right START it stays inside the right training range, so it is not a refusal here)
    for mode, exp in (("jump", ["row_step"]), ("nan", ["finite"]), ("relative", ["first_step"])):
        B.goto(START, 6)
        stub = tvr_stub(mode)
        try:
            g0, t0 = B.goals_seen(), time.monotonic()
            r = Roll(f"{s}_{mode}", TVRX + ["--max-steps", "40"])
            rc = r.wait(60); q = r.recs()
            fg = failed_gates(q[0]) if q else None
            check(s, f"{mode}: exit 3, refused at the first query ({exp} failed), nothing sent, arm still", rc == 3 and len(q) == 1 and q[0]["sent"] is False and all(e in (fg or []) for e in exp) and B.ptp(t0, time.monotonic()) < 1e-4, f"rc={rc} failed {fg}")
        finally:
            stub.stop()


@scenario
def tvr_arm_channels_mismatch():
    s = "tvr_arm_channels_mismatch"
    B.goto(START, 6)
    cases = [("server announces left [0, 6]", ["--tv-arm-channels", "0,6"], []),
             ("server announces nothing (pre-right-arm server)", ["--tv-arm-channels", "none"], [])]
    for ci, (label, extra, cargs) in enumerate(cases):
        stub = tvr_stub("good", extra)
        try:
            g0, t0 = B.goals_seen(), time.monotonic()
            r = Roll(f"{s}_exec{ci}", TVRX + ["--max-steps", "16", *cargs])
            rc = r.wait(60)
            check(s, f"{label}: execute refused (exit 4, reason names arm_channels), no query, arm still, no chunk goal",
                  rc == 4 and "arm_channels" in r.end().get("reason", "") and not r.recs() and B.ptp(t0, time.monotonic()) < 1e-4 and len(B.goals_seen() - g0) <= 1, f"rc={rc} {r.end().get('reason')}")
            r = Roll(f"{s}_obs{ci}", TVR + ["--max-queries", "1"])
            rc = r.wait(60); q = r.recs()
            check(s, f"{label}: observe warns (MODALITY MISMATCH ... arm_channels) and goes on; the stub rejects the right-arm state (server_error), nothing sent",
                  rc == 0 and "MODALITY MISMATCH" in r.console() and "arm_channels" in r.console() and len(q) == 1 and q[0]["verdict"] == "server_error" and q[0]["sent"] is False, f"rc={rc} {[x['verdict'] for x in q]}")
        finally:
            stub.stop()
    stub = tvr_stub("good")   # right server, client config says left channels
    try:
        r = Roll(s + "_cfg_left", TVRX + ["--max-steps", "16", "--set", "model.arm_channels=[0, 6]"])
        rc = r.wait(60)
        check(s, "right server, client model.arm_channels [0, 6]: execute refused (exit 4, arm_channels), no query", rc == 4 and "arm_channels" in r.end().get("reason", "") and not r.recs(), f"rc={rc} {r.end().get('reason')}")
        r = Roll(s + "_ok", TVRX + ["--max-steps", "6"])
        rc = r.wait(60)
        check(s, "matching server and config: run works", rc == 0 and len(r.recs()) == 6, f"rc={rc}")
    finally:
        stub.stop()
    for label, extra, word in [("the left language", ["--tv-language", "Open the box flaps."], "language")]:
        stub = tvr_stub("good", extra)
        try:
            r = Roll(s + "_" + word, TVRX + ["--max-steps", "16"])
            rc = r.wait(60)
            check(s, f"server announcing {label}: execute refused (exit 4, names {word}), no query", rc == 4 and word in r.end().get("reason", "") and not r.recs(), f"rc={rc} {r.end().get('reason')}")
        finally:
            stub.stop()
    stub = tvr_stub("good", ["--tv-sha", "ab" * 32])
    try:
        r = Roll(s + "_sha", TVRX + ["--max-steps", "16"])
        rc = r.wait(60)
        check(s, f"server announcing another checkpoint sha (config pins {RM['ckpt']}): execute refused (exit 4, checkpoint_sha256), no query", rc == 4 and "checkpoint_sha256" in r.end().get("reason", "") and not r.recs(), f"rc={rc} {r.end().get('reason')}")
    finally:
        stub.stop()


@scenario
def tvr_stale_camera():
    """the cameras the right config subscribes to (head + /arm_right/gripper_camera) going stale ends a running execute with exit 4"""
    s = "tvr_stale_camera"
    B.goto(START, 6)
    stub = tvr_stub("good", ["--speed", "0.08"])
    try:
        r = Roll(s, TVRX + ["--max-steps", "400"])
        wait_for(lambda: len(r.recs()) >= 5, 40)
        n_before = len(r.recs())
        t_stop = time.monotonic()
        cams_unit(False)
        rc = r.wait(60)
        q = r.recs(); last = q[-1]
        check(s, "ran fine before the cameras stopped (only the right-side topics are published in this suite)", n_before >= 5 and all(x["verdict"] == "pass" for x in q[:-1]), f"{n_before} steps passed before the stop")
        check(s, "after the stop: verdict stale, exit 4, names an image topic", rc == 4 and last["verdict"] == "stale" and any("image" in p and ("/camera/color" in p or "/arm_right/gripper_camera" in p) for p in last["problems"]), f"rc={rc} {last['problems']}")
        check(s, "refused within ~2.5 s of the stop, arm holds still, hold sequence ran", r.t1 - t_stop < 2.5 and (r.end().get("stop") or {}).get("held") is True, f"{r.t1 - t_stop:.2f} s; stop {r.end().get('stop')}")
    finally:
        stub.stop()
        cams(True)
        B.goto(START, 6)


# ---- real server of the selected --model (fake images): plumbing, latency, gates
def real_server_up():
    if not REAL_PORT:
        return False
    try:
        c = G.Gr00tClient("127.0.0.1", int(REAL_PORT), 3.0, label="TurboVLA")
        r = c.call("get_modality_config", timeout_s=3.0)
        c.close()
        return isinstance(r, dict) and r.get("arm_channels") == [7, 13] and r.get("checkpoint_sha256") == RIGHT_SHA   # the server of the selected --model
    except Exception:  # noqa: BLE001
        return False


def tvr_real(name, argv, **kw):
    return Roll(name, TVR + ["--port", REAL_PORT, *argv], **kw)


@scenario
def tvr_real_modality():
    s = "tvr_real_modality"
    if not real_server_up():
        check(s, f"real {MODEL} server on --real-port (arm_channels [7, 13], sha256 {RIGHT_SHA[:12]})", False, f"not reachable / not the {MODEL} right-arm server: start it with start_turbovla_server.sh start --profile {RM['profile']} and pass --real-port")
        return
    B.goto(REAL_START, 6)
    r = tvr_real(s, ["--max-queries", "1"])
    rc = r.wait(90); con = r.console()
    check(s, "real server vs right config: modality line shows arm_channels=[7, 13], the training language and the pinned sha; no MODALITY MISMATCH; exit 0",
          rc == 0 and "arm_channels=[7, 13]" in con and RIGHT_LANG in con and RIGHT_SHA[:16] in con and "MODALITY MISMATCH" not in con, f"rc={rc}")
    r = tvr_real(s + "_wrong_cfg", ["--max-queries", "1", "--set", "model.arm_channels=[0, 6]"])
    rc = r.wait(90)
    check(s, "real server, client arm_channels [0, 6] (observe): warns MODALITY MISMATCH arm_channels", "MODALITY MISMATCH" in r.console() and "arm_channels" in r.console(), f"rc={rc}")
    r = tvr_real(s + "_exec_wrong", TVRX + ["--max-steps", "4", "--set", "model.arm_channels=[0, 6]"])
    rc = r.wait(90)
    check(s, "real server, client arm_channels [0, 6] (execute): refused exit 4, no query", rc == 4 and "arm_channels" in r.end().get("reason", "") and not r.recs(), f"rc={rc} {r.end().get('reason')}")
    r = tvr_real(s + "_left_lang", TVRX + ["--max-steps", "4", "--set", 'model.language="Open the box flaps."'])
    rc = r.wait(90)
    check(s, "real server, client language = left language (execute): refused exit 4 (language)", rc == 4 and "language" in r.end().get("reason", "") and not r.recs(), f"rc={rc} {r.end().get('reason')}")


@scenario
def tvr_real_observe():
    s = "tvr_real_observe"
    if not real_server_up():
        check(s, f"real {MODEL} server reachable", False, "skipped: no --real-port server")
        return
    B.goto(REAL_START, 6)
    g0, t0 = B.goals_seen(), time.monotonic()
    r = tvr_real(s, ["--max-queries", "30", "--set", "observe.period_s=0.05"])
    rc = r.wait(180); t1 = time.monotonic(); q = r.recs()
    verd = {v: sum(1 for x in q if x["verdict"] == v) for v in sorted({x["verdict"] for x in q})}
    check(s, "exit 0, 30 queries answered, no server_error / stale / input_error (verdict is pass or refuse only)", rc == 0 and len(q) == 30 and set(verd) <= {"pass", "refuse"}, f"rc={rc} verdicts {verd}")
    check(s, "arm not moved, nothing sent, no goal", B.ptp(t0, t1) < 1e-4 and B.goals_seen() == g0 and not any(x["sent"] for x in q), "")
    lat = np.array([x["latency_ms"] for x in q[1:]])
    srv = np.array([x["server_info"]["inference_ms"] for x in q[1:]])
    check(s, "latency (client round trip incl. 224x224 payload; first query excluded): median < 100 ms; server model time median < 60 ms", np.median(lat) < 100 and np.median(srv) < 60,
          f"round trip median {np.median(lat):.1f} p95 {np.percentile(lat, 95):.1f} ms; server median {np.median(srv):.1f} p95 {np.percentile(srv, 95):.1f} ms; first query {q[0]['latency_ms']:.0f} ms")
    cyc = np.array([x["cycle_ms"] for x in q[1:]])
    check(s, f"preprocess (decode 1920x1080 + 1280x720, BILINEAR {RIGHT_REC[0]}x{RIGHT_REC[1]}, 224) + query cycle (informational)", True, f"cycle median {np.median(cyc):.0f} ms, p95 {np.percentile(cyc, 95):.0f} ms")
    names = collections.Counter(n for x in q for n in failed_gates(x))
    check(s, "every record has the 4+ raw gates evaluated and an ensemble target (never short-circuited); failed gate names (informational)", all(len(x["gates"]) >= 6 for x in q), f"failed gates over 30 fake-image queries: {dict(names)}")
    ch = np.array([x["chunk_summary"]["row0"] for x in q])
    check(s, "the model answered with finite right-arm joint values near the state (row0 within 0.5 rad of the state on every query; fake images)", np.isfinite(ch).all() and max(x["chunk_summary"]["row0_minus_state_max"] for x in q) < 0.5, f"max |row0-state| {max(x['chunk_summary']['row0_minus_state_max'] for x in q):.3f}")


@scenario
def tvr_real_execute():
    s = "tvr_real_execute"
    if not real_server_up():
        check(s, f"real {MODEL} server reachable", False, "skipped: no --real-port server")
        return
    # (a) strict gates (the real config): either every executed step passed every gate, or the run was refused before any motion
    B.goto(REAL_START, 6)
    g0, t0 = B.goals_seen(), time.monotonic()
    r = Roll(s + "_strict", TVRX + ["--port", REAL_PORT, "--max-steps", "12"])
    rc = r.wait(240); t1 = time.monotonic(); q = r.recs()
    sent = [x for x in q if x.get("sent")]
    ok_sent = all(x["verdict"] == "pass" and all(g["ok"] for g in x["gates"]) and x["exec"]["error_code"] == 0 for x in sent)
    refused = [x for x in q if x["verdict"] != "pass"]
    check(s, "strict gates: exit 0 (12 steps) or 3 (refused); every step that was sent had ALL gates ok; a refused step was not sent and ended the run with the hold",
          rc in (0, 3) and ok_sent and all(x["sent"] is False for x in refused) and (rc == 0 or ((r.end().get("stop") or {}).get("held") is True and len(refused) == 1)),
          f"rc={rc} queries={len(q)} sent={len(sent)} refused={len(refused)} failed gates {[failed_gates(x) for x in refused]}")
    pk = B.peak(t0, t1)
    check(s, "velocity cap holds: measured peak <= 0.105 rad/s", pk <= 0.105, f"peak {pk:.4f} rad/s")
    check(s, "goals: one per sent step (+ one hold goal if refused)", len(B.goals_seen() - g0) == len(sent) + (0 if rc == 0 else 1), f"new goals {len(B.goals_seen() - g0)}")
    if rc == 3:
        print(f"  [info] strict run refused after {len(sent)} executed step(s): {failed_gates(refused[0]) if refused else None}")
    # (b) PLUMBING ONLY (mock, fake images): first_step / target_step tightened OFF to 0.3 rad (config files untouched; --set on this one run) so the
    #     execute loop runs several steps with the real server; everything else (row_step, urdf, training range, speed 0.1 rad/s) stays.
    B.goto(REAL_START, 6)
    g0, t0 = B.goals_seen(), time.monotonic()
    r = Roll(s + "_plumbing", TVRX + ["--port", REAL_PORT, "--max-steps", "10", "--set", "gates.first_step_max_rad=0.3", "--set", "gates.max_target_step_rad=0.3"])
    rc = r.wait(300); t1 = time.monotonic(); q = r.recs()
    sent = [x for x in q if x.get("sent")]
    check(s, "PLUMBING run (first_step/target_step 0.3 on mock, fake images): exit 0 (10 steps) or 3 (another gate refused); sent steps all passed every gate; goals succeeded",
          rc in (0, 3) and all(x["verdict"] == "pass" and x["exec"]["error_code"] == 0 for x in sent) and all(x["sent"] is False for x in q if x["verdict"] != "pass"),
          f"rc={rc} queries={len(q)} sent={len(sent)} verdicts {collections.Counter(x['verdict'] for x in q)} failed {[failed_gates(x) for x in q if x['verdict'] != 'pass']}")
    pk = B.peak(t0, t1)
    check(s, "plumbing run: velocity cap holds, measured peak <= 0.105 rad/s", pk <= 0.105, f"peak {pk:.4f} rad/s")
    if sent:
        lat = np.array([x["latency_ms"] for x in q]); cyc = np.array([x["cycle_ms"] for x in sent])
        check(s, "plumbing run: ensemble ran on the real chunks (history 1..n), per-step latency (informational)", [x["ensemble"]["history_size"] for x in sent] == list(range(1, len(sent) + 1)),
              f"{len(sent)} steps executed; get_action round trip median {np.median(lat):.1f} ms; cycle median {np.median(cyc):.0f} ms (includes the 0.1 rad/s stretched goal)")
    B.goto(START, 6)


def main():
    global B
    rclpy.init()
    B = Bench()
    time.sleep(1.0)
    # hard safety check: this bench only ever moves mock hardware
    from groot_ros_rollout import hardware_is_mock
    cli = B.create_client(__import__("controller_manager_msgs.srv", fromlist=["ListHardwareComponents"]).ListHardwareComponents, "/controller_manager/list_hardware_components")
    assert cli.wait_for_service(10), "controller_manager not up"
    fut = cli.call_async(__import__("controller_manager_msgs.srv", fromlist=["ListHardwareComponents"]).ListHardwareComponents.Request())
    while not fut.done(): time.sleep(0.05)
    comps = [{"class_type": c.class_type} for c in fut.result().component]
    print("hardware:", [c["class_type"] for c in comps])
    assert hardware_is_mock(comps), "NOT mock hardware: refusing to run any scenario"
    names = [n for n in SCN if n.startswith("tvr_") == (SUITE == "right") and (not ONLY or n in ONLY or any(n.startswith(o) for o in ONLY))]
    for n in names:
        print(f"\n=== {n}", flush=True)
        try:
            SCN[n]()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            check(n, "scenario completed without exception", False, traceback.format_exc().strip().splitlines()[-1])
    B.goto(START, 6)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))
    print("\n================ SUMMARY ================")
    byscn = {}
    for r in RESULTS:
        byscn.setdefault(r["scenario"], []).append(r["ok"])
    for k, v in byscn.items():
        print(f"{'PASS' if all(v) else 'FAIL'}  {k}  ({sum(v)}/{len(v)} checks)")
    bad = [r for r in RESULTS if not r["ok"]]
    print(f"TOTAL {len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed; results: {OUT}/results.json")
    rclpy.shutdown()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF

# ------------------------------------------------------------------------------------------------ bring up the mock
# the domain must be empty (other sessions share mock domains)
NODES=$(timeout 30 ros2 node list --no-daemon 2>/dev/null)
if [ -n "$NODES" ]; then
  echo "ROS domain $DOMAIN is not empty (nodes: $(echo $NODES | tr '\n' ' ')): pick another with --domain"; exit 3
fi
for u in "$U_STACK" "$U_CAMS"; do
  if systemctl --user is-active --quiet "$u"; then echo "unit $u already active: stop it or pick another domain"; exit 3; fi
done
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "TCP port $PORT is busy: pick another with --port"; exit 3; fi

echo "[mock] starting $SUITE-arm mock stack ($LAUNCH) on domain $DOMAIN (unit $U_STACK) ..."
systemd-run --user --unit="$U_STACK" --collect "$WORK/stack.sh" >/dev/null && STARTED_UNITS+=("$U_STACK")
systemd-run --user --unit="$U_CAMS" --collect "$WORK/cams.sh" >/dev/null && STARTED_UNITS+=("$U_CAMS")
for i in $(seq 1 90); do
  if [ "$SUITE" = right ]; then CTRL=arm_right_trajectory_controller; else CTRL=rebel_arm_trajectory_controller; fi
  CTRLS=$(timeout 10 ros2 control list_controllers 2>/dev/null)   # (timeout: the verb waits forever for a missing controller_manager)
  if echo "$CTRLS" | grep -q "$CTRL.*active" && echo "$CTRLS" | grep -q "joint_state_broadcaster.*active"; then break; fi
  sleep 1
  [ "$i" = 90 ] && { echo "mock controllers did not become active"; exit 4; }
done
timeout 20 ros2 control list_hardware_components 2>/dev/null | grep -q "mock_components/GenericSystem" || { echo "hardware is not mock_components/GenericSystem: abort"; exit 4; }
for i in $(seq 1 30); do
  timeout 5 ros2 topic echo --once /camera/color/image_raw/compressed --no-arr >/dev/null 2>&1 && break
  [ "$i" = 30 ] && { echo "fake cameras not publishing"; exit 4; }
done
echo "[mock] stack up (mock_components/GenericSystem confirmed); running scenarios, output in $OUT"

G_TOOLDIR="$HERE" G_OUT="$OUT" G_PORT="$PORT" G_PYV="$PYV" G_ONLY="$ONLY" G_UNIT_CAMS="$U_CAMS" G_CAMS_SH="$WORK/cams.sh" G_SUITE="$SUITE" G_MODEL="$MODEL" G_REAL_PORT="$REAL_PORT" \
  nice -n 19 "$PYV" "$WORK/driver.py" 2>&1 | tee "$OUT/driver.log"
RC=${PIPESTATUS[0]}
echo "[mock] driver exit code $RC; tearing down"
exit "$RC"
