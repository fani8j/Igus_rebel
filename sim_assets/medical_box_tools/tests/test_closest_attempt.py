"""Headless Isaac Sim 5.1: "closest attempt" videos for the UNREACHABLE targets (T1, T2), front mode.  The arm moves from home along the collision-free joint path computed by
tools/closest_attempt.py to the best-effort pose (closest to the press / grasp pose under joint limits and collision), holds 3 s, a thin translucent red box (visual only, no collision)
marks the gap between the pad TCP and the target point.  Suction arm first (returns home), then the gripper arm.  No vacuum, no grasp.
Run: ROS domain is taken from CLOSEST_DOMAIN (default 174).  nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_closest_attempt.py --target T1 --record
Needs data/plans/closest_<T>.json (python3 tools/closest_attempt.py --targets T1).  Footer caption only (no header).
Watch live: add --gui (Isaac Sim window, real-time playback, waits for the window to be closed; result goes to unreachable_closest_run_<T>_gui.json)."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from isaac_common import *            # noqa: E402,F401
os.environ['ROS_DOMAIN_ID'] = os.environ.get('CLOSEST_DOMAIN', '174')
os.environ['ROS_LOCALHOST_ONLY'] = '1'
import numpy as np                    # noqa: E402
from test_cabinet_grasp import TARGETS, BOX, Cabinet, mat_from, set_matrix, quat_wxyz   # noqa: E402

DT = 1 / 240.0
PLANS = Path(__file__).resolve().parent.parent / 'data' / 'plans'
VIDEO_DEFAULT = Path('/media/farhan/My Passport/igus_nvidia/scratch_usd/videos/robot_20261015')
U = '/World/D200/UpperAssembly/dual_arm_rig'
ARM_J = {s: [f'arm_{s}_joint{k}' for k in range(1, 7)] for s in ('left', 'right')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--video-dir', default=str(VIDEO_DEFAULT))
    ap.add_argument('--hold', type=float, default=3.0)
    ap.add_argument('--gui', action='store_true', help='show the run live in the Isaac Sim window (physics unchanged; not with --record)')
    ap.add_argument('--gui-speed', type=float, default=1.0, help='--gui playback speed vs real time (it cannot run faster than the physics)')
    ap.add_argument('--gui-close', action='store_true', help='--gui: exit at the end instead of waiting for the window to be closed')
    args = ap.parse_args()
    if args.gui and args.record:
        ap.error('--gui and --record cannot be combined')
    app = start_app(headless=not args.gui, extra_exts=('isaacsim.robot.surface_gripper', 'omni.kit.viewport.utility', 'omni.replicator.core'))
    print('### STAGE_OPEN', os.environ['ROS_DOMAIN_ID'])
    apply_d200_physics_settings()
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Usd, Vt
    from isaacsim.core.prims import Articulation
    stage, sim = open_scene(app, 'cabinet_robot_test_scene.usda', DT)
    gui = GuiViewer(app, sim, args.gui_speed, dt=DT) if args.gui else None
    cab = Cabinet(stage)
    tn = args.target
    plan = json.load(open(PLANS / f'closest_{tn}.json'))
    rf = RESULTS / f'unreachable_closest_{tn}.json'                       # written by tools/closest_attempt.py --targets <T>
    res = json.load(open(rf if rf.exists() else RESULTS / 'unreachable_closest.json'))[tn]     # merged T1 + T2 file
    shelf, box, which = TARGETS[tn]
    name, tpath, Mc = cab.pick(shelf, box, which)
    c = Mc[:3, 3]
    wf = stage.GetPrimAtPath('/World/D200/joints/world_fix')
    wf_pos0 = np.array(wf.GetAttribute('physics:localPos0').Get(), float)
    contacts = {}

    def on_contact(headers, data):
        for h in headers:
            a0, a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            contacts[(a0, a1)] = contacts.get((a0, a1), 0) + 1
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/D200')):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.0)
    from omni.physx import get_physx_simulation_interface
    csub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    # ---- gap marker: visual only (no CollisionAPI / RigidBodyAPI), translucent red
    mk = UsdGeom.Cube.Define(stage, '/World/GapMarker')
    mk.CreateSizeAttr(1.0)
    mk.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.05, 0.05)])
    mk.CreateDisplayOpacityAttr([0.55])
    mk.CreateVisibilityAttr('invisible')
    ball = UsdGeom.Sphere.Define(stage, '/World/GapTarget')
    ball.CreateRadiusAttr(0.012)
    ball.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.05, 0.05)])
    ball.CreateDisplayOpacityAttr([0.7])
    ball.CreateVisibilityAttr('invisible')

    def show_marker(p_tcp, p_tgt):
        p_tcp, p_tgt = np.array(p_tcp), np.array(p_tgt)
        v = p_tgt - p_tcp
        L = float(np.linalg.norm(v))
        z = v / max(L, 1e-9)
        a = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(z, a)
        Rm = np.column_stack([a, b, z])
        M = mat_from(Rm, (p_tcp + p_tgt) / 2)
        ops = UsdGeom.Xformable(mk.GetPrim())
        ops.ClearXformOpOrder()
        ops.AddTransformOp().Set(M)
        ops.AddScaleOp().Set(Gf.Vec3f(0.010, 0.010, max(L, 0.002)))
        set_matrix(ball.GetPrim(), mat_from(np.eye(3), p_tgt))
        mk.GetVisibilityAttr().Set('inherited')
        ball.GetVisibilityAttr().Set('inherited')

    def hide_marker():
        mk.GetVisibilityAttr().Set('invisible')
        ball.GetVisibilityAttr().Set('invisible')
    vdir = Path(args.video_dir)
    rec_tools = {}
    VIEWS = ('3q', 'side', 'head')
    if args.record:
        import subprocess
        import omni.replicator.core as rep
        from PIL import Image, ImageDraw, ImageFont
        for nm, fl in (('RecCam3Q', 20.0), ('RecCamSide', 28.0)):
            cm = UsdGeom.Camera.Define(stage, f'/World/{nm}')
            cm.CreateFocalLengthAttr(fl)
            cm.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        cam_paths = {'3q': '/World/RecCam3Q', 'side': '/World/RecCamSide', 'head': f'{U}/stand/gemini336L_camera_link/gemini336L_rgb'}
        rec_tools['ann'] = {}
        for v in VIEWS:
            rp = rep.create.render_product(cam_paths[v], (1280, 720))
            a = rep.AnnotatorRegistry.get_annotator('rgb')
            a.attach([rp])
            rec_tools['ann'][v] = a
        rec_tools['rep'] = rep
        rec_tools['font'] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
        vdir.mkdir(parents=True, exist_ok=True)
    # ---- place the robot at the parking pose (lift at its lowest limit, sliders home)
    Rcw, tcw = np.array(plan['cabinet_to_world']['R']), np.array(plan['cabinet_to_world']['t'])
    Rrc, trc = Rcw.T, -Rcw.T @ tcw
    set_matrix(stage.GetPrimAtPath('/World/D200'), mat_from(Rrc, trc))
    wf.GetAttribute('physics:localPos0').Set(Gf.Vec3f(*[float(x) for x in (Rrc @ wf_pos0 + trc)]))
    w, x, y, z = quat_wxyz(Rrc)
    wf.GetAttribute('physics:localRot0').Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    par = plan['parking']
    for jn, v in (('lift_joint', par['lift_m'] + 0.0001), ('slider_left_joint', par['slider_left']), ('slider_right_joint', par['slider_right'])):
        jp = stage.GetPrimAtPath(f'{U}/Joints/{jn}')
        jp.GetAttribute('state:linear:physics:position').Set(float(v))
        jp.GetAttribute('drive:linear:physics:targetPosition').Set(float(v))
    sim.initialize_physics()
    sim.play()
    art = Articulation('/World/D200/base_link')
    art.initialize()
    dofs = {n: i for i, n in enumerate(art.dof_names)}
    arm_idx = {s: np.array([dofs[n] for n in ARM_J[s]]) for s in ARM_J}
    il, ir = dofs['arm_right_finger_left_joint'], dofs['arm_right_finger_right_joint']
    tgt = np.array(art.get_joint_positions()[0], float)
    tgt[il] = tgt[ir] = 0.024
    step_i = [0]
    cap = ['']
    n_frames = [0]
    procs = {}

    def look_at_M(eye, tg_):
        f = tg_ - eye
        f = f / np.linalg.norm(f)
        sx_ = np.cross(f, [0, 0, 1.0])
        sx_ /= np.linalg.norm(sx_)
        u = np.cross(sx_, f)
        return mat_from(np.column_stack([sx_, u, -f]), eye)

    def rec_frame(extra=None):
        rec_tools['rep'].orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        imgs = {}
        for v, ann in rec_tools['ann'].items():
            img = Image.fromarray(np.ascontiguousarray(ann.get_data()[..., :3]))
            dr = ImageDraw.Draw(img, 'RGBA')
            dr.rectangle([0, 668, 1280, 720], fill=(0, 0, 0, 150))
            dr.text((12, 676), cap[0], font=rec_tools['font'], fill=(255, 220, 80, 255))
            imgs[v] = img
            procs[v].stdin.write(img.tobytes())
        tile = Image.new('RGB', (1920, 720))
        tile.paste(imgs['3q'], (0, 0))
        tile.paste(imgs['side'].resize((640, 360)), (1280, 0))
        tile.paste(imgs['head'].resize((640, 360)), (1280, 360))
        procs['tiled'].stdin.write(tile.tobytes())
        n_frames[0] += 1
    rec_on = [False]

    def step(n=1):
        for _ in range(n):
            art.set_joint_position_targets(tgt.reshape(1, -1))
            sim.step(render=False)
            step_i[0] += 1
            if rec_on[0] and step_i[0] % 8 == 0:
                rec_frame()
            if gui:
                gui.tick(step_i[0])

    def drive(arm, q0, qs, ts):
        pts = np.vstack([q0, np.asarray(qs)])
        tt = np.concatenate([[0.0], np.asarray(ts)])
        T = float(tt[-1])
        for k in range(1, int(np.ceil(T / DT)) + 1):
            tgt[arm_idx[arm]] = [np.interp(min(k * DT, T), tt, pts[:, j]) for j in range(6)]
            step(1)
        tgt[arm_idx[arm]] = pts[-1]

    def set_side_cam(tcp, tg_):
        if not args.record:                              # the video cameras only exist with --record
            return
        mid = (np.array(tcp) + np.array(tg_)) / 2
        eye = mid + np.array([-0.55, -0.45, 0.20])
        set_matrix(stage.GetPrimAtPath('/World/RecCamSide'), look_at_M(eye, mid))
    if args.record:
        set_matrix(stage.GetPrimAtPath('/World/RecCam3Q'), look_at_M(np.array([c[0] - 1.30, -2.00, c[2] + 0.60]), np.array([c[0] + 0.05, -0.40, c[2] + 0.22])))
    set_side_cam(plan['arms']['left']['tcp_cabinet'], plan['arms']['left']['target_cabinet'])
    if args.record:
        for _ in range(8):
            rec_tools['rep'].orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        for v, size in (('3q', '1280x720'), ('side', '1280x720'), ('head', '1280x720'), ('tiled', '1920x720')):
            procs[v] = subprocess.Popen(['/usr/bin/ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', size, '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                                         '-pix_fmt', 'yuv420p', str(vdir / f'front_{tn}_closest_{v}.mp4')], stdin=subprocess.PIPE)
        rec_on[0] = True
    if gui:
        gui.look(np.array([c[0] - 1.30, -2.00, c[2] + 0.60]), np.array([c[0] + 0.05, -0.40, c[2] + 0.22]))      # same as the 3/4 video view
    cap[0] = f'robot at the parking pose (d {par["d_m"] * 1000:.0f} mm, lift at its lowest limit {par["lift_m"]:.3f} m), arms at home'
    step(int(1.5 / DT))
    out = dict(target=tn, arms={})
    for arm, label, what in (('left', 'suction pad', 'press point'), ('right', 'gripper TCP', 'grasp point (handoff position)')):
        pa, ra = plan['arms'][arm], res['arms'][arm]
        e = ra['error']
        set_side_cam(pa['tcp_cabinet'], pa['target_cabinet'])
        cap[0] = f"{'suction' if arm == 'left' else 'gripper'} arm moves toward the carton (collision-free path)"
        drive(arm, np.zeros(6), pa['q'], pa['t'])
        step(int(0.3 / DT))
        if e['total_mm'] > 2.0:
            show_marker(pa['tcp_cabinet'], pa['target_cabinet'])
        cap[0] = ra['caption']
        if gui:
            gui.caption(cap[0])
        out['arms'][arm] = dict(hold_start_s=step_i[0] / 240.0)
        step(int(args.hold / DT))
        qm = np.array(art.get_joint_positions()[0], float)[arm_idx[arm]]
        out['arms'][arm].update(joint_tracking_error_deg_at_hold=float(np.degrees(np.max(np.abs(qm - np.array(pa['q_best']))))), planned=e)
        hide_marker()
        cap[0] = f"{'suction' if arm == 'left' else 'gripper'} arm returns home"
        drive(arm, np.array(pa['q_best']), pa['q_back'], pa['t_back'])
        step(int(0.5 / DT))
    if rec_on[0]:
        for _ in range(30):
            rec_frame()
        for v, pr in procs.items():
            pr.stdin.close()
            pr.wait()
    un = {}
    for (a0, a1), n in contacts.items():
        if a0 and a1:
            for r_, o_ in ((a0, a1), (a1, a0)):
                if r_.startswith('/World/D200') and (o_.startswith('/World/Cabinet') or o_.startswith('/World/Cartons')):
                    k = f"{r_.split('/')[-1] if '/' in r_ else r_} -> {o_.split('/')[2]}"
                    un[k] = un.get(k, 0) + n
    out['robot_cabinet_or_stock_contacts'] = un
    out['frames'] = n_frames[0]
    json.dump(out, open(RESULTS / f"unreachable_closest_run_{tn}{'_gui' if gui else ''}.json", 'w'), indent=1, default=float)
    print('RUN', json.dumps(out, default=float), flush=True)
    if gui and not args.gui_close:
        print('### DONE: close the Isaac Sim window to exit', flush=True)
        gui.hold(None)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
