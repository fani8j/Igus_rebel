"""Headless Isaac Sim 5.1: two-tool grasp of one carton in the REAL stocked medicine cabinet with the tools mounted on the REAL D200 dual-arm robot (static AMR placeholder).
The ARM JOINTS are driven (joint position drives, USD stiffness / damping unchanged) along the joint trajectories planned by tools/parking.py + tools/export_plan.py (IK with branch
continuity, collision checked, time-parametrised with the medicine-grasp joint limits).  Lift and sliders are fixed at the parking pose (or, for the top-down split-lift result, the lift is moved
between the phases).  Same sequence, metrics and pass criteria as tests/test_cabinet_grasp.py (kinematic carriers) plus: max arm joint tracking error, arm joint efforts, robot-cabinet contacts.
Run: nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_cabinet_grasp_robot.py --mode front [--targets T5,T3] [--record] [--yaw180 --targets E2]
Plans: data/plans/<mode>_<target>.json (python3 tools/export_plan.py ...).   Results: tests/results/cabinet_grasp_robot_<mode>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from isaac_common import *            # noqa: E402,F401
import numpy as np                    # noqa: E402
import tool_api                       # noqa: E402
from test_cabinet_grasp import (BOX, TARGETS, Cabinet, tool_vertices, aabb_dist, obb_dist, mat_from, set_matrix, quat_wxyz, angle_between,   # noqa: E402
                                SUCTION_TCP_COMP, GRIP_TCP, GRIP_STOP, G_NOM, VACUUM_DELAY_S)

DT = 1 / 240.0
PLANS = Path(__file__).resolve().parent.parent / 'data' / 'plans'
VIDEO_DEFAULT = Path('/media/farhan/My Passport/igus_nvidia/scratch_usd/videos/robot_20261015')
U = '/World/D200/UpperAssembly/dual_arm_rig'
ARM_J = {s: [f'arm_{s}_joint{k}' for k in range(1, 7)] for s in ('left', 'right')}
SX_ROOT, GR_ROOT = f'{U}/arm_left/left_suction_tool', f'{U}/arm_right/right_gripper_tool'
SG = SX_ROOT + '/surface_gripper'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='front', choices=['topdown', 'front'])
    ap.add_argument('--targets', default='all')
    ap.add_argument('--yaw180', action='store_true')
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--header', action='store_true', help='also draw the top info bar (mode/shelf/time/parking/lift) on the videos; default: footer caption only')
    ap.add_argument('--video-dir', default=str(VIDEO_DEFAULT))
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    app = start_app(extra_exts=('isaacsim.robot.surface_gripper', 'omni.kit.viewport.utility', 'omni.replicator.core'))
    print('### STAGE_OPEN')
    apply_d200_physics_settings()
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from isaacsim.core.prims import Articulation
    from scipy.spatial.transform import Rotation as Rot
    from scipy.spatial import cKDTree
    from pxr import Usd
    stage, sim = open_scene(app, 'cabinet_robot_test_scene.usda', DT)
    cab = Cabinet(stage)
    iface = tool_api.suction_interface()
    wf = stage.GetPrimAtPath('/World/D200/joints/world_fix')
    wf_pos0 = np.array(wf.GetAttribute('physics:localPos0').Get(), float)
    # ---- tool collision vertices (for clearances) and contact reporting on every robot body
    verts = {'SX': tool_vertices(stage, SX_ROOT), 'GR': tool_vertices(stage, GR_ROOT)}
    arm_verts = {}
    xc = UsdGeom.XformCache()
    for side in ('left', 'right'):
        for p_ in Usd.PrimRange(stage.GetPrimAtPath(f'{U}/arm_{side}')):
            if not (p_.HasAPI(UsdPhysics.CollisionAPI) and (p_.IsA(UsdGeom.Cube) or p_.IsA(UsdGeom.Cylinder))):
                continue
            if '/left_suction_tool' in str(p_.GetPath()) or '/right_gripper_tool' in str(p_.GetPath()):
                continue
            body = p_.GetParent()
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            Mrel = np.array(xc.ComputeRelativeTransform(p_, body)[0]).T
            if p_.IsA(UsdGeom.Cube):
                hs = float(UsdGeom.Cube(p_).GetSizeAttr().Get()) / 2
                g = np.array([[a, b, c_] for a in (-hs, 0, hs) for b in (-hs, 0, hs) for c_ in (-hs, 0, hs)])
            else:
                cy = UsdGeom.Cylinder(p_)
                r, hh = float(cy.GetRadiusAttr().Get()), float(cy.GetHeightAttr().Get()) / 2
                ax = {'X': 0, 'Y': 1, 'Z': 2}[str(cy.GetAxisAttr().Get())]
                g = []
                for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    for hz in (-hh, 0, hh):
                        v = np.zeros(3)
                        o = [i for i in range(3) if i != ax]
                        v[o[0]], v[o[1]], v[ax] = r * np.cos(ang), r * np.sin(ang), hz
                        g.append(v)
                g = np.array(g + [[0, 0, 0]])
            sc = np.array(Mrel[:3, :3])
            arm_verts.setdefault(str(body.GetPath()), []).append(g @ Mrel[:3, :3].T + Mrel[:3, 3])
    arm_verts = {k: np.vstack(v) for k, v in arm_verts.items()}
    robot_bodies = []
    from pxr import Usd
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/D200')):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            robot_bodies.append(str(p.GetPath()))
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.0)
    print('robot rigid bodies', len(robot_bodies), 'arm collision bodies', len(arm_verts), flush=True)
    contacts = {}

    def on_contact(headers, data):
        for h in headers:
            a0, a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            contacts[(a0, a1)] = contacts.get((a0, a1), 0) + 1
    from omni.physx import get_physx_simulation_interface
    csub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
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
        HEADCAM = f'{U}/stand/gemini336L_camera_link/gemini336L_rgb'                  # real model camera: 94 deg HFOV (focal 16.785 mm / aperture 36 mm), 0.17 .. 20 m
        cam_paths = {'3q': '/World/RecCam3Q', 'side': '/World/RecCamSide', 'head': HEADCAM}
        rec_tools['rp'] = {v: rep.create.render_product(cam_paths[v], (1280, 720)) for v in VIEWS}
        rec_tools['ann'] = {}
        for v, rpd in rec_tools['rp'].items():
            a = rep.AnnotatorRegistry.get_annotator('rgb')
            a.attach([rpd])
            rec_tools['ann'][v] = a
        rec_tools['rep'] = rep
        rec_tools['font'] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
        rec_tools['font_s'] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
        vdir.mkdir(parents=True, exist_ok=True)
    results = {'mode': args.mode, 'runs': [], 'cabinet_clear_mm': {k: cab.clear[k] * 1000 for k in cab.clear}}

    def run_target(tname):
        shelf, box, which = TARGETS[tname]
        h, l, t = BOX[box]
        plan = json.load(open(PLANS / f"{args.mode}_{tname}{'_yaw180' if args.yaw180 else ''}.json"))
        name, tpath, Mc = cab.pick(shelf, box, which)
        assert name == plan['carton'], (name, plan['carton'])
        c = Mc[:3, 3]
        spine_y = c[1] - l / 2
        PULL = plan['pull_m']
        lifts = plan['lift'] if isinstance(plan['lift'], list) else [plan['lift'], plan['lift']]
        sim.stop()
        # ---------- place the robot: world -> cabinet frame (plan['cabinet_to_world'] = cabinet pose in the robot world frame)
        Rcw, tcw = np.array(plan['cabinet_to_world']['R']), np.array(plan['cabinet_to_world']['t'])
        Rrc, trc = Rcw.T, -Rcw.T @ tcw                                   # robot (D200 footprint) pose in the cabinet frame
        Trc = np.eye(4)
        Trc[:3, :3], Trc[:3, 3] = Rrc, trc
        set_matrix(stage.GetPrimAtPath('/World/D200'), mat_from(Rrc, trc))
        wf.GetAttribute('physics:localPos0').Set(Gf.Vec3f(*[float(x) for x in (Rrc @ wf_pos0 + trc)]))
        w, x, y, z = quat_wxyz(Rrc)
        wf.GetAttribute('physics:localRot0').Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
        for jn, v in (('lift_joint', lifts[0]), ('slider_left_joint', plan['slider_left']), ('slider_right_joint', plan['slider_right'])):
            jp = stage.GetPrimAtPath(f'{U}/Joints/{jn}')
            jp.GetAttribute('state:linear:physics:position').Set(float(v))
            jp.GetAttribute('drive:linear:physics:targetPosition').Set(float(v))
        tool_api.set_vacuum_level(stage, SG, 50.0)
        co_phys, _ = tool_api.physical_limits(stage, SG)
        contacts.clear()
        sim.initialize_physics()
        sim.play()
        art = Articulation('/World/D200/base_link')
        art.initialize()
        dofs = {n: i for i, n in enumerate(art.dof_names)}
        arm_idx = {s: np.array([dofs[n] for n in ARM_J[s]]) for s in ARM_J}
        il, ir = dofs['arm_right_finger_left_joint'], dofs['arm_right_finger_right_joint']
        pad_dofs = [dofs[f'arm_left_pad_{i}_buffer_joint'] for i in (1, 2, 3)]
        lift_i = dofs['lift_joint']
        tgt = np.array(art.get_joint_positions()[0], float)
        tgt[il] = tgt[ir] = 0.024
        art.set_joint_position_targets(tgt.reshape(1, -1))
        efforts_lim = {}
        for s in ARM_J:
            for n in ARM_J[s]:
                efforts_lim[n] = float(stage.GetPrimAtPath(f'{U}/Joints/{n}').GetAttribute('drive:angular:physics:maxForce').Get())
        import omni.physics.tensors as pt
        sview = pt.create_simulation_view('numpy')
        rbv = sview.create_rigid_body_view(f'/World/Cartons/Shelf_0{shelf}/*')

        class _SV:
            prim_paths = rbv.prim_paths

            @staticmethod
            def get_world_poses():
                tr = np.array(rbv.get_transforms())
                return tr[:, :3], np.column_stack([tr[:, 6], tr[:, 3], tr[:, 4], tr[:, 5]])
        shelf_view = _SV()
        paths = [str(p) for p in shelf_view.prim_paths]
        ti = paths.index(tpath)
        pos0, q0 = [np.array(a, float) for a in shelf_view.get_world_poses()]
        order = np.argsort(pos0[:, 0])
        rank = list(order).index(ti)
        near = [int(order[r]) for r in range(max(0, rank - 6), min(len(order), rank + 7)) if r != rank]
        adj = [int(order[r]) for r in (rank - 2, rank - 1, rank + 1, rank + 2) if 0 <= r < len(order)]
        mon = tool_api.SuctionLimitMonitor(art, iface, SG, co_phys, pad_dofs=pad_dofs).attach()
        links = [str(x) for x in art._physics_view.link_paths[0]]
        M = dict(pull_out_mm=PULL * 1000, target=tname, shelf=shelf, box=box, which=which, variant=('yaw180' if args.yaw180 else 'base'), mode=args.mode, thickness_mm=t * 1000,
                 clear_height_mm=cab.clear[shelf] * 1000, carton_path=tpath, vacuum_kpa=50.0, steps={}, notes=[],
                 parking=dict(d_mm=plan['d'] * 1000, lateral_mm=plan['lat'] * 1000, lift_m=plan['lift'], slider_left_m=plan['slider_left'], slider_right_m=plan['slider_right']))
        trk = dict(nb_disp=0.0, nb_tilt=0.0, shelf_disp=0.0, tgt_tilt=0.0, tgt_dz=0.0, clr={}, phase='start', clr_phase={}, jerr=0.0, jerr_phase={}, jerr_seg={}, eff={}, eff_phase={})

        def link_pose(path):
            i = links.index(path)
            tr = np.array(art._physics_view.get_link_transforms()[0][i])
            return tr[:3].copy(), np.array([tr[6], tr[3], tr[4], tr[5]])

        def sample(do_clear=True):
            pos, qq = [np.array(a, float) for a in shelf_view.get_world_poses()]
            d = np.linalg.norm(pos - pos0, axis=1)
            trk['nb_disp'] = max(trk['nb_disp'], max([d[i] for i in adj] + [0.0]))
            others = [i for i in range(len(paths)) if i != ti]
            trk['shelf_disp'] = max(trk['shelf_disp'], float(d[others].max()))
            trk['nb_tilt'] = max(trk['nb_tilt'], max([angle_between(q0[i], qq[i]) for i in adj] + [0.0]))
            trk['tgt_tilt'] = max(trk['tgt_tilt'], angle_between(q0[ti], qq[ti]))
            trk['tgt_dz'] = max(trk['tgt_dz'], float(pos[ti][2] - pos0[ti][2]))
            if not do_clear:
                return
            ptsd = {}
            groups = dict(verts)
            for tl, vs in list(groups.items()) + [('ARM', arm_verts)]:
                pts_all = []
                for b, pv in vs.items():
                    pw, qw = link_pose(b)
                    Rb = Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
                    pts_all.append(pv @ Rb.T + pw)
                P = np.vstack(pts_all)
                ptsd[tl] = P
                above, below = f'Shelf_0{shelf + 1}', f'Shelf_0{shelf}'
                cats = {'shelf_above': aabb_dist(P, *cab.boxes[above]).min(), 'shelf_below': aabb_dist(P, *cab.boxes[below]).min(),
                        'walls_trim_back': min(aabb_dist(P, *cab.boxes[k]).min() for k in ('Body_Left', 'Body_Right', 'Body_Back', 'FrontTrim_Left', 'FrontTrim_Right'))}
                nbd = []
                for i in near:
                    Ri = Rot.from_quat([qq[i][1], qq[i][2], qq[i][3], qq[i][0]]).as_matrix()
                    hh, ll, tt = BOX[paths[i].split('/')[-1].rsplit('_', 1)[0]]
                    nbd.append(obb_dist(P, pos[i], Ri, np.array([hh / 2, ll / 2, tt / 2])).min())
                cats['neighbours'] = min(nbd) if nbd else 9.9
                if tl == 'GR':
                    cats['to_other_tool'] = float(cKDTree(ptsd['SX'][::2]).query(P[::2])[0].min())
                for k, v in cats.items():
                    key = f'{tl}.{k}'
                    trk['clr'][key] = min(trk['clr'].get(key, 9.9), float(v))
                    ph = trk['clr_phase'].setdefault(trk['phase'], {})
                    ph[key] = min(ph.get(key, 9.9), float(v))
        step_i = [0]
        cap = ['']
        rec = {'on': False, 'procs': {}, 'n': 0}

        def say(txt):
            cap[0] = txt

        def look_at_M(eye, tgp):
            f = tgp - eye
            f = f / np.linalg.norm(f)
            sx_ = np.cross(f, [0, 0, 1.0])
            sx_ /= np.linalg.norm(sx_)
            u = np.cross(sx_, f)
            return mat_from(np.column_stack([sx_, u, -f]), eye)

        def rec_frame(extra=None):
            rep_ = rec_tools['rep']
            rep_.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            imgs = {}
            for v, ann in rec_tools['ann'].items():
                d = ann.get_data()
                img = Image.fromarray(np.ascontiguousarray(d[..., :3]))
                dr = ImageDraw.Draw(img, 'RGBA')
                dr.rectangle([0, 668, 1280, 720], fill=(0, 0, 0, 150))
                if args.header:
                    dr.rectangle([0, 0, 1280, 64], fill=(0, 0, 0, 150))
                    lf = lift_now()
                    head = f"{args.mode.upper()}  |  shelf {shelf} (clear {cab.clear[shelf] * 1000:.0f} mm)  |  {box} {t * 1000:.0f} mm  |  vacuum -{M['vacuum_kpa']:.0f} kPa  |  {M['variant']}  |  {tname}"
                    dr.text((12, 8), head, font=rec_tools['font_s'], fill=(255, 255, 255, 255))
                    dr.text((12, 34), f"t = {step_i[0] / 240.0:5.1f} s  |  parking d = {plan['d'] * 1000:.0f} mm  |  lift = {lf:+.3f} m  |  view: {dict(zip(VIEWS, ('3/4 front-left', 'side close-up', 'head camera gemini336L_rgb')))[v]}",
                            font=rec_tools['font_s'], fill=(200, 200, 200, 255))
                dr.text((12, 676), extra[0] if extra else cap[0], font=rec_tools['font'], fill=(255, 220, 80, 255) if not extra else extra[1])
                imgs[v] = img
                rec['procs'][v].stdin.write(img.tobytes())
            tile = Image.new('RGB', (1920, 720))
            tile.paste(imgs['3q'], (0, 0))
            tile.paste(imgs['side'].resize((640, 360)), (1280, 0))
            tile.paste(imgs['head'].resize((640, 360)), (1280, 360))
            rec['procs']['tiled'].stdin.write(tile.tobytes())
            rec['n'] += 1
            if rec['n'] in (1, 400):
                pass

        def lift_now():
            return float(art.get_joint_positions()[0][lift_i])

        def step(n=1):
            for _ in range(n):
                art.set_joint_position_targets(tgt.reshape(1, -1))
                sim.step(render=False)
                step_i[0] += 1
                if step_i[0] % 3 == 0:                             # tracking error / effort bookkeeping (80 Hz)
                    qm = np.array(art.get_joint_positions()[0], float)
                    ef = np.array(art.get_measured_joint_efforts()[0], float)
                    for s in ARM_J:
                        e = float(np.max(np.abs(qm[arm_idx[s]] - tgt[arm_idx[s]])))
                        trk['jerr'] = max(trk['jerr'], e)
                        trk['jerr_phase'][trk['phase']] = max(trk['jerr_phase'].get(trk['phase'], 0.0), e)
                        trk['jerr_seg'][trk['seg']] = max(trk['jerr_seg'].get(trk['seg'], 0.0), e)
                        for k, n_ in enumerate(ARM_J[s]):
                            fr = abs(ef[arm_idx[s][k]]) / efforts_lim[n_]
                            trk['eff'][n_] = max(trk['eff'].get(n_, 0.0), float(abs(ef[arm_idx[s][k]])))
                if step_i[0] % 12 == 0:
                    sample()
                if rec['on'] and step_i[0] % 8 == 0:
                    rec_frame()
        trk['seg'] = 'start'

        def run_traj(arm, seg):
            """drive the 6 joints of `arm` along the time-parametrised segment (q0 -> q(t)), then hold the final value"""
            trk['seg'] = seg['name']
            ts = np.array(seg['t'])
            qs = np.vstack([seg['q0'], np.array(seg['q'])])
            tt = np.concatenate([[0.0], ts])
            T = float(tt[-1])
            n = int(np.ceil(T / DT))
            for k in range(1, n + 1):
                tc = k * DT
                tgt[arm_idx[arm]] = [np.interp(min(tc, T), tt, qs[:, j]) for j in range(6)]
                step(1)
            tgt[arm_idx[arm]] = qs[-1]

        def ramp_lift(v, speed=0.1):
            a = float(tgt[lift_i])
            n = max(1, int(abs(v - a) / speed / DT))
            for k in range(n):
                tgt[lift_i] = a + (v - a) * (k + 1) / n
                step(1)
            trk['lift_err_end'] = float(abs(lift_now() - v))

        def fingers(q):
            tgt[il] = tgt[ir] = q

        def close_profile(q_from, q_to, push):
            for q in tool_api.finger_target_profile(q_from, q_to, DT, push=push):
                fingers(q)
                step(1)

        def carton_pose():
            pos, qq = [np.array(a, float) for a in shelf_view.get_world_poses()]
            return pos[ti].copy(), qq[ti].copy()

        def gripper_frame():
            pw, qw = link_pose(GR_ROOT + '/base')
            return pw, Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()

        def rel_to_gripper():
            pw, Rg_ = gripper_frame()
            return Rg_.T @ (carton_pose()[0] - pw)
        S = M['steps']
        if args.record:
            import subprocess
            ctr = c.copy()
            Mcam = {'RecCam3Q': look_at_M(np.array([c[0] - 1.45, -1.95, c[2] + 0.75]), np.array([c[0] + 0.05, -0.30, c[2] - 0.15])),
                    'RecCamSide': look_at_M(np.array([c[0] - 0.45, spine_y - 0.35, c[2] + 0.15]), np.array([c[0], spine_y - 0.04, c[2] + 0.02]))}
            for nm, Mm in Mcam.items():
                set_matrix(stage.GetPrimAtPath(f'/World/{nm}'), Mm)
            for _ in range(8):
                rec_tools['rep'].orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            base_name = f"{args.mode}_{tname}{'_yaw180' if args.yaw180 else ''}"
            vids = {}
            for v, size, suffix in (('3q', '1280x720', '3q'), ('side', '1280x720', 'side'), ('head', '1280x720', 'head'), ('tiled', '1920x720', 'tiled')):
                out_mp4 = vdir / f'{base_name}_{suffix}.mp4'
                vids[v] = str(out_mp4)
                rec['procs'][v] = subprocess.Popen(['/usr/bin/ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', size, '-r', '30', '-i', '-',
                                                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', str(out_mp4)], stdin=subprocess.PIPE)
            rec['on'] = True
            M['videos'] = vids
        say('settle (cartons drop 1 mm onto the shelf, arms hold home)')
        trk['phase'] = '0_settle'
        step(int(1.5 / DT))
        sample()
        pos0, q0 = [np.array(a, float) for a in shelf_view.get_world_poses()]
        trk.update(nb_disp=0.0, nb_tilt=0.0, shelf_disp=0.0, tgt_tilt=0.0, tgt_dz=0.0)
        c0, cq0 = carton_pose()
        L = {s['name']: s for s in plan['arms']['left']}
        Rg = {s['name']: s for s in plan['arms']['right']}
        # ---- left arm: transit to the stage pose, approach, press
        trk['phase'] = '1_press_2_vacuum_3_pull'
        say('suction arm moves to the cabinet (free-space transit)')
        for sg in plan['arms']['left']:
            if sg['name'].startswith(('L0', )):
                run_traj('left', sg)
        say('suction approach + press (buffers ~5 mm)')
        run_traj('left', next(s for s in plan['arms']['left'] if s['name'].startswith('L1')))
        step(int(0.3 / DT))
        comp = [-float(art.get_joint_positions()[0][d_]) * 1000 for d_ in pad_dofs]
        pushed = float((carton_pose()[0] - c0)[1] * 1000)
        S['1_press'] = dict(compression_mm=comp, carton_pushed_back_mm=pushed, travel_past_contact_mm=float(np.mean(comp) + pushed),
                            ok=(0.0 <= pushed <= 6.0 and abs(np.mean(comp) + pushed - 5.0) <= 1.5 and all(x >= -0.5 for x in comp)))
        say('vacuum on (0.15 s build-up, then Surface Gripper closes)')
        step(int(VACUUM_DELAY_S / DT))
        tool_api.suction_close(iface, SG)
        step(int(0.3 / DT))
        stt, objs = tool_api.suction_status(iface, SG), list(iface.get_gripped_objects(SG))
        S['2_vacuum'] = dict(status=stt, gripped=objs, ok=(stt == 'Closed' and len(objs) == 1 and objs[0].startswith(tpath)))
        say('suction pulls the carton out (50 mm/s)')
        y_start = float(carton_pose()[0][1])
        run_traj('left', next(s for s in plan['arms']['left'] if s['name'].startswith('L3')))
        step(int(0.3 / DT))
        disp = (y_start - float(carton_pose()[0][1])) * 1000
        net_disp = (c0[1] - float(carton_pose()[0][1])) * 1000
        S['3_pull'] = dict(carton_disp_mm=disp, carton_net_disp_from_rest_mm=net_disp, status=tool_api.suction_status(iface, SG), monitor_tripped=mon.tripped_at, monitor_max_pad_tension_N=mon.max_tension,
                           neighbour_disp_mm=trk['nb_disp'] * 1000, neighbour_tilt_deg=trk['nb_tilt'], carton_yaw_tilt_deg=angle_between(cq0, carton_pose()[1]),
                           arm_joint_err_max_rad_during_pull=trk['jerr_seg'].get(next(s['name'] for s in plan['arms']['left'] if s['name'].startswith('L3')), 0.0),
                           ok=abs(net_disp - PULL * 1000) <= 3.0 and tool_api.suction_status(iface, SG) == 'Closed' and trk['nb_disp'] < 0.003)
        # ---- release, suction arm retracts and goes home
        trk['phase'] = '5_release_park'
        say('vacuum off, suction arm retracts 100 mm and returns home')
        tool_api.suction_open(iface, SG)
        step(int(0.3 / DT))
        ca = carton_pose()[0]
        for sg in plan['arms']['left']:
            if sg['name'].startswith(('L4', 'L5')):
                run_traj('left', sg)
        S['5_release_park'] = dict(status=tool_api.suction_status(iface, SG), carton_shift_mm=float(np.linalg.norm(carton_pose()[0] - ca) * 1000),
                                   ok=tool_api.suction_status(iface, SG) == 'Open' and np.linalg.norm(carton_pose()[0] - ca) < 0.002)
        if lifts[1] != lifts[0]:
            say(f'lift moves {lifts[0]:+.3f} -> {lifts[1]:+.3f} m (arms at home)')
            ramp_lift(lifts[1])
        # ---- right arm
        trk['phase'] = '4_gripper_approach_clamp'
        t_pre = (t + 0.004 - 0.008) / 2.0
        say('gripper arm moves in front of the cabinet')
        for sg in plan['arms']['right']:
            if sg['name'].startswith(('R0', 'R1')) and 'descend' not in sg['name'] and 'down' not in sg['name']:
                run_traj('right', sg)
        say('gripper pre-close to t + 4 mm')
        close_profile(0.024, t_pre, False)
        step(int(0.2 / DT))
        c_pre = carton_pose()[0]
        say('gripper ' + ('approaches horizontally' if args.mode == 'front' else 'descends') + ' (<= 20 mm/s)')
        for sg in plan['arms']['right']:
            if sg['name'].startswith('R2') or 'descend' in sg['name']:
                run_traj('right', sg)
        step(int(0.3 / DT))
        c_app = carton_pose()[0]
        S['4a_approach'] = dict(carton_shift_mm=float(np.linalg.norm(c_app - c_pre) * 1000), neighbour_disp_mm=trk['nb_disp'] * 1000, ok=np.linalg.norm(c_app - c_pre) < 0.002)
        say('clamp (24.3 N per finger)')
        c_before_clamp = carton_pose()[0]
        close_profile(t_pre, tool_api.GRIP_CLOSE_Q, True)
        step(int(0.5 / DT))
        qq_ = art.get_joint_positions()[0]
        gap = 0.008 + qq_[il] + qq_[ir]
        e = art.get_measured_joint_efforts()[0]
        c_clamped = carton_pose()[0]
        S['4b_clamp'] = dict(gap_mm=float(gap * 1000), gap_error_mm=float((gap - t) * 1000), carton_shift_during_close_mm=float(np.linalg.norm(c_clamped - c_before_clamp) * 1000),
                             finger_effort_N=float(abs(e[il])), ok=abs(gap - t) <= 0.001 and gap > 0.0085)
        rel_ref = rel_to_gripper()
        trk['phase'] = '6_extract_7_lift'
        say('gripper pulls the carton straight out')
        cz0, cy0 = float(carton_pose()[0][2]), float(carton_pose()[0][1])
        gy0 = float(gripper_frame()[0][1])
        ext = next(s for s in plan['arms']['right'] if s['name'].startswith('R4'))
        run_traj('right', ext)
        step(int(0.3 / DT))
        cy1, gy1 = float(carton_pose()[0][1]), float(gripper_frame()[0][1])
        rel1 = rel_to_gripper()
        S['6_extract'] = dict(carton_travel_mm=float((cy0 - cy1) * 1000), gripper_travel_mm=float((gy0 - gy1) * 1000), slip_mm=float(np.linalg.norm(rel1 - rel_ref) * 1000),
                              carton_lift_max_mm=float(trk['tgt_dz'] * 1000), neighbour_disp_mm=trk['nb_disp'] * 1000, neighbour_tilt_deg=trk['nb_tilt'], carton_rotation_deg=trk['tgt_tilt'],
                              carton_fully_clear=bool((carton_pose()[0][1] + l / 2) < cab.boxes['Body_Left'][0][1] - 0.005),
                              arm_joint_err_max_rad=trk['jerr_seg'].get(ext['name'], 0.0),
                              ok=abs((cy0 - cy1) - (gy0 - gy1)) < 0.002 and np.linalg.norm(rel1 - rel_ref) < 0.002 and trk['tgt_dz'] < 0.002 and trk['nb_disp'] < 0.003 and trk['nb_tilt'] < 5.0)
        say('lift 50 mm')
        run_traj('right', next(s for s in plan['arms']['right'] if s['name'].startswith('R5')))
        step(int(0.5 / DT))
        cz2 = float(carton_pose()[0][2])
        rel2 = rel_to_gripper()
        S['7_lift50'] = dict(carton_rise_mm=float((cz2 - cz0) * 1000), slip_mm_total=float(np.linalg.norm(rel2 - rel_ref) * 1000), carton_rotation_deg=trk['tgt_tilt'],
                             carton_rotation_final_deg=angle_between(cq0, carton_pose()[1]),
                             ok=abs((cz2 - cz0) * 1000 - 50.0) <= 2.0 and np.linalg.norm(rel2 - rel_ref) < 0.002 and trk['tgt_tilt'] < 5.0 and angle_between(cq0, carton_pose()[1]) < 5.0)
        sample()

        def classify(a):
            if a.startswith(tpath):
                return 'target'
            if a.startswith('/World/Cartons'):
                return 'neighbour'
            if a.startswith('/World/Cabinet'):
                return 'cabinet'
            if a.startswith(SX_ROOT):
                return 'SX'
            if a.startswith(GR_ROOT):
                return 'GR'
            if a.startswith(f'{U}/arm_left'):
                return 'ARM_L'
            if a.startswith(f'{U}/arm_right'):
                return 'ARM_R'
            if a.startswith('/World/D200'):
                return 'ROBOT_OTHER'
            return 'other'
        unexpected, robot_cab = {}, {}
        for (a0, a1), n in contacts.items():
            if not a0 or not a1:
                continue
            ca_, cb_ = classify(a0), classify(a1)
            for rb, kind in ((ca_, cb_), (cb_, ca_)):
                if rb in ('SX', 'GR', 'ARM_L', 'ARM_R', 'ROBOT_OTHER'):
                    if kind in ('cabinet', 'neighbour'):
                        unexpected[f'{rb}->{kind}'] = unexpected.get(f'{rb}->{kind}', 0) + n
                        robot_cab[f'{rb}->{kind}'] = robot_cab.get(f'{rb}->{kind}', 0) + n
                    elif kind == 'target' and rb not in ('SX', 'GR'):
                        unexpected[f'{rb}->target'] = unexpected.get(f'{rb}->target', 0) + n
                    elif (rb, kind) in (('SX', 'GR'), ('SX', 'ARM_R'), ('GR', 'ARM_L'), ('ARM_L', 'ARM_R'), ('SX', 'ROBOT_OTHER')) and rb < kind or (rb, kind) == ('GR', 'SX') and False:
                        pass
        # arm-arm / tool-other-arm contacts (each unordered pair once)
        for (a0, a1), n in contacts.items():
            if not a0 or not a1:
                continue
            p = tuple(sorted((classify(a0), classify(a1))))
            if p in (('GR', 'SX'), ('ARM_L', 'ARM_R'), ('ARM_R', 'SX'), ('ARM_L', 'GR'), ('ARM_L', 'ARM_L') if False else ('ARM_R', 'SX')):
                unexpected[f'{p[0]}<->{p[1]}'] = unexpected.get(f'{p[0]}<->{p[1]}', 0) + n
        M['unexpected_contacts'] = unexpected
        M['robot_cabinet_contacts'] = robot_cab
        M['contact_pairs_debug'] = {f'{a0} | {a1}': n for (a0, a1), n in list(contacts.items())[:40]}
        M['min_clearance_mm'] = {k: (v * 1000 if v < 9 else None) for k, v in trk['clr'].items()}
        M['min_clearance_by_phase_mm'] = {ph: {k: (v * 1000 if v < 9 else None) for k, v in d.items()} for ph, d in trk['clr_phase'].items()}
        M['neighbour_max_disp_mm'] = trk['nb_disp'] * 1000
        M['neighbour_max_tilt_deg'] = trk['nb_tilt']
        M['shelf_max_disp_mm'] = trk['shelf_disp'] * 1000
        M['carton_max_rotation_deg'] = trk['tgt_tilt']
        M['arm_joint_tracking_error_max_rad'] = trk['jerr']
        M['arm_joint_tracking_error_by_phase_rad'] = trk['jerr_phase']
        M['arm_joint_tracking_error_by_segment_rad'] = trk['jerr_seg']
        M['arm_joint_effort_max'] = trk['eff']
        M['arm_joint_effort_limit'] = efforts_lim
        M['arm_effort_ratio_max'] = float(max(trk['eff'].get(n, 0.0) / efforts_lim[n] for n in efforts_lim))
        M['plan_stage_variant'] = plan['stage_variant']
        M['planned_min_clearance_m'] = plan['clearance_planned_m']
        M['success'] = all(v.get('ok', True) for v in S.values()) and not unexpected
        if rec['on']:
            failed = [k for k, v in S.items() if not v.get('ok', True)] + (['contacts'] if unexpected else [])
            card = ('RESULT: SUCCESS  (all steps OK, no unexpected contact)', (120, 255, 120, 255)) if M['success'] else ('RESULT: FAIL  ' + ', '.join(failed), (255, 110, 110, 255))
            for _ in range(60):
                rec_frame(card)
            for nm, pr in rec['procs'].items():
                pr.stdin.close()
                pr.wait()
            rec['on'] = False
            M['video_frames'] = rec['n']
        mon.detach()
        sim.stop()
        print('RESULT', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in M.items() if k in ('target', 'mode', 'variant', 'success', 'arm_joint_tracking_error_max_rad', 'arm_effort_ratio_max', 'unexpected_contacts')}), flush=True)
        for k, v in S.items():
            print('   ', k, 'OK' if v.get('ok', True) else 'FAIL', json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'ok'}), flush=True)
        print('    clearance mm', json.dumps({k: (round(v, 1) if v is not None else None) for k, v in M['min_clearance_mm'].items()}), flush=True)
        return M

    names = list(TARGETS) if args.targets == 'all' else args.targets.split(',')
    for tn in names:
        if not (PLANS / f"{args.mode}_{tn}{'_yaw180' if args.yaw180 else ''}.json").exists():
            results['runs'].append(dict(target=tn, skipped='no feasible parking pose / plan'))
            continue
        try:
            results['runs'].append(run_target(tn))
        except Exception as ex:
            import traceback
            traceback.print_exc()
            results['runs'].append(dict(target=tn, error=str(ex)))
        json.dump(results, open(RESULTS / (args.out or f'cabinet_grasp_robot_{args.mode}.json'), 'w'), indent=1, default=float)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
