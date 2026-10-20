"""Headless Isaac Sim 5.1: two-tool grasp of one carton in the REAL stocked medicine cabinet, kinematic tool carriers (no arms, no IK).
  suction presses the spine (buffers ~5 mm), vacuum on (0.15 s delay, calibrated limits, physical pull-off monitor), pulls 100 mm at 50 mm/s,
  then the right gripper takes over:   --mode topdown : descends from above on the protruding part (cad gripper README registration v3), pre-close t+4, clamp, suction off, suction retracts and
                                                         steps aside, gripper pulls the carton straight out, lifts 50 mm
                                       --mode front   : suction off and retracts >= 100 mm and parks, gripper approaches horizontally along the shelf depth (jaws across the cabinet),
                                                         pre-close, advance to the grip depth, clamp, pull straight out horizontally, lift 50 mm
Run: nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_cabinet_grasp.py --mode topdown [--targets T5,T1] [--variants vac35,squeeze,lat2,tilt3] [--shots]
Results: tests/results/cabinet_grasp_<mode>.json (+ cabinet_grasp_<mode>_<target>.png with --shots)
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

DT = 1 / 240.0
SHOTS = Path('/media/farhan/My Passport/igus_nvidia/sim_assets_ext/cabinet_grasp_shots')
SHOTS.mkdir(parents=True, exist_ok=True)
BOX = {'agent_monorail': (0.230, 0.250, 0.015), 'angiolite': (0.240, 0.250, 0.016), 'accuforce': (0.220, 0.250, 0.016),
       'ultimaster_nagomi': (0.250, 0.295, 0.016), 'conqueror_nc_pro': (0.230, 0.250, 0.012)}          # height, length, thickness (m), boxes.yaml
TARGETS = {   # name: (shelf, box, which in group)
    'T1': (1, 'ultimaster_nagomi', 'mid'), 'T2': (2, 'accuforce', 'mid'), 'T3': (3, 'agent_monorail', 'mid'), 'T4': (3, 'conqueror_nc_pro', 'mid'),
    'T5': (4, 'angiolite', 'mid'), 'T6': (5, 'agent_monorail', 'mid'), 'T7': (6, 'accuforce', 'mid'), 'T8': (6, 'conqueror_nc_pro', 'mid'),
    'E1': (3, 'agent_monorail', 'last'),       # end of a group, next to a DIFFERENT product (conqueror)
    'E2': (6, 'angiolite', 'first'),           # end of a group, next to the left side wall
}
VARIANTS = ('vac35', 'squeeze', 'lat2', 'tilt3')
# tool geometry (link6 frame, m): suction lip z 0.159 / compressed TCP 0.154; gripper stop face z 0.1675, TCP (pad centre) 0.1932, jaw plane 0.150, tips 0.2269
SUCTION_TCP_COMP, GRIP_TCP, GRIP_STOP, G_NOM = 0.154, 0.1932, 0.1675, 0.004
PULL_OUT = 0.100
# top-down grasp on Shelf_06: the closed TopClosedPanel (y 0..0.02, z >= 1.96) is in the way of the gripper body (z up to stop face + 0.1675 - 0.067) when the spine is only 100 mm out;
# the gripper body / yoke (TCP y +-27.5 mm) must stay at y < 0 -> pull 150 mm there (found by the first run: GR hit the panel, 39907 contacts)
PULL_BY_MODE_SHELF = {('topdown', 6): 0.150}
VACUUM_DELAY_S = 0.15
SPEED_APPROACH, SPEED_PULL, SPEED_GRIP_DESCENT, SPEED_LIFT = 0.020, 0.050, 0.020, 0.025


def mat_from(R, t):
    from pxr import Gf
    m = Gf.Matrix4d(1.0)
    m.SetRotateOnly(Gf.Matrix3d(*[float(x) for x in np.asarray(R, float).T.reshape(9)]))
    m.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in t]))
    return m


def set_matrix(prim, M):
    from pxr import UsdGeom
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(M)


def quat_wxyz(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()
    return q[3], q[0], q[1], q[2]


def angle_between(q0, q1):
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_quat([q0[1], q0[2], q0[3], q0[0]]).inv() * Rot.from_quat([q1[1], q1[2], q1[3], q1[0]])
    return float(np.degrees(r.magnitude()))


class Cabinet:
    def __init__(self, stage):
        from pxr import Usd, UsdGeom
        self.stage = stage
        bb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render, UsdGeom.Tokens.guide])
        self.boxes = {}
        for ch in stage.GetPrimAtPath('/World/Cabinet/Cabinet').GetChildren():
            r = bb.ComputeWorldBound(ch).ComputeAlignedRange()
            if not r.IsEmpty():
                self.boxes[ch.GetName()] = (np.array(r.GetMin()), np.array(r.GetMax()))
        self.shelf_top = {k: self.boxes[f'Shelf_0{k}'][1][2] for k in range(1, 8)}
        self.clear = {k: (self.boxes[f'Shelf_0{k + 1}'][0][2] - self.shelf_top[k]) for k in range(1, 7)}
        self.front_edge_y = self.boxes['Shelf_01'][0][1]          # shelf front edge (scene layer: board from y = 0.05)
        self.xf = UsdGeom.XformCache()
        self.cartons = {}
        for k in range(1, 7):
            sc = stage.GetPrimAtPath(f'/World/Cartons/Shelf_0{k}')
            items = []
            for c in sc.GetChildren():
                M = np.array(self.xf.GetLocalToWorldTransform(c)).T
                items.append((c.GetName(), str(c.GetPath()), M))
            self.cartons[k] = items

    def pick(self, shelf, box, which):
        items = self.cartons[shelf]
        idx = [i for i, it in enumerate(items) if it[0].rsplit('_', 1)[0] == box]
        # group = consecutive in x order
        order = sorted(idx, key=lambda i: items[i][2][0, 3])
        i = {'mid': order[len(order) // 2], 'first': order[0], 'last': order[-1]}[which]
        return items[i]


def tool_vertices(stage, root):
    """collision hull vertices of a tool in its body frames: list of (body path, Nx3)"""
    from pxr import Usd, UsdGeom, UsdPhysics
    xc = UsdGeom.XformCache()
    out = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            q = p.GetParent()
            while q and not q.HasAPI(UsdPhysics.RigidBodyAPI):
                q = q.GetParent()
            M = np.array(xc.ComputeRelativeTransform(p, q)[0]).T
            pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
            out.setdefault(str(q.GetPath()), []).append(pts @ M[:3, :3].T + M[:3, 3])
    return {k: np.vstack(v) for k, v in out.items()}


def aabb_dist(P, lo, hi):
    d = np.maximum(np.maximum(lo - P, P - hi), 0.0)
    return np.linalg.norm(d, axis=1)


def obb_dist(P, c, R, half):
    L = (P - c) @ R
    d = np.maximum(np.abs(L) - half, 0.0)
    return np.linalg.norm(d, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='topdown', choices=['topdown', 'front'])
    ap.add_argument('--targets', default='all')
    ap.add_argument('--variants', default='')
    ap.add_argument('--shots', action='store_true')
    ap.add_argument('--record', action='store_true', help='render MP4s (3/4 view + side view) of every run, physics unchanged (checked)')
    ap.add_argument('--video-dir', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    app = start_app(extra_exts=('isaacsim.robot.surface_gripper', 'omni.kit.viewport.utility', 'omni.replicator.core'))
    print('### STAGE_OPEN')
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from isaacsim.core.prims import Articulation, RigidPrim
    from scipy.spatial.transform import Rotation as Rot
    stage, sim = open_scene(app, 'cabinet_grasp_test_scene.usda', DT)
    cab = Cabinet(stage)
    print('shelf tops', {k: round(v, 3) for k, v in cab.shelf_top.items()}, 'clear mm', {k: round(v * 1000) for k, v in cab.clear.items()}, flush=True)
    iface = tool_api.suction_interface()
    SX, GR = '/World/Tools/SX', '/World/Tools/GR'
    SG = SX + '/tool/surface_gripper'
    verts = {'SX': tool_vertices(stage, SX + '/tool'), 'GR': tool_vertices(stage, GR + '/tool')}
    tool_bodies = {'SX': list(verts['SX'].keys()), 'GR': list(verts['GR'].keys())}
    for grp in tool_bodies.values():
        for b in grp:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(b)).CreateThresholdAttr(0.0)
    contacts = {}

    def on_contact(headers, data):
        for h in headers:
            a0, a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            contacts[(a0, a1)] = contacts.get((a0, a1), 0) + 1
    from omni.physx import get_physx_simulation_interface
    csub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    vdir = Path(args.video_dir) if args.video_dir else RESULTS / 'videos'
    vdir.mkdir(parents=True, exist_ok=True)
    rec_tools = {}
    if args.record:
        import subprocess
        import omni.replicator.core as rep
        from PIL import Image, ImageDraw, ImageFont
        for nm in ('RecCam3Q', 'RecCamSide'):
            cm = UsdGeom.Camera.Define(stage, f'/World/{nm}')
            cm.CreateFocalLengthAttr(24.0 if nm == 'RecCam3Q' else 28.0)
            cm.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        rec_tools['rp'] = {nm: rep.create.render_product(f'/World/{nm}', (1280, 720)) for nm in ('RecCam3Q', 'RecCamSide')}
        rec_tools['ann'] = {}
        for nm, rpd in rec_tools['rp'].items():
            a = rep.AnnotatorRegistry.get_annotator('rgb')
            a.attach([rpd])
            rec_tools['ann'][nm] = a
        rec_tools['rep'] = rep
        try:
            rec_tools['font'] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
            rec_tools['font_s'] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
        except Exception:
            rec_tools['font'] = rec_tools['font_s'] = ImageFont.load_default()
    results = {'mode': args.mode, 'cabinet_clear_mm': {k: cab.clear[k] * 1000 for k in cab.clear}, 'runs': []}

    def place(name, T, ly0, lz0):
        base = f'/World/Tools/{name}'
        set_matrix(stage.GetPrimAtPath(base + '/root'), mat_from(T[:3, :3], T[:3, 3]))
        Ty = T.copy()
        Ty[:3, 3] = T[:3, 3] + T[:3, 1] * ly0
        set_matrix(stage.GetPrimAtPath(base + '/stage_body'), mat_from(Ty[:3, :3], Ty[:3, 3]))
        Tz = Ty.copy()
        Tz[:3, 3] = Ty[:3, 3] + T[:3, 2] * lz0
        set_matrix(stage.GetPrimAtPath(base + '/tool'), mat_from(Tz[:3, :3], Tz[:3, 3]))
        rj = stage.GetPrimAtPath(base + '/joints/root_joint')
        rj.GetAttribute('physics:localPos0').Set(Gf.Vec3f(*[float(x) for x in T[:3, 3]]))
        w, x, y, z = quat_wxyz(T[:3, :3])
        rj.GetAttribute('physics:localRot0').Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
        for jn, v in (('jy', ly0), ('jz', lz0)):
            jp = stage.GetPrimAtPath(f'{base}/joints/{name}_{jn}')
            jp.GetAttribute('state:linear:physics:position').Set(float(v))
            jp.GetAttribute('drive:linear:physics:targetPosition').Set(float(v))

    def run_target(tname, variant=None, shots=False):
        shelf, box, which = TARGETS[tname]
        h, l, t = BOX[box]
        name, tpath, Mc = cab.pick(shelf, box, which)
        R_c, c = Mc[:3, :3], Mc[:3, 3]
        spine_y = c[1] - l / 2
        z_b = c[2] - h / 2
        # ---------- poses
        Rs = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float)
        Ts = np.eye(4)
        Ts[:3, :3] = Rs
        Ts[:3, 3] = np.array([c[0], spine_y, c[2]]) - Rs[:, 2] * SUCTION_TCP_COMP
        if variant == 'lat2':
            Ts[:3, 3] += Rs[:, 1] * 0.002
        if variant == 'tilt3':
            Rz = Rot.from_euler('z', 3.0, degrees=True).as_matrix()
            piv = np.array([c[0], spine_y, c[2]])
            Ts[:3, :3] = Rz @ Ts[:3, :3]
            Ts[:3, 3] = piv + Rz @ (Ts[:3, 3] - piv)
        PULL = PULL_BY_MODE_SHELF.get((args.mode, shelf), PULL_OUT)
        spine_pulled = spine_y - PULL
        Tg = np.eye(4)
        if args.mode == 'topdown':
            Rg = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(float)
            tcp = np.array([c[0], spine_pulled + 0.0425, z_b + h + G_NOM - (GRIP_TCP - GRIP_STOP)])
            lz0_g, ly0_g = -0.070, 0.0
        else:
            Rg = np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)       # tool x = +X (jaws across the cabinet), y = -Z, z = +Y (into the shelf)
            l6y = spine_pulled - G_NOM - GRIP_STOP                                         # stop face 4 mm before the spine (the crossbar sits between the jaws)
            tcp = np.array([c[0], l6y + GRIP_TCP, c[2]])
            lz0_g, ly0_g = -0.100, -0.300                                                  # starts 0.3 m ABOVE its grip height (tool y = -Z_w), 100 mm before the final depth
        Tg[:3, :3] = Rg
        Tg[:3, 3] = tcp - Rg[:, 2] * GRIP_TCP
        sim.stop()
        # ---------- variant edits of the carton row
        shift_info = None
        orig = {}
        if variant == 'squeeze':
            row = sorted([it for it in cab.cartons[shelf] if it[0].rsplit('_', 1)[0] == box], key=lambda it: it[2][0, 3])
            ti = [it[1] for it in row].index(tpath)
            for k, it in enumerate(row):
                d = (ti - k)                                                       # cartons left of the target move right by d mm, right ones left (gap 1 mm -> 0)
                if d != 0:
                    p = stage.GetPrimAtPath(it[1])
                    M = Gf.Matrix4d(*[float(x) for x in it[2].T.reshape(16)])
                    Mn = Gf.Matrix4d(M)
                    Mn.SetTranslateOnly(M.ExtractTranslation() + Gf.Vec3d(0.001 * d, 0, 0))
                    set_matrix(p, Mn)
                    orig[it[1]] = M
            shift_info = len(orig)
        place('SX', Ts, 0.0, -0.030)
        place('GR', Tg, ly0_g, lz0_g)
        tool_api.set_vacuum_level(stage, SG, 35.0 if variant == 'vac35' else 50.0)
        co_phys, _ = tool_api.physical_limits(stage, SG)
        contacts.clear()
        sim.initialize_physics()
        sim.play()
        sx = Articulation(SX + '/root')
        sx.initialize()
        gr = Articulation(GR + '/root')
        gr.initialize()
        sdof = {n: i for i, n in enumerate(sx.dof_names)}
        gdof = {n: i for i, n in enumerate(gr.dof_names)}
        # NOTE: isaacsim.core.prims.RigidPrim on these instanceable cartons makes them invisible in the render (it writes default poses to the instance proxies);
        # the raw physics tensor view only reads the transforms.
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
        mon = tool_api.SuctionLimitMonitor(sx, iface, SG, co_phys).attach()
        links = {'SX': [str(x) for x in sx._physics_view.link_paths[0]], 'GR': [str(x) for x in gr._physics_view.link_paths[0]]}
        arts = {'SX': sx, 'GR': gr}
        dofs = {'SX': sdof, 'GR': gdof}
        tgt = {'SX': np.zeros(len(sx.dof_names)), 'GR': np.zeros(len(gr.dof_names))}
        tgt['SX'][sdof['SX_jy']], tgt['SX'][sdof['SX_jz']] = 0.0, -0.030
        tgt['GR'][gdof['GR_jy']], tgt['GR'][gdof['GR_jz']] = ly0_g, lz0_g
        gnames = {'jf': gdof['finger_left_joint']}
        cur = {'SX': tgt['SX'].copy(), 'GR': tgt['GR'].copy()}
        # per-run metrics
        M = dict(pull_out_mm=PULL * 1000, target=tname, shelf=shelf, box=box, which=which, variant=variant or 'base', mode=args.mode, thickness_mm=t * 1000, clear_height_mm=cab.clear[shelf] * 1000,
                 carton_path=tpath, vacuum_kpa=35.0 if variant == 'vac35' else 50.0, steps={}, notes=[])
        M['squeezed_cartons'] = shift_info
        trk = dict(nb_disp=0.0, nb_tilt=0.0, shelf_disp=0.0, shelf_tilt=0.0, tgt_tilt=0.0, tgt_dz=0.0, clr={}, phase='start', clr_phase={})
        base0 = {}

        def link_pose(tool, path):
            i = links[tool].index(path)
            tr = np.array(arts[tool]._physics_view.get_link_transforms()[0][i])
            return tr[:3].copy(), np.array([tr[6], tr[3], tr[4], tr[5]])

        def sample(do_clear=True):
            pos, qq = [np.array(a, float) for a in shelf_view.get_world_poses()]
            d = np.linalg.norm(pos - pos0, axis=1)
            nbd = max([d[i] for i in adj] + [0.0])
            trk['nb_disp'] = max(trk['nb_disp'], nbd)
            others = [i for i in range(len(paths)) if i != ti]
            trk['shelf_disp'] = max(trk['shelf_disp'], float(d[others].max()))
            tilts = [angle_between(q0[i], qq[i]) for i in adj]
            trk['nb_tilt'] = max(trk['nb_tilt'], max(tilts + [0.0]))
            trk['tgt_tilt'] = max(trk['tgt_tilt'], angle_between(q0[ti], qq[ti]))
            trk['tgt_dz'] = max(trk['tgt_dz'], float(pos[ti][2] - pos0[ti][2]))
            if not do_clear:
                return
            ptsd = {}
            for tl in ('SX', 'GR'):
                pts_all = []
                for b, pv in verts[tl].items():
                    pw, qw = link_pose(tl, b)
                    Rb = Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
                    pts_all.append(pv @ Rb.T + pw)
                P = np.vstack(pts_all)
                ptsd[tl] = P
                above, below = f'Shelf_0{shelf + 1}', f'Shelf_0{shelf}'
                cats = {'shelf_above': aabb_dist(P, *cab.boxes[above]).min(),
                        'shelf_below': aabb_dist(P, *cab.boxes[below]).min(),
                        'walls_trim_back': min(aabb_dist(P, *cab.boxes[k]).min() for k in ('Body_Left', 'Body_Right', 'Body_Back', 'FrontTrim_Left', 'FrontTrim_Right'))}
                nbd = []
                for i in near:
                    Ri = Rot.from_quat([qq[i][1], qq[i][2], qq[i][3], qq[i][0]]).as_matrix()
                    bn = [BOX[paths[i].split('/')[-1].rsplit('_', 1)[0]]]
                    hh, ll, tt = bn[0]
                    nbd.append(obb_dist(P, pos[i], Ri, np.array([hh / 2, ll / 2, tt / 2])).min())      # carton frame: x height, y length, z thickness
                cats['neighbours'] = min(nbd) if nbd else 9.9
                if tl == 'GR':
                    from scipy.spatial import cKDTree
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

        def look_at_M(eye, tg):
            f = tg - eye
            f = f / np.linalg.norm(f)
            sx_ = np.cross(f, [0, 0, 1.0])
            sx_ /= np.linalg.norm(sx_)
            u = np.cross(sx_, f)
            return mat_from(np.column_stack([sx_, u, -f]), eye)

        def rec_frame(extra=None):
            rep_ = rec_tools['rep']
            rep_.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            for nm, ann in rec_tools['ann'].items():
                d = ann.get_data()
                img = Image.fromarray(np.ascontiguousarray(d[..., :3]))
                dr = ImageDraw.Draw(img, 'RGBA')
                dr.rectangle([0, 0, 1280, 64], fill=(0, 0, 0, 150))
                dr.rectangle([0, 668, 1280, 720], fill=(0, 0, 0, 150))
                dr.text((12, 8), f"{args.mode.upper()}  |  shelf {shelf} (clear {cab.clear[shelf] * 1000:.0f} mm)  |  {box} {t * 1000:.0f} mm  |  vacuum -{M['vacuum_kpa']:.0f} kPa  |  {variant or 'baseline'}  |  {tname}", font=rec_tools['font_s'], fill=(255, 255, 255, 255))
                dr.text((12, 34), f"t = {step_i[0] / 240.0:5.1f} s", font=rec_tools['font_s'], fill=(200, 200, 200, 255))
                dr.text((12, 676), extra[0] if extra else cap[0], font=rec_tools['font'], fill=(255, 220, 80, 255) if not extra else extra[1])
                rec['procs'][nm].stdin.write(img.tobytes())
            rec['n'] += 1

        def step(n=1):
            for _ in range(n):
                for tl in ('SX', 'GR'):
                    arts[tl].set_joint_position_targets(cur[tl].reshape(1, -1))
                sim.step(render=False)
                step_i[0] += 1
                if step_i[0] % 12 == 0:
                    sample()
                if rec['on'] and step_i[0] % 8 == 0:
                    rec_frame()

        def ramp(moves, speed):
            """moves: list of (tool, dofname, target); all move together at the given speed (the longest one sets the duration)."""
            n = max(1, int(max(abs(v - cur[tl][dofs[tl][d]]) for tl, d, v in moves) / speed / DT))
            st = [(tl, dofs[tl][d], cur[tl][dofs[tl][d]], v) for tl, d, v in moves]
            for k in range(n):
                for tl, i, a, b in st:
                    cur[tl][i] = a + (b - a) * (k + 1) / n
                step(1)

        def carton_pose():
            pos, qq = [np.array(a, float) for a in shelf_view.get_world_poses()]
            return pos[ti].copy(), qq[ti].copy()

        def gripper_frame():
            pw, qw = link_pose('GR', GR + '/tool/base')
            return pw, Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()

        def rel_to_gripper():
            pw, Rg_ = gripper_frame()
            return Rg_.T @ (carton_pose()[0] - pw)

        def shot(tag):
            if not shots:
                return
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            cpos = carton_pose()[0]
            set_camera_view(eye=cpos + np.array([0.55, -0.75, 0.35]), target=cpos + np.array([0, 0.0, 0.0]), camera_prim_path='/OmniverseKit_Persp')
            sim.pause()
            for _ in range(30):
                app.update()
            capture_viewport_to_file(get_active_viewport(), str(SHOTS / f'cabinet_grasp_{args.mode}_{tname}{("_" + variant) if variant else ""}_{tag}.png'))
            for _ in range(30):
                app.update()
            sim.play()

        S = M['steps']
        if args.record:
            import subprocess
            ctr = c.copy()
            Mcam = {'RecCam3Q': look_at_M(ctr + np.array([-0.65, -0.65, 0.35]), ctr + np.array([0.0, -0.05, 0.0])),
                    'RecCamSide': look_at_M(np.array([c[0] - 0.35, spine_y - 0.30, c[2] + 0.12]), np.array([c[0], spine_y - 0.04, c[2] + 0.02]))}
            for nm, Mm in Mcam.items():
                set_matrix(stage.GetPrimAtPath(f'/World/{nm}'), Mm)
            for _ in range(8):
                rec_tools['rep'].orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            base_name = f"{args.mode}_{tname}{('_' + variant) if variant else ''}"
            for nm, suffix in (('RecCam3Q', '3q'), ('RecCamSide', 'side')):
                out_mp4 = vdir / f'{base_name}_{suffix}.mp4'
                rec['procs'][nm] = subprocess.Popen(['/usr/bin/ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '30', '-i', '-',
                                                     '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p', str(out_mp4)], stdin=subprocess.PIPE)
            rec['on'] = True
            M['videos'] = [str(vdir / f'{base_name}_3q.mp4'), str(vdir / f'{base_name}_side.mp4')]
        say('settle (cartons drop 1 mm onto the shelf)')
        step(int(1.0 / DT))                                                         # cartons drop their 1 mm onto the shelf, tools settle
        sample()
        pos0, q0 = [np.array(a, float) for a in shelf_view.get_world_poses()]       # reference after the settle
        trk.update(nb_disp=0.0, nb_tilt=0.0, shelf_disp=0.0, tgt_tilt=0.0, tgt_dz=0.0)
        c0, cq0 = carton_pose()
        ok_all = True
        shot('0_start')
        # ---- 1 press
        say('suction approach + press (buffers ~5 mm)')
        trk['phase'] = '1_press_2_vacuum_3_pull'
        ramp([('SX', 'SX_jz', 0.0)], SPEED_APPROACH)
        step(int(0.3 / DT))
        comp = [-float(sx.get_joint_positions()[0][sdof[f'pad_{i}_buffer_joint']]) * 1000 for i in (1, 2, 3)]
        pushed = float((carton_pose()[0] - c0)[1] * 1000)
        # a loose carton yields to the 1..3 N pad springs (shelf friction ~0.4 N): the 5 mm of press travel past first contact is shared between buffer compression and carton push-back
        S['1_press'] = dict(compression_mm=comp, carton_pushed_back_mm=pushed, travel_past_contact_mm=float(np.mean(comp) + pushed), ok=(0.0 <= pushed <= 6.0 and abs(np.mean(comp) + pushed - 5.0) <= 1.5 and all(x >= -0.5 for x in comp)))
        # ---- 2 vacuum
        say('vacuum on (0.15 s build-up, then Surface Gripper closes)')
        step(int(VACUUM_DELAY_S / DT))
        tool_api.suction_close(iface, SG)
        step(int(0.3 / DT))
        stt, objs = tool_api.suction_status(iface, SG), list(iface.get_gripped_objects(SG))
        S['2_vacuum'] = dict(status=stt, gripped=objs, ok=(stt == 'Closed' and len(objs) == 1 and objs[0].startswith(tpath)))
        # ---- 3 pull 100 mm
        say('suction pulls the carton out (50 mm/s)')
        y_start = float(carton_pose()[0][1])
        errs = []
        n_pull = int(((PULL + 0.005) / SPEED_PULL) / DT)
        s0 = cur['SX'][sdof['SX_jz']]
        for k in range(n_pull):
            cur['SX'][sdof['SX_jz']] = s0 - (PULL + 0.005) * (k + 1) / n_pull
            step(1)
            if k % 24 == 0 and os.environ.get('CG_DEBUG'):
                ee = sx.get_measured_joint_efforts()[0]
                print('PULLDBG', k, 'cartonY', round(float(carton_pose()[0][1]), 4), 'jz', round(float(cur['SX'][sdof['SX_jz']]), 4), 'tension', [round(-float(ee[sdof[f'pad_{i}_buffer_joint']]), 2) for i in (1, 2, 3)], 'status', tool_api.suction_status(iface, SG), flush=True)
            if k % 12 == 0:
                pw, _ = link_pose('SX', SX + '/tool/pad_2')
                errs.append(abs((carton_pose()[0][1] - l / 2) - (pw[1] + 0.159 - 0.0 + float(sx.get_joint_positions()[0][sdof['pad_2_buffer_joint']]))))
        step(int(0.3 / DT))
        disp = (y_start - float(carton_pose()[0][1])) * 1000
        net_disp = (c0[1] - float(carton_pose()[0][1])) * 1000                      # from the rest position before the press
        rot_now = angle_between(cq0, carton_pose()[1])
        S['3_pull'] = dict(carton_disp_mm=disp, carton_net_disp_from_rest_mm=net_disp, follow_error_mm=float(max(errs)) * 1000, status=tool_api.suction_status(iface, SG), monitor_tripped=mon.tripped_at,
                           neighbour_disp_mm=trk['nb_disp'] * 1000, neighbour_tilt_deg=trk['nb_tilt'], carton_yaw_tilt_deg=rot_now,
                           ok=abs(net_disp - PULL * 1000) <= 3.0 and tool_api.suction_status(iface, SG) == 'Closed' and trk['nb_disp'] < 0.003)
        shot('1_pulled')
        carton_after_pull, _ = carton_pose()
        # ---- gripper + suction release sequences
        trk['phase'] = '5_release_park' if args.mode == 'front' else '4_gripper_approach_clamp'
        def gripper_close_profile(q_from, q_to, push):
            for q in tool_api.finger_target_profile(q_from, q_to, DT, push=push):
                cur['GR'][gnames['jf']] = q
                step(1)

        def grip_gap():
            q = gr.get_joint_positions()[0]
            return 0.008 + q[gdof['finger_left_joint']] + q[gdof['finger_right_joint']]
        q_pre = (t + 0.004 - 0.008) / 2.0
        cur['GR'][gnames['jf']] = 0.024
        step(int(0.3 / DT))
        if args.mode == 'front':
            # suction off, retract straight back >= 100 mm along the spine normal, park laterally (away from the side with the cabinet wall)
            say('vacuum off, suction retracts 100 mm and parks aside')
            tool_api.suction_open(iface, SG)
            step(int(0.3 / DT))
            ca = carton_pose()[0]
            ramp([('SX', 'SX_jz', cur['SX'][sdof['SX_jz']] - 0.100)], SPEED_PULL)
            side = -1.0 if c[0] > 0 else 1.0
            ramp([('SX', 'SX_jy', side * 0.30)], SPEED_PULL)                     # tool y = +X_w
            step(int(0.3 / DT))
            S['5_release_park'] = dict(status=tool_api.suction_status(iface, SG), carton_shift_mm=float(np.linalg.norm(carton_pose()[0] - ca) * 1000), retract_mm=100, park_lateral_mm=side * 300,
                                       ok=tool_api.suction_status(iface, SG) == 'Open' and np.linalg.norm(carton_pose()[0] - ca) < 0.002)
        trk['phase'] = '4_gripper_approach_clamp'
        if args.mode == 'front':
            say('gripper lowered in free air')
            ramp([('GR', 'GR_jy', 0.0)], SPEED_PULL)                                    # lower the gripper to the pad-row height in free air in front of the cabinet
        # pre-close to t + 4 mm (jaws still clear of the carton)
        say('gripper pre-close to t + 4 mm')
        gripper_close_profile(0.024, q_pre, False)
        step(int(0.2 / DT))
        c_pre = carton_pose()[0]
        # approach / descend at <= 20 mm/s to the final pose (position based, no search)
        say('gripper ' + ('descends' if args.mode == 'topdown' else 'approaches horizontally') + ' (<= 20 mm/s)')
        ramp([('GR', 'GR_jz', 0.0)], SPEED_GRIP_DESCENT)
        step(int(0.3 / DT))
        c_app = carton_pose()[0]
        S['4a_approach'] = dict(carton_shift_mm=float(np.linalg.norm(c_app - c_pre) * 1000), neighbour_disp_mm=trk['nb_disp'] * 1000,
                                carton_shift_xyz_mm=[float(x) for x in (c_app - c_pre) * 1000], ok=np.linalg.norm(c_app - c_pre) < 0.002)
        shot('2_gripper_in_place')
        # clamp
        say('clamp (24.3 N per finger)')
        c_before_clamp = carton_pose()[0]
        gripper_close_profile(q_pre, tool_api.GRIP_CLOSE_Q, True)
        step(int(0.5 / DT))
        gap = grip_gap()
        e = gr.get_measured_joint_efforts()[0]
        c_clamped = carton_pose()[0]
        S['4b_clamp'] = dict(gap_mm=float(gap * 1000), gap_error_mm=float((gap - t) * 1000), carton_shift_during_close_mm=float(np.linalg.norm(c_clamped - c_before_clamp) * 1000),
                             carton_shift_xyz_mm=[float(x) for x in (c_clamped - c_before_clamp) * 1000], finger_effort_N=float(abs(e[gdof['finger_left_joint']])),
                             ok=abs(gap - t) <= 0.001 and gap > 0.0085)
        rel_ref = rel_to_gripper()
        trk['phase'] = '5_release_aside'
        if args.mode == 'topdown':
            say('vacuum off, suction retracts 30 mm and steps aside')
        if args.mode == 'topdown':
            tool_api.suction_open(iface, SG)
            step(int(0.3 / DT))
            ca = carton_pose()[0]
            ramp([('SX', 'SX_jz', cur['SX'][sdof['SX_jz']] - 0.030)], SPEED_PULL)
            side = -1.0 if c[0] > 0 else 1.0
            ramp([('SX', 'SX_jy', side * 0.12)], SPEED_PULL)
            step(int(0.3 / DT))
            S['5_release_aside'] = dict(status=tool_api.suction_status(iface, SG), carton_shift_mm=float(np.linalg.norm(carton_pose()[0] - ca) * 1000), retract_mm=30, aside_lateral_mm=side * 120,
                                        ok=tool_api.suction_status(iface, SG) == 'Open' and np.linalg.norm(carton_pose()[0] - ca) < 0.002)
        rel_ref = rel_to_gripper()
        # ---- 6 extraction: straight out until the carton fully clears the cabinet, then lift 50 mm
        trk['phase'] = '6_extract_7_lift'
        say('gripper pulls the carton straight out')
        y_rear = (c[1] + l / 2) - PULL
        D = y_rear - (cab.boxes['Body_Left'][0][1] - 0.02)                           # rear end 20 mm outside the cabinet front plane (y = 0)
        cz0 = float(carton_pose()[0][2])
        cy0 = float(carton_pose()[0][1])
        gz0 = float(gripper_frame()[0][2])
        gy0 = float(gripper_frame()[0][1])
        if args.mode == 'topdown':
            ramp([('GR', 'GR_jy', cur['GR'][gdof['GR_jy']] + D)], SPEED_PULL)          # tool y = -Y_w
        else:
            ramp([('GR', 'GR_jz', cur['GR'][gdof['GR_jz']] - D)], SPEED_PULL)          # tool z = +Y_w
        step(int(0.3 / DT))
        cy1 = float(carton_pose()[0][1])
        gy1 = float(gripper_frame()[0][1])
        rel1 = rel_to_gripper()
        S['6_extract'] = dict(distance_mm=float(D * 1000), carton_travel_mm=float((cy0 - cy1) * 1000), gripper_travel_mm=float((gy0 - gy1) * 1000),
                              slip_mm=float(np.linalg.norm(rel1 - rel_ref) * 1000), carton_lift_max_mm=float(trk['tgt_dz'] * 1000), neighbour_disp_mm=trk['nb_disp'] * 1000,
                              neighbour_tilt_deg=trk['nb_tilt'], carton_rotation_deg=trk['tgt_tilt'], carton_fully_clear=bool((carton_pose()[0][1] + l / 2) < cab.boxes['Body_Left'][0][1] - 0.005),
                              ok=abs((cy0 - cy1) - (gy0 - gy1)) < 0.002 and np.linalg.norm(rel1 - rel_ref) < 0.002 and trk['tgt_dz'] < 0.002 and trk['nb_disp'] < 0.003 and trk['nb_tilt'] < 5.0)
        shot('3_extracted')
        say('lift 50 mm')
        if args.mode == 'topdown':
            ramp([('GR', 'GR_jz', cur['GR'][gdof['GR_jz']] - 0.050)], SPEED_LIFT)
        else:
            ramp([('GR', 'GR_jy', cur['GR'][gdof['GR_jy']] - 0.050)], SPEED_LIFT)      # tool y = -Z_w : -0.05 = up
        step(int(0.5 / DT))
        cz2 = float(carton_pose()[0][2])
        rel2 = rel_to_gripper()
        S['7_lift50'] = dict(carton_rise_mm=float((cz2 - cz0) * 1000), slip_mm_total=float(np.linalg.norm(rel2 - rel_ref) * 1000), carton_rotation_deg=trk['tgt_tilt'],
                             carton_rotation_final_deg=angle_between(cq0, carton_pose()[1]),
                             ok=abs((cz2 - cz0) * 1000 - 50.0) <= 2.0 and np.linalg.norm(rel2 - rel_ref) < 0.002 and trk['tgt_tilt'] < 5.0 and angle_between(cq0, carton_pose()[1]) < 5.0)
        shot('4_lifted')
        sample()
        # ---- contacts / clearances
        def classify(a):
            if a.startswith(tpath):
                return 'target'
            if a.startswith('/World/Cartons'):
                return 'neighbour'
            if a.startswith('/World/Cabinet'):
                return 'cabinet'
            if a.startswith(SX):
                return 'SX'
            if a.startswith(GR):
                return 'GR'
            return 'other'
        unexpected = {}
        for (a0, a1), n in contacts.items():
            if not a0 or not a1:
                continue                                                         # actor already removed at sim.stop
            ca, cb = classify(a0), classify(a1)
            for tl, kind in ((ca, cb), (cb, ca)):
                if tl in ('SX', 'GR') and kind not in ('target', tl):
                    unexpected[f'{tl}->{kind}'] = unexpected.get(f'{tl}->{kind}', 0) + n
        M['unexpected_contacts'] = unexpected
        M['contact_pairs_debug'] = {f'{a0} | {a1}': n for (a0, a1), n in list(contacts.items())[:40]}
        M['min_clearance_mm'] = {k: (v * 1000 if v < 9 else None) for k, v in trk['clr'].items()}
        M['min_clearance_by_phase_mm'] = {ph: {k: (v * 1000 if v < 9 else None) for k, v in d.items()} for ph, d in trk['clr_phase'].items()}
        M['neighbour_max_disp_mm'] = trk['nb_disp'] * 1000
        M['neighbour_max_tilt_deg'] = trk['nb_tilt']
        M['shelf_max_disp_mm'] = trk['shelf_disp'] * 1000
        M['carton_max_rotation_deg'] = trk['tgt_tilt']
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
        for pth, Mo in orig.items():
            set_matrix(stage.GetPrimAtPath(pth), Mo)
        print('RESULT', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in M.items() if k not in ('steps', 'min_clearance_mm', 'notes')}), flush=True)
        for k, v in S.items():
            print('   ', k, 'OK' if v.get('ok', True) else 'FAIL', json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'ok'}), flush=True)
        print('    clearance mm', json.dumps({k: (round(v, 1) if v is not None else None) for k, v in M['min_clearance_mm'].items()}), 'unexpected contacts', unexpected, flush=True)
        return M

    names = list(TARGETS) if args.targets == 'all' else args.targets.split(',')
    for tn in names:
        try:
            results['runs'].append(run_target(tn, None, args.shots))
        except Exception as ex:
            import traceback
            traceback.print_exc()
            results['runs'].append(dict(target=tn, error=str(ex)))
        json.dump(results, open(RESULTS / (args.out or f'cabinet_grasp_{args.mode}.json'), 'w'), indent=1, default=float)
    for v in [x for x in args.variants.split(',') if x]:
        try:
            results['runs'].append(run_target('T5', v, False))
        except Exception as ex:
            import traceback
            traceback.print_exc()
            results['runs'].append(dict(target='T5', variant=v, error=str(ex)))
        json.dump(results, open(RESULTS / (args.out or f'cabinet_grasp_{args.mode}.json'), 'w'), indent=1, default=float)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
