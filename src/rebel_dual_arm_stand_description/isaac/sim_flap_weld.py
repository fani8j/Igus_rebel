"""Run this INSIDE Isaac Sim (Window > Script Editor) after dual_arm_rig.usda is open and
dual_arm_rig_ros2_bridge.py has been run. It adds the two sim-only services the lid-opening
data collection (scripts/lid_open/) needs on top of Isaac's own ROS 2 simulation interfaces:

    /sim/attach_flap   (std_srvs/SetBool, data: true = arm_right, false = arm_left)
        Welds the gripper finger of that arm to whichever lid flap (LidBack / LidFront) is
        nearest to it, with a FixedJoint frozen at their CURRENT relative pose (nothing jumps).
    /sim/detach_flap   (std_srvs/Trigger)
        Deletes the weld; the flap stays where it is (hinge drive is pure damping).
    /sim/set_kinematic_pose   (simulation_interfaces/SetEntityState, twist ignored)
        Moves a KINEMATIC rigid body (the box Body) while simulating. Isaac's /set_entity_state
        cannot: its RigidPrim.set_world_poses() goes through the tensor API's set_transforms(),
        which PhysX ignores for kinematic bodies (it still answers RESULT_OK, and its
        set_velocities() call adds "PxRigidDynamic::setLinearVelocity: Body must be
        non-kinematic!"). This sets a kinematic target instead, waits two frames and reads the
        pose back; the response is RESULT_OPERATION_FAILED if the body did not get there.
        It also writes the prim's USD transform: PhysX never writes kinematic bodies back to
        USD/Fabric, so with the target alone the collider moves but the rendered box (and every
        camera image) stays where it was.

Box placement does NOT need custom code: this script also enables Isaac's
isaacsim.ros2.sim_control extension, which serves the standard simulation_interfaces
/set_entity_state, /get_entity_state, ... services. scripts/lid_open/lid_common.py moves
Body / LidFront / LidBack with /set_entity_state.

Why a weld instead of a real friction grasp: same reason as ~/openvla_poc/poc_common.py
(attach_flap_to_gripper) - pinching a 5 mm cardboard flap with physics friction is unstable in
sim and not worth tuning before the data deadline. Known shortcut; must be replaced before any
real-hardware use.

The services are registered on the sim_control extension's ROS2ServiceManager node, so they run
on the same executor/thread as Isaac's own /set_entity_state handlers (which also touch USD
directly from the callback). Re-running this script is safe: it unregisters its old services
first.

Press Play before calling the services - RigidPrim poses are only valid while simulating.
"""
import numpy as np
import omni.kit.app
import warp as wp
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

FLAP_PATHS = ("/World/CardboardBox/LidBack", "/World/CardboardBox/LidFront")
WELD_PATH = "/World/CardboardBox/GripperFlapWeld"
# Welded body on each arm. The commanded finger (left_carriage) - the gripper controller only
# drives this joint, see dual_arm_ros_controllers.yaml.
FINGER_LINK_NAME = "xeg32_left_carriage_link"
SERVICE_NAMES = ("/sim/attach_flap", "/sim/detach_flap", "/sim/set_kinematic_pose")

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.sim_control", True)

from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.ros2.sim_control import ROS2ServiceManager  # noqa: E402
from simulation_interfaces.msg import Result  # noqa: E402
from simulation_interfaces.srv import SetEntityState  # noqa: E402
from std_srvs.srv import SetBool, Trigger  # noqa: E402


def find_finger_body(arm: str) -> str:
    """Rigid-body prim of the arm's commanded finger. The USD nests visual Xforms with the same
    name inside the link, so only prims that actually carry PhysicsRigidBodyAPI count."""
    stage = omni.usd.get_context().get_stage()
    hits = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.GetName().endswith(FINGER_LINK_NAME)
        and f"/{arm}/" in prim.GetPath().pathString
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one rigid {FINGER_LINK_NAME} under /{arm}/, found {hits}")
    return hits[0]


def world_pose(path: str):
    positions, orientations = RigidPrim(paths=path).get_world_poses()
    pos = np.asarray(positions.numpy() if hasattr(positions, "numpy") else positions).reshape(-1)[:3]
    quat_wxyz = np.asarray(orientations.numpy() if hasattr(orientations, "numpy") else orientations).reshape(-1)[:4]
    return pos, quat_wxyz


def attach(arm: str) -> str:
    stage = omni.usd.get_context().get_stage()
    finger_path = find_finger_body(arm)
    finger_pos, finger_q = world_pose(finger_path)
    # weld the nearest flap - the caller only says which arm
    flap_path, flap_pos, flap_q = min(
        ((p, *world_pose(p)) for p in FLAP_PATHS), key=lambda t: np.linalg.norm(t[1] - finger_pos))

    # Joint frame = flap origin/orientation. body1 (flap) side is identity, so the flap's scale
    # (it is a scaled Cube) never enters; body0 (finger, unscaled) side gets the flap pose
    # expressed in the finger frame.
    finger_rot = Gf.Rotation(Gf.Quatd(*finger_q.tolist()))
    flap_rot = Gf.Rotation(Gf.Quatd(*flap_q.tolist()))
    inv_finger = finger_rot.GetInverse()
    local_pos0 = inv_finger.TransformDir(Gf.Vec3d(*(flap_pos - finger_pos).tolist()))
    local_rot0 = (flap_rot * inv_finger).GetQuat()  # Gf composes left-to-right: flap then inverse finger

    stage.RemovePrim(WELD_PATH)
    joint = UsdPhysics.FixedJoint.Define(stage, WELD_PATH)
    joint.CreateBody0Rel().SetTargets([finger_path])
    joint.CreateBody1Rel().SetTargets([flap_path])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(local_rot0.GetReal(), Gf.Vec3f(local_rot0.GetImaginary())))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # The finger is an articulation link, the flap is not: keep the weld out of the
    # articulation's reduced-coordinate tree, otherwise PhysX tries to merge the flap into it.
    joint.CreateExcludeFromArticulationAttr(True)
    return f"{finger_path} -> {flap_path}"


async def handle_attach(request, response):
    arm = "arm_right" if request.data else "arm_left"
    try:
        response.message = f"welded {attach(arm)}"
        response.success = True
    except Exception as e:  # report over ROS instead of killing the executor thread
        response.success = False
        response.message = f"attach failed: {e}"
    print(f"[sim_flap_weld] {response.message}")
    return response


async def handle_detach(request, response):
    stage = omni.usd.get_context().get_stage()
    existed = stage.GetPrimAtPath(WELD_PATH).IsValid()
    stage.RemovePrim(WELD_PATH)
    response.success = True
    response.message = "weld removed" if existed else "no weld to remove"
    print(f"[sim_flap_weld] {response.message}")
    return response


_kinematic_views = {}  # path -> (simulation view it came from, rigid body view), reused between calls


def set_usd_world_pose(path: str, pos, quat_wxyz):
    """Author the prim's local transform so that its world pose is (pos, quat), as xformOp:translate +
    xformOp:orient (+ scale if authored). Kept as translate/orient on purpose: sim_scene_control.py
    resizes the box by editing Body's xformOp:translate, which a matrix op would silently override.
    Existing attributes keep their precision (Body's translate is double3; RigidPrim may have added
    a float orient)."""
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    world = Gf.Matrix4d()
    world.SetTransform(Gf.Rotation(Gf.Quatd(*quat_wxyz)), Gf.Vec3d(*pos))
    local = world * xf.ComputeParentToWorldTransform(Usd.TimeCode.Default()).GetInverse()  # USD row vectors
    t, q = local.ExtractTranslation(), local.ExtractRotationQuat()

    t_attr = prim.GetAttribute("xformOp:translate") or prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
    t_attr.Set(Gf.Vec3d(t) if t_attr.GetTypeName() == Sdf.ValueTypeNames.Double3 else Gf.Vec3f(t))
    o_attr = prim.GetAttribute("xformOp:orient") or prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd)
    o_attr.Set(Gf.Quatd(q) if o_attr.GetTypeName() == Sdf.ValueTypeNames.Quatd else Gf.Quatf(q))
    ops = [UsdGeom.XformOp(t_attr), UsdGeom.XformOp(o_attr)]
    s_attr = prim.GetAttribute("xformOp:scale")
    if s_attr and s_attr.HasAuthoredValue():
        ops.append(UsdGeom.XformOp(s_attr))
    xf.SetXformOpOrder(ops)


def kinematic_view(path: str, fresh: bool = False):
    """Rigid body view of one kinematic prim, made from the simulation view Isaac's SimulationManager
    already created at Play (warp frontend; a fresh tensors.create_simulation_view() from here fails).
    Stop/Play makes SimulationManager create a new simulation view and invalidates the old one together
    with every view made from it; using those logs "Simulation view object is invalidated and cannot be
    used again". So the cache is keyed on the current simulation view object, never on view.check()."""
    sim_view = SimulationManager.get_physics_simulation_view()
    if sim_view is None:
        raise RuntimeError("physics is not initialized (press Play)")
    cached = _kinematic_views.get(path)
    if not fresh and cached is not None and cached[0] is sim_view:
        return cached[1]
    view = sim_view.create_rigid_body_view(path)
    if view.count != 1:
        raise RuntimeError(f"{path} is not a single rigid body (view count {view.count})")
    _kinematic_views[path] = (sim_view, view)
    return view


async def handle_set_kinematic_pose(request, response):
    try:
        p, q = request.state.pose.position, request.state.pose.orientation
        target = np.array([[p.x, p.y, p.z, q.x, q.y, q.z, q.w]], dtype=np.float32)
        set_usd_world_pose(request.entity, (p.x, p.y, p.z), (q.w, q.x, q.y, q.z))
        device = SimulationManager.get_device()
        targets = wp.array(target, dtype=wp.float32, device=device)
        indices = wp.array([0], dtype=wp.uint32, device=device)
        view = kinematic_view(request.entity)
        try:
            view.set_kinematic_targets(targets, indices)
        except Exception:
            # sim_scene_control.py 改尺寸（改了 Body 底下碰撞體的 scale）後 PhysX 會重建這個剛體，
            # simulation view 還是同一個，但舊的 rigid body view 已經失效（"Failed to set rigid body
            # kinematic targets in backend"）：重建一次再試
            view = kinematic_view(request.entity, fresh=True)
            view.set_kinematic_targets(targets, indices)
        for _ in range(2):  # the target is applied on the next physics step
            await omni.kit.app.get_app().next_update_async()
        got = view.get_transforms().numpy().reshape(-1)[:3]
        err = float(np.linalg.norm(got - target[0, :3]))
        if err > 0.005:
            raise RuntimeError(f"body is at {got.round(4).tolist()}, {err:.3f} m from the target (timeline playing?)")
        response.result = Result(result=Result.RESULT_OK, error_message=f"kinematic target set for {request.entity}")
    except Exception as e:  # report over ROS instead of killing the executor thread
        response.result = Result(result=Result.RESULT_OPERATION_FAILED, error_message=f"set pose failed: {e}")
        print(f"[sim_flap_weld] {response.result.error_message}")
    return response


manager = ROS2ServiceManager()
for name in SERVICE_NAMES:
    if name in manager.services:
        manager.unregister_service(name)
manager.register_service("/sim/attach_flap", SetBool, handle_attach)
manager.register_service("/sim/detach_flap", Trigger, handle_detach)
manager.register_service("/sim/set_kinematic_pose", SetEntityState, handle_set_kinematic_pose)
print("[sim_flap_weld] serving /sim/attach_flap (SetBool: true=arm_right), /sim/detach_flap, "
      "/sim/set_kinematic_pose; "
      "isaacsim.ros2.sim_control enabled (/set_entity_state etc.)")

