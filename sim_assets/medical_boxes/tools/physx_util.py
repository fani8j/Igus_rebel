"""Raw PhysX attribute authoring/reading for usd-core (no PhysxSchema python module).

Attribute names and types are copied from Isaac Sim 5.1 omni.usd.schema.physx generatedSchema.usda.
The API names are added to apiSchemas with AddAppliedSchema; usd-core does not know them and
treats them as unknown (harmless) tokens, Isaac Sim resolves them.
"""
from pxr import Gf, Sdf, UsdPhysics, UsdShade

T = Sdf.ValueTypeNames


def _set(prim, name, vtype, value, uniform=False):
    prim.CreateAttribute(name, vtype, custom=False, variability=(
        Sdf.VariabilityUniform if uniform else Sdf.VariabilityVarying)).Set(value)


def author_material(prim, m):
    """UsdPhysics.MaterialAPI + PhysxMaterialAPI from a dict with static_friction,
    dynamic_friction, restitution, friction_combine_mode, restitution_combine_mode."""
    api = UsdPhysics.MaterialAPI.Apply(prim)
    api.CreateStaticFrictionAttr(float(m["static_friction"]))
    api.CreateDynamicFrictionAttr(float(m["dynamic_friction"]))
    api.CreateRestitutionAttr(float(m["restitution"]))
    prim.AddAppliedSchema("PhysxMaterialAPI")
    _set(prim, "physxMaterial:frictionCombineMode", T.Token, m["friction_combine_mode"], True)
    _set(prim, "physxMaterial:restitutionCombineMode", T.Token, m["restitution_combine_mode"], True)


def author_collision(prim, c):
    prim.AddAppliedSchema("PhysxCollisionAPI")
    _set(prim, "physxCollision:contactOffset", T.Float, float(c["contact_offset_m"]))
    _set(prim, "physxCollision:restOffset", T.Float, float(c["rest_offset_m"]))


def author_rigid_body(prim, r):
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    _set(prim, "physxRigidBody:solverPositionIterationCount", T.Int, int(r["solver_position_iterations"]))
    _set(prim, "physxRigidBody:solverVelocityIterationCount", T.Int, int(r["solver_velocity_iterations"]))


def author_scene(prim, s):
    prim.AddAppliedSchema("PhysxSceneAPI")
    _set(prim, "physxScene:solverType", T.Token, s["solver_type"], True)
    _set(prim, "physxScene:timeStepsPerSecond", T.UInt, int(s["time_steps_per_second"]))
    _set(prim, "physxScene:enableStabilization", T.Bool, bool(s["enable_stabilization"]))


def has_api(prim, name):
    # usd-core drops unknown schemas from GetAppliedSchemas(); read the authored metadata
    lo = prim.GetMetadata("apiSchemas")
    return bool(lo) and name in lo.GetAddedOrExplicitItems()


def get(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a else None


def check_material(prim, m, tol=1e-6):
    """-> (ok, detail) for a material prim against a config dict."""
    if not prim or not prim.HasAPI(UsdPhysics.MaterialAPI) or not has_api(prim, "PhysxMaterialAPI"):
        return False, "missing UsdPhysicsMaterialAPI/PhysxMaterialAPI"
    api = UsdPhysics.MaterialAPI(prim)
    got = (api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(),
           api.GetRestitutionAttr().Get())
    want = (m["static_friction"], m["dynamic_friction"], m["restitution"])
    modes = (get(prim, "physxMaterial:frictionCombineMode"), get(prim, "physxMaterial:restitutionCombineMode"))
    ok = (all(g is not None and abs(g - w) < tol for g, w in zip(got, want))
          and modes == (m["friction_combine_mode"], m["restitution_combine_mode"]))
    return ok, f"{prim.GetPath()} sf/df/e={tuple(round(v, 4) for v in got)} combine f/r={modes}"


def check_collision(mesh_prim, coll, tol=1e-7):
    mc = UsdPhysics.MeshCollisionAPI(mesh_prim)
    appr = mc.GetApproximationAttr().Get() if mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
    co, ro = get(mesh_prim, "physxCollision:contactOffset"), get(mesh_prim, "physxCollision:restOffset")
    ok = (mesh_prim.HasAPI(UsdPhysics.CollisionAPI) and has_api(mesh_prim, "PhysxCollisionAPI")
          and appr == coll["approximation"] and co is not None and ro is not None
          and abs(co - coll["contact_offset_m"]) < tol and abs(ro - coll["rest_offset_m"]) < tol)
    return ok, f"approx={appr} contactOffset={co} restOffset={ro}"


def check_rigid_body(root, r):
    p = get(root, "physxRigidBody:solverPositionIterationCount")
    v = get(root, "physxRigidBody:solverVelocityIterationCount")
    ok = (has_api(root, "PhysxRigidBodyAPI") and p == r["solver_position_iterations"]
          and v == r["solver_velocity_iterations"])
    return ok, f"position/velocity iterations={p}/{v}"


def check_scene(prim, s):
    got = (get(prim, "physxScene:solverType"), get(prim, "physxScene:timeStepsPerSecond"),
           get(prim, "physxScene:enableStabilization"))
    ok = (has_api(prim, "PhysxSceneAPI") and got == (
        s["solver_type"], s["time_steps_per_second"], bool(s["enable_stabilization"])))
    return ok, f"solver/steps/stabilization={got}"
