"""Run this INSIDE Isaac Sim (Window > Script Editor) after dual_arm_rig.usda is open and
dual_arm_rig_ros2_bridge.py + sim_flap_weld.py have been run, BEFORE pressing Play. It applies the
box size + table in scripts/dual_arm_gui/scene.yaml (written by the GUI's 儲存 buttons) to the USD.

Only while the timeline is STOPPED. Tested 2026-10-04: editing these prims while simulating (even
with the timeline paused around the edit) leaves the bridge's box TF frozen at stale values, PhysX
does not re-read the lid hinge joints' localPos (lids get pulled open ~9 deg when width/height
change) and the kinematic Body's tensor view goes stale. So changing the box size means: save in
the GUI -> stop the launch -> File > Open dual_arm_rig.usda again -> run the three scripts -> Play
-> start the launch (scripts/lid_open/README.md, "改紙箱尺寸").

Services (std_srvs/Trigger; they refuse with success=false unless the timeline is stopped). The GUI
no longer calls them; /sim/apply_box_size's presence is what tells lid_common / the GUI that this
script has run:

    /sim/apply_box_size        box length/width/height/wall/lid/gap <- scene.yaml
    /sim/apply_table           table top height + horizontal position <- scene.yaml
    /sim/restore_default_scene box size + table back to the USD's original numbers

Prims are edited IN PLACE (translate / scale / joint localPos of the existing prims). Geometry
formulas live in scripts/dual_arm_gui/scene_config.py (shared with lid_open, the GUI's MoveIt
collision objects and RViz markers). Edits are made only in memory - nothing is saved back to
dual_arm_rig.usda. Box POSE is not handled here (lid_open's `r` / GUI 移動紙箱 do that).
Re-running this script is safe: it unregisters its old services first.
"""
import asyncio
import importlib
import os
import types
import sys
from pathlib import Path

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf

REPO_ROOT = Path(os.environ.get("COMPAL_ROBOT_ROOT", Path.home() / "compal_robot"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dual_arm_gui"))
import scene_config  # noqa: E402

importlib.reload(scene_config)   # 重跑這支時拿到最新的 scene_config（Script Editor 會沿用舊的 module）

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.sim_control", True)

from isaacsim.ros2.sim_control import ROS2ServiceManager  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

BOX_ROOT = "/World/CardboardBox"
TABLE_ROOT = "/World/table"
SERVICE_NAMES = ("/sim/apply_box_size", "/sim/apply_table", "/sim/restore_default_scene")


def _set_vec3(path, attr_name, value):
    """依屬性原本的型別（float3 / double3 / point3f）寫入，USD 型別不符會直接被拒絕。"""
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"找不到 prim {path}（USD 是不是 dual_arm_rig.usda？紙箱是否還在？）")
    attr = prim.GetAttribute(attr_name)
    if not attr.IsValid():
        raise RuntimeError(f"{path} 沒有屬性 {attr_name}")
    cls = Gf.Vec3d if "double" in str(attr.GetTypeName()) else Gf.Vec3f
    attr.Set(cls(*(float(v) for v in value)))


def _get_vec3(path, attr_name):
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    return tuple(prim.GetAttribute(attr_name).Get())


def apply_box_size(box):
    # Body 在 CardboardBox 座標系下：x/y 不動（USD 是 (0.4, 0)），z = 高度一半 -> 底面貼齊桌面
    bx, by, _ = _get_vec3(f"{BOX_ROOT}/Body", "xformOp:translate")
    _set_vec3(f"{BOX_ROOT}/Body", "xformOp:translate", (bx, by, box["height"] / 2))
    for name, (center, size) in scene_config.box_walls(box).items():
        _set_vec3(f"{BOX_ROOT}/Body/{name}", "xformOp:translate", center)
        _set_vec3(f"{BOX_ROOT}/Body/{name}", "xformOp:scale", size)
    for name, (center, size) in scene_config.box_lids(box).items():
        # 蓋子是 CardboardBox 的直接子節點（不在 Body 底下），朝向與 Body 相同 -> 直接平移
        _set_vec3(f"{BOX_ROOT}/{name}", "xformOp:translate",
                  (bx + center[0], by + center[1], box["height"] / 2 + center[2]))
        _set_vec3(f"{BOX_ROOT}/{name}", "xformOp:scale", size)
        # localPos0 在 Body（無 scale）座標系 = 公尺；localPos1 在蓋子（有 scale）座標系 = 分率，不用改
        _set_vec3(f"{BOX_ROOT}/{name}Joint", "physics:localPos0", scene_config.lid_hinge(box, name))
    return (f"box {box['length']:.3f} x {box['width']:.3f} x {box['height']:.3f} "
            f"(wall {box['wall']:.3f}, lid {box['lid']:.3f}, gap {box['gap']:.3f})")


def apply_table(table):
    top_z = table["top_z"]
    for name, (center, size) in scene_config.table_parts(table).items():
        _set_vec3(f"{TABLE_ROOT}/{name}", "xformOp:translate", center)
        _set_vec3(f"{TABLE_ROOT}/{name}", "xformOp:scale", size)
    # 紙箱根節點跟著桌面走（USD 原本就是 z = 桌面頂 0.73）
    cx, cy, _ = _get_vec3(BOX_ROOT, "xformOp:translate")
    _set_vec3(BOX_ROOT, "xformOp:translate", (cx, cy, top_z))
    return f"table top z = {top_z:.3f}, center = ({table['x']:.3f}, {table['y']:.3f})"


async def _edit(edit_fn, response):
    """改 prim，只在 timeline 停止時做（見 module docstring）。"""
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        response.success = False
        response.message = ("refused: timeline is playing/paused. Editing the box/table while simulating breaks "
                            "the box TF and the lid hinges. File > Open dual_arm_rig.usda again, run the bridge, "
                            "sim_flap_weld.py and this script, then press Play.")
    else:
        try:
            response.message = edit_fn()
            response.success = True
        except Exception as e:  # report over ROS instead of killing the executor thread
            response.success = False
            response.message = f"failed: {e}"
    print(f"[sim_scene_control] {response.message}")
    return response


async def handle_apply_box_size(request, response):
    return await _edit(lambda: apply_box_size(scene_config.load()["box"]), response)


async def handle_apply_table(request, response):
    return await _edit(lambda: apply_table(scene_config.load()["table"]), response)


async def handle_restore_default(request, response):
    d = scene_config.DEFAULTS
    return await _edit(
        lambda: f"{apply_box_size(d['box'])}; {apply_table(d['table'])} (USD defaults)", response)


def apply_scene_file():
    """跑這支腳本時把 scene.yaml（GUI 最後一次的設定）套到 USD。"""
    cfg = scene_config.load()
    response = types.SimpleNamespace(success=False, message="")
    note = "" if scene_config.is_default(cfg) else "  <- not the USD default (lid_open uses it too)"
    return _edit(lambda: f"{apply_box_size(cfg['box'])}; {apply_table(cfg['table'])} (from scene.yaml){note}",
                 response)


def check_rig():
    """scene.yaml 的 rig: 要跟目前開的 USD 同一版（GUI / lid_open 的 launch 依 rig 選 xacro），
    否則錄到的資料是「Isaac 一版、MoveIt 另一版」。v2 USD 才有 /World/ch_v4_joints。"""
    want = scene_config.load()["rig"]
    stage = omni.usd.get_context().get_stage()
    have = "v2" if stage.GetPrimAtPath("/World/ch_v4_joints").IsValid() else "v1"
    if want != have:
        raise RuntimeError(f"scene.yaml 是 rig: {want}，但 Isaac 開的是 {have} 的 USD"
                           f"（{scene_config.RIGS[have]['usda'].name}）。請開 {scene_config.RIGS[want]['usda'].name}，"
                           "或在 GUI 把機架切成一致後再跑這支")
    print(f"[sim_scene_control] rig {have} matches scene.yaml")


check_rig()
manager = ROS2ServiceManager()
for name in SERVICE_NAMES:
    if name in manager.services:
        manager.unregister_service(name)
manager.register_service("/sim/apply_box_size", Trigger, handle_apply_box_size)
manager.register_service("/sim/apply_table", Trigger, handle_apply_table)
manager.register_service("/sim/restore_default_scene", Trigger, handle_restore_default)
print(f"[sim_scene_control] serving {', '.join(SERVICE_NAMES)}; scene file {scene_config.SCENE_FILE}")
asyncio.ensure_future(apply_scene_file())
