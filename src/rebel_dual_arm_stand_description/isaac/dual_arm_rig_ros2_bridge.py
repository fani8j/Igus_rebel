"""Run this INSIDE Isaac Sim (Window > Script Editor, or as a startup script) after
dual_arm_rig.usda is open and the stage has finished loading. It wires up the ROS2 Bridge
OmniGraph so Isaac Sim publishes joint states and accepts joint commands over the exact same
topic names dual_arm_rig.control.xacro's isaac_topic hardware_protocol uses:

    /dual_arm/isaac_joint_commands   (subscribed  -> drives the articulation)
    /dual_arm/isaac_joint_states     (published    <- articulation's current joint state)

Also publishes the gemini336L camera (see dual_arm_rig_README.md / dual_arm_rig.urdf.xacro for
the modeled mount geometry) so compal_box_perception's box_perception_node has something to
subscribe to, on the topics dual_arm_box_perception.yaml expects:

    /dual_arm/gemini336L/rgb/image_raw     (sensor_msgs/Image, frame_id gemini336L_rgb_optical_frame)
    /dual_arm/gemini336L/rgb/camera_info   (sensor_msgs/CameraInfo)
    /dual_arm/gemini336L/depth/image_raw   (sensor_msgs/Image, frame_id gemini336L_depth_optical_frame)

and both wrist cameras (USD Camera "wrist_camera" under each arm's xeg32_camera_mount_link), RGB
only - the VLA policy input is top + wrist_left + wrist_right RGB:

    /dual_arm/wrist_left/rgb/image_raw     (sensor_msgs/Image, frame_id arm_left_wrist_camera_optical_frame)
    /dual_arm/wrist_left/rgb/camera_info   (sensor_msgs/CameraInfo)
    /dual_arm/wrist_right/rgb/image_raw    (sensor_msgs/Image, frame_id arm_right_wrist_camera_optical_frame)
    /dual_arm/wrist_right/rgb/camera_info  (sensor_msgs/CameraInfo)

and the cardboard box's TF (Body, LidBack, LidFront, parent "world"), which
scripts/cardboard_box_markers.py hangs its RViz markers on - the single-arm USD has this
ROS2PublishTransformTree baked in, the dual-arm USD does not:

    /tf   world -> Body / LidBack / LidFront   (frame names = USD prim names)

parentPrim is left empty so Isaac reports the poses in stage-world coordinates under the frame
"world". That equals the URDF's "world" link only because dual_arm_rig and its stand sit at
the stage origin (root_joint has zero offsets) - if the rig is referenced in at an offset, the
box TF will be off by that offset.

The wrist camera prims are found by searching the stage for Camera prims named "wrist_camera"
whose path contains /arm_left/ or /arm_right/ (they sit ~15 levels deep under the arm, so the
path is not hardcoded). Their optical frame links are defined in dual_arm_rig.urdf.xacro and
ride on camera_tilt_joint, so TF follows the simulated tilt.

The camera prims are found relative to the articulation root, at
<articulation_root>/gemini336L_camera_link/gemini336L_{rgb,depth} - this matches the USD's own
customLayerData.cameraSettings.boundCamera default
(/World/dual_arm_rig/stand/gemini336L_camera_link/gemini336L_rgb), so it should hold regardless
of how the file was loaded/referenced. The rgb/depth optical frame_ids above must match the
"gemini336L_rgb_optical_frame"/"gemini336L_depth_optical_frame" links added to
dual_arm_rig.urdf.xacro - robot_state_publisher publishes their TF (they are fixed relative to
"stand"), Isaac only needs to publish the images/camera_info themselves.

dual_arm_rig_README.md's own "Isaac Sim usage" section only shows how to add the rig to the
stage (add_reference_to_stage) -- it does NOT set up this ROS2 bridge, so this script fills
that gap. It does not add the rig to the stage itself; run it after the rig is already there
(either because you opened dual_arm_rig.usda directly, or referenced it in with
add_reference_to_stage as the README shows).

IMPORTANT / not verified end-to-end in this Isaac Sim install:
  Node type names and input/output pin names below (isaacsim.ros2.bridge.*,
  isaacsim.core.nodes.*) match NVIDIA's standard ROS2-joint-state OmniGraph tutorial for the
  "isaacsim.*" extension naming used since Isaac Sim 4.x / 2023.1.1. If your Isaac Sim build is
  older it may still use the "omni.isaac.*" extension names instead (e.g.
  omni.isaac.ros2_bridge.ROS2PublishJointState, omni.isaac.core_nodes.IsaacArticulationController)
  -- swap the strings in NODE_TYPES below if og.Controller.edit() raises an unknown-node-type
  error. Likewise if a SET_VALUES call below errors with an unknown attribute, open that node in
  Window > Visualize > Action Graph and check its Property panel for the current pin name
  (common drift: targetPrim vs robotPath, topicName vs "Topic Name").

  The camera nodes (IsaacCreateRenderProduct -> ROS2CameraHelper, head + both wrists) are EVEN LESS verified than
  the joint-state graph above: they follow NVIDIA's standard ROS2 camera OmniGraph tutorial
  shape, but were never run against this project's actual Isaac Sim install. Likely drift
  points: inputs:type token spelling ("rgb"/"depth" vs "rgb"/"depth_pcl"/...),
  and whether ROS2CameraHelper wants inputs:renderProductPath as a single string or an array.
  camera_info is published by the separate ROS2CameraInfoHelper node (sharing the rgb render
  product): Isaac Sim 6.1's ROS2CameraHelper no longer accepts type "camera_info" and logs
  "Sensor type 'camera_info' is not supported" every tick if you try. Check the debug_image
  topic / `ros2 topic hz` on the topics above before trusting that box_perception_node is
  actually receiving frames.

You must press Play in Isaac Sim for anything here to actually execute (OnPlaybackTick only
fires during simulation, and root_joint / lift_joint / etc. only move under physics stepping).
"""
import omni.graph.core as og
from pxr import UsdPhysics
import omni.kit.commands
import omni.usd


JOINT_COMMANDS_TOPIC = "/dual_arm/isaac_joint_commands"
JOINT_STATES_TOPIC = "/dual_arm/isaac_joint_states"
GRAPH_PATH = "/World/DualArmRigROS2Bridge"

NODE_TYPES = {
    "OnTick": "omni.graph.action.OnPlaybackTick",
    "Context": "isaacsim.ros2.bridge.ROS2Context",
    "PublishClock": "isaacsim.ros2.bridge.ROS2PublishClock",
    "PublishJointState": "isaacsim.ros2.bridge.ROS2PublishJointState",
    "SubscribeJointState": "isaacsim.ros2.bridge.ROS2SubscribeJointState",
    "ArticulationController": "isaacsim.core.nodes.IsaacArticulationController",
    "ReadSimTime": "isaacsim.core.nodes.IsaacReadSimulationTime",
    "RgbRenderProduct": "isaacsim.core.nodes.IsaacCreateRenderProduct",
    "DepthRenderProduct": "isaacsim.core.nodes.IsaacCreateRenderProduct",
    "RgbImageHelper": "isaacsim.ros2.bridge.ROS2CameraHelper",
    "RgbInfoHelper": "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
    "DepthImageHelper": "isaacsim.ros2.bridge.ROS2CameraHelper",
    "WristLRenderProduct": "isaacsim.core.nodes.IsaacCreateRenderProduct",
    "WristLImageHelper": "isaacsim.ros2.bridge.ROS2CameraHelper",
    "WristLInfoHelper": "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
    "WristRRenderProduct": "isaacsim.core.nodes.IsaacCreateRenderProduct",
    "WristRImageHelper": "isaacsim.ros2.bridge.ROS2CameraHelper",
    "WristRInfoHelper": "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
}
# Only created when the stage has a CardboardBox (see find_cardboard_box_prims()).
BOX_TF_NODE_TYPE = "isaacsim.ros2.bridge.ROS2PublishTransformTree"

RGB_IMAGE_TOPIC = "/dual_arm/gemini336L/rgb/image_raw"
RGB_INFO_TOPIC = "/dual_arm/gemini336L/rgb/camera_info"
DEPTH_IMAGE_TOPIC = "/dual_arm/gemini336L/depth/image_raw"
RGB_OPTICAL_FRAME = "gemini336L_rgb_optical_frame"
DEPTH_OPTICAL_FRAME = "gemini336L_depth_optical_frame"
# Head camera: rgb and depth render at the SAME resolution. Together with the USD giving
# gemini336L_depth the same pose and intrinsics as gemini336L_rgb, this makes depth
# pixel-registered to RGB (like the real driver's align_target_stream:=COLOR), which
# box_perception assumes - it reads depth pixels with the RGB camera_info.
# 1280x800 is the real Gemini 336L resolution and matches the USD aperture ratio (36 : 22.64).
HEAD_WIDTH = 1280
HEAD_HEIGHT = 800

# Wrist cameras (real hardware: Orbbec Gemini 305, color stream only): keyed by the node-name
# prefix used in NODE_TYPES ("WristL"/"WristR"). The USD wrist_camera intrinsics model the
# Gemini 305 color FOV (H 94 x V 68 deg, 16:10 sensor). 848x530 is one of its native 16:10
# color modes (up to 60 fps) and keeps the render cost of two more cameras down; switch to
# 1280x800 (its full resolution) here if the real recordings use that.
CARDBOARD_BOX_PRIM_NAME = "CardboardBox"
# Frame names scripts/cardboard_box_markers.py expects (Isaac uses the prim name as the frame id).
CARDBOARD_BOX_PARTS = ("Body", "LidBack", "LidFront")

WRIST_CAMERA_PRIM_NAME = "wrist_camera"
WRIST_WIDTH = 848
WRIST_HEIGHT = 530
WRIST_CAMERAS = {
    "WristL": {
        "arm": "arm_left",
        "image_topic": "/dual_arm/wrist_left/rgb/image_raw",
        "info_topic": "/dual_arm/wrist_left/rgb/camera_info",
        "frame_id": "arm_left_wrist_camera_optical_frame",
    },
    "WristR": {
        "arm": "arm_right",
        "image_topic": "/dual_arm/wrist_right/rgb/image_raw",
        "info_topic": "/dual_arm/wrist_right/rgb/camera_info",
        "frame_id": "arm_right_wrist_camera_optical_frame",
    },
}


def find_articulation_root() -> str:
    """First prim in the open stage with PhysicsArticulationRootAPI applied.
    Per dual_arm_rig_README.md this is .../dual_arm_rig/stand (PhysicsArticulationRootAPI is
    applied there, with root_joint fixing it to its parent) -- found by search rather than
    hardcoded, since the exact absolute path depends on where/how you loaded the usda."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No stage is open in Isaac Sim yet.")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    raise RuntimeError(
        "No prim with PhysicsArticulationRootAPI found on the current stage. "
        "Make sure dual_arm_rig.usda is open (or referenced in) and fully loaded first."
    )


def find_wrist_cameras() -> dict:
    """{"arm_left": prim_path, "arm_right": prim_path} for the Camera prims named
    WRIST_CAMERA_PRIM_NAME. Searched rather than hardcoded for the same reason as
    find_articulation_root(); the arm is told apart by the /arm_left/ or /arm_right/ path segment."""
    stage = omni.usd.get_context().get_stage()
    found = {}
    for prim in stage.Traverse():
        if prim.GetName() != WRIST_CAMERA_PRIM_NAME or prim.GetTypeName() != "Camera":
            continue
        path = prim.GetPath().pathString
        for arm in ("arm_left", "arm_right"):
            if f"/{arm}/" in path:
                if arm in found:
                    raise RuntimeError(f"More than one {WRIST_CAMERA_PRIM_NAME} under {arm}: {found[arm]}, {path}")
                found[arm] = path
    missing = [arm for arm in ("arm_left", "arm_right") if arm not in found]
    if missing:
        raise RuntimeError(
            f"No Camera prim named {WRIST_CAMERA_PRIM_NAME!r} found under {missing}. "
            "Expected .../<arm>/.../xeg32_camera_mount_link/wrist_camera (dual_arm_rig_README.md)."
        )
    return found


def find_cardboard_box_prims() -> list:
    """[Body, LidBack, LidFront] prim paths under the first prim named CARDBOARD_BOX_PRIM_NAME,
    or [] if the stage has no box (the rest of the bridge still works without it)."""
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() != CARDBOARD_BOX_PRIM_NAME:
            continue
        paths = [prim.GetPath().AppendChild(part) for part in CARDBOARD_BOX_PARTS]
        missing = [str(p) for p in paths if not stage.GetPrimAtPath(p).IsValid()]
        if missing:
            print(f"[dual_arm_rig_ros2_bridge] WARNING: {prim.GetPath()} is missing {missing}, "
                  "not publishing box TF")
            return []
        return [p.pathString for p in paths]
    print(f"[dual_arm_rig_ros2_bridge] WARNING: no {CARDBOARD_BOX_PRIM_NAME} prim on the stage, "
          "not publishing box TF")
    return []


def build_bridge():
    articulation_root = find_articulation_root()
    print(f"[dual_arm_rig_ros2_bridge] articulation root = {articulation_root}")

    # Matches dual_arm_rig.urdf.xacro's gemini336L_camera_link mount (fixed to "stand") and the
    # USD's own customLayerData.cameraSettings.boundCamera default - see module docstring.
    rgb_camera_prim = f"{articulation_root}/gemini336L_camera_link/gemini336L_rgb"
    depth_camera_prim = f"{articulation_root}/gemini336L_camera_link/gemini336L_depth"
    wrist_camera_prims = find_wrist_cameras()
    for arm, path in wrist_camera_prims.items():
        print(f"[dual_arm_rig_ros2_bridge] {arm} wrist camera = {path}")
    box_prims = find_cardboard_box_prims()

    # Box TF: parentPrim left unset -> poses relative to stage world, parent frame "world".
    box_nodes, box_connect, box_values = [], [], []
    if box_prims:
        box_nodes = [("PublishBoxTF", BOX_TF_NODE_TYPE)]
        box_connect = [
            ("OnTick.outputs:tick", "PublishBoxTF.inputs:execIn"),
            ("Context.outputs:context", "PublishBoxTF.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "PublishBoxTF.inputs:timeStamp"),
        ]
        box_values = [("PublishBoxTF.inputs:targetPrims", box_prims)]

    # Same wiring as the gemini336L rgb pair (render product -> image helper + camera_info helper).
    wrist_connect = []
    wrist_values = []
    for n, cam in WRIST_CAMERAS.items():
        wrist_connect += [
            ("OnTick.outputs:tick", f"{n}RenderProduct.inputs:execIn"),
            ("Context.outputs:context", f"{n}ImageHelper.inputs:context"),
            ("Context.outputs:context", f"{n}InfoHelper.inputs:context"),
            (f"{n}RenderProduct.outputs:execOut", f"{n}ImageHelper.inputs:execIn"),
            (f"{n}RenderProduct.outputs:execOut", f"{n}InfoHelper.inputs:execIn"),
            (f"{n}RenderProduct.outputs:renderProductPath", f"{n}ImageHelper.inputs:renderProductPath"),
            (f"{n}RenderProduct.outputs:renderProductPath", f"{n}InfoHelper.inputs:renderProductPath"),
        ]
        wrist_values += [
            (f"{n}RenderProduct.inputs:cameraPrim", [wrist_camera_prims[cam["arm"]]]),
            (f"{n}RenderProduct.inputs:width", WRIST_WIDTH),
            (f"{n}RenderProduct.inputs:height", WRIST_HEIGHT),
            (f"{n}ImageHelper.inputs:type", "rgb"),
            (f"{n}ImageHelper.inputs:topicName", cam["image_topic"]),
            (f"{n}ImageHelper.inputs:frameId", cam["frame_id"]),
            (f"{n}InfoHelper.inputs:topicName", cam["info_topic"]),
            (f"{n}InfoHelper.inputs:frameId", cam["frame_id"]),
        ]

    # 重跑時 graph 已經在了，og.Controller.edit 會丟 "Failed to wrap graph in node"：先刪掉舊的再建
    # （例如紙箱 TF 卡住時重跑本腳本重建）
    if omni.usd.get_context().get_stage().GetPrimAtPath(GRAPH_PATH).IsValid():
        omni.kit.commands.execute("DeletePrims", paths=[GRAPH_PATH])
        print(f"[dual_arm_rig_ros2_bridge] removed existing graph at {GRAPH_PATH}")

    keys = og.Controller.Keys
    graph, nodes, _, _ = og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: list(NODE_TYPES.items()) + box_nodes,
            keys.CONNECT: [
                ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("OnTick.outputs:tick", "RgbRenderProduct.inputs:execIn"),
                ("OnTick.outputs:tick", "DepthRenderProduct.inputs:execIn"),
                ("Context.outputs:context", "RgbImageHelper.inputs:context"),
                ("Context.outputs:context", "RgbInfoHelper.inputs:context"),
                ("Context.outputs:context", "DepthImageHelper.inputs:context"),
                ("RgbRenderProduct.outputs:execOut", "RgbImageHelper.inputs:execIn"),
                ("RgbRenderProduct.outputs:execOut", "RgbInfoHelper.inputs:execIn"),
                ("DepthRenderProduct.outputs:execOut", "DepthImageHelper.inputs:execIn"),
                ("RgbRenderProduct.outputs:renderProductPath", "RgbImageHelper.inputs:renderProductPath"),
                ("RgbRenderProduct.outputs:renderProductPath", "RgbInfoHelper.inputs:renderProductPath"),
                ("DepthRenderProduct.outputs:renderProductPath", "DepthImageHelper.inputs:renderProductPath"),
            ] + wrist_connect + box_connect,
            keys.SET_VALUES: [
                ("PublishJointState.inputs:targetPrim", [articulation_root]),
                ("PublishJointState.inputs:topicName", JOINT_STATES_TOPIC),
                ("SubscribeJointState.inputs:topicName", JOINT_COMMANDS_TOPIC),
                ("ArticulationController.inputs:targetPrim", [articulation_root]),
                ("ArticulationController.inputs:robotPath", articulation_root),
                ("RgbRenderProduct.inputs:cameraPrim", [rgb_camera_prim]),
                ("DepthRenderProduct.inputs:cameraPrim", [depth_camera_prim]),
                ("RgbRenderProduct.inputs:width", HEAD_WIDTH),
                ("RgbRenderProduct.inputs:height", HEAD_HEIGHT),
                ("DepthRenderProduct.inputs:width", HEAD_WIDTH),
                ("DepthRenderProduct.inputs:height", HEAD_HEIGHT),
                ("RgbImageHelper.inputs:type", "rgb"),
                ("RgbImageHelper.inputs:topicName", RGB_IMAGE_TOPIC),
                ("RgbImageHelper.inputs:frameId", RGB_OPTICAL_FRAME),
                ("RgbInfoHelper.inputs:topicName", RGB_INFO_TOPIC),
                ("RgbInfoHelper.inputs:frameId", RGB_OPTICAL_FRAME),
                ("DepthImageHelper.inputs:type", "depth"),
                ("DepthImageHelper.inputs:topicName", DEPTH_IMAGE_TOPIC),
                ("DepthImageHelper.inputs:frameId", DEPTH_OPTICAL_FRAME),
            ] + wrist_values + box_values,
        },
    )
    print(f"[dual_arm_rig_ros2_bridge] graph created at {GRAPH_PATH}")
    print(f"  publishing  {JOINT_STATES_TOPIC}")
    print(f"  subscribing {JOINT_COMMANDS_TOPIC}")
    print(f"  publishing  {RGB_IMAGE_TOPIC} + {RGB_INFO_TOPIC} (frame {RGB_OPTICAL_FRAME})")
    print(f"  publishing  {DEPTH_IMAGE_TOPIC} (frame {DEPTH_OPTICAL_FRAME}, registered to rgb, "
          f"{HEAD_WIDTH}x{HEAD_HEIGHT})")
    for cam in WRIST_CAMERAS.values():
        print(f"  publishing  {cam['image_topic']} + {cam['info_topic']} "
              f"(frame {cam['frame_id']}, {WRIST_WIDTH}x{WRIST_HEIGHT})")
    if box_prims:
        print(f"  publishing  /tf world -> {', '.join(CARDBOARD_BOX_PARTS)} ({box_prims[0].rsplit('/', 2)[0]})")
    print("  press Play to start ticking the graph.")
    return graph, nodes


if __name__ == "__main__":
    build_bridge()
