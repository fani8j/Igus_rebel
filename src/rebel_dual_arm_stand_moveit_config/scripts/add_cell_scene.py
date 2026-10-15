#!/usr/bin/env python3
"""Add the dual-arm cell's carton and carton table to move_group's planning scene (collision boxes).

DERIVED FROM compal_robot scripts/add_scene_boxes.py @ bb76327 (MIT, Copyright (c) 2026 James, see ../NOTICE): the /apply_planning_scene
call pattern and the explicit `objects:` list format are kept; the layout -> world transform (config/cell_scene.yaml) is new.

Usage
  ros2 run rebel_dual_arm_stand_moveit_config add_cell_scene.py --ros-args -p scene_file:=/path/cell_scene.yaml [-p lift_ref:=-0.2] [-p service_timeout:=60.0]
  python3 add_cell_scene.py --print [--scene-file f] [--lift-ref x]       # no ROS: print the objects in the world frame as JSON

The yaml either holds an explicit `objects:` list (compal_robot format, frame_id + boxes with size/position/orientation) or, as shipped, a
W-frame `layout:` plus the `w_frame:` transform to the URDF world frame (derivation in the yaml header).  The planning scene diff is sent once
(waits for /apply_planning_scene) and the node exits; move_group keeps the objects.
"""
import json
import math
import os
import sys

import numpy as np
import yaml


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_z(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def default_scene_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('rebel_dual_arm_stand_moveit_config'), 'config', 'cell_scene.yaml')
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'cell_scene.yaml')


def layout_to_objects(spec, lift_ref=None):
    """W-frame layout -> list of {id, color, boxes:[{name,size,position,orientation}]} in the world frame"""
    wf = spec['w_frame']
    lift = wf['lift_ref'] if lift_ref is None else lift_ref
    O = np.array([wf['origin_xy'][0], wf['origin_xy'][1], wf['mount_plane_z_at_lift0'] + lift])
    Rw = Rz(math.radians(wf['yaw_deg']))              # W axes -> world axes
    yaw_w = math.radians(wf['yaw_deg'])

    def to_world(p):
        return (O + Rw @ np.asarray(p, float)).tolist()

    objs = []
    lay = spec['layout']
    c = lay['carton']
    A, B, H = c['size_AxBxH']
    psi = math.radians(c['yaw_deg'])
    phi = psi + math.pi / 2                           # cellgeom.carton_geometry: box frame = Rz(psi + 90 deg), size [A, B, H]
    pw = [c['centre_xy'][0], c['centre_xy'][1], -c['h'] - H / 2]
    objs.append(dict(id=c['id'], color=c['rgba'],
                     boxes=[dict(name='box', size=[A, B, H], position=to_world(pw), orientation=quat_z(phi + yaw_w))]))
    t = lay.get('carton_table')
    if t:
        top_w = -c['h'] - H                           # table plane = carton bottom
        top_world = to_world([0, 0, top_w])[2]
        thick = t.get('thickness')
        if thick is None:
            thick = top_world - t['floor_z_world']
        sx, sy = t['depth'], 2 * t['half_width']
        cw = [t['x_front'] + sx / 2, 0.0, top_w - thick / 2]
        objs.append(dict(id=t['id'], color=t['rgba'],
                         boxes=[dict(name='top', size=[sx, sy, thick], position=to_world(cw), orientation=quat_z(yaw_w))]))
    return objs


def load_objects(path, lift_ref=None):
    spec = yaml.safe_load(open(path))
    frame = spec.get('frame_id', 'world')
    if 'layout' in spec:
        objs = layout_to_objects(spec, lift_ref)
    else:
        objs = spec['objects']
    return frame, objs


def build_scene(frame, objs):
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
    from shape_msgs.msg import SolidPrimitive
    from std_msgs.msg import ColorRGBA

    def make_pose(position, orientation=(0.0, 0.0, 0.0, 1.0)):
        p = Pose()
        p.position.x, p.position.y, p.position.z = (float(v) for v in position)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in orientation)
        return p

    scene = PlanningScene(is_diff=True)
    for obj in objs:
        co = CollisionObject(id=obj['id'], operation=CollisionObject.ADD)
        co.header.frame_id = frame
        co.pose = make_pose((0.0, 0.0, 0.0))
        for box in obj['boxes']:
            co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in box['size']]))
            co.primitive_poses.append(make_pose(box['position'], box.get('orientation', (0.0, 0.0, 0.0, 1.0))))
        scene.world.collision_objects.append(co)
        if 'color' in obj:
            r, g, b, a = (float(v) for v in obj['color'])
            scene.object_colors.append(ObjectColor(id=obj['id'], color=ColorRGBA(r=r, g=g, b=b, a=a)))
    return scene


def main_ros():
    import rclpy
    from moveit_msgs.srv import ApplyPlanningScene
    from rclpy.node import Node
    rclpy.init()
    node = Node('add_cell_scene')
    scene_file = node.declare_parameter('scene_file', '').value or default_scene_file()
    timeout = node.declare_parameter('service_timeout', 60.0).value
    lift_ref = node.declare_parameter('lift_ref', float('nan')).value
    frame, objs = load_objects(scene_file, None if math.isnan(lift_ref) else lift_ref)
    scene = build_scene(frame, objs)
    client = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
    if not client.wait_for_service(timeout_sec=timeout):
        node.get_logger().error('/apply_planning_scene not available after %.0fs (is move_group running?)' % timeout)
        return 1
    future = client.call_async(ApplyPlanningScene.Request(scene=scene))
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    ids = [co.id for co in scene.world.collision_objects]
    if future.result() is None or not future.result().success:
        node.get_logger().error('applying %s from %s failed' % (ids, scene_file))
        return 1
    node.get_logger().info('added %s from %s to the planning scene' % (ids, scene_file))
    node.destroy_node()
    rclpy.shutdown()
    return 0


def main():
    if '--print' in sys.argv:
        argv = sys.argv
        f = argv[argv.index('--scene-file') + 1] if '--scene-file' in argv else default_scene_file()
        lr = float(argv[argv.index('--lift-ref') + 1]) if '--lift-ref' in argv else None
        frame, objs = load_objects(f, lr)
        print(json.dumps(dict(frame_id=frame, scene_file=f, lift_ref=lr, objects=objs), indent=1))
        return 0
    return main_ros()


if __name__ == '__main__':
    sys.exit(main())
