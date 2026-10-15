#!/usr/bin/env python3
"""MoveIt - medicine grasp: add the medicine shelf, its plinth and the packed cartons to move_group's planning scene (collision boxes).

Adapted from rebel_dual_arm_stand_moveit_config/scripts/add_cell_scene.py (itself derived from compal_robot scripts/add_scene_boxes.py, MIT, see ../NOTICE): the
/apply_planning_scene call pattern is kept, the shelf -> world transform (config/medicine_shelf_scene.yaml) is new.

Objects (ids): `medicine_shelf` (5 boards), `shelf_plinth`, `carton_0` .. `carton_4` (carton_2 = the target, box 3 "Terumo Accuforce").

Usage
  ros2 run rebel_medicine_grasp_moveit_config add_medicine_shelf.py --ros-args -p scene_file:=/path/medicine_shelf_scene.yaml [-p pull_out:=0.1] [-p service_timeout:=60.0]
  python3 add_medicine_shelf.py --print [--scene-file f] [--pull-out 0.1]       # no ROS: print the objects in the world frame as JSON
  python3 add_medicine_shelf.py --write-pose-json medicine_shelf_pose.json [--scene-file f]   # shelf pose for the Isaac scene
"""
import json
import math
import os
import sys

import numpy as np
import yaml

TARGET_ID = 'carton_2'


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_z(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def default_scene_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('rebel_medicine_grasp_moveit_config'), 'config', 'medicine_shelf_scene.yaml')
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'medicine_shelf_scene.yaml')


def carton_layout(spec):
    """[(index, box_id, centre_x_w, t, h, l)] in W (metres); the target carton is centred on x_w = 0, zero gap"""
    c = spec['cartons']
    seq = c['sequence']
    ti = c['target_index']
    t = [c['boxes_mm'][b]['t'] * 1e-3 for b in seq]
    xs = [0.0] * len(seq)
    for i in range(ti - 1, -1, -1):
        xs[i] = xs[i + 1] - t[i + 1] / 2 - t[i] / 2
    for i in range(ti + 1, len(seq)):
        xs[i] = xs[i - 1] + t[i - 1] / 2 + t[i] / 2
    out = []
    for i, b in enumerate(seq):
        bx = c['boxes_mm'][b]
        out.append((i, b, xs[i], bx['t'] * 1e-3, bx['h'] * 1e-3, bx['l'] * 1e-3))
    return out


def load_objects(path, pull_out=None, w_frame=None):
    """w_frame: optional {'origin': [x,y,z], 'yaw_deg': a} overriding the yaml (placement search / verification)"""
    spec = yaml.safe_load(open(path))
    if w_frame:
        spec['w_frame'] = dict(w_frame)
    wf = spec['w_frame']
    O = np.array(wf['origin'], float)
    yaw = math.radians(wf['yaw_deg'])
    Rw = Rz(yaw)

    def to_world(p):
        return (O + Rw @ np.asarray(p, float)).tolist()

    def box(name, centre_w, size_w):
        # size is given along the W axes; the W -> world rotation is a pure yaw, so the box keeps its size and gets the yaw orientation
        return dict(name=name, size=list(size_w), position=to_world(centre_w), orientation=quat_z(yaw))

    objs = [dict(id='medicine_shelf', color=spec['shelf_rgba'], boxes=[box(n, b['centre'], b['size']) for n, b in spec['shelf_boards'].items()])]
    p = spec['plinth']
    top_world = to_world([0, 0, p['top_z_w']])[2]
    height = top_world - p['bottom_z_world']
    cw = to_world([p['centre_xy_w'][0], p['centre_xy_w'][1], p['top_z_w'] - height / 2])
    objs.append(dict(id='shelf_plinth', color=spec['plinth_rgba'],
                     boxes=[dict(name='plinth', size=[p['size_xy'][0], p['size_xy'][1], height], position=cw, orientation=quat_z(yaw))]))
    po = spec['cartons']['target_pull_out_m'] if pull_out is None else pull_out
    for i, bid, x, t, h, l in carton_layout(spec):
        target = i == spec['cartons']['target_index']
        y = l / 2 - (po if target else 0.0)
        objs.append(dict(id='carton_%d' % i, color=spec['cartons']['target_rgba'] if target else spec['cartons']['rgba'],
                         boxes=[box('box%d_%s' % (bid, spec['cartons']['boxes_mm'][bid]['name'].split()[0]), [x, y, h / 2], [t, l, h])]))
    return spec.get('frame_id', 'world'), objs


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
        for b in obj['boxes']:
            co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in b['size']]))
            co.primitive_poses.append(make_pose(b['position'], b.get('orientation', (0.0, 0.0, 0.0, 1.0))))
        scene.world.collision_objects.append(co)
        if 'color' in obj:
            r, g, b_, a = (float(v) for v in obj['color'])
            scene.object_colors.append(ObjectColor(id=obj['id'], color=ColorRGBA(r=r, g=g, b=b_, a=a)))
    return scene


def main_ros():
    import rclpy
    from moveit_msgs.srv import ApplyPlanningScene
    from rclpy.node import Node
    rclpy.init()
    node = Node('add_medicine_shelf')
    scene_file = node.declare_parameter('scene_file', '').value or default_scene_file()
    timeout = node.declare_parameter('service_timeout', 60.0).value
    pull = node.declare_parameter('pull_out', float('nan')).value
    frame, objs = load_objects(scene_file, None if math.isnan(pull) else pull)
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
    node.get_logger().info('MoveIt - medicine grasp: added %s from %s to the planning scene' % (ids, scene_file))
    node.destroy_node()
    rclpy.shutdown()
    return 0


AMR_BOX = ([-0.484, -0.327, 0.0], [0.484, 0.327, 0.2515])     # d200_base_link collision box in d200_base_footprint (urdf)


def amr_clearance(path, pull_out=None):
    """smallest gap (m) between the AMR box and any scene box (axis-aligned bounding box of the rotated box; conservative), per object"""
    _, objs = load_objects(path, pull_out)
    lo, hi = np.array(AMR_BOX[0]), np.array(AMR_BOX[1])
    out = {}
    for o in objs:
        gap = 9.0
        for b in o['boxes']:
            yaw = 2 * math.atan2(b['orientation'][2], b['orientation'][3])
            R = Rz(yaw)
            h = np.array(b['size']) / 2
            corners = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ R.T + np.array(b['position'])
            blo, bhi = corners.min(0), corners.max(0)
            d = np.maximum(0, np.maximum(lo - bhi, blo - hi))
            gap = min(gap, float(np.linalg.norm(d)))
        out[o['id']] = round(gap, 4)
    return out


def pose_json(path, pull_out=None):
    """the shelf pose for other consumers (Isaac): W in the d200_base_footprint frame + board heights + carton layout"""
    spec = yaml.safe_load(open(path))
    wf = spec['w_frame']
    layout = [dict(index=i, box_id=bid, name=spec['cartons']['boxes_mm'][bid]['name'], target=(i == spec['cartons']['target_index']),
                   centre_x_w_m=round(x, 6), thickness_m=t, height_m=h, length_m=l, centre_y_w_m=l / 2, centre_z_w_m=h / 2)
              for i, bid, x, t, h, l in carton_layout(spec)]
    return dict(
        schema='medicine_shelf_pose/v1', frame='d200_base_footprint',
        note='W = shelf frame (origin shelf floor at the front edge, centred on the target carton, +X lateral, +Y into the shelf, +Z up); '
             'world_T_W = translate xyz, then yaw about +Z. d200_base_footprint = D200 root (floor under the AMR centre, +X forward, Z up). '
             'The AMR parks with its front facing the shelf. Source of truth: config/medicine_shelf_scene.yaml.',
        W_in_frame=dict(xyz=wf['origin'], yaw_deg=wf['yaw_deg']),
        shelf_floor_top_z_m=wf['origin'][2],
        shelf_boards_W=spec['shelf_boards'],
        shelf_inner_height_m=spec['shelf_boards']['upper_board']['centre'][2] - spec['shelf_boards']['upper_board']['size'][2] / 2,
        target_spine_xyz_in_frame=(np.array(wf['origin']) + Rz(math.radians(wf['yaw_deg'])) @ np.array([0, 0, spec['cartons']['boxes_mm'][spec['cartons']['sequence'][spec['cartons']['target_index']]]['h'] * 5e-4])).tolist(),
        plinth=spec['plinth'], cartons=layout, target_pull_out_m=spec['cartons']['target_pull_out_m'],
        handoff_pull_out_m=0.100,
        amr_box_in_frame=dict(min=AMR_BOX[0], max=AMR_BOX[1], note='d200_base_link collision box (rebel_medicine_grasp.urdf.xacro)'),
        amr_clearance_m_packed=amr_clearance(path, 0.0), amr_clearance_m_target_pulled_out=amr_clearance(path, 0.100),
        amr_parking='AMR front faces the shelf (shelf front plane perpendicular to the AMR X axis), shelf centred on the AMR centre line')


def main():
    if '--write-pose-json' in sys.argv:
        argv = sys.argv
        f = argv[argv.index('--scene-file') + 1] if '--scene-file' in argv else default_scene_file()
        out = argv[argv.index('--write-pose-json') + 1]
        json.dump(pose_json(f), open(out, 'w'), indent=1)
        print('wrote', out)
        return 0
    if '--print' in sys.argv:
        argv = sys.argv
        f = argv[argv.index('--scene-file') + 1] if '--scene-file' in argv else default_scene_file()
        po = float(argv[argv.index('--pull-out') + 1]) if '--pull-out' in argv else None
        frame, objs = load_objects(f, po)
        print(json.dumps(dict(frame_id=frame, scene_file=f, objects=objs), indent=1))
        return 0
    return main_ros()


if __name__ == '__main__':
    sys.exit(main())
