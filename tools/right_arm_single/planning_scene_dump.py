import rclpy, sys
from rclpy.node import Node
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents
rclpy.init(); n=Node('scene_q'); c=n.create_client(GetPlanningScene,'/get_planning_scene'); c.wait_for_service(10)
r=GetPlanningScene.Request(); r.components.components=PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
f=c.call_async(r); rclpy.spin_until_future_complete(n,f,timeout_sec=10)
for o in f.result().scene.world.collision_objects:
    p=o.pose.position
    print(o.id, o.header.frame_id, [round(v,3) for v in o.primitives[0].dimensions] if o.primitives else 'mesh', None if p is None else [round(p.x,3),round(p.y,3),round(p.z,3)])
