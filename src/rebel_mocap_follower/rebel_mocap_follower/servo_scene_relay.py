"""Planning scene feed for the two mocap Servo nodes (dual-arm table cell).

Mirrors dual_teleop's selector: polls /get_planning_scene at 2 Hz and republishes the full scene with every
same-arm link pair allowed in the ACM (otherwise Servo crawls because of pairs that are close by construction,
e.g. wrist camera vs link5). Cross-arm, arm-vs-table and arm-vs-object entries stay exactly as move_group has them.
Needs dual_teleop (imports its pure allow_same_arm_pairs).
"""
import copy

import rclpy
from dual_teleop.selector_core import allow_same_arm_pairs
from moveit_msgs.msg import AllowedCollisionEntry, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

CALL_TIMEOUT_S = 3.0


class ServoSceneRelay(Node):
    def __init__(self):
        super().__init__("servo_scene_relay")
        d = self.declare_parameter
        self.out_topic = d("scene_topic", "/mocap/servo_scene").value
        self.allow_same_arm = bool(d("allow_same_arm_pairs", True).value)
        self.client = self.create_client(GetPlanningScene, d("planning_scene_service", "/get_planning_scene").value)
        self.pub = self.create_publisher(PlanningScene, self.out_topic,
                                         QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.pending = None
        self.create_timer(0.5, self.poll)
        self.get_logger().info(f"relaying /get_planning_scene -> {self.out_topic} (same-arm pairs allowed: "
                               f"{self.allow_same_arm})")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def poll(self):
        if self.pending is not None and self.now() - self.pending < CALL_TIMEOUT_S:
            return
        self.pending = None
        if not self.client.service_is_ready():
            return
        req = GetPlanningScene.Request()
        c = PlanningSceneComponents
        req.components.components = (c.SCENE_SETTINGS | c.ROBOT_STATE | c.WORLD_OBJECT_NAMES | c.WORLD_OBJECT_GEOMETRY
                                     | c.OCTOMAP | c.TRANSFORMS | c.ALLOWED_COLLISION_MATRIX
                                     | c.ROBOT_STATE_ATTACHED_OBJECTS)
        self.pending = self.now()
        self.client.call_async(req).add_done_callback(self.on_scene)

    def on_scene(self, future):
        self.pending = None
        try:
            scene = copy.deepcopy(future.result().scene)
        except Exception as error:
            self.get_logger().warn(f"get_planning_scene failed: {error}")
            return
        acm = scene.allowed_collision_matrix
        if acm.entry_names and self.allow_same_arm:
            rows = allow_same_arm_pairs(list(acm.entry_names), [list(e.enabled) for e in acm.entry_values])
            acm.entry_values = [AllowedCollisionEntry(enabled=r) for r in rows]
        scene.is_diff = False
        self.pub.publish(scene)


def main(args=None):
    rclpy.init(args=args)
    node = ServoSceneRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
