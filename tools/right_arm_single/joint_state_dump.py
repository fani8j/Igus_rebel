import rclpy, sys, json
from rclpy.node import Node
from sensor_msgs.msg import JointState
rclpy.init(); n=Node('js_q'); got=[]
n.create_subscription(JointState,'/joint_states',lambda m: got.append(m),10)
import time; t=time.time()
while time.time()-t<5 and not got: rclpy.spin_once(n,timeout_sec=0.2)
m=got[-1]; print(json.dumps(dict(zip(m.name,m.position))))
