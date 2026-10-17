import os

# Isolated DDS domain; never the robot's domain 11. Set before rclpy is imported anywhere.
os.environ["ROS_DOMAIN_ID"] = "87"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ["BOX_SEAM_CAPTURE_STAGING"] = "off"  # node tests write straight into their tmp output_root

_SHM = os.path.join(os.path.dirname(__file__), "..", "e2e", "fastdds_shm_only.xml")
if os.path.exists(_SHM):  # loopback UDP drops multi-MB images under ROS_LOCALHOST_ONLY on Humble
    os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = os.path.abspath(_SHM)

import threading  # noqa: E402

import pytest  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _rclpy():
    rclpy.init()
    yield
    rclpy.shutdown()


class Spinner:
    def __init__(self, *nodes):
        self.nodes = list(nodes)
        self.ex = MultiThreadedExecutor(num_threads=8)
        for n in nodes:
            self.ex.add_node(n)
        self.th = threading.Thread(target=self.ex.spin, daemon=True)
        self.th.start()

    def stop(self):
        self.ex.shutdown()
        self.th.join(timeout=5)
        for n in self.nodes:
            n.destroy_node()


@pytest.fixture
def spinner():
    made = []

    def make(*nodes):
        s = Spinner(*nodes)
        made.append(s)
        return s

    yield make
    for s in made:
        s.stop()
