"""The probe must stay strictly passive: subscribers + its own two output publishers, nothing else."""
import os
import re

import pytest

import synth
from side_cut_vision import mock_camera
from side_cut_vision.domains import MAX_DOMAIN_ID, check_domain_param, plan_domains

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "side_cut_vision")
LAUNCH = os.path.join(os.path.dirname(ROOT), "launch", "side_cut_probe.launch.py")
PROBE_SOURCES = ("probe_node.py", "probe_core.py", "trigger.py", "record.py", "measure.py", "detect.py")

BANNED = ("create_client", "ActionClient", "create_service", "create_action", "set_parameters", "set_parameter(",
          "call_async", "declare_parameter", "SetParameters", "FollowJointTrajectory", "JointTrajectory",
          "trajectory_msgs", "control_msgs", "controller_manager_msgs", "std_srvs", "Trigger.Request", "srv import",
          "cancel_goal", "send_goal", "ros2 param", "subprocess", "os.system")


def src(name):
    return open(os.path.join(ROOT, name)).read()


@pytest.mark.parametrize("name", PROBE_SOURCES)
def test_no_active_robot_interfaces(name):
    text = src(name)
    for banned in BANNED:
        assert banned not in text, f"{name}: {banned}"


def test_only_subscriptions_timers_and_two_output_publishers():
    text = src("probe_node.py")
    assert set(re.findall(r"\.create_(\w+)\(", text)) <= {"node", "subscription", "publisher"}
    pubs = re.findall(r"create_publisher\((\w+),\s*f?\"([^\"]+)\"", text)
    assert sorted(p[0] for p in pubs) == ["CompressedImage", "String"]
    assert all(p[1].startswith("/side_cut_vision/arm_") for p in pubs)
    # every .publish( is on the two output publishers
    assert sorted(set(re.findall(r"(\w+)\.publish\(", text))) == ["pub_img", "pub_meas"]
    # message types imported by the node: only read-only sensor/state types + its two outputs
    imported = set(re.findall(r"from (\w+)\.msg import ([\w, ]+)", text))
    flat = {(m, n.strip()) for m, names in imported for n in names.split(",")}
    assert flat == {("sensor_msgs", "CameraInfo"), ("sensor_msgs", "CompressedImage"), ("sensor_msgs", "Image"),
                    ("sensor_msgs", "JointState"), ("std_msgs", "String"), ("dual_cut_msgs", "CoordinatorStatus")}


def test_only_tf_listener_no_broadcaster():
    for name in os.listdir(ROOT):
        if name.endswith(".py") and name != "mock_camera.py":
            text = src(name)
            assert "TransformBroadcaster" not in text and "StaticTransformBroadcaster" not in text, name


def test_launch_runs_one_process_with_required_camera_domain():
    text = open(LAUNCH).read()
    assert text.count("ExecuteProcess(") == 1
    assert "DeclareLaunchArgument('camera_domain_id', description" in text     # no default_value: required
    assert "'nice', '-n', '10'" in text


def test_mock_camera_refuses_real_domains_and_non_localhost():
    ok = {"ROS_DOMAIN_ID": "203", "ROS_LOCALHOST_ONLY": "1"}
    assert mock_camera.safety_refusal(ok) == ""
    for dom in ("11", "12"):
        assert "real-cell" in mock_camera.safety_refusal({**ok, "ROS_DOMAIN_ID": dom})
    assert "LOCALHOST" in mock_camera.safety_refusal({"ROS_DOMAIN_ID": "203"})
    assert "LOCALHOST" in mock_camera.safety_refusal({"ROS_DOMAIN_ID": "203", "ROS_LOCALHOST_ONLY": "0"})
    assert mock_camera.safety_refusal({"ROS_DOMAIN_ID": "x", "ROS_LOCALHOST_ONLY": "1"})
    text = src("mock_camera.py")
    assert "mock_components/GenericSystem" in text and "refusing to start" in text


def test_domain_planning():
    p = plan_domains(12, 11, -1)
    assert p.separate_camera_context and p.camera_domain == 11
    p = plan_domains(203, 203, -1)
    assert not p.separate_camera_context
    p = plan_domains(203, -1, -1)
    assert not p.separate_camera_context and p.camera_domain is None
    for bad in (233, -2, 1.5, True):
        with pytest.raises(ValueError):
            check_domain_param("camera_domain_id", bad)
    assert MAX_DOMAIN_ID == 232
