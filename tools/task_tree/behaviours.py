"""Custom py_trees leaf behaviours for the dual-arm task tree (see run_task_tree.py / README.md).

Only the leaves that py_trees / py_trees_ros do not already provide live here. Service calls
that just need "send a constant request, store the response" reuse
py_trees_ros.service_clients.FromConstant directly in run_task_tree.py.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import py_trees
import py_trees_ros
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


class CheckResponseSuccess(py_trees.behaviour.Behaviour):
    """SUCCESS if the service response stored at ``key`` has ``.success == True``.

    py_trees_ros service clients return SUCCESS as soon as *any* response arrives, even when the
    server answered success=False, so every call is followed by this check. The server's
    ``.message`` becomes this behaviour's feedback message (shown by tree watchers) and is copied
    to ``message_key`` on the blackboard for the end-of-run summary. If ``flag_key`` is given,
    the verdict is also stored there as a bool (e.g. ``is_tape_cut``).
    """

    def __init__(self, name: str, key: str, message_key: str, flag_key: str = None):
        super().__init__(name)
        self.key = key
        self.message_key = message_key
        self.flag_key = flag_key
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key, access=py_trees.common.Access.READ)
        self.bb.register_key(message_key, access=py_trees.common.Access.WRITE)
        if flag_key:
            self.bb.register_key(flag_key, access=py_trees.common.Access.WRITE)

    def update(self):
        response = self.bb.get(self.key)
        message = getattr(response, "message", "")
        success = bool(getattr(response, "success", False))
        self.bb.set(self.message_key, message)
        if self.flag_key:
            self.bb.set(self.flag_key, success)
        self.feedback_message = message
        if success:
            self.logger.info(f"{self.name}: OK - {message}")
            return py_trees.common.Status.SUCCESS
        self.logger.error(f"{self.name}: FAILED - {message}")
        return py_trees.common.Status.FAILURE


class DeriveSnapshotPath(py_trees.behaviour.Behaviour):
    """Turn a CaptureBoxSnapshot response into the JSON path box_perception_node just wrote.

    box_perception_node saves ``<snapshot_output_dir>/<snapshot_id>.json`` but only returns the
    snapshot message, so the path is rebuilt here from the same directory
    (dual_arm_box_perception.yaml: ``snapshot_output_dir: ~/dual_arm_box_snapshots``).
    """

    def __init__(self, name: str, response_key: str, snapshot_dir: Path):
        super().__init__(name)
        self.response_key = response_key
        self.snapshot_dir = snapshot_dir
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(response_key, access=py_trees.common.Access.READ)
        self.bb.register_key("snapshot_file", access=py_trees.common.Access.WRITE)

    def update(self):
        snapshot_id = self.bb.get(self.response_key).snapshot.snapshot_id
        path = self.snapshot_dir / f"{snapshot_id}.json"
        if not path.is_file():
            self.feedback_message = f"snapshot JSON not found: {path}"
            self.logger.error(self.feedback_message)
            return py_trees.common.Status.FAILURE
        self.bb.snapshot_file = str(path)
        self.feedback_message = str(path)
        return py_trees.common.Status.SUCCESS


class WriteFixtureSnapshot(py_trees.behaviour.Behaviour):
    """Offline stand-in for perception: write a fresh snapshot JSON of the USD scene's carton.

    Geometry from dual_arm_rig.usda /World/CardboardBox (Xform at (0, -1, 0.73), rotated 90 deg
    about Z; Body 0.32 x 0.45 x 0.25 m centred 0.4 m along the box's local X): top face centred
    at world (0, -0.6), z = 0.98, 0.45 m along world X, 0.32 m along world Y, the two lid flaps
    meeting along world X. Corner/seam layout follows box_perception_node.py (seam runs
    left -> right, parallel to the back edge); "back" = the side farther from the camera
    (more negative Y). Field set = what box_h_cut_executor's loadExternalSnapshot() requires.
    Rewritten on every run so it passes the executor's external_snapshot_max_age_s check.
    """

    CENTER = (0.0, -0.6)
    TOP_Z = 0.98
    HALF_X = 0.225   # along the seam
    HALF_Y = 0.16    # across the seam

    def __init__(self, name: str, path: Path):
        super().__init__(name)
        self.path = path
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key("snapshot_file", access=py_trees.common.Access.WRITE)

    def update(self):
        cx, cy, z = self.CENTER[0], self.CENTER[1], self.TOP_Z

        def p(x, y):
            return {"x": x, "y": y, "z": z}

        report = {
            "snapshot_id": f"offline_fixture_{int(time.time())}",
            "frame_id": "world",
            "seam_start": p(cx + self.HALF_X, cy),
            "seam_end": p(cx - self.HALF_X, cy),
            "seam_center": p(cx, cy),
            "back_left": p(cx + self.HALF_X, cy - self.HALF_Y),
            "back_right": p(cx - self.HALF_X, cy - self.HALF_Y),
            "front_left": p(cx + self.HALF_X, cy + self.HALF_Y),
            "front_right": p(cx - self.HALF_X, cy + self.HALF_Y),
            "surface_normal": {"x": 0.0, "y": 0.0, "z": 1.0},
            "plane_rms_m": 0.0,
            "source": "scripts/task_tree offline fixture (USD /World/CardboardBox), not a camera capture",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.bb.snapshot_file = str(self.path)
        self.feedback_message = str(self.path)
        self.logger.warning(f"{self.name}: using OFFLINE fixture snapshot {self.path}")
        return py_trees.common.Status.SUCCESS


class SetRemoteParameter(py_trees.behaviour.Behaviour):
    """Set a string parameter on another node from a blackboard value (replaces `ros2 param set`)."""

    def __init__(self, name: str, target_node: str, parameter: str, value_key: str):
        super().__init__(name)
        self.target_node = target_node
        self.parameter = parameter
        self.value_key = value_key
        self.client = None
        self.future = None
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(value_key, access=py_trees.common.Access.READ)

    def setup(self, **kwargs):
        self.client = kwargs["node"].create_client(SetParameters, f"{self.target_node}/set_parameters")

    def initialise(self):
        self.future = None
        if not self.client.service_is_ready():
            return
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=str(self.bb.get(self.value_key)))
        request = SetParameters.Request(parameters=[Parameter(name=self.parameter, value=value)])
        self.future = self.client.call_async(request)

    def update(self):
        if self.future is None:
            self.feedback_message = f"{self.target_node}/set_parameters not available"
            self.logger.error(self.feedback_message)
            return py_trees.common.Status.FAILURE
        if not self.future.done():
            return py_trees.common.Status.RUNNING
        result = self.future.result().results[0]
        if not result.successful:
            self.feedback_message = f"rejected: {result.reason}"
            self.logger.error(f"{self.name}: {self.feedback_message}")
            return py_trees.common.Status.FAILURE
        self.feedback_message = f"{self.parameter} = {self.bb.get(self.value_key)}"
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        if self.future is not None and not self.future.done():
            self.client.remove_pending_request(self.future)


class LogAndSucceed(py_trees.behaviour.Behaviour):
    """Always SUCCESS; records why (e.g. 'plan only, execution skipped') on the blackboard."""

    def __init__(self, name: str, message: str, message_key: str):
        super().__init__(name)
        self.message = message
        self.message_key = message_key
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(message_key, access=py_trees.common.Access.WRITE)

    def update(self):
        self.bb.set(self.message_key, self.message)
        self.feedback_message = self.message
        self.logger.info(f"{self.name}: {self.message}")
        return py_trees.common.Status.SUCCESS


class NotImplementedStub(py_trees.behaviour.Behaviour):
    """Placeholder for a task step that has no implementation yet.

    Reports RUNNING for ``ticks`` ticks (so the step is visible in a tree watcher) and then
    SUCCESS, logging a WARNING every time it starts. Replace it with a real behaviour (usually
    a py_trees_ros action/service client) once the step exists on the ROS side.
    """

    def __init__(self, name: str, ticks: int = 4):
        super().__init__(f"{name} [STUB]")
        self.ticks = ticks
        self.count = 0

    def initialise(self):
        self.count = 0
        self.logger.warning(f"{self.name}: NOT IMPLEMENTED - simulating success, nothing moves")

    def update(self):
        self.count += 1
        if self.count < self.ticks:
            self.feedback_message = f"simulating ({self.count}/{self.ticks})"
            return py_trees.common.Status.RUNNING
        self.feedback_message = "not implemented (simulated success)"
        return py_trees.common.Status.SUCCESS


class MockActionLeaf(py_trees.behaviour.Behaviour):
    """Mock of an action-client leaf: RUNNING for ``ticks`` ticks, then writes a result to the blackboard.

    The result is a ``SimpleNamespace(success, message)`` stored at ``result_key`` - the same
    shape ActionResultToBlackboard stores for a real action - and the behaviour itself returns
    SUCCESS ("result recorded"). Follow it with CheckResponseSuccess, which turns
    ``result.success`` into SUCCESS / FAILURE, so swapping mock -> real action changes one leaf.

    The first ``fail_first`` attempts (counted across Retry re-runs) report success=False.
    Nothing moves; this only simulates time passing.
    """

    def __init__(self, name: str, result_key: str, ticks: int = 4, fail_first: int = 0,
                 success_message: str = "done", failure_message: str = "failed"):
        super().__init__(f"{name} [MOCK]")
        self.result_key = result_key
        self.ticks = ticks
        self.fail_first = fail_first
        self.success_message = success_message
        self.failure_message = failure_message
        self.attempt = 0
        self.count = 0
        self.bb = self.attach_blackboard_client(name=self.name)
        self.bb.register_key(result_key, access=py_trees.common.Access.WRITE)

    def initialise(self):
        self.attempt += 1
        self.count = 0
        self.logger.info(f"{self.name}: attempt {self.attempt} (mock, nothing moves)")

    def update(self):
        self.count += 1
        if self.count < self.ticks:
            self.feedback_message = f"attempt {self.attempt}: running ({self.count}/{self.ticks})"
            return py_trees.common.Status.RUNNING
        success = self.attempt > self.fail_first
        message = self.success_message if success else self.failure_message
        message = f"attempt {self.attempt}: {message} [mock]"
        self.bb.set(self.result_key, SimpleNamespace(success=success, message=message))
        self.feedback_message = message
        return py_trees.common.Status.SUCCESS


class ActionResultToBlackboard(py_trees_ros.action_clients.FromConstant):
    """Action client that stores the action's Result at ``result_key`` (for CheckResponseSuccess).

    py_trees_ros action clients only report whether the goal reached SUCCEEDED; a server that
    finishes normally but answers ``success=False`` (e.g. CutBox's IK_FAILED) would still be
    SUCCESS. So, like MockActionLeaf, this returns SUCCESS once *any* outcome is known and leaves
    the verdict to the following CheckResponseSuccess. If there is no usable result (goal
    rejected, aborted, server missing) it stores ``success=False`` with the reason.
    """

    def __init__(self, name: str, action_type, action_name: str, action_goal, result_key: str,
                 wait_for_server_timeout_sec: float = 0.0):
        super().__init__(name=name, action_type=action_type, action_name=action_name,
                         action_goal=action_goal, wait_for_server_timeout_sec=wait_for_server_timeout_sec)
        self.result_key = result_key
        self.result_bb = self.attach_blackboard_client(name=f"{name}_result")
        self.result_bb.register_key(result_key, access=py_trees.common.Access.WRITE)

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.RUNNING:
            return status
        result = getattr(self.result_message, "result", None)
        if status == py_trees.common.Status.SUCCESS and result is not None:
            self.result_bb.set(self.result_key, result)
        else:
            reason = f"{self.feedback_message} ({self.result_status_string or 'no result'})"
            self.result_bb.set(self.result_key, SimpleNamespace(success=False, message=reason))
        return py_trees.common.Status.SUCCESS
