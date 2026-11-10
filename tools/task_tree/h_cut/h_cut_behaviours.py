"""Leaf / decorator behaviours used only by the H-cut tree (h_cut_tree.py).

The shared ones (CheckResponseSuccess, ...) live in ../behaviours.py.
"""
from types import SimpleNamespace

import py_trees
import rclpy.action
from action_msgs.msg import GoalStatus

STATUS_NAMES = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED", GoalStatus.STATUS_ABORTED: "ABORTED",
                GoalStatus.STATUS_CANCELED: "CANCELED"}


class ActionLeaf(py_trees.behaviour.Behaviour):
    """Send one action goal, wait for the result, store a verdict on the blackboard.

    The goal is built fresh every time the leaf starts: ``make_goal(ctx)`` where ``ctx`` holds
    the blackboard keys listed in ``read_keys``. The Result is turned into
    ``SimpleNamespace(success, message, raw)`` by ``interpret(result) -> (success, message)``
    and stored at ``result_key``.

    Like every action / service leaf in this project the leaf itself returns SUCCESS as soon as
    *any* outcome is known (also: server missing, goal rejected, aborted) and the following
    CheckResponseSuccess turns ``.success`` into SUCCESS / FAILURE. If the tree interrupts the
    leaf while it is running, the goal is cancelled.
    """

    def __init__(self, name, action_type, action_name, make_goal, interpret, result_key,
                 read_keys=(), feedback_text=None):
        super().__init__(name)
        self.action_type = action_type
        self.action_name = action_name
        self.make_goal = make_goal
        self.interpret = interpret
        self.result_key = result_key
        self.read_keys = tuple(read_keys)
        self.feedback_text = feedback_text
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(result_key, access=py_trees.common.Access.WRITE)
        for key in self.read_keys:
            self.bb.register_key(key, access=py_trees.common.Access.READ)
        self.node = None
        self.client = None
        self._reset()

    def _reset(self):
        self.send_future = None
        self.goal_handle = None
        self.result_future = None
        self.error = None

    def setup(self, **kwargs):
        self.node = kwargs["node"]
        self.client = rclpy.action.ActionClient(self.node, self.action_type, self.action_name)

    def initialise(self):
        self._reset()
        if not self.client.server_is_ready():
            self.error = f"action server {self.action_name} not available"
            return
        ctx = SimpleNamespace(**{key: self.bb.get(key) for key in self.read_keys})
        try:
            goal = self.make_goal(ctx)
        except Exception as exc:  # a bug in make_goal() should fail this leaf, not crash the tree
            self.error = f"make_goal() failed: {exc!r}"
            return
        self.send_future = self.client.send_goal_async(goal, feedback_callback=self._on_feedback)
        self.feedback_message = "goal sent"

    def _on_feedback(self, msg):
        if self.feedback_text is not None:
            try:
                self.feedback_message = self.feedback_text(msg.feedback)
            except Exception as exc:
                self.feedback_message = f"feedback_text() failed: {exc!r}"

    def _finish(self, success, message, raw=None):
        self.bb.set(self.result_key, SimpleNamespace(success=success, message=message, raw=raw))
        self.feedback_message = message
        return py_trees.common.Status.SUCCESS

    def update(self):
        if self.error:
            self.logger.error(f"{self.name}: {self.error}")
            return self._finish(False, self.error)
        if self.goal_handle is None:
            if not self.send_future.done():
                return py_trees.common.Status.RUNNING
            self.goal_handle = self.send_future.result()
            if self.goal_handle is None or not self.goal_handle.accepted:
                return self._finish(False, f"goal rejected by {self.action_name}")
            self.result_future = self.goal_handle.get_result_async()
        if not self.result_future.done():
            return py_trees.common.Status.RUNNING
        response = self.result_future.result()
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            name = STATUS_NAMES.get(response.status, str(response.status))
            return self._finish(False, f"goal {name} by {self.action_name}", response.result)
        try:
            success, message = self.interpret(response.result)
        except Exception as exc:
            return self._finish(False, f"interpret_result() failed: {exc!r}", response.result)
        return self._finish(bool(success), str(message), response.result)

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.INVALID and self.goal_handle is not None \
                and self.result_future is not None and not self.result_future.done():
            self.logger.warning(f"{self.name}: interrupted - cancelling the goal")
            self.goal_handle.cancel_goal_async()

    def shutdown(self):
        if self.client is not None:
            self.client.destroy()


class BeginAttempt(py_trees.behaviour.Behaviour):
    """Start of one cut -> check attempt: count it and prepare the re-cut hint.

    attempt += 1; the previous CheckCut report becomes ``recut_hint_json`` for the next H-cut
    goal; ``check_result`` is cleared so a stale verdict cannot trigger a retry.
    """

    def __init__(self, name):
        super().__init__(name)
        self.bb = self.attach_blackboard_client(name=name)
        for key in ("attempt", "recut_hint_json", "check_result", "is_tape_cut"):
            self.bb.register_key(key, access=py_trees.common.Access.WRITE)

    def update(self):
        previous = self.bb.check_result
        self.bb.attempt += 1
        self.bb.recut_hint_json = getattr(getattr(previous, "raw", None), "report_json", "") if previous else ""
        self.bb.check_result = None
        self.bb.is_tape_cut = False
        self.feedback_message = f"attempt {self.bb.attempt}"
        self.logger.info(f"{self.name}: === attempt {self.bb.attempt} ===")
        return py_trees.common.Status.SUCCESS


class RetryWhenNotFullyCut(py_trees.decorators.Retry):
    """Retry, but only when the tape check said NOT_FULLY_CUT.

    A failed H-cut (planning / execution error), a failed start of the recording or a check that
    could not judge (NOT_CHECKED) stops at once: cutting again would not fix those, and moving the
    arm again after an execution error is not something to do automatically.
    """

    def __init__(self, name, child, num_failures):
        super().__init__(name=name, child=child, num_failures=num_failures)
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key("check_result", access=py_trees.common.Access.READ)

    def update(self):
        if self.decorated.status == py_trees.common.Status.FAILURE:
            result = self.bb.check_result
            verdict = getattr(getattr(result, "raw", None), "verdict", "")
            if verdict != "NOT_FULLY_CUT":
                self.failures += 1
                self.feedback_message = "failed without a NOT_FULLY_CUT verdict - not re-cutting"
                return py_trees.common.Status.FAILURE
        return super().update()
