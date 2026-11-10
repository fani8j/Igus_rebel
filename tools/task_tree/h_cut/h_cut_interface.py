"""THE ONE FILE TO EDIT when plugging your own H-cut action server into the behaviour tree.

h_cut_tree.py never touches the H-cut action type directly; it only uses the five things below.
Out of the box they point at compal_cut_task_msgs/action/HCut on /h_cut, which is what
mock_h_cut_server.py serves. To use your own server, either

  A. make your server implement compal_cut_task_msgs/action/HCut on /h_cut  -> change nothing here
     (at most ACTION_NAME), or
  B. keep your own action type and change the import, ACTION_NAME, make_goal(),
     interpret_result() (and optionally run_folder_from_result() / feedback_text()).

See README.md, section 4, for a worked example of B.
"""
from compal_cut_task_msgs.action import HCut as ACTION_TYPE  # B: from your_pkg.action import YourCut as ACTION_TYPE

# Where your action server listens (ros2 action list).
ACTION_NAME = "/h_cut"


def make_goal(ctx):
    """Build the goal for one cutting attempt.

    ctx.execute_allowed  bool  True only if the user gave BOTH --execute and
                               --i-understand-this-moves-the-real-robot. If False, the goal MUST
                               make your server plan only / not move.
    ctx.attempt          int   1 = first cut, 2, 3, ... = re-cut after CheckCut said NOT_FULLY_CUT
    ctx.recut_hint_json  str   "" on attempt 1, else the previous CheckCut report (JSON text)
    """
    goal = ACTION_TYPE.Goal()
    goal.execute = ctx.execute_allowed
    goal.i_understand_this_moves_the_real_robot = ctx.execute_allowed
    goal.attempt = ctx.attempt
    goal.recut_hint_json = ctx.recut_hint_json
    return goal


def interpret_result(result):
    """Return (success: bool, message: str) from your action's Result.

    success=True means "the H-cut finished, go on and check the tape". Anything else stops the
    tree without retrying (a re-cut only happens when CheckCut says NOT_FULLY_CUT).
    """
    return bool(result.success), f"{result.message} (did_move={result.did_move})"


def run_folder_from_result(result):
    """Folder with the wrist-camera videos of this cut, if your server records them itself.

    Return "" to let cut_check_server.py use its own recording of the wrist-camera topics.
    """
    return getattr(result, "run_folder", "")


def feedback_text(feedback):
    """One line for the tree watcher while the cut is running (return "" to show nothing)."""
    return f"{feedback.stage} {feedback.progress:.0%} {feedback.message}"
