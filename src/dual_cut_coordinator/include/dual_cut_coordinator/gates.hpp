// Pure gate logic of the coordinator (WP6): frame-consistency, cut-line geometry and tip-on-line gate,
// at-rest tracking, start barriers and the monitor's stop decision. No ROS node, no MoveIt scene, so
// everything here has plain gtests.
#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "dual_cut_coordinator/phase_plan.hpp"
#include "dual_cut_coordinator/trajectory_tools.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace dual_cut
{

// ---- frame-consistency gate -------------------------------------------------------------------

/// URDF fixed-axis rpy (R = Rz(yaw) Ry(pitch) Rx(roll)) plus translation.
Eigen::Isometry3d poseFromXyzRpy(const std::array<double, 3> & xyz, const std::array<double, 3> & rpy);

/// The transform an executor exports (snapshot_planning_base_xyz/rpy) must equal the dual URDF's
/// world -> arm_X_base_link: max |dt| <= tol and rotation angle <= tol (default 1e-6).
Status checkFrameConsistency(
  const Eigen::Isometry3d & exported, const Eigen::Isometry3d & expected, double tol = 1e-6);

// ---- the same box seen by both arms ----------------------------------------------------------

/// Both per-arm executors place `detected_carton` in their own scene (right: after the fixed snapshot
/// transform). In the dual world they must be the same physical box: dimensions equal within tol, centres
/// within tol, orientations equal up to the box's own symmetries (0 or 180 degrees about its vertical axis).
Status checkBoxConsistency(
  const Eigen::Isometry3d & a_world, const Eigen::Vector3d & a_size, const Eigen::Isometry3d & b_world,
  const Eigen::Vector3d & b_size, double tol_m, double tol_rad, double * center_error_m = nullptr,
  double * rotation_error_rad = nullptr);

// ---- cut lines and the tip-on-line gate ---------------------------------------------------------

struct CutLine
{
  Eigen::Vector3d start = Eigen::Vector3d::Zero();   // world frame
  Eigen::Vector3d end = Eigen::Vector3d::Zero();
};

/// Distance from a point to the finite segment [a, b].
double distancePointToSegment(const Eigen::Vector3d & p, const Eigen::Vector3d & a, const Eigen::Vector3d & b);

/// World cut lines of a phase's CUT segments from the exported targets (planning_frame poses):
/// exactly 2 per CUT segment (start, end), all with frame_id == expected_frame, finite. Fails closed on
/// a count or frame mismatch. base_to_world is the (verified) world -> arm base transform.
Status cutLinesFromTargets(
  const std::vector<geometry_msgs::msg::PoseStamped> & targets, size_t expected_cut_segments,
  const std::string & expected_frame, const Eigen::Isometry3d & base_to_world, std::vector<CutLine> & out);

/// Tool-tip position in world for joints given by name (unprefixed or prefixed, as in the trajectory).
using TipFk = std::function<Status(const std::vector<std::string> &, const std::vector<double> &, Eigen::Vector3d &)>;

struct TipGateResult
{
  bool ok = false;
  std::string message;
  size_t cut_segments = 0;
  size_t samples = 0;
  double max_deviation_m = 0.0;       // largest distance of a sample from its cut line
  double max_endpoint_error_m = 0.0;  // distance of the cut's first/last point from the line ends
  bool safe() const { return ok; }
};

/// In every CUT segment of the arm's phase, the tool tip (FK of the dual model) must stay within tol
/// of the exported world cut line and start/end within tol of its end points. Fails closed if the
/// number of cut lines does not match the number of CUT segments, or FK fails.
TipGateResult checkTipOnLine(
  const ArmPhase & arm, const std::vector<CutLine> & lines, const TipFk & fk, double tol_m,
  const SampleOptions & sample);

// ---- at rest and start barriers ---------------------------------------------------------------

/// Tracks "all |v| below the threshold for at least N seconds" from timestamped velocity samples.
class RestTracker
{
public:
  explicit RestTracker(double velocity_threshold = 0.03) : threshold_(velocity_threshold) {}
  void reset();
  /// t_s: monotonic time of the sample (seconds). max_abs_velocity of all tracked joints.
  void update(double t_s, double max_abs_velocity);
  /// Seconds the joints have been at rest as of now_s (0 if moving or no sample yet).
  double restDuration(double now_s) const;
  double threshold() const { return threshold_; }

private:
  double threshold_;
  bool have_ = false;
  double rest_since_ = 0.0;  // start of the current at-rest stretch (valid if at_rest_)
  bool at_rest_ = false;
  double last_t_ = 0.0;
};

struct BarrierLimits
{
  double joint_state_max_age_s = 0.1;
  double start_tolerance_rad = 0.01;
  double rest_duration_s = 0.5;
  double snapshot_max_age_s = 240.0;
  double monitor_max_age_s = 0.2;
};

struct BarrierInputs
{
  double joint_state_age_s = std::numeric_limits<double>::infinity();
  double max_joint_error_to_start_rad = std::numeric_limits<double>::infinity();
  double rest_duration_s = 0.0;
  bool controllers_active = false;
  bool hardware_active = false;
  double snapshot_age_s = std::numeric_limits<double>::infinity();
  bool stop_latched = true;
  double monitor_age_s = std::numeric_limits<double>::infinity();   // time since the last monitor tick
  bool monitor_healthy = false;                                     // last tick found no stop condition
};

/// ok iff every barrier holds. failures lists each violated one.
Status evaluateBarriers(const BarrierInputs & in, const BarrierLimits & limits, std::vector<std::string> * failures = nullptr);

// ---- monitor ----------------------------------------------------------------------------------

struct MonitorLimits
{
  double joint_state_stale_s = 0.4;
  double stop_distance_m = 0.03;
  double tracking_stop_rad = 0.2;
  double tracking_warn_rad = 0.1;
  double status_poll_stale_s = 1.0;   // controller / hardware poll must have answered this recently
};

struct MonitorInputs
{
  bool have_joint_state = false;
  double joint_state_age_s = std::numeric_limits<double>::infinity();
  bool distance_valid = false;
  double cross_distance_m = std::numeric_limits<double>::quiet_NaN();   // conservative (tolerance subtracted)
  bool unexpected_goal_end = false;                                      // aborted / canceled / rejected, not ours
  std::string goal_end_reason;
  bool poll_valid = false;
  double poll_age_s = std::numeric_limits<double>::infinity();
  bool controllers_active = false;
  bool hardware_active = false;
  double tracking_error_rad = 0.0;
};

struct MonitorDecision
{
  bool stop = false;
  bool warn = false;
  std::string reason;   // first stop condition found, or the warning text
};

/// Stop conditions are evaluated in a fixed order; any invalid input (no joint state, no distance, no
/// poll) while the phase is executing is itself a stop condition (fail closed).
MonitorDecision evaluateMonitor(const MonitorInputs & in, const MonitorLimits & limits);

}  // namespace dual_cut
