// Robot-side (CRI "CMD Move Joint") execution of eligible free-space transit legs: pure logic, no ROS node.
//
// WHAT IT IS. In a SERIAL phase of the phases-mode plan an arm's long, straight joint-space transit leg may be
// sent to the robot controller as one igus_rebel_msgs/action/MoveJoint goal instead of the JTC trajectory. The
// robot plans a synchronous joint-linear move itself (measured path deviation <= 0.9 mrad, exact goal, 0.46-0.93x
// of the driver's duration model, scan_results/cri_single_arm_20261022/REPORT.md). Opt-in, default OFF.
//
// WHAT IS HERE (all pure, all unit-tested in test_cri_transit.cpp):
//   * parameter ranges (rejected, never clamped);
//   * the physical constants of the stop and of the duration model, the certificate padding;
//   * plan-time ELIGIBILITY of one Transit segment (geometry only) and the chord resampling the certificate uses;
//   * the split of one arm-phase into [JTC run | CRI run | JTC run] (cri off = one JTC run, as before);
//   * the pre-dispatch gate, the per-run monitor decision, the controller-state gate, the classification of a
//     MoveJoint result (ERROR / latched are FAULT whatever error_code says) and the doStop step order.
//
// SAFETY FACTS the code relies on (driver lead): once a MoveJoint is sent the robot completes it even after
// driver death or cable pull; a controller claim DURING a move stops it and LATCHES the driver; /joint_states is
// FROZEN with fresh stamps while the driver is latched, so everything here that decides about a CRI arm uses the
// driver's live ~/robot_joint_states and ~/hardware_state; an arm is NEVER safe while hardware_state.latched.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

#include "dual_cut_coordinator/gates.hpp"
#include "dual_cut_coordinator/phase_plan.hpp"
#include "dual_cut_coordinator/swept_volume.hpp"
#include "dual_cut_coordinator/trajectory_tools.hpp"

namespace dual_cut
{

// ---- constants --------------------------------------------------------------------------------

constexpr double kCriPi = 3.14159265358979323846;
/// Measured cruise speed of the real arm: 0.66 deg/s per CRI vel % (REPORT.md: 16.6/33.0/49.4/65.4 deg/s at 25/50/75/100 %).
constexpr double kCriMeasuredRadPerSecPerVelPercent = 0.66 * kCriPi / 180.0;
/// Move Stop: travel after the cancel ~ v * 0.06 s + 0.2 deg (measured <= 90 ms / <= 0.72 deg at 25 %). INFORMATION ONLY: since
/// the lead's decision of 2026-10-24 it is NOT part of the certificate padding. A Move Stop decelerates ALONG the robot-side
/// joint-linear path, which is the certified chord itself, and a lost link makes the robot complete the same chord. The extra
/// deviation of an emergency stop applies to JTC legs just the same, so it is not CRI-specific and not certified here.
constexpr double kCriStopDecelS = 0.06;
constexpr double kCriStopExtraRad = 0.2 * kCriPi / 180.0;
/// The driver's duration model (MoveSupervisor.hpp): deliberately slower than the robot (0.60 deg/s per %, 1 rad/s^2 at
/// acc 40, 0.2 s start latency). Used only for the monitor's timeout.
constexpr double kCriModelRadPerSecPerVelPercent = 0.60 * kCriPi / 180.0;
constexpr double kCriModelAccelRadS2AtAcc40 = 1.0;
constexpr double kCriModelStartLatencyS = 0.2;
constexpr double kCriTimeoutFactor = 2.0;
constexpr double kCriTimeoutMarginS = 10.0;
/// Eligibility: every exported waypoint (and every sampled spline point) within this of the straight chord.
constexpr double kCriChordTolRad = 1e-3;
/// Certificate: chord re-sampled with at most this joint step.
constexpr double kCriResampleStepRad = 0.01;
/// The arm has 6 joints; a MoveJoint goal must name exactly these.
constexpr size_t kCriJointCount = 6;

// ---- parameters -------------------------------------------------------------------------------

struct CriParams
{
  bool enabled = false;
  double velocity_percent = 20.0;          // (0, 25]
  double acceleration_percent = 30.0;      // (0, 40]
  double min_length_rad = 0.05;
  double path_tolerance_rad = 0.005;      // measured real path deviation <= 0.9 mrad
  double goal_tolerance_rad = 0.005;
  double start_tolerance_rad = 0.01;
  int max_runs_per_arm_phase = 1;
  double driver_max_velocity_cap_percent = 25.0;
  double driver_max_acceleration_cap_percent = 40.0;
  bool assume_driver_cap_ok = false;       // until HardwareState carries the effective caps
  /// Lever arm of the certificate padding (padding = path_tolerance * reach). 0 (DEFAULT): derive the rigorous bound from the
  /// model, the sum over the six joints of the largest distance from the joint's frame origin to any collision geometry it
  /// carries (armReachBound(): a deviation of path_tolerance in EVERY joint moves no point of the arm by more). > 0: this
  /// constant [m] (e.g. 1.0 = "link + tool reach", which is NOT a rigorous bound for the real arm: 3.6 m).
  double reach_m = 0.0;
  /// Time parameterisation of the JTC sub-chords that remain on both sides of a partial CRI stretch: rest-to-rest quintics along
  /// the chord, leading joint within these limits. Defaults = the executor's transit scalings read as FRACTIONS OF THE
  /// joint_limits.yaml LIMITS: transit_velocity_scaling 0.20 * 0.785 rad/s = 0.157 rad/s, transit_acceleration_scaling 0.0763 *
  /// 5.24 rad/s^2 = 0.400 rad/s^2 (box_h_cut_executor.yaml; the 2026-10-31 cap), applied to the leading joint (the others move proportionally less).
  double subchord_velocity_rad_s = 0.157;
  double subchord_acceleration_rad_s2 = 0.400;
};

/// Range check of ONE parameter (name without the "cri_transit_" prefix handled by the caller): out-of-range
/// values are REJECTED, never clamped. `name` is the full parameter name; unknown names return ok.
Status validateCriParameter(const std::string & name, double value);
Status validateCriParameter(const std::string & name, int64_t value);
/// The whole set (cross checks included).
Status validateCriParams(const CriParams & p);

// ---- stop travel, padding, duration ----------------------------------------------------------

/// Travel of the leading joint after a Move Stop at the given CRI velocity percent: v*0.06 s + 0.2 deg (rad).
double criStopTravelRad(double velocity_percent);
/// Certificate padding: path_tolerance * reach (no stop travel, see kCriStopDecelS). The monitor stops the run when the live
/// state is further than path_tolerance (per joint) from the chord; a per-joint error of that size moves no point of the arm
/// further than path_tolerance * reach (reach = armReachBound() or the constant).
double criPaddingM(double path_tolerance_rad, double reach_m);

/// Rest-to-rest quintic along the chord from -> to (one 2-point trajectory with p, v, a at both ends, which the JTC interpolates
/// as exactly this quintic), leading joint within the limits: T = max(1.875 D / v, sqrt(5.7735 D / a)) for D = leading joint
/// delta. All joints share one profile, so every sample lies on the chord. Kind Transit, no contact. Fails (empty trajectory) on
/// non-finite input, mismatched sizes or non-positive limits.
PlannedSegmentData makeChordSegment(
  const std::vector<std::string> & joint_names, const std::vector<double> & from, const std::vector<double> & to, double velocity_limit_rad_s,
  double acceleration_limit_rad_s2, const std::string & stage_name);
/// Model duration [s] of a trapezoidal move whose largest joint delta is max_delta_rad (the driver's model).
double criModelDurationS(double max_delta_rad, double velocity_percent, double acceleration_percent);
/// 2 x model + 10 s.
double criTimeoutS(double model_s);

/// Rigorous reach bound of one arm for a joint-angle error: for each active joint of the arm the largest distance from
/// the origin of the joint's child frame (the axis passes through it) to any point of the collision geometry of the
/// links it carries (chain of fixed joint origins + |bounding sphere centre| + radius), summed over the joints.
/// A deviation of delta in EVERY joint then moves any point of the arm by at most delta * sum. per_joint (optional)
/// receives the six terms.
Status armReachBound(const moveit::core::RobotModel & model, const ArmSpec & arm, double & sum_m, std::vector<double> * per_joint = nullptr);

// ---- eligibility of one Transit segment -------------------------------------------------------

struct CriGeomLimits
{
  size_t joint_count = kCriJointCount;
  double min_length_rad = 0.05;
  double goal_tolerance_rad = 0.005;     // chord end vs next segment start (and start vs previous end)
  double chord_tolerance_rad = kCriChordTolRad;
};

struct CriLegGeometry
{
  bool eligible = false;
  std::string reason;                    // why not (empty when eligible)
  size_t segment_index = 0;
  std::string stage_name;
  std::vector<std::string> joint_names;  // arm.joint_names order (unprefixed)
  std::vector<double> start, goal;       // same order
  double length_rad = 0.0;               // largest |goal - start| over the joints (the leading joint)
  double max_chord_deviation_rad = 0.0;  // worst waypoint / sample distance from the chord
};

/// Geometry-only eligibility of segment `index` of the arm phase (SegmentKind::Transit, exactly joint_count joints, all
/// waypoints and spline samples within chord_tolerance of the straight chord first->last, monotone progress,
/// length >= min_length, chord end == next segment's first point and chord start == previous segment's last point
/// (arm.start_q / arm.end_q at the phase ends) within goal_tolerance).
CriLegGeometry checkTransitGeometry(const ArmPhase & arm, size_t index, const CriGeomLimits & limits);
/// checkTransitGeometry() of every Transit segment of the arm phase (in segment order); other kinds are skipped.
std::vector<CriLegGeometry> scanTransitLegs(const ArmPhase & arm, const CriGeomLimits & limits);

/// Straight joint-space chord start->goal as >= 2 configurations with at most max_step between consecutive points.
std::vector<std::vector<double>> resampleChord(const std::vector<double> & start, const std::vector<double> & goal, double max_step_rad = kCriResampleStepRad);

// ---- certificate result and the per-arm-phase plan --------------------------------------------

struct CriLegRecord
{
  CriLegGeometry geometry;
  bool certified = false;                // geometry eligible AND the padded checks passed (full chord or a middle stretch)
  bool selected = false;                 // goes to the robot (at most max_runs_per_arm_phase per arm-phase)
  std::string reason;                    // geometry reason, certificate failure, or "not selected: ..."
  double reach_m = 0.0, padding_m = 0.0;
  /// The stretch of the chord that goes to the robot: progress [s0, s1] of the geometry's chord (0, 1 = the whole leg). A
  /// partial stretch is the maximal contiguous middle run of samples that pass with the padding; the rest stays on the JTC as
  /// rest-to-rest sub-chords (head [0, s0], tail [s1, 1]) on the same chord.
  double s0 = 0.0, s1 = 1.0;
  bool partial = false;
  std::vector<double> cri_start, cri_goal;   // chord points at s0 / s1, plan joint order
  double cri_length_rad = 0.0;               // leading joint change of the CRI stretch
  PlannedSegmentData head, tail;             // JTC sub-chords (empty trajectory = none)
  double head_s = 0.0, tail_s = 0.0;         // their durations
  double cross_effective_min_m = std::numeric_limits<double>::quiet_NaN();   // vs the parked other arm
  double cross_required_m = std::numeric_limits<double>::quiet_NaN();
  double table_min_m = std::numeric_limits<double>::quiet_NaN();
  double table_required_m = std::numeric_limits<double>::quiet_NaN();
  double object_min_m = std::numeric_limits<double>::quiet_NaN();            // vs the scene objects (carton, pole, ...)
  double object_required_m = std::numeric_limits<double>::quiet_NaN();
  size_t samples = 0;
};

struct CriArmPhasePlan
{
  std::string phase;                     // "A" / "B"
  std::string side;                      // "left" / "right"
  std::string skipped;                   // non-empty: no leg considered (e.g. "phase runs in parallel")
  std::vector<CriLegRecord> legs;        // every Transit segment
  std::vector<size_t> selected_segments; // sorted segment indices
};

/// Marks at most max_runs eligible+certified legs as selected (longest first, ties: earlier segment); every other
/// certified leg gets reason "not selected: ...". Fills selected_segments (sorted).
void selectCriLegs(CriArmPhasePlan & plan, int max_runs);

struct CriPlan
{
  bool computed = false;                 // plan-time analysis ran (cri enabled, phases mode)
  std::string skipped;                   // why it did not run (e.g. "sequence plan")
  CriParams params;                      // the values the certificate was made with (execution uses THESE)
  std::vector<CriArmPhasePlan> arm_phases;
  const CriArmPhasePlan * find(const std::string & phase, const std::string & side) const;
};

// ---- runs --------------------------------------------------------------------------------------

enum class CriRunType
{
  Jtc,
  Cri
};
const char * toString(CriRunType t);

struct CriRun
{
  CriRunType type = CriRunType::Jtc;
  size_t first_segment = 0;
  size_t end_segment = 0;   // exclusive
  bool operator==(const CriRun & o) const { return type == o.type && first_segment == o.first_segment && end_segment == o.end_segment; }
};

/// [JTC run | CRI run | JTC run | ...] over `segment_count` segments. cri_segments (sorted, unique, < segment_count) each
/// become a one-segment CRI run; the JTC runs between them are maximal; empty JTC runs are dropped. cri_segments empty
/// (CRI off, or nothing selected): ONE JTC run over every segment, which is exactly today's single FJT goal.
std::vector<CriRun> splitArmPhaseRuns(size_t segment_count, const std::vector<size_t> & cri_segments);

/// The run plan of one arm-phase as the execution uses it. CRI off, driver cap not ok, no plan-time analysis for this
/// arm-phase or no selected leg: ONE JTC run over all segments, i.e. exactly the single FJT goal of today.
std::vector<CriRun> criRunsForArmPhase(bool cri_enabled, bool cap_ok, const CriArmPhasePlan * plan, size_t segment_count);

struct CriArmRuns
{
  ArmPhase phase;                          // the arm phase as executed: a selected PARTIAL leg is replaced by [head, leg, tail]
  std::vector<CriRun> runs;                // over phase.segments
  std::vector<const CriLegRecord *> leg;   // parallel to runs: the leg record of a Cri run, nullptr for a Jtc run
};
/// criRunsForArmPhase() on the executed phase. With CRI off (or nothing selected) phase == arm and one JTC run.
CriArmRuns buildCriArmRuns(bool cri_enabled, bool cap_ok, const CriArmPhasePlan * plan, const ArmPhase & arm);

// ---- live measurement helpers -----------------------------------------------------------------

/// Maximum joint speed over a sliding window of the last `window_s`..: |q(now) - q(oldest sample at least min_span_s
/// old)| / dt. Noise of single STATUS samples (1e-3 rad / 33 ms) therefore does not look like motion. Zero span (too
/// few samples) returns +inf so that "at rest" is never claimed without a baseline.
class WindowVelocity
{
public:
  explicit WindowVelocity(double min_span_s = 0.15, double keep_s = 0.5) : min_span_(min_span_s), keep_(keep_s) {}
  void reset() { samples_.clear(); }
  void add(double t_s, const std::vector<double> & q);
  double maxAbsVelocity() const;

private:
  struct S
  {
    double t;
    std::vector<double> q;
  };
  double min_span_, keep_;
  std::deque<S> samples_;
};

// ---- chord supervision ------------------------------------------------------------------------

struct ChordState
{
  double progress = 0.0;                 // unclamped projection on start->goal, 0 = start, 1 = goal
  double deviation_rad = std::numeric_limits<double>::infinity();   // largest per-joint distance from the clamped line point
};
/// Geometric (not time-indexed) deviation of q from the chord. Wrong size or non-finite q: deviation +inf.
ChordState chordState(const std::vector<double> & q, const std::vector<double> & start, const std::vector<double> & goal);
double maxAbsDiff(const std::vector<double> & a, const std::vector<double> & b);

// ---- hardware state, cap, gates ----------------------------------------------------------------

/// The fields of igus_rebel_msgs/msg/HardwareState the coordinator uses (+ the effective MoveJoint caps once the driver
/// publishes them). Defaults are fail closed.
struct CriHwView
{
  bool have = false;
  double age_s = std::numeric_limits<double>::infinity();   // message age (local receive time and header stamp, the larger)
  bool connected = false;
  bool status_fresh = false;
  bool latched = true;
  bool move_stop_pending = true;
  bool move_active = true;
  bool activated = false;
  bool cap_known = false;
  double cap_velocity_percent = std::numeric_limits<double>::quiet_NaN();
  double cap_acceleration_percent = std::numeric_limits<double>::quiet_NaN();
};

/// Health of one arm's component. require_idle: no move in progress (barriers, pre-dispatch, rearm); during a CRI run
/// move_active is expected and this is false. failures (optional) lists every violated condition.
Status evaluateCriHardware(const CriHwView & hw, double max_age_s, bool require_idle, std::vector<std::string> * failures = nullptr);
/// Effective driver caps against the planned percentages: cap known and cap_vel <= driver_max_velocity_cap_percent and
/// cap_acc <= driver_max_acceleration_cap_percent and velocity/acceleration percent within the caps. Unknown cap:
/// refuse unless p.assume_driver_cap_ok.
bool criCapOk(const CriHwView & hw, const CriParams & p, std::string * why = nullptr);

/// Live measurement of one arm (the driver's ~/robot_joint_states), spec joint order.
struct CriLiveView
{
  bool have = false;
  double age_s = std::numeric_limits<double>::infinity();
  std::vector<double> q;
  double rest_s = 0.0;                   // seconds at rest (WindowVelocity + RestTracker)
};

struct CriPreDispatchLimits
{
  double hardware_max_age_s = 0.5;
  double live_max_age_s = 0.2;
  double rest_duration_s = 0.5;
};

struct CriPreDispatchInputs
{
  CriHwView hw;
  CriLiveView live;
  std::vector<double> chord_start;       // spec order
  double start_tolerance_rad = 0.01;
  bool stop_latched = true;
  bool controller_active = false;        // the arm's JTC (it is still active here and is deactivated next)
  bool cap_ok = false;
};

/// Pre-dispatch gate (a): arm at rest on LIVE state (fresh), hardware_state fresh, not latched, no move_stop_pending, no
/// move_active, activated, cap ok, measured start within the start tolerance of the chord start.
Status evaluateCriPreDispatch(const CriPreDispatchInputs & in, const CriPreDispatchLimits & lim, std::vector<std::string> * failures = nullptr);

/// Barrier / next-phase / rearm gate of ONE arm: hardware idle+healthy and live measurement fresh. (Both arms must pass.)
struct CriArmGateLimits
{
  double hardware_max_age_s = 0.5;
  double live_max_age_s = 0.1;
};
Status evaluateCriArmGate(const CriHwView & hw, const CriLiveView & live, const CriArmGateLimits & lim, std::vector<std::string> * failures = nullptr);

struct CriRunMonitorInputs
{
  CriHwView hw;
  CriLiveView live;
  std::vector<double> start, goal;       // spec order
  double path_tolerance_rad = 0.01;
  double elapsed_s = 0.0;                // since the goal was sent
  double timeout_s = 0.0;                // 2 x model + 10
  bool result_received = false;          // the timeout only applies while the move is outstanding
  double hardware_max_age_s = 0.5;
  double live_stale_s = 0.4;
};
/// Per-tick decision for an arm with a MoveJoint run: hardware_state stale / latched / not activated -> FAULT STOP; live
/// measurement missing or stale -> STOP; deviation from the chord > path tolerance -> STOP; timeout -> STOP.
MonitorDecision evaluateCriRunMonitor(const CriRunMonitorInputs & in);

// ---- controller gate ---------------------------------------------------------------------------

/// Per-arm expectation of the JTC. Normal: active. Switching (STRICT switch in progress): active or inactive. CriRun: the
/// JTC is deactivated and the robot owns the arm: inactive.
enum class CtrlGateMode
{
  Normal,
  Switching,
  CriRun
};
const char * toString(CtrlGateMode m);
/// May the controller's reported state be what we see? `poll_after_mode_start` false (the poll was sent before the mode
/// began, so its answer may still show the previous state): the union of the previous and the new expectation.
bool controllerStateAllowed(CtrlGateMode mode, const std::string & state, bool poll_after_mode_start);

// ---- MoveJoint result --------------------------------------------------------------------------

/// igus_rebel_msgs/action/MoveJoint result codes.
enum MoveJointCode : uint8_t
{
  kMjSuccess = 0,
  kMjRejected = 1,
  kMjControllerActive = 2,
  kMjStartDeviation = 3,
  kMjCmdError = 4,
  kMjAckTimeout = 5,
  kMjMotionTimeout = 6,
  kMjPathDeviation = 7,
  kMjStatusGap = 8,
  kMjLatched = 9,
  kMjFault = 10,
  kMjCanceled = 11,
  kMjRobotEnded = 12,
  kMjGoalDeviation = 13
};

enum class ActionOutcome
{
  Succeeded,
  Aborted,
  Canceled,
  Unknown
};

struct MoveJointResultView
{
  ActionOutcome outcome = ActionOutcome::Unknown;
  uint8_t error_code = kMjFault;
  std::string end_reason;
  bool stop_sent = false;
  bool stop_confirmed = false;
  std::string message;
  bool hardware_latched = true;          // hardware_state at the time of the decision
  bool live_valid = false;
  std::vector<double> live_q;            // spec order
};

struct CriVerdict
{
  bool success = false;
  std::string reason;                    // FAULT text (empty on success)
};
/// Success only if the action SUCCEEDED, error_code SUCCESS, end_reason PLAN, the arm is at the goal (goal tolerance) on
/// LIVE state and the hardware is not latched. A robot ERROR (end_reason "ERROR", or latched) is a FAULT whatever
/// error_code says (e.g. CANCELED with stop_confirmed). Anything else is a FAULT.
CriVerdict classifyMoveJointResult(const MoveJointResultView & r, const std::vector<double> & goal, double goal_tolerance_rad);

// ---- doStop ------------------------------------------------------------------------------------

enum class StopAction
{
  CancelMoveJoint,   // -> CMD Move Stop on the robot
  HoldJtc,           // one-point hold trajectory at the measured position (needs the arm's joint states)
  CancelFjt
};
struct StopStep
{
  StopAction action = StopAction::CancelFjt;
  int arm = 0;       // 0 left, 1 right
  bool operator==(const StopStep & o) const { return action == o.action && arm == o.arm; }
};
const char * toString(StopAction a);
/// The order doStop() executes: FIRST every MoveJoint cancel (cri enabled: both arms), THEN the JTC holds (not for an arm
/// whose JTC is not in the Normal mode: it is inactive and its joint states may be frozen), THEN the FJT cancels.
/// cri disabled: today's order (holds, then cancels), no MoveJoint step.
std::vector<StopStep> planStopSteps(bool cri_enabled, const std::array<bool, 2> & jtc_hold_allowed);

}  // namespace dual_cut
