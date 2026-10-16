#ifndef IGUS_REBEL__MOVE_SUPERVISOR_HPP_
#define IGUS_REBEL__MOVE_SUPERVISOR_HPP_

#include <array>
#include <string>
#include <vector>

// Pure helpers for the robot-side joint move (CRI "CMD Move Joint"): request validation, the
// duration model and the supervision of one move along its start->goal line. No I/O and no ROS
// dependency, so the safety arithmetic is unit tested without a robot.
namespace Igus
{
namespace Move
{
using Joints = std::array<double, 6>;

// Driver-side caps on the CRI vel/acc percentages. The hardware parameters
// move_joint_max_velocity_percent / move_joint_max_acceleration_percent may lower the defaults
// but never exceed the hard caps.
constexpr double kDefaultMaxVelocityPercent = 25.0;
constexpr double kDefaultMaxAccelerationPercent = 40.0;
constexpr double kHardMaxVelocityPercent = 50.0;
constexpr double kHardMaxAccelerationPercent = 40.0;

// Sanity caps on the caller-chosen tolerances [rad]; a larger value would defeat the check.
constexpr double kMaxStartTolerance = 0.5;
constexpr double kMaxPathTolerance = 0.5;
constexpr double kMaxGoalTolerance = 0.2;
constexpr double kMaxTimeoutS = 600.0;

// Smallest start->goal distance (largest joint delta) that is worth a move [rad].
constexpr double kMinMoveDistance = 1e-3;

// Progress beyond 1 + kProgressEpsilon (overshoot of the goal) stops the move.
constexpr double kProgressEpsilon = 0.02;
constexpr double kProgressMinMarginRad = 0.003;

// Speed model, deliberately SLOWER than the measurement on the real arm (RobotControl V980-15,
// 2026-10-12: 0.70 deg/s per vel% on J1, ~1.5 rad/s^2 at vel 25 / acc 40) so the derived
// timeout is conservative. The leading (largest) joint delta determines the duration.
constexpr double kModelDegPerSecPerVelPercent = 0.60;
constexpr double kModelAccelRadS2AtFullAcc = 1.0;  // at acc 40 %, scaled linearly with acc %
constexpr double kModelStartLatencyS = 0.2;        // measured 0.10-0.15 s
constexpr double kTimeoutFactor = 2.0;
constexpr double kTimeoutMarginS = 1.0;

// Joint limits of the ReBeL (igus_rebel2.urdf), used when the hardware description carries none.
const std::array<std::array<double, 2>, 6> & HardwareJointLimits();

// Model duration [s] of a trapezoidal move whose largest joint delta is maxDeltaRad.
double ModelDurationS(double maxDeltaRad, double velocityPercent, double accelerationPercent);

struct GoalRequest
{
  std::vector<std::string> jointNames;
  std::vector<double> startPositions;
  std::vector<double> goalPositions;
  double velocityPercent = 0.0;
  double accelerationPercent = 0.0;
  double startTolerance = 0.0;
  double pathTolerance = 0.0;
  double goalTolerance = 0.0;
  double timeoutS = 0.0;
};

struct Caps
{
  double maxVelocityPercent = kDefaultMaxVelocityPercent;
  double maxAccelerationPercent = kDefaultMaxAccelerationPercent;
  std::array<std::array<double, 2>, 6> jointLimits = HardwareJointLimits();
};

// Static checks that need no robot state: names equal the hardware's in order, sizes, finite
// values, vel/acc within (0, cap] (REJECTED, never clamped), tolerances in (0, max], goal and
// start inside the joint limits, a move that is not zero length, timeout >= 0. Returns true if
// the request is acceptable; otherwise `reason` says why. On success `start`/`goal` are filled.
bool ValidateGoalRequest(
  const GoalRequest & request, const std::array<std::string, 6> & hardwareJointNames,
  const Caps & caps, Joints & start, Joints & goal, std::string & reason);

// Largest |a - b| over the six joints.
double MaxAbsDifference(const Joints & a, const Joints & b);

enum class Verdict
{
  Ok,
  PathDeviation,    // a joint is further than path_tolerance from the start->goal line
  ProgressOverrun,  // progress beyond 1 + margin (or before the start by more than the margin)
  Timeout           // elapsed time beyond the timeout
};

const char * VerdictName(Verdict verdict);

struct Sample
{
  double progress = 0.0;   // projection on the start->goal line, 0 = start, 1 = goal
  double deviation = 0.0;  // largest per-joint distance from the line point at that progress
};

// Supervises one move along the straight joint-space line start -> goal.
class MoveSupervisor
{
public:
  MoveSupervisor(
    const Joints & start, const Joints & goal, double velocityPercent, double accelerationPercent,
    double pathTolerance, double goalTolerance, double timeoutOverrideS = 0.0);

  const Joints & start() const {return start_;}
  const Joints & goal() const {return goal_;}
  double pathTolerance() const {return pathTolerance_;}
  double goalTolerance() const {return goalTolerance_;}
  double ModelDurationS() const {return modelDurationS_;}
  // goal timeout_s if > 0, else kTimeoutFactor * model + kTimeoutMarginS.
  double TimeoutS() const {return timeoutS_;}

  Sample Evaluate(const Joints & q) const;
  // Path deviation is checked first, then the progress overrun, then the timeout.
  Verdict Check(const Joints & q, double elapsedS) const;
  // max |q - goal| <= goal_tolerance.
  bool AtGoal(const Joints & q) const;

private:
  Joints start_;
  Joints goal_;
  Joints delta_;
  double deltaNormSquared_ = 0.0;
  double pathTolerance_;
  double goalTolerance_;
  double modelDurationS_ = 0.0;
  double timeoutS_ = 0.0;
};
}  // namespace Move
}  // namespace Igus

#endif  // IGUS_REBEL__MOVE_SUPERVISOR_HPP_
