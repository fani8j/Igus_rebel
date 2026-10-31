#ifndef IGUS_REBEL__REBEL_SAFETY_HPP_
#define IGUS_REBEL__REBEL_SAFETY_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "igus_rebel/CriMessages.hpp"

// Pure helpers used by the Rebel hardware interface. They have no I/O and no ROS
// dependency so that the safety-relevant arithmetic can be unit tested without a robot.
namespace Igus
{
// jog [%] = JOINT_VELOCITY_SCALE * velocity [deg/s]. Empirical scale of the original driver.
constexpr double JOINT_VELOCITY_SCALE = 2.0;
// Optional hardware parameter jog_velocity_scale replaces JOINT_VELOCITY_SCALE (default unchanged).
// Sanity range: a lower value commands LESS jog per velocity, a higher one more; the +-100 % jog
// clamp applies in every case.
constexpr double kMinJogVelocityScale = 0.5;
constexpr double kMaxJogVelocityScale = 4.0;

// ---- Cartesian jog (CRI "CMD MotionTypeCartBase" + ALIVEJOG X Y Z A B C), see Safety::CartJog* ----
// The two scales are PLACEHOLDERS: how many percent of the controller's maximum Cartesian speed one
// m/s or one rad/s corresponds to, and the meaning of the A/B/C axes, are NOT documented and have
// NOT been measured on the arm yet (to be measured on the real robot). The real protection is the
// per-axis clamp kDefaultCartJogMaxPercent (10 %), whatever the true mapping is; the first real
// test must be done at low speed with the operator at the e-stop.
constexpr double kDefaultCartJogPercentPerMps = 200.0;    // 0.05 m/s -> 10 % (unmeasured)
constexpr double kDefaultCartJogPercentPerRadps = 100.0;  // 0.10 rad/s -> 10 % (unmeasured)
constexpr double kDefaultCartJogMaxPercent = 10.0;
constexpr double kDefaultCartJogTimeoutS = 0.15;
constexpr double kMinCartJogTimeoutS = 0.03;
constexpr double kMaxCartJogTimeoutS = 1.0;
// No cart_jog_cmd for this long while the mode is on: leave it (0 disables). Covers a teleop that
// died (SIGKILL) and left the controller in Cartesian mode.
constexpr double kDefaultCartJogIdleExitS = 10.0;
constexpr double kMaxCartJogIdleExitS = 3600.0;
constexpr double kMaxCartJogScale = 2000.0;  // sanity bound of the two scale parameters
constexpr const char * kDefaultCartJogBaseFrame = "base_link";

// ---- Direct joint jog (~/joint_jog_enable, ~/joint_jog_cmd): ALIVEJOG joint % in MotionTypeJoint ----
// jog_i [%] = jog_velocity_scale * velocity_i [deg/s], clamped to +-joint_jog_max_percent. The mode
// switches nothing on the controller; the real protection is that clamp (10 % by default).
constexpr double kDefaultJointJogMaxPercent = 10.0;
constexpr double kDefaultJointJogTimeoutS = 0.15;
constexpr double kMinJointJogTimeoutS = 0.03;
constexpr double kMaxJointJogTimeoutS = 1.0;
// No joint_jog_cmd for this long while the mode is on: leave it (0 disables).
constexpr double kDefaultJointJogIdleExitS = 10.0;
constexpr double kMaxJointJogIdleExitS = 3600.0;

namespace Safety
{
using SteadyClock = std::chrono::steady_clock;

// pi / 180
constexpr double kDegToRad = 0.0174532925199432957692369076848861271344287188854172545609719144;

// CRI jog values are percentages; anything outside [-100, 100] is clamped.
constexpr float kMaxJogPercent = 100.0f;

// ALIVEJOG period.
constexpr std::chrono::milliseconds kAlivePeriod{10};

// The ALIVEJOG thread sends a zero jog if write() has not refreshed the command for this
// long (5 missed cycles of the 100 Hz controller manager).
constexpr std::chrono::milliseconds kCommandWatchdogTimeout{50};

// read() reports ERROR if no valid STATUS arrived for this long. The robot sends STATUS
// every 20-30 ms (measured on the real arm), so this is ~8 missed messages.
constexpr std::chrono::milliseconds kStatusStaleTimeout{200};

// STATUS age tiers after kStatusStaleTimeout. Up to kStatusCoastTimeout the arm "coasts": the
// last jog sent is frozen and re-sent (new write() commands are ignored, they are computed on
// frozen state), so a single retransmit on the robot link does not make the arm stop and jerk.
// Beyond it (up to kStatusLatchTimeout) it is a "hold": jog is forced to zero, read() stays OK, and control resumes when STATUS returns. A longer gap (or a
// reconnect) latches the hardware error. The robot link showed single lost segments stalling
// STATUS by one TCP RTO (~200 ms), which must not require an operator reset.
constexpr std::chrono::milliseconds kStatusCoastTimeout{400};
constexpr std::chrono::milliseconds kStatusLatchTimeout{1000};

// A robot-owned joint move (CRI "CMD Move Joint") keeps running on the controller regardless of
// this driver, so there is no coasting through a STATUS gap: the supervisor cannot see a path
// deviation or a stall while STATUS is missing, and only CMD Move Stop (which also needs the
// link) ends it. The move is therefore aborted (Move Stop + latch) as soon as STATUS is stale,
// i.e. at the tighter kStatusStaleTimeout instead of the jog "coast" tier kStatusCoastTimeout.
constexpr std::chrono::milliseconds kMoveStatusAbortTimeout = kStatusStaleTimeout;

// CMD Move Stop is re-sent at this period until STATUS confirms the robot stands still (a TCP
// write that succeeded is not a confirmation: the controller may be in the middle of a retransmit
// or may ignore the command).
constexpr std::chrono::milliseconds kMoveStopResendPeriod{100};
// Standstill (joint change below the supervisor's epsilon on fresh STATUS) for this long
// confirms a stop.
constexpr std::chrono::milliseconds kMoveStopStillWindow{300};
// Largest joint change [rad] between two STATUS that still counts as standing still.
constexpr double kMoveStillEpsilonRad = 0.0005;
// Bit i = joint i reports 'Motor not enabled': all six joints (see Rebel::drivesDisabledMask_).
constexpr unsigned kAllDrivesDisabledMask = 0x3Fu;

// Next-send deadline of a repeated command (CMD Move Stop) with an explicit "send now" that a
// concurrent deadline update can never overwrite. The sender observes the timer BEFORE it sends
// and defers with that token afterwards; ArmNow() (a new request) changes the value to a fresh
// negative number, so the compare-and-swap of a sender that started earlier fails and the
// "send now" survives. Negative values are always due; deadlines are steady-clock ns (> 0).
class ResendTimer
{
public:
  using Token = std::int64_t;
  void ArmNow() {value_.store(-(resets_.fetch_add(1) + 1));}
  Token Observe() const {return value_.load();}
  bool Due(std::int64_t nowNs) const {return value_.load() <= nowNs;}
  // False if ArmNow() (or another Defer) happened since Observe().
  bool Defer(Token observed, std::int64_t deadlineNs)
  {
    return value_.compare_exchange_strong(observed, deadlineNs);
  }

private:
  std::atomic<std::int64_t> value_{0};
  std::atomic<std::int64_t> resets_{0};
};

// A STATUS hold may only auto-resume if the arm was idle. write() commands whose largest joint
// |velocity| exceeds this [rad/s] count as motion (normal idle JTC hold commands are ~P*1e-3).
constexpr double kHoldIdleVelocityRadS = 0.02;
// Motion is looked for from this long before the last STATUS preceding the gap until resume.
constexpr std::chrono::milliseconds kHoldMotionLookback{100};

// on_init fails if no valid STATUS arrives within this time after start-up.
constexpr std::chrono::milliseconds kFirstStatusTimeout{5000};

// Statuses closer together than this (bursts after a delayed STATUS, two in one TCP
// segment) are not used as a velocity baseline; the velocity is computed over the longer
// interval instead. Below the nominal 20-30 ms STATUS period, so normally every STATUS
// updates the velocity; limits the error from arrival jitter to roughly 0.6x-1.5x.
constexpr std::chrono::milliseconds kMinVelocityInterval{15};

// Fault groups: a STATUS drive/kinematics error only counts as "stopped by a fault" once this
// instance was armed, i.e. it saw a fully clean STATUS at least this long after the start-up
// sequence (SetActive/Reset/Enable) was sent. Keeps the normal disabled state before the first
// enable, and the transient while Reset/Enable are processed, from tripping the group.
constexpr std::chrono::milliseconds kFaultArmSettleTime{500};

// Fault groups: does this CRI kinematics state count as "the controller stopped this arm"?
// This is the ONE place that decides. It is deliberately strict for bring-up: every state other
// than NO_ERROR trips the group (also JOINT_LIMIT_MIN/MAX, singularities, virtual-box limits).
// Candidates to whitelist later, once the real behaviour is known, because a tool that merely
// touches a joint limit or virtual box may still be recoverable by jogging away:
// JOINT_LIMIT_MIN, JOINT_LIMIT_MAX (and possibly TOOL_AT_VIRTUAL_BOX_LIMIT_1..6). A state that
// is not tripping is treated as clean STATUS for arming and for clearing a root-cause fault.
inline bool IsGroupTrippingKinstate(CriMessages::Kinstate kinstate)
{
  return kinstate != CriMessages::Kinstate::NO_ERROR;
}

// Velocity override requested at start-up and the tolerance for the STATUS check.
constexpr float kExpectedOverridePercent = 100.0f;
constexpr float kOverrideTolerancePercent = 0.5f;

struct JogCommand
{
  float percent = 0.0f;
  bool nonFinite = false;  // the input (or the scaled value) was NaN/inf; percent is 0
  bool saturated = false;  // the scaled value exceeded +-kMaxJogPercent and was clamped
};

// Converts a joint velocity command [rad/s] to a CRI jog percentage:
// scale * velocity / degToRad (scale defaults to JOINT_VELOCITY_SCALE), non-finite -> 0, clamped
// to [-100, 100].
JogCommand JogPercentFromVelocity(
  double velocityRadPerSec, double scale = JOINT_VELOCITY_SCALE);

// Largest accepted value of the driver parameter jtc_jog_deadband_percent.
constexpr double kMaxJtcJogDeadbandPercent = 2.0;

// True if `deadbandPercent` is finite and within [0, kMaxJtcJogDeadbandPercent].
bool IsValidJogDeadband(double deadbandPercent);

// Optional deadband of the ros2_control (JTC) jog path only: every joint whose |percent| is
// strictly below `deadbandPercent` is set to 0.0; |percent| == deadband is kept, the sign of a
// kept value is unchanged. A deadband <= 0, NaN or infinite (invalid) changes nothing, and a
// non-finite jog value is passed through untouched (FormatAliveJog turns it into 0 as before).
// Never apply this to the Cartesian jog, the joint jog topic or the move_joint paths.
std::array<float, 6> ApplyJogDeadband(
  const std::array<float, 6> & jogPercent, double deadbandPercent);

// Parses the optional hardware parameter jog_velocity_scale. True and *scale set only for a
// finite number in [kMinJogVelocityScale, kMaxJogVelocityScale] with no trailing characters.
bool ParseJogVelocityScale(const std::string & text, double & scale);

struct CartJogConfig
{
  double percentPerMps = kDefaultCartJogPercentPerMps;
  double percentPerRadps = kDefaultCartJogPercentPerRadps;
  double maxPercent = kDefaultCartJogMaxPercent;  // per-axis clamp, <= 100
  bool allowRotation = false;
  // Per-axis sign (+1 or -1) for x, y, z, rx, ry, rz -> ALIVEJOG slots 1..6.
  std::array<double, 6> sign{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double timeoutS = kDefaultCartJogTimeoutS;
  double idleExitS = kDefaultCartJogIdleExitS;
  std::string baseFrame = kDefaultCartJogBaseFrame;
};

struct CartJogResult
{
  std::array<float, 6> percent{};
  bool nonFinite = false;        // at least one input axis was NaN/inf (that axis is 0)
  bool saturated = false;        // at least one axis was clamped to maxPercent
  bool rotationDropped = false;  // rotation disabled and a non-zero angular input was zeroed
};

// twist = {vx, vy, vz [m/s], wx, wy, wz [rad/s]}. percent = sign * scale * value, non-finite -> 0,
// each axis clamped to +-min(maxPercent, 100); angular axes are 0 unless allowRotation.
CartJogResult CartJogFromTwist(const CartJogConfig & config, const std::array<double, 6> & twist);

struct JointJogConfig
{
  double maxPercent = kDefaultJointJogMaxPercent;  // per-joint clamp, <= 100
  double timeoutS = kDefaultJointJogTimeoutS;
  double idleExitS = kDefaultJointJogIdleExitS;
};

struct JointJogResult
{
  std::array<float, 6> percent{};
  bool nonFinite = false;  // at least one input was NaN/inf (that joint is 0)
  bool saturated = false;  // at least one joint was clamped to maxPercent
};

// velocities in rad/s per hardware joint (0 for joints not commanded). percent =
// JogPercentFromVelocity(v, scale), non-finite -> 0, then clamped to +-min(maxPercent, 100).
JointJogResult JointJogFromVelocities(
  double scale, double maxPercent, const std::array<double, 6> & velocities);

// Maps a JointJog message onto the hardware's six joints. `names` and `velocities` must have the
// same length, every name must be one of `hardwareJoints` exactly once, and `displacements` must be
// empty. On success `out` holds the velocity per hardware joint (0 for joints not listed). On
// failure `out` is all zero and `error` says why; the whole message is to be rejected.
bool MapJointJogMessage(
  const std::array<std::string, 6> & hardwareJoints, const std::vector<std::string> & names,
  const std::vector<double> & velocities, std::size_t displacementCount,
  std::array<double, 6> & out, std::string & error);

// Direct jog (cart_jog_cmd / joint_jog_cmd) command lifetime, counted from the message STAMP and
// not from its receipt: timeout - age, with age clamped to >= 0 (a stamp slightly in the future
// gets the full timeout) and the result clamped to [0, timeout]. A message that sat in a queue or
// in a paused callback for part of its timeout therefore expires on time.
std::int64_t JogCommandLifetimeNs(double timeoutS, double ageS);

// Replay / ordering rule inside one jog session: a command is accepted only if its stamp is
// strictly newer than the newest accepted one (0 = none yet). Duplicates and older stamps are
// rejected whatever their age.
bool JogStampIsNewer(std::int64_t stampNs, std::int64_t lastAcceptedStampNs);

// Parsers of the optional cart_jog_* hardware parameters (the whole string must parse).
bool ParseBoundedNumber(const std::string & text, double minimum, double maximum, double & value);
bool ParseBoolText(const std::string & text, bool & value);
// "1,1,-1,1,1,1" (also space separated): six values, each exactly +1 or -1.
bool ParseCartJogSigns(const std::string & text, std::array<double, 6> & signs);
// Non-empty; letters, digits, '_', '/', '.', '-' only.
bool IsValidFrameName(const std::string & name);

// True if the last command is older than timeout, or if no command was ever written
// (lastWrite == time_point{}).
bool IsCommandStale(
  SteadyClock::time_point lastWrite, SteadyClock::time_point now,
  std::chrono::nanoseconds timeout);

// True if no valid STATUS was ever received or the last one is older than timeout.
bool IsStatusStale(
  SteadyClock::time_point lastStatus, SteadyClock::time_point now,
  std::chrono::nanoseconds timeout);

enum class StatusHealth
{
  Fresh,  // age <= staleTimeout
  Coast,  // staleTimeout < age <= coastTimeout: re-send the frozen last jog
  Hold,   // coastTimeout < age <= latchTimeout: zero jog, keep waiting
  Latch   // no STATUS ever, or age > latchTimeout: fail closed
};

// Classifies the STATUS age. haveStatus=false (no valid STATUS yet) -> Latch.
StatusHealth ClassifyStatusAge(
  bool haveStatus, std::chrono::nanoseconds age, std::chrono::nanoseconds staleTimeout,
  std::chrono::nanoseconds coastTimeout, std::chrono::nanoseconds latchTimeout);

// May control resume when STATUS returns after a gap that reached `gapTier`? Fresh/Coast: yes
// (the arm kept moving at the frozen jog, the controller state is close). Hold: only if no motion
// was commanded during the gap. Latch: never.
bool GapResumeAllowed(StatusHealth gapTier, bool motionDuringGap);

const char * StatusHealthName(StatusHealth tier);

// True if the largest |commanded joint velocity| [rad/s] exceeds kHoldIdleVelocityRadS
// (non-finite counts as motion).
bool IsCommandMoving(double maxAbsVelocityRadS);

// True if a moving command was seen (lastMovingNs != 0) at or after
// lastStatusBeforeGapNs - lookback (all steady-clock ns). Such a gap must latch, not resume.
bool MotionDuringGap(
  std::int64_t lastMovingNs, std::int64_t lastStatusBeforeGapNs, std::chrono::nanoseconds lookback);

// Velocity between two STATUS samples, divided by the actual time between them.
// Returns false (and leaves velocity untouched) if the interval is shorter than
// minInterval, so the caller keeps the previous estimate and the old baseline.
bool EstimateVelocity(
  const std::array<double, 6> & previousPosition, SteadyClock::time_point previousStamp,
  const std::array<double, 6> & position, SteadyClock::time_point stamp,
  std::chrono::nanoseconds minInterval, std::array<double, 6> & velocity);

// Joint 1-6 positions of a STATUS in radians. False if any of them is not finite.
bool JointPositionsRad(const CriMessages::Status & status, std::array<double, 6> & positions);

// True if the STATUS override is within kOverrideTolerancePercent of the expected value.
bool IsOverrideAsExpected(float overridePercent);

// "CRISTART <counter> ALIVEJOG j1 .. j6 0 0 0 CRIEND\n", 8 decimals, non-finite -> 0.
std::string FormatAliveJog(int counter, const std::array<float, 6> & jogPercent);

// "CRISTART <counter> CMD <command> CRIEND\n"
std::string FormatCommand(int counter, const std::string & command);

// Command text (without counter/framing) of the CRI joint move:
// "Move Joint a1 a2 a3 a4 a5 a6 0 0 0 <vel%> <acc%>". Angles in DEGREES with 3 decimals, the
// external axes e1..e3 are 0. vel/acc are written as integers when integral, else 3 decimals.
// Returns an empty string if any value is not finite.
std::string MoveJointCommandText(
  const std::array<double, 6> & jointsDeg, double velocityPercent, double accelerationPercent);

// "CRISTART <counter> CMD Move Joint ... CRIEND\n"; empty if any value is not finite.
std::string FormatMoveJoint(
  int counter, const std::array<double, 6> & jointsDeg, double velocityPercent,
  double accelerationPercent);
}  // namespace Safety
}  // namespace Igus

#endif  // IGUS_REBEL__REBEL_SAFETY_HPP_
