// ROS-free logic of the Servo -> JTC trajectory relay (see servo_trajectory_relay.cpp).
//
// Why: MoveIt Servo 2.5.10 anchors every command on the MEASURED joint state and the joint_trajectory_controller
// restarts every stamp-0 trajectory from its current state, so only ~1/3 of the commanded speed is realised. The
// relay integrates Servo's output VELOCITY into a reference position it owns and sends the JTC a three-point
// trajectory whose first point (time_from_start 0) IS that reference, so the JTC follows the reference instead of
// restarting from the measured state every Servo period.
//
// The reference is not integrated separately: at every Servo message it is the PREVIOUSLY PUBLISHED trajectory
// evaluated at the elapsed time (the same cubic Hermite the JTC samples), so it follows what the JTC is executing.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace rebel_servo_relay
{
struct RelayParams
{
  double publish_period_s = 0.034;      // Servo's publish_period (nominal message interval)
  double reset_timeout_s = -1.0;        // gap that re-seeds the reference; <= 0: 3 x publish_period (= horizon)
  double horizon_s = -1.0;              // length of the constant-velocity segment; <= 0: 3 x publish_period
  double stop_time_s = -1.0;            // length of the final constant-deceleration segment to velocity 0;
                                        // < 0: horizon_s; 0, NaN or inf are rejected
  double max_lead_rad = 0.05;           // absolute bound on |reference - measured| per joint
  double lead_time_s = 0.2;             // lead bound per joint: min(max_lead_rad, |v| * lead_time_s + lead_floor_rad)
  double lead_floor_rad = 0.005;
  double joint_limit_margin_rad = 0.02617994;  // reference stays this far inside the URDF limits (same as Servo)
  double joint_state_timeout_s = 0.2;   // measured positions older than this are "stale"
  double silence_hold_s = -1.0;         // no valid Servo message for this long while active -> hold; <= 0: 2 x period
  double zero_velocity_eps = 1e-4;      // rad/s; all |v| below this = Servo halted

  // Fills the derived defaults; throws std::invalid_argument on a non-positive/non-finite parameter.
  void resolve();
};

struct JointLimit
{
  bool has_limits = false;
  double lower = 0.0;
  double upper = 0.0;
};

struct Measured
{
  std::vector<double> position;  // empty = no joint_states for these joints
  double age_s = 1e9;            // age of the oldest sample used
  bool unstamped = false;        // some joint_states sample had header.stamp == 0 (freshness cannot be checked)
};

enum class Action
{
  kNone,     // nothing to publish and nothing to say
  kForward,  // publish {start @0 (velocity), mid @H (velocity_mid), end @H+T_stop (velocity 0)}
  kHold,     // publish a single point at `start`, zero velocity (Servo halted, or a fault/silence while active)
  kReject    // publish nothing; `reason` says why (only when there was no reference to hold)
};

struct RelayOutput
{
  Action action = Action::kNone;
  std::string reason;
  bool fault = false;  // kHold caused by a fault/silence (joint names = the active reference's names)
  // kForward: three points. Per joint: start r (v'), mid r + v' H (v'), end r + v' (H + T_stop / 2) (0), where v' is v
  // scaled down so that the end point does not pass limit -/+ margin: v' = sign(v) min(|v|, d / (H + T_stop/2)) with d the
  // distance to that bound in the direction of v. The last segment is a constant deceleration (a cubic Hermite
  // reproduces it exactly); |velocity| <= |v| everywhere and the path is monotone and inside the bounds.
  std::vector<double> start;         // reference (kForward) or hold position (kHold)
  std::vector<double> mid;           // position at t = horizon
  std::vector<double> end;           // position at t = horizon + stop_time, velocity 0
  std::vector<double> velocity;      // velocity at start (and mid)
  std::vector<double> velocity_mid;  // velocity at mid
};

class RelayCore
{
public:
  explicit RelayCore(RelayParams params);  // calls params.resolve()

  // now_s: monotonic seconds. names/velocity/limits/measured are in the joint order of the Servo message.
  // All validation happens first. A fault while a reference is active returns kHold with the OLD reference's joints
  // (the caller must use the joint names the active trajectory used): at the measured position if it belongs to those
  // joints and is fresh, else at the reference evaluated from the last trajectory. Only a VALID message with a different
  // joint name list (order included) drops the old reference and reseeds.
  RelayOutput process(double now_s, const std::vector<std::string>& names, const std::vector<double>& velocity,
                      const Measured& measured, const std::vector<JointLimit>& limits);

  // Watchdog tick: kHold (once) if active and no valid Servo message for silence_hold_s, else kNone.
  // `measured` is for names(); empty/stale -> the reference is used.
  RelayOutput checkSilence(double now_s, const Measured& measured);

  // Hold now (node-level faults). kHold if active, else kReject with `reason`.
  RelayOutput holdNow(double now_s, const Measured& measured, const std::string& reason);

  // Forget the reference; the next moving message re-seeds it from the measured state.
  void reset();

  bool active() const { return active_; }
  const std::vector<std::string>& names() const { return names_; }
  const std::vector<double>& reference() const { return reference_; }
  const RelayParams& params() const { return params_; }

private:
  struct Trajectory
  {
    std::vector<double> p0, v0, p1, v1, p2;  // times 0, H, H + T_stop; velocity at p2 is 0
  };
  double evaluate(std::size_t joint, double elapsed_s) const;
  RelayOutput reject(const std::string& reason);
  RelayOutput fault(double now_s, const std::vector<std::string>& names, const Measured& measured,
                    const std::string& reason);

  RelayParams params_;
  bool active_ = false;
  double last_time_ = 0.0;          // publish time of traj_
  std::vector<std::string> names_;
  std::vector<double> reference_;
  Trajectory traj_;
};

}  // namespace rebel_servo_relay
