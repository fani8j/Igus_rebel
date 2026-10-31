#ifndef REBEL_HPP_
#define REBEL_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp/node_interfaces/node_parameters_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "igus_rebel_msgs/action/move_joint.hpp"
#include "igus_rebel_msgs/msg/cri_diagnostics.hpp"
#include "igus_rebel_msgs/msg/digital_output.hpp"
#include "igus_rebel_msgs/msg/hardware_state.hpp"
#include "igus_rebel_msgs/msg/robot_cart_pose.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "control_msgs/msg/joint_jog.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "igus_rebel_msgs/srv/set_digital_output.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include <hardware_interface/system_interface.hpp>

#include "igus_rebel/RebelSocket.hpp"
#include "igus_rebel/CriMessages.hpp"
#include "igus_rebel/FaultGroup.hpp"
#include "igus_rebel/MoveSupervisor.hpp"
#include "igus_rebel/RebelSafety.hpp"

using namespace hardware_interface;

namespace Igus
{
class Rebel : public SystemInterface
{
public:
  // Defaults of the optional hardware parameters node_name, digital_output_service and
  // trajectory_controller_name.
  static constexpr const char * kDefaultNodeName = "igus_rebel";
  static constexpr const char * kDefaultDigitalOutputService = "set_digital_output";
  static constexpr const char * kDefaultTrajectoryControllerName =
    "rebel_arm_trajectory_controller";

private:
  // Latest valid STATUS, written by the message thread and read by read()/on_init.
  struct JointSnapshot
  {
    std::array<double, 6> positionRad{};
    Safety::SteadyClock::time_point stamp{};
    // Wall-clock (system_clock) time of the same instant, ns since the Unix epoch: the header
    // stamp of ~/robot_joint_states (the acquisition time of the sample, never "now").
    std::int64_t systemStampNs = 0;
    std::uint64_t sequence = 0;  // 0 = no valid STATUS yet
    float overridePercent = 0.0f;
    // Cumulative count of CRI message counter values skipped (any message type) up to this
    // STATUS; a gap that skipped none was a delayed (retransmitted) segment, not lost messages.
    std::uint64_t counterSkipped = 0;
  };

  rclcpp::Node::SharedPtr node_;
  // Two threads: the move_joint cancel callback must never wait behind reset_and_enable (which
  // blocks up to 3 s). The action server has its own callback group.
  std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread executorThread;

  std::shared_ptr<RebelSocket> rebelSocket;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_enable_srv_;

  rclcpp::Logger logger_;
  rclcpp::Clock throttleClock_{RCL_STEADY_TIME};

  // Current commanded jog [%] for joints 1-6, written by write(), read by the ALIVEJOG thread.
  std::array<std::atomic<float>, 6> jog;
  // steady_clock time of the last write() in ns since the clock's epoch; 0 = never.
  std::atomic<std::int64_t> lastWriteNs{0};

  std::atomic<bool> continueAlive{false};
  std::atomic<bool> continueMessage{false};
  // ALIVEJOG is only transmitted after SetActive/Reset/Enable were sent on this connection.
  std::atomic<bool> aliveTransmit{false};
  std::atomic<bool> stopping{false};
  std::atomic<bool> stopped{false};

  // Set by read() (stale STATUS or an unacknowledged reconnect) and by on_error(); fail-closed
  // and never cleared automatically. Only on_activate() clears it, and only after confirming a
  // fresh STATUS. The component name (from the URDF ros2_control tag) is used in the log/recovery
  // text so the operator has the exact commands to run.
  //
  // WHILE LATCHED /joint_states IS FROZEN AND MUST NOT BE TRUSTED: read() returns ERROR, so
  // ros2_control moves the component to UNCONFIGURED and stops calling read(); the state
  // interfaces keep their last values and joint_state_broadcaster keeps publishing them with
  // fresh stamps (measured: the arm coasted on while /joint_states stayed put). The driver does
  // not touch the state interfaces from any other thread (data race) and does not write NaN into
  // them. Use the node's own telemetry instead, which keeps running while latched, UNCONFIGURED
  // or INACTIVE: ~/robot_joint_states (latest STATUS, stamped with its receive time) and
  // ~/hardware_state (latched, latch_reason, status age, ...), see PublishTelemetry().
  std::atomic<bool> hardwareErrorLatched_{false};
  // First cause of the current latch (set right after the latch flag by LatchHardwareError /
  // OnPeerFault, cleared when the latch clears). latchVersion_ changes on every set/clear.
  mutable std::mutex latchReasonLock_;
  std::string latchReason_;
  std::atomic<std::uint64_t> latchVersion_{0};
  std::string componentName_;
  // Optional hardware parameters (see on_init); the defaults reproduce the single-arm names.
  // Two Igus/Rebel instances in one controller_manager need distinct node_name values (and
  // distinct digital_output_service names, or a service that only one of them provides).
  std::string nodeName_ = kDefaultNodeName;
  std::string digitalOutputService_ = kDefaultDigitalOutputService;
  std::string trajectoryControllerName_ = kDefaultTrajectoryControllerName;
  // Optional hardware parameter jog_velocity_scale: jog [%] = scale * velocity [deg/s]. Default is
  // JOINT_VELOCITY_SCALE (2.0); set once in on_init, read-only afterwards.
  double jogVelocityScale_ = JOINT_VELOCITY_SCALE;
  // Optional hardware parameter fault_group: Rebel instances in one process with the same group
  // name stop together (see FaultGroup.hpp). Empty = not in a group = no change in behaviour.
  // faultMember_ is created before the message thread starts (so it can be read there without
  // synchronisation) but only registered with the group at the end of a successful on_init.
  std::string faultGroup_;
  FaultGroupMemberPtr faultMember_;
  // Message thread (ProcessStatus) and OnConnected: drive/kinematics errors only count as a
  // fault after a clean STATUS kFaultArmSettleTime after the last start-up sequence.
  std::atomic<bool> faultArmed_{false};
  // steady ns when OnConnected() finished its start-up sequence; 0 = not (yet) connected.
  std::atomic<std::int64_t> connectedAtNs_{0};
  // steady ns of the start of the current uninterrupted run of clean STATUS messages (no drive
  // error, Kinstate not tripping, no STATUS gap longer than kStatusStaleTimeout); 0 = no clean
  // run in progress. Arming and the root-cause clear both require kFaultArmSettleTime of it.
  std::atomic<std::int64_t> cleanSinceNs_{0};
  // steady ns of the last STATUS processed by CheckFaultGroupStatus (message thread only).
  std::int64_t lastCheckedStatusNs_{0};
  // Names of the six joints exactly as given in the ros2_control tag (info_.joints), in CRI
  // joint order 1..6. Used for the exported state/command interfaces.
  std::array<std::string, 6> jointNames_;

  // Serialises the start-up command sequence: OnConnected() runs on the socket thread (reconnect),
  // in on_activate() and from the reset_and_enable service (executor thread).
  std::mutex startupLock_;
  // Set while ResetAndEnable() is re-enabling the drives; write() then forces jog to zero.
  std::atomic<bool> holdJogZero_{false};
  // Set by read() while STATUS is late (kStatusStaleTimeout..kStatusLatchTimeout): write() forces
  // jog to zero. Separate from holdJogZero_ (ResetAndEnable). read() thread owns the transitions.
  std::atomic<bool> statusHold_{false};
  // Set by read() while STATUS age is in the coast tier: write() keeps the last jog frozen
  // (still refreshes the write watchdog) instead of storing new commands.
  std::atomic<bool> statusCoast_{false};
  std::uint64_t lastFreshCounterSkipped_ = 0;  // read() thread
  // Message thread: CRI counter tracking.
  int lastCriCounter_ = -1;
  std::uint64_t counterSkipped_ = 0;
  // write() thread -> read(): steady ns of the last write() whose commanded velocity exceeded
  // kHoldIdleVelocityRadS (0 = none) and that command's max |velocity| [rad/s].
  std::atomic<std::int64_t> lastMovingNs_{0};
  std::atomic<double> lastMovingCmd_{0.0};
  // read()-thread only: gap bookkeeping.
  Safety::SteadyClock::time_point lastFreshStamp_{};
  std::uint64_t gapCount_ = 0;
  double gapMaxMs_ = 0.0;
  // Bit i set = STATUS reports "Motor not enabled" (ErrorJoint::MNE) for joint i (0..5).
  // Written by the message thread.
  std::atomic<unsigned> drivesDisabledMask_{0};
  std::thread aliveThread;
  std::thread messageThread;
  std::mutex aliveWakeLock;
  std::condition_variable aliveWake;

  int current_ccnt = 1;
  std::mutex cntLock;

  std::mutex snapshotLock;
  std::condition_variable snapshotCondition;
  JointSnapshot snapshot;

  double vel_cmd[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double pos[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double vel[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // Velocity baseline (position and arrival time of the STATUS the last velocity used).
  std::array<double, 6> last_pos{};
  Safety::SteadyClock::time_point lastPosStamp{};
  std::uint64_t lastConsumedSequence = 0;
  // Socket connection count when on_init finished; a change means a reconnect happened.
  std::uint64_t initialConnectionCount = 0;

  // Message thread only.
  CriMessages::Kinstate lastKinstate = CriMessages::Kinstate::NO_ERROR;
  std::array<int, 16> lastErrorJoints{};
  std::string kinstateMessage;
  bool haveOverride = false;
  float lastOverride = 0.0f;

  std::mutex commandsLock;
  std::unordered_map<int, std::string> unacknowledgedCommands;

  // ROS2 communication
  rclcpp::Service<igus_rebel_msgs::srv::SetDigitalOutput>::SharedPtr digital_output_srv_;

  // ---- Robot-side joint move (CRI "CMD Move Joint"), see ExecuteMoveGoal() ----
  // Once a Move Joint is sent the robot owns the motion: zero jog, a paused ALIVEJOG or a closed
  // socket do NOT stop it, only "CMD Move Stop" does. Hence every stop/fault path calls
  // RequestMoveStop().
  using MoveJointAction = igus_rebel_msgs::action::MoveJoint;
  using MoveGoalHandle = rclcpp_action::ServerGoalHandle<MoveJointAction>;
  enum MoveState : int {kMoveIdle = 0, kMoveSending = 1, kMoveMoving = 2, kMoveStopping = 3};
  // Events of the current move, filled by the message thread, read by the move thread.
  struct MoveEvents
  {
    bool moveAck = false;     // MOVETOEXECACK
    bool cmdAck = false;      // CMDACK for the Move Joint counter
    bool cmdError = false;    // CMDERROR for the Move Joint counter
    std::string cmdErrorText;
    bool endSeen = false;     // MOVETOEXECEND
    int endCode = 0;
    std::string endReason;
    // MOVETOEXECERROR (or an EXECERROR while the move is active): the robot failed the move,
    // e.g. "605 MOVETOEXECERROR 0 0 MoveTo Hardware error _ESTOP". The numeric code 0 does NOT
    // mean success. The first error of the goal is kept. It never confirms a pending Move Stop.
    bool errorSeen = false;
    std::string errorType;    // "MOVETOEXECERROR" or "EXECERROR"
    std::string errorReason;  // everything after the program token
  };

  std::atomic<int> moveState_{kMoveIdle};
  // One goal at a time: set when a goal is accepted, cleared when its result was sent.
  std::atomic<bool> moveGoalBusy_{false};
  std::atomic<bool> moveCancelRequested_{false};
  // "CMD Move Stop" wanted. Sent by the ALIVEJOG thread (within ~10 ms of the request, then every
  // Safety::kMoveStopResendPeriod) and PERSISTENT: a successful socket write is not a
  // confirmation. While it is set jog is held at zero and new moves are refused. It is cleared
  // ONLY when the robot is known to have stopped: (a) MOVETOEXECEND of the move, (b) standstill
  // on fresh STATUS after motion was seen, or (c) all six drives disabled on fresh STATUS (the
  // robot cannot move) -- or when a fresh start-up sequence / on_init resets the driver, or the
  // move that owned it ended without a stop. A MOVETOEXECERROR / EXECERROR (the robot reporting
  // that it failed, e.g. an e-stop) is NOT a confirmation: it faults the move and requests the
  // stop, which then needs one of (a)-(c). Standstill without ever seeing motion does NOT
  // confirm (the robot may not have started yet). Recovery of a never-confirmed stop: e-stop
  // (or switch the drives off) -> "Pending CMD Move Stop confirmed" is logged -> inactive ->
  // active (set_hardware_component_state) -> cycle the controllers.
  std::atomic<bool> moveStopRequested_{false};
  // When the next CMD Move Stop may go out; ArmNow() on every false->true transition of
  // moveStopRequested_ (see Safety::ResendTimer: a concurrent deadline update cannot overwrite it).
  Safety::ResendTimer moveStopTimer_;
  // Standstill tracking for a pending stop with no move thread (ALIVEJOG thread only).
  Move::Joints stopStillRef_{};
  bool stopStillRefValid_ = false;
  Safety::SteadyClock::time_point stopStillSince_{};
  // Incremented by every RequestMoveStop() while a move is active; the move thread compares it
  // with the value at the start of its goal, so no request is ever lost.
  std::atomic<std::uint64_t> moveStopSeq_{0};
  std::atomic<bool> peerFaulted_{false};
  // Between on_activate() and on_deactivate()/on_error(): read()/write() are being called.
  std::atomic<bool> activated_{false};
  std::atomic<int> kinstateValue_{0};
  std::atomic<int> moveCommandCounter_{-1};
  std::mutex moveEventLock_;
  MoveEvents moveEvents_;
  std::mutex moveThreadLock_;
  std::thread moveThread_;
  // Velocity command interfaces of this arm that are claimed by an active controller (bit i =
  // joint i), tracked through perform_command_mode_switch(). claimLock_ makes "accept a goal"
  // and "prepare a claim" mutually exclusive; the pending deadline covers prepare -> perform.
  std::mutex claimLock_;
  std::atomic<unsigned> velocityClaimMask_{0};
  // A controller claimed a velocity interface while a move was active (set under claimLock_ by
  // perform_command_mode_switch, which cannot refuse the switch: Humble still activates the
  // controllers). The move thread aborts and latches on it; cleared when a move starts.
  std::atomic<bool> claimDuringMove_{false};
  // ---- State shared by the two direct jog modes (Cartesian jog and joint jog) ----
  // Latest accepted command of one mode. Written by the subscription thread (CommitDirectJogCmd),
  // read by the ALIVEJOG thread (target/lastCmdNs/deadline under `lock`; the atomics also by
  // AliveFrameStillValid() under the socket write lock, which may only use atomics).
  struct DirectJogCmd
  {
    std::mutex lock;                  // held for a few instructions only
    std::array<float, 6> target{};    // jog [%]; all zero = no motion
    std::int64_t lastCmdNs = 0;       // steady ns of the last accepted command (idle-exit reference)
    std::int64_t lastStampNs = 0;     // ROS ns of the newest accepted stamp of THIS session; 0 = none
    // Steady ns after which the target must no longer be sent: receipt + (timeout - age of the
    // stamp at receipt), so a command that was already old when it arrived expires on time.
    // 0 = no command / expired.
    std::atomic<std::int64_t> deadlineNs{0};
    // Incremented on every commit to `target`, including zeroing for an invalid / rejected message
    // and every session reset. A frame remembers the revision it was built from.
    std::atomic<std::uint64_t> revision{0};
    std::atomic<bool> nonZero{false};  // the current target has a non-zero element
  };
  // Session generation of the direct jog modes (one counter for both): incremented on every entry
  // AND every exit / reset path (service enter/leave, RequestCartExit, RequestJointJogExit, latch,
  // reconnect, deactivate, ...). The ALIVEJOG thread captures it before it builds a frame and a
  // command callback captures it before it validates a message; a frame / commit made in an older
  // session is dropped even if the mode is "active" again by then (disable + re-enable).
  std::atomic<std::uint64_t> jogGeneration_{0};
  void BumpJogGeneration() {jogGeneration_.fetch_add(1);}
  // What an ALIVEJOG frame was built from (captured by the ALIVEJOG thread; generation FIRST).
  struct AliveFrameBuild
  {
    std::uint64_t generation = 0;
    int cartState = 0;       // CartState
    int jointJogState = 0;   // JointJogState
    std::uint64_t revision = 0;  // DirectJogCmd::revision of the active direct mode at build
  };
  enum class JogCommit {kCommitted, kModeOff, kSessionChanged, kStamp, kReplay};

  // ---- Cartesian jog mode (CRI "CMD MotionTypeCartBase"), see SetCartJogEnabled() ----
  // The controller does the IK: ~/cart_jog_cmd (TwistStamped, base frame) is converted to jog [%]
  // for X Y Z A B C and sent in the same ALIVEJOG stream. Off by default; every hardware parameter
  // (cart_jog_*) leaves the joint-jog behaviour byte-identical while the mode is never entered.
  //   Off -> Entering (SetCartJogEnabled(true) checked, MotionTypeCartBase sent) -> Active (CMDACK
  //   and a STATUS with MODE cartbase seen) -> [exit requested] -> Off (MotionTypeJoint CMDACK and
  //   a STATUS with MODE joint). Every exit except the explicit service call (latch, peer fault,
  //   deactivate, error, claim, timeout) only sets cartExitRequested_ (lock-free); the ALIVEJOG
  //   thread then sends zero jog and MotionTypeJoint until confirmed (CartExitStep). A (re)connect
  //   goes straight to Off because OnConnected() re-sends MotionTypeJoint.
  enum CartState : int {kCartOff = 0, kCartEntering = 1, kCartActive = 2};
  struct TrackedCommand
  {
    std::atomic<int> counter{-1};  // CRI counter of the tracked CMD; -1 = none
    std::atomic<bool> acked{false};
    std::atomic<bool> rejected{false};
  };
  Safety::CartJogConfig cartConfig_;  // set in on_init, read-only afterwards
  std::atomic<int> cartState_{kCartOff};
  std::atomic<bool> cartExitRequested_{false};
  // One enter/leave at a time (service threads, tests).
  std::mutex cartTransitionLock_;
  // Latest accepted cart_jog_cmd (jog [%], receipt time, stamp order, deadline, revision).
  DirectJogCmd cartCmd_;
  TrackedCommand cartEnterCmd_;
  TrackedCommand cartExitCmd_;
  // STATUS MODE of the latest STATUS (CriMessages::Mode as int) and a counter that counts STATUS
  // messages; the mode is stored before the counter is incremented (message thread).
  std::atomic<int> statusMode_{static_cast<int>(CriMessages::Mode::UNKNOWN)};
  std::atomic<std::uint64_t> statusModeSeq_{0};
  CriMessages::Mode lastLoggedMode_ = CriMessages::Mode::UNKNOWN;  // message thread only
  // steady ns when the mode last became Active (idle auto-exit reference); 0 = not active.
  std::atomic<std::int64_t> cartActiveSinceNs_{0};
  // node clock (ROS time) ns of the same instant: cart_jog_cmd stamps before it are rejected.
  std::atomic<std::int64_t> cartActiveSinceRosNs_{0};
  // ALIVEJOG thread only: STATUS mode vs driver state cross-check (see ModeGuardZeroesJog).
  bool modeMismatch_ = false;
  Safety::SteadyClock::time_point modeMismatchSince_{};
  bool modeMismatchLatched_ = false;
  std::array<float, 6> cartLastSent_{};
  bool cartWasStale_ = true;
  bool cartExitActive_ = false;
  Safety::SteadyClock::time_point cartExitFirstAt_{};
  Safety::SteadyClock::time_point cartExitLastSendAt_{};
  std::uint64_t cartExitSeqAtSend_ = 0;
  bool cartExitSent_ = false;
  bool cartExitTimeoutLatched_ = false;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr cartJogSrv_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cartJogSub_;
  rclcpp::CallbackGroup::SharedPtr cartSrvGroup_;
  rclcpp::CallbackGroup::SharedPtr cartCmdGroup_;

  bool EnterCartJog(std::string & message);
  bool LeaveCartJog(std::string & message);
  // Lock-free; callable from any thread/path that latches, faults or deactivates.
  void RequestCartExit();
  // ALIVEJOG thread: drives a pending exit (zero jog is guaranteed by the caller's inhibit).
  void CartExitStep();
  // ALIVEJOG thread: true if the latest STATUS MODE contradicts the driver's jog mode (jog is then
  // sent as zero). Active but not cartbase -> leave the mode; Off but cartbase/carttool for longer
  // than a grace period (a MotionTypeJoint that did not take) -> latch. Unknown mode: never trips.
  bool ModeGuardZeroesJog(Safety::SteadyClock::time_point now);
  // ALIVEJOG thread: leaves the mode after cart_jog_idle_exit_s without a cart_jog_cmd.
  void CartIdleStep(Safety::SteadyClock::time_point now);
  // ALIVEJOG thread: jog [%] to send in the Cartesian mode (watchdog, coast freeze).
  // `revision` receives the DirectJogCmd revision the returned frame was built from.
  std::array<float, 6> CartJogForTransmit(
    Safety::SteadyClock::time_point now, std::uint64_t & revision);
  void OnCartJogCmd(const geometry_msgs::msg::TwistStamped & msg);
  // Resets the mode bookkeeping to "off" (on_init, OnConnected).
  void ResetCartJogState();

  // ---- Direct joint jog mode, see SetJointJogEnabled() ----
  // ~/joint_jog_cmd (control_msgs/JointJog, rad/s per named hardware joint) is converted to joint
  // ALIVEJOG [%] with the same scale as the controller path and sent in the same stream. The robot
  // stays in MotionTypeJoint (no CRI mode switch), so there is no entering/leaving handshake:
  // Off <-> Active, both transitions are instant and every exit is lock-free (jointJogState_ is set
  // to off; the ALIVEJOG thread then builds the next frame from jog[], which is zero). Exclusive
  // with the Cartesian jog mode and with every other jog source (controller claims, joint moves).
  enum JointJogState : int {kJointJogOff = 0, kJointJogActive = 1};
  Safety::JointJogConfig jointJogConfig_;  // set in on_init, read-only afterwards
  std::atomic<int> jointJogState_{kJointJogOff};
  // Held for a few instructions only (subscription thread <-> ALIVEJOG thread). One enter/leave at
  // a time through jointJogTransitionLock_.
  std::mutex jointJogTransitionLock_;
  DirectJogCmd jointCmd_;  // latest accepted joint_jog_cmd
  std::atomic<std::int64_t> jointJogActiveSinceNs_{0};
  std::atomic<std::int64_t> jointJogActiveSinceRosNs_{0};
  std::atomic<bool> jointJogWasStale_{true};  // log edge only; set at entry, flipped by the ALIVEJOG thread
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr jointJogSrv_;
  rclcpp::Subscription<control_msgs::msg::JointJog>::SharedPtr jointJogSub_;
  rclcpp::CallbackGroup::SharedPtr jointSrvGroup_;
  rclcpp::CallbackGroup::SharedPtr jointCmdGroup_;

  bool EnterJointJog(std::string & message);
  bool LeaveJointJog(std::string & message);
  // Lock-free and allocation-free; callable from any thread/path that latches, faults or deactivates.
  void RequestJointJogExit();
  // ALIVEJOG thread: leaves the mode after joint_jog_idle_exit_s without a joint_jog_cmd.
  void JointJogIdleStep(Safety::SteadyClock::time_point now);
  // ALIVEJOG thread: jog [%] to send in the joint jog mode (watchdog, no coast).
  std::array<float, 6> JointJogForTransmit(
    Safety::SteadyClock::time_point now, std::uint64_t & revision);
  void OnJointJogCmd(const control_msgs::msg::JointJog & msg);
  void ResetJointJogState();
  // Evaluated by the socket under its write lock right before a non-zero ALIVEJOG goes out: atomics
  // only. Besides latch / hold / move / connection (the socket) it requires the same jog session
  // (generation), the same mode states, and for a frame built in a direct jog mode: no controller
  // claim, STATUS not coasting, the expected STATUS MODE, an unexpired command deadline, and that
  // the command was not replaced by an all-zero one since the frame was built.
  // A frame already handed to the kernel / TCP cannot be retracted: a stop that lands after this
  // check can be preceded by at most one ALIVEJOG frame (one transmit period) built before it.
  bool AliveFrameStillValid(const AliveFrameBuild & build) const;
  bool DirectJogActive(bool cart) const
  {
    return cart ? (cartState_.load() == kCartActive && !cartExitRequested_.load()) :
           jointJogState_.load() == kJointJogActive;
  }
  // Commits one command to `cmd` under its lock, after re-checking that the mode is still active,
  // that the session generation is the one captured at the start of the callback, and (when
  // `percent` is non-null) that the stamp still passes JogStampRejection() and is newer than the
  // newest accepted stamp. `percent == nullptr` commits an all-zero target for an invalid message
  // (watchdog and stamp order untouched). Otherwise drops without changing the target.
  JogCommit CommitDirectJogCmd(
    DirectJogCmd & cmd, bool cart, std::uint64_t generationAtStart,
    const std::array<float, 6> * percent, const builtin_interfaces::msg::Time * stamp,
    double timeoutS, const std::atomic<std::int64_t> & activeSinceRosNs, const char *& why,
    double & ageS);
  // Clears a direct jog command (session start / end): zero target, no deadline, no stamp order,
  // revision bumped.
  static void ClearDirectJogCmd(DirectJogCmd & cmd);
  // Shared by the two direct jog modes. Empty string = acceptable; otherwise why the header.stamp is
  // rejected (zero, older than the watchdog, too far in the future, before the mode entry).
  const char * JogStampRejection(
    const builtin_interfaces::msg::Time & stamp, double timeoutS, std::int64_t activeSinceRosNs,
    double & ageS) const;
  // ALIVEJOG thread: true if a STATUS gap (coast tier or worse, or a stale snapshot) forbids any
  // non-zero direct jog: those modes never coast.
  bool StatusGapZeroesDirectJog(Safety::SteadyClock::time_point now);

  // Serializes the actual CMD Move Stop send (ALIVEJOG thread, Stop()) with the CMD Move Joint
  // send, and makes "may this move still go out?" atomic with the send.
  // Lock order (outer -> inner), never the reverse:
  //   claimLock_ -> moveEventLock_
  //   moveSendLock_ -> moveEventLock_ (short, never held across a send)
  //   moveSendLock_ -> startupLock_ (try_lock only) / commandsLock / the socket write lock
  // moveEventLock_ is a leaf for everything else. LatchHardwareError()/RequestMoveStop() take no
  // lock that is held while moveSendLock_ or moveEventLock_ is taken (atomics, a leaf string lock
  // and the fault group), and are never called with moveSendLock_ or moveEventLock_ held.
  std::mutex moveSendLock_;
  std::atomic<bool> failNextMoveSend_{false};
  std::shared_ptr<std::function<void()>> beforeStopSendHook_;
  std::shared_ptr<std::function<void()>> beforeMoveSendHook_;
  std::shared_ptr<std::function<void()>> beforeHandleMessageHook_;
  std::shared_ptr<std::function<void()>> beforeMoveFinishHook_;
  std::shared_ptr<std::function<void()>> beforeAliveSendHook_;
  std::shared_ptr<std::function<void(bool)>> beforeJogCommitHook_;
  // STATUS showed joint motion since the current/last CMD Move Joint was sent (see
  // ExecuteMoveGoal): standstill only confirms a Move Stop for a robot that was moving.
  std::atomic<bool> moveMotionSeen_{false};
  bool stopDrivesOff_ = false;  // ALIVEJOG thread only: all drives disabled since stopDrivesOffSince_
  Safety::SteadyClock::time_point stopDrivesOffSince_{};
  bool stopMotionSeen_ = false;  // ALIVEJOG thread only: motion observed while a stop is pending
  std::atomic<std::int64_t> claimPendingUntilNs_{0};
  double moveMaxVelocityPercent_ = Move::kDefaultMaxVelocityPercent;
  double moveMaxAccelerationPercent_ = Move::kDefaultMaxAccelerationPercent;
  std::array<std::array<double, 2>, 6> jointLimits_ = Move::HardwareJointLimits();
  rclcpp::CallbackGroup::SharedPtr moveGroup_;
  rclcpp_action::Server<MoveJointAction>::SharedPtr moveServer_;

  // ---- Telemetry that survives a latch (see hardwareErrorLatched_) ----
  // ~/robot_joint_states: published by the message thread once per NEW valid STATUS. The
  // velocity estimate state below is message-thread only.
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr robotJointStatePub_;
  std::array<double, 6> pubLastPos_{};
  std::int64_t pubLastStampNs_ = 0;  // steady ns of the sample pubLastPos_ belongs to; 0 = none
  std::array<double, 6> pubVel_{};
  bool pubVelValid_ = false;
  // ~/robot_cart_pose: the controller's own POSCARTROBOT of the same STATUS sample (raw), with
  // that sample's joint angles. Message thread only; diagnostic, never read by any control path.
  rclcpp::Publisher<igus_rebel_msgs::msg::RobotCartPose>::SharedPtr robotCartPosePub_;
  // ~/cri_diagnostics: raw per-joint CURRENTJOINTS, set-point/actual position and the jog percent
  // of the last ALIVEJOG frame, once per valid STATUS (message thread). Read-only instrumentation:
  // nothing here is read by any control path.
  rclcpp::Publisher<igus_rebel_msgs::msg::CriDiagnostics>::SharedPtr criDiagnosticsPub_;
  void PublishCriDiagnostics(const JointSnapshot & sample, const CriMessages::Status & status);
  // Where the most recent ALIVEJOG frame came from (jog_source of ~/cri_diagnostics).
  enum JogSource : int
  {
    kJogSourceNone = 0,       // ALIVEJOG not being transmitted
    kJogSourceJtc = 1,        // the ros2_control write() path (jog[])
    kJogSourceJointJog = 2,   // ~/joint_jog_cmd
    kJogSourceCartJog = 3,    // ~/cart_jog_cmd
    kJogSourceStale = 4,      // no fresh write(): watchdog zero
    kJogSourceInhibited = 5,  // latch / hold / move / pending stop / mode guard: zero
    kJogSourceDropped = 6     // a prepared non-zero frame was rejected, zero went out instead
  };
  // Written by the ALIVEJOG thread after each frame, read lock-free by the message thread. The six
  // values and the source are separate atomics: a reader may see a mix of two adjacent frames.
  std::array<std::atomic<float>, 6> lastSentJog_;
  std::atomic<int> lastSentSource_{kJogSourceNone};
  void StoreLastSent(const std::array<float, 6> & jogPercent, int source);
  // Optional deadband [%] of the ros2_control (JTC) jog path only (parameter
  // jtc_jog_deadband_percent of the driver node, [0, 2], default 0 = off). Applied in write().
  std::atomic<double> jtcJogDeadbandPercent_{0.0};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr paramCallbackHandle_;
  // ~/hardware_state: a timer (own callback group, so reset_and_enable cannot stall it) polls
  // every kTelemetryPollPeriod and publishes at 10 Hz and immediately when a flag changed.
  rclcpp::Publisher<igus_rebel_msgs::msg::HardwareState>::SharedPtr hardwareStatePub_;
  rclcpp::CallbackGroup::SharedPtr telemetryGroup_;
  rclcpp::TimerBase::SharedPtr hardwareStateTimer_;
  // Timer callback only.
  std::uint64_t hwPubLastKey_ = 0;
  std::int64_t hwPubLastNs_ = 0;
  std::uint64_t hwPubLastVersion_ = 0;
  void PublishRobotJointStates(const JointSnapshot & sample);
  void PublishRobotCartPose(const JointSnapshot & sample, const CriMessages::Status & status);
  void PublishHardwareState();
  void SetLatchReason(const std::string & reason);
  void ClearLatchReason();

  void RequestMoveStop();
  bool SendMoveStopNow();
  void ConsumeMoveStopRequest();
  bool MoveStopDue() const;
  rclcpp_action::GoalResponse OnMoveGoal(
    const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const MoveJointAction::Goal> goal);
  rclcpp_action::CancelResponse OnMoveCancel(const std::shared_ptr<MoveGoalHandle> goalHandle);
  void OnMoveAccepted(const std::shared_ptr<MoveGoalHandle> goalHandle);
  void ExecuteMoveGoal(const std::shared_ptr<MoveGoalHandle> goalHandle);
  // Returns 0 (SUCCESS) if the goal may be sent, otherwise the result error code; `reason` says
  // why. Called with claimLock_ held. Fills the supervisor inputs.
  std::uint8_t CheckMoveAcceptable(
    const MoveJointAction::Goal & goal, Move::Joints & measured, Move::Joints & start,
    Move::Joints & target, std::string & reason);
  void RecordMoveEvent(const CriMessages::ExecEvent & event, CriMessages::MessageType type);

  // Thread functions
  void AliveThreadFunction();
  void MessageThreadFunction();

  // Other functions
  int Ccnt();
  // quiet: no warning when the send fails (used by the repeated Move Stop retry).
  bool Command(const std::string &, bool quiet = false, TrackedCommand * tracked = nullptr);
  void OnConnected(bool reconnect);
  void ZeroJog();
  // Latches the hardware error. If it was not latched yet and this instance is in a fault group,
  // this is a root-cause fault and every other member is stopped.
  void LatchHardwareError(const std::string & reason);
  // Fault-group callback: another member faulted. Zero jog and latch; never takes a lock.
  void OnPeerFault(const std::string & originName, const std::string & reason);
  // ProcessStatus helper: drive/kinematics error state after the arm was active and enabled.
  void CheckFaultGroupStatus(
    CriMessages::Kinstate kinstate, const std::array<int, 16> & errorJoints);
  std::string FaultGroupRecoveryHint() const;
  // `receivedAt`/`receivedSystemNs`: when the socket read the bytes (not when this thread gets
  // to them); stamps the STATUS snapshot.
  void HandleMessage(
    const std::string &, Safety::SteadyClock::time_point receivedAt, std::int64_t receivedSystemNs);

  // on_init helper: exactly six joints in CRI order with the interfaces this driver exports.
  bool ValidateJoints(const HardwareInfo & hardware_info);

  // Function to react to specific status values, to display warnings, error messages, etc.
  void ProcessStatus(const CriMessages::Status &);

public:
  // Joint names of the exported interfaces come from the ros2_control tag (info_.joints), so
  // prefixed names (arm_left_joint1 ...) work; see JointNames().
  const std::array<std::string, 6> & JointNames() const {return jointNames_;}

  // pi / 180
  const double degToRad = Safety::kDegToRad;

  // Controller address. Default is the ReBeL wired LAN address; override with the
  // ros2_control hardware parameters robot_ip / robot_port (launch arg robot_ip:=...).
  static constexpr const char * kDefaultIp = "192.168.3.11";
  static constexpr int kDefaultPort = 3920;
  std::string ip = kDefaultIp;
  int port = kDefaultPort;

  Rebel();
  // For tests against a local fake CRI server only; the plugin uses Rebel().
  Rebel(const std::string & testIp, int testPort);
  ~Rebel();

  bool SetDigitalOut(const int &, const bool &);

  // Operator-driven re-enable for the case where the robot controller disabled the drives on its
  // own while the driver is healthy. Refuses (returns false, reason in `message`) unless STATUS is
  // fresh, no hardware error is latched and the connection was not re-established. Otherwise
  // re-sends SetActive/Reset/Enable/MotionTypeJoint/Override with jog held at zero until the
  // drives report enabled (max ~3 s). Never called automatically.
  bool ResetAndEnable(std::string & message);
  // Bit i set = joint i (0..5) reports "Motor not enabled" in the latest STATUS.
  unsigned DrivesDisabledMask() const {return drivesDisabledMask_.load();}
  bool StatusHoldActive() const {return statusHold_.load();}
  // Joint move state for tests/diagnostics: 0 idle, 1 sending, 2 moving, 3 stopping.
  int MoveStateValue() const {return moveState_.load();}
  bool MoveGoalBusy() const {return moveGoalBusy_.load();}
  // Bit i set = the velocity command interface of joint i is claimed by a controller.
  unsigned VelocityClaimMask() const {return velocityClaimMask_.load();}
  // Test seams (never used by the plugin): the next CMD Move Joint send reports failure; a hook the
  // ALIVEJOG thread runs between the check and the send of a CMD Move Stop (nullptr clears it).
  void FailNextMoveJointSendForTest() {failNextMoveSend_ = true;}
  void SetBeforeStopSendHookForTest(std::function<void()> hook)
  {
    std::atomic_store(
      &beforeStopSendHook_,
      hook ? std::make_shared<std::function<void()>>(std::move(hook)) :
      std::shared_ptr<std::function<void()>>());
  }
  // Runs under the move/stop send lock just before the CMD Move Joint is written (nullptr clears).
  void SetBeforeMoveSendHookForTest(std::function<void()> hook)
  {
    std::atomic_store(
      &beforeMoveSendHook_,
      hook ? std::make_shared<std::function<void()>>(std::move(hook)) :
      std::shared_ptr<std::function<void()>>());
  }
  // Runs on the message thread after a message was dequeued and before it is handled: a test can
  // delay the processing (a slow consumer) to check receive-time stamps and dropped messages.
  void SetBeforeHandleMessageHookForTest(std::function<void()> hook)
  {
    std::atomic_store(
      &beforeHandleMessageHook_,
      hook ? std::make_shared<std::function<void()>>(std::move(hook)) :
      std::shared_ptr<std::function<void()>>());
  }
  // Runs in the move thread at the start of ending a goal that started a move, before the last
  // look at the robot's events (a test can make a late MOVETOEXECERROR deterministic).
  void SetBeforeMoveFinishHookForTest(std::function<void()> hook)
  {
    std::atomic_store(
      &beforeMoveFinishHook_,
      hook ? std::make_shared<std::function<void()>>(std::move(hook)) :
      std::shared_ptr<std::function<void()>>());
  }
  // Cartesian jog mode (service ~/cart_jog_enable). enable=true refuses (false, reason in
  // `message`) unless the component is active and not latched, STATUS is fresh, no joint move or
  // Move Stop is pending and NO controller claims a command interface of this arm; it then sends
  // CMD MotionTypeCartBase and waits (max 2 s) for CMDACK and STATUS MODE cartbase. enable=false
  // zeroes the jog and sends CMD MotionTypeJoint, confirmed the same way. Blocks up to ~3 s.
  bool SetCartJogEnabled(bool enable, std::string & message);
  // True while the mode is confirmed on and no exit is pending.
  bool CartJogActive() const {return cartState_.load() == kCartActive && !cartExitRequested_.load();}
  // 0 off, 1 entering, 2 active, 3 leaving (exit requested, MotionTypeJoint not confirmed yet).
  int CartJogStateValue() const
  {
    const int state = cartState_.load();
    return (state != kCartOff && cartExitRequested_.load()) ? 3 : state;
  }
  const Safety::CartJogConfig & CartJogSettings() const {return cartConfig_;}
  // Direct joint jog mode (service ~/joint_jog_enable). enable=true refuses (false, reason in
  // `message`) on the same conditions as the Cartesian jog mode, and additionally while the
  // Cartesian jog mode is not off; it sends nothing to the controller (the robot stays in
  // MotionTypeJoint) and requires STATUS mode joint. enable=false zeroes the jog at once.
  bool SetJointJogEnabled(bool enable, std::string & message);
  bool JointJogActive() const {return jointJogState_.load() == kJointJogActive;}
  int JointJogStateValue() const {return jointJogState_.load();}
  const Safety::JointJogConfig & JointJogSettings() const {return jointJogConfig_;}
  // Runs in the ALIVEJOG thread after a NON-ZERO frame was built, right before it is sent (nullptr
  // clears): a test can hold the thread there across a reconnect.
  void SetBeforeAliveSendHookForTest(std::function<void()> hook)
  {
    std::atomic_store(
      &beforeAliveSendHook_,
      hook ? std::make_shared<std::function<void()>>(std::move(hook)) :
      std::shared_ptr<std::function<void()>>());
  }
  // Runs in the cart_jog_cmd / joint_jog_cmd callback (argument true = Cartesian) after the message
  // was validated and right before the commit under the command lock (nullptr clears): a test can
  // hold the callback there across a mode exit + re-entry.
  void SetBeforeJogCommitHookForTest(std::function<void(bool)> hook)
  {
    std::atomic_store(
      &beforeJogCommitHook_,
      hook ? std::make_shared<std::function<void(bool)>>(std::move(hook)) :
      std::shared_ptr<std::function<void(bool)>>());
  }
  std::uint64_t JogGeneration() const {return jogGeneration_.load();}
  // Pretends a controller claims velocity interfaces (bit i = joint i) without latching; 0 clears.
  void SetVelocityClaimMaskForTest(unsigned mask) {velocityClaimMask_.store(mask);}
  // Overrides the remembered STATUS mode (CriMessages::Mode as int) until the next STATUS.
  void SetStatusModeForTest(int mode) {statusMode_.store(mode);}
  bool StatusCoastActive() const {return statusCoast_.load();}
  bool HardwareErrorLatched() const {return hardwareErrorLatched_.load();}
  // Fault groups: drive/kinematics errors are only monitored once armed (see RebelSafety.hpp),
  // and this instance blocks the recovery of its peers while it holds a root-cause fault.
  bool FaultMonitorArmed() const {return faultArmed_.load();}
  bool FaultGroupRootCause() const {return faultMember_ && faultMember_->HasRootFault();}

  // Interaction with hardware for ROS2
  CallbackReturn on_init(const HardwareInfo & hardware_info) override;
  // Reached from UNCONFIGURED on every startup and, after a latched hardware error, via
  // `ros2 control set_hardware_component_state <component> inactive`. Does not clear the
  // latch; on_activate() does that, and only if STATUS is fresh.
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State & previous_state) override;

  // The joint move needs this arm's velocity command interfaces unclaimed: these track the
  // claims, and prepare refuses a claim while a move is active (so the JTC cannot be activated
  // mid-move).
  return_type prepare_command_mode_switch(
    const std::vector<std::string> & start_interfaces,
    const std::vector<std::string> & stop_interfaces) override;
  return_type perform_command_mode_switch(
    const std::vector<std::string> & start_interfaces,
    const std::vector<std::string> & stop_interfaces) override;

  std::vector<StateInterface> export_state_interfaces() override;
  std::vector<CommandInterface> export_command_interfaces() override;
  return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

  void dio_callback(
    const std::shared_ptr<igus_rebel_msgs::srv::SetDigitalOutput::Request> request,
    std::shared_ptr<igus_rebel_msgs::srv::SetDigitalOutput::Response> response);

  void Start();
  void Stop();
};
}  // namespace Igus

#endif
