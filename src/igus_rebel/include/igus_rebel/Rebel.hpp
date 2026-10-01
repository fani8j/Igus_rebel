#ifndef REBEL_HPP_
#define REBEL_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "igus_rebel_msgs/msg/digital_output.hpp"
#include "igus_rebel_msgs/srv/set_digital_output.hpp"
#include "std_srvs/srv/trigger.hpp"
#include <hardware_interface/system_interface.hpp>

#include "igus_rebel/RebelSocket.hpp"
#include "igus_rebel/CriMessages.hpp"
#include "igus_rebel/RebelSafety.hpp"

using namespace hardware_interface;

namespace Igus
{
class Rebel : public SystemInterface
{
private:
  // Latest valid STATUS, written by the message thread and read by read()/on_init.
  struct JointSnapshot
  {
    std::array<double, 6> positionRad{};
    Safety::SteadyClock::time_point stamp{};
    std::uint64_t sequence = 0;  // 0 = no valid STATUS yet
    float overridePercent = 0.0f;
    // Cumulative count of CRI message counter values skipped (any message type) up to this
    // STATUS; a gap that skipped none was a delayed (retransmitted) segment, not lost messages.
    std::uint64_t counterSkipped = 0;
  };

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
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
  std::atomic<bool> hardwareErrorLatched_{false};
  std::string componentName_;

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

  // Thread functions
  void AliveThreadFunction();
  void MessageThreadFunction();

  // Other functions
  int Ccnt();
  bool Command(const std::string &);
  void OnConnected(bool reconnect);
  void ZeroJog();
  void HandleMessage(const std::string &);

  // Function to react to specific status values, to display warnings, error messages, etc.
  void ProcessStatus(const CriMessages::Status &);

public:
  const std::vector<std::string> JOINT_NAME = {
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

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
  bool StatusCoastActive() const {return statusCoast_.load();}
  bool HardwareErrorLatched() const {return hardwareErrorLatched_.load();}

  // Interaction with hardware for ROS2
  CallbackReturn on_init(const HardwareInfo & hardware_info) override;
  // Reached from UNCONFIGURED on every startup and, after a latched hardware error, via
  // `ros2 control set_hardware_component_state <component> inactive`. Does not clear the
  // latch; on_activate() does that, and only if STATUS is fresh.
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State & previous_state) override;

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
