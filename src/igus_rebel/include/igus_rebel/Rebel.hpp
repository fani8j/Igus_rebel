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
  };

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread executorThread;

  std::shared_ptr<RebelSocket> rebelSocket;

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

  // Interaction with hardware for ROS2
  CallbackReturn on_init(const HardwareInfo & hardware_info) override;
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
