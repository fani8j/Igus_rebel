/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2023, PickNik LLC
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PickNik LLC nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/*      Title     : servo_keyboard_input.cpp
 *      Project   : moveit_servo
 *      Created   : 05/31/2021
 *      Author    : Adam Pettinger, V Mohammed Ibrahim
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include <control_msgs/msg/joint_jog.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <igus_rebel_msgs/msg/hardware_state.hpp>
#include <igus_rebel_msgs/srv/set_cartesian_increment.hpp>
#if __has_include(<moveit_msgs/srv/servo_command_type.hpp>)
#include <moveit_msgs/srv/servo_command_type.hpp>
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 1
#else
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 0
#include <std_srvs/srv/trigger.hpp>
#endif
#include <std_msgs/msg/int8.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <rclcpp/rclcpp.hpp>
#ifndef WIN32
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#else
#include <conio.h>
#endif

// A single keypress (a tap) holds one velocity command for this long; Servo's own input timeout is shorter.
constexpr double KEYPRESS_DURATION_S = 0.3;
// Terminal auto-repeat sends the held key every ~30-40 ms. A key event that follows the same key
// within REPEAT_GAP_S is a repeat: it only extends the command by REPEAT_HOLD_S, so the arm stops
// at most that long after the key is released (a full KEYPRESS_DURATION_S overran every release).
constexpr double REPEAT_GAP_S = 0.1;
constexpr double REPEAT_HOLD_S = 0.12;
// When a command ends, a zero command is streamed this long. Without it Servo just goes quiet and
// the controller keeps its last point, whose velocity is still nonzero: with the velocity
// interface (ff_velocity_scale 0.4, p 1.0) the arm then crept 0.4 * v past the target and settled
// over ~1 s (the "floaty" stop). A zero command makes Servo send its last position with zero velocity.
constexpr double STOP_PUBLISH_S = 0.2;
constexpr double DEFAULT_CARTESIAN_INCREMENT_M = 0.005;
constexpr double MAX_CARTESIAN_INCREMENT_M = 0.03;
constexpr double DEFAULT_LINEAR_VELOCITY_MPS =
    DEFAULT_CARTESIAN_INCREMENT_M / KEYPRESS_DURATION_S;
constexpr double DEFAULT_ANGULAR_VELOCITY_RPS = 0.15;
constexpr double MIN_LINEAR_VELOCITY_MPS = 0.001;
constexpr double MAX_LINEAR_VELOCITY_MPS = 0.10;
constexpr double MIN_ANGULAR_VELOCITY_RPS = 0.01;
constexpr double MAX_ANGULAR_VELOCITY_RPS = 0.5;
constexpr double SPEED_SCALE_FACTOR = 1.25;
// Joint jog speed (rad/s). It was a fixed 1.0 rad/s (= the 0.785 rad/s joint limit) that '-' / '='
// did not change; with the robot's ~0.5 rad/s^2 deceleration that alone is ~30 deg of stopping travel.
constexpr double DEFAULT_JOINT_VELOCITY_RPS = 0.1;
constexpr double MIN_JOINT_VELOCITY_RPS = 0.01;
constexpr double MAX_JOINT_VELOCITY_RPS = 0.4;
// output:=cri: the driver maps joint rad/s to jog % (jog_velocity_scale * deg/s) and clamps it at
// joint_jog_max_percent, so faster keyboard speeds were silently cut back to the clamp ('=' did nothing).
// The keyboard caps its joint speed at that clamp and starts at this fraction of it.
constexpr double CRI_DEFAULT_JOINT_SPEED_FRACTION = 0.6;
// Terminal auto-repeat starts this long after the first key event (GNOME/X default 500 ms). A first
// press shorter than that stopped the arm and restarted it when the repeats began; with output:=cri a
// first press holds until the repeats can arrive.
constexpr double DEFAULT_KEY_REPEAT_DELAY_S = 0.5;
constexpr double KEY_REPEAT_MARGIN_S = 0.08;

// Define used keys
namespace
{
constexpr int8_t KEYCODE_RIGHT = 0x43;
constexpr int8_t KEYCODE_LEFT = 0x44;
constexpr int8_t KEYCODE_UP = 0x41;
constexpr int8_t KEYCODE_DOWN = 0x42;
constexpr int8_t KEYCODE_PERIOD = 0x2E;
constexpr int8_t KEYCODE_SEMICOLON = 0x3B;
constexpr int8_t KEYCODE_1 = 0x31;
constexpr int8_t KEYCODE_2 = 0x32;
constexpr int8_t KEYCODE_3 = 0x33;
constexpr int8_t KEYCODE_4 = 0x34;
constexpr int8_t KEYCODE_5 = 0x35;
constexpr int8_t KEYCODE_6 = 0x36;
constexpr int8_t KEYCODE_Q = 0x71;
constexpr int8_t KEYCODE_C = 0x63;
constexpr int8_t KEYCODE_R = 0x72;
constexpr int8_t KEYCODE_J = 0x6A;
constexpr int8_t KEYCODE_T = 0x74;
constexpr int8_t KEYCODE_W = 0x77;
constexpr int8_t KEYCODE_E = 0x65;
constexpr int8_t KEYCODE_S = 0x73;
constexpr int8_t KEYCODE_I = 0x69;
constexpr int8_t KEYCODE_K = 0x6b;
constexpr int8_t KEYCODE_N = 0x6e;
constexpr int8_t KEYCODE_M = 0x6d;
constexpr int8_t KEYCODE_U = 0x75;
constexpr int8_t KEYCODE_O = 0x6f;
constexpr int8_t KEYCODE_MINUS = 0x2d;
constexpr int8_t KEYCODE_EQUAL = 0x3d;
}  // namespace

// Some constants used in the Servo Teleop demo
namespace
{
const std::string TWIST_TOPIC = "delta_twist_cmds";
const std::string JOINT_TOPIC = "delta_joint_cmds";
const size_t ROS_QUEUE_SIZE = 10;
const std::string PLANNING_FRAME_ID = "base_link";
const std::string EE_FRAME_ID = "xeg32_tool_tip";
constexpr auto SERVICE_WAIT_TIMEOUT = std::chrono::seconds(5);
constexpr auto SERVICE_RESPONSE_TIMEOUT = std::chrono::seconds(2);
constexpr int KEY_POLL_TIMEOUT_MS = 100;

// Set by the SIGINT/SIGTERM handler; the key loop polls it. The handler only
// touches this lock-free flag (async-signal-safe).
volatile std::sig_atomic_t g_quit_requested = 0;

void requestQuit(int /*signal*/)
{
  g_quit_requested = 1;
}
}  // namespace

// Puts the terminal in raw mode for its lifetime and restores it on destruction.
class KeyboardReader
{
public:
  KeyboardReader() : file_descriptor_(0)
  {
#ifndef WIN32
    if (tcgetattr(file_descriptor_, &cooked_) == 0)
    {
      struct termios raw;
      std::memcpy(&raw, &cooked_, sizeof(struct termios));
      raw.c_lflag &= ~(ICANON | ECHO);
      // Setting a new line, then end of file
      raw.c_cc[VEOL] = 1;
      raw.c_cc[VEOF] = 2;
      terminal_configured_ = tcsetattr(file_descriptor_, TCSANOW, &raw) == 0;
    }
#endif
  }
  ~KeyboardReader()
  {
    shutdown();
  }
  KeyboardReader(const KeyboardReader&) = delete;
  KeyboardReader& operator=(const KeyboardReader&) = delete;

  // Returns true when a key was read, false when the poll timed out or was
  // interrupted by a signal. Throws on read errors or end of input.
  bool readOne(char* c, int timeout_ms)
  {
#ifndef WIN32
    struct pollfd descriptor
    {
      file_descriptor_, POLLIN, 0
    };
    const int ready = poll(&descriptor, 1, timeout_ms);
    if (ready < 0)
    {
      if (errno == EINTR)
        return false;
      throw std::runtime_error(std::string("poll failed: ") + std::strerror(errno));
    }
    if (ready == 0)
      return false;
    const ssize_t rc = read(file_descriptor_, c, 1);
    if (rc < 0)
    {
      if (errno == EINTR)
        return false;
      throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
    }
    if (rc == 0)
      throw std::runtime_error("keyboard input closed");
    return true;
#else
    (void)timeout_ms;
    *c = static_cast<char>(_getch());
    return true;
#endif
  }
  void shutdown()
  {
#ifndef WIN32
    if (terminal_configured_)
    {
      tcsetattr(file_descriptor_, TCSANOW, &cooked_);
      terminal_configured_ = false;
    }
#endif
  }

private:
  int file_descriptor_;
#ifndef WIN32
  struct termios cooked_;
  bool terminal_configured_{ false };
#endif
};

// Converts key-presses to Twist or Jog commands for Servo, in lieu of a controller.
// Parameter output:=cri sends them to the Igus/Rebel driver's own jog modes instead; no MoveIt/Servo
// is involved: Twist -> controller-side Cartesian jog (cart_jog_cmd / cart_jog_enable), JointJog ->
// direct joint jog (joint_jog_cmd / joint_jog_enable). One of the two is selected ('t' / 'j');
// the driver allows only one at a time.
class KeyboardServo
{
public:
  KeyboardServo();
  ~KeyboardServo();
  int keyLoop(KeyboardReader& input);
  // CRI route: zero twist and joint jog, then disable whichever mode is on. No-op in servo mode. Call
  // before the object is destroyed (the spin thread must still run to receive the answer).
  void shutdownCri();

private:
  void setCartesianIncrement(
    const std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Request> request,
    std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Response> response);
  void stopSpinning();
  void publishActiveCommand();
  void logCriJointSpeed() const;
  void queueTwist(const geometry_msgs::msg::TwistStamped& command, double hold_s);
  void queueJoint(const control_msgs::msg::JointJog& command, double hold_s);
  // CRI route: asks the driver to enter/leave a jog mode and waits for the answer.
  enum CriMode : int { CRI_TWIST = 0, CRI_JOINT = 1 };
  bool setCriEnabled(CriMode mode, bool enable);
  // 'j' / 't': switch the selected mode (the other one is disabled first, the answer awaited).
  bool selectCriMode(CriMode mode);
  const char* criModeName(CriMode mode) const { return mode == CRI_JOINT ? "Joint jog" : "Cartesian (twist) jog"; }
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
  void switchCommandType(int8_t command_type, const char* label);
#else
  void startServo();
#endif

  rclcpp::Node::SharedPtr nh_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread spin_thread_;
  std::atomic<bool> spin_finished_{ false };

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Publisher<control_msgs::msg::JointJog>::SharedPtr joint_pub_;
  rclcpp::Service<igus_rebel_msgs::srv::SetCartesianIncrement>::SharedPtr increment_service_;
  rclcpp::TimerBase::SharedPtr command_timer_;
  std::mutex command_mutex_;
  enum class CommandType { NONE, TWIST, JOINT };
  CommandType active_type_{ CommandType::NONE };
  geometry_msgs::msg::TwistStamped active_twist_;
  control_msgs::msg::JointJog active_joint_;
  std::chrono::steady_clock::time_point command_deadline_;
  std::chrono::steady_clock::time_point stop_deadline_;
  bool stopping_{ false };
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
  rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_input_;
#else
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr start_servo_;
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr servo_status_;
  std::atomic<bool> servo_active_{ false };
#endif
  // CRI route (output:=cri)
  bool cri_{ false };
  std::string cri_cmd_topic_{ "/igus_rebel/cart_jog_cmd" };
  std::string cri_enable_service_{ "/igus_rebel/cart_jog_enable" };
  std::string cri_joint_cmd_topic_{ "/igus_rebel/joint_jog_cmd" };
  std::string cri_joint_enable_service_{ "/igus_rebel/joint_jog_enable" };
  std::vector<std::string> cri_joint_names_;  // hardware joint names (joint1..joint6, or arm_left_joint1..)
  bool cri_auto_enable_{ true };
  CriMode cri_selected_{ CRI_TWIST };  // key loop thread only
  // Belief per mode, corrected by the driver's hardware_state.
  std::atomic<bool> cri_on_[2]{};
  std::atomic<std::int64_t> cri_on_set_ns_[2]{};
  rclcpp::Subscription<igus_rebel_msgs::msg::HardwareState>::SharedPtr cri_state_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cri_pub_;
  rclcpp::Publisher<control_msgs::msg::JointJog>::SharedPtr cri_joint_pub_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr cri_enable_client_[2];

  std::atomic<double> cartesian_velocity_mps_;
  std::atomic<double> angular_velocity_rps_;
  double joint_speed_rps_;
  double max_joint_speed_rps_{ MAX_JOINT_VELOCITY_RPS };
  double cri_jog_velocity_scale_{ 1.65 };  // must match the driver's jog_velocity_scale (launch default 1.65 since 2026-10-31)
  double first_press_hold_s_{ KEYPRESS_DURATION_S };
  double joint_direction_;
  std::string command_frame_id_;
};

KeyboardServo::KeyboardServo()
  : cartesian_velocity_mps_(DEFAULT_LINEAR_VELOCITY_MPS),
    angular_velocity_rps_(DEFAULT_ANGULAR_VELOCITY_RPS),
    joint_speed_rps_(DEFAULT_JOINT_VELOCITY_RPS), joint_direction_(1.0), command_frame_id_{ PLANNING_FRAME_ID }
{
  nh_ = rclcpp::Node::make_shared("servo_keyboard_input");

  const std::string output = nh_->declare_parameter("output", std::string("servo"));
  if (output != "servo" && output != "cri")
  {
    throw std::runtime_error("parameter output must be 'servo' or 'cri', got '" + output + "'");
  }
  cri_ = output == "cri";
  cri_cmd_topic_ = nh_->declare_parameter("cri_cmd_topic", cri_cmd_topic_);
  cri_enable_service_ = nh_->declare_parameter("cri_enable_service", cri_enable_service_);
  cri_auto_enable_ = nh_->declare_parameter("cri_auto_enable", cri_auto_enable_);
  cri_joint_cmd_topic_ = nh_->declare_parameter("cri_joint_cmd_topic", cri_joint_cmd_topic_);
  cri_joint_enable_service_ = nh_->declare_parameter("cri_joint_enable_service", cri_joint_enable_service_);
  cri_joint_names_ = nh_->declare_parameter(
    "cri_joint_names", std::vector<std::string>{ "joint1", "joint2", "joint3", "joint4", "joint5", "joint6" });
  if (cri_ && cri_joint_names_.size() != 6)
  {
    throw std::runtime_error("parameter cri_joint_names needs exactly 6 names, got " +
                             std::to_string(cri_joint_names_.size()));
  }
  // Both must match the driver's launch args; a higher value here is still clamped by the driver.
  cri_jog_velocity_scale_ = nh_->declare_parameter("cri_jog_velocity_scale", cri_jog_velocity_scale_);
  const double cri_joint_max_percent = nh_->declare_parameter("cri_joint_jog_max_percent", 10.0);
  const double key_repeat_delay_s = nh_->declare_parameter("key_repeat_delay_s", DEFAULT_KEY_REPEAT_DELAY_S);
  if (cri_)
  {
    if (!(cri_jog_velocity_scale_ > 0.0) || !(cri_joint_max_percent > 0.0) || !(key_repeat_delay_s >= 0.0) ||
        key_repeat_delay_s > 2.0)
    {
      throw std::runtime_error("cri_jog_velocity_scale and cri_joint_jog_max_percent must be > 0, "
                               "key_repeat_delay_s in [0, 2]");
    }
    max_joint_speed_rps_ = std::clamp(cri_joint_max_percent / cri_jog_velocity_scale_ * M_PI / 180.0,
                                      MIN_JOINT_VELOCITY_RPS, MAX_JOINT_VELOCITY_RPS);
    joint_speed_rps_ = CRI_DEFAULT_JOINT_SPEED_FRACTION * max_joint_speed_rps_;
    first_press_hold_s_ = std::max(KEYPRESS_DURATION_S, key_repeat_delay_s + KEY_REPEAT_MARGIN_S);
  }

  if (cri_)
  {
    // base_link only: the driver rejects every other frame_id.
    cri_pub_ = nh_->create_publisher<geometry_msgs::msg::TwistStamped>(cri_cmd_topic_, 2);
    cri_joint_pub_ = nh_->create_publisher<control_msgs::msg::JointJog>(cri_joint_cmd_topic_, 2);
    cri_enable_client_[CRI_TWIST] = nh_->create_client<std_srvs::srv::SetBool>(cri_enable_service_);
    cri_enable_client_[CRI_JOINT] = nh_->create_client<std_srvs::srv::SetBool>(cri_joint_enable_service_);
    const std::string state_topic =
      nh_->declare_parameter("cri_state_topic", std::string("/igus_rebel/hardware_state"));
    // The driver is the truth about the modes (it can leave one by itself: latch, idle timeout).
    cri_state_sub_ = nh_->create_subscription<igus_rebel_msgs::msg::HardwareState>(
      state_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
      [this](igus_rebel_msgs::msg::HardwareState::ConstSharedPtr msg) {
        const std::int64_t now_ns = nh_->get_clock()->now().nanoseconds();
        const bool driver_on[2] = { msg->cart_jog_state == "active", msg->joint_jog_state == "active" };
        const std::string* driver_state_text[2] = { &msg->cart_jog_state, &msg->joint_jog_state };
        for (const CriMode mode : { CRI_TWIST, CRI_JOINT })
        {
          if (driver_on[mode])
            cri_on_[mode].store(true);
          else if (cri_on_[mode].load() && now_ns - cri_on_set_ns_[mode].load() > 500000000LL)
          {
            cri_on_[mode].store(false);
            RCLCPP_ERROR(nh_->get_logger(), "The driver left the %s mode (state '%s'): press %s to re-enable",
                         criModeName(mode), driver_state_text[mode]->c_str(),
                         mode == cri_selected_ ? "'c'" : (mode == CRI_JOINT ? "'j'" : "'t'"));
          }
        }
      });
  }
  else
  {
    twist_pub_ = nh_->create_publisher<geometry_msgs::msg::TwistStamped>(TWIST_TOPIC, ROS_QUEUE_SIZE);
    joint_pub_ = nh_->create_publisher<control_msgs::msg::JointJog>(JOINT_TOPIC, ROS_QUEUE_SIZE);
  }
  command_timer_ = nh_->create_wall_timer(std::chrono::milliseconds(20), [this]() { publishActiveCommand(); });
  increment_service_ = nh_->create_service<igus_rebel_msgs::srv::SetCartesianIncrement>(
    "~/set_cartesian_increment",
    [this](
      const std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Request> request,
      std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Response> response)
    {
      setCartesianIncrement(request, response);
    });

  if (!cri_)
  {
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
    // MoveIt 2.10 and newer require explicitly selecting the active command type.
    switch_input_ = nh_->create_client<moveit_msgs::srv::ServoCommandType>("servo_node/switch_command_type");
#else
    // Humble's servo_node stays idle until ~/start_servo is called.
    start_servo_ = nh_->create_client<std_srvs::srv::Trigger>("servo_node/start_servo");
    servo_status_ = nh_->create_subscription<std_msgs::msg::Int8>(
      "/status", 10, [this](std_msgs::msg::Int8::ConstSharedPtr) { servo_active_.store(true); });
#endif
  }

  // Service callbacks and client responses are handled on a dedicated thread,
  // which is joined in the destructor.
  executor_.add_node(nh_);
  spin_thread_ = std::thread([this]() {
    executor_.spin();
    spin_finished_.store(true);
  });
}

KeyboardServo::~KeyboardServo()
{
  stopSpinning();
}

void KeyboardServo::stopSpinning()
{
  if (!spin_thread_.joinable())
  {
    return;
  }
  // cancel() is a no-op if spin() has not started yet, so repeat until it returns.
  while (!spin_finished_.load())
  {
    executor_.cancel();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  spin_thread_.join();
}

#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
void KeyboardServo::switchCommandType(int8_t command_type, const char* label)
{
  auto request = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
  request->command_type = command_type;
  if (!switch_input_->wait_for_service(std::chrono::seconds(1)))
  {
    RCLCPP_WARN(nh_->get_logger(), "servo_node/switch_command_type unavailable; input not switched to %s",
                label);
    return;
  }
  auto result = switch_input_->async_send_request(request);
  if (result.wait_for(SERVICE_RESPONSE_TIMEOUT) != std::future_status::ready)
  {
    switch_input_->remove_pending_request(result);
    RCLCPP_WARN(nh_->get_logger(), "Timed out switching input to %s", label);
  }
  else if (result.get()->success)
  {
    RCLCPP_INFO(nh_->get_logger(), "Switched to input type: %s", label);
  }
  else
  {
    RCLCPP_WARN(nh_->get_logger(), "Could not switch input to: %s", label);
  }
}
#else
void KeyboardServo::startServo()
{
  if (!start_servo_->wait_for_service(SERVICE_WAIT_TIMEOUT))
  {
    RCLCPP_WARN(nh_->get_logger(),
                "servo_node/start_servo not available after %ld s; start Servo manually "
                "(ros2 service call /servo_node/start_servo std_srvs/srv/Trigger) before jogging",
                static_cast<long>(SERVICE_WAIT_TIMEOUT.count()));
    return;
  }
  // Humble's second start call stops and reinitializes its calculation thread.
  // Status is emitted only from that running thread, so avoid a redundant start.
  const auto status_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!servo_active_.load() && std::chrono::steady_clock::now() < status_deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (servo_active_.load())
  {
    RCLCPP_INFO(nh_->get_logger(), "Servo already active; skipping start_servo");
    return;
  }
  auto result = start_servo_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  if (result.wait_for(SERVICE_RESPONSE_TIMEOUT) != std::future_status::ready)
  {
    start_servo_->remove_pending_request(result);
    RCLCPP_WARN(nh_->get_logger(), "Timed out waiting for servo_node/start_servo response");
    return;
  }
  const auto response = result.get();
  if (response->success)
  {
    RCLCPP_INFO(nh_->get_logger(), "Servo started");
  }
  else
  {
    RCLCPP_WARN(nh_->get_logger(), "servo_node/start_servo failed: %s", response->message.c_str());
  }
}
#endif

void KeyboardServo::setCartesianIncrement(
  const std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Request> request,
  std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Response> response)
{
  const double increment = request->increment_meters;
  if (!std::isfinite(increment) || increment <= 0.0 || increment > MAX_CARTESIAN_INCREMENT_M)
  {
    response->success = false;
    response->message = "increment_meters must be finite and in the range (0, " +
                        std::to_string(MAX_CARTESIAN_INCREMENT_M) + "] m";
    return;
  }

  cartesian_velocity_mps_.store(increment / KEYPRESS_DURATION_S, std::memory_order_relaxed);
  response->success = true;
  response->message = "Cartesian keyboard increment set to " + std::to_string(increment) + " m";
  RCLCPP_INFO(nh_->get_logger(), "%s", response->message.c_str());
}

namespace
{
std::chrono::steady_clock::time_point afterSeconds(double seconds)
{
  return std::chrono::steady_clock::now() +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
}
}  // namespace

void KeyboardServo::queueTwist(const geometry_msgs::msg::TwistStamped& command, double hold_s)
{
  std::lock_guard<std::mutex> lock(command_mutex_);
  active_twist_ = command;
  command_deadline_ = afterSeconds(hold_s);
  stopping_ = false;
  active_type_ = CommandType::TWIST;
}

void KeyboardServo::queueJoint(const control_msgs::msg::JointJog& command, double hold_s)
{
  std::lock_guard<std::mutex> lock(command_mutex_);
  active_joint_ = command;
  command_deadline_ = afterSeconds(hold_s);
  stopping_ = false;
  active_type_ = CommandType::JOINT;
}

void KeyboardServo::logCriJointSpeed() const
{
  const double max_percent = max_joint_speed_rps_ * 180.0 / M_PI * cri_jog_velocity_scale_;
  RCLCPP_INFO(nh_->get_logger(), "CRI joint jog: %.1f %% of robot speed (max %.1f %%)%s",
              joint_speed_rps_ * 180.0 / M_PI * cri_jog_velocity_scale_, max_percent,
              joint_speed_rps_ >= max_joint_speed_rps_ - 1e-9 ? " - at the cap; raise joint_jog_max_percent on the "
                                                                "driver and cri_joint_jog_max_percent here for more" :
                                                                "");
}

void KeyboardServo::publishActiveCommand()
{
  std::lock_guard<std::mutex> lock(command_mutex_);
  if (active_type_ == CommandType::NONE)
    return;
  const auto now = std::chrono::steady_clock::now();
  if (!stopping_ && now >= command_deadline_)
  {
    // The command ended: stream zeros of the same type so Servo commands zero velocity.
    stopping_ = true;
    stop_deadline_ = afterSeconds(STOP_PUBLISH_S);
    active_twist_.twist = geometry_msgs::msg::Twist();
    active_joint_.velocities.assign(active_joint_.velocities.size(), 0.0);
  }
  if (stopping_ && now >= stop_deadline_)
  {
    stopping_ = false;
    active_type_ = CommandType::NONE;
    return;
  }
  if (active_type_ == CommandType::TWIST)
  {
    active_twist_.header.stamp = nh_->now();
    (cri_ ? cri_pub_ : twist_pub_)->publish(active_twist_);
  }
  else
  {
    active_joint_.header.stamp = nh_->now();
    (cri_ ? cri_joint_pub_ : joint_pub_)->publish(active_joint_);
  }
}

bool KeyboardServo::setCriEnabled(CriMode mode, bool enable)
{
  const std::string& service = mode == CRI_JOINT ? cri_joint_enable_service_ : cri_enable_service_;
  const auto& client = cri_enable_client_[mode];
  const char* name = mode == CRI_JOINT ? "joint_jog_enable" : "cart_jog_enable";
  if (!client->wait_for_service(enable ? SERVICE_WAIT_TIMEOUT : std::chrono::seconds(1)))
  {
    RCLCPP_ERROR(nh_->get_logger(), "%s not available", service.c_str());
    return false;
  }
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = enable;
  auto result = client->async_send_request(request);
  // The Cartesian mode waits up to ~3 s for the controller to confirm; the joint mode answers at once.
  if (result.wait_for(std::chrono::seconds(6)) != std::future_status::ready)
  {
    client->remove_pending_request(result);
    RCLCPP_ERROR(nh_->get_logger(), "Timed out waiting for %s", service.c_str());
    return false;
  }
  const auto response = result.get();
  if (!response->success)
  {
    RCLCPP_ERROR(nh_->get_logger(), "%s(%s) refused: %s", name, enable ? "true" : "false",
                 response->message.c_str());
    if (!enable)
      cri_on_[mode] = false;
    return false;
  }
  cri_on_[mode] = enable;
  cri_on_set_ns_[mode] = nh_->get_clock()->now().nanoseconds();
  RCLCPP_WARN(nh_->get_logger(), "%s mode %s: %s", criModeName(mode), enable ? "ON" : "OFF",
              response->message.c_str());
  return true;
}

bool KeyboardServo::selectCriMode(CriMode mode)
{
  const CriMode other = mode == CRI_JOINT ? CRI_TWIST : CRI_JOINT;
  {
    // Drop whatever key command is active: it belongs to the mode being left (the driver zeroes
    // the jog itself when the mode is disabled, and its watchdog backs that up).
    std::lock_guard<std::mutex> lock(command_mutex_);
    active_type_ = CommandType::NONE;
  }
  // Always ask the driver to switch the other mode off and wait for the answer, whatever this node
  // believes (it is idempotent there); only then is the new mode enabled. A refusal aborts, so two
  // modes are never requested at once.
  if (!setCriEnabled(other, false))
  {
    RCLCPP_ERROR(nh_->get_logger(), "Could not switch the %s mode off; staying in the %s selection",
                 criModeName(other), criModeName(cri_selected_));
    return false;
  }
  if (!setCriEnabled(mode, true))
  {
    // The old mode is off and the new one refused: keep the selection on the mode that was asked
    // for so 'c' retries it; nothing moves meanwhile.
    cri_selected_ = mode;
    RCLCPP_ERROR(nh_->get_logger(), "The %s mode was NOT enabled (the %s mode is off); fix the cause and press 'c'",
                 criModeName(mode), criModeName(other));
    return false;
  }
  cri_selected_ = mode;
  return true;
}

void KeyboardServo::shutdownCri()
{
  if (!cri_)
    return;
  {
    // Drop whatever key command is active and stop the republishing timer first.
    std::lock_guard<std::mutex> lock(command_mutex_);
    active_type_ = CommandType::NONE;
  }
  command_timer_->cancel();
  for (int i = 0; i < 3; ++i)
  {
    geometry_msgs::msg::TwistStamped zero;
    zero.header.frame_id = PLANNING_FRAME_ID;
    zero.header.stamp = nh_->now();
    cri_pub_->publish(zero);
    control_msgs::msg::JointJog zero_joint;
    zero_joint.header.stamp = nh_->now();
    zero_joint.joint_names = cri_joint_names_;
    zero_joint.velocities.assign(cri_joint_names_.size(), 0.0);
    cri_joint_pub_->publish(zero_joint);
  }
  if (cri_on_[CRI_JOINT] || cri_selected_ == CRI_JOINT)
    setCriEnabled(CRI_JOINT, false);  // idempotent in the driver
  if (cri_on_[CRI_TWIST] || cri_auto_enable_)
    setCriEnabled(CRI_TWIST, false);  // idempotent in the driver
}

int main(int argc, char** argv)
{
  // The key loop owns SIGINT/SIGTERM handling so the terminal is always restored
  // and the spin thread is joined before rclcpp shuts down.
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, requestQuit);
  std::signal(SIGTERM, requestQuit);

  int rc = 0;
  try
  {
    KeyboardReader input;
    KeyboardServo keyboard_servo;
    rc = keyboard_servo.keyLoop(input);
    keyboard_servo.shutdownCri();
  }
  catch (const std::exception& error)
  {
    fprintf(stderr, "%s\n", error.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}

int KeyboardServo::keyLoop(KeyboardReader& input)
{
  char c;
  bool publish_twist = false;
  bool publish_joint = false;
  // Last key that produced a motion command (arrow keys arrive as ESC '[' 'A', so only the final
  // byte of a motion key is compared).
  char last_motion_key = 0;
  auto last_motion_key_time = std::chrono::steady_clock::time_point();

#if !IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
  if (!cri_)
    startServo();
#endif

  puts("Reading from keyboard");
  puts("---------------------------");
  if (cri_)
  {
    puts("OUTPUT = cri: no Servo; the driver's own jog modes are used (the trajectory controller must be deactivated first).");
    puts("  't' selects twist: Twist goes to the controller-side Cartesian jog (base_link only; arrows, '.', ';', i/k, n/m, u/o).");
    puts("  'j' selects joint jog: keys 1-6 jog the hardware joints directly (robot stays in joint mode); 'r' reverses.");
    puts("  Only one of the two is on at a time: switching turns the other off first and waits for the driver's answer.");
    puts("  'c' toggles the selected mode on/off; 's' stops. Twist keys do nothing in joint jog and vice versa.");
    puts("  The tool frame ('e') is not supported. The driver clamps joint jog to joint_jog_max_percent (default 10 %).");
    printf("A terminal only sends key presses (auto-repeat while held), no key-release events: a press moves for\n"
           "%.2f s (key_repeat_delay_s + %.2f s, so a held key does not stop before auto-repeat starts), each repeat\n"
           "extends it by %.2f s, so the arm stops <= ~%.2f s after the last repeat; then zeros are streamed and the\n"
           "driver's 0.15 s command watchdog zeroes the jog by itself. 's' stops at once.\n",
           first_press_hold_s_, KEY_REPEAT_MARGIN_S, REPEAT_HOLD_S, REPEAT_HOLD_S);
    printf("NOT a release-sensitive deadman: terminal key buffering can extend the motion; one tap at the current\n"
           "joint speed is ~%.1f deg. '-'/'=' change the joint speed up to the driver clamp. The gamepad is the deadman.\n",
           joint_speed_rps_ * 180.0 / M_PI * first_press_hold_s_);
  }
  puts("Twist commands use the selected command frame ('w': base_link, 'e': xeg32_tool_tip)");
  puts("Use arrow keys and the '.' and ';' keys to Cartesian jog");
  puts("Use i/k for TCP roll, n/m for TCP pitch, and u/o for TCP yaw (press 'e' first)");
  puts("Use '-' and '=' to decrease/increase the speed (Cartesian and joint)");
  puts("Use 1|2|3|4|5|6 keys to joint jog. 'r' to reverse the direction of jogging.");
  puts("Use 'j' to select joint jog (output:=cri: also turns the joint jog mode on).");
  puts("Use 't' to select twist (output:=cri: also turns the Cartesian jog mode on).");
  puts("Use 'w' and 'e' to switch between sending command in planning frame or end effector frame");
  printf("Change the Cartesian increment with the set_cartesian_increment service "
         "(default %.3f m, max %.3f m)\n",
         DEFAULT_CARTESIAN_INCREMENT_M, MAX_CARTESIAN_INCREMENT_M);
  puts("'q' or Ctrl-C to quit.");
  printf("Speeds: %.4f m/s, %.3f rad/s, joint %.3f rad/s (%.1f deg/s)\n", cartesian_velocity_mps_.load(),
         angular_velocity_rps_.load(), joint_speed_rps_, joint_speed_rps_ * 180.0 / M_PI);
  if (cri_)
    logCriJointSpeed();

  if (cri_ && cri_auto_enable_ && !setCriEnabled(CRI_TWIST, true))
  {
    RCLCPP_ERROR(nh_->get_logger(), "Cartesian jog mode was NOT enabled; fix the cause and press 'c' to retry");
  }

  while (!g_quit_requested && rclcpp::ok())
  {
    // get the next event from the keyboard
    try
    {
      if (!input.readOne(&c, KEY_POLL_TIMEOUT_MS))
      {
        continue;
      }
    }
    catch (const std::runtime_error& error)
    {
      fprintf(stderr, "%s\n", error.what());
      return -1;
    }

    RCLCPP_DEBUG(nh_->get_logger(), "value: 0x%02X\n", c);

    // // Create the messages we might publish
    auto twist_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
    auto joint_msg = std::make_unique<control_msgs::msg::JointJog>();

    joint_msg->joint_names = cri_ ? cri_joint_names_ :
                                    std::vector<std::string>{ "joint1", "joint2", "joint3", "joint4", "joint5", "joint6" };

    joint_msg->velocities.resize(6);
    std::fill(joint_msg->velocities.begin(), joint_msg->velocities.end(), 0.0);
    const double cartesian_velocity_mps =
      cartesian_velocity_mps_.load(std::memory_order_relaxed);
    const double angular_velocity_rps = angular_velocity_rps_.load(std::memory_order_relaxed);
    if (cri_)
    {
      // Keys that only make sense for the Servo route are explained and ignored; keys of the mode that
      // is not selected only print a hint.
      bool handled = true;
      switch (c)
      {
        case KEYCODE_J:
          selectCriMode(CRI_JOINT);
          break;
        case KEYCODE_T:
          selectCriMode(CRI_TWIST);
          break;
        case KEYCODE_E:
          RCLCPP_WARN(nh_->get_logger(), "Tool-frame twists are not supported in output:=cri (the driver accepts "
                                         "base_link only); command frame stays %s", PLANNING_FRAME_ID.c_str());
          break;
        case KEYCODE_C:
          setCriEnabled(cri_selected_, !cri_on_[cri_selected_]);
          break;
        case KEYCODE_1:
        case KEYCODE_2:
        case KEYCODE_3:
        case KEYCODE_4:
        case KEYCODE_5:
        case KEYCODE_6:
          if (cri_selected_ != CRI_JOINT)
            RCLCPP_WARN(nh_->get_logger(), "Twist mode is selected: joint keys do nothing (press 'j' for joint jog)");
          else
            handled = false;
          break;
        case KEYCODE_LEFT:
        case KEYCODE_RIGHT:
        case KEYCODE_UP:
        case KEYCODE_DOWN:
        case KEYCODE_PERIOD:
        case KEYCODE_SEMICOLON:
        case KEYCODE_I:
        case KEYCODE_K:
        case KEYCODE_N:
        case KEYCODE_M:
        case KEYCODE_U:
        case KEYCODE_O:
          if (cri_selected_ == CRI_JOINT)
            RCLCPP_WARN(nh_->get_logger(), "Joint jog is selected: twist keys do nothing (press 't' for twist)");
          else
            handled = false;
          break;
        default:
          handled = false;
      }
      if (handled)
        continue;
    }
    // Use read key-press
    switch (c)
    {
      case KEYCODE_LEFT:
        RCLCPP_DEBUG(nh_->get_logger(), "LEFT");
        twist_msg->twist.linear.y = -cartesian_velocity_mps;
        publish_twist = true;
        break;
      case KEYCODE_RIGHT:
        RCLCPP_DEBUG(nh_->get_logger(), "RIGHT");
        twist_msg->twist.linear.y = cartesian_velocity_mps;
        publish_twist = true;
        break;
      case KEYCODE_UP:
        RCLCPP_DEBUG(nh_->get_logger(), "UP");
        twist_msg->twist.linear.x = cartesian_velocity_mps;
        publish_twist = true;
        break;
      case KEYCODE_DOWN:
        RCLCPP_DEBUG(nh_->get_logger(), "DOWN");
        twist_msg->twist.linear.x = -cartesian_velocity_mps;
        publish_twist = true;
        break;
      case KEYCODE_PERIOD:
        RCLCPP_DEBUG(nh_->get_logger(), "PERIOD");
        twist_msg->twist.linear.z = -cartesian_velocity_mps;
        publish_twist = true;
        break;
      case KEYCODE_SEMICOLON:
        RCLCPP_DEBUG(nh_->get_logger(), "SEMICOLON");
        twist_msg->twist.linear.z = cartesian_velocity_mps;
        publish_twist = true;
        break;
      case KEYCODE_S:
        RCLCPP_DEBUG(nh_->get_logger(), "S");
        if (cri_ && cri_selected_ == CRI_JOINT)
        {
          publish_joint = true;  // all-zero JointJog (velocities were zero-filled above)
          break;
        }
        twist_msg->twist.linear.x = 0.0;
        twist_msg->twist.linear.y = 0.0;
        twist_msg->twist.linear.z = 0.0;
        publish_twist = true;
        break;
      case KEYCODE_I:
        RCLCPP_DEBUG(nh_->get_logger(), "I");
        twist_msg->twist.angular.x = angular_velocity_rps;
        publish_twist = true;
        break;
      case KEYCODE_K:
        RCLCPP_DEBUG(nh_->get_logger(), "K");
        twist_msg->twist.angular.x = -angular_velocity_rps;
        publish_twist = true;
        break;
      case KEYCODE_N:
        RCLCPP_DEBUG(nh_->get_logger(), "N");
        twist_msg->twist.angular.y = angular_velocity_rps;
        publish_twist = true;
        break;
      case KEYCODE_M:
        RCLCPP_DEBUG(nh_->get_logger(), "M");
        twist_msg->twist.angular.y = -angular_velocity_rps;
        publish_twist = true;
        break;
      case KEYCODE_U:
        RCLCPP_DEBUG(nh_->get_logger(), "U");
        twist_msg->twist.angular.z = angular_velocity_rps;
        publish_twist = true;
        break;
      case KEYCODE_O:
        RCLCPP_DEBUG(nh_->get_logger(), "O");
        twist_msg->twist.angular.z = -angular_velocity_rps;
        publish_twist = true;
        break;
      case KEYCODE_MINUS:
      case KEYCODE_EQUAL:
      {
        const double scale = c == KEYCODE_MINUS ? 1.0 / SPEED_SCALE_FACTOR : SPEED_SCALE_FACTOR;
        const double linear_speed = std::clamp(
          cartesian_velocity_mps * scale, MIN_LINEAR_VELOCITY_MPS, MAX_LINEAR_VELOCITY_MPS);
        const double angular_speed = std::clamp(
          angular_velocity_rps * scale, MIN_ANGULAR_VELOCITY_RPS, MAX_ANGULAR_VELOCITY_RPS);
        joint_speed_rps_ = std::clamp(joint_speed_rps_ * scale, MIN_JOINT_VELOCITY_RPS, max_joint_speed_rps_);
        cartesian_velocity_mps_.store(linear_speed, std::memory_order_relaxed);
        angular_velocity_rps_.store(angular_speed, std::memory_order_relaxed);
        RCLCPP_INFO(nh_->get_logger(), "Speeds: %.4f m/s, %.3f rad/s, joint %.3f rad/s (%.1f deg/s)", linear_speed,
                    angular_speed, joint_speed_rps_, joint_speed_rps_ * 180.0 / M_PI);
        if (cri_)
          logCriJointSpeed();
        break;
      }
      case KEYCODE_1:
        RCLCPP_DEBUG(nh_->get_logger(), "1");
        joint_msg->velocities[0] = joint_direction_ * joint_speed_rps_;
        publish_joint = true;
        break;
      case KEYCODE_2:
        RCLCPP_DEBUG(nh_->get_logger(), "2");
        joint_msg->velocities[1] = joint_direction_ * joint_speed_rps_;
        publish_joint = true;
        break;
      case KEYCODE_3:
        RCLCPP_DEBUG(nh_->get_logger(), "3");
        joint_msg->velocities[2] = joint_direction_ * joint_speed_rps_;
        publish_joint = true;
        break;
      case KEYCODE_4:
        RCLCPP_DEBUG(nh_->get_logger(), "4");
        joint_msg->velocities[3] = joint_direction_ * joint_speed_rps_;
        publish_joint = true;
        break;
      case KEYCODE_5:
        RCLCPP_DEBUG(nh_->get_logger(), "5");
        joint_msg->velocities[4] = joint_direction_ * joint_speed_rps_;
        publish_joint = true;
        break;
      case KEYCODE_6:
        RCLCPP_DEBUG(nh_->get_logger(), "6");
        joint_msg->velocities[5] = joint_direction_ * joint_speed_rps_;
        publish_joint = true;
        break;
      case KEYCODE_R:
        RCLCPP_DEBUG(nh_->get_logger(), "r");
        joint_direction_ *= -1;
        RCLCPP_INFO(nh_->get_logger(), "Joint jog direction: %s", joint_direction_ > 0 ? "+" : "-");
        break;
      case KEYCODE_J:
        RCLCPP_DEBUG(nh_->get_logger(), "j");
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
        switchCommandType(moveit_msgs::srv::ServoCommandType::Request::JOINT_JOG, "JointJog");
#endif
        break;
      case KEYCODE_T:
        RCLCPP_DEBUG(nh_->get_logger(), "t");
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
        switchCommandType(moveit_msgs::srv::ServoCommandType::Request::TWIST, "Twist");
#endif
        break;
      case KEYCODE_W:
        RCLCPP_DEBUG(nh_->get_logger(), "w");
        RCLCPP_INFO_STREAM(nh_->get_logger(), "Command frame set to: " << PLANNING_FRAME_ID);
        command_frame_id_ = PLANNING_FRAME_ID;
        break;
      case KEYCODE_E:
        RCLCPP_DEBUG(nh_->get_logger(), "e");
        RCLCPP_INFO_STREAM(nh_->get_logger(), "Command frame set to: " << EE_FRAME_ID);
        command_frame_id_ = EE_FRAME_ID;
        break;
      case KEYCODE_Q:
        RCLCPP_DEBUG(nh_->get_logger(), "quit");
        return 0;
    }

    // A new key replaces the active velocity and extends its deadline.
    double hold_s = first_press_hold_s_;
    if (publish_twist || publish_joint)
    {
      const auto key_time = std::chrono::steady_clock::now();
      // The first auto-repeat comes key_repeat_delay_s after the press, later ones every ~30 ms.
      // Any same-key event while the press still holds is a repeat (a quick double tap is too: it
      // then extends by REPEAT_HOLD_S only, never more than a single press).
      const double repeat_gap_s = std::max(REPEAT_GAP_S, first_press_hold_s_);
      if (c == last_motion_key &&
          std::chrono::duration<double>(key_time - last_motion_key_time).count() < repeat_gap_s)
        hold_s = REPEAT_HOLD_S;
      last_motion_key = c;
      last_motion_key_time = key_time;
    }
    if (publish_twist)
    {
      twist_msg->header.frame_id = command_frame_id_;
      queueTwist(*twist_msg, hold_s);
      publish_twist = false;
    }
    else if (publish_joint)
    {
      joint_msg->header.frame_id = PLANNING_FRAME_ID;
      queueJoint(*joint_msg, hold_s);
      publish_joint = false;
    }
  }

  return 0;
}