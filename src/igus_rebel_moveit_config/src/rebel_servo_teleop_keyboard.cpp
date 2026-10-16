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

#include <control_msgs/msg/joint_jog.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <igus_rebel_msgs/srv/set_cartesian_increment.hpp>
#if __has_include(<moveit_msgs/srv/servo_command_type.hpp>)
#include <moveit_msgs/srv/servo_command_type.hpp>
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 1
#else
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 0
#include <std_srvs/srv/trigger.hpp>
#endif
#include <std_msgs/msg/int8.hpp>
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

// Converts key-presses to Twist or Jog commands for Servo, in lieu of a controller
class KeyboardServo
{
public:
  KeyboardServo();
  ~KeyboardServo();
  int keyLoop(KeyboardReader& input);

private:
  void setCartesianIncrement(
    const std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Request> request,
    std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Response> response);
  void stopSpinning();
  void publishActiveCommand();
  void queueTwist(const geometry_msgs::msg::TwistStamped& command, double hold_s);
  void queueJoint(const control_msgs::msg::JointJog& command, double hold_s);
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
  std::atomic<double> cartesian_velocity_mps_;
  std::atomic<double> angular_velocity_rps_;
  double joint_speed_rps_;
  double joint_direction_;
  std::string command_frame_id_;
};

KeyboardServo::KeyboardServo()
  : cartesian_velocity_mps_(DEFAULT_LINEAR_VELOCITY_MPS),
    angular_velocity_rps_(DEFAULT_ANGULAR_VELOCITY_RPS),
    joint_speed_rps_(DEFAULT_JOINT_VELOCITY_RPS), joint_direction_(1.0), command_frame_id_{ PLANNING_FRAME_ID }
{
  nh_ = rclcpp::Node::make_shared("servo_keyboard_input");

  twist_pub_ = nh_->create_publisher<geometry_msgs::msg::TwistStamped>(TWIST_TOPIC, ROS_QUEUE_SIZE);
  joint_pub_ = nh_->create_publisher<control_msgs::msg::JointJog>(JOINT_TOPIC, ROS_QUEUE_SIZE);
  command_timer_ = nh_->create_wall_timer(std::chrono::milliseconds(20), [this]() { publishActiveCommand(); });
  increment_service_ = nh_->create_service<igus_rebel_msgs::srv::SetCartesianIncrement>(
    "~/set_cartesian_increment",
    [this](
      const std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Request> request,
      std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Response> response)
    {
      setCartesianIncrement(request, response);
    });

#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
  // MoveIt 2.10 and newer require explicitly selecting the active command type.
  switch_input_ = nh_->create_client<moveit_msgs::srv::ServoCommandType>("servo_node/switch_command_type");
#else
  // Humble's servo_node stays idle until ~/start_servo is called.
  start_servo_ = nh_->create_client<std_srvs::srv::Trigger>("servo_node/start_servo");
  servo_status_ = nh_->create_subscription<std_msgs::msg::Int8>(
    "/status", 10, [this](std_msgs::msg::Int8::ConstSharedPtr) { servo_active_.store(true); });
#endif

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
    twist_pub_->publish(active_twist_);
  }
  else
  {
    active_joint_.header.stamp = nh_->now();
    joint_pub_->publish(active_joint_);
  }
}

int main(int argc, char** argv)
{
  // The key loop owns SIGINT/SIGTERM handling so the terminal is always restored
  // and the spin thread is joined before rclcpp shuts down.
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, requestQuit);
  std::signal(SIGTERM, requestQuit);

  int rc = 0;
  {
    KeyboardReader input;
    KeyboardServo keyboard_servo;
    rc = keyboard_servo.keyLoop(input);
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
  startServo();
#endif

  puts("Reading from keyboard");
  puts("---------------------------");
  puts("Twist commands use the selected command frame ('w': base_link, 'e': xeg32_tool_tip)");
  puts("Use arrow keys and the '.' and ';' keys to Cartesian jog");
  puts("Use i/k for TCP roll, n/m for TCP pitch, and u/o for TCP yaw (press 'e' first)");
  puts("Use '-' and '=' to decrease/increase the speed (Cartesian and joint)");
  puts("Use 1|2|3|4|5|6 keys to joint jog. 'r' to reverse the direction of jogging.");
  puts("Use 'j' to select joint jog. ");
  puts("Use 't' to select twist ");
  puts("Use 'w' and 'e' to switch between sending command in planning frame or end effector frame");
  printf("Change the Cartesian increment with the set_cartesian_increment service "
         "(default %.3f m, max %.3f m)\n",
         DEFAULT_CARTESIAN_INCREMENT_M, MAX_CARTESIAN_INCREMENT_M);
  puts("'q' or Ctrl-C to quit.");
  printf("Speeds: %.4f m/s, %.3f rad/s, joint %.3f rad/s (%.1f deg/s)\n", cartesian_velocity_mps_.load(),
         angular_velocity_rps_.load(), joint_speed_rps_, joint_speed_rps_ * 180.0 / M_PI);

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

    joint_msg->joint_names = { "joint1", "joint2", "joint3", "joint4",
                               "joint5", "joint6"};

    joint_msg->velocities.resize(6);
    std::fill(joint_msg->velocities.begin(), joint_msg->velocities.end(), 0.0);
    const double cartesian_velocity_mps =
      cartesian_velocity_mps_.load(std::memory_order_relaxed);
    const double angular_velocity_rps = angular_velocity_rps_.load(std::memory_order_relaxed);
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
        joint_speed_rps_ = std::clamp(joint_speed_rps_ * scale, MIN_JOINT_VELOCITY_RPS, MAX_JOINT_VELOCITY_RPS);
        cartesian_velocity_mps_.store(linear_speed, std::memory_order_relaxed);
        angular_velocity_rps_.store(angular_speed, std::memory_order_relaxed);
        RCLCPP_INFO(nh_->get_logger(), "Speeds: %.4f m/s, %.3f rad/s, joint %.3f rad/s (%.1f deg/s)", linear_speed,
                    angular_speed, joint_speed_rps_, joint_speed_rps_ * 180.0 / M_PI);
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
    double hold_s = KEYPRESS_DURATION_S;
    if (publish_twist || publish_joint)
    {
      const auto key_time = std::chrono::steady_clock::now();
      if (c == last_motion_key &&
          std::chrono::duration<double>(key_time - last_motion_key_time).count() < REPEAT_GAP_S)
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