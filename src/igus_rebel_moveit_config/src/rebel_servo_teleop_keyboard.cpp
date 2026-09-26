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
#include <cmath>
#include <chrono>
#include <control_msgs/msg/joint_jog.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <igus_rebel_msgs/srv/set_cartesian_increment.hpp>
#if __has_include(<moveit_msgs/srv/servo_command_type.hpp>)
#include <moveit_msgs/srv/servo_command_type.hpp>
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 1
#else
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 0
#endif
#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include <stdio.h>
#ifndef WIN32
#include <termios.h>
#include <unistd.h>
#else
#include <conio.h>
#endif

// Servo applies each command for its incoming-command timeout.
constexpr double SERVO_COMMAND_TIMEOUT_S = 0.3;
constexpr double DEFAULT_CARTESIAN_INCREMENT_M = 0.005;
constexpr double MAX_CARTESIAN_INCREMENT_M = 0.03;
constexpr double DEFAULT_LINEAR_VELOCITY_MPS =
    DEFAULT_CARTESIAN_INCREMENT_M / SERVO_COMMAND_TIMEOUT_S;
constexpr double DEFAULT_ANGULAR_VELOCITY_RPS = 0.15;
constexpr double MIN_LINEAR_VELOCITY_MPS = 0.001;
constexpr double MAX_LINEAR_VELOCITY_MPS = 0.10;
constexpr double MIN_ANGULAR_VELOCITY_RPS = 0.01;
constexpr double MAX_ANGULAR_VELOCITY_RPS = 0.5;
constexpr double SPEED_SCALE_FACTOR = 1.25;

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
constexpr int8_t KEYCODE_7 = 0x37;
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
}  // namespace

// A class for reading the key inputs from the terminal
class KeyboardReader
{
public:
  KeyboardReader() : file_descriptor_(0)
  {
#ifndef WIN32
    // get the console in raw mode
    tcgetattr(file_descriptor_, &cooked_);
    struct termios raw;
    memcpy(&raw, &cooked_, sizeof(struct termios));
    raw.c_lflag &= ~(ICANON | ECHO);
    // Setting a new line, then end of file
    raw.c_cc[VEOL] = 1;
    raw.c_cc[VEOF] = 2;
    tcsetattr(file_descriptor_, TCSANOW, &raw);
#endif
  }
  void readOne(char* c)
  {
#ifndef WIN32
    int rc = read(file_descriptor_, c, 1);
    if (rc < 0)
    {
      throw std::runtime_error("read failed");
    }
#else
    *c = static_cast<char>(_getch());
#endif
  }
  void shutdown()
  {
#ifndef WIN32
    tcsetattr(file_descriptor_, TCSANOW, &cooked_);
#endif
  }

private:
  int file_descriptor_;
#ifndef WIN32
  struct termios cooked_;
#endif
};

// Converts key-presses to Twist or Jog commands for Servo, in lieu of a controller
class KeyboardServo
{
public:
  KeyboardServo();
  int keyLoop();

private:
  void setCartesianIncrement(
    const std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Request> request,
    std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Response> response);
  void spin();

  rclcpp::Node::SharedPtr nh_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Publisher<control_msgs::msg::JointJog>::SharedPtr joint_pub_;
  rclcpp::Service<igus_rebel_msgs::srv::SetCartesianIncrement>::SharedPtr increment_service_;
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
  rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_input_;
  std::shared_ptr<moveit_msgs::srv::ServoCommandType::Request> request_;
#endif
  std::atomic<double> cartesian_velocity_mps_;
  std::atomic<double> angular_velocity_rps_;
  double joint_vel_cmd_;
  std::string command_frame_id_;
};

KeyboardServo::KeyboardServo()
  : cartesian_velocity_mps_(DEFAULT_LINEAR_VELOCITY_MPS),
    angular_velocity_rps_(DEFAULT_ANGULAR_VELOCITY_RPS),
    joint_vel_cmd_(1.0), command_frame_id_{ "base_link" }
{
  nh_ = rclcpp::Node::make_shared("servo_keyboard_input");

  twist_pub_ = nh_->create_publisher<geometry_msgs::msg::TwistStamped>(TWIST_TOPIC, ROS_QUEUE_SIZE);
  joint_pub_ = nh_->create_publisher<control_msgs::msg::JointJog>(JOINT_TOPIC, ROS_QUEUE_SIZE);
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
#endif
}

void KeyboardServo::setCartesianIncrement(
  const std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Request> request,
  std::shared_ptr<igus_rebel_msgs::srv::SetCartesianIncrement::Response> response)
{
  const double increment = request->increment_meters;
  if (!std::isfinite(increment) || increment <= 0.0 || increment > MAX_CARTESIAN_INCREMENT_M)
  {
    response->success = false;
    response->message = "increment_meters must be finite and in the range (0, 0.01]";
    return;
  }

  cartesian_velocity_mps_.store(increment / SERVO_COMMAND_TIMEOUT_S, std::memory_order_relaxed);
  response->success = true;
  response->message = "Cartesian keyboard increment set to " + std::to_string(increment) + " m";
  RCLCPP_INFO(nh_->get_logger(), "%s", response->message.c_str());
}

KeyboardReader input;

void quit(int sig)
{
  (void)sig;
  input.shutdown();
  rclcpp::shutdown();
  exit(0);
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  KeyboardServo keyboard_servo;

  signal(SIGINT, quit);

  int rc = keyboard_servo.keyLoop();
  input.shutdown();
  rclcpp::shutdown();

  return rc;
}

void KeyboardServo::spin()
{
  while (rclcpp::ok())
  {
    rclcpp::spin_some(nh_);
  }
}

int KeyboardServo::keyLoop()
{
  char c;
  bool publish_twist = false;
  bool publish_joint = false;

  std::thread{ [this]() { return spin(); } }.detach();

  puts("Reading from keyboard");
  puts("---------------------------");
  puts("Twist commands use the selected command frame ('w': base_link, 'e': xeg32_tool_tip)");
  puts("Use arrow keys and the '.' and ';' keys to Cartesian jog");
  puts("Use i/k for TCP roll, n/m for TCP pitch, and u/o for TCP yaw (press 'e' first)");
  puts("Use '-' and '=' to decrease/increase Cartesian linear and angular speed");
  puts("Use 1|2|3|4|5|6|7 keys to joint jog. 'r' to reverse the direction of jogging.");
  puts("Use 'j' to select joint jog. ");
  puts("Use 't' to select twist ");
  puts("Use 'w' and 'e' to switch between sending command in planning frame or end effector frame");
  puts("Change the Cartesian increment with the set_cartesian_increment service (default 0.001 m)");
  puts("'Q' to quit.");

  for (;;)
  {
    // get the next event from the keyboard
    try
    {
      input.readOne(&c);
    }
    catch (const std::runtime_error&)
    {
      perror("read():");
      return -1;
    }

    RCLCPP_DEBUG(nh_->get_logger(), "value: 0x%02X\n", c);

    // // Create the messages we might publish
    auto twist_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
    auto joint_msg = std::make_unique<control_msgs::msg::JointJog>();

    joint_msg->joint_names.resize(6);
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
        cartesian_velocity_mps_.store(linear_speed, std::memory_order_relaxed);
        angular_velocity_rps_.store(angular_speed, std::memory_order_relaxed);
        RCLCPP_INFO(
          nh_->get_logger(), "Cartesian speeds: %.4f m/s, %.3f rad/s", linear_speed, angular_speed);
        break;
      }
      case KEYCODE_1:
        RCLCPP_DEBUG(nh_->get_logger(), "1");
        joint_msg->velocities[0] = joint_vel_cmd_;
        publish_joint = true;
        break;
      case KEYCODE_2:
        RCLCPP_DEBUG(nh_->get_logger(), "2");
        joint_msg->velocities[1] = joint_vel_cmd_;
        publish_joint = true;
        break;
      case KEYCODE_3:
        RCLCPP_DEBUG(nh_->get_logger(), "3");
        joint_msg->velocities[2] = joint_vel_cmd_;
        publish_joint = true;
        break;
      case KEYCODE_4:
        RCLCPP_DEBUG(nh_->get_logger(), "4");
        joint_msg->velocities[3] = joint_vel_cmd_;
        publish_joint = true;
        break;
      case KEYCODE_5:
        RCLCPP_DEBUG(nh_->get_logger(), "5");
        joint_msg->velocities[4] = joint_vel_cmd_;
        publish_joint = true;
        break;
      case KEYCODE_6:
        RCLCPP_DEBUG(nh_->get_logger(), "6");
        joint_msg->velocities[5] = joint_vel_cmd_;
        publish_joint = true;
        break;
      // case KEYCODE_7:
      //   RCLCPP_DEBUG(nh_->get_logger(), "7");
      //   joint_msg->velocities[6] = joint_vel_cmd_;
      //   publish_joint = true;
      //   break;
      case KEYCODE_R:
        RCLCPP_DEBUG(nh_->get_logger(), "r");
        joint_vel_cmd_ *= -1;
        break;
      case KEYCODE_J:
        RCLCPP_DEBUG(nh_->get_logger(), "j");
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
        request_ = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
        request_->command_type = moveit_msgs::srv::ServoCommandType::Request::JOINT_JOG;
        if (switch_input_->wait_for_service(std::chrono::seconds(1)))
        {
          auto result = switch_input_->async_send_request(request_);
          if (result.get()->success)
          {
            RCLCPP_INFO_STREAM(nh_->get_logger(), "Switched to input type: JointJog");
          }
          else
          {
            RCLCPP_WARN_STREAM(nh_->get_logger(), "Could not switch input to: JointJog");
          }
        }
#endif
        break;
      case KEYCODE_T:
        RCLCPP_DEBUG(nh_->get_logger(), "t");
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
        request_ = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
        request_->command_type = moveit_msgs::srv::ServoCommandType::Request::TWIST;
        if (switch_input_->wait_for_service(std::chrono::seconds(1)))
        {
          auto result = switch_input_->async_send_request(request_);
          if (result.get()->success)
          {
            RCLCPP_INFO_STREAM(nh_->get_logger(), "Switched to input type: Twist");
          }
          else
          {
            RCLCPP_WARN_STREAM(nh_->get_logger(), "Could not switch input to: Twist");
          }
        }
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

    // If a key requiring a publish was pressed, publish the message now
    if (publish_twist)
    {
      twist_msg->header.stamp = nh_->now();
      twist_msg->header.frame_id = command_frame_id_;
      twist_pub_->publish(std::move(twist_msg));
      publish_twist = false;
    }
    else if (publish_joint)
    {
      joint_msg->header.stamp = nh_->now();
      joint_msg->header.frame_id = PLANNING_FRAME_ID;
      joint_pub_->publish(std::move(joint_msg));
      publish_joint = false;
    }
  }

  return 0;
}