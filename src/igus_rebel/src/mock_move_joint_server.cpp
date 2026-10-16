// Mock of the robot-side joint move for hardware_protocol:=mock_hardware.
//
// mock_components/GenericSystem has no CRI, so there is no "CMD Move Joint". This node serves the
// same igus_rebel_msgs/action/MoveJoint action as the Igus::Rebel plugin (/<node name>/move_joint)
// and executes the move by streaming positions to an otherwise inactive
// forward_command_controller (mock_move_joint_controller) at 100 Hz: a trapezoid along the
// start->goal joint-space line, speeds from the measured speed model of the real robot.
//
// Like the real driver it refuses a goal while the joint trajectory controller is active (they
// would fight over the position interfaces), activates its own controller for the duration of the
// move and deactivates it at the end. Cancel brakes to a stop and ends the move with reason USER.
// Fault injection (ROS parameters, read at every goal): reject, abort_after_s, stall, deviate_rad.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "controller_manager_msgs/srv/list_controllers.hpp"
#include "controller_manager_msgs/srv/switch_controller.hpp"
#include "igus_rebel/MoveSupervisor.hpp"
#include "igus_rebel_msgs/action/move_joint.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace
{
using MoveJoint = igus_rebel_msgs::action::MoveJoint;
using GoalHandle = rclcpp_action::ServerGoalHandle<MoveJoint>;
using Result = MoveJoint::Result;
using Joints = Igus::Move::Joints;
using Clock = std::chrono::steady_clock;

constexpr double kDegToRad = 0.0174532925199432957692369076848861271344287188854172545609719144;
// Measured on the real robot: 0.70 deg/s per vel% on J1; ~1.5 rad/s^2 at vel 25 / acc 40.
constexpr double kDegPerSecPerVelPercent = 0.70;
constexpr double kAccelRadS2AtFullAcc = 1.5;
constexpr double kStartLatencyS = 0.12;
constexpr double kStreamPeriodS = 0.01;

class MockMoveJointServer : public rclcpp::Node
{
public:
  MockMoveJointServer()
  : Node("igus_rebel")
  {
    declare_parameter<std::vector<std::string>>(
      "joint_names", {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"});
    declare_parameter<std::string>("controller_name", "mock_move_joint_controller");
    declare_parameter<std::string>("trajectory_controller_name", "rebel_arm_trajectory_controller");
    declare_parameter<std::string>("controller_manager", "/controller_manager");
    declare_parameter<double>("max_velocity_percent", Igus::Move::kDefaultMaxVelocityPercent);
    declare_parameter<double>(
      "max_acceleration_percent",
      Igus::Move::kDefaultMaxAccelerationPercent);
    // Fault injection.
    declare_parameter<bool>("reject", false);
    declare_parameter<double>("abort_after_s", 0.0);
    declare_parameter<bool>("stall", false);
    declare_parameter<double>("deviate_rad", 0.0);

    const auto names = get_parameter("joint_names").as_string_array();
    if (names.size() != 6) {
      throw std::runtime_error("joint_names must have 6 entries");
    }
    for (std::size_t i = 0; i < 6; ++i) {
      jointNames_[i] = names[i];
    }
    controllerName_ = get_parameter("controller_name").as_string();
    trajectoryController_ = get_parameter("trajectory_controller_name").as_string();
    const std::string manager = get_parameter("controller_manager").as_string();

    clientGroup_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    listClient_ = create_client<controller_manager_msgs::srv::ListControllers>(
      manager + "/list_controllers", rmw_qos_profile_services_default, clientGroup_);
    switchClient_ = create_client<controller_manager_msgs::srv::SwitchController>(
      manager + "/switch_controller", rmw_qos_profile_services_default, clientGroup_);
    commandPublisher_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/" + controllerName_ + "/commands", 10);
    stateSubscription_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::JointState::SharedPtr message) {OnJointState(*message);});

    actionGroup_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    server_ = rclcpp_action::create_server<MoveJoint>(
      this, "~/move_joint",
      [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const MoveJoint::Goal>) {
        if (busy_.exchange(true)) {
          RCLCPP_WARN(get_logger(), "goal rejected: another move is active");
          return rclcpp_action::GoalResponse::REJECT;
        }
        cancel_ = false;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [this](const std::shared_ptr<GoalHandle>) {
        cancel_ = true;
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this](const std::shared_ptr<GoalHandle> handle) {
        std::lock_guard<std::mutex> lock(threadLock_);
        if (worker_.joinable()) {
          worker_.join();
        }
        worker_ = std::thread([this, handle]() {Execute(handle);});
      },
      rcl_action_server_get_default_options(), actionGroup_);
    RCLCPP_INFO(
      get_logger(), "Mock joint move server ready on %s/move_joint (controller '%s')",
      get_fully_qualified_name(), controllerName_.c_str());
  }

  ~MockMoveJointServer() override
  {
    cancel_ = true;
    std::lock_guard<std::mutex> lock(threadLock_);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  void OnJointState(const sensor_msgs::msg::JointState & message)
  {
    std::array<double, 6> positions{};
    std::array<bool, 6> found{};
    for (std::size_t m = 0; m < message.name.size() && m < message.position.size(); ++m) {
      for (std::size_t i = 0; i < 6; ++i) {
        if (message.name[m] == jointNames_[i]) {
          positions[i] = message.position[m];
          found[i] = true;
        }
      }
    }
    if (std::all_of(found.begin(), found.end(), [](bool f) {return f;})) {
      std::lock_guard<std::mutex> lock(stateLock_);
      measured_ = positions;
      measuredStamp_ = Clock::now();
      haveState_ = true;
    }
  }

  bool Measured(Joints & joints, double & ageS)
  {
    std::lock_guard<std::mutex> lock(stateLock_);
    if (!haveState_) {
      return false;
    }
    joints = measured_;
    ageS = std::chrono::duration<double>(Clock::now() - measuredStamp_).count();
    return true;
  }

  // Name of the first joint trajectory controller that is active, "" if none, or "?" if the
  // controller manager did not answer (fail closed).
  std::string ActiveTrajectoryController()
  {
    if (!listClient_->wait_for_service(std::chrono::seconds(1))) {
      return "?";
    }
    auto future = listClient_->async_send_request(
      std::make_shared<controller_manager_msgs::srv::ListControllers::Request>());
    if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
      return "?";
    }
    const auto response = future.get();  // keep it alive for the loop
    for (const auto & controller : response->controller) {
      if (controller.state != "active") {
        continue;
      }
      if (controller.name == trajectoryController_ ||
        controller.type.find("JointTrajectoryController") != std::string::npos)
      {
        return controller.name;
      }
    }
    return "";
  }

  bool Switch(bool activate)
  {
    if (!switchClient_->wait_for_service(std::chrono::seconds(1))) {
      return false;
    }
    auto request = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
    (activate ? request->activate_controllers : request->deactivate_controllers)
    .push_back(controllerName_);
    request->strictness = controller_manager_msgs::srv::SwitchController::Request::STRICT;
    request->activate_asap = true;
    request->timeout = rclcpp::Duration::from_seconds(2.0);
    auto future = switchClient_->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(4)) != std::future_status::ready) {
      return false;
    }
    return future.get()->ok;
  }

  void Publish(const Joints & joints)
  {
    std_msgs::msg::Float64MultiArray message;
    message.data.assign(joints.begin(), joints.end());
    commandPublisher_->publish(message);
  }

  void Execute(const std::shared_ptr<GoalHandle> handle)
  {
    const auto goal = handle->get_goal();
    auto result = std::make_shared<Result>();
    bool stopSent = false;
    bool controllerActive = false;
    std::string endReason;

    const auto finish = [&](std::uint8_t code, const std::string & message) {
        if (controllerActive) {
          Switch(false);  // always hand the interfaces back
          controllerActive = false;
        }
        result->error_code = code;
        result->message = message;
        result->end_reason = endReason;
        result->stop_sent = stopSent;
        result->stop_confirmed = stopSent;
        Joints last{};
        double age = 0.0;
        if (Measured(last, age)) {
          result->final_positions.assign(last.begin(), last.end());
        }
        RCLCPP_INFO(get_logger(), "move ended: code %u %s", code, message.c_str());
        try {
          if (code == Result::SUCCESS) {
            handle->succeed(result);
          } else if (code == Result::CANCELED && handle->is_canceling()) {
            handle->canceled(result);
          } else {
            handle->abort(result);
          }
        } catch (const std::exception & error) {
          RCLCPP_WARN(get_logger(), "could not report the result: %s", error.what());
        }
        busy_ = false;
      };

    // ---- validation ----
    if (get_parameter("reject").as_bool()) {
      finish(Result::REJECTED, "rejected by fault injection (reject:=true)");
      return;
    }
    Igus::Move::GoalRequest request;
    request.jointNames = goal->joint_names;
    request.startPositions = goal->start_positions;
    request.goalPositions = goal->goal_positions;
    request.velocityPercent = goal->velocity_percent;
    request.accelerationPercent = goal->acceleration_percent;
    request.startTolerance = goal->start_tolerance;
    request.pathTolerance = goal->path_tolerance;
    request.goalTolerance = goal->goal_tolerance;
    request.timeoutS = goal->timeout_s;
    Igus::Move::Caps caps;
    caps.maxVelocityPercent = std::min(
      get_parameter("max_velocity_percent").as_double(), Igus::Move::kHardMaxVelocityPercent);
    caps.maxAccelerationPercent = std::min(
      get_parameter("max_acceleration_percent").as_double(),
      Igus::Move::kHardMaxAccelerationPercent);
    Joints start{};
    Joints target{};
    std::string reason;
    if (!Igus::Move::ValidateGoalRequest(request, jointNames_, caps, start, target, reason)) {
      finish(Result::REJECTED, reason);
      return;
    }
    const std::string active = ActiveTrajectoryController();
    if (!active.empty()) {
      finish(
        Result::CONTROLLER_ACTIVE,
        active == "?" ? "controller manager did not answer (list_controllers)" :
        "joint trajectory controller '" + active + "' is active; deactivate it first");
      return;
    }
    Joints measured{};
    double age = 0.0;
    if (!Measured(measured, age) || age > 0.5) {
      finish(Result::STATUS_GAP, "no fresh /joint_states");
      return;
    }
    if (Igus::Move::MaxAbsDifference(start, measured) > goal->start_tolerance) {
      finish(Result::START_DEVIATION, "measured joints differ from start_positions");
      return;
    }
    if (Igus::Move::MaxAbsDifference(measured, target) < Igus::Move::kMinMoveDistance) {
      finish(Result::REJECTED, "already at the goal");
      return;
    }
    if (cancel_) {
      finish(Result::CANCELED, "canceled before the move started");
      return;
    }
    if (!Switch(true)) {
      finish(Result::CONTROLLER_ACTIVE, "could not activate " + controllerName_);
      return;
    }
    controllerActive = true;

    // ---- execution ----
    const bool stall = get_parameter("stall").as_bool();
    const double abortAfter = get_parameter("abort_after_s").as_double();
    const double deviation = get_parameter("deviate_rad").as_double();
    Igus::Move::MoveSupervisor supervisor(
      measured, target, goal->velocity_percent, goal->acceleration_percent, goal->path_tolerance,
      goal->goal_tolerance, goal->timeout_s);

    double distance = 0.0;  // leading joint distance
    Joints delta{};
    for (std::size_t i = 0; i < 6; ++i) {
      delta[i] = target[i] - measured[i];
      distance = std::max(distance, std::fabs(delta[i]));
    }
    const double vmax = kDegPerSecPerVelPercent * kDegToRad * goal->velocity_percent;
    // ~1.5 rad/s^2 at vel 25 / acc 40, growing to ~3 at vel 100 (measured).
    const double amax = kAccelRadS2AtFullAcc * (goal->acceleration_percent / 40.0) *
      (0.667 + 1.333 * goal->velocity_percent / 100.0);
    double position = 0.0;  // along the leading joint
    double speed = 0.0;
    bool braking = false;
    std::uint8_t brakeCode = Result::SUCCESS;
    std::string brakeMessage;
    const auto begin = Clock::now();
    auto next = begin;
    auto lastFeedback = begin;
    double maxDeviation = 0.0;
    Publish(measured);  // hold where we are (the controller starts from its own state)

    while (rclcpp::ok()) {
      next += std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(kStreamPeriodS));
      std::this_thread::sleep_until(next);
      const double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();

      if (!braking) {
        if (cancel_) {
          braking = true;
          stopSent = true;
          brakeCode = Result::CANCELED;
          brakeMessage = "canceled by the client";
        } else if (abortAfter > 0.0 && elapsed > abortAfter) {
          braking = true;
          brakeCode = Result::ROBOT_ENDED;
          brakeMessage = "fault injection: the 'robot' ended the move (abort_after_s)";
        } else if (elapsed > supervisor.TimeoutS()) {
          braking = true;
          stopSent = true;
          brakeCode = Result::MOTION_TIMEOUT;
          brakeMessage = "no completion within the timeout";
        }
      }

      if (elapsed >= kStartLatencyS && !stall) {
        const double remaining = distance - position;
        const double stopDistance = speed * speed / (2.0 * amax);
        if (braking || remaining <= stopDistance) {
          speed = std::max(0.0, speed - amax * kStreamPeriodS);
        } else {
          speed = std::min(vmax, speed + amax * kStreamPeriodS);
        }
        position = std::min(distance, position + speed * kStreamPeriodS);
        if (!braking && position >= distance - 1e-9) {
          position = distance;
          speed = 0.0;
        }
      }
      const double s = distance > 0.0 ? position / distance : 1.0;
      Joints command{};
      for (std::size_t i = 0; i < 6; ++i) {
        command[i] = measured[i] + s * delta[i];
      }
      command[4] += deviation * s;
      Publish(command);

      Joints now{};
      double ageNow = 0.0;
      Measured(now, ageNow);
      const auto sample = supervisor.Evaluate(now);
      maxDeviation = std::max(maxDeviation, sample.deviation);
      if (!braking && supervisor.Check(now, 0.0) != Igus::Move::Verdict::Ok) {
        braking = true;
        stopSent = true;
        brakeCode = Result::PATH_DEVIATION;
        brakeMessage = std::string("supervisor: ") +
          Igus::Move::VerdictName(supervisor.Check(now, 0.0));
      }
      if (Clock::now() - lastFeedback > std::chrono::milliseconds(50)) {
        lastFeedback = Clock::now();
        auto feedback = std::make_shared<MoveJoint::Feedback>();
        feedback->state = braking ? "STOPPING" : "MOVING";
        feedback->progress = sample.progress;
        feedback->max_path_deviation = maxDeviation;
        feedback->actual_positions.assign(now.begin(), now.end());
        handle->publish_feedback(feedback);
      }

      if (braking && speed <= 0.0) {
        endReason = "USER";
        finish(brakeCode, brakeMessage);
        return;
      }
      if (!braking && position >= distance) {
        break;
      }
    }

    // Arrived: hold the goal, let the arm settle, then check it (like the driver).
    endReason = "PLAN";
    Publish(target);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    Joints finalPositions{};
    double finalAge = 0.0;
    Measured(finalPositions, finalAge);
    if (supervisor.AtGoal(finalPositions)) {
      finish(Result::SUCCESS, "goal reached (mock)");
    } else {
      finish(Result::GOAL_DEVIATION, "mock arm did not reach the goal");
    }
  }

  std::array<std::string, 6> jointNames_;
  std::string controllerName_;
  std::string trajectoryController_;
  rclcpp::CallbackGroup::SharedPtr clientGroup_;
  rclcpp::CallbackGroup::SharedPtr actionGroup_;
  rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedPtr listClient_;
  rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr switchClient_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr commandPublisher_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr stateSubscription_;
  rclcpp_action::Server<MoveJoint>::SharedPtr server_;
  std::mutex stateLock_;
  std::array<double, 6> measured_{};
  Clock::time_point measuredStamp_{};
  bool haveState_ = false;
  std::atomic<bool> busy_{false};
  std::atomic<bool> cancel_{false};
  std::mutex threadLock_;
  std::thread worker_;
};
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MockMoveJointServer>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
