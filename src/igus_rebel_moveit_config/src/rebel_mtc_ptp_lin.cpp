#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/joint_interpolation.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/trajectory_execution_info.h>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/rclcpp.hpp>

#include <std_srvs/srv/trigger.hpp>
#include <array>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mtc = moveit::task_constructor;

namespace
{
constexpr std::array<const char*, 6> JOINT_NAMES = {
  "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

template <typename T>
T getOrDeclareParameter(
  const rclcpp::Node::SharedPtr& node, const std::string& name, const T& default_value)
{
  if (node->has_parameter(name)) {
    return node->get_parameter(name).get_value<T>();
  }
  return node->declare_parameter<T>(name, default_value);
}

mtc::Task createTask(const rclcpp::Node::SharedPtr& node)
{
  const auto group = getOrDeclareParameter<std::string>(node, "planning_group", "igus_rebel_arm");
  const auto tool_link = getOrDeclareParameter<std::string>(node, "tool_link", "link6");
  const auto controller = getOrDeclareParameter<std::string>(
    node, "controller", "rebel_arm_trajectory_controller");
  const auto left_1_goal = getOrDeclareParameter<std::vector<double>>(
    node, "left_1_joint_goal", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  const auto left_2_goal = getOrDeclareParameter<std::vector<double>>(
    node, "left_2_joint_goal", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  const auto left_3_goal = getOrDeclareParameter<std::vector<double>>(
    node, "left_3_joint_goal", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  const auto home_goal = getOrDeclareParameter<std::vector<double>>(
    node, "home_joint_goal", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  const double velocity_scaling =
    getOrDeclareParameter<double>(node, "velocity_scaling", 0.05);
  const double acceleration_scaling =
    getOrDeclareParameter<double>(node, "acceleration_scaling", 0.05);
  const double cartesian_step = getOrDeclareParameter<double>(node, "cartesian_step", 0.005);

  const auto validate_goal = [](const std::vector<double>& goal, const char* name) {
    if (goal.size() != JOINT_NAMES.size()) {
      throw std::invalid_argument(std::string(name) + " must contain exactly six joint values in radians");
    }
  };
  validate_goal(left_1_goal, "left_1_joint_goal");
  validate_goal(left_2_goal, "left_2_joint_goal");
  validate_goal(left_3_goal, "left_3_joint_goal");
  validate_goal(home_goal, "home_joint_goal");
  if (velocity_scaling <= 0.0 || velocity_scaling > 1.0 || acceleration_scaling <= 0.0 ||
      acceleration_scaling > 1.0) {
    throw std::invalid_argument("velocity_scaling and acceleration_scaling must be in (0, 1]");
  }
  if (cartesian_step <= 0.0) {
    throw std::invalid_argument("cartesian_step must be positive");
  }

  mtc::Task task;
  task.stages()->setName("ReBeL Left 1 to Left 2 to Left 3 to Home");
  task.loadRobotModel(node);
  task.add(std::make_unique<mtc::stages::CurrentState>("current robot state"));

  auto ptp_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
  ptp_planner->setMaxVelocityScalingFactor(velocity_scaling);
  ptp_planner->setMaxAccelerationScalingFactor(acceleration_scaling);

  auto linear_planner = std::make_shared<mtc::solvers::CartesianPath>();
  linear_planner->setIKFrame(tool_link);
  linear_planner->setStepSize(cartesian_step);
  linear_planner->setJumpThreshold(0.0);
  linear_planner->setMinFraction(1.0);
  linear_planner->setMaxVelocityScalingFactor(velocity_scaling);
  linear_planner->setMaxAccelerationScalingFactor(acceleration_scaling);

  mtc::TrajectoryExecutionInfo execution_info;
  execution_info.controller_names = {controller};

  const auto add_move = [&](const char* name, const auto& planner,
                            const std::vector<double>& goal, bool cartesian) {
    auto stage = std::make_unique<mtc::stages::MoveTo>(name, planner);
    stage->setGroup(group);
    if (cartesian) {
      stage->setIKFrame(tool_link);
    }
    std::map<std::string, double> joint_goal;
    for (std::size_t index = 0; index < JOINT_NAMES.size(); ++index) {
      joint_goal.emplace(JOINT_NAMES[index], goal[index]);
    }
    stage->setGoal(joint_goal);
    stage->setProperty("trajectory_execution_info", execution_info);
    task.add(std::move(stage));
  };

  add_move("PTP to Left 1", ptp_planner, left_1_goal, false);
  add_move("LIN to Left 2", linear_planner, left_2_goal, true);
  add_move("LIN to Left 3", linear_planner, left_3_goal, true);
  add_move("PTP to Home", ptp_planner, home_goal, false);

  return task;
}

bool executeSolution(const rclcpp::Node::SharedPtr& node, const mtc::SolutionBase& solution)
{
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  using namespace std::chrono_literals;

  const auto action_name = getOrDeclareParameter<std::string>(
    node, "controller_action", "/rebel_arm_trajectory_controller/follow_joint_trajectory");
  const double timeout_seconds =
    getOrDeclareParameter<double>(node, "execution_timeout", 120.0);
  const double controller_wait_seconds =
    getOrDeclareParameter<double>(node, "controller_wait_timeout", 30.0);
  if (timeout_seconds <= 0.0 || controller_wait_seconds <= 0.0) {
    throw std::invalid_argument("execution_timeout and controller_wait_timeout must be positive");
  }

  auto client = rclcpp_action::create_client<FollowJointTrajectory>(node, action_name);
  if (!client->wait_for_action_server(std::chrono::duration<double>(controller_wait_seconds))) {
    RCLCPP_ERROR(node->get_logger(), "Controller action %s is unavailable", action_name.c_str());
    return false;
  }

  moveit_task_constructor_msgs::msg::Solution solution_message;
  solution.toMsg(solution_message);
  std::size_t motion_index = 0;
  for (const auto& sub_trajectory : solution_message.sub_trajectory) {
    if (sub_trajectory.trajectory.joint_trajectory.points.empty()) {
      continue;
    }
    ++motion_index;

    FollowJointTrajectory::Goal goal;
    goal.trajectory = sub_trajectory.trajectory.joint_trajectory;
    auto goal_future = client->async_send_goal(goal);
    if (goal_future.wait_for(5s) != std::future_status::ready) {
      RCLCPP_ERROR(node->get_logger(), "Timed out sending motion %zu", motion_index);
      return false;
    }

    const auto goal_handle = goal_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(node->get_logger(), "Controller rejected motion %zu", motion_index);
      return false;
    }

    RCLCPP_INFO(node->get_logger(), "Executing motion %zu", motion_index);
    auto result_future = client->async_get_result(goal_handle);
    if (result_future.wait_for(std::chrono::duration<double>(timeout_seconds)) != std::future_status::ready) {
      RCLCPP_ERROR(node->get_logger(), "Motion %zu exceeded its execution timeout", motion_index);
      client->async_cancel_goal(goal_handle);
      return false;
    }

    const auto result = result_future.get();
    if (result.code != rclcpp_action::ResultCode::SUCCEEDED ||
        result.result->error_code != FollowJointTrajectory::Result::SUCCESSFUL) {
      RCLCPP_ERROR(
        node->get_logger(), "Motion %zu failed: %s", motion_index,
        result.result ? result.result->error_string.c_str() : "no controller result");
      return false;
    }
    RCLCPP_INFO(node->get_logger(), "Motion %zu completed", motion_index);
  }

  if (motion_index == 0) {
    RCLCPP_ERROR(node->get_logger(), "The MTC solution contains no executable trajectories");
    return false;
  }
  return true;
}


bool runTask(const rclcpp::Node::SharedPtr& node, bool execute)
{
  try {
    const int max_solutions = getOrDeclareParameter<int>(node, "max_solutions", 1);
    if (max_solutions < 1) {
      throw std::invalid_argument("max_solutions must be at least one");
    }

    auto task = createTask(node);
    task.init();
    const auto result = task.plan(static_cast<std::size_t>(max_solutions));
    if (result != moveit::core::MoveItErrorCode::SUCCESS || task.solutions().empty()) {
      RCLCPP_ERROR(node->get_logger(), "PTP/LIN task planning failed");
      task.explainFailure();
      return false;
    }

    const auto& solution = *task.solutions().front();
    task.introspection().publishSolution(solution);
    RCLCPP_INFO(node->get_logger(), "PTP/LIN task planned successfully");
    if (!execute) {
      RCLCPP_INFO(node->get_logger(), "Planning only: call ~/replay_motion after inspecting the solution");
      return true;
    }

    RCLCPP_WARN(node->get_logger(), "Executing the planned PTP/LIN task");
    if (!executeSolution(node, solution)) {
      RCLCPP_ERROR(node->get_logger(), "PTP/LIN task execution failed");
      return false;
    }
    RCLCPP_INFO(node->get_logger(), "PTP/LIN task execution completed");
    return true;
  } catch (const mtc::InitStageException& exception) {
    RCLCPP_ERROR(node->get_logger(), "Task initialization failed: %s", exception.what());
  } catch (const std::exception& exception) {
    RCLCPP_ERROR(node->get_logger(), "PTP/LIN task failed: %s", exception.what());
  }
  return false;
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("rebel_mtc_ptp_lin");

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  std::mutex task_mutex;
  auto replay_group =
    node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto replay_service = node->create_service<std_srvs::srv::Trigger>(
    "~/replay_motion",
    [node, &task_mutex](
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
      std::unique_lock<std::mutex> lock(task_mutex, std::try_to_lock);
      if (!lock.owns_lock()) {
        response->success = false;
        response->message = "The motion task is already running";
        return;
      }
      response->success = runTask(node, true);
      response->message = response->success ? "Replay completed" : "Replay failed";
    },
    rmw_qos_profile_services_default, replay_group);

  const bool execute = getOrDeclareParameter<bool>(node, "execute", false);
  bool initial_run_succeeded = false;
  {
    std::lock_guard<std::mutex> lock(task_mutex);
    initial_run_succeeded = runTask(node, execute);
  }

  if (!initial_run_succeeded) {
    rclcpp::shutdown();
  }
  spinner.join();
  (void)replay_service;
  return initial_run_succeeded ? 0 : 1;
}
