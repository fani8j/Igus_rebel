#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/srv/get_parameters.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "handeye_pose_replay/pose_loader.hpp"

struct CalibrationPose
{
  std::string name;
  std::array<double, 6> q;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
      "handeye_pose_replay",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // Replay settings.
  if (!node->has_parameter("planning_group"))
    node->declare_parameter<std::string>("planning_group", "igus_rebel_arm");

  if (!node->has_parameter("model_source_node"))
    node->declare_parameter<std::string>("model_source_node", "/move_group");

  if (!node->has_parameter("wait_for_enter"))
    node->declare_parameter<bool>("wait_for_enter", true);

  if (!node->has_parameter("settle_seconds"))
    node->declare_parameter<double>("settle_seconds", 1.0);

  if (!node->has_parameter("velocity_scale"))
    node->declare_parameter<double>("velocity_scale", 0.05);

  if (!node->has_parameter("acceleration_scale"))
    node->declare_parameter<double>("acceleration_scale", 0.05);

  // Replay the joint poses of a calibration-recorder session instead of the
  // built-in poses. Empty poses_file keeps the built-in 19 poses.
  if (!node->has_parameter("poses_file"))
    node->declare_parameter<std::string>("poses_file", "");

  if (!node->has_parameter("arm"))
    node->declare_parameter<std::string>("arm", "");

  if (!node->has_parameter("start_index"))
    node->declare_parameter<int>("start_index", 0);

  if (!node->has_parameter("max_poses"))
    node->declare_parameter<int>("max_poses", 0);

  std::string planning_group;
  std::string model_source_node;
  bool wait_for_enter;
  double settle_seconds;
  double velocity_scale;
  double acceleration_scale;

  node->get_parameter("planning_group", planning_group);
  node->get_parameter("model_source_node", model_source_node);
  node->get_parameter("wait_for_enter", wait_for_enter);
  node->get_parameter("settle_seconds", settle_seconds);
  node->get_parameter("velocity_scale", velocity_scale);
  node->get_parameter("acceleration_scale", acceleration_scale);

  std::string poses_file;
  std::string arm;
  int start_index = 0;
  int max_poses = 0;
  node->get_parameter("poses_file", poses_file);
  node->get_parameter("arm", arm);
  node->get_parameter("start_index", start_index);
  node->get_parameter("max_poses", max_poses);

  // Each pose is a real robot motion. Without a terminal, std::getline would
  // hit EOF immediately and every "confirmation" would pass unattended, so
  // refuse to start instead of silently degrading to automatic replay.
  if (wait_for_enter && !isatty(STDIN_FILENO))
  {
    RCLCPP_FATAL(
        node->get_logger(),
        "wait_for_enter:=true but stdin is not a terminal, so ENTER cannot be "
        "read. Use the launch default interactive_terminal:=true (opens a "
        "GNOME Terminal), run the executable directly in a terminal, or pass "
        "wait_for_enter:=false only after the poses were verified.");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Planning group: %s", planning_group.c_str());
  RCLCPP_INFO(node->get_logger(), "Robot model source: %s", model_source_node.c_str());
  RCLCPP_INFO(node->get_logger(), "Velocity scale: %.3f", velocity_scale);
  RCLCPP_INFO(node->get_logger(), "Acceleration scale: %.3f", acceleration_scale);

  // ------------------------------------------------------------------------
  // Parameter-client responses require an executor before the synchronous copy.
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread executor_thread([&executor]() { executor.spin(); });
  // Stop the spinning thread before rclcpp::shutdown() so the executor never
  // spins on a context that is being torn down. Used on every exit path.
  const auto stop_executor = [&executor, &executor_thread]() {
    executor.cancel();
    if (executor_thread.joinable()) executor_thread.join();
  };

  // Copy the already-working robot model from the running MoveIt move_group.
  //
  // This deliberately avoids processing the project's URDF/Xacro again.
  // The normal MoveIt launch has already resolved all Xacro arguments and
  // initial positions correctly.
  // ------------------------------------------------------------------------
  using GetParameters = rcl_interfaces::srv::GetParameters;
  const auto service_name = model_source_node + "/get_parameters";
  const auto parameter_client = node->create_client<GetParameters>(service_name);

  RCLCPP_INFO(
      node->get_logger(),
      "Waiting for parameter service from %s ...",
      model_source_node.c_str());

  if (!parameter_client->wait_for_service(std::chrono::seconds(15)))
  {
    RCLCPP_FATAL(
        node->get_logger(),
        "Could not contact %s parameter service. Start your normal MoveIt "
        "launch first, then run this replay launch.",
        model_source_node.c_str());
    stop_executor();
    rclcpp::shutdown();
    return 1;
  }

  auto request = std::make_shared<GetParameters::Request>();
  request->names = {"robot_description", "robot_description_semantic"};
  auto result = parameter_client->async_send_request(request);
  if (result.wait_for(std::chrono::seconds(15)) != std::future_status::ready)
  {
    RCLCPP_FATAL(
        node->get_logger(),
        "Timed out reading robot model parameters from %s.",
        model_source_node.c_str());
    stop_executor();
    rclcpp::shutdown();
    return 1;
  }
  const auto model_parameters = result.get()->values;

  if (model_parameters.size() != 2 ||
      model_parameters[0].type != rcl_interfaces::msg::ParameterType::PARAMETER_STRING ||
      model_parameters[1].type != rcl_interfaces::msg::ParameterType::PARAMETER_STRING ||
      model_parameters[0].string_value.empty() ||
      model_parameters[1].string_value.empty())
  {
    RCLCPP_FATAL(
        node->get_logger(),
        "%s does not provide non-empty robot_description and "
        "robot_description_semantic parameters.",
        model_source_node.c_str());
    stop_executor();
    rclcpp::shutdown();
    return 1;
  }


  // Declare these locally so MoveGroupInterface / RobotModelLoader can use them.
  if (!node->has_parameter("robot_description"))
    node->declare_parameter<std::string>("robot_description", "");

  if (!node->has_parameter("robot_description_semantic"))
    node->declare_parameter<std::string>("robot_description_semantic", "");

  node->set_parameter(
      rclcpp::Parameter(
          "robot_description",
          model_parameters[0].string_value));

  node->set_parameter(
      rclcpp::Parameter(
          "robot_description_semantic",
          model_parameters[1].string_value));

  RCLCPP_INFO(
      node->get_logger(),
      "Copied robot_description and robot_description_semantic from %s.",
      model_source_node.c_str());



  moveit::planning_interface::MoveGroupInterface move_group(
      node,
      planning_group);

  move_group.setMaxVelocityScalingFactor(velocity_scale);
  move_group.setMaxAccelerationScalingFactor(acceleration_scale);
  move_group.setPlanningTime(10.0);
  move_group.setNumPlanningAttempts(10);
  move_group.allowReplanning(true);

  // Exact joint names used by the recorded controller data.
  const std::array<std::string, 6> joint_names = {
      "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
  };

  // Actual joint positions from /home/farhan/Desktop/pose_h2e/v2.zip, in archive order.
  // p_17.txt is absent from the supplied archive.
  const std::vector<CalibrationPose> builtin_poses = {
      {"p_0", {-0.62552600657792878, 0.72221726311738121, 1.8074629307547503, -1.7259560866445480, 1.0529571696861122, 2.0820032806830024}},
      {"p_1", {-0.62570051020835626, 0.72204269290792655, 1.8074629307547503, -1.7259560866445480, 1.0529571696861122, 1.6987289929240141}},
      {"p_2", {-0.62552600657792878, 0.72204269290792655, 1.8074629307547503, -1.7259560866445480, 1.0529571696861122, 1.0667452201768066}},
      {"p_3", {-0.62552600657792878, 0.72221726311738121, 1.8072884937033500, -1.7259560866445480, 1.0529571696861122, 0.66811201635814066}},
      {"p_4", {-0.62552600657792878, 0.72204269290792655, 1.8072884937033500, -1.7259560866445480, 1.0529571696861122, 0.26354472370904647}},
      {"p_5", {-0.69638639817734838, 0.62325705990820857, 1.8982201317532881, -1.9760618004132686, 0.60318581345769007, 0.39898227765854810}},
      {"p_6", {-0.74281215361723651, 0.62238440859801680, 1.9528488894038110, -2.5982715722228034, 0.61575213879831070, 1.8161895770147218}},
      {"p_7", {-0.74281215361723651, 0.62220990496758943, 1.9528488894038110, -2.5982715722228034, 0.61592670900776547, 2.3661429947604451}},
      {"p_8", {-0.74281215361723651, 0.62255891222844428, 1.9528488894038110, -2.5980971351714035, 0.61592670900776547, 1.1901400595455107}},
      {"p_9", {-0.74281215361723651, 0.62238440859801680, 1.9526743191943563, -2.5982715722228034, 0.61575213879831070, 0.74682640290734015}},
      {"p_10", {-0.78295471309730058, 0.69952799615726036, 1.9701276117142996, -3.0162781411283328, 1.0107201671667552, 1.0503391491765097}},
      {"p_11", {-0.78278014288784592, 0.69952799615726036, 1.9701276117142996, -3.0161034377608233, 1.0107201671667552, 1.4688691623942940}},
      {"p_12", {-0.78278014288784592, 0.69935342594780570, 1.9701276117142996, -3.0161034377608233, 1.0105456635363277, 2.0781635350233261}},
      {"p_13", {-0.78278014288784592, 0.69952799615726036, 1.9701276117142996, -3.0161034377608233, 1.0107201671667552, 2.4066344949223368}},
      {"p_14", {-0.78278014288784592, 0.69952799615726036, 1.9701276117142996, -3.0159290007094230, 1.0107201671667552, 2.8174848836627140}},
      {"p_15", {-0.65380032181810932, 0.78819068753747812, 1.8680258771042422, -2.2947588259536338, 1.3131856919162783, 2.5113542500419963}},
      {"p_16", {-0.65380032181810932, 0.78819068753747812, 1.8680258771042422, -2.2947588259536338, 1.3131856919162783, 2.0528563141326299}},
      {"p_18", {-0.62796945687807648, 0.87790060043734230, 1.7729054861337730, -2.1308724198440108, 1.5957545404247377, 2.0247565025228771}},
      {"p_19", {-0.60737456637822118, 0.79918628046717000, 1.6582373596040678, -1.3357004552459717, 1.3245303586858523, 1.7263052270634576}},
  };

  // One replay target: a label for the logs and the joint values in the order
  // of target_joint_names.
  struct Target
  {
    std::string label;
    std::string detail;  // e.g. "(capture #5)"
    std::vector<double> q;
  };

  std::vector<std::string> target_joint_names(joint_names.begin(), joint_names.end());
  std::vector<Target> targets;
  // Session replay: stop on the first planning failure instead of skipping.
  const bool session_replay = !poses_file.empty();
  // 0-based position of each target in the filtered session list (for resuming).
  std::vector<std::size_t> session_index;

  if (!session_replay)
  {
    for (const auto& p : builtin_poses)
      targets.push_back({p.name, "", std::vector<double>(p.q.begin(), p.q.end())});
  }
  else
  {
    // Validate the WHOLE list against the loaded model before anything moves.
    const auto* group = move_group.getRobotModel()->getJointModelGroup(planning_group);
    if (group == nullptr)
    {
      RCLCPP_FATAL(node->get_logger(), "Planning group %s not found in the robot model.", planning_group.c_str());
      stop_executor();
      rclcpp::shutdown();
      return 1;
    }
    target_joint_names.clear();
    std::vector<handeye_pose_replay::JointBounds> bounds;
    for (const auto* jm : group->getActiveJointModels())
    {
      if (jm->getVariableCount() != 1)
        continue;
      const auto& vb = jm->getVariableBounds(jm->getName());
      target_joint_names.push_back(jm->getName());
      bounds.push_back(
          {vb.position_bounded_ ? vb.min_position_ : -std::numeric_limits<double>::infinity(),
           vb.position_bounded_ ? vb.max_position_ : std::numeric_limits<double>::infinity()});
    }

    if (start_index < 0 || max_poses < 0)
    {
      RCLCPP_FATAL(node->get_logger(), "start_index and max_poses must not be negative.");
      stop_executor();
      rclcpp::shutdown();
      return 1;
    }
    handeye_pose_replay::LoadOptions options;
    options.arm = arm;
    options.start_index = static_cast<std::size_t>(start_index);
    options.max_poses = static_cast<std::size_t>(max_poses);

    const auto loaded = handeye_pose_replay::loadPoses(poses_file, target_joint_names, bounds, options);
    if (!loaded.ok())
    {
      RCLCPP_FATAL(
          node->get_logger(), "Refusing to replay %s: %zu problem(s). Nothing was moved.",
          loaded.source_path.c_str(), loaded.errors.size());
      for (const auto& e : loaded.errors) RCLCPP_ERROR(node->get_logger(), "  %s", e.c_str());
      stop_executor();
      rclcpp::shutdown();
      return 1;
    }
    RCLCPP_INFO(
        node->get_logger(),
        "Session %s: arm filter '%s', %zu poses after filtering (%zu near-duplicates skipped), "
        "replaying %zu starting at index %d.",
        loaded.source_path.c_str(), arm.c_str(), loaded.filtered_total, loaded.duplicates_skipped,
        loaded.poses.size(), start_index);
    for (const auto& p : loaded.poses)
    {
      std::string detail = p.capture >= 0 ? "(capture #" + std::to_string(p.capture) + ")" : "(line " + std::to_string(p.line) + ")";
      targets.push_back({"pose " + std::to_string(p.index + 1) + "/" + std::to_string(loaded.filtered_total), detail, p.q});
      session_index.push_back(p.index);
    }
  }

  RCLCPP_INFO(node->get_logger(), "Loaded %zu poses.", targets.size());

  std::size_t reached = 0;
  for (std::size_t i = 0; i < targets.size() && rclcpp::ok(); ++i)
  {
    const auto& pose = targets[i];
    const std::string name = session_replay ? pose.label + " " + pose.detail : pose.label;

    RCLCPP_INFO(node->get_logger(),
                "============================================");
    if (session_replay)
      RCLCPP_INFO(node->get_logger(), "%s", name.c_str());
    else
      RCLCPP_INFO(node->get_logger(),
                  "Target %zu / %zu : %s",
                  i + 1, targets.size(), pose.label.c_str());

    if (session_replay)
    {
      std::string text = "  target [deg]:";
      char buf[32];
      for (const double v : pose.q)
      {
        std::snprintf(buf, sizeof(buf), " %8.2f", v * 180.0 / M_PI);
        text += buf;
      }
      RCLCPP_INFO(node->get_logger(), "%s", text.c_str());

      const auto current = move_group.getCurrentState(5.0);
      if (current)
      {
        std::vector<double> now;
        current->copyJointGroupPositions(planning_group, now);
        if (now.size() == pose.q.size())
        {
          double largest = 0.0;
          for (std::size_t j = 0; j < now.size(); ++j)
            largest = std::max(largest, std::fabs(now[j] - pose.q[j]));
          RCLCPP_INFO(
              node->get_logger(), "  largest joint change from current state: %.1f deg",
              largest * 180.0 / M_PI);
        }
      }
      else
      {
        RCLCPP_WARN(node->get_logger(), "  current state unavailable; cannot show the joint change.");
      }
    }

    if (wait_for_enter)
    {
      RCLCPP_INFO(
          node->get_logger(),
          "Press ENTER to plan/move to %s (Ctrl+C to stop)...", name.c_str());

      std::string input;
      if (!std::getline(std::cin, input))
      {
        RCLCPP_ERROR(
            node->get_logger(),
            "stdin closed before %s was confirmed. Stopping replay.",
            name.c_str());
        break;
      }

      if (!rclcpp::ok())
        break;
    }

    move_group.setStartStateToCurrentState();

    std::map<std::string, double> target;
    for (std::size_t j = 0; j < target_joint_names.size(); ++j)
      target[target_joint_names[j]] = pose.q[j];

    const bool target_ok = move_group.setJointValueTarget(target);

    if (!target_ok)
    {
      if (session_replay)
      {
        RCLCPP_ERROR(
            node->get_logger(),
            "Target %s is invalid or outside a joint limit. Stopping replay.",
            name.c_str());
        break;
      }
      RCLCPP_ERROR(
          node->get_logger(),
          "Target %s is invalid or outside a joint limit. Skipping.",
          name.c_str());
      continue;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto plan_result = move_group.plan(plan);

    if (plan_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      if (session_replay)
      {
        RCLCPP_ERROR(
            node->get_logger(),
            "Planning FAILED for %s. Stopping replay (not skipping ahead); resume with start_index:=%zu.",
            name.c_str(), session_index[i]);
        break;
      }
      RCLCPP_ERROR(
          node->get_logger(),
          "Planning FAILED for %s. Robot will not move to this pose.",
          name.c_str());
      continue;
    }

    RCLCPP_INFO(
        node->get_logger(),
        "Plan successful for %s. Executing...",
        name.c_str());

    const auto exec_result = move_group.execute(plan);

    if (exec_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(
          node->get_logger(),
          "Execution FAILED at %s. Stopping replay.",
          name.c_str());
      break;
    }
    ++reached;

    RCLCPP_INFO(
        node->get_logger(),
        "Reached %s. Settling for %.2f s...",
        name.c_str(),
        settle_seconds);

    std::this_thread::sleep_for(
        std::chrono::duration<double>(settle_seconds));

    RCLCPP_INFO(
        node->get_logger(),
        "POSE_READY: %s",
        name.c_str());
  }

  RCLCPP_INFO(node->get_logger(), "Replay finished.");
  RCLCPP_INFO(node->get_logger(), "Summary: reached %zu / %zu poses.", reached, targets.size());

  stop_executor();
  rclcpp::shutdown();
  return 0;
}
