#include <Eigen/Geometry>
#include <compal_box_msgs/srv/capture_box_snapshot.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stage.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/task.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace mtc = moveit::task_constructor;
using CaptureSnapshot = compal_box_msgs::srv::CaptureBoxSnapshot;

struct SideCutCandidate {
  geometry_msgs::msg::Point start;
  geometry_msgs::msg::Point end;
  double length_m;
  double roll_deg;
  double pitch_deg;
  double yaw_deg;
};

struct OrientationOffset {
  double roll_deg;
  double pitch_deg;
  double yaw_deg;
};

struct CutSegment {
  std::string name;
  geometry_msgs::msg::Point start;
  geometry_msgs::msg::Point end;
  double roll_deg{0.0};
  double pitch_deg{0.0};
  double yaw_deg{0.0};
  bool stop_after_start{false};
};

class BoxHCutExecutor : public rclcpp::Node
{
public:
  BoxHCutExecutor() : Node("box_h_cut_executor")
  {
    planning_group_ = declare_parameter<std::string>("planning_group", "igus_rebel_arm");
    planning_frame_ = declare_parameter<std::string>("planning_frame", "base_link");
    tool_link_ = declare_parameter<std::string>("tool_link", "link6");
    planning_pipeline_ = declare_parameter<std::string>("planning_pipeline", "ompl");
    chomp_pipeline_ = declare_parameter<std::string>("chomp_pipeline", "chomp");
    pilz_pipeline_ = declare_parameter<std::string>(
      "pilz_pipeline", "pilz_industrial_motion_planner");
    const auto ompl_plugin_parameter = planning_pipeline_ + ".planning_plugin";
    if (!has_parameter(ompl_plugin_parameter)) {
      declare_parameter<std::string>(ompl_plugin_parameter, "ompl_interface/OMPLPlanner");
    }
    const auto ompl_adapters_parameter = planning_pipeline_ + ".request_adapters";
    if (!has_parameter(ompl_adapters_parameter)) {
      declare_parameter<std::string>(
        ompl_adapters_parameter,
        "default_planner_request_adapters/ResolveConstraintFrames "
        "default_planner_request_adapters/FixWorkspaceBounds "
        "default_planner_request_adapters/FixStartStateBounds "
        "default_planner_request_adapters/FixStartStateCollision "
        "default_planner_request_adapters/FixStartStatePathConstraints "
        "default_planner_request_adapters/AddTimeOptimalParameterization");
    }
    const auto kinematics_prefix = "robot_description_kinematics." + planning_group_ + ".";
    if (!has_parameter(kinematics_prefix + "kinematics_solver")) {
      declare_parameter<std::string>(
        kinematics_prefix + "kinematics_solver",
        "kdl_kinematics_plugin/KDLKinematicsPlugin");
      declare_parameter<int>(kinematics_prefix + "kinematics_solver_attempts", 5);
      declare_parameter<double>(
        kinematics_prefix + "kinematics_solver_search_resolution", 0.005);
      declare_parameter<double>(kinematics_prefix + "kinematics_solver_timeout", 0.05);
      declare_parameter<bool>(kinematics_prefix + "position_only_ik", false);
    }
    motion_profile_ = declare_parameter<std::string>("motion_profile", "ompl_pilz");
    snapshot_service_ = declare_parameter<std::string>(
      "snapshot_service", "/box_perception/capture_snapshot");
    carton_collision_id_ = declare_parameter<std::string>("carton_collision_id", "detected_carton");
    carton_height_m_ = declare_parameter<double>("carton_height_m", 0.20);
    hover_clearance_m_ = declare_parameter<double>("hover_clearance_m", 0.10);
    cut_clearance_m_ = declare_parameter<double>("cut_clearance_m", 0.05);
    middle_entry_alignment_m_ = declare_parameter<double>("middle_entry_alignment_m", 0.010);
    side_cut_tilt_deg_ = declare_parameter<double>("side_cut_tilt_deg", 0.0);
    side_cut_orientation_mode_ = declare_parameter<std::string>(
      "side_cut_orientation_mode", "edge_aligned");
    cut_pattern_ = declare_parameter<std::string>("cut_pattern", "left_partial");
    partial_reverse_ = declare_parameter<bool>("partial_reverse", false);
    partial_start_fraction_ = declare_parameter<double>("partial_start_fraction", 0.25);
    partial_end_fraction_ = declare_parameter<double>("partial_end_fraction", 0.75);
    use_approach_pose_as_cut_start_ = declare_parameter<bool>(
      "use_approach_pose_as_cut_start", false);
    approach_cut_start_xyz_ = declare_parameter<std::vector<double>>(
      "approach_cut_start_xyz", std::vector<double>{});
    approach_cut_orientation_xyzw_ = declare_parameter<std::vector<double>>(
      "approach_cut_orientation_xyzw", std::vector<double>{});
    approach_cut_length_m_ = declare_parameter<double>("approach_cut_length_m", 0.04);
    partial_start_joint_degrees_ = declare_parameter<std::vector<double>>(
      "partial_start_joint_degrees", std::vector<double>{});
    approach_joint_degrees_ = declare_parameter<std::vector<double>>(
      "approach_joint_degrees", std::vector<double>{});
    cartesian_step_m_ = declare_parameter<double>("cartesian_step_m", 0.003);
    planning_timeout_s_ = declare_parameter<double>("planning_timeout_s", 15.0);
    velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.05);
    acceleration_scaling_ = declare_parameter<double>("acceleration_scaling", 0.05);
    combined_search_budget_s_ = declare_parameter<double>("combined_search_budget_s", 60.0);
    side_candidate_grid_m_ = declare_parameter<double>("side_candidate_grid_m", 0.010);
    side_candidate_corner_inset_m_ = declare_parameter<double>(
      "side_candidate_corner_inset_m", 0.010);
    side_candidate_min_length_m_ = declare_parameter<double>(
      "side_candidate_min_length_m", 0.040);
    side_orientation_offsets_deg_ = declare_parameter<std::vector<double>>(
      "side_orientation_offsets_deg", std::vector<double>{-5.0, -2.5, 0.0, 2.5, 5.0});
    combined_first_side_link6_frame_ = declare_parameter<std::string>(
      "combined_first_side_link6_frame", "");
    combined_first_side_link6_xyz_ = declare_parameter<std::vector<double>>(
      "combined_first_side_link6_xyz", std::vector<double>{});
    combined_first_side_link6_orientation_xyzw_ = declare_parameter<std::vector<double>>(
      "combined_first_side_link6_orientation_xyzw", std::vector<double>{});
    manual_first_side_max_length_m_ = declare_parameter<double>(
      "manual_first_side_max_length_m", 0.060);
    manual_first_side_contact_offset_xyz_ = declare_parameter<std::vector<double>>(
      "manual_first_side_contact_offset_xyz", std::vector<double>{});
    saved_snapshot_dir_ = declare_parameter<std::string>("saved_snapshot_dir", "~/box_snapshots");
    declare_parameter<std::string>("saved_snapshot_file", "");
    saved_snapshot_max_age_s_ = declare_parameter<double>("saved_snapshot_max_age_s", 60.0);
    declare_parameter<bool>("execution_enabled", false);
    max_solutions_ = declare_parameter<int>("max_solutions", 1);
    const std::array<std::string, 4> profiles{
      "ompl_cartesian", "ompl_pilz", "pilz_only", "chomp_pilz"};
    if (std::find(profiles.begin(), profiles.end(), motion_profile_) == profiles.end()) {
      throw std::invalid_argument("unsupported H-cut motion_profile: " + motion_profile_);
    }
    if (saved_snapshot_max_age_s_ < 0.0) {
      throw std::invalid_argument("saved_snapshot_max_age_s must be non-negative");
    }
    const bool has_manual_first_side_pose = !combined_first_side_link6_frame_.empty();
    if (has_manual_first_side_pose &&
        (combined_first_side_link6_frame_.empty() ||
         combined_first_side_link6_xyz_.size() != 3U ||
         combined_first_side_link6_orientation_xyzw_.size() != 4U ||
         manual_first_side_contact_offset_xyz_.size() != 3U)) {
      throw std::invalid_argument(
        "manual first-side pose must provide frame, XYZ, XYZW, and detected-point offset XYZ");
    }
    if (planning_frame_.empty() || carton_collision_id_.empty() || carton_height_m_ <= 0.0 ||
        hover_clearance_m_ < 0.005 || cut_clearance_m_ < 0.0 ||
        cut_clearance_m_ >= hover_clearance_m_ || middle_entry_alignment_m_ < 0.0 ||
        middle_entry_alignment_m_ > 0.10 || cartesian_step_m_ <= 0.0 ||
        planning_timeout_s_ <= 0.0 || combined_search_budget_s_ <= 0.0 ||
        side_candidate_grid_m_ <= 0.0 || side_candidate_corner_inset_m_ < 0.0 ||
        side_candidate_min_length_m_ <= 0.0 || manual_first_side_max_length_m_ <= 0.0 ||
        side_orientation_offsets_deg_.empty() || velocity_scaling_ <= 0.0 ||
        velocity_scaling_ > 1.0 || acceleration_scaling_ <= 0.0 ||
        acceleration_scaling_ > 1.0 || max_solutions_ < 1 ||
        (cut_pattern_ != "h" && cut_pattern_ != "middle_only" &&
         cut_pattern_ != "middle_entry_debug" && cut_pattern_ != "left_partial" &&
         cut_pattern_ != "right_partial" && cut_pattern_ != "combined") ||
        ((cut_pattern_ == "left_partial" || cut_pattern_ == "right_partial") &&
         partial_start_joint_degrees_.size() != 6U) ||
        (!approach_joint_degrees_.empty() && approach_joint_degrees_.size() != 6U)) {
      throw std::invalid_argument("invalid H-cut planning safety parameters");
    }
    if ((cut_pattern_ == "left_partial" || cut_pattern_ == "right_partial") &&
        use_approach_pose_as_cut_start_ &&
        (approach_joint_degrees_.size() != 6U ||
         approach_cut_start_xyz_.size() != 3U ||
         approach_cut_orientation_xyzw_.size() != 4U || approach_cut_length_m_ <= 0.0)) {
      throw std::invalid_argument("invalid supplied fixed-orientation cut parameters");
    }
    if (side_cut_orientation_mode_ != "edge_aligned" &&
        side_cut_orientation_mode_ != "middle_aligned" &&
        side_cut_orientation_mode_ != "auto") {
      throw std::invalid_argument("unsupported side_cut_orientation_mode");
    }
    cut_pattern_callback_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto& parameter : parameters) {
          if (parameter.get_name() != "cut_pattern") continue;
          const auto pattern = parameter.as_string();
          if (pattern != "h" && pattern != "middle_only" &&
              pattern != "middle_entry_debug" && pattern != "left_partial" &&
              pattern != "right_partial" && pattern != "combined") {
            result.reason = "unsupported cut_pattern";
            return result;
          }
          std::lock_guard<std::mutex> lock(plan_mutex_);
          if (execution_in_progress_) {
            result.successful = false;
            result.reason = "cannot change cut_pattern during trajectory execution";
            return result;
          }
          cut_pattern_ = pattern;
          planned_task_.reset();
          planned_snapshot_.reset();
        }
        return result;
      });

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      [this](sensor_msgs::msg::JointState::SharedPtr message) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_joint_state_ = *message;
      });
    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    snapshot_client_ = create_client<CaptureSnapshot>(
      snapshot_service_, rmw_qos_profile_services_default, callback_group_);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/box_motion/h_cut_markers", rclcpp::QoS(1).reliable().transient_local());
    target_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      "/box_motion/cut_target_poses", rclcpp::QoS(1).reliable().transient_local());
    carton_collision_pub_ = create_publisher<moveit_msgs::msg::CollisionObject>(
      "/collision_object", rclcpp::QoS(1).reliable().transient_local());

    plan_service_ = create_service<std_srvs::srv::Trigger>(
      "~/plan_h_cut",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
          response->success = false;
          response->message = "another H-cut plan operation is active";
          return;
        }
        if (execution_in_progress_) {
          response->success = false;
          response->message = "cannot plan while an H-cut trajectory execution is active";
          return;
        }
        const auto snapshot_file = get_parameter("saved_snapshot_file").as_string();
        response->success = snapshot_file.empty() ?
          captureAndPlan(response->message) : planSavedSnapshot(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
    live_detection_plan_service_ = create_service<std_srvs::srv::Trigger>(
      "~/plan_from_live_detection",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || execution_in_progress_) {
          response->success = false;
          response->message = "cannot plan while another planning or execution operation is active";
          return;
        }
        response->success = captureAndPlan(response->message);
        if (response->success) response->message = "fresh live detection: " + response->message;
      },
      rmw_qos_profile_services_default, callback_group_);
    tcp_debug_start_service_ = create_service<std_srvs::srv::Trigger>(
      "~/start_tcp_debug",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || execution_in_progress_) {
          response->success = false;
          response->message = "cannot start TCP debug while another planning or execution operation is active";
          return;
        }
        response->success = captureAndStartTcpDebug(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
    tcp_debug_record_service_ = create_service<std_srvs::srv::Trigger>(
      "~/record_tcp_debug_point",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || execution_in_progress_) {
          response->success = false;
          response->message = "cannot record TCP debug point while another planning or execution operation is active";
          return;
        }
        response->success = recordTcpDebugPoint(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
    section_diagnostic_service_ = create_service<std_srvs::srv::Trigger>(
      "~/diagnose_saved_snapshot_sections",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || execution_in_progress_) {
          response->success = false;
          response->message = "cannot diagnose while another planning or execution operation is active";
          return;
        }
        response->success = diagnoseSavedSnapshotSections(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
    execute_service_ = create_service<std_srvs::srv::Trigger>(
      "~/execute_planned_h_cut",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
          response->success = false;
          response->message = "another H-cut plan operation is active";
          return;
        }
        if (!get_parameter("execution_enabled").as_bool()) {
          response->success = false;
          response->message = "execution service is disabled by execution_enabled";
          return;
        }
        if (execution_in_progress_) {
          response->success = false;
          response->message = "an H-cut trajectory execution is already active";
          return;
        }
        if (!planned_task_ || !planned_snapshot_ || planned_task_->solutions().empty()) {
          response->success = false;
          response->message = "no valid planned H-cut is available to execute";
          return;
        }
        if (execution_thread_.joinable()) execution_thread_.join();
        execution_in_progress_ = true;
        auto task = std::move(planned_task_);
        planned_snapshot_.reset();
        execution_thread_ = std::thread([this, task = std::move(task)]() mutable {
          try {
            const auto result = task->execute(*task->solutions().front());
            RCLCPP_INFO(
              get_logger(), "H-cut trajectory execution %s",
              result == moveit::core::MoveItErrorCode::SUCCESS ? "completed" : "failed");
          } catch (const std::exception& exception) {
            RCLCPP_ERROR(get_logger(), "H-cut trajectory execution threw: %s", exception.what());
          }
          std::lock_guard<std::mutex> completion_lock(plan_mutex_);
          execution_in_progress_ = false;
        });
        response->success = true;
        response->message = "H-cut trajectory dispatch accepted; monitor controller execution";
      },
      rmw_qos_profile_services_default, callback_group_);
    clear_service_ = create_service<std_srvs::srv::Trigger>(
      "~/clear_plan",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(plan_mutex_);
        if (execution_in_progress_) {
          response->success = false;
          response->message = "cannot clear a plan while an H-cut trajectory execution is active";
          return;
        }
        planned_task_.reset();
        planned_snapshot_.reset();
        response->success = true;
        response->message = "H-cut plan cleared";
      },
      rmw_qos_profile_services_default, callback_group_);
    RCLCPP_INFO(
      get_logger(), "Snapshot H-cut MTC planner ready; hover %.0f mm, cut %.0f mm",
      hover_clearance_m_ * 1000.0, cut_clearance_m_ * 1000.0);
  }

  ~BoxHCutExecutor() override
  {
    if (execution_thread_.joinable()) execution_thread_.join();
  }

private:
  using Snapshot = compal_box_msgs::msg::BoxSnapshot;

  bool captureAndPlan(std::string& message)
  {
    using namespace std::chrono_literals;
    if (!snapshot_client_->wait_for_service(5s)) {
      message = "snapshot service is unavailable";
      return false;
    }
    auto future = snapshot_client_->async_send_request(
      std::make_shared<CaptureSnapshot::Request>());
    if (future.wait_for(10s) != std::future_status::ready) {
      message = "snapshot capture timed out";
      return false;
    }
    const auto response = future.get();
    if (!response->success) {
      message = "snapshot rejected: " + response->message;
      return false;
    }
    return planSnapshot(response->snapshot, message);
  }

  bool captureAndStartTcpDebug(std::string& message)
  {
    using namespace std::chrono_literals;
    if (!snapshot_client_->wait_for_service(5s)) {
      message = "snapshot service is unavailable";
      return false;
    }
    auto future = snapshot_client_->async_send_request(
      std::make_shared<CaptureSnapshot::Request>());
    if (future.wait_for(10s) != std::future_status::ready) {
      message = "snapshot capture timed out";
      return false;
    }
    const auto response = future.get();
    if (!response->success) {
      message = "snapshot rejected: " + response->message;
      return false;
    }
    tcp_debug_snapshot_ = response->snapshot;
    tcp_debug_index_ = 0U;
    return planTcpDebugPoint(message);
  }

  bool planTcpDebugPoint(std::string& message)
  {
    if (!tcp_debug_snapshot_ || tcp_debug_index_ >= 3U) {
      message = "TCP debug session is complete or not started";
      return false;
    }
    std::string validation;
    if (!validateSnapshot(*tcp_debug_snapshot_, validation)) {
      message = "snapshot rejected: " + validation;
      return false;
    }
    const auto& snapshot = *tcp_debug_snapshot_;
    const std::array<std::string, 3> labels{{"seam_start", "seam_end", "seam_center"}};
    CutSegment segment;
    if (tcp_debug_index_ == 0U) {
      segment = {"tcp debug seam start", snapshot.seam_start, snapshot.seam_end, 0.0, 0.0, 0.0, true};
    } else if (tcp_debug_index_ == 1U) {
      segment = {"tcp debug seam end", snapshot.seam_end, snapshot.seam_start, 0.0, 0.0, 0.0, true};
    } else {
      segment = {"tcp debug seam center", snapshot.seam_center, snapshot.seam_end, 0.0, 0.0, 0.0, true};
    }
    std::unique_ptr<mtc::Task> task;
    std::vector<geometry_msgs::msg::PoseStamped> targets;
    std::string planning_message;
    publishCartonGeometry(snapshot);
    if (!planCutSegments(
          snapshot, {segment}, false, true, planning_timeout_s_,
          get_parameter("motion_profile").as_string(),
          "TCP debug " + labels[tcp_debug_index_] + " dry run", true, task, targets,
          planning_message)) {
      message = "TCP debug " + labels[tcp_debug_index_] + " planning failed: " + planning_message;
      return false;
    }
    if (targets.size() != 1U) {
      message = "TCP debug planner produced an unexpected target count";
      return false;
    }
    publishMarkers(targets);
    publishTargetPoses(snapshot, targets);
    planned_snapshot_ = snapshot;
    planned_task_ = std::move(task);
    tcp_debug_target_ = targets.front();
    message = "TCP debug " + labels[tcp_debug_index_] +
      " dry-run plan published to RViz; manually jog only after reviewing it, then call record_tcp_debug_point; this service never dispatches motion";
    return true;
  }

  bool recordTcpDebugPoint(std::string& message)
  {
    if (!tcp_debug_snapshot_ || !tcp_debug_target_ || !planned_task_ || tcp_debug_index_ >= 3U) {
      message = "no TCP debug target is active; call start_tcp_debug first";
      return false;
    }
    sensor_msgs::msg::JointState joint_state;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_joint_state_ || current_joint_state_->name.empty() || current_joint_state_->position.empty()) {
        message = "no /joint_states received; cannot record TCP observation";
        return false;
      }
      joint_state = *current_joint_state_;
    }
    moveit::core::RobotState state(planned_task_->getRobotModel());
    state.setToDefaultValues();
    const auto& variables = planned_task_->getRobotModel()->getVariableNames();
    for (std::size_t i = 0; i < joint_state.name.size() && i < joint_state.position.size(); ++i) {
      if (std::find(variables.begin(), variables.end(), joint_state.name[i]) != variables.end()) {
        state.setVariablePosition(joint_state.name[i], joint_state.position[i]);
      }
    }
    state.update();
    const Eigen::Vector3d observed = state.getGlobalLinkTransform(tool_link_).translation();
    const auto& target = tcp_debug_target_->pose.position;
    const Eigen::Vector3d error(target.x - observed.x(), target.y - observed.y(), target.z - observed.z());
    const std::array<std::string, 3> labels{{"seam_start", "seam_end", "seam_center"}};
    namespace fs = std::filesystem;
    const char* home = std::getenv("HOME");
    if (home == nullptr) {
      message = "cannot save TCP debug observation without HOME";
      return false;
    }
    const fs::path directory = fs::path(home) / "tcp_debug_measurements";
    std::error_code error_code;
    fs::create_directories(directory, error_code);
    if (error_code) {
      message = "cannot create TCP debug measurement directory: " + error_code.message();
      return false;
    }
    const fs::path output = directory / (tcp_debug_snapshot_->snapshot_id + "_" + labels[tcp_debug_index_] + ".json");
    std::ofstream file(output);
    if (!file) {
      message = "cannot write TCP debug observation: " + output.string();
      return false;
    }
    file << "{\n  \"snapshot_id\": \"" << tcp_debug_snapshot_->snapshot_id
         << "\",\n  \"point\": \"" << labels[tcp_debug_index_]
         << "\",\n  \"frame_id\": \"" << planning_frame_
         << "\",\n  \"target_m\": [" << target.x << ", " << target.y << ", " << target.z
         << "],\n  \"observed_tool_tip_m\": [" << observed.x() << ", " << observed.y() << ", " << observed.z()
         << "],\n  \"target_minus_observed_m\": [" << error.x() << ", " << error.y() << ", " << error.z() << "]\n}\n";
    file.close();
    ++tcp_debug_index_;
    if (tcp_debug_index_ == 3U) {
      planned_task_.reset();
      message = "saved final TCP debug observation to " + output.string() +
        "; compare all three target_minus_observed_m values before changing the TCP; no robot motion was dispatched";
      return true;
    }
    std::string next_message;
    if (!planTcpDebugPoint(next_message)) {
      message = "saved TCP debug observation to " + output.string() + "; next target failed: " + next_message;
      return false;
    }
    message = "saved TCP debug observation to " + output.string() + "; " + next_message;
    return true;
  }

  bool planSavedSnapshot(std::string& message)
  {
    std::string source_path;
    const auto snapshot = loadLatestSavedSnapshot(source_path, message);
    if (!snapshot) return false;
    const bool planned = planSnapshot(*snapshot, message);
    message = "saved snapshot " + source_path + ": " + message;
    return planned;
  }

  bool planNamedSavedSnapshot(const std::string& snapshot_file, std::string& message)
  {
    std::string source_path;
    const auto snapshot = loadLatestSavedSnapshot(source_path, message, snapshot_file);
    if (!snapshot) return false;
    const bool planned = planSnapshot(*snapshot, message);
    message = "saved snapshot " + source_path + ": " + message;
    return planned;
  }

  bool diagnoseSavedSnapshotSections(std::string& message)
  {
    std::string source_path;
    const auto snapshot = loadLatestSavedSnapshot(source_path, message);
    if (!snapshot) return false;
    std::string validation;
    const bool canonical = validateSnapshot(*snapshot, validation);
    const auto motion_profile = get_parameter("motion_profile").as_string();
    const std::array<CutSegment, 4> sections{{
      {"left side cut", snapshot->back_left, snapshot->front_left},
      {"middle seam cut", snapshot->seam_start, snapshot->seam_end},
      {"right side cut forward", snapshot->back_right, snapshot->front_right},
      {"right side cut reverse", snapshot->front_right, snapshot->back_right},
    }};
    std::ostringstream report;
    report << "diagnostic_only snapshot=" << snapshot->snapshot_id
           << " canonical=" << (canonical ? "true" : "false")
           << " validation=" << (canonical ? "ok" : validation);
    for (const auto& section : sections) {
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> targets;
      std::string result;
      const bool planned = planCutSegments(
        *snapshot, {section}, false, true, planning_timeout_s_, motion_profile,
        "section diagnostic " + section.name, true, task, targets, result);
      report << "; " << section.name << "=" << (planned ? "planned" : "failed")
             << " (" << result << ")";
    }
    message = "saved snapshot " + source_path + ": " + report.str();
    return true;
  }

  std::optional<Snapshot> loadLatestSavedSnapshot(
    std::string& source_path, std::string& message,
    const std::string& requested_file_name = "") const
  {
    namespace fs = std::filesystem;
    std::string directory = saved_snapshot_dir_;
    if (directory.rfind("~/", 0) == 0) {
      const char* home = std::getenv("HOME");
      if (home == nullptr) {
        message = "cannot expand saved_snapshot_dir without HOME";
        return std::nullopt;
      }
      directory = std::string(home) + directory.substr(1);
    }
    std::error_code error;
    fs::path selected;
    const auto requested_file = requested_file_name.empty()
      ? get_parameter("saved_snapshot_file").as_string()
      : requested_file_name;
    if (!requested_file.empty()) {
      const fs::path requested_path(requested_file);
      if (requested_path.filename() != requested_path || requested_path.extension() != ".json") {
        message = "saved_snapshot_file must be a .json filename inside " + directory;
        return std::nullopt;
      }
      selected = fs::path(directory) / requested_path;
      if (!fs::is_regular_file(selected, error) || error) {
        message = "selected saved snapshot is unavailable: " + selected.string();
        return std::nullopt;
      }
    } else {
      fs::file_time_type newest_time;
      for (const auto& entry : fs::directory_iterator(directory, error)) {
        if (error) break;
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
        const auto modified = entry.last_write_time(error);
        if (error) break;
        if (selected.empty() || modified > newest_time) {
          selected = entry.path();
          newest_time = modified;
        }
      }
      if (error || selected.empty()) {
        message = "no saved snapshot JSON is available in " + directory;
        return std::nullopt;
      }
    }
    const auto modified = fs::last_write_time(selected, error);
    if (error) {
      message = "cannot read saved snapshot modification time: " + selected.string();
      return std::nullopt;
    }
    const auto age_s = std::chrono::duration<double>(
      fs::file_time_type::clock::now() - modified).count();
    if (saved_snapshot_max_age_s_ > 0.0 && age_s > saved_snapshot_max_age_s_) {
      std::ostringstream age_message;
      age_message << "saved snapshot is stale (" << age_s << " s; maximum "
                  << saved_snapshot_max_age_s_ << " s): " << selected.string();
      message = age_message.str();
      return std::nullopt;
    }
    std::ifstream input(selected);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string json = buffer.str();
    const std::string number = "([-+0-9.eE]+)";
    const auto scalar = [&json, &number](const std::string& name, double& value) {
      const std::regex expression("\"" + name + "\"" + R"([[:space:]]*:[[:space:]]*)" + number);
      std::smatch match;
      if (!std::regex_search(json, match, expression)) return false;
      value = std::stod(match[1].str());
      return true;
    };
    const auto point = [&json, &number](const std::string& name, geometry_msgs::msg::Point& value) {
      const std::regex expression(
        "\"" + name + "\"" + R"([[:space:]]*:[[:space:]]*[{][[:space:]]*"x"[[:space:]]*:[[:space:]]*)" + number +
        R"([[:space:]]*,[[:space:]]*"y"[[:space:]]*:[[:space:]]*)" + number +
        R"([[:space:]]*,[[:space:]]*"z"[[:space:]]*:[[:space:]]*)" + number);
      std::smatch match;
      if (!std::regex_search(json, match, expression)) return false;
      value.x = std::stod(match[1].str());
      value.y = std::stod(match[2].str());
      value.z = std::stod(match[3].str());
      return true;
    };
    std::smatch frame_match, id_match;
    if (!std::regex_search(json, id_match, std::regex(R"json("snapshot_id"[[:space:]]*:[[:space:]]*"([^"]+)")json")) ||
        !std::regex_search(json, frame_match, std::regex(R"json("frame_id"[[:space:]]*:[[:space:]]*"([^"]+)")json"))) {
      message = "saved snapshot JSON is missing snapshot_id or frame_id: " + selected.string();
      return std::nullopt;
    }
    Snapshot snapshot;
    snapshot.snapshot_id = id_match[1].str();
    snapshot.header.frame_id = frame_match[1].str();
    geometry_msgs::msg::Point normal;
    if (!point("seam_start", snapshot.seam_start) || !point("seam_end", snapshot.seam_end) ||
        !point("seam_center", snapshot.seam_center) || !point("back_left", snapshot.back_left) ||
        !point("back_right", snapshot.back_right) || !point("front_left", snapshot.front_left) ||
        !point("front_right", snapshot.front_right) || !point("surface_normal", normal) ||
        !scalar("plane_rms_m", snapshot.plane_rms_m)) {
      message = "saved snapshot JSON is incomplete: " + selected.string();
      return std::nullopt;
    }
    snapshot.surface_normal.x = normal.x;
    snapshot.surface_normal.y = normal.y;
    snapshot.surface_normal.z = normal.z;
    source_path = selected.string();
    return snapshot;
  }

  moveit_msgs::msg::CollisionObject cartonCollisionObject(const Snapshot& snapshot) const
  {
    const auto vector = [](const geometry_msgs::msg::Point& from,
                           const geometry_msgs::msg::Point& to) {
      return std::array<double, 3>{to.x - from.x, to.y - from.y, to.z - from.z};
    };
    const auto dot = [](const std::array<double, 3>& left, const std::array<double, 3>& right) {
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    };
    const auto magnitude = [&dot](const std::array<double, 3>& value) {
      return std::sqrt(dot(value, value));
    };
    const auto normalize = [&magnitude](std::array<double, 3> value) {
      const double length = magnitude(value);
      for (auto& component : value) component /= length;
      return value;
    };
    const auto cross = [](const std::array<double, 3>& left, const std::array<double, 3>& right) {
      return std::array<double, 3>{
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]};
    };
    const auto normal = normalize({
      snapshot.surface_normal.x, snapshot.surface_normal.y, snapshot.surface_normal.z});
    std::array<double, 3> top_center{
      (snapshot.back_left.x + snapshot.back_right.x + snapshot.front_left.x + snapshot.front_right.x) / 4.0,
      (snapshot.back_left.y + snapshot.back_right.y + snapshot.front_left.y + snapshot.front_right.y) / 4.0,
      (snapshot.back_left.z + snapshot.back_right.z + snapshot.front_left.z + snapshot.front_right.z) / 4.0};
    auto x_axis = vector(snapshot.seam_start, snapshot.seam_end);
    const double normal_component = dot(x_axis, normal);
    for (std::size_t i = 0; i < 3; ++i) x_axis[i] -= normal_component * normal[i];
    x_axis = normalize(x_axis);
    const auto y_axis = normalize(cross(normal, x_axis));
    const double width = (magnitude(vector(snapshot.back_left, snapshot.back_right)) +
                          magnitude(vector(snapshot.front_left, snapshot.front_right))) / 2.0;
    const double depth = (magnitude(vector(snapshot.back_left, snapshot.front_left)) +
                          magnitude(vector(snapshot.back_right, snapshot.front_right))) / 2.0;

    const double m00=x_axis[0],m01=y_axis[0],m02=normal[0];
    const double m10=x_axis[1],m11=y_axis[1],m12=normal[1];
    const double m20=x_axis[2],m21=y_axis[2],m22=normal[2];
    geometry_msgs::msg::Quaternion q;
    const double trace=m00+m11+m22;
    if(trace>0){double s=std::sqrt(trace+1.0)*2;q.w=.25*s;q.x=(m21-m12)/s;q.y=(m02-m20)/s;q.z=(m10-m01)/s;}
    else if(m00>m11&&m00>m22){double s=std::sqrt(1+m00-m11-m22)*2;q.w=(m21-m12)/s;q.x=.25*s;q.y=(m01+m10)/s;q.z=(m02+m20)/s;}
    else if(m11>m22){double s=std::sqrt(1+m11-m00-m22)*2;q.w=(m02-m20)/s;q.x=(m01+m10)/s;q.y=.25*s;q.z=(m12+m21)/s;}
    else{double s=std::sqrt(1+m22-m00-m11)*2;q.w=(m10-m01)/s;q.x=(m02+m20)/s;q.y=(m12+m21)/s;q.z=.25*s;}

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {width, depth, carton_height_m_};
    geometry_msgs::msg::Pose pose;
    pose.position.x = top_center[0] - normal[0] * carton_height_m_ / 2.0;
    pose.position.y = top_center[1] - normal[1] * carton_height_m_ / 2.0;
    pose.position.z = top_center[2] - normal[2] * carton_height_m_ / 2.0;
    pose.orientation = q;
    moveit_msgs::msg::CollisionObject object;
    object.id = carton_collision_id_;
    object.header = snapshot.header;
    object.primitives.push_back(primitive);
    object.primitive_poses.push_back(pose);
    object.operation = moveit_msgs::msg::CollisionObject::ADD;
    return object;
  }

  bool validateSnapshot(const Snapshot& snapshot, std::string& message) const
  {
    const auto finite = [](const geometry_msgs::msg::Point& point) {
      return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
    };
    const auto vector = [](const geometry_msgs::msg::Point& from,
                           const geometry_msgs::msg::Point& to) {
      return std::array<double, 3>{to.x - from.x, to.y - from.y, to.z - from.z};
    };
    const auto dot = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    const auto length = [&dot](const std::array<double, 3>& value) { return std::sqrt(dot(value, value)); };
    const std::array<const geometry_msgs::msg::Point*, 7> points{
      &snapshot.seam_start, &snapshot.seam_end, &snapshot.back_left, &snapshot.back_right,
      &snapshot.front_left, &snapshot.front_right, &snapshot.seam_center};
    if (snapshot.snapshot_id.empty() || snapshot.header.frame_id != planning_frame_) {
      message = "snapshot must have an ID and use planning frame " + planning_frame_ +
                "; got frame=" + snapshot.header.frame_id;
      return false;
    }
    for (const auto* point : points) {
      if (!finite(*point)) { message = "snapshot contains a non-finite geometric point"; return false; }
    }
    const std::array<double, 3> normal{
      snapshot.surface_normal.x, snapshot.surface_normal.y, snapshot.surface_normal.z};
    const double normal_length = length(normal);
    if (!std::isfinite(normal_length) || std::abs(normal_length - 1.0) > 0.05) {
      message = "snapshot surface_normal must be unit length; norm=" + formatMeters(normal_length);
      return false;
    }
    const auto back = vector(snapshot.back_left, snapshot.back_right);
    const auto left = vector(snapshot.back_left, snapshot.front_left);
    const auto right = vector(snapshot.back_right, snapshot.front_right);
    const auto front = vector(snapshot.front_left, snapshot.front_right);
    const double width = length(back), depth = length(left);
    if (width < 0.04 || depth < 0.04 || length(right) < 0.04 || length(front) < 0.04) {
      message = "snapshot carton edges are degenerate";
      return false;
    }
    if (std::abs(dot(left, right) / (depth * length(right))) < 0.95 ||
        std::abs(dot(back, front) / (width * length(front))) < 0.95) {
      message = "snapshot corner ordering is not a rectangular carton top face";
      return false;
    }
    const auto on_top_face = [&](const geometry_msgs::msg::Point& point) {
      const auto relative = vector(snapshot.back_left, point);
      const double x = dot(relative, back) / width;
      const double y = dot(relative, left) / depth;
      return x >= -0.01 && x <= width + 0.01 && y >= -0.01 && y <= depth + 0.01 &&
             std::abs(dot(relative, normal)) <= 0.01;
    };
    if (!on_top_face(snapshot.seam_start) || !on_top_face(snapshot.seam_end) ||
        !on_top_face(snapshot.seam_center)) {
      message = "snapshot seam is outside the detected carton top face";
      return false;
    }
    return true;
  }

  void publishCartonGeometry(const Snapshot& snapshot)
  {
    const auto carton = cartonCollisionObject(snapshot);
    carton_collision_pub_->publish(carton);
    RCLCPP_INFO(
      get_logger(), "Published canonical carton geometry from snapshot %s in frame %s",
      snapshot.snapshot_id.c_str(), snapshot.header.frame_id.c_str());
  }

  bool planSnapshot(const Snapshot& snapshot, std::string& message)
  {
    try {
      std::string validation;
      if (!validateSnapshot(snapshot, validation)) {
        message = "snapshot rejected: " + validation;
        return false;
      }
      const auto motion_profile = get_parameter("motion_profile").as_string();
      const std::array<std::string, 4> supported_profiles{
        "ompl_cartesian", "ompl_pilz", "pilz_only", "chomp_pilz"};
      if (std::find(
            supported_profiles.begin(), supported_profiles.end(), motion_profile) ==
          supported_profiles.end()) {
        throw std::invalid_argument("unsupported H-cut motion_profile: " + motion_profile);
      }
      if (cut_pattern_ == "combined" && hasManualFirstSideLink6Pose()) {
        RCLCPP_INFO(
          get_logger(),
          "Using manual first-side link6 pose in %s for snapshot %s",
          combined_first_side_link6_frame_.c_str(), snapshot.snapshot_id.c_str());
      }
      publishCartonGeometry(snapshot);
      if (cut_pattern_ == "combined") {
        return planCombinedSnapshot(snapshot, motion_profile, message);
      }

      const bool partial_cut = cut_pattern_ == "left_partial" || cut_pattern_ == "right_partial";
      std::vector<CutSegment> segments;
      if (cut_pattern_ == "h") {
        segments = {
          {"left side cut", snapshot.back_left, snapshot.front_left},
          {"middle seam cut", snapshot.seam_start, snapshot.seam_end},
          {"right side cut", snapshot.back_right, snapshot.front_right},
        };
      } else if (cut_pattern_ == "middle_only") {
        segments = {{"middle seam cut", snapshot.seam_start, snapshot.seam_end}};
      } else if (cut_pattern_ == "middle_entry_debug") {
        segments = {{"middle seam entry debug", snapshot.seam_start, snapshot.seam_end,
          0.0, 0.0, 0.0, true}};
      } else if (cut_pattern_ == "left_partial") {
        const auto& start = partial_reverse_ ? snapshot.front_left : snapshot.back_left;
        const auto& end = partial_reverse_ ? snapshot.back_left : snapshot.front_left;
        segments = {{"left partial side cut",
          interpolate(start, end, partial_start_fraction_),
          interpolate(start, end, partial_end_fraction_)}};
      } else {
        segments = {{"right partial side cut",
          interpolate(snapshot.front_right, snapshot.back_right, partial_start_fraction_),
          interpolate(snapshot.front_right, snapshot.back_right, partial_end_fraction_)}};
      }
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> cut_targets;
      const bool allow_cutter_carton_contact =
        cut_pattern_ == "middle_only" || cut_pattern_ == "middle_entry_debug" || partial_cut;

      const bool automatic_side_orientation =
        partial_cut && side_cut_orientation_mode_ == "auto";
      const std::string requested_side_orientation_mode = side_cut_orientation_mode_;
      if (automatic_side_orientation) side_cut_orientation_mode_ = "edge_aligned";
      bool planned = planCutSegments(
        snapshot, segments, partial_cut, allow_cutter_carton_contact, planning_timeout_s_,
        motion_profile, "Snapshot-relative " + cut_pattern_ + " dry run", true, task,
        cut_targets, message);
      if (!planned && automatic_side_orientation) {
        side_cut_orientation_mode_ = "middle_aligned";
        planned = planCutSegments(
          snapshot, segments, partial_cut, allow_cutter_carton_contact, planning_timeout_s_,
          motion_profile, "Snapshot-relative " + cut_pattern_ + " top-orientation fallback dry run",
          true, task, cut_targets, message);
      }
      side_cut_orientation_mode_ = requested_side_orientation_mode;
      if (!planned) return false;
      publishMarkers(cut_targets);
      publishTargetPoses(snapshot, cut_targets);
      planned_snapshot_ = snapshot;
      planned_task_ = std::move(task);
      const bool includes_middle_seam =
        cut_pattern_ == "h" || cut_pattern_ == "middle_only" ||
        cut_pattern_ == "middle_entry_debug";
      message = "Snapshot-relative " + cut_pattern_ + " dry-run plan [" + motion_profile +
                "] published to RViz" +
                (includes_middle_seam ? "; middle entry alignment_m=" +
                  formatMeters(middle_entry_alignment_m_) : "") +
                (cut_pattern_ == "middle_entry_debug" ?
                  "; stops at seam_start before the cut traverse" : "") +
                "; execution remains disabled until execution_enabled is set";
      RCLCPP_INFO(
        get_logger(), "Planned snapshot %s with profile %s and pattern %s; middle entry alignment %.1f mm",
        snapshot.snapshot_id.c_str(), motion_profile.c_str(), cut_pattern_.c_str(),
        includes_middle_seam ? middle_entry_alignment_m_ * 1000.0 : 0.0);
      return true;
    } catch (const std::exception& exception) {
      RCLCPP_ERROR(
        get_logger(), "H-cut %s planning initialization failed: %s",
        cut_pattern_.c_str(), exception.what());
      message = cut_pattern_ + " planning failed: " + exception.what();
      return false;
    }
  }

  bool planCombinedSnapshot(
    const Snapshot& snapshot, const std::string& motion_profile, std::string& message)
  {
    const bool use_manual_left_start = hasManualFirstSideLink6Pose();
    std::vector<SideCutCandidate> left_intervals;
    if (use_manual_left_start) {
      const double detected_left_length = distanceBetween(snapshot.back_left, snapshot.front_left);
      if (detected_left_length <= 0.0) {
        message = "combined planning failed: detected left edge is degenerate";
        return false;
      }
      const double left_stroke_length = std::min(manual_first_side_max_length_m_, detected_left_length);
      left_intervals.push_back({
        snapshot.back_left,
        interpolate(snapshot.back_left, snapshot.front_left, left_stroke_length / detected_left_length),
        left_stroke_length, 0.0, 0.0, 0.0});
    } else {
      left_intervals = sideIntervals(snapshot.back_left, snapshot.front_left);
    }
    auto right_intervals = sideIntervals(snapshot.back_right, snapshot.front_right);
    const auto reverse_right_intervals = sideIntervals(snapshot.front_right, snapshot.back_right);
    right_intervals.insert(
      right_intervals.end(), reverse_right_intervals.begin(), reverse_right_intervals.end());
    const auto orientations = orientationOffsets();
    const std::vector<OrientationOffset> manual_left_orientations{{0.0, 0.0, 0.0}};
    const auto& left_orientations = use_manual_left_start ? manual_left_orientations : orientations;
    const std::size_t left_candidate_count = left_intervals.size() * left_orientations.size();
    const std::size_t right_candidate_count = right_intervals.size() * orientations.size();
    const std::size_t total_candidate_pairs = left_candidate_count * right_candidate_count;
    if (left_intervals.empty() || right_intervals.empty()) {
      message = "combined planning failed: no side intervals satisfy inset/grid/min-length limits; "
                "left_intervals=" + std::to_string(left_intervals.size()) +
                " right_intervals=" + std::to_string(right_intervals.size());
      return false;
    }

    struct IntervalPair {
      std::size_t left_index;
      std::size_t right_index;
      double total_length_m;
    };
    std::vector<IntervalPair> interval_pairs;
    interval_pairs.reserve(left_intervals.size() * right_intervals.size());
    for (std::size_t left_index = 0; left_index < left_intervals.size(); ++left_index) {
      for (std::size_t right_index = 0; right_index < right_intervals.size(); ++right_index) {
        interval_pairs.push_back({
          left_index, right_index,
          left_intervals[left_index].length_m + right_intervals[right_index].length_m});
      }
    }
    std::sort(
      interval_pairs.begin(), interval_pairs.end(),
      [&left_intervals, &right_intervals](const IntervalPair& left, const IntervalPair& right) {
        if (left.total_length_m != right.total_length_m) {
          return left.total_length_m > right.total_length_m;
        }
        if (left_intervals[left.left_index].length_m !=
            left_intervals[right.left_index].length_m) {
          return left_intervals[left.left_index].length_m >
                 left_intervals[right.left_index].length_m;
        }
        return right_intervals[left.right_index].length_m >
               right_intervals[right.right_index].length_m;
      });

    const auto start_time = std::chrono::steady_clock::now();
    const auto deadline = start_time + std::chrono::duration<double>(combined_search_budget_s_);
    std::size_t attempted_pairs = 0U;
    std::size_t left_preflight_attempts = 0U;
    bool budget_exhausted = false;
    std::string last_failure = "no candidate attempted";
    std::vector<bool> viable_left(left_intervals.size(), false);
    for (std::size_t index = 0; index < left_intervals.size(); ++index) {
      const auto now_time = std::chrono::steady_clock::now();
      if (now_time >= deadline) { budget_exhausted = true; break; }
      const auto& interval = left_intervals[index];
      for (const auto& orientation : left_orientations) {
        const double remaining_s = std::chrono::duration<double>(
          deadline - std::chrono::steady_clock::now()).count();
        if (remaining_s <= 0.0) { budget_exhausted = true; break; }
        const SideCutCandidate candidate{interval.start, interval.end, interval.length_m,
          orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg};
        const std::vector<CutSegment> left_segment{{"left side cut", candidate.start, candidate.end,
          candidate.roll_deg, candidate.pitch_deg, candidate.yaw_deg}};
        std::unique_ptr<mtc::Task> preflight_task;
        std::vector<geometry_msgs::msg::PoseStamped> ignored_targets;
        ++left_preflight_attempts;
        if (planCutSegments(snapshot, left_segment, false, true, std::min(4.0, remaining_s),
                            motion_profile, "left-side feasibility preflight", false,
                            preflight_task, ignored_targets, last_failure)) {
          viable_left[index] = true;
          break;
        }
      }
      if (budget_exhausted) break;
    }
    const auto viable_left_count = static_cast<std::size_t>(
      std::count(viable_left.begin(), viable_left.end(), true));
    if (viable_left_count == 0U) {
      message = "combined planning rejected before pair search: viable_left_candidates=0/" +
                std::to_string(left_intervals.size()) + " preflight_attempts=" +
                std::to_string(left_preflight_attempts) + " budget_state=" +
                (budget_exhausted ? "exhausted" : "complete") + " failure=" + last_failure;
      return false;
    }

    for (const auto& interval_pair : interval_pairs) {
      if (!viable_left[interval_pair.left_index]) continue;
      const auto& left_interval = left_intervals[interval_pair.left_index];
      const auto& right_interval = right_intervals[interval_pair.right_index];
      for (const auto& left_orientation : left_orientations) {
        for (const auto& right_orientation : orientations) {
          const auto now_time = std::chrono::steady_clock::now();
          if (now_time >= deadline) {
            budget_exhausted = true;
            break;
          }
          const double remaining_s = std::chrono::duration<double>(deadline - now_time).count();
          const double candidate_timeout_s = std::min(planning_timeout_s_, remaining_s);
          const SideCutCandidate left_candidate{
            left_interval.start, left_interval.end, left_interval.length_m,
            left_orientation.roll_deg, left_orientation.pitch_deg, left_orientation.yaw_deg};
          const SideCutCandidate right_candidate{
            right_interval.start, right_interval.end, right_interval.length_m,
            right_orientation.roll_deg, right_orientation.pitch_deg, right_orientation.yaw_deg};
          const std::vector<CutSegment> segments = {
            {"left side cut", left_candidate.start, left_candidate.end,
             left_candidate.roll_deg, left_candidate.pitch_deg, left_candidate.yaw_deg},
            {"middle seam cut", snapshot.seam_start, snapshot.seam_end},
            {"right side cut", right_candidate.start, right_candidate.end,
             right_candidate.roll_deg, right_candidate.pitch_deg, right_candidate.yaw_deg},
          };
          std::unique_ptr<mtc::Task> candidate_task;
          std::vector<geometry_msgs::msg::PoseStamped> cut_targets;
          ++attempted_pairs;
          if (planCutSegments(
                snapshot, segments, false, true, candidate_timeout_s, motion_profile,
                "Snapshot-relative combined bounded candidate", false, candidate_task,
                cut_targets, last_failure)) {
            const auto finish_time = std::chrono::steady_clock::now();
            const bool expired_after_plan = finish_time >= deadline;
            publishMarkers(cut_targets);
            publishTargetPoses(snapshot, cut_targets);
            planned_snapshot_ = snapshot;
            planned_task_ = std::move(candidate_task);
            const std::string budget_state = expired_after_plan ?
              "bounded_not_globally_proven_budget_expired_after_feasible_plan" :
              "highest_ranked_feasible_pair_found_before_budget";
            message = "Snapshot-relative combined dry-run plan [" + motion_profile +
                      "] published to RViz; left_length_m=" +
                      formatMeters(left_candidate.length_m) + " right_length_m=" +
                      formatMeters(right_candidate.length_m) + " total_length_m=" +
                      formatMeters(left_candidate.length_m + right_candidate.length_m) +
                      " left_rpy_deg=" + formatRpy(left_candidate) +
                      " right_rpy_deg=" + formatRpy(right_candidate) +
                      " interval_candidates_left=" + std::to_string(left_intervals.size()) +
                      " interval_candidates_right=" + std::to_string(right_intervals.size()) +
                      " orientation_count=" + std::to_string(orientations.size()) +
                      " side_candidates_left=" + std::to_string(left_candidate_count) +
                      " side_candidates_right=" + std::to_string(right_candidate_count) +
                      " candidate_pairs=" + std::to_string(total_candidate_pairs) +
                      " attempted_pairs=" + std::to_string(attempted_pairs) +
                      " budget_s=" + formatMeters(combined_search_budget_s_) +
                      " budget_state=" + budget_state +
                      "; execution remains disabled until execution_enabled is set";
            RCLCPP_INFO(
              get_logger(),
              "Planned snapshot %s combined bounded candidate: left %.1f mm rpy %s, "
              "right %.1f mm rpy %s, attempted %zu/%zu, %s",
              snapshot.snapshot_id.c_str(), left_candidate.length_m * 1000.0,
              formatRpy(left_candidate).c_str(), right_candidate.length_m * 1000.0,
              formatRpy(right_candidate).c_str(), attempted_pairs, total_candidate_pairs,
              budget_state.c_str());
            return true;
          }
        }
        if (budget_exhausted) break;
      }
      if (budget_exhausted) break;
    }

    message = "combined bounded search found no feasible plan; interval_candidates_left=" +
              std::to_string(left_intervals.size()) + " interval_candidates_right=" +
              std::to_string(right_intervals.size()) + " orientation_count=" +
              std::to_string(orientations.size()) + " side_candidates_left=" +
              std::to_string(left_candidate_count) + " side_candidates_right=" +
              std::to_string(right_candidate_count) + " candidate_pairs=" +
              std::to_string(total_candidate_pairs) + " attempted_pairs=" +
              std::to_string(attempted_pairs) + " budget_s=" +
              formatMeters(combined_search_budget_s_) + " budget_state=" +
              (budget_exhausted ? "bounded_not_globally_proven_budget_exhausted" :
                                  "complete_no_feasible_candidate") +
              " last_failure=" + last_failure;
    return false;
  }

  bool planCutSegments(
    const Snapshot& snapshot, const std::vector<CutSegment>& segments, bool partial_cut,
    bool allow_cutter_carton_contact, double planner_timeout_s,
    const std::string& motion_profile, const std::string& task_name, bool explain_failure,
    std::unique_ptr<mtc::Task>& task, std::vector<geometry_msgs::msg::PoseStamped>& cut_targets,
    std::string& message)
  {
    task = std::make_unique<mtc::Task>();
    task->stages()->setName(task_name);
    task->loadRobotModel(shared_from_this());
    task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));
    const auto carton = cartonCollisionObject(snapshot);
    auto add_carton = std::make_unique<mtc::stages::ModifyPlanningScene>(
      "add detected carton collision object");
    add_carton->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    add_carton->addObject(carton);
    task->add(std::move(add_carton));
    moveit::core::RobotState ik_state(task->getRobotModel());
    ik_state.setToDefaultValues();
    const auto& model_variables = task->getRobotModel()->getVariableNames();
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_joint_state_ || current_joint_state_->name.empty() ||
          current_joint_state_->position.empty()) {
        throw std::runtime_error(
          "no /joint_states received; start the ReBeL driver before planning an H-cut");
      }
      for (std::size_t index = 0; index < current_joint_state_->name.size() &&
                                  index < current_joint_state_->position.size(); ++index) {
        if (std::find(
              model_variables.begin(), model_variables.end(),
              current_joint_state_->name[index]) != model_variables.end()) {
          ik_state.setVariablePosition(
            current_joint_state_->name[index], current_joint_state_->position[index]);
        }
      }
      ik_state.update();
    }
    const auto* joint_group = task->getRobotModel()->getJointModelGroup(planning_group_);
    if (joint_group == nullptr) {
      throw std::runtime_error("planning group is missing from robot model");
    }

    const auto configure_pipeline = [this, planner_timeout_s](const std::string& pipeline_name) {
      auto planner = std::make_shared<mtc::solvers::PipelinePlanner>(
        shared_from_this(), pipeline_name);
      planner->setTimeout(planner_timeout_s);
      planner->setMaxVelocityScalingFactor(velocity_scaling_);
      planner->setMaxAccelerationScalingFactor(acceleration_scaling_);
      return planner;
    };
    auto ompl = configure_pipeline(planning_pipeline_);
    ompl->setPlannerId("RRTConnectkConfigDefault");
    auto chomp = configure_pipeline(chomp_pipeline_);
    auto pilz_ptp = configure_pipeline(pilz_pipeline_);
    pilz_ptp->setPlannerId("PTP");
    auto pilz_lin = configure_pipeline(pilz_pipeline_);
    pilz_lin->setPlannerId("LIN");
    auto cartesian = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian->setIKFrame(tool_link_);
    cartesian->setStepSize(cartesian_step_m_);
    cartesian->setJumpThreshold(0.0);
    cartesian->setMinFraction(1.0);
    cartesian->setMaxVelocityScalingFactor(velocity_scaling_);
    cartesian->setMaxAccelerationScalingFactor(acceleration_scaling_);

    mtc::solvers::PlannerInterfacePtr transit_planner;
    mtc::solvers::PlannerInterfacePtr stroke_planner;
    if (motion_profile == "ompl_cartesian") {
      transit_planner = ompl;
      stroke_planner = cartesian;
    } else if (motion_profile == "ompl_pilz") {
      transit_planner = ompl;
      stroke_planner = pilz_lin;
    } else if (motion_profile == "pilz_only") {
      transit_planner = pilz_ptp;
      stroke_planner = pilz_lin;
    } else {
      transit_planner = chomp;
      stroke_planner = pilz_lin;
    }
    if (!allow_cutter_carton_contact && !approach_joint_degrees_.empty()) {
      auto approach = std::make_unique<mtc::stages::MoveTo>(
        "transit to supplied link6 approach [" + motion_profile + "]", transit_planner);
      approach->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
      approach->setGroup(planning_group_);
      approach->setGoal(jointGoalDegrees(joint_group, approach_joint_degrees_));
      task->add(std::move(approach));
    }
    if (allow_cutter_carton_contact) {
      auto allow_cutter_contact = std::make_unique<mtc::stages::ModifyPlanningScene>(
        "allow cutter/carton contact for bounded combined candidate");
      allow_cutter_contact->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
      allow_cutter_contact->allowCollisions(carton_collision_id_, "xeg32_cutter_link", true);
      task->add(std::move(allow_cutter_contact));
    }

    cut_targets.clear();
    const std::optional<geometry_msgs::msg::PoseStamped> manual_first_side_pose =
      cut_pattern_ == "combined" && hasManualFirstSideLink6Pose() ?
      std::make_optional(manualFirstSideToolPose(snapshot, task->getRobotModel())) : std::nullopt;
    for (const auto& segment : segments) {
      const bool is_middle_seam =
        segment.name == "middle seam cut" || segment.name == "middle seam entry debug";
      const bool use_manual_first_side_pose =
        manual_first_side_pose && segment.name == "left side cut";
      const double tilt_deg = is_middle_seam ? 0.0 : side_cut_tilt_deg_;
      auto orientation_start_target = segment.end;
      auto orientation_end_target = segment.end;
      orientation_end_target.x += segment.end.x - segment.start.x;
      orientation_end_target.y += segment.end.y - segment.start.y;
      orientation_end_target.z += segment.end.z - segment.start.z;
      if (!is_middle_seam && side_cut_orientation_mode_ == "middle_aligned") {
        const double seam_dx = snapshot.seam_end.x - snapshot.seam_start.x;
        const double seam_dy = snapshot.seam_end.y - snapshot.seam_start.y;
        const double seam_dz = snapshot.seam_end.z - snapshot.seam_start.z;
        orientation_start_target = segment.start;
        orientation_start_target.x += seam_dx;
        orientation_start_target.y += seam_dy;
        orientation_start_target.z += seam_dz;
        orientation_end_target = segment.end;
        orientation_end_target.x += seam_dx;
        orientation_end_target.y += seam_dy;
        orientation_end_target.z += seam_dz;
      }
      auto hover_start = targetPose(
        snapshot, segment.start, orientation_start_target, hover_clearance_m_, tilt_deg,
        segment.roll_deg, segment.pitch_deg, segment.yaw_deg);
      auto cut_start = targetPose(
        snapshot, segment.start, orientation_start_target, cut_clearance_m_, tilt_deg,
        segment.roll_deg, segment.pitch_deg, segment.yaw_deg);
      auto hover_end = targetPose(
        snapshot, segment.end, orientation_end_target, hover_clearance_m_, tilt_deg,
        segment.roll_deg, segment.pitch_deg, segment.yaw_deg);
      auto cut_end = targetPose(
        snapshot, segment.end, orientation_end_target, cut_clearance_m_, tilt_deg,
        segment.roll_deg, segment.pitch_deg, segment.yaw_deg);
      if (use_manual_first_side_pose) {
        cut_start = *manual_first_side_pose;
        cut_start.pose.position.x = segment.start.x + manual_first_side_contact_offset_xyz_[0];
        cut_start.pose.position.y = segment.start.y + manual_first_side_contact_offset_xyz_[1];
        cut_start.pose.position.z = segment.start.z + manual_first_side_contact_offset_xyz_[2];
        hover_start = cut_start;
        hover_start.pose.position.x += hover_clearance_m_ * snapshot.surface_normal.x;
        hover_start.pose.position.y += hover_clearance_m_ * snapshot.surface_normal.y;
        hover_start.pose.position.z += hover_clearance_m_ * snapshot.surface_normal.z;
        const double dx = segment.end.x - segment.start.x;
        const double dy = segment.end.y - segment.start.y;
        const double dz = segment.end.z - segment.start.z;
        const double direction_length = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (direction_length <= 0.0) throw std::runtime_error("zero-length manual left cut direction");
        const double requested_stroke_length = std::min(
          manual_first_side_max_length_m_, direction_length);
        cut_end = cut_start;
        cut_end.pose.position.x += requested_stroke_length * dx / direction_length;
        cut_end.pose.position.y += requested_stroke_length * dy / direction_length;
        cut_end.pose.position.z += requested_stroke_length * dz / direction_length;
        cut_end.pose.orientation = cut_start.pose.orientation;
        hover_end = cut_end;
        hover_end.pose.position.x += hover_clearance_m_ * snapshot.surface_normal.x;
        hover_end.pose.position.y += hover_clearance_m_ * snapshot.surface_normal.y;
        hover_end.pose.position.z += hover_clearance_m_ * snapshot.surface_normal.z;
      }
      const bool use_supplied_start = partial_cut && use_approach_pose_as_cut_start_;
      if (use_supplied_start) {
        const double dx = segment.end.x - segment.start.x;
        const double dy = segment.end.y - segment.start.y;
        const double dz = segment.end.z - segment.start.z;
        const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (length <= 0.0) throw std::runtime_error("zero-length supplied cut direction");
        cut_start = poseFromParameters(snapshot, approach_cut_start_xyz_, approach_cut_orientation_xyzw_);
        if (side_cut_orientation_mode_ == "middle_aligned") {
          cut_start.pose.orientation = targetPose(
            snapshot, segment.start, orientation_start_target, 0.0).pose.orientation;
        }
        cut_end = cut_start;
        cut_end.pose.position.x += approach_cut_length_m_ * dx / length;
        cut_end.pose.position.y += approach_cut_length_m_ * dy / length;
        cut_end.pose.position.z += approach_cut_length_m_ * dz / length;
      }
      if (segment.name.rfind("right side cut", 0) == 0U) {
        moveit::core::RobotState approach_ik_state(ik_state);
        if (!approach_ik_state.setFromIK(joint_group, hover_start.pose, tool_link_, 0.25)) {
          message = "right-side approach IK infeasible before Cartesian stroke";
          return false;
        }
      }
      auto entry_alignment = hover_start;
      if (is_middle_seam && middle_entry_alignment_m_ > 0.0) {
        const double dx = segment.end.x - segment.start.x;
        const double dy = segment.end.y - segment.start.y;
        const double dz = segment.end.z - segment.start.z;
        const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (length <= 0.0) throw std::runtime_error("zero-length middle seam direction");
        entry_alignment.pose.position.x -= middle_entry_alignment_m_ * dx / length;
        entry_alignment.pose.position.y -= middle_entry_alignment_m_ * dy / length;
        entry_alignment.pose.position.z -= middle_entry_alignment_m_ * dz / length;
      }
      cut_targets.push_back(cut_start);
      if (!segment.stop_after_start) cut_targets.push_back(cut_end);
      if (!use_supplied_start) {
        auto transit = std::make_unique<mtc::stages::MoveTo>(
          "transit above " + segment.name + " entry [" + motion_profile + "]", transit_planner);
        transit->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
        transit->setGroup(planning_group_);
        transit->setIKFrame(tool_link_);
        if (partial_cut) {
          transit->setGoal(jointGoalDegrees(joint_group, partial_start_joint_degrees_));
        } else if (motion_profile == "chomp_pilz") {
          transit->setGoal(jointGoal(ik_state, joint_group, entry_alignment));
        } else {
          transit->setGoal(entry_alignment);
        }
        task->add(std::move(transit));
      }
      if (is_middle_seam && middle_entry_alignment_m_ > 0.0) {
        auto align = std::make_unique<mtc::stages::MoveTo>(
          "align above middle seam start [" + motion_profile + "]", stroke_planner);
        align->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
        align->setGroup(planning_group_);
        align->setIKFrame(tool_link_);
        align->setGoal(hover_start);
        task->add(std::move(align));
      }
      if (!partial_cut) {
        auto descend = std::make_unique<mtc::stages::MoveTo>(
          "descend from aligned " + segment.name + " start [" + motion_profile + "]",
          stroke_planner);
        descend->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
        descend->setGroup(planning_group_);
        descend->setIKFrame(tool_link_);
        descend->setGoal(cut_start);
        task->add(std::move(descend));
      }
      if (segment.stop_after_start) {
        continue;
      }

      auto traverse = std::make_unique<mtc::stages::MoveTo>(
        "cut " + segment.name + " [" + motion_profile + "]", stroke_planner);
      traverse->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
      traverse->setGroup(planning_group_);
      traverse->setIKFrame(tool_link_);
      traverse->setGoal(cut_end);
      task->add(std::move(traverse));
      if (!partial_cut) {
        auto ascend = std::make_unique<mtc::stages::MoveTo>(
          "retreat above " + segment.name + " [" + motion_profile + "]", stroke_planner);
        ascend->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
        ascend->setGroup(planning_group_);
        ascend->setIKFrame(tool_link_);
        ascend->setGoal(hover_end);
        task->add(std::move(ascend));
      }
    }

    try {
      task->init();
    } catch (const mtc::InitStageException& exception) {
      std::ostringstream details;
      details << exception;
      throw std::runtime_error("MTC stage initialization failed: " + details.str());
    }
    const auto result = task->plan(static_cast<std::size_t>(max_solutions_));
    if (result != moveit::core::MoveItErrorCode::SUCCESS || task->solutions().empty()) {
      if (explain_failure) {
        task->explainFailure();
      }
      message = "MTC candidate failed: task=" + task_name + " profile=" + motion_profile +
                " segments=" + std::to_string(segments.size());
      return false;
    }
    task->introspection().publishSolution(*task->solutions().front());
    return true;
  }

  geometry_msgs::msg::Point interpolate(
    const geometry_msgs::msg::Point& start, const geometry_msgs::msg::Point& end,
    double fraction) const
  {
    geometry_msgs::msg::Point point;
    point.x = start.x + fraction * (end.x - start.x);
    point.y = start.y + fraction * (end.y - start.y);
    point.z = start.z + fraction * (end.z - start.z);
    return point;
  }

  double distanceBetween(
    const geometry_msgs::msg::Point& start, const geometry_msgs::msg::Point& end) const
  {
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double dz = end.z - start.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  std::vector<SideCutCandidate> sideIntervals(
    const geometry_msgs::msg::Point& back_corner,
    const geometry_msgs::msg::Point& front_corner) const
  {
    const double edge_length = distanceBetween(back_corner, front_corner);
    std::vector<SideCutCandidate> intervals;
    if (edge_length <= 0.0 ||
        edge_length < 2.0 * side_candidate_corner_inset_m_ + side_candidate_min_length_m_) {
      return intervals;
    }
    const int first_step = static_cast<int>(
      std::ceil(side_candidate_corner_inset_m_ / side_candidate_grid_m_));
    const int max_step = static_cast<int>(
      std::floor((edge_length - side_candidate_corner_inset_m_) / side_candidate_grid_m_));
    for (int start_step = first_step; start_step <= max_step; ++start_step) {
      const double start_inset = start_step * side_candidate_grid_m_;
      for (int end_step = first_step; end_step <= max_step; ++end_step) {
        const double end_inset = end_step * side_candidate_grid_m_;
        const double candidate_length = edge_length - start_inset - end_inset;
        if (candidate_length + 1e-9 < side_candidate_min_length_m_) {
          continue;
        }
        intervals.push_back({
          interpolate(back_corner, front_corner, start_inset / edge_length),
          interpolate(back_corner, front_corner, 1.0 - end_inset / edge_length),
          candidate_length, 0.0, 0.0, 0.0});
      }
    }
    std::sort(
      intervals.begin(), intervals.end(),
      [](const SideCutCandidate& left, const SideCutCandidate& right) {
        return left.length_m > right.length_m;
      });
    return intervals;
  }

  std::vector<OrientationOffset> orientationOffsets() const
  {
    std::vector<OrientationOffset> orientations;
    orientations.reserve(
      side_orientation_offsets_deg_.size() * side_orientation_offsets_deg_.size() *
      side_orientation_offsets_deg_.size());
    for (const double roll_deg : side_orientation_offsets_deg_) {
      for (const double pitch_deg : side_orientation_offsets_deg_) {
        for (const double yaw_deg : side_orientation_offsets_deg_) {
          orientations.push_back({roll_deg, pitch_deg, yaw_deg});
        }
      }
    }
    std::sort(
      orientations.begin(), orientations.end(),
      [](const OrientationOffset& left, const OrientationOffset& right) {
        const double left_sum =
          std::abs(left.roll_deg) + std::abs(left.pitch_deg) + std::abs(left.yaw_deg);
        const double right_sum =
          std::abs(right.roll_deg) + std::abs(right.pitch_deg) + std::abs(right.yaw_deg);
        if (left_sum != right_sum) return left_sum < right_sum;
        const double left_max = std::max(
          {std::abs(left.roll_deg), std::abs(left.pitch_deg), std::abs(left.yaw_deg)});
        const double right_max = std::max(
          {std::abs(right.roll_deg), std::abs(right.pitch_deg), std::abs(right.yaw_deg)});
        if (left_max != right_max) return left_max < right_max;
        if (left.roll_deg != right.roll_deg) return left.roll_deg < right.roll_deg;
        if (left.pitch_deg != right.pitch_deg) return left.pitch_deg < right.pitch_deg;
        return left.yaw_deg < right.yaw_deg;
      });
    return orientations;
  }

  geometry_msgs::msg::PoseStamped poseFromParameters(
    const Snapshot& snapshot, const std::vector<double>& xyz,
    const std::vector<double>& orientation) const
  {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = snapshot.header;
    pose.pose.position.x = xyz[0];
    pose.pose.position.y = xyz[1];
    pose.pose.position.z = xyz[2];
    pose.pose.orientation.x = orientation[0];
    pose.pose.orientation.y = orientation[1];
    pose.pose.orientation.z = orientation[2];
    pose.pose.orientation.w = orientation[3];
    return pose;
  }

  bool hasManualFirstSideLink6Pose() const
  {
    return !get_parameter("combined_first_side_link6_frame").as_string().empty();
  }

  geometry_msgs::msg::PoseStamped manualFirstSideToolPose(
    const Snapshot& snapshot, const moveit::core::RobotModelConstPtr& robot_model) const
  {
    if (combined_first_side_link6_frame_ != snapshot.header.frame_id) {
      throw std::runtime_error(
        "manual first-side link6 pose frame " + combined_first_side_link6_frame_ +
        " does not match snapshot frame " + snapshot.header.frame_id);
    }
    if (!robot_model->getLinkModel("link6") || !robot_model->getLinkModel(tool_link_)) {
      throw std::runtime_error("manual first-side pose requires link6 and configured tool_link");
    }

    moveit::core::RobotState default_state(robot_model);
    default_state.setToDefaultValues();
    const Eigen::Isometry3d link6_to_tool =
      default_state.getGlobalLinkTransform("link6").inverse() *
      default_state.getGlobalLinkTransform(tool_link_);

    Eigen::Quaterniond link6_orientation(
      combined_first_side_link6_orientation_xyzw_[3],
      combined_first_side_link6_orientation_xyzw_[0],
      combined_first_side_link6_orientation_xyzw_[1],
      combined_first_side_link6_orientation_xyzw_[2]);
    if (link6_orientation.squaredNorm() <= 1e-12) {
      throw std::runtime_error("manual first-side link6 orientation is zero");
    }
    link6_orientation.normalize();
    Eigen::Isometry3d link6_pose = Eigen::Isometry3d::Identity();
    link6_pose.translation() = Eigen::Vector3d(
      combined_first_side_link6_xyz_[0], combined_first_side_link6_xyz_[1],
      combined_first_side_link6_xyz_[2]);
    link6_pose.linear() = link6_orientation.toRotationMatrix();
    const Eigen::Isometry3d tool_pose = link6_pose * link6_to_tool;
    const Eigen::Quaterniond tool_orientation(tool_pose.rotation());

    geometry_msgs::msg::PoseStamped pose;
    pose.header = snapshot.header;
    pose.pose.position.x = tool_pose.translation().x();
    pose.pose.position.y = tool_pose.translation().y();
    pose.pose.position.z = tool_pose.translation().z();
    pose.pose.orientation.x = tool_orientation.x();
    pose.pose.orientation.y = tool_orientation.y();
    pose.pose.orientation.z = tool_orientation.z();
    pose.pose.orientation.w = tool_orientation.w();
    return pose;
  }
  std::string formatMeters(double value) const
  {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(3);
    stream << value;
    return stream.str();
  }

  std::string formatRpy(const SideCutCandidate& candidate) const
  {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(1);
    stream << "[" << candidate.roll_deg << "," << candidate.pitch_deg << "," <<
      candidate.yaw_deg << "]";
    return stream.str();
  }

  geometry_msgs::msg::PoseStamped targetPose(
    const Snapshot& snapshot, const geometry_msgs::msg::Point& point,
    const geometry_msgs::msg::Point& direction_target, double clearance_m,
    double tilt_deg = 0.0, double roll_deg = 0.0, double pitch_deg = 0.0,
    double yaw_deg = 0.0) const
  {
    const auto normalize = [](std::array<double, 3> value) {
      const double norm = std::sqrt(
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
      for (auto& component : value) component /= norm;
      return value;
    };
    const auto cross = [](const auto& a, const auto& b) {
      return std::array<double, 3>{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]};
    };
    const auto z_axis = normalize({
      -snapshot.surface_normal.x, -snapshot.surface_normal.y,
      -snapshot.surface_normal.z});
    // The cutter's local Y axis is its blade length. Keep it along the seam,
    // rather than using local X as was appropriate only for the top-seam trial.
    std::array<double, 3> y_axis{
      direction_target.x - point.x,
      direction_target.y - point.y,
      direction_target.z - point.z};
    const double projection =
      y_axis[0] * z_axis[0] + y_axis[1] * z_axis[1] + y_axis[2] * z_axis[2];
    for (std::size_t i = 0; i < 3; ++i) y_axis[i] -= projection * z_axis[i];
    y_axis = normalize(y_axis);
    auto x_axis = normalize(cross(y_axis, z_axis));
    auto tilted_z_axis = z_axis;
    // Side tilt is a roll about the blade, so it cannot rotate the blade off seam.
    const double radians = tilt_deg * std::acos(-1.0) / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    for (std::size_t i = 0; i < 3; ++i) {
      const double original_x = x_axis[i];
      x_axis[i] = cosine * original_x - sine * z_axis[i];
      tilted_z_axis[i] = sine * original_x + cosine * z_axis[i];
    }

    const double m00=x_axis[0],m01=y_axis[0],m02=tilted_z_axis[0];
    const double m10=x_axis[1],m11=y_axis[1],m12=tilted_z_axis[1];
    const double m20=x_axis[2],m21=y_axis[2],m22=tilted_z_axis[2];
    geometry_msgs::msg::Quaternion q;
    const double trace=m00+m11+m22;
    if(trace>0){double s=std::sqrt(trace+1.0)*2;q.w=.25*s;q.x=(m21-m12)/s;q.y=(m02-m20)/s;q.z=(m10-m01)/s;}
    else if(m00>m11&&m00>m22){double s=std::sqrt(1+m00-m11-m22)*2;q.w=(m21-m12)/s;q.x=.25*s;q.y=(m01+m10)/s;q.z=(m02+m20)/s;}
    else if(m11>m22){double s=std::sqrt(1+m11-m00-m22)*2;q.w=(m02-m20)/s;q.x=(m01+m10)/s;q.y=.25*s;q.z=(m12+m21)/s;}
    else{double s=std::sqrt(1+m22-m00-m11)*2;q.w=(m10-m01)/s;q.x=(m02+m20)/s;q.y=(m12+m21)/s;q.z=.25*s;}
    q = normalizedQuaternion(q);
    if (roll_deg != 0.0 || pitch_deg != 0.0 || yaw_deg != 0.0) {
      q = normalizedQuaternion(multiplyQuaternions(
        q, localRpyQuaternion(roll_deg, pitch_deg, yaw_deg)));
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header = snapshot.header;
    pose.pose.position.x = point.x + clearance_m * snapshot.surface_normal.x;
    pose.pose.position.y = point.y + clearance_m * snapshot.surface_normal.y;
    pose.pose.position.z = point.z + clearance_m * snapshot.surface_normal.z;
    pose.pose.orientation = q;
    return pose;
  }

  geometry_msgs::msg::Quaternion normalizedQuaternion(
    geometry_msgs::msg::Quaternion quaternion) const
  {
    const double norm = std::sqrt(
      quaternion.x * quaternion.x + quaternion.y * quaternion.y +
      quaternion.z * quaternion.z + quaternion.w * quaternion.w);
    if (norm <= 0.0) {
      throw std::runtime_error("cannot normalize a zero-length quaternion");
    }
    quaternion.x /= norm;
    quaternion.y /= norm;
    quaternion.z /= norm;
    quaternion.w /= norm;
    return quaternion;
  }

  geometry_msgs::msg::Quaternion multiplyQuaternions(
    const geometry_msgs::msg::Quaternion& left,
    const geometry_msgs::msg::Quaternion& right) const
  {
    geometry_msgs::msg::Quaternion result;
    result.w = left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z;
    result.x = left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y;
    result.y = left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x;
    result.z = left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w;
    return result;
  }

  geometry_msgs::msg::Quaternion localRpyQuaternion(
    double roll_deg, double pitch_deg, double yaw_deg) const
  {
    const double half_roll = roll_deg * std::acos(-1.0) / 360.0;
    const double half_pitch = pitch_deg * std::acos(-1.0) / 360.0;
    const double half_yaw = yaw_deg * std::acos(-1.0) / 360.0;
    const double cr = std::cos(half_roll);
    const double sr = std::sin(half_roll);
    const double cp = std::cos(half_pitch);
    const double sp = std::sin(half_pitch);
    const double cy = std::cos(half_yaw);
    const double sy = std::sin(half_yaw);
    geometry_msgs::msg::Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
    return normalizedQuaternion(q);
  }

  std::map<std::string, double> jointGoal(
    moveit::core::RobotState& state,
    const moveit::core::JointModelGroup* group,
    const geometry_msgs::msg::PoseStamped& pose) const
  {
    if (!state.setFromIK(group, pose.pose, tool_link_, 0.25)) {
      throw std::runtime_error("CHOMP transit IK failed");
    }
    std::vector<double> positions;
    state.copyJointGroupPositions(group, positions);
    std::map<std::string, double> goal;
    const auto& names = group->getVariableNames();
    for (std::size_t index = 0; index < names.size(); ++index) {
      goal.emplace(names[index], positions[index]);
    }
    return goal;
  }

  std::map<std::string, double> jointGoalDegrees(
    const moveit::core::JointModelGroup* group, const std::vector<double>& degrees) const
  {
    const auto& names = group->getVariableNames();
    if (degrees.size() != names.size()) {
      throw std::runtime_error("joint-degree goal does not match planning group variable count");
    }
    std::map<std::string, double> goal;
    for (std::size_t index = 0; index < names.size(); ++index) {
      goal.emplace(names[index], degrees[index] * std::acos(-1.0) / 180.0);
    }
    return goal;
  }
  void publishTargetPoses(
    const Snapshot& snapshot, const std::vector<geometry_msgs::msg::PoseStamped>& cut_targets)
  {
    geometry_msgs::msg::PoseArray targets;
    targets.header = snapshot.header;
    for (const auto& target : cut_targets) {
      targets.poses.push_back(target.pose);
    }
    target_pose_pub_->publish(targets);
  }

  void publishMarkers(const std::vector<geometry_msgs::msg::PoseStamped>& cut_targets)
  {
    visualization_msgs::msg::MarkerArray array;
    for (std::size_t index = 0; index + 1U < cut_targets.size(); index += 2U) {
      visualization_msgs::msg::Marker marker;
      marker.header = cut_targets[index].header;
      marker.ns = "frozen_h_cut";
      marker.id = static_cast<int>(index / 2U);
      marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.points = {cut_targets[index].pose.position, cut_targets[index + 1U].pose.position};
      marker.scale.x = 0.008;
      const bool middle_seam = index == 2U;
      marker.color.r = middle_seam ? 0.0 : 1.0;
      marker.color.g = middle_seam ? 1.0 : 0.4;
      marker.color.b = middle_seam ? 1.0 : 0.0;
      marker.color.a = 1.0;
      array.markers.push_back(marker);
    }
    if (cut_targets.size() % 2U != 0U) {
      const auto& target = cut_targets.back();
      visualization_msgs::msg::Marker marker;
      marker.header = target.header;
      marker.ns = "middle_entry_debug";
      marker.id = 0;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = target.pose;
      marker.scale.x = marker.scale.y = marker.scale.z = 0.030;
      marker.color.r = 1.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      array.markers.push_back(marker);
    }
    marker_pub_->publish(array);
  }
  std::mutex plan_mutex_;
  std::mutex state_mutex_;
  std::optional<sensor_msgs::msg::JointState> current_joint_state_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::Client<CaptureSnapshot>::SharedPtr snapshot_client_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr target_pose_pub_;
  rclcpp::Publisher<moveit_msgs::msg::CollisionObject>::SharedPtr carton_collision_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr section_diagnostic_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr execute_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr live_detection_plan_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr tcp_debug_start_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr tcp_debug_record_service_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr cut_pattern_callback_;
  std::optional<Snapshot> planned_snapshot_;
  std::optional<Snapshot> tcp_debug_snapshot_;
  std::optional<geometry_msgs::msg::PoseStamped> tcp_debug_target_;
  std::size_t tcp_debug_index_{0U};
  std::unique_ptr<mtc::Task> planned_task_;
  std::thread execution_thread_;
  std::string saved_snapshot_dir_, cut_pattern_, side_cut_orientation_mode_,
    combined_first_side_link6_frame_;
  std::string planning_frame_, planning_group_, tool_link_, planning_pipeline_, chomp_pipeline_;
  std::string pilz_pipeline_, motion_profile_, snapshot_service_, carton_collision_id_;
  double carton_height_m_, hover_clearance_m_, cut_clearance_m_, middle_entry_alignment_m_,
    side_cut_tilt_deg_,
    partial_start_fraction_, partial_end_fraction_, approach_cut_length_m_, cartesian_step_m_,
    planning_timeout_s_, combined_search_budget_s_, side_candidate_grid_m_,
    side_candidate_corner_inset_m_, side_candidate_min_length_m_, manual_first_side_max_length_m_,
    velocity_scaling_, acceleration_scaling_, saved_snapshot_max_age_s_;
  std::vector<double> partial_start_joint_degrees_, approach_joint_degrees_;
  std::vector<double> approach_cut_start_xyz_, approach_cut_orientation_xyzw_;
  std::vector<double> side_orientation_offsets_deg_, combined_first_side_link6_xyz_,
    combined_first_side_link6_orientation_xyzw_, manual_first_side_contact_offset_xyz_;
  int max_solutions_;
  bool partial_reverse_, use_approach_pose_as_cut_start_, execution_in_progress_{false};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BoxHCutExecutor>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
