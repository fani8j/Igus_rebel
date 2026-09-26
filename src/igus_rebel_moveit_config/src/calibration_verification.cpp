#include <Eigen/Geometry>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stage.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/trajectory_execution_info.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mtc = moveit::task_constructor;
using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using GoalHandleFollowJointTrajectory = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;

namespace
{
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kEpsilon = 1e-9;
// Board reference points point_1..point_N (parameters point_<n>_xy_m, services ~/plan_point_<n>).
constexpr std::size_t kBoardPointCount = 7U;

std::string boardPointName(std::size_t index) { return "point_" + std::to_string(index + 1U); }

bool isFinite(double value) { return std::isfinite(value); }

bool zeroStamp(const builtin_interfaces::msg::Time& stamp)
{
  return stamp.sec == 0 && stamp.nanosec == 0U;
}

std::string formatDouble(double value, int precision = 6)
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string formatVector3(const Eigen::Vector3d& value)
{
  std::ostringstream stream;
  stream << '[' << formatDouble(value.x()) << ", " << formatDouble(value.y()) << ", "
         << formatDouble(value.z()) << ']';
  return stream.str();
}

std::string jsonEscape(const std::string& input)
{
  std::ostringstream escaped;
  for (const char c : input) {
    switch (c) {
      case '"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\b': escaped << "\\b"; break;
      case '\f': escaped << "\\f"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20U) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<int>(static_cast<unsigned char>(c)) << std::dec;
        } else {
          escaped << c;
        }
    }
  }
  return escaped.str();
}

std::string csvEscape(const std::string& input)
{
  std::string escaped;
  escaped.reserve(input.size() + 2U);
  escaped.push_back('"');
  for (const char c : input) {
    if (c == '"') escaped.push_back('"');
    escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

std::string wallTimestampForFile()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &now_time);
#else
  localtime_r(&now_time, &tm);
#endif
  std::ostringstream stream;
  stream << std::put_time(&tm, "%Y%m%d_%H%M%S");
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
    now.time_since_epoch()) % 1000;
  stream << '_' << std::setw(3) << std::setfill('0') << milliseconds.count();
  return stream.str();
}

std::filesystem::path expandUserPath(const std::string& path)
{
  if (path.empty()) return {};
  if (path[0] != '~') return std::filesystem::path(path);
  const char* home = std::getenv("HOME");
  if (home == nullptr || std::string(home).empty()) {
    throw std::runtime_error("output_directory uses ~ but HOME is not set");
  }
  if (path.size() == 1U) return std::filesystem::path(home);
  if (path[1] == '/') return std::filesystem::path(home) / path.substr(2U);
  throw std::runtime_error("output_directory supports only '~' or '~/' expansion");
}

Eigen::Quaterniond quaternionFromRpy(double roll, double pitch, double yaw)
{
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
}

Eigen::Quaterniond quaternionFromMsg(const geometry_msgs::msg::Quaternion& msg)
{
  Eigen::Quaterniond quaternion(msg.w, msg.x, msg.y, msg.z);
  quaternion.normalize();
  return quaternion;
}

geometry_msgs::msg::Quaternion quaternionMsg(const Eigen::Quaterniond& input)
{
  Eigen::Quaterniond quaternion = input;
  quaternion.normalize();
  geometry_msgs::msg::Quaternion msg;
  msg.x = quaternion.x();
  msg.y = quaternion.y();
  msg.z = quaternion.z();
  msg.w = quaternion.w();
  return msg;
}

Eigen::Isometry3d poseToEigen(const geometry_msgs::msg::Pose& pose)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  transform.linear() = quaternionFromMsg(pose.orientation).toRotationMatrix();
  return transform;
}

geometry_msgs::msg::Pose poseMsg(const Eigen::Isometry3d& transform)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  pose.orientation = quaternionMsg(Eigen::Quaterniond(transform.linear()));
  return pose;
}

Eigen::Isometry3d transformFromXyzRpy(
  const std::vector<double>& xyz, const std::vector<double>& rpy)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation() = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
  transform.linear() = quaternionFromRpy(rpy[0], rpy[1], rpy[2]).toRotationMatrix();
  return transform;
}

bool finiteVector(const std::vector<double>& values)
{
  return std::all_of(values.begin(), values.end(), [](double value) { return isFinite(value); });
}

bool finitePose(const geometry_msgs::msg::Pose& pose)
{
  return isFinite(pose.position.x) && isFinite(pose.position.y) && isFinite(pose.position.z) &&
         isFinite(pose.orientation.x) && isFinite(pose.orientation.y) && isFinite(pose.orientation.z) &&
         isFinite(pose.orientation.w);
}

bool quaternionWellFormed(const geometry_msgs::msg::Quaternion& q)
{
  if (!isFinite(q.x) || !isFinite(q.y) || !isFinite(q.z) || !isFinite(q.w)) return false;
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  return isFinite(norm) && norm > 0.5 && norm < 1.5;
}

double quaternionAngularDistance(
  const geometry_msgs::msg::Quaternion& first, const geometry_msgs::msg::Quaternion& second)
{
  Eigen::Quaterniond a = quaternionFromMsg(first);
  Eigen::Quaterniond b = quaternionFromMsg(second);
  const double dot = std::clamp(std::abs(a.dot(b)), 0.0, 1.0);
  return 2.0 * std::acos(dot);
}

Eigen::Vector3d pointVector(const geometry_msgs::msg::Point& point)
{
  return Eigen::Vector3d(point.x, point.y, point.z);
}

geometry_msgs::msg::Point pointMsg(const Eigen::Vector3d& point)
{
  geometry_msgs::msg::Point msg;
  msg.x = point.x();
  msg.y = point.y();
  msg.z = point.z();
  return msg;
}

std::string poseJson(const geometry_msgs::msg::PoseStamped& pose)
{
  std::ostringstream stream;
  stream << "{\"frame_id\":\"" << jsonEscape(pose.header.frame_id) << "\","
         << "\"position_m\":[" << pose.pose.position.x << ',' << pose.pose.position.y << ','
         << pose.pose.position.z << "],"
         << "\"orientation_xyzw\":[" << pose.pose.orientation.x << ',' << pose.pose.orientation.y
         << ',' << pose.pose.orientation.z << ',' << pose.pose.orientation.w << "]}";
  return stream.str();
}

}  // namespace

class CalibrationVerification : public rclcpp::Node
{
public:
  CalibrationVerification() : Node("calibration_verification")
  {
    planning_group_ = declare_parameter<std::string>("planning_group", "igus_rebel_arm");
    planning_frame_ = declare_parameter<std::string>("planning_frame", "base_link");
    tool_link_ = declare_parameter<std::string>("tool_link", "xeg32_tool_tip");
    planning_pipeline_ = declare_parameter<std::string>("planning_pipeline", "ompl");
    planner_id_ = declare_parameter<std::string>("planner_id", "RRTConnectkConfigDefault");
    camera_pole_required_ = declare_parameter<bool>("camera_pole_required", true);
    camera_pole_topic_ = declare_parameter<std::string>(
      "camera_pole_topic", "/box_motion/camera_pole_collision");
    controller_name_ = declare_parameter<std::string>(
      "controller", "rebel_arm_trajectory_controller");
    controller_action_ = declare_parameter<std::string>(
      "controller_action", "/rebel_arm_trajectory_controller/follow_joint_trajectory");

    const auto pipeline_plugin_parameter = planning_pipeline_ + ".planning_plugin";
    if (!has_parameter(pipeline_plugin_parameter)) {
      declare_parameter<std::string>(pipeline_plugin_parameter, "ompl_interface/OMPLPlanner");
    }
    const auto pipeline_adapters_parameter = planning_pipeline_ + ".request_adapters";
    if (!has_parameter(pipeline_adapters_parameter)) {
      declare_parameter<std::string>(
        pipeline_adapters_parameter,
        "default_planner_request_adapters/ResolveConstraintFrames "
        "default_planner_request_adapters/FixWorkspaceBounds "
        "default_planner_request_adapters/FixStartStateBounds "
        "default_planner_request_adapters/FixStartStateCollision "
        "default_planner_request_adapters/FixStartStatePathConstraints "
        "default_planner_request_adapters/AddTimeOptimalParameterization");
    }
    // Kinematics must come from the package kinematics.yaml (passed by
    // move_group.launch.py); a silent in-code fallback could disagree with it.
    const auto kinematics_solver_parameter =
      "robot_description_kinematics." + planning_group_ + ".kinematics_solver";
    if (!has_parameter(kinematics_solver_parameter) &&
        get_node_parameters_interface()->get_parameter_overrides().count(kinematics_solver_parameter) == 0U) {
      throw std::invalid_argument(
        "missing parameter " + kinematics_solver_parameter +
        "; launch calibration_verification via move_group.launch.py so robot_description_kinematics "
        "is loaded from igus_rebel_moveit_config/config/kinematics.yaml");
    }

    for (std::size_t index = 0U; index < kBoardPointCount; ++index) {
      point_xy_m_[index] = declare_parameter<std::vector<double>>(
        boardPointName(index) + "_xy_m", std::vector<double>{});
    }
    board_target_height_m_ = declare_parameter<double>("board_target_height_m", 0.0);
    preapproach_only_ = declare_parameter<bool>("preapproach_only", false);
    retract_after_approach_ = declare_parameter<bool>("retract_after_approach", true);
    reference_origin_xyz_m_ = declare_parameter<std::vector<double>>(
      "reference_origin_xyz_m", std::vector<double>{0.0, 0.0, 0.0});
    board_to_tcp_orientation_xyzw_ = declare_parameter<std::vector<double>>(
      "board_to_tcp_orientation_xyzw", std::vector<double>{});
    board_to_tcp_rpy_rad_ = declare_parameter<std::vector<double>>(
      "board_to_tcp_rpy_rad", std::vector<double>{});

    board_collision_id_ = declare_parameter<std::string>("board_collision_id", "calibration_board");
    board_box_center_xyz_m_ = declare_parameter<std::vector<double>>(
      "board_box_center_xyz_m", std::vector<double>{});
    board_box_dimensions_xyz_m_ = declare_parameter<std::vector<double>>(
      "board_box_dimensions_xyz_m", std::vector<double>{});
    table_collision_id_ = declare_parameter<std::string>("table_collision_id", "calibration_table");
    table_box_center_xyz_m_ = declare_parameter<std::vector<double>>(
      "table_box_center_xyz_m", std::vector<double>{});
    table_box_dimensions_xyz_m_ = declare_parameter<std::vector<double>>(
      "table_box_dimensions_xyz_m", std::vector<double>{});
    cube_collision_id_ = declare_parameter<std::string>("cube_collision_id", "calibration_cube");
    cube_pose_xyz_m_ = declare_parameter<std::vector<double>>(
      "cube_pose_xyz_m", std::vector<double>{});
    cube_pose_rpy_rad_ = declare_parameter<std::vector<double>>(
      "cube_pose_rpy_rad", std::vector<double>{});
    cube_dimensions_xyz_m_ = declare_parameter<std::vector<double>>(
      "cube_dimensions_xyz_m", std::vector<double>{});
    cube_reference_xyz_m_ = declare_parameter<std::vector<double>>(
      "cube_reference_xyz_m", std::vector<double>{});
    cube_approach_direction_xyz_ = declare_parameter<std::vector<double>>(
      "cube_approach_direction_xyz", std::vector<double>{});
    cube_to_tcp_orientation_xyzw_ = declare_parameter<std::vector<double>>(
      "cube_to_tcp_orientation_xyzw", std::vector<double>{});
    cube_to_tcp_rpy_rad_ = declare_parameter<std::vector<double>>(
      "cube_to_tcp_rpy_rad", std::vector<double>{});

    target_standoff_m_ = declare_parameter<double>("target_standoff_m", 0.020);
    final_standoff_m_ = declare_parameter<double>("final_standoff_m", 0.020);
    preapproach_distance_m_ = declare_parameter<double>("preapproach_distance_m", 0.080);
    cartesian_step_m_ = declare_parameter<double>("cartesian_step_m", 0.003);
    planning_timeout_s_ = declare_parameter<double>("planning_timeout_s", 15.0);
    max_solutions_ = declare_parameter<int>("max_solutions", 1);
    velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.05);
    acceleration_scaling_ = declare_parameter<double>("acceleration_scaling", 0.05);

    max_board_pose_age_s_ = declare_parameter<double>("max_board_pose_age_s", 2.0);
    max_joint_state_age_s_ = declare_parameter<double>("max_joint_state_age_s", 2.0);
    plan_expiry_s_ = declare_parameter<double>("plan_expiry_s", 60.0);
    start_state_tolerance_rad_ = declare_parameter<double>("start_state_tolerance_rad", 0.01);
    board_pose_position_tolerance_m_ = declare_parameter<double>(
      "board_pose_position_tolerance_m", 0.002);
    board_pose_angular_tolerance_rad_ = declare_parameter<double>(
      "board_pose_angular_tolerance_rad", 1.0 * kPi / 180.0);
    future_stamp_tolerance_s_ = declare_parameter<double>("future_stamp_tolerance_s", 0.1);

    execution_allowed_ = declare_parameter<bool>("execution_allowed", false);
    preview_only_ = declare_parameter<bool>("preview_only", false);
    controller_wait_timeout_s_ = declare_parameter<double>("controller_wait_timeout_s", 30.0);
    controller_goal_send_timeout_s_ = declare_parameter<double>("controller_goal_send_timeout_s", 5.0);
    execution_timeout_s_ = declare_parameter<double>("execution_timeout_s", 120.0);
    controller_cancel_timeout_s_ = declare_parameter<double>("controller_cancel_timeout_s", 5.0);
    post_execution_state_settle_s_ = declare_parameter<double>("post_execution_state_settle_s", 0.25);

    joint_states_topic_ = declare_parameter<std::string>("joint_states_topic", "/joint_states");
    planning_scene_service_ = declare_parameter<std::string>(
      "planning_scene_service", "/get_planning_scene");
    planning_scene_topic_ = declare_parameter<std::string>("planning_scene_topic", "/planning_scene");
    planning_scene_world_topic_ = declare_parameter<std::string>(
      "planning_scene_world_topic", "/planning_scene_world");
    collision_object_topic_ = declare_parameter<std::string>("collision_object_topic", "/collision_object");
    planning_scene_state_wait_s_ = declare_parameter<double>("planning_scene_state_wait_s", 2.0);

    output_directory_ = declare_parameter<std::string>("output_directory", "");
    record_format_ = declare_parameter<std::string>("record_format", "json");

    // Launch supplies the same MoveIt configuration as move_group. Plugins
    // enumerate declared parameters, not undeclared parameter overrides.
    for (const auto& [name, value] : get_node_parameters_interface()->get_parameter_overrides()) {
      if (!has_parameter(name)) declare_parameter(name, value);
    }
    validateStartupConfig();

    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group = callback_group_;
    board_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/calibration_verification/board_pose", rclcpp::QoS(10),
      [this](geometry_msgs::msg::PoseStamped::SharedPtr message) {
        std::lock_guard<std::mutex> lock(board_mutex_);
        latest_board_pose_ = LatestBoardPose{*message, now()};
      },
      subscription_options);
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_states_topic_, rclcpp::QoS(50),
      [this](sensor_msgs::msg::JointState::SharedPtr message) {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        latest_joint_state_ = LatestJointState{*message, now()};
      },
      subscription_options);
    camera_pole_sub_ = create_subscription<moveit_msgs::msg::CollisionObject>(
      camera_pole_topic_, rclcpp::QoS(1).reliable().transient_local(),
      [this](moveit_msgs::msg::CollisionObject::SharedPtr message) {
        if (message->id != "camera_pole" || message->header.frame_id != planning_frame_ ||
            message->operation != moveit_msgs::msg::CollisionObject::ADD ||
            message->primitives.size() != 1U || message->primitive_poses.size() != 1U ||
            message->primitives.front().type != shape_msgs::msg::SolidPrimitive::BOX ||
            message->primitives.front().dimensions.size() != 3U) {
          RCLCPP_ERROR(get_logger(), "Rejected invalid camera pole collision object");
          return;
        }
        std::lock_guard<std::mutex> lock(camera_pole_mutex_);
        camera_pole_ = *message;
      },
      subscription_options);

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/calibration_verification/markers", rclcpp::QoS(1).reliable().transient_local());

    controller_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
      get_node_base_interface(), get_node_graph_interface(), get_node_logging_interface(),
      get_node_waitables_interface(), controller_action_, callback_group_);

    capture_service_ = create_service<std_srvs::srv::Trigger>(
      "~/capture",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
          response->success = false;
          response->message = "another calibration verification operation is active";
          return;
        }
        response->success = captureLocked(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);

    plan_center_service_ = createPlanService("~/plan_center", TargetKind::CENTER);
    for (std::size_t index = 0U; index < kBoardPointCount; ++index) {
      plan_point_services_[index] =
        createPlanService("~/plan_" + boardPointName(index), TargetKind::POINT, index);
    }
    plan_cube_service_ = createPlanService("~/plan_cube", TargetKind::CUBE);

    execute_service_ = create_service<std_srvs::srv::Trigger>(
      "~/execute_plan",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        response->success = executePlanService(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);

    clear_service_ = create_service<std_srvs::srv::Trigger>(
      "~/clear_plan",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
          response->success = false;
          response->message = "another calibration verification operation is active";
          return;
        }
        if (execution_in_progress_.load()) {
          response->success = false;
          response->message = "cannot clear a retained plan while execution is active";
          return;
        }
        retained_plan_.reset();
        response->success = true;
        response->message = "retained calibration verification plan cleared";
      },
      rmw_qos_profile_services_default, callback_group_);

    stop_service_ = create_service<std_srvs::srv::Trigger>(
      "~/stop",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        response->success = stopExecution(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);

    RCLCPP_INFO(
      get_logger(),
      "calibration_verification ready: capture /calibration_verification/board_pose, then plan; execution_allowed=%s",
      execution_allowed_ ? "true" : "false");
  }

  ~CalibrationVerification() override
  {
    std::lock_guard<std::mutex> lock(execution_mutex_);
    stop_requested_.store(true);
    if (active_goal_handle_) {
      controller_client_->async_cancel_goal(active_goal_handle_);
    }
  }

private:
  enum class TargetKind
  {
    CENTER,
    POINT,  // board reference point; the index selects point_<index + 1>
    CUBE
  };

  struct LatestBoardPose
  {
    geometry_msgs::msg::PoseStamped pose;
    rclcpp::Time received_at;
  };

  struct CapturedBoardPose
  {
    geometry_msgs::msg::PoseStamped pose;
    rclcpp::Time captured_at;
    std::uint64_t sequence{0U};
  };

  struct LatestJointState
  {
    sensor_msgs::msg::JointState state;
    rclcpp::Time received_at;
  };

  struct TargetSpec
  {
    std::string name;
    geometry_msgs::msg::PoseStamped target_pose;
    geometry_msgs::msg::PoseStamped final_pose;
    geometry_msgs::msg::PoseStamped preapproach_pose;
    Eigen::Vector3d reference_point_base{Eigen::Vector3d::Zero()};
    Eigen::Vector3d approach_direction_base{Eigen::Vector3d::UnitZ()};
  };

  struct RetainedPlan
  {
    moveit_task_constructor_msgs::msg::Solution solution;
    moveit::core::RobotModelConstPtr robot_model;
    CapturedBoardPose board;
    TargetSpec target;
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    std::map<std::string, double> start_joint_positions;
    rclcpp::Time planned_at;
    std::uint64_t plan_sequence{0U};
    std::size_t trajectory_count{0U};
    std::size_t point_count{0U};
  };

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr createPlanService(
    const std::string& name, TargetKind target_kind, std::size_t point_index = 0U)
  {
    return create_service<std_srvs::srv::Trigger>(
      name,
      [this, target_kind, point_index](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
          response->success = false;
          response->message = "another calibration verification operation is active";
          return;
        }
        if (execution_in_progress_.load()) {
          response->success = false;
          response->message = "cannot plan while calibration verification execution is active";
          return;
        }
        response->success = planTargetLocked(target_kind, point_index, response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
  }

  void validateStartupConfig() const
  {
    const auto require_nonempty_string = [](const std::string& value, const char* name) {
      if (value.empty()) throw std::invalid_argument(std::string(name) + " must not be empty");
    };
    require_nonempty_string(planning_group_, "planning_group");
    require_nonempty_string(planning_frame_, "planning_frame");
    require_nonempty_string(tool_link_, "tool_link");
    require_nonempty_string(planning_pipeline_, "planning_pipeline");
    if (camera_pole_required_ && camera_pole_topic_.empty()) {
      throw std::invalid_argument("camera_pole_topic must not be empty when camera_pole_required is true");
    }
    require_nonempty_string(controller_name_, "controller");
    require_nonempty_string(controller_action_, "controller_action");
    require_nonempty_string(board_collision_id_, "board_collision_id");
    require_nonempty_string(table_collision_id_, "table_collision_id");
    require_nonempty_string(cube_collision_id_, "cube_collision_id");
    require_nonempty_string(joint_states_topic_, "joint_states_topic");
    require_nonempty_string(planning_scene_service_, "planning_scene_service");
    require_nonempty_string(planning_scene_topic_, "planning_scene_topic");
    require_nonempty_string(planning_scene_world_topic_, "planning_scene_world_topic");
    require_nonempty_string(collision_object_topic_, "collision_object_topic");

    if (!configuredVector3(reference_origin_xyz_m_)) {
      throw std::invalid_argument("reference_origin_xyz_m must contain three finite metre values");
    }
    validateOptionalPosition(reference_origin_xyz_m_, "reference_origin_xyz_m");
    for (std::size_t index = 0U; index < kBoardPointCount; ++index) {
      const std::string name = boardPointName(index) + "_xy_m";
      validateOptionalVector(point_xy_m_[index], 2U, name.c_str());
      validateOptionalPosition(point_xy_m_[index], name.c_str());
    }
    validateOptionalVector(board_to_tcp_orientation_xyzw_, 4U, "board_to_tcp_orientation_xyzw");
    validateOptionalVector(board_to_tcp_rpy_rad_, 3U, "board_to_tcp_rpy_rad");
    validateOptionalVector(board_box_center_xyz_m_, 3U, "board_box_center_xyz_m");
    validateOptionalVector(board_box_dimensions_xyz_m_, 3U, "board_box_dimensions_xyz_m");
    validateOptionalVector(table_box_center_xyz_m_, 3U, "table_box_center_xyz_m");
    validateOptionalVector(table_box_dimensions_xyz_m_, 3U, "table_box_dimensions_xyz_m");
    validateOptionalVector(cube_pose_xyz_m_, 3U, "cube_pose_xyz_m");
    validateOptionalVector(cube_pose_rpy_rad_, 3U, "cube_pose_rpy_rad");
    validateOptionalVector(cube_dimensions_xyz_m_, 3U, "cube_dimensions_xyz_m");
    validateOptionalVector(cube_reference_xyz_m_, 3U, "cube_reference_xyz_m");
    validateOptionalVector(cube_approach_direction_xyz_, 3U, "cube_approach_direction_xyz");
    validateOptionalVector(cube_to_tcp_orientation_xyzw_, 4U, "cube_to_tcp_orientation_xyzw");
    validateOptionalVector(cube_to_tcp_rpy_rad_, 3U, "cube_to_tcp_rpy_rad");

    validateOptionalQuaternion(board_to_tcp_orientation_xyzw_, "board_to_tcp_orientation_xyzw");
    validateOptionalQuaternion(cube_to_tcp_orientation_xyzw_, "cube_to_tcp_orientation_xyzw");
    if (!board_to_tcp_orientation_xyzw_.empty() && !board_to_tcp_rpy_rad_.empty()) {
      throw std::invalid_argument(
        "configure only one of board_to_tcp_orientation_xyzw or board_to_tcp_rpy_rad");
    }
    if (!cube_to_tcp_orientation_xyzw_.empty() && !cube_to_tcp_rpy_rad_.empty()) {
      throw std::invalid_argument(
        "configure only one of cube_to_tcp_orientation_xyzw or cube_to_tcp_rpy_rad");
    }

    validateOptionalDimensions(board_box_dimensions_xyz_m_, "board_box_dimensions_xyz_m");
    validateOptionalDimensions(table_box_dimensions_xyz_m_, "table_box_dimensions_xyz_m");
    validateOptionalDimensions(cube_dimensions_xyz_m_, "cube_dimensions_xyz_m");
    validateOptionalPosition(board_box_center_xyz_m_, "board_box_center_xyz_m");
    validateOptionalPosition(table_box_center_xyz_m_, "table_box_center_xyz_m");
    validateOptionalPosition(cube_pose_xyz_m_, "cube_pose_xyz_m");
    validateOptionalPosition(cube_reference_xyz_m_, "cube_reference_xyz_m");

    if (!cube_approach_direction_xyz_.empty()) {
      const Eigen::Vector3d direction(
        cube_approach_direction_xyz_[0], cube_approach_direction_xyz_[1],
        cube_approach_direction_xyz_[2]);
      if (direction.norm() < kEpsilon) {
        throw std::invalid_argument("cube_approach_direction_xyz must not be zero-length");
      }
    }

    if (!isFinite(board_target_height_m_) || std::abs(board_target_height_m_) > 2.0) {
      throw std::invalid_argument("board_target_height_m must be finite and within +/-2 m");
    }
    if (!isFinite(target_standoff_m_) || target_standoff_m_ <= 0.0 || target_standoff_m_ > 0.5) {
      throw std::invalid_argument("target_standoff_m must be finite and in (0, 0.5] m");
    }
    if (!isFinite(final_standoff_m_) || final_standoff_m_ <= 0.0 ||
        final_standoff_m_ > target_standoff_m_) {
      throw std::invalid_argument("final_standoff_m must be finite and in (0, target_standoff_m] m");
    }
    if (!isFinite(preapproach_distance_m_) || preapproach_distance_m_ <= 0.0 ||
        preapproach_distance_m_ > 1.0) {
      throw std::invalid_argument("preapproach_distance_m must be finite and in (0, 1] m");
    }
    if (!isFinite(cartesian_step_m_) || cartesian_step_m_ <= 0.0 || cartesian_step_m_ > 0.05) {
      throw std::invalid_argument("cartesian_step_m must be finite and in (0, 0.05] m");
    }
    if (!isFinite(planning_timeout_s_) || planning_timeout_s_ <= 0.0 || planning_timeout_s_ > 300.0) {
      throw std::invalid_argument("planning_timeout_s must be finite and in (0, 300] s");
    }
    if (max_solutions_ < 1 || max_solutions_ > 100) {
      throw std::invalid_argument("max_solutions must be in [1, 100]");
    }
    if (!isFinite(velocity_scaling_) || velocity_scaling_ <= 0.0 || velocity_scaling_ > 1.0 ||
        !isFinite(acceleration_scaling_) || acceleration_scaling_ <= 0.0 || acceleration_scaling_ > 1.0) {
      throw std::invalid_argument("velocity_scaling and acceleration_scaling must be finite and in (0, 1]");
    }
    if (!isFinite(max_board_pose_age_s_) || max_board_pose_age_s_ <= 0.0 || max_board_pose_age_s_ > 60.0 ||
        !isFinite(max_joint_state_age_s_) || max_joint_state_age_s_ <= 0.0 || max_joint_state_age_s_ > 60.0 ||
        !isFinite(plan_expiry_s_) || plan_expiry_s_ <= 0.0 || plan_expiry_s_ > 3600.0) {
      throw std::invalid_argument("freshness and plan expiry durations must be finite positive bounded seconds");
    }
    if (!isFinite(start_state_tolerance_rad_) || start_state_tolerance_rad_ < 0.0 ||
        start_state_tolerance_rad_ > 0.5) {
      throw std::invalid_argument("start_state_tolerance_rad must be finite and in [0, 0.5] rad");
    }
    if (!isFinite(board_pose_position_tolerance_m_) || board_pose_position_tolerance_m_ < 0.0 ||
        board_pose_position_tolerance_m_ > 0.1 || !isFinite(board_pose_angular_tolerance_rad_) ||
        board_pose_angular_tolerance_rad_ < 0.0 || board_pose_angular_tolerance_rad_ > kPi) {
      throw std::invalid_argument("board pose execution tolerances are invalid");
    }
    if (!isFinite(future_stamp_tolerance_s_) || future_stamp_tolerance_s_ < 0.0 ||
        future_stamp_tolerance_s_ > 10.0) {
      throw std::invalid_argument("future_stamp_tolerance_s must be finite and in [0, 10] s");
    }
    if (!isFinite(controller_wait_timeout_s_) || controller_wait_timeout_s_ <= 0.0 ||
        !isFinite(controller_goal_send_timeout_s_) || controller_goal_send_timeout_s_ <= 0.0 ||
        !isFinite(execution_timeout_s_) || execution_timeout_s_ <= 0.0 ||
        !isFinite(controller_cancel_timeout_s_) || controller_cancel_timeout_s_ <= 0.0 ||
        !isFinite(post_execution_state_settle_s_) || post_execution_state_settle_s_ < 0.0 ||
        !isFinite(planning_scene_state_wait_s_) || planning_scene_state_wait_s_ <= 0.0) {
      throw std::invalid_argument("execution and planning-scene timeouts must be finite positive seconds");
    }
    if (record_format_ != "json" && record_format_ != "csv") {
      throw std::invalid_argument("record_format must be 'json' or 'csv'");
    }
  }

  static void validateOptionalVector(
    const std::vector<double>& values, std::size_t expected_size, const char* name)
  {
    if (values.empty()) return;
    if (values.size() != expected_size) {
      throw std::invalid_argument(
        std::string(name) + " must be empty or contain exactly " + std::to_string(expected_size) +
        " values");
    }
    if (!finiteVector(values)) {
      throw std::invalid_argument(std::string(name) + " contains a non-finite value");
    }
  }

  static void validateOptionalQuaternion(const std::vector<double>& values, const char* name)
  {
    if (values.empty()) return;
    const double norm = std::sqrt(
      values[0] * values[0] + values[1] * values[1] + values[2] * values[2] +
      values[3] * values[3]);
    if (!isFinite(norm) || norm < 0.5 || norm > 1.5) {
      throw std::invalid_argument(std::string(name) + " must be a non-zero XYZW quaternion");
    }
  }

  static void validateOptionalDimensions(const std::vector<double>& values, const char* name)
  {
    if (values.empty()) return;
    for (const double value : values) {
      if (value <= 0.0 || value > 5.0) {
        throw std::invalid_argument(std::string(name) + " dimensions must be in (0, 5] m");
      }
    }
  }

  static void validateOptionalPosition(const std::vector<double>& values, const char* name)
  {
    if (values.empty()) return;
    for (const double value : values) {
      if (std::abs(value) > 5.0) {
        throw std::invalid_argument(std::string(name) + " coordinates must be within +/-5 m");
      }
    }
  }

  bool captureLocked(std::string& message)
  {
    if (execution_in_progress_.load()) {
      message = "cannot capture a new board snapshot while execution is active";
      return false;
    }

    LatestBoardPose latest;
    {
      std::lock_guard<std::mutex> lock(board_mutex_);
      if (!latest_board_pose_) {
        message = "no /calibration_verification/board_pose has been received";
        return false;
      }
      latest = *latest_board_pose_;
    }

    if (!validateFreshBoardPose(latest, message)) {
      message = "board pose capture rejected: " + message;
      return false;
    }

    captured_board_pose_ = CapturedBoardPose{latest.pose, now(), ++capture_sequence_};
    retained_plan_.reset();
    publishMarkers(*captured_board_pose_, std::nullopt, {});
    message = "captured fresh board pose snapshot " + std::to_string(captured_board_pose_->sequence) +
              "; any retained plan was invalidated";
    RCLCPP_INFO(get_logger(), "%s", message.c_str());
    return true;
  }

  bool planTargetLocked(TargetKind target_kind, std::size_t point_index, std::string& message)
  {
    retained_plan_.reset();
    if (!captured_board_pose_) {
      message = "no captured board pose; call ~/capture with a fresh board pose first";
      return false;
    }

    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    if (!buildCollisionObjects(*captured_board_pose_, collision_objects, message)) {
      message = "planning blocked: " + message;
      publishMarkers(*captured_board_pose_, std::nullopt, {});
      return false;
    }

    TargetSpec target;
    if (!buildTargetSpec(target_kind, point_index, *captured_board_pose_, target, message)) {
      message = "planning blocked: " + message;
      publishMarkers(*captured_board_pose_, std::nullopt, collision_objects);
      return false;
    }
    if (preapproach_only_) {
      target.name += "_preapproach";
      target.target_pose = target.preapproach_pose;
    }

    publishMarkers(*captured_board_pose_, target, collision_objects);

    try {
      auto task = createTask(target, collision_objects);
      task->init();
      const auto result = task->plan(static_cast<std::size_t>(max_solutions_));
      if (result != moveit::core::MoveItErrorCode::SUCCESS || task->solutions().empty()) {
        task->explainFailure();
        message = "MTC planning failed for " + target.name +
                  "; no motion was executed and no retained plan is available";
        RCLCPP_ERROR(get_logger(), "%s", message.c_str());
        return false;
      }

      moveit_task_constructor_msgs::msg::Solution solution_message;
      task->solutions().front()->toMsg(solution_message);
      std::map<std::string, double> start_joint_positions;
      std::size_t trajectory_count = 0U;
      std::size_t point_count = 0U;
      if (!extractSolutionStart(
            solution_message, start_joint_positions, trajectory_count, point_count, message)) {
        message = "MTC planning produced an unusable solution for " + target.name + ": " + message;
        return false;
      }

      task->introspection().publishSolution(*task->solutions().front());

      auto retained = std::make_unique<RetainedPlan>();
      retained->solution = std::move(solution_message);
      retained->robot_model = task->getRobotModel();
      retained->board = *captured_board_pose_;
      retained->target = target;
      retained->collision_objects = collision_objects;
      retained->start_joint_positions = std::move(start_joint_positions);
      retained->planned_at = now();
      retained->plan_sequence = ++plan_sequence_;
      retained->trajectory_count = trajectory_count;
      retained->point_count = point_count;
      retained_plan_ = std::move(retained);

      message = "planned and retained calibration verification target " + target.name +
                " with " + std::to_string(trajectory_count) + " joint trajectories / " +
                std::to_string(point_count) +
                " waypoints; planning service did not execute motion; inspect in RViz before ~/execute_plan";
      RCLCPP_INFO(get_logger(), "%s", message.c_str());
      return true;
    } catch (const mtc::InitStageException& exception) {
      std::ostringstream details;
      details << exception;
      message = "MTC initialization failed for " + target.name + ": " + details.str();
    } catch (const std::exception& exception) {
      message = "MTC planning failed for " + target.name + ": " + exception.what();
    }
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    return false;
  }

  std::unique_ptr<mtc::Task> createTask(
    const TargetSpec& target, const std::vector<moveit_msgs::msg::CollisionObject>& collision_objects)
  {
    auto task = std::make_unique<mtc::Task>();
    task->stages()->setName("Calibration verification " + target.name);
    {
      // Loading the model parses URDF/SRDF and kinematics plugins; do it once.
      std::lock_guard<std::mutex> lock(robot_model_mutex_);
      if (!robot_model_) {
        task->loadRobotModel(shared_from_this());
        robot_model_ = task->getRobotModel();
      } else {
        task->setRobotModel(robot_model_);
      }
    }
    if (!task->getRobotModel()) {
      throw std::runtime_error("failed to load robot model from robot_description");
    }
    if (task->getRobotModel()->getJointModelGroup(planning_group_) == nullptr) {
      throw std::runtime_error("planning_group " + planning_group_ + " is missing from robot model");
    }
    if (task->getRobotModel()->getLinkModel(tool_link_) == nullptr) {
      throw std::runtime_error("tool_link " + tool_link_ + " is missing from robot model");
    }

    task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));

    auto add_scene = std::make_unique<mtc::stages::ModifyPlanningScene>(
      "add calibration verification collision objects");
    add_scene->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    for (const auto& object : collision_objects) {
      add_scene->addObject(object);
    }
    {
      std::optional<moveit_msgs::msg::CollisionObject> camera_pole;
      {
        std::lock_guard<std::mutex> lock(camera_pole_mutex_);
        camera_pole = camera_pole_;
      }
      if (camera_pole_required_ && !camera_pole) {
        throw std::runtime_error(
          "camera pole collision object is unavailable; wait for camera_pole_collision.py");
      }
      if (camera_pole) {
        add_scene->addObject(*camera_pole);
      }
    }
    task->add(std::move(add_scene));

    auto ptp_planner = std::make_shared<mtc::solvers::PipelinePlanner>(
      shared_from_this(), planning_pipeline_);
    if (!planner_id_.empty()) ptp_planner->setPlannerId(planner_id_);
    ptp_planner->setTimeout(planning_timeout_s_);
    ptp_planner->setMaxVelocityScalingFactor(velocity_scaling_);
    ptp_planner->setMaxAccelerationScalingFactor(acceleration_scaling_);

    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setIKFrame(tool_link_);
    cartesian_planner->setStepSize(cartesian_step_m_);
    cartesian_planner->setJumpThreshold(0.0);
    cartesian_planner->setMinFraction(1.0);
    cartesian_planner->setMaxVelocityScalingFactor(velocity_scaling_);
    cartesian_planner->setMaxAccelerationScalingFactor(acceleration_scaling_);

    mtc::TrajectoryExecutionInfo execution_info;
    execution_info.controller_names = {controller_name_};

    auto transit = std::make_unique<mtc::stages::MoveTo>(
      "PTP collision-checked transit to " + target.name + " preapproach", ptp_planner);
    transit->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    transit->setGroup(planning_group_);
    transit->setIKFrame(tool_link_);
    transit->setGoal(target.preapproach_pose);
    transit->setProperty("trajectory_execution_info", execution_info);
    task->add(std::move(transit));
    if (preapproach_only_) return task;

    auto approach = std::make_unique<mtc::stages::MoveTo>(
      "Cartesian approach to positive stand-off for " + target.name, cartesian_planner);
    approach->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    approach->setGroup(planning_group_);
    approach->setIKFrame(tool_link_);
    approach->setGoal(target.target_pose);
    approach->setProperty("trajectory_execution_info", execution_info);
    task->add(std::move(approach));

    auto final_descent = std::make_unique<mtc::stages::MoveTo>(
      "Cartesian final descent to positive clearance for " + target.name, cartesian_planner);
    final_descent->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    final_descent->setGroup(planning_group_);
    final_descent->setIKFrame(tool_link_);
    final_descent->setGoal(target.final_pose);
    final_descent->setProperty("trajectory_execution_info", execution_info);
    task->add(std::move(final_descent));

    if (!retract_after_approach_) return task;

    auto retract = std::make_unique<mtc::stages::MoveTo>(
      "Cartesian retract from " + target.name + " stand-off", cartesian_planner);
    retract->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    retract->setGroup(planning_group_);
    retract->setIKFrame(tool_link_);
    retract->setGoal(target.preapproach_pose);
    retract->setProperty("trajectory_execution_info", execution_info);
    task->add(std::move(retract));

    return task;
  }

  bool validateFreshBoardPose(const LatestBoardPose& latest, std::string& message) const
  {
    if (latest.pose.header.frame_id != planning_frame_) {
      message = "expected frame " + planning_frame_ + ", got " + latest.pose.header.frame_id;
      return false;
    }
    if (!finitePose(latest.pose.pose) || !quaternionWellFormed(latest.pose.pose.orientation)) {
      message = "pose contains non-finite coordinates or an invalid quaternion";
      return false;
    }

    const rclcpp::Time now_time = now();
    const double receive_age = (now_time - latest.received_at).seconds();
    if (!isFinite(receive_age) || receive_age < -future_stamp_tolerance_s_ ||
        receive_age > max_board_pose_age_s_) {
      message = "received board pose is not fresh; receive_age_s=" + formatDouble(receive_age);
      return false;
    }
    if (!zeroStamp(latest.pose.header.stamp)) {
      const double stamp_age = (now_time - rclcpp::Time(latest.pose.header.stamp)).seconds();
      if (!isFinite(stamp_age) || stamp_age < -future_stamp_tolerance_s_ ||
          stamp_age > max_board_pose_age_s_) {
        message = "stamped board pose is not fresh; stamp_age_s=" + formatDouble(stamp_age);
        return false;
      }
    }
    return true;
  }

  std::optional<LatestJointState> freshJointState(std::string& message) const
  {
    LatestJointState latest;
    {
      std::lock_guard<std::mutex> lock(joint_state_mutex_);
      if (!latest_joint_state_) {
        message = "no " + joint_states_topic_ + " message has been received";
        return std::nullopt;
      }
      latest = *latest_joint_state_;
    }
    if (latest.state.name.empty() || latest.state.position.empty() ||
        latest.state.name.size() != latest.state.position.size()) {
      message = joint_states_topic_ + " has empty or mismatched name/position arrays";
      return std::nullopt;
    }
    for (const double position : latest.state.position) {
      if (!isFinite(position)) {
        message = joint_states_topic_ + " contains a non-finite joint position";
        return std::nullopt;
      }
    }
    const rclcpp::Time now_time = now();
    const double receive_age = (now_time - latest.received_at).seconds();
    if (!isFinite(receive_age) || receive_age < -future_stamp_tolerance_s_ ||
        receive_age > max_joint_state_age_s_) {
      message = joint_states_topic_ + " is stale; receive_age_s=" + formatDouble(receive_age);
      return std::nullopt;
    }
    if (!zeroStamp(latest.state.header.stamp)) {
      const double stamp_age = (now_time - rclcpp::Time(latest.state.header.stamp)).seconds();
      if (!isFinite(stamp_age) || stamp_age < -future_stamp_tolerance_s_ ||
          stamp_age > max_joint_state_age_s_) {
        message = joint_states_topic_ + " stamp is stale; stamp_age_s=" + formatDouble(stamp_age);
        return std::nullopt;
      }
    }
    return latest;
  }

  bool buildCollisionObjects(
    const CapturedBoardPose& board, std::vector<moveit_msgs::msg::CollisionObject>& objects,
    std::string& message) const
  {
    if (preview_only_) {
      objects.clear();
      return true;
    }
    if (!configuredVector3(board_box_center_xyz_m_) || !configuredVector3(board_box_dimensions_xyz_m_)) {
      message = "board collision geometry unset: board_box_center_xyz_m and "
                "board_box_dimensions_xyz_m must each contain three finite metre values; dimensions include board thickness";
      return false;
    }
    if (!configuredVector3(table_box_center_xyz_m_) || !configuredVector3(table_box_dimensions_xyz_m_)) {
      message = "table collision geometry unset: table_box_center_xyz_m and "
                "table_box_dimensions_xyz_m must each contain three finite metre values; dimensions include table thickness";
      return false;
    }
    if (!configuredVector3(cube_pose_xyz_m_) || !configuredVector3(cube_pose_rpy_rad_) ||
        !configuredVector3(cube_dimensions_xyz_m_)) {
      message = "cube collision geometry unset: cube_pose_xyz_m, cube_pose_rpy_rad, and "
                "cube_dimensions_xyz_m must each contain three finite metre/radian values";
      return false;
    }
    std::string dimension_error;
    if (!positiveDimensions(board_box_dimensions_xyz_m_, "board_box_dimensions_xyz_m", dimension_error) ||
        !positiveDimensions(table_box_dimensions_xyz_m_, "table_box_dimensions_xyz_m", dimension_error) ||
        !positiveDimensions(cube_dimensions_xyz_m_, "cube_dimensions_xyz_m", dimension_error)) {
      message = dimension_error;
      return false;
    }

    const Eigen::Isometry3d base_board = poseToEigen(board.pose.pose);
    const Eigen::Isometry3d base_cube = base_board * transformFromXyzRpy(cube_pose_xyz_m_, cube_pose_rpy_rad_);

    objects.clear();
    objects.reserve(3U);
    objects.push_back(makeBoxCollisionObject(
      board_collision_id_, base_board, board_box_center_xyz_m_, board_box_dimensions_xyz_m_));
    objects.push_back(makeBoxCollisionObject(
      table_collision_id_, Eigen::Isometry3d::Identity(), table_box_center_xyz_m_, table_box_dimensions_xyz_m_));
    objects.push_back(makeBoxCollisionObject(
      cube_collision_id_, base_cube, {0.0, 0.0, 0.0}, cube_dimensions_xyz_m_));
    return true;
  }

  bool buildTargetSpec(
    TargetKind target_kind, std::size_t point_index, const CapturedBoardPose& board, TargetSpec& target,
    std::string& message) const
  {
    if (preview_only_ && target_kind == TargetKind::CUBE) {
      message = "cube target unavailable in unmeasured-fixture preview mode";
      return false;
    }
    switch (target_kind) {
      case TargetKind::CENTER:
        return buildBoardTarget("center", {0.0, 0.0}, board, target, message);
      case TargetKind::POINT:
        if (point_index >= kBoardPointCount) {
          message = "board point index out of range";
          return false;
        }
        return buildOptionalBoardTarget(
          boardPointName(point_index), point_xy_m_[point_index], board, target, message);
      case TargetKind::CUBE:
        return buildCubeTarget(board, target, message);
    }
    message = "unknown target kind";
    return false;
  }

  bool buildOptionalBoardTarget(
    const std::string& name, const std::vector<double>& xy, const CapturedBoardPose& board,
    TargetSpec& target, std::string& message) const
  {
    if (xy.empty()) {
      message = name + " is disabled because " + name + "_xy_m is empty";
      return false;
    }
    return buildBoardTarget(name, {xy[0], xy[1]}, board, target, message);
  }

  bool buildBoardTarget(
    const std::string& name, const std::array<double, 2>& xy, const CapturedBoardPose& board,
    TargetSpec& target, std::string& message) const
  {
    if (!isFinite(xy[0]) || !isFinite(xy[1]) || std::abs(xy[0]) > 5.0 || std::abs(xy[1]) > 5.0) {
      message = name + " board XY target is non-finite or outside +/-5 m";
      return false;
    }

    Eigen::Quaterniond board_to_tcp;
    if (!configuredOrientation(
          "board_to_tcp", board_to_tcp_orientation_xyzw_, board_to_tcp_rpy_rad_, board_to_tcp,
          message)) {
      return false;
    }

    const Eigen::Isometry3d base_board = poseToEigen(board.pose.pose);
    const Eigen::Vector3d approach_board = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d reference_board = referencePoint(xy[0], xy[1]);
    const Eigen::Vector3d final_board = reference_board + final_standoff_m_ * approach_board;
    const Eigen::Vector3d target_board = reference_board + target_standoff_m_ * approach_board;
    const Eigen::Vector3d preapproach_board = target_board + preapproach_distance_m_ * approach_board;

    target.name = name;
    target.reference_point_base = base_board * reference_board;
    target.approach_direction_base = base_board.linear() * approach_board;
    target.target_pose = poseStampedFromLocal(board.pose.header, base_board, target_board, board_to_tcp);
    target.final_pose = poseStampedFromLocal(board.pose.header, base_board, final_board, board_to_tcp);
    target.preapproach_pose = poseStampedFromLocal(board.pose.header, base_board, preapproach_board, board_to_tcp);
    return true;
  }

  bool buildCubeTarget(
    const CapturedBoardPose& board, TargetSpec& target, std::string& message) const
  {
    if (!configuredVector3(cube_pose_xyz_m_) || !configuredVector3(cube_pose_rpy_rad_)) {
      message = "cube pose unset: cube_pose_xyz_m and cube_pose_rpy_rad must be configured before ~/plan_cube";
      return false;
    }
    if (!configuredVector3(cube_reference_xyz_m_)) {
      message = "cube reference unset: cube_reference_xyz_m must contain the selected corner/face reference point in the cube frame";
      return false;
    }
    if (!configuredVector3(cube_approach_direction_xyz_)) {
      message = "cube approach direction unset: cube_approach_direction_xyz must point from the selected cube feature toward the safe side";
      return false;
    }

    Eigen::Quaterniond cube_to_tcp;
    if (!configuredOrientation(
          "cube_to_tcp", cube_to_tcp_orientation_xyzw_, cube_to_tcp_rpy_rad_, cube_to_tcp,
          message)) {
      return false;
    }

    Eigen::Vector3d approach_cube(
      cube_approach_direction_xyz_[0], cube_approach_direction_xyz_[1], cube_approach_direction_xyz_[2]);
    const double approach_norm = approach_cube.norm();
    if (!isFinite(approach_norm) || approach_norm < kEpsilon) {
      message = "cube_approach_direction_xyz must be finite and non-zero";
      return false;
    }
    approach_cube /= approach_norm;

    const Eigen::Isometry3d base_board = poseToEigen(board.pose.pose);
    const Eigen::Isometry3d base_cube = base_board * transformFromXyzRpy(cube_pose_xyz_m_, cube_pose_rpy_rad_);
    const Eigen::Vector3d reference_cube(
      cube_reference_xyz_m_[0], cube_reference_xyz_m_[1], cube_reference_xyz_m_[2]);
    const Eigen::Vector3d final_cube = reference_cube + final_standoff_m_ * approach_cube;
    const Eigen::Vector3d target_cube = reference_cube + target_standoff_m_ * approach_cube;
    const Eigen::Vector3d preapproach_cube = target_cube + preapproach_distance_m_ * approach_cube;

    target.name = "cube";
    target.reference_point_base = base_cube * reference_cube;
    target.approach_direction_base = base_cube.linear() * approach_cube;
    target.target_pose = poseStampedFromLocal(board.pose.header, base_cube, target_cube, cube_to_tcp);
    target.final_pose = poseStampedFromLocal(board.pose.header, base_cube, final_cube, cube_to_tcp);
    target.preapproach_pose = poseStampedFromLocal(board.pose.header, base_cube, preapproach_cube, cube_to_tcp);
    return true;
  }

  geometry_msgs::msg::PoseStamped poseStampedFromLocal(
    const std_msgs::msg::Header& header, const Eigen::Isometry3d& base_reference,
    const Eigen::Vector3d& local_position, const Eigen::Quaterniond& local_to_tcp) const
  {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = base_reference * local_position;
    pose.linear() = base_reference.linear() * local_to_tcp.toRotationMatrix();

    geometry_msgs::msg::PoseStamped stamped;
    stamped.header = header;
    stamped.header.frame_id = planning_frame_;
    stamped.pose = poseMsg(pose);
    return stamped;
  }

  bool configuredOrientation(
    const std::string& prefix, const std::vector<double>& xyzw, const std::vector<double>& rpy,
    Eigen::Quaterniond& quaternion, std::string& message) const
  {
    if (xyzw.empty() && rpy.empty()) {
      if (preview_only_) {
        quaternion = Eigen::Quaterniond::Identity();
        return true;
      }
      message = prefix + " orientation unset: configure " + prefix +
                "_orientation_xyzw or " + prefix + "_rpy_rad explicitly";
      return false;
    }
    if (!xyzw.empty() && !rpy.empty()) {
      message = prefix + " orientation is ambiguous: configure only quaternion or RPY";
      return false;
    }
    if (!xyzw.empty()) {
      quaternion = Eigen::Quaterniond(xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
      const double norm = quaternion.norm();
      if (!isFinite(norm) || norm < 0.5 || norm > 1.5) {
        message = prefix + "_orientation_xyzw is not a valid non-zero quaternion";
        return false;
      }
      quaternion.normalize();
      return true;
    }
    quaternion = quaternionFromRpy(rpy[0], rpy[1], rpy[2]);
    quaternion.normalize();
    return true;
  }

  bool extractSolutionStart(
    const moveit_task_constructor_msgs::msg::Solution& solution,
    std::map<std::string, double>& start_joint_positions, std::size_t& trajectory_count,
    std::size_t& point_count, std::string& message) const
  {
    start_joint_positions.clear();
    trajectory_count = 0U;
    point_count = 0U;
    for (const auto& sub_trajectory : solution.sub_trajectory) {
      const auto& trajectory = sub_trajectory.trajectory.joint_trajectory;
      if (trajectory.points.empty()) continue;
      if (trajectory.joint_names.empty()) {
        message = "joint trajectory has no joint_names";
        return false;
      }
      ++trajectory_count;
      point_count += trajectory.points.size();
      for (const auto& point : trajectory.points) {
        if (point.positions.size() != trajectory.joint_names.size()) {
          message = "joint trajectory point count does not match joint_names";
          return false;
        }
        for (const double position : point.positions) {
          if (!isFinite(position)) {
            message = "joint trajectory contains non-finite positions";
            return false;
          }
        }
      }
      if (start_joint_positions.empty()) {
        const auto& first_point = trajectory.points.front();
        for (std::size_t index = 0; index < trajectory.joint_names.size(); ++index) {
          start_joint_positions.emplace(trajectory.joint_names[index], first_point.positions[index]);
        }
      }
    }
    if (trajectory_count == 0U) {
      message = "solution contains no executable joint trajectories";
      return false;
    }
    return true;
  }

  bool executePlanService(std::string& message)
  {
    std::unique_ptr<RetainedPlan> plan;
    {
      std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
      if (!lock.owns_lock()) {
        message = "another calibration verification operation is active";
        return false;
      }
      if (preview_only_) {
        message = "execution blocked: preview_only omits fixture collision geometry";
        return false;
      }
      if (!execution_allowed_) {
        message = "execution blocked: execution_allowed is false; planning remains read-only";
        return false;
      }
      if (execution_in_progress_.load()) {
        message = "execution blocked: a calibration verification trajectory is already active";
        return false;
      }
      if (!retained_plan_) {
        message = "execution blocked: no retained exact plan is available";
        return false;
      }
      plan = std::move(retained_plan_);  // consume the exact plan before validation or dispatch.
      execution_in_progress_.store(true);
      stop_requested_.store(false);
    }

    bool success = false;
    std::string validation_or_execution_message;
    try {
      if (validateBeforeExecution(*plan, validation_or_execution_message)) {
        success = executeRetainedPlan(*plan, validation_or_execution_message);
      }
    } catch (const std::exception& exception) {
      validation_or_execution_message = std::string("execution threw before completion: ") + exception.what();
    }

    std::optional<geometry_msgs::msg::PoseStamped> achieved_pose;
    std::string achieved_message;
    if (computeAchievedTcpPose(plan->robot_model, achieved_pose, achieved_message)) {
      const Eigen::Vector3d target_position = pointVector(plan->target.target_pose.pose.position);
      const Eigen::Vector3d achieved_position = pointVector(achieved_pose->pose.position);
      const Eigen::Vector3d target_minus_achieved = target_position - achieved_position;
      validation_or_execution_message += "; target_tcp_m=" + formatVector3(target_position) +
                                         "; achieved_model_fk_tcp_m=" +
                                         formatVector3(achieved_position) +
                                         "; target_minus_achieved_model_m=" +
                                         formatVector3(target_minus_achieved) +
                                         " (reported for TCP/target bookkeeping, not a calibration residual)";
    } else {
      validation_or_execution_message += "; achieved TCP unavailable: " + achieved_message;
    }

    std::string record_message;
    if (!writeRecord(*plan, success, validation_or_execution_message, achieved_pose, record_message)) {
      validation_or_execution_message += "; record not written: " + record_message;
    } else if (!record_message.empty()) {
      validation_or_execution_message += "; " + record_message;
    }

    {
      std::lock_guard<std::mutex> lock(execution_mutex_);
      active_goal_handle_.reset();
    }
    execution_in_progress_.store(false);
    stop_requested_.store(false);

    message = validation_or_execution_message;
    if (success) {
      RCLCPP_INFO(get_logger(), "%s", message.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "%s", message.c_str());
    }
    return success;
  }

  bool validateBeforeExecution(const RetainedPlan& plan, std::string& message)
  {
    const double plan_age = (now() - plan.planned_at).seconds();
    if (!isFinite(plan_age) || plan_age < -future_stamp_tolerance_s_ || plan_age > plan_expiry_s_) {
      message = "execution blocked: retained plan expired or has invalid age; plan_age_s=" +
                formatDouble(plan_age);
      return false;
    }

    const auto latest_joint_state = freshJointState(message);
    if (!latest_joint_state) {
      message = "execution blocked: " + message;
      return false;
    }
    if (!jointStateMatchesPlanStart(latest_joint_state->state, plan, message)) {
      message = "execution blocked: " + message;
      return false;
    }

    LatestBoardPose latest_board;
    {
      std::lock_guard<std::mutex> lock(board_mutex_);
      if (!latest_board_pose_) {
        message = "execution blocked: no live board pose is available";
        return false;
      }
      latest_board = *latest_board_pose_;
    }
    if (!validateFreshBoardPose(latest_board, message)) {
      message = "execution blocked: live board pose is not fresh: " + message;
      return false;
    }
    if (!boardPoseMatchesSnapshot(latest_board.pose, plan.board.pose, message)) {
      message = "execution blocked: live board pose changed since planning: " + message;
      return false;
    }

    if (!revalidateSceneAndTrajectories(plan, latest_joint_state->state, message)) {
      message = "execution blocked: " + message;
      return false;
    }
    message = "pre-execution checks passed for retained plan " + std::to_string(plan.plan_sequence);
    return true;
  }

  bool jointStateMatchesPlanStart(
    const sensor_msgs::msg::JointState& joint_state, const RetainedPlan& plan,
    std::string& message) const
  {
    for (const auto& [name, planned_position] : plan.start_joint_positions) {
      const auto iterator = std::find(joint_state.name.begin(), joint_state.name.end(), name);
      if (iterator == joint_state.name.end()) {
        message = "current joint state is missing planned start joint " + name;
        return false;
      }
      const std::size_t index = static_cast<std::size_t>(
        std::distance(joint_state.name.begin(), iterator));
      if (index >= joint_state.position.size()) {
        message = "current joint state has no position for " + name;
        return false;
      }
      const double delta = std::abs(joint_state.position[index] - planned_position);
      if (!isFinite(delta) || delta > start_state_tolerance_rad_) {
        message = "current joint " + name + " differs from retained plan start by " +
                  formatDouble(delta) + " rad (tolerance " +
                  formatDouble(start_state_tolerance_rad_) + ")";
        return false;
      }
    }
    return true;
  }

  bool boardPoseMatchesSnapshot(
    const geometry_msgs::msg::PoseStamped& live, const geometry_msgs::msg::PoseStamped& snapshot,
    std::string& message) const
  {
    if (live.header.frame_id != snapshot.header.frame_id) {
      message = "frame changed from " + snapshot.header.frame_id + " to " + live.header.frame_id;
      return false;
    }
    const double position_delta = (pointVector(live.pose.position) - pointVector(snapshot.pose.position)).norm();
    const double angular_delta = quaternionAngularDistance(live.pose.orientation, snapshot.pose.orientation);
    if (position_delta > board_pose_position_tolerance_m_ ||
        angular_delta > board_pose_angular_tolerance_rad_) {
      message = "delta position_m=" + formatDouble(position_delta) + " angular_rad=" +
                formatDouble(angular_delta) + " exceeds tolerances position_m=" +
                formatDouble(board_pose_position_tolerance_m_) + " angular_rad=" +
                formatDouble(board_pose_angular_tolerance_rad_);
      return false;
    }
    return true;
  }

  bool ensurePlanningSceneMonitor(std::string& message)
  {
    std::lock_guard<std::mutex> lock(scene_monitor_mutex_);
    if (planning_scene_monitor_) return true;

    planning_scene_monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      shared_from_this(), "robot_description", "calibration_verification_scene_monitor");
    if (!planning_scene_monitor_->getPlanningScene()) {
      planning_scene_monitor_.reset();
      message = "could not construct a planning scene from robot_description";
      return false;
    }
    planning_scene_monitor_->startStateMonitor(joint_states_topic_);
    planning_scene_monitor_->startSceneMonitor(planning_scene_topic_);
    planning_scene_monitor_->startWorldGeometryMonitor(
      collision_object_topic_, planning_scene_world_topic_, false);
    return true;
  }

  bool revalidateSceneAndTrajectories(
    const RetainedPlan& plan, const sensor_msgs::msg::JointState& current_joint_state,
    std::string& message)
  {
    if (!ensurePlanningSceneMonitor(message)) return false;
    if (!planning_scene_monitor_->requestPlanningSceneState(planning_scene_service_)) {
      message = "failed to request current planning scene from " + planning_scene_service_;
      return false;
    }
    if (!planning_scene_monitor_->waitForCurrentRobotState(now(), planning_scene_state_wait_s_)) {
      message = "planning scene monitor did not receive a fresh robot state from " + joint_states_topic_;
      return false;
    }

    planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
    if (!locked_scene) {
      message = "planning scene is unavailable";
      return false;
    }
    planning_scene::PlanningScenePtr scene = locked_scene->diff();
    for (const auto& object : plan.collision_objects) {
      if (!scene->processCollisionObjectMsg(object)) {
        message = "planning scene rejected collision object " + object.id;
        return false;
      }
    }
    for (const auto& object : plan.collision_objects) {
      if (!scene->getWorld()->hasObject(object.id)) {
        message = "collision object " + object.id + " is missing after scene revalidation";
        return false;
      }
    }

    moveit::core::RobotState state(scene->getCurrentState());
    if (!setRobotStateFromJointState(state, current_joint_state, message)) {
      return false;
    }
    const auto* joint_group = state.getRobotModel()->getJointModelGroup(planning_group_);
    if (joint_group == nullptr) {
      message = "planning group " + planning_group_ + " is missing during revalidation";
      return false;
    }
    state.update();
    if (!state.satisfiesBounds(joint_group)) {
      message = "current start state violates joint bounds";
      return false;
    }
    if (scene->isStateColliding(state, planning_group_, false)) {
      message = "current start state collides with retained board/cube/table or existing planning-scene obstacles";
      return false;
    }

    std::size_t checked_points = 0U;
    for (std::size_t trajectory_index = 0U; trajectory_index < plan.solution.sub_trajectory.size(); ++trajectory_index) {
      const auto& joint_trajectory = plan.solution.sub_trajectory[trajectory_index].trajectory.joint_trajectory;
      if (joint_trajectory.points.empty()) continue;
      if (joint_trajectory.joint_names.empty()) {
        message = "trajectory " + std::to_string(trajectory_index) + " has no joint names";
        return false;
      }
      for (std::size_t point_index = 0U; point_index < joint_trajectory.points.size(); ++point_index) {
        const auto& point = joint_trajectory.points[point_index];
        if (point.positions.size() != joint_trajectory.joint_names.size()) {
          message = "trajectory " + std::to_string(trajectory_index) + " point " +
                    std::to_string(point_index) + " position count mismatch";
          return false;
        }
        for (std::size_t joint_index = 0U; joint_index < joint_trajectory.joint_names.size(); ++joint_index) {
          if (!isFinite(point.positions[joint_index])) {
            message = "trajectory " + std::to_string(trajectory_index) + " point " +
                      std::to_string(point_index) + " contains a non-finite joint position";
            return false;
          }
          state.setVariablePosition(joint_trajectory.joint_names[joint_index], point.positions[joint_index]);
        }
        state.update();
        ++checked_points;
        if (!state.satisfiesBounds(joint_group)) {
          message = "trajectory " + std::to_string(trajectory_index) + " point " +
                    std::to_string(point_index) + " violates joint bounds";
          return false;
        }
        if (scene->isStateColliding(state, planning_group_, false)) {
          message = "trajectory " + std::to_string(trajectory_index) + " point " +
                    std::to_string(point_index) +
                    " collides with retained board/cube/table or existing planning-scene obstacles";
          return false;
        }
      }
    }
    if (checked_points == 0U) {
      message = "retained solution has no executable points during scene revalidation";
      return false;
    }
    return true;
  }

  bool setRobotStateFromJointState(
    moveit::core::RobotState& state, const sensor_msgs::msg::JointState& joint_state,
    std::string& message) const
  {
    const auto& variables = state.getRobotModel()->getVariableNames();
    for (std::size_t index = 0U; index < joint_state.name.size() && index < joint_state.position.size(); ++index) {
      if (std::find(variables.begin(), variables.end(), joint_state.name[index]) != variables.end()) {
        if (!isFinite(joint_state.position[index])) {
          message = "joint " + joint_state.name[index] + " position is non-finite";
          return false;
        }
        state.setVariablePosition(joint_state.name[index], joint_state.position[index]);
      }
    }
    state.update();
    return true;
  }

  bool executeRetainedPlan(const RetainedPlan& plan, std::string& message)
  {
    using namespace std::chrono_literals;
    if (!controller_client_->wait_for_action_server(std::chrono::duration<double>(controller_wait_timeout_s_))) {
      message = "controller action " + controller_action_ + " is unavailable";
      return false;
    }

    std::size_t motion_index = 0U;
    for (const auto& sub_trajectory : plan.solution.sub_trajectory) {
      if (stop_requested_.load()) {
        message = "execution stopped before dispatching next trajectory";
        return false;
      }
      const auto& joint_trajectory = sub_trajectory.trajectory.joint_trajectory;
      if (joint_trajectory.points.empty()) continue;
      ++motion_index;

      FollowJointTrajectory::Goal goal;
      goal.trajectory = joint_trajectory;
      auto goal_future = controller_client_->async_send_goal(goal);
      if (!waitForFuture(goal_future, controller_goal_send_timeout_s_)) {
        message = "timed out sending trajectory " + std::to_string(motion_index) +
                  " to controller action";
        return false;
      }
      if (stop_requested_.load()) {
        message = "execution stopped before controller accepted trajectory " +
                  std::to_string(motion_index);
        return false;
      }
      auto goal_handle = goal_future.get();
      if (!goal_handle) {
        message = "controller rejected trajectory " + std::to_string(motion_index);
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(execution_mutex_);
        active_goal_handle_ = goal_handle;
      }

      RCLCPP_INFO(
        get_logger(), "Executing retained calibration verification trajectory %zu/%zu",
        motion_index, plan.trajectory_count);
      auto result_future = controller_client_->async_get_result(goal_handle);
      const auto execution_deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(execution_timeout_s_));
      bool cancel_sent = false;
      auto cancel_deadline = std::chrono::steady_clock::time_point::max();
      while (result_future.wait_for(100ms) != std::future_status::ready) {
        if (stop_requested_.load() && !cancel_sent) {
          controller_client_->async_cancel_goal(goal_handle);
          cancel_sent = true;
          cancel_deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(controller_cancel_timeout_s_));
        }
        const auto steady_now = std::chrono::steady_clock::now();
        if (cancel_sent && steady_now > cancel_deadline) {
          clearActiveGoal(goal_handle);
          message = "stop requested; controller did not report cancellation within " +
                    formatDouble(controller_cancel_timeout_s_) + " s";
          return false;
        }
        if (!cancel_sent && steady_now > execution_deadline) {
          controller_client_->async_cancel_goal(goal_handle);
          clearActiveGoal(goal_handle);
          message = "trajectory " + std::to_string(motion_index) +
                    " exceeded execution_timeout_s=" + formatDouble(execution_timeout_s_);
          return false;
        }
      }

      const auto wrapped_result = result_future.get();
      clearActiveGoal(goal_handle);
      if (wrapped_result.code == rclcpp_action::ResultCode::CANCELED) {
        message = "trajectory " + std::to_string(motion_index) + " was canceled by ~/stop";
        return false;
      }
      if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED || !wrapped_result.result ||
          wrapped_result.result->error_code != FollowJointTrajectory::Result::SUCCESSFUL) {
        message = "trajectory " + std::to_string(motion_index) + " failed: ";
        if (wrapped_result.result) {
          message += wrapped_result.result->error_string;
        } else {
          message += "no controller result";
        }
        return false;
      }
    }

    if (motion_index == 0U) {
      message = "retained plan contained no executable trajectories";
      return false;
    }
    if (post_execution_state_settle_s_ > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(post_execution_state_settle_s_));
    }
    message = "execution completed for retained calibration verification target " + plan.target.name;
    return true;
  }

  template <typename FutureT>
  bool waitForFuture(FutureT& future, double timeout_s) const
  {
    using namespace std::chrono_literals;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    while (future.wait_for(50ms) != std::future_status::ready) {
      if (stop_requested_.load()) return false;
      if (std::chrono::steady_clock::now() > deadline) return false;
    }
    return true;
  }

  void clearActiveGoal(const GoalHandleFollowJointTrajectory::SharedPtr& goal_handle)
  {
    std::lock_guard<std::mutex> lock(execution_mutex_);
    if (active_goal_handle_ == goal_handle) active_goal_handle_.reset();
  }

  bool stopExecution(std::string& message)
  {
    stop_requested_.store(true);
    std::lock_guard<std::mutex> lock(execution_mutex_);
    if (!execution_in_progress_.load()) {
      stop_requested_.store(false);
      message = "no calibration verification execution is active";
      return true;
    }
    if (active_goal_handle_) {
      controller_client_->async_cancel_goal(active_goal_handle_);
      message = "stop requested; active controller goal cancellation sent";
      return true;
    }
    message = "stop requested; execution will halt before the next controller goal";
    return true;
  }

  bool computeAchievedTcpPose(
    const moveit::core::RobotModelConstPtr& robot_model,
    std::optional<geometry_msgs::msg::PoseStamped>& achieved_pose, std::string& message) const
  {
    if (!robot_model) {
      message = "retained plan has no robot model";
      return false;
    }
    if (robot_model->getLinkModel(tool_link_) == nullptr) {
      message = "tool_link " + tool_link_ + " is missing from robot model";
      return false;
    }
    const auto latest_joint_state = freshJointState(message);
    if (!latest_joint_state) return false;

    moveit::core::RobotState state(robot_model);
    state.setToDefaultValues();
    if (!setRobotStateFromJointState(state, latest_joint_state->state, message)) return false;
    const Eigen::Isometry3d tcp = state.getGlobalLinkTransform(tool_link_);
    geometry_msgs::msg::PoseStamped stamped;
    stamped.header.frame_id = planning_frame_;
    stamped.header.stamp = now();
    stamped.pose = poseMsg(tcp);
    achieved_pose = stamped;
    return true;
  }

  bool writeRecord(
    const RetainedPlan& plan, bool success, const std::string& outcome_message,
    const std::optional<geometry_msgs::msg::PoseStamped>& achieved_pose,
    std::string& record_message) const
  {
    record_message.clear();
    if (output_directory_.empty()) return true;

    try {
      const std::filesystem::path directory = expandUserPath(output_directory_);
      std::error_code error_code;
      std::filesystem::create_directories(directory, error_code);
      if (error_code) {
        record_message = "could not create output_directory " + directory.string() + ": " +
                         error_code.message();
        return false;
      }
      if (record_format_ == "json") {
        const std::filesystem::path output = directory /
          (wallTimestampForFile() + "_plan_" + std::to_string(plan.plan_sequence) + "_" +
           plan.target.name + ".json");
        std::ofstream file(output);
        if (!file) {
          record_message = "could not open " + output.string();
          return false;
        }
        writeJsonRecord(file, plan, success, outcome_message, achieved_pose);
        file.close();
        if (!file) {
          record_message = "failed while writing " + output.string();
          return false;
        }
        record_message = "recorded " + output.string();
        return true;
      }

      const std::filesystem::path output = directory / "calibration_verification_records.csv";
      const bool write_header = !std::filesystem::exists(output);
      std::ofstream file(output, std::ios::app);
      if (!file) {
        record_message = "could not open " + output.string();
        return false;
      }
      if (write_header) {
        file << "wall_time,target,success,message,board_pose,target_pose,preapproach_pose,"
                "achieved_pose,target_minus_achieved_position_m,note\n";
      }
      writeCsvRecord(file, plan, success, outcome_message, achieved_pose);
      file.close();
      if (!file) {
        record_message = "failed while writing " + output.string();
        return false;
      }
      record_message = "recorded " + output.string();
      return true;
    } catch (const std::exception& exception) {
      record_message = exception.what();
      return false;
    }
  }

  void writeJsonRecord(
    std::ostream& file, const RetainedPlan& plan, bool success,
    const std::string& outcome_message,
    const std::optional<geometry_msgs::msg::PoseStamped>& achieved_pose) const
  {
    file << std::setprecision(15);
    file << "{\n";
    file << "  \"wall_time\": \"" << wallTimestampForFile() << "\",\n";
    file << "  \"plan_sequence\": " << plan.plan_sequence << ",\n";
    file << "  \"capture_sequence\": " << plan.board.sequence << ",\n";
    file << "  \"target\": \"" << jsonEscape(plan.target.name) << "\",\n";
    file << "  \"success\": " << (success ? "true" : "false") << ",\n";
    file << "  \"message\": \"" << jsonEscape(outcome_message) << "\",\n";
    file << "  \"note\": \"achieved_tcp_pose is computed from reported joint state and the configured robot model; it is not an external physical calibration residual\",\n";
    file << "  \"board_pose\": " << poseJson(plan.board.pose) << ",\n";
    file << "  \"target_pose\": " << poseJson(plan.target.target_pose) << ",\n";
    file << "  \"preapproach_pose\": " << poseJson(plan.target.preapproach_pose) << ",\n";
    file << "  \"reference_point_base_m\": [" << plan.target.reference_point_base.x() << ','
         << plan.target.reference_point_base.y() << ',' << plan.target.reference_point_base.z()
         << "],\n";
    file << "  \"approach_direction_base\": [" << plan.target.approach_direction_base.x() << ','
         << plan.target.approach_direction_base.y() << ',' << plan.target.approach_direction_base.z()
         << "],\n";
    file << "  \"planned_start_joints\": {";
    bool first = true;
    for (const auto& [name, position] : plan.start_joint_positions) {
      if (!first) file << ',';
      first = false;
      file << "\"" << jsonEscape(name) << "\":" << position;
    }
    file << "},\n";
    if (achieved_pose) {
      const Eigen::Vector3d target_position = pointVector(plan.target.target_pose.pose.position);
      const Eigen::Vector3d achieved_position = pointVector(achieved_pose->pose.position);
      const Eigen::Vector3d target_minus_achieved = target_position - achieved_position;
      file << "  \"achieved_tcp_pose\": " << poseJson(*achieved_pose) << ",\n";
      file << "  \"target_minus_achieved_position_m\": [" << target_minus_achieved.x() << ','
           << target_minus_achieved.y() << ',' << target_minus_achieved.z() << "]\n";
    } else {
      file << "  \"achieved_tcp_pose\": null,\n";
      file << "  \"target_minus_achieved_position_m\": null\n";
    }
    file << "}\n";
  }

  void writeCsvRecord(
    std::ostream& file, const RetainedPlan& plan, bool success,
    const std::string& outcome_message,
    const std::optional<geometry_msgs::msg::PoseStamped>& achieved_pose) const
  {
    std::string achieved = "";
    std::string delta = "";
    if (achieved_pose) {
      achieved = poseJson(*achieved_pose);
      const Eigen::Vector3d target_position = pointVector(plan.target.target_pose.pose.position);
      const Eigen::Vector3d achieved_position = pointVector(achieved_pose->pose.position);
      delta = formatVector3(target_position - achieved_position);
    }
    file << csvEscape(wallTimestampForFile()) << ',' << csvEscape(plan.target.name) << ','
         << (success ? "true" : "false") << ',' << csvEscape(outcome_message) << ','
         << csvEscape(poseJson(plan.board.pose)) << ',' << csvEscape(poseJson(plan.target.target_pose))
         << ',' << csvEscape(poseJson(plan.target.preapproach_pose)) << ',' << csvEscape(achieved)
         << ',' << csvEscape(delta) << ','
         << csvEscape(
              "achieved pose is model FK from reported joint state, not an external physical calibration residual")
         << '\n';
  }

  bool configuredVector3(const std::vector<double>& values) const
  {
    return values.size() == 3U && finiteVector(values);
  }

  bool positiveDimensions(
    const std::vector<double>& dimensions, const std::string& name, std::string& message) const
  {
    if (!configuredVector3(dimensions)) {
      message = name + " must contain three finite dimensions";
      return false;
    }
    for (const double dimension : dimensions) {
      if (dimension <= 0.0 || dimension > 5.0) {
        message = name + " dimensions must be in (0, 5] m";
        return false;
      }
    }
    return true;
  }

  moveit_msgs::msg::CollisionObject makeBoxCollisionObject(
    const std::string& id, const Eigen::Isometry3d& base_reference,
    const std::vector<double>& local_center_xyz, const std::vector<double>& dimensions_xyz) const
  {
    Eigen::Isometry3d box_pose = base_reference;
    box_pose.translation() = base_reference * Eigen::Vector3d(
      local_center_xyz[0], local_center_xyz[1], local_center_xyz[2]);

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {dimensions_xyz[0], dimensions_xyz[1], dimensions_xyz[2]};

    moveit_msgs::msg::CollisionObject object;
    object.id = id;
    object.header.frame_id = planning_frame_;
    object.header.stamp = now();
    object.primitives.push_back(primitive);
    object.primitive_poses.push_back(poseMsg(box_pose));
    object.operation = moveit_msgs::msg::CollisionObject::ADD;
    return object;
  }

  Eigen::Vector3d referencePoint(double x, double y) const
  {
    return Eigen::Vector3d(reference_origin_xyz_m_[0] + x,
                           reference_origin_xyz_m_[1] + y,
                           reference_origin_xyz_m_[2] + board_target_height_m_);
  }

  void publishMarkers(
    const CapturedBoardPose& board, const std::optional<TargetSpec>& target,
    const std::vector<moveit_msgs::msg::CollisionObject>& collision_objects) const
  {
    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker delete_all;
    delete_all.header.frame_id = planning_frame_;
    delete_all.header.stamp = now();
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    array.markers.push_back(delete_all);

    const Eigen::Isometry3d base_board = poseToEigen(board.pose.pose);
    Eigen::Isometry3d base_reference = base_board;
    base_reference.translation() = base_board * referencePoint(0.0, 0.0);
    addAxisMarkers(array, base_reference, "captured_board", 0, 0.075);
    std::vector<std::pair<std::string, std::vector<double>>> points{{"center", {0.0, 0.0}}};
    for (std::size_t index = 0U; index < kBoardPointCount; ++index) {
      points.emplace_back(boardPointName(index), point_xy_m_[index]);
    }
    int point_id = 300;
    for (const auto& [name, xy] : points) {
      if (xy.size() != 2U) continue;
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = planning_frame_;
      marker.header.stamp = now();
      marker.ns = "board_reference_points";
      marker.id = point_id++;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position = pointMsg(base_board * referencePoint(xy[0], xy[1]));
      marker.pose.orientation.w = 1.0;
      marker.scale.x = marker.scale.y = marker.scale.z = 0.012;
      marker.color.r = 1.0; marker.color.g = 0.8; marker.color.a = 1.0;
      array.markers.push_back(marker);
      marker.id = point_id++;
      marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      marker.text = name;
      marker.pose.position.z += 0.02;
      marker.scale.z = 0.018;
      array.markers.push_back(marker);
    }

    int marker_id = 100;
    for (const auto& object : collision_objects) {
      if (object.primitives.empty() || object.primitive_poses.empty()) continue;
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = planning_frame_;
      marker.header.stamp = now();
      marker.ns = "calibration_collision_objects";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = object.primitive_poses.front();
      const auto& dimensions = object.primitives.front().dimensions;
      if (dimensions.size() >= 3U) {
        marker.scale.x = dimensions[0];
        marker.scale.y = dimensions[1];
        marker.scale.z = dimensions[2];
      } else {
        marker.scale.x = marker.scale.y = marker.scale.z = 0.01;
      }
      marker.color.a = 0.25;
      if (object.id == cube_collision_id_) {
        marker.color.r = 0.1;
        marker.color.g = 0.4;
        marker.color.b = 1.0;
      } else if (object.id == table_collision_id_) {
        marker.color.r = 0.55;
        marker.color.g = 0.35;
        marker.color.b = 0.15;
      } else {
        marker.color.r = 0.1;
        marker.color.g = 0.9;
        marker.color.b = 0.2;
      }
      array.markers.push_back(marker);
    }

    if (target) {
      visualization_msgs::msg::Marker line;
      line.header.frame_id = planning_frame_;
      line.header.stamp = now();
      line.ns = "calibration_target";
      line.id = 200;
      line.type = visualization_msgs::msg::Marker::ARROW;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.points = {target->preapproach_pose.pose.position, target->target_pose.pose.position};
      line.scale.x = 0.006;
      line.scale.y = 0.014;
      line.scale.z = 0.020;
      line.color.r = 1.0;
      line.color.g = 0.75;
      line.color.b = 0.0;
      line.color.a = 1.0;
      array.markers.push_back(line);

      visualization_msgs::msg::Marker reference;
      reference.header.frame_id = planning_frame_;
      reference.header.stamp = now();
      reference.ns = "calibration_target";
      reference.id = 201;
      reference.type = visualization_msgs::msg::Marker::SPHERE;
      reference.action = visualization_msgs::msg::Marker::ADD;
      reference.pose.position = pointMsg(target->reference_point_base);
      reference.pose.orientation.w = 1.0;
      reference.scale.x = reference.scale.y = reference.scale.z = 0.018;
      reference.color.r = 1.0;
      reference.color.g = 1.0;
      reference.color.a = 1.0;
      array.markers.push_back(reference);

      visualization_msgs::msg::Marker target_sphere;
      target_sphere.header.frame_id = planning_frame_;
      target_sphere.header.stamp = now();
      target_sphere.ns = "calibration_target";
      target_sphere.id = 202;
      target_sphere.type = visualization_msgs::msg::Marker::SPHERE;
      target_sphere.action = visualization_msgs::msg::Marker::ADD;
      target_sphere.pose = target->target_pose.pose;
      target_sphere.scale.x = target_sphere.scale.y = target_sphere.scale.z = 0.025;
      target_sphere.color.g = 1.0;
      target_sphere.color.b = 1.0;
      target_sphere.color.a = 1.0;
      array.markers.push_back(target_sphere);

      visualization_msgs::msg::Marker preapproach;
      preapproach.header.frame_id = planning_frame_;
      preapproach.header.stamp = now();
      preapproach.ns = "calibration_target";
      preapproach.id = 203;
      preapproach.type = visualization_msgs::msg::Marker::SPHERE;
      preapproach.action = visualization_msgs::msg::Marker::ADD;
      preapproach.pose = target->preapproach_pose.pose;
      preapproach.scale.x = preapproach.scale.y = preapproach.scale.z = 0.018;
      preapproach.color.r = 1.0;
      preapproach.color.b = 1.0;
      preapproach.color.a = 1.0;
      array.markers.push_back(preapproach);
    }

    marker_pub_->publish(array);
  }

  void addAxisMarkers(
    visualization_msgs::msg::MarkerArray& array, const Eigen::Isometry3d& transform,
    const std::string& name, int start_id, double length) const
  {
    const std::array<Eigen::Vector3d, 3> axes{
      Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ()};
    for (std::size_t index = 0U; index < axes.size(); ++index) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = planning_frame_;
      marker.header.stamp = now();
      marker.ns = name;
      marker.id = start_id + static_cast<int>(index);
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;
      const Eigen::Vector3d origin = transform.translation();
      const Eigen::Vector3d endpoint = origin + length * (transform.linear() * axes[index]);
      marker.points = {pointMsg(origin), pointMsg(endpoint)};
      marker.scale.x = 0.004;
      marker.scale.y = 0.010;
      marker.scale.z = 0.014;
      marker.color.a = 1.0;
      marker.color.r = index == 0U ? 1.0 : 0.0;
      marker.color.g = index == 1U ? 1.0 : 0.0;
      marker.color.b = index == 2U ? 1.0 : 0.0;
      array.markers.push_back(marker);
    }
  }

  std::mutex board_mutex_;
  mutable std::mutex joint_state_mutex_;
  std::mutex plan_mutex_;
  std::mutex execution_mutex_;
  std::mutex scene_monitor_mutex_;
  std::mutex camera_pole_mutex_;

  std::optional<LatestBoardPose> latest_board_pose_;
  std::optional<CapturedBoardPose> captured_board_pose_;
  std::optional<LatestJointState> latest_joint_state_;
  std::unique_ptr<RetainedPlan> retained_plan_;
  std::mutex robot_model_mutex_;
  moveit::core::RobotModelConstPtr robot_model_;
  std::uint64_t capture_sequence_{0U};
  std::uint64_t plan_sequence_{0U};

  std::atomic_bool execution_in_progress_{false};
  std::atomic_bool stop_requested_{false};
  GoalHandleFollowJointTrajectory::SharedPtr active_goal_handle_;

  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr board_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr camera_pole_sub_;
  std::optional<moveit_msgs::msg::CollisionObject> camera_pole_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr controller_client_;
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr capture_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_center_service_;
  std::array<rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr, kBoardPointCount> plan_point_services_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_cube_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr execute_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_service_;

  std::string planning_group_;
  std::string planning_frame_;
  std::string tool_link_;
  std::string planning_pipeline_;
  std::string planner_id_;
  bool camera_pole_required_{true};
  std::string camera_pole_topic_;
  std::string controller_name_;
  std::string controller_action_;
  std::string board_collision_id_;
  std::string table_collision_id_;
  std::string cube_collision_id_;
  std::string joint_states_topic_;
  std::string planning_scene_service_;
  std::string planning_scene_topic_;
  std::string planning_scene_world_topic_;
  std::string collision_object_topic_;
  std::string output_directory_;
  std::string record_format_;

  std::array<std::vector<double>, kBoardPointCount> point_xy_m_;
  std::vector<double> board_to_tcp_orientation_xyzw_;
  std::vector<double> board_to_tcp_rpy_rad_;
  std::vector<double> board_box_center_xyz_m_;
  std::vector<double> board_box_dimensions_xyz_m_;
  std::vector<double> table_box_center_xyz_m_;
  std::vector<double> table_box_dimensions_xyz_m_;
  std::vector<double> cube_pose_xyz_m_;
  std::vector<double> cube_pose_rpy_rad_;
  std::vector<double> cube_dimensions_xyz_m_;
  std::vector<double> cube_reference_xyz_m_;
  std::vector<double> cube_approach_direction_xyz_;
  std::vector<double> cube_to_tcp_orientation_xyzw_;
  std::vector<double> cube_to_tcp_rpy_rad_;

  double board_target_height_m_{0.0};
  std::vector<double> reference_origin_xyz_m_;
  bool preapproach_only_{false};
  bool retract_after_approach_{true};
  double target_standoff_m_{0.020};
  double final_standoff_m_{0.001};
  double preapproach_distance_m_{0.080};
  double cartesian_step_m_{0.003};
  double planning_timeout_s_{15.0};
  double velocity_scaling_{0.05};
  double acceleration_scaling_{0.05};
  double max_board_pose_age_s_{2.0};
  double max_joint_state_age_s_{2.0};
  double plan_expiry_s_{60.0};
  double start_state_tolerance_rad_{0.01};
  double board_pose_position_tolerance_m_{0.002};
  double board_pose_angular_tolerance_rad_{1.0 * kPi / 180.0};
  double future_stamp_tolerance_s_{0.1};
  double controller_wait_timeout_s_{30.0};
  double controller_goal_send_timeout_s_{5.0};
  double execution_timeout_s_{120.0};
  double controller_cancel_timeout_s_{5.0};
  double post_execution_state_settle_s_{0.25};
  double planning_scene_state_wait_s_{2.0};
  int max_solutions_{1};
  bool execution_allowed_{false};
  bool preview_only_{false};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  std::shared_ptr<CalibrationVerification> node;
  try {
    node = std::make_shared<CalibrationVerification>();
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("calibration_verification"), "startup failed: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
