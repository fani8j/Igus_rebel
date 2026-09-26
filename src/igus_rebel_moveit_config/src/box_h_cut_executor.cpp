#include <Eigen/Geometry>
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/joint_interpolation.h>
#include <moveit/task_constructor/solvers/multi_planner.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stage.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/task.h>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
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
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>
#include <system_error>

namespace mtc = moveit::task_constructor;


class StationaryDwell : public mtc::PropagatingEitherWay
{
public:
  StationaryDwell(std::string name, std::string planning_group, double duration_s)
  : mtc::PropagatingEitherWay(std::move(name)),
    planning_group_(std::move(planning_group)), duration_s_(duration_s)
  {
    restrictDirection(FORWARD);
  }

protected:
  void computeForward(const mtc::InterfaceState& from) override
  {
    auto trajectory = std::make_shared<robot_trajectory::RobotTrajectory>(
      from.scene()->getRobotModel(), planning_group_);
    moveit::core::RobotState stationary_state(from.scene()->getCurrentState());
    const std::vector<double> zero(stationary_state.getVariableCount(), 0.0);
    stationary_state.setVariableVelocities(zero);
    stationary_state.setVariableAccelerations(zero);
    trajectory->addSuffixWayPoint(stationary_state, 0.0);
    trajectory->addSuffixWayPoint(stationary_state, duration_s_);

    mtc::InterfaceState to(from);
    mtc::SubTrajectory dwell(trajectory);
    dwell.setPlannerId("stationary_dwell");
    dwell.setComment("stationary pre-cut stabilization");
    sendForward(from, std::move(to), std::move(dwell));
  }

private:
  std::string planning_group_;
  double duration_s_;
};

// Free-space transit wrapper that resolves a Cartesian (pose) goal to the exact-pose,
// collision-free IK branch nearest the propagated start state before planning. Seeds
// are the start state and its spherical-wrist flip (joint4 +/- pi, -joint5, joint6 +/- pi);
// every IK result is expanded with the equivalent wrist-flip/joint6 +/- 2pi branches,
// each re-verified by forward kinematics, bounds and collision. Candidates whose joint6
// change would exceed pi are ranked last, then by L1 joint distance, so the transit takes
// the short wrist branch. The target pose is never altered and the trajectory is still
// produced by the wrapped planner; the solution-level wrist guard remains authoritative.
class NearestBranchTransitPlanner : public mtc::solvers::PlannerInterface
{
public:
  explicit NearestBranchTransitPlanner(mtc::solvers::PlannerInterfacePtr inner)
  : inner_(std::move(inner)) {}

  void init(const moveit::core::RobotModelConstPtr& robot_model) override
  {
    inner_->init(robot_model);
  }

  Result plan(
    const planning_scene::PlanningSceneConstPtr& from,
    const planning_scene::PlanningSceneConstPtr& to,
    const moveit::core::JointModelGroup* jmg, double timeout,
    robot_trajectory::RobotTrajectoryPtr& result,
    const moveit_msgs::msg::Constraints& path_constraints) override
  {
    return inner_->plan(from, to, jmg, timeout, result, path_constraints);
  }

  Result plan(
    const planning_scene::PlanningSceneConstPtr& from, const moveit::core::LinkModel& link,
    const Eigen::Isometry3d& offset, const Eigen::Isometry3d& target,
    const moveit::core::JointModelGroup* jmg, double timeout,
    robot_trajectory::RobotTrajectoryPtr& result,
    const moveit_msgs::msg::Constraints& path_constraints) override
  {
    const auto logger = rclcpp::get_logger("box_h_cut_executor.transit_ik");
    constexpr double kPi = 3.14159265358979323846;
    const auto& variables = jmg->getVariableNames();
    const auto index_of = [&](const std::string& name) -> std::optional<std::size_t> {
      const auto it = std::find(variables.begin(), variables.end(), name);
      if (it == variables.end()) return std::nullopt;
      return static_cast<std::size_t>(std::distance(variables.begin(), it));
    };
    const auto j4 = index_of("joint4");
    const auto j5 = index_of("joint5");
    const auto j6 = index_of("joint6");
    if (!j6) return inner_->plan(from, link, offset, target, jmg, timeout, result, path_constraints);

    const moveit::core::RobotState& start = from->getCurrentState();
    std::vector<double> start_values;
    start.copyJointGroupPositions(jmg, start_values);
    const Eigen::Isometry3d link_target = target * offset.inverse();

    const auto is_valid = [&](moveit::core::RobotState& state) {
      state.update();
      return state.satisfiesBounds(jmg) && from->isStateValid(state, jmg->getName());
    };
    const auto exact_pose = [&](moveit::core::RobotState& state) {
      state.update();
      const Eigen::Isometry3d& reached = state.getGlobalLinkTransform(&link);
      const double position_error = (reached.translation() - link_target.translation()).norm();
      const double angle_error = Eigen::AngleAxisd(
        reached.linear().transpose() * link_target.linear()).angle();
      return position_error <= 1e-5 && std::abs(angle_error) <= 1e-4;
    };

    std::vector<std::vector<double>> candidates;
    // `ik_verified`: the raw IK solution already passed is_valid inside the IK
    // validity callback, so only its exact-pose check remains.
    const auto add_candidate = [&](const std::vector<double>& values, bool ik_verified) {
      for (const auto& existing : candidates) {
        double max_diff = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
          max_diff = std::max(max_diff, std::abs(existing[i] - values[i]));
        }
        if (max_diff < 1e-4) return;
      }
      moveit::core::RobotState probe(start);
      probe.setJointGroupPositions(jmg, values);
      if ((!ik_verified && !is_valid(probe)) || !exact_pose(probe)) return;
      candidates.push_back(values);
    };
    const auto wrist_variants = [&](const std::vector<double>& values) {
      std::vector<std::vector<double>> variants;
      for (const double turn6 : {-2.0 * kPi, 2.0 * kPi}) {
        auto shifted = values;
        shifted[*j6] += turn6;
        variants.push_back(shifted);
      }
      if (j4 && j5) {
        for (const double turn4 : {-kPi, kPi}) {
          for (const double turn6 : {-kPi, kPi}) {
            auto flipped = values;
            flipped[*j4] += turn4;
            flipped[*j5] = -flipped[*j5];
            flipped[*j6] += turn6;
            variants.push_back(flipped);
          }
        }
      }
      return variants;
    };

    std::vector<std::vector<double>> seeds{start_values};
    for (const auto& variant : wrist_variants(start_values)) {
      moveit::core::RobotState probe(start);
      probe.setJointGroupPositions(jmg, variant);
      probe.update();
      if (probe.satisfiesBounds(jmg)) seeds.push_back(variant);
    }
    const moveit::core::GroupStateValidityCallbackFn ik_valid =
      [&](moveit::core::RobotState* state, const moveit::core::JointModelGroup* group,
          const double* values) {
        state->setJointGroupPositions(group, values);
        return is_valid(*state);
      };
    for (const auto& seed : seeds) {
      moveit::core::RobotState ik_state(start);
      ik_state.setJointGroupPositions(jmg, seed);
      ik_state.update();
      if (!ik_state.setFromIK(jmg, link_target, link.getName(), 0.0, ik_valid)) continue;
      std::vector<double> solution;
      ik_state.copyJointGroupPositions(jmg, solution);
      add_candidate(solution, true);
      for (const auto& variant : wrist_variants(solution)) add_candidate(variant, false);
    }
    if (candidates.empty()) {
      RCLCPP_WARN(
        logger, "no seeded collision-free IK branch for transit goal; using wrapped planner");
      return inner_->plan(from, link, offset, target, jmg, timeout, result, path_constraints);
    }

    const auto wrist_travel = [&](const std::vector<double>& values) {
      return std::abs(values[*j6] - start_values[*j6]);
    };
    const auto joint_distance = [&](const std::vector<double>& values) {
      double sum = 0.0;
      for (std::size_t i = 0; i < values.size(); ++i) sum += std::abs(values[i] - start_values[i]);
      return sum;
    };
    std::stable_sort(
      candidates.begin(), candidates.end(),
      [&](const std::vector<double>& lhs, const std::vector<double>& rhs) {
        const bool lhs_long = wrist_travel(lhs) > kPi;
        const bool rhs_long = wrist_travel(rhs) > kPi;
        if (lhs_long != rhs_long) return !lhs_long;
        return joint_distance(lhs) < joint_distance(rhs);
      });

    std::ostringstream summary;
    summary << std::fixed << std::setprecision(4) << "start joint6=" << start_values[*j6]
            << "; " << candidates.size() << " IK branch(es):";
    for (const auto& candidate : candidates) {
      summary << " [joint6=" << candidate[*j6] << " d6=" << wrist_travel(candidate)
              << " L1=" << joint_distance(candidate) << "]";
    }
    RCLCPP_INFO(logger, "transit goal %s", summary.str().c_str());

    Result last{false, "no IK branch could be planned"};
    for (const auto& candidate : candidates) {
      auto goal_scene = from->diff();
      goal_scene->getCurrentStateNonConst().setJointGroupPositions(jmg, candidate);
      goal_scene->getCurrentStateNonConst().update();
      last = inner_->plan(from, goal_scene, jmg, timeout, result, path_constraints);
      if (last) {
        RCLCPP_INFO(
          logger, "transit planned to IK branch joint6=%.4f (joint6 change %.4f rad)",
          candidate[*j6], wrist_travel(candidate));
        return last;
      }
    }
    return last;
  }

  std::string getPlannerId() const override { return "NearestBranchTransit"; }

private:
  mtc::solvers::PlannerInterfacePtr inner_;
};

struct CutSegment {
  std::string name;
  geometry_msgs::msg::Point start;
  geometry_msgs::msg::Point end;
  bool stop_after_start{false};
  bool hover_only{false};
  bool force_fixed_orientation{false};
  bool handoff_end{false};
  bool terminal_after_cut{false};
  double handoff_fraction{0.0};
};
struct SegmentPlanOptions {
  // Skip the supplied joint-space approach (approach_joint_degrees); only "h" uses it.
  bool skip_supplied_approach{true};
  double timeout_s{15.0};
  bool explain_failure{true};
  // Staged cut-debug start plan: the carton is added together with the scoped
  // tool/carton allowance in one atomic scene change (the arm may still be in contact
  // from an earlier session), and the left roll search probes only the start poses.
  bool cut_debug_start{false};
};
struct Snapshot {
  std::string snapshot_id;
  std::string capture_id;
  std::int64_t acquisition_stamp_ns{0};
  std_msgs::msg::Header header;
  geometry_msgs::msg::Point seam_start;
  geometry_msgs::msg::Point seam_end;
  geometry_msgs::msg::Point seam_center;
  geometry_msgs::msg::Point back_left;
  geometry_msgs::msg::Point back_right;
  geometry_msgs::msg::Point front_left;
  geometry_msgs::msg::Point front_right;
  geometry_msgs::msg::Point back_middle_left;
  geometry_msgs::msg::Point front_middle_left;
  geometry_msgs::msg::Point back_middle_right;
  geometry_msgs::msg::Point front_middle_right;
  geometry_msgs::msg::Vector3 surface_normal;
  double plane_rms_m{0.0};
};
enum class CutDebugStage : std::uint8_t {
  kNone = 0,
  kStartPlanned,
  kStartExecuted,
  kEndExecuted,
};
struct CutLocalDeltas {
  double along_m{0.0};
  double across_m{0.0};
  double normal_m{0.0};
};
struct CutDebugCorrection {
  std::string cut;
  CutLocalDeltas start_local{};
  CutLocalDeltas end_local{};
  std::array<double, 3> recorded_along{};
  std::array<double, 3> recorded_across{};
  std::array<double, 3> recorded_normal{};
  geometry_msgs::msg::Quaternion observed_start{};
  geometry_msgs::msg::Quaternion observed_end{};
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
    const auto motion_profile = declare_parameter<std::string>("motion_profile", "ompl_pilz");
    declare_parameter<std::string>("external_snapshot_file", "");
    external_snapshot_max_age_s_ = declare_parameter<double>(
      "external_snapshot_max_age_s", 30.0);
    debug_allow_stale_snapshot_ = declare_parameter<bool>(
      "debug_allow_stale_snapshot", false);
    carton_collision_id_ = declare_parameter<std::string>("carton_collision_id", "detected_carton");
    allow_tool_carton_contact_ = declare_parameter<bool>("allow_tool_carton_contact", false);
    camera_pole_required_ = declare_parameter<bool>("camera_pole_required", true);
    camera_pole_topic_ = declare_parameter<std::string>(
      "camera_pole_topic", "/box_motion/camera_pole_collision");
    carton_height_m_ = declare_parameter<double>("carton_height_m", 0.20);
    hover_clearance_m_ = declare_parameter<double>("hover_clearance_m", 0.10);
    left_lateral_standoff_m_ = declare_parameter<double>("left_lateral_standoff_m", 0.010);
    cut_clearance_m_ = declare_parameter<double>("cut_clearance_m", 0.05);
    middle_cut_clearance_m_ = declare_parameter<double>(
      "middle_cut_clearance_m", cut_clearance_m_);
    middle_entry_alignment_m_ = declare_parameter<double>("middle_entry_alignment_m", 0.010);
    pre_cut_stabilization_duration_s_ = declare_parameter<double>(
      "pre_cut_stabilization_duration_s", 1.0);
    left_side_cut_tilt_deg_ = declare_parameter<double>("left_side_cut_tilt_deg", 0.0);
    right_side_cut_tilt_deg_ = declare_parameter<double>("right_side_cut_tilt_deg", 0.0);
    side_cut_orientation_mode_ = declare_parameter<std::string>(
      "side_cut_orientation_mode", "edge_aligned");
    left_adaptive_orientation_enabled_ = declare_parameter<bool>(
      "left_adaptive_orientation_enabled", true);
    left_adaptive_orientation_max_deg_ = declare_parameter<double>(
      "left_adaptive_orientation_max_deg", 45.0);
    left_adaptive_orientation_step_deg_ = declare_parameter<double>(
      "left_adaptive_orientation_step_deg", 5.0);
    // Expensive per-pose IK/collision probing that explains each rejected left roll
    // candidate. Read at use time; the roll search itself is unaffected.
    declare_parameter<bool>("left_roll_diagnostics_enabled", false);
    debug_motion_distance_m_ = declare_parameter<double>("debug_motion_distance_m", 0.010);
    debug_motion_clearance_m_ = declare_parameter<double>("debug_motion_clearance_m", 0.030);
    debug_fixed_orientation_ = declare_parameter<bool>("debug_fixed_orientation", false);
    cut_pattern_ = declare_parameter<std::string>("cut_pattern", "t_cut");
    t_cut_middle_handoff_fraction_ = declare_parameter<double>(
      "t_cut_middle_handoff_fraction", 0.0);
    // Supplied joint-space approach used only by the "h" pattern before its first cut.
    approach_joint_degrees_ = declare_parameter<std::vector<double>>(
      "approach_joint_degrees", std::vector<double>{});
    cartesian_step_m_ = declare_parameter<double>("cartesian_step_m", 0.003);
    planning_timeout_s_ = declare_parameter<double>("planning_timeout_s", 15.0);
    velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.05);
    acceleration_scaling_ = declare_parameter<double>("acceleration_scaling", 0.05);
    transit_velocity_scaling_ = declare_parameter<double>("transit_velocity_scaling", 0.05);
    transit_acceleration_scaling_ = declare_parameter<double>("transit_acceleration_scaling", 0.05);
    declare_parameter<bool>("execution_enabled", false);
    declare_parameter<bool>("debug_motion_execution_enabled", false);
    cut_debug_cut_ = declare_parameter<std::string>("cut_debug_cut", "middle");
    cut_debug_correction_enabled_ = declare_parameter<bool>("cut_debug_correction_enabled", false);
    cut_debug_left_correction_file_ = declare_parameter<std::string>(
      "cut_debug_left_correction_file", "");
    cut_debug_middle_correction_file_ = declare_parameter<std::string>(
      "cut_debug_middle_correction_file", "");
    cut_debug_right_correction_file_ = declare_parameter<std::string>(
      "cut_debug_right_correction_file", "");
    cut_debug_output_dir_ = declare_parameter<std::string>("cut_debug_output_dir", "");
    max_solutions_ = declare_parameter<int>("max_solutions", 1);
    if (external_snapshot_max_age_s_ <= 0.0) {
      throw std::invalid_argument("external_snapshot_max_age_s must be positive");
    }
    if (!isSupportedMotionProfile(motion_profile)) {
      throw std::invalid_argument("unsupported H-cut motion_profile: " + motion_profile);
    }
    if (planning_frame_.empty() || carton_collision_id_.empty() ||
        (camera_pole_required_ && camera_pole_topic_.empty()) ||
        carton_height_m_ <= 0.0 ||
        hover_clearance_m_ < 0.005 || left_lateral_standoff_m_ < 0.005 ||
        left_lateral_standoff_m_ > 0.025 || cut_clearance_m_ < -0.003 ||
        middle_cut_clearance_m_ < -0.005 ||
        ((cut_clearance_m_ < 0.0 || middle_cut_clearance_m_ < 0.0) &&
         !allow_tool_carton_contact_) ||
        cut_clearance_m_ >= hover_clearance_m_ ||
        middle_cut_clearance_m_ >= hover_clearance_m_ || middle_entry_alignment_m_ < 0.0 ||
        middle_entry_alignment_m_ > 0.10 ||
        !std::isfinite(pre_cut_stabilization_duration_s_) ||
        pre_cut_stabilization_duration_s_ < 0.0 || pre_cut_stabilization_duration_s_ > 10.0 ||
        cartesian_step_m_ <= 0.0 || planning_timeout_s_ <= 0.0 ||
        debug_motion_distance_m_ <= 0.0 || debug_motion_distance_m_ > 0.020 ||
        debug_motion_clearance_m_ < 0.010 || debug_motion_clearance_m_ > 0.100 ||
        std::abs(left_side_cut_tilt_deg_) > 90.0 ||
        std::abs(right_side_cut_tilt_deg_) > 90.0 ||
        !std::isfinite(left_adaptive_orientation_max_deg_) ||
        left_adaptive_orientation_max_deg_ <= 0.0 ||
        left_adaptive_orientation_max_deg_ > 90.0 ||
        !std::isfinite(left_adaptive_orientation_step_deg_) ||
        left_adaptive_orientation_step_deg_ <= 0.0 ||
        left_adaptive_orientation_step_deg_ > 90.0 ||
        left_adaptive_orientation_step_deg_ > left_adaptive_orientation_max_deg_ ||
        velocity_scaling_ <= 0.0 || velocity_scaling_ > 1.0 || acceleration_scaling_ <= 0.0 ||
        acceleration_scaling_ > 1.0 ||
        !std::isfinite(transit_velocity_scaling_) || transit_velocity_scaling_ <= 0.0 ||
        transit_velocity_scaling_ > 1.0 || !std::isfinite(transit_acceleration_scaling_) ||
        transit_acceleration_scaling_ <= 0.0 || transit_acceleration_scaling_ > 1.0 ||
        max_solutions_ < 1 ||
        !isValidHandoffFraction(t_cut_middle_handoff_fraction_) ||
        !isSupportedCutPattern(cut_pattern_) ||
        (!approach_joint_degrees_.empty() && approach_joint_degrees_.size() != 6U) ||
        (cut_debug_cut_ != "left" && cut_debug_cut_ != "middle" && cut_debug_cut_ != "right") ||
        !isAbsoluteJsonOrEmpty(cut_debug_left_correction_file_) ||
        !isAbsoluteJsonOrEmpty(cut_debug_middle_correction_file_) ||
        !isAbsoluteJsonOrEmpty(cut_debug_right_correction_file_) ||
        (!cut_debug_output_dir_.empty() && cut_debug_output_dir_.front() != '/')) {
      throw std::invalid_argument("invalid H-cut planning safety parameters");
    }
    if (side_cut_orientation_mode_ != "edge_aligned" &&
        side_cut_orientation_mode_ != "middle_aligned") {
      throw std::invalid_argument(
        "unsupported side_cut_orientation_mode (edge_aligned or middle_aligned)");
    }
    cut_pattern_callback_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        std::optional<std::string> requested_pattern;
        std::optional<bool> requested_debug_orientation;
        std::optional<bool> requested_debug_stale_snapshot;
        std::optional<double> requested_left_tilt;
        std::optional<double> requested_right_tilt;
        std::optional<double> requested_pre_cut_dwell;
        std::optional<std::string> requested_debug_cut;
        std::optional<bool> requested_correction_enabled;
        std::optional<bool> requested_left_adaptive_enabled;
        std::optional<double> requested_left_adaptive_max_deg;
        std::optional<double> requested_left_adaptive_step_deg;
        std::optional<std::string> requested_left_correction_file;
        std::optional<std::string> requested_middle_correction_file;
        std::optional<std::string> requested_right_correction_file;
        std::optional<std::string> requested_output_dir;
        std::optional<double> requested_handoff_fraction;
        for (const auto& parameter : parameters) {
          if (parameter.get_name() == "cut_pattern") {
            const auto pattern = parameter.as_string();
            if (!isSupportedCutPattern(pattern)) {
              result.successful = false;
              result.reason = "unsupported cut_pattern";
              return result;
            }
            requested_pattern = pattern;
          } else if (parameter.get_name() == "motion_profile") {
            // Validated only: every planning request reads motion_profile at use time.
            if (!isSupportedMotionProfile(parameter.as_string())) {
              result.successful = false;
              result.reason =
                "unsupported motion_profile (ompl_cartesian, ompl_pilz, pilz_only, chomp_pilz)";
              return result;
            }
          } else if (parameter.get_name() == "t_cut_middle_handoff_fraction") {
            const double fraction = parameter.as_double();
            if (!isValidHandoffFraction(fraction)) {
              result.successful = false;
              result.reason = kHandoffFractionRule;
              return result;
            }
            requested_handoff_fraction = fraction;
          } else if (parameter.get_name() == "debug_fixed_orientation") {
            requested_debug_orientation = parameter.as_bool();
          } else if (parameter.get_name() == "debug_allow_stale_snapshot") {
            requested_debug_stale_snapshot = parameter.as_bool();
          } else if (parameter.get_name() == "pre_cut_stabilization_duration_s") {
            const double duration_s = parameter.as_double();
            if (!std::isfinite(duration_s) || duration_s < 0.0 || duration_s > 10.0) {
              result.successful = false;
              result.reason =
                "pre_cut_stabilization_duration_s must be finite and within [0,10] seconds";
              return result;
            }
            requested_pre_cut_dwell = duration_s;
          } else if (parameter.get_name() == "left_side_cut_tilt_deg" ||
                     parameter.get_name() == "right_side_cut_tilt_deg") {
            const double tilt = parameter.as_double();
            if (!std::isfinite(tilt) || std::abs(tilt) > 90.0) {
              result.successful = false;
              result.reason = "side cut tilt must be finite and within [-90,90] degrees";
              return result;
            }
            if (parameter.get_name() == "left_side_cut_tilt_deg") {
              requested_left_tilt = tilt;
            } else {
              requested_right_tilt = tilt;
            }
          } else if (parameter.get_name() == "cut_debug_cut") {
            const auto cut = parameter.as_string();
            if (cut != "left" && cut != "middle" && cut != "right") {
              result.successful = false;
              result.reason = "cut_debug_cut must be left, middle, or right";
              return result;
            }
            requested_debug_cut = cut;
          } else if (parameter.get_name() == "cut_debug_correction_enabled") {
            requested_correction_enabled = parameter.as_bool();
          } else if (parameter.get_name() == "left_adaptive_orientation_enabled") {
            requested_left_adaptive_enabled = parameter.as_bool();
          } else if (parameter.get_name() == "left_adaptive_orientation_max_deg") {
            const double value = parameter.as_double();
            if (!std::isfinite(value) || value <= 0.0 || value > 90.0) {
              result.successful = false;
              result.reason =
                "left_adaptive_orientation_max_deg must be finite and within (0,90] degrees";
              return result;
            }
            requested_left_adaptive_max_deg = value;
          } else if (parameter.get_name() == "left_adaptive_orientation_step_deg") {
            const double value = parameter.as_double();
            if (!std::isfinite(value) || value <= 0.0 || value > 90.0) {
              result.successful = false;
              result.reason =
                "left_adaptive_orientation_step_deg must be finite and within (0,90] degrees";
              return result;
            }
            requested_left_adaptive_step_deg = value;
          } else if (parameter.get_name() == "cut_debug_left_correction_file") {
            const auto file = parameter.as_string();
            if (!isAbsoluteJsonOrEmpty(file)) {
              result.successful = false;
              result.reason =
                "cut_debug_left_correction_file must be an absolute .json path or empty";
              return result;
            }
            requested_left_correction_file = file;
          } else if (parameter.get_name() == "cut_debug_middle_correction_file") {
            const auto file = parameter.as_string();
            if (!isAbsoluteJsonOrEmpty(file)) {
              result.successful = false;
              result.reason =
                "cut_debug_middle_correction_file must be an absolute .json path or empty";
              return result;
            }
            requested_middle_correction_file = file;
          } else if (parameter.get_name() == "cut_debug_right_correction_file") {
            const auto file = parameter.as_string();
            if (!isAbsoluteJsonOrEmpty(file)) {
              result.successful = false;
              result.reason =
                "cut_debug_right_correction_file must be an absolute .json path or empty";
              return result;
            }
            requested_right_correction_file = file;
          } else if (parameter.get_name() == "cut_debug_output_dir") {
            const auto dir = parameter.as_string();
            if (!dir.empty() && dir.front() != '/') {
              result.successful = false;
              result.reason = "cut_debug_output_dir must be an absolute path or empty";
              return result;
            }
            requested_output_dir = dir;
          }
        }
        if (!requested_pattern && !requested_debug_orientation &&
            !requested_debug_stale_snapshot && !requested_left_tilt &&
            !requested_right_tilt && !requested_pre_cut_dwell && !requested_debug_cut &&
            !requested_correction_enabled && !requested_left_adaptive_enabled &&
            !requested_left_adaptive_max_deg && !requested_left_adaptive_step_deg &&
            !requested_left_correction_file &&
            !requested_middle_correction_file && !requested_right_correction_file &&
            !requested_output_dir && !requested_handoff_fraction) {
          return result;
        }
        std::lock_guard<std::mutex> lock(plan_mutex_);
        if (execution_in_progress_) {
          result.successful = false;
          result.reason = "cannot change planning parameters during trajectory execution";
          return result;
        }
        const bool any_planning_mutation = requested_pattern || requested_debug_orientation ||
          requested_debug_stale_snapshot || requested_left_tilt || requested_right_tilt ||
          requested_pre_cut_dwell || requested_debug_cut || requested_correction_enabled ||
          requested_left_adaptive_enabled || requested_left_adaptive_max_deg ||
          requested_left_adaptive_step_deg || requested_left_correction_file ||
          requested_middle_correction_file || requested_right_correction_file ||
          requested_output_dir || requested_handoff_fraction;
        if (any_planning_mutation && cut_debug_stage_ != CutDebugStage::kNone) {
          result.successful = false;
          result.reason = "cannot change H-cut planning or cut-debug correction parameters while a cut debug session is active for cut " +
            cut_debug_session_cut_ + " stage " + cutDebugStageName(cut_debug_stage_) +
            "; call clear_plan first; execution_enabled and debug_motion_execution_enabled remain operable gates";
          return result;
        }
        if (requested_pattern) cut_pattern_ = *requested_pattern;
        if (requested_handoff_fraction) t_cut_middle_handoff_fraction_ = *requested_handoff_fraction;
        if (requested_debug_orientation) debug_fixed_orientation_ = *requested_debug_orientation;
        if (requested_debug_stale_snapshot) {
          debug_allow_stale_snapshot_ = *requested_debug_stale_snapshot;
        }
        if (requested_left_tilt) left_side_cut_tilt_deg_ = *requested_left_tilt;
        if (requested_right_tilt) right_side_cut_tilt_deg_ = *requested_right_tilt;
        if (requested_pre_cut_dwell) {
          pre_cut_stabilization_duration_s_ = *requested_pre_cut_dwell;
        }
        if (requested_debug_cut) cut_debug_cut_ = *requested_debug_cut;
        if (requested_correction_enabled) {
          cut_debug_correction_enabled_ = *requested_correction_enabled;
        }
        if (requested_left_adaptive_enabled) {
          left_adaptive_orientation_enabled_ = *requested_left_adaptive_enabled;
        }
        if (requested_left_adaptive_max_deg || requested_left_adaptive_step_deg) {
          const double next_max = requested_left_adaptive_max_deg ?
            *requested_left_adaptive_max_deg : left_adaptive_orientation_max_deg_;
          const double next_step = requested_left_adaptive_step_deg ?
            *requested_left_adaptive_step_deg : left_adaptive_orientation_step_deg_;
          if (!std::isfinite(next_max) || next_max <= 0.0 || next_max > 90.0 ||
              !std::isfinite(next_step) || next_step <= 0.0 || next_step > 90.0 ||
              next_step > next_max) {
            result.successful = false;
            result.reason =
              "left adaptive orientation bounds must satisfy 0 < step <= max <= 90 degrees";
            return result;
          }
          left_adaptive_orientation_max_deg_ = next_max;
          left_adaptive_orientation_step_deg_ = next_step;
        }
        if (requested_left_correction_file) {
          cut_debug_left_correction_file_ = *requested_left_correction_file;
        }
        if (requested_middle_correction_file) {
          cut_debug_middle_correction_file_ = *requested_middle_correction_file;
        }
        if (requested_right_correction_file) {
          cut_debug_right_correction_file_ = *requested_right_correction_file;
        }
        if (requested_output_dir) cut_debug_output_dir_ = *requested_output_dir;
        clearRetainedPlan();
        return result;
      });

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      [this](sensor_msgs::msg::JointState::SharedPtr message) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_joint_state_ = *message;
        joint_state_received_ = now();
      });
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
      });
    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
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
        if (!lock.owns_lock() || execution_in_progress_) {
          response->success = false;
          response->message = "cannot plan while another planning or execution operation is active";
          return;
        }
        if (cut_debug_stage_ != CutDebugStage::kNone) {
          response->success = false;
          response->message = "cut debug session active for cut " + cut_debug_session_cut_ +
            " stage " + cutDebugStageName(cut_debug_stage_) +
            "; call clear_plan before planning a normal H-cut";
          return;
        }
        clearRetainedPlan();
        response->success = planExternalSnapshot(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
    cut_debug_start_service_ = create_service<std_srvs::srv::Trigger>(
      "~/start_cut_debug_session",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || execution_in_progress_) {
          response->success = false;
          response->message = "cannot start cut debug session while another planning or execution operation is active";
          return;
        }
        response->success = startCutDebugSession(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
    cut_debug_save_service_ = create_service<std_srvs::srv::Trigger>(
      "~/save_cut_debug_point",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || execution_in_progress_) {
          response->success = false;
          response->message = "cannot save cut debug point while another planning or execution operation is active";
          return;
        }
        response->success = saveCutDebugPoint(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
    section_diagnostic_service_ = create_service<std_srvs::srv::Trigger>(
      "~/diagnose_external_snapshot_sections",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || execution_in_progress_) {
          response->success = false;
          response->message = "cannot diagnose while another planning or execution operation is active";
          return;
        }
        if (cut_debug_stage_ != CutDebugStage::kNone) {
          response->success = false;
          response->message = "cut debug session active for cut " + cut_debug_session_cut_ +
            "; call clear_plan before running section diagnostics";
          return;
        }
        response->success = diagnoseExternalSnapshotSections(response->message);
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
        if (debug_fixed_orientation_) {
          response->success = false;
          response->message = "execution is blocked while debug_fixed_orientation is enabled";
          return;
        }
        const bool in_debug_session = cut_debug_stage_ != CutDebugStage::kNone;
        if (in_debug_session &&
            !get_parameter("debug_motion_execution_enabled").as_bool()) {
          response->success = false;
          response->message =
            "cut debug session execution requires both execution_enabled and debug_motion_execution_enabled";
          return;
        }
        if (!in_debug_session && cut_pattern_ == "middle_debug_short" &&
            !get_parameter("debug_motion_execution_enabled").as_bool()) {
          response->success = false;
          response->message =
            "debug trajectory execution is disabled by debug_motion_execution_enabled";
          return;
        }
        if (execution_in_progress_) {
          response->success = false;
          response->message = "an H-cut trajectory execution is already active";
          return;
        }
        if (in_debug_session && cut_debug_stage_ != CutDebugStage::kStartPlanned) {
          response->success = false;
          if (cut_debug_stage_ == CutDebugStage::kStartExecuted) {
            response->message = "cut debug start already executed for cut " +
              cut_debug_session_cut_ +
              "; teleop correct the TCP then call save_cut_debug_point before any further execution";
          } else if (cut_debug_stage_ == CutDebugStage::kEndExecuted) {
            response->message = "cut debug end already executed for cut " +
              cut_debug_session_cut_ +
              "; teleop correct the TCP then call save_cut_debug_point before any further execution";
          } else {
            response->message = "no executable cut debug plan is retained for cut " +
              cut_debug_session_cut_ + " stage " + cutDebugStageName(cut_debug_stage_);
          }
          return;
        }
        if (in_debug_session && cut_debug_dispatch_failed_) {
          response->success = false;
          response->message = "previous cut debug " + cutDebugStageName(cut_debug_stage_) +
            " dispatch for cut " + cut_debug_session_cut_ +
            " failed during execution and was NOT marked executed; no retained plan remains; call clear_plan then start_cut_debug_session to restart";
          return;
        }
        if (!planned_task_ || !planned_snapshot_ || planned_task_->solutions().empty()) {
          response->success = false;
          response->message = "no valid planned H-cut is available to execute";
          return;
        }
        // Normal retained plans refuse a stale snapshot so a previous carton's result
        // can never dispatch. A staged session instead freezes its snapshot at session
        // start and each stage is planned from the live state seconds before its
        // explicit execution; teleop correction inherently takes longer than the
        // snapshot age bound, so the age check does not apply while the session owns
        // the retained plan. Both execution gates still apply to every stage.
        if (!in_debug_session) {
          std::string freshness_error;
          if (!snapshotIsFresh(*planned_snapshot_, freshness_error)) {
            clearRetainedPlan();
            response->success = false;
            response->message = freshness_error + "; retained plan was cleared";
            return;
          }
        }
        if (execution_thread_.joinable()) execution_thread_.join();
        execution_in_progress_ = true;
        const CutDebugStage dispatched_stage = cut_debug_stage_;
        const std::string dispatched_cut = cut_debug_session_cut_;
        auto task = std::move(planned_task_);
        planned_snapshot_.reset();
        execution_thread_ = std::thread(
          [this, task = std::move(task), dispatched_stage, dispatched_cut]() mutable {
          bool succeeded = false;
          std::string failure_detail;
          try {
            const auto result = task->execute(*task->solutions().front());
            succeeded = (result == moveit::core::MoveItErrorCode::SUCCESS);
            if (succeeded) {
              RCLCPP_INFO(get_logger(), "H-cut trajectory execution completed");
            } else {
              failure_detail =
                "controller reported failure status " + std::to_string(result.val);
              RCLCPP_ERROR(
                get_logger(), "H-cut trajectory execution failed: %s",
                failure_detail.c_str());
            }
          } catch (const std::exception& exception) {
            failure_detail = std::string("threw exception: ") + exception.what();
            RCLCPP_ERROR(
              get_logger(), "H-cut trajectory execution %s", failure_detail.c_str());
          }
          std::lock_guard<std::mutex> completion_lock(plan_mutex_);
          execution_in_progress_ = false;
          if (dispatched_stage == CutDebugStage::kNone) return;
          if (!succeeded) {
            // Never mark the stage executed: the arm did not reach the stage pose,
            // so capturing an observation here would persist a false correction.
            // The dispatched task was consumed, so nothing executable remains.
            cut_debug_dispatch_failed_ = true;
            RCLCPP_ERROR(
              get_logger(),
              "Cut debug %s dispatch for cut %s did NOT complete (%s); the stage was NOT marked executed and no plan is retained; call clear_plan then start_cut_debug_session to restart",
              cutDebugStageName(dispatched_stage).c_str(), dispatched_cut.c_str(),
              failure_detail.c_str());
            return;
          }
          cut_debug_dispatch_failed_ = false;
          // The staged session retains only a start plan; every later stage is teleop.
          if (dispatched_stage == CutDebugStage::kStartPlanned) {
            cut_debug_stage_ = CutDebugStage::kStartExecuted;
          }
        });
        if (in_debug_session) {
          response->success = true;
          response->message = "cut debug start dispatch accepted for cut " + dispatched_cut +
            "; monitor controller execution, then teleop correct and call save_cut_debug_point; no further motion is automatic";
        } else {
          response->success = true;
          response->message = "H-cut trajectory dispatch accepted; monitor controller execution";
        }
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
        clearRetainedPlan();
        clearCutDebugSessionState();
        response->success = true;
        response->message = "H-cut plan, cut debug session, and RViz outputs cleared";
      },
      rmw_qos_profile_services_default, callback_group_);
    RCLCPP_INFO(
      get_logger(),
      "Snapshot H-cut MTC planner ready; hover %.0f mm, cut %.0f mm, pre-cut stabilization %.3f s",
      hover_clearance_m_ * 1000.0, cut_clearance_m_ * 1000.0,
      pre_cut_stabilization_duration_s_);
  }

  ~BoxHCutExecutor() override
  {
    if (execution_thread_.joinable()) execution_thread_.join();
  }

private:
  void clearRetainedPlan()
  {
    planned_task_.reset();
    planned_snapshot_.reset();
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_all);
    marker_pub_->publish(markers);
    geometry_msgs::msg::PoseArray empty_targets;
    empty_targets.header.frame_id = planning_frame_;
    target_pose_pub_->publish(empty_targets);
    moveit_msgs::msg::CollisionObject remove_carton;
    remove_carton.id = carton_collision_id_;
    remove_carton.header.frame_id = planning_frame_;
    remove_carton.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    carton_collision_pub_->publish(remove_carton);
  }

  void clearCutDebugSessionState()
  {
    cut_debug_stage_ = CutDebugStage::kNone;
    cut_debug_session_cut_.clear();
    cut_debug_snapshot_.reset();
    cut_debug_robot_model_.reset();
    cut_debug_has_observed_start_ = false;
    cut_debug_has_observed_end_ = false;
    cut_debug_dispatch_failed_ = false;
    cut_debug_record_path_.clear();
    cut_debug_created_stamp_ns_ = 0;
  }

  bool planExternalSnapshot(std::string& message)
  {
    const auto source_path = get_parameter("external_snapshot_file").as_string();
    const auto snapshot = loadExternalSnapshot(source_path, message);
    if (!snapshot) return false;
    const bool planned = planSnapshot(*snapshot, message);
    message = "external snapshot " + source_path + ": " + message;
    return planned;
  }

  static const char* cutDebugCorrectionSchema() { return "box_h_cut_correction/v1"; }
  static constexpr const char* kHandoffFractionRule =
    "t_cut_middle_handoff_fraction must be finite and either exactly 0.0 (automatic) or within [0.05,0.95]";
  static bool isValidHandoffFraction(double fraction)
  {
    return std::isfinite(fraction) && (fraction == 0.0 || (fraction >= 0.05 && fraction <= 0.95));
  }
  static bool isSupportedCutPattern(const std::string& pattern)
  {
    static const std::array<const char*, 9> patterns{
      "h", "t_cut", "t_cut_handoff", "combined", "left_only", "middle_only", "right_only",
      "middle_entry_debug", "middle_debug_short"};
    return std::any_of(patterns.begin(), patterns.end(), [&pattern](const char* candidate) {
      return pattern == candidate;
    });
  }
  static bool isSupportedMotionProfile(const std::string& profile)
  {
    static const std::array<const char*, 4> profiles{
      "ompl_cartesian", "ompl_pilz", "pilz_only", "chomp_pilz"};
    return std::any_of(profiles.begin(), profiles.end(), [&profile](const char* candidate) {
      return profile == candidate;
    });
  }
  static bool isAbsoluteJsonOrEmpty(const std::string& path)
  {
    return path.empty() || (path.front() == '/' && path.size() >= 6U &&
                            path.substr(path.size() - 5U) == ".json");
  }
  const std::string& correctionFileForCut(const std::string& cut) const
  {
    if (cut == "left") return cut_debug_left_correction_file_;
    if (cut == "right") return cut_debug_right_correction_file_;
    return cut_debug_middle_correction_file_;
  }
  std::string cutDebugStageName(CutDebugStage stage) const
  {
    switch (stage) {
      case CutDebugStage::kNone: return "idle";
      case CutDebugStage::kStartPlanned: return "start_planned";
      case CutDebugStage::kStartExecuted: return "start_executed";
      case CutDebugStage::kEndExecuted: return "end_executed";
    }
    return "unknown";
  }
  bool cutDebugEndpoints(
    const Snapshot& snapshot, const std::string& cut,
    geometry_msgs::msg::Point& start, geometry_msgs::msg::Point& end) const
  {
    if (cut == "left") {
      start = snapshot.back_middle_left;
      end = snapshot.front_middle_left;
      return true;
    }
    if (cut == "middle") {
      start = snapshot.seam_start;
      end = snapshot.seam_end;
      return true;
    }
    if (cut == "right") {
      start = snapshot.back_middle_right;
      end = snapshot.front_middle_right;
      return true;
    }
    return false;
  }
  bool cutDebugBasis(
    const Snapshot& snapshot, const std::string& cut,
    std::array<double, 3>& along, std::array<double, 3>& across,
    std::array<double, 3>& normal, std::string& error) const
  {
    geometry_msgs::msg::Point start;
    geometry_msgs::msg::Point end;
    if (!cutDebugEndpoints(snapshot, cut, start, end)) {
      error = "unknown cut debug cut: " + cut;
      return false;
    }
    normal = {snapshot.surface_normal.x, snapshot.surface_normal.y, snapshot.surface_normal.z};
    const double normal_length =
      std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (!std::isfinite(normal_length) || normal_length <= 1e-9) {
      error = "snapshot surface normal has zero length";
      return false;
    }
    for (auto& component : normal) component /= normal_length;
    std::array<double, 3> axis{
      end.x - start.x, end.y - start.y, end.z - start.z};
    const double projection = axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2];
    for (std::size_t i = 0; i < 3; ++i) axis[i] -= projection * normal[i];
    const double axis_length = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (!std::isfinite(axis_length) || axis_length < 1e-6) {
      error = "cut debug cut has degenerate length for cut " + cut;
      return false;
    }
    for (auto& component : axis) component /= axis_length;
    along = axis;
    across = {
      normal[1] * along[2] - normal[2] * along[1],
      normal[2] * along[0] - normal[0] * along[2],
      normal[0] * along[1] - normal[1] * along[0]};
    return true;
  }
  bool offsetLeftPoseOutward(
    const Snapshot& snapshot, geometry_msgs::msg::PoseStamped& pose, double distance_m,
    std::string& error) const
  {
    std::array<double, 3> along{};
    std::array<double, 3> across{};
    std::array<double, 3> normal{};
    if (!cutDebugBasis(snapshot, "left", along, across, normal, error)) return false;
    const std::array<double, 3> midpoint{
      0.5 * (snapshot.back_middle_left.x + snapshot.front_middle_left.x),
      0.5 * (snapshot.back_middle_left.y + snapshot.front_middle_left.y),
      0.5 * (snapshot.back_middle_left.z + snapshot.front_middle_left.z)};
    const std::array<double, 3> toward_center{
      snapshot.seam_center.x - midpoint[0], snapshot.seam_center.y - midpoint[1],
      snapshot.seam_center.z - midpoint[2]};
    const double projection = across[0] * toward_center[0] +
      across[1] * toward_center[1] + across[2] * toward_center[2];
    if (!std::isfinite(projection) || std::abs(projection) < 1e-6) {
      error = "cannot determine outward side of left cut from carton geometry";
      return false;
    }
    const double sign = projection > 0.0 ? -1.0 : 1.0;
    pose.pose.position.x += sign * distance_m * across[0];
    pose.pose.position.y += sign * distance_m * across[1];
    pose.pose.position.z += sign * distance_m * across[2];
    return true;
  }
  Eigen::Quaterniond cutDebugBasisQuaternion(
    const std::array<double, 3>& along, const std::array<double, 3>& across,
    const std::array<double, 3>& normal) const
  {
    Eigen::Matrix3d basis;
    basis.col(0) = Eigen::Vector3d(along[0], along[1], along[2]);
    basis.col(1) = Eigen::Vector3d(across[0], across[1], across[2]);
    basis.col(2) = Eigen::Vector3d(normal[0], normal[1], normal[2]);
    return Eigen::Quaterniond(basis);
  }
  bool loadCutDebugCorrectionFile(
    const std::string& path, CutDebugCorrection& correction, std::string& error) const
  {
    namespace fs = std::filesystem;
    if (path.empty() || path.front() != '/' || path.size() < 6U ||
        path.substr(path.size() - 5U) != ".json") {
      error = "per-cut correction file path must be an absolute .json path";
      return false;
    }
    std::error_code code;
    if (!fs::is_regular_file(path, code) || code) {
      error = "cut debug correction file is unavailable: " + path;
      return false;
    }
    std::ifstream input(path);
    if (!input) {
      error = "cannot read cut debug correction file: " + path;
      return false;
    }
    using boost::property_tree::ptree;
    ptree document;
    try {
      boost::property_tree::read_json(input, document);
    } catch (const std::exception& exception) {
      error = "cut debug correction file is not valid JSON: " + std::string(exception.what());
      return false;
    }
    const auto schema = document.get_optional<std::string>("schema");
    const auto cut = document.get_optional<std::string>("cut");
    if (!schema || *schema != cutDebugCorrectionSchema()) {
      error = "cut debug correction schema mismatch; expected " +
        std::string(cutDebugCorrectionSchema());
      return false;
    }
    if (!cut || (*cut != "left" && *cut != "middle" && *cut != "right")) {
      error = "cut debug correction file has an invalid cut field";
      return false;
    }
    const auto frame = document.get_optional<std::string>("planning_frame");
    if (frame && *frame != planning_frame_) {
      error = "cut debug correction planning_frame mismatch: file=" + *frame +
        " active=" + planning_frame_;
      return false;
    }
    const auto tool = document.get_optional<std::string>("tool_link");
    if (tool && !tool->empty() && *tool != tool_link_) {
      error = "cut debug correction tool_link mismatch: file=" + *tool + " active=" + tool_link_;
      return false;
    }
    const auto status = document.get_optional<std::string>("status");
    if (!status || *status != "complete") {
      error =
        "cut debug correction file is not a complete start+end record; finish the staged session before opting in";
      return false;
    }
    // The local orientation delta is schema-validated (finite, unit length) but not
    // stored: planning maps the full observed quaternions instead.
    const auto read_local = [&document, &error](
      const std::string& prefix, CutLocalDeltas& deltas) {
      const auto along = document.get_optional<double>(prefix + "_along_m");
      const auto across = document.get_optional<double>(prefix + "_across_m");
      const auto normal = document.get_optional<double>(prefix + "_normal_m");
      const auto qx = document.get_optional<double>(prefix + "_orientation_delta_local_xyzw.x");
      const auto qy = document.get_optional<double>(prefix + "_orientation_delta_local_xyzw.y");
      const auto qz = document.get_optional<double>(prefix + "_orientation_delta_local_xyzw.z");
      const auto qw = document.get_optional<double>(prefix + "_orientation_delta_local_xyzw.w");
      if (!along || !across || !normal || !qx || !qy || !qz || !qw) {
        error = "cut debug correction file is missing " + prefix + " fields";
        return false;
      }
      if (!std::isfinite(*along) || !std::isfinite(*across) || !std::isfinite(*normal) ||
          !std::isfinite(*qx) || !std::isfinite(*qy) || !std::isfinite(*qz) ||
          !std::isfinite(*qw)) {
        error = "cut debug correction file has non-finite " + prefix + " values";
        return false;
      }
      const double norm = std::sqrt(*qx * *qx + *qy * *qy + *qz * *qz + *qw * *qw);
      if (!std::isfinite(norm) || std::abs(norm - 1.0) > 0.01) {
        error = "cut debug correction " + prefix + " local orientation is not unit length";
        return false;
      }
      deltas.along_m = *along;
      deltas.across_m = *across;
      deltas.normal_m = *normal;
      return true;
    };
    const auto read_array3 = [&document, &error](
      const std::string& field, std::array<double, 3>& values) {
      const auto child = document.get_child_optional(field);
      if (!child || child->size() != values.size()) {
        error = "cut debug correction file is missing three-element " + field;
        return false;
      }
      std::size_t index = 0;
      try {
        for (const auto& entry : *child) values[index++] = entry.second.get_value<double>();
      } catch (const std::exception&) {
        error = "cut debug correction file has invalid " + field;
        return false;
      }
      if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
          })) {
        error = "cut debug correction file has non-finite " + field;
        return false;
      }
      return true;
    };
    const auto read_observed_orientation = [&document, &error](
      const std::string& field, geometry_msgs::msg::Quaternion& orientation) {
      const auto child = document.get_child_optional(field);
      if (!child || child->size() != 4U) {
        error = "cut debug correction file is missing four-element " + field;
        return false;
      }
      std::array<double, 4> values{};
      std::size_t index = 0;
      try {
        for (const auto& entry : *child) values[index++] = entry.second.get_value<double>();
      } catch (const std::exception&) {
        error = "cut debug correction file has invalid " + field;
        return false;
      }
      const double norm = std::sqrt(
        values[0] * values[0] + values[1] * values[1] +
        values[2] * values[2] + values[3] * values[3]);
      if (!std::isfinite(norm) || std::abs(norm - 1.0) > 0.01) {
        error = "cut debug correction " + field + " is not unit length";
        return false;
      }
      orientation.x = values[0] / norm;
      orientation.y = values[1] / norm;
      orientation.z = values[2] / norm;
      orientation.w = values[3] / norm;
      return true;
    };
    CutDebugCorrection loaded;
    loaded.cut = *cut;
    if (!read_array3("cut_basis_along_m", loaded.recorded_along) ||
        !read_array3("cut_basis_across_m", loaded.recorded_across) ||
        !read_array3("cut_basis_normal_m", loaded.recorded_normal) ||
        !read_observed_orientation("observed_start_orientation_xyzw", loaded.observed_start) ||
        !read_observed_orientation("observed_end_orientation_xyzw", loaded.observed_end)) {
      return false;
    }
    const Eigen::Vector3d recorded_along(
      loaded.recorded_along[0], loaded.recorded_along[1], loaded.recorded_along[2]);
    const Eigen::Vector3d recorded_across(
      loaded.recorded_across[0], loaded.recorded_across[1], loaded.recorded_across[2]);
    const Eigen::Vector3d recorded_normal(
      loaded.recorded_normal[0], loaded.recorded_normal[1], loaded.recorded_normal[2]);
    if (std::abs(recorded_along.norm() - 1.0) > 0.01 ||
        std::abs(recorded_across.norm() - 1.0) > 0.01 ||
        std::abs(recorded_normal.norm() - 1.0) > 0.01 ||
        std::abs(recorded_along.dot(recorded_across)) > 0.01 ||
        std::abs(recorded_along.dot(recorded_normal)) > 0.01 ||
        std::abs(recorded_across.dot(recorded_normal)) > 0.01 ||
        recorded_along.cross(recorded_across).dot(recorded_normal) < 0.99) {
      error = "cut debug correction recorded cut basis is not orthonormal";
      return false;
    }
    if (!read_local("start_delta_cut_local", loaded.start_local)) {
      return false;
    }
    const auto has_end = document.get_optional<double>("end_delta_cut_local_along_m");
    if (has_end) {
      if (!read_local("end_delta_cut_local", loaded.end_local)) {
        return false;
      }
    } else {
      loaded.end_local = loaded.start_local;
    }
    correction = loaded;
    return true;
  }
  // Loads each configured correction slot for `cuts` exactly once per planning request;
  // the resulting records are passed down to every planned segment. Empty when
  // cut_debug_correction_enabled is false. Schema/slot mismatches throw (fail closed).
  // `require_any`: whole-plan requests refuse to run opted-in without any record.
  std::map<std::string, CutDebugCorrection> loadActiveCorrections(
    const std::vector<std::string>& cuts, bool require_any) const
  {
    std::map<std::string, CutDebugCorrection> active;
    if (!cut_debug_correction_enabled_) return active;
    for (const auto& cut : cuts) {
      const std::string& correction_path = correctionFileForCut(cut);
      if (correction_path.empty()) continue;
      CutDebugCorrection loaded;
      std::string correction_error;
      if (!loadCutDebugCorrectionFile(correction_path, loaded, correction_error)) {
        throw std::runtime_error("cut debug correction rejected: " + correction_error);
      }
      if (loaded.cut != cut) {
        throw std::runtime_error(
          "cut debug correction record is for cut " + loaded.cut + " but was configured in cut_debug_" +
          cut + "_correction_file; refusing to silently apply a mismatched correction");
      }
      active.emplace(cut, loaded);
    }
    if (require_any && active.empty()) {
      throw std::runtime_error(
        "cut_debug_correction_enabled is true but no per-cut correction file is set; refusing to plan without an explicit correction record");
    }
    return active;
  }
  std::map<std::string, CutDebugCorrection> loadAllActiveCorrections() const
  {
    return loadActiveCorrections({"left", "middle", "right"}, true);
  }
  static geometry_msgs::msg::Point applyCutLocalDelta(
    geometry_msgs::msg::Point point, const std::array<double, 3>& along,
    const std::array<double, 3>& across, const std::array<double, 3>& normal,
    const CutLocalDeltas& local)
  {
    point.x += along[0] * local.along_m + across[0] * local.across_m + normal[0] * local.normal_m;
    point.y += along[1] * local.along_m + across[1] * local.across_m + normal[1] * local.normal_m;
    point.z += along[2] * local.along_m + across[2] * local.across_m + normal[2] * local.normal_m;
    return point;
  }
  // Nominal blade-direction targets for a cut: the blade (tool Y) follows the cut
  // segment, or the middle seam for side cuts in middle_aligned mode. Side cuts face
  // 180 degrees away from travel (facing sign -1). Returns {start target, end target}.
  std::pair<geometry_msgs::msg::Point, geometry_msgs::msg::Point> orientationTargets(
    const Snapshot& snapshot, bool is_middle, const geometry_msgs::msg::Point& segment_start,
    const geometry_msgs::msg::Point& segment_end, const geometry_msgs::msg::Point& start,
    const geometry_msgs::msg::Point& end) const
  {
    const double facing_sign = is_middle ? 1.0 : -1.0;
    const bool use_seam = !is_middle && side_cut_orientation_mode_ == "middle_aligned";
    const auto& from = use_seam ? snapshot.seam_start : segment_start;
    const auto& to = use_seam ? snapshot.seam_end : segment_end;
    const double dx = to.x - from.x;
    const double dy = to.y - from.y;
    const double dz = to.z - from.z;
    auto start_target = start;
    start_target.x += facing_sign * dx;
    start_target.y += facing_sign * dy;
    start_target.z += facing_sign * dz;
    auto end_target = end;
    end_target.x += facing_sign * dx;
    end_target.y += facing_sign * dy;
    end_target.z += facing_sign * dz;
    return {start_target, end_target};
  }
  geometry_msgs::msg::Quaternion interpolateTaughtOrientation(
    const geometry_msgs::msg::Quaternion& start,
    const geometry_msgs::msg::Quaternion& end, double fraction) const
  {
    Eigen::Quaterniond start_q(start.w, start.x, start.y, start.z);
    Eigen::Quaterniond end_q(end.w, end.x, end.y, end.z);
    if (start_q.dot(end_q) < 0.0) end_q.coeffs() *= -1.0;
    const Eigen::Quaterniond blended = start_q.slerp(fraction, end_q).normalized();
    geometry_msgs::msg::Quaternion result;
    result.x = blended.x();
    result.y = blended.y();
    result.z = blended.z();
    result.w = blended.w();
    return result;
  }
  bool applyCutDebugObservedOrientation(
    const Snapshot& snapshot, const std::string& cut, const CutDebugCorrection& correction,
    const geometry_msgs::msg::Quaternion& observed, geometry_msgs::msg::PoseStamped& pose,
    std::string& error) const
  {
    std::array<double, 3> current_along{};
    std::array<double, 3> current_across{};
    std::array<double, 3> current_normal{};
    if (!cutDebugBasis(
        snapshot, cut, current_along, current_across, current_normal, error)) {
      return false;
    }
    const Eigen::Quaterniond current_basis =
      cutDebugBasisQuaternion(current_along, current_across, current_normal).normalized();
    const Eigen::Quaterniond recorded_basis = cutDebugBasisQuaternion(
      correction.recorded_along, correction.recorded_across, correction.recorded_normal).normalized();
    const Eigen::Quaterniond recorded_observed(
      observed.w, observed.x, observed.y, observed.z);
    const Eigen::Quaterniond corrected =
      (current_basis * recorded_basis.inverse() * recorded_observed).normalized();
    const Eigen::Vector3d into_surface(
      -snapshot.surface_normal.x, -snapshot.surface_normal.y, -snapshot.surface_normal.z);
    const Eigen::Vector3d into_unit = into_surface.normalized();
    if (cut == "left") {
      const double dot_x = (corrected * Eigen::Vector3d::UnitX()).dot(into_unit);
      const double dot_z = (corrected * Eigen::Vector3d::UnitZ()).dot(into_unit);
      if (dot_x < 0.7 && dot_z < 0.7) {
        std::ostringstream detail;
        detail << std::fixed << std::setprecision(4);
        detail << "corrected left tool X/Z penetration axes no longer point into the "
          "carton surface (x=" << dot_x << ", z=" << dot_z <<
          "); refusing to apply correction";
        error = detail.str();
        return false;
      }
    } else {
      const Eigen::Vector3d penetration_axis = corrected * Eigen::Vector3d::UnitZ();
      if (penetration_axis.dot(into_unit) < 0.7) {
        error = "corrected " + cut + " tool penetration axis no longer points into the "
          "carton surface; refusing to apply correction";
        return false;
      }
    }
    pose.pose.orientation.x = corrected.x();
    pose.pose.orientation.y = corrected.y();
    pose.pose.orientation.z = corrected.z();
    pose.pose.orientation.w = corrected.w();
    return true;
  }
  std::vector<double> leftAdaptiveOrientationCandidates() const
  {
    // Deterministic smallest-magnitude order: 0, +step, -step, ..., ±max.
    std::vector<double> candidates{0.0};
    const double max_deg = left_adaptive_orientation_max_deg_;
    const double step_deg = left_adaptive_orientation_step_deg_;
    const auto magnitudes =
      static_cast<std::size_t>(std::floor(max_deg / step_deg + 1e-9));
    for (std::size_t multiple = 1; multiple <= magnitudes; ++multiple) {
      const double magnitude = static_cast<double>(multiple) * step_deg;
      candidates.push_back(magnitude);
      candidates.push_back(-magnitude);
    }
    if (static_cast<double>(magnitudes) * step_deg < max_deg - 1e-9) {
      candidates.push_back(max_deg);
      candidates.push_back(-max_deg);
    }
    return candidates;
  }
  Eigen::Quaterniond rollLeftOrientationAboutToolY(
    const Eigen::Quaterniond& mapped, double theta_deg) const
  {
    // TCP-local Y is the blade-length/seam axis: post-multiplying preserves the
    // blade Y world direction and the TCP position; only the X/Z penetration
    // roll changes.
    const double radians = theta_deg * std::acos(-1.0) / 180.0;
    return (mapped * Eigen::AngleAxisd(radians, Eigen::Vector3d::UnitY())).normalized();
  }
  bool adaptLeftCorrectedOrientations(
    const Snapshot& snapshot,
    const moveit::core::JointModelGroup* joint_group,
    const moveit::core::RobotState& seed_state,
    const std::vector<std::pair<geometry_msgs::msg::PoseStamped*, const char*> >& labeled_poses,
    const std::string& pose_context, std::string& error)
  {
    // Match the objects that the task adds before its left approach. Kinematic
    // IK alone can select a pose with no collision-free goal state.
    planning_scene::PlanningScene scene(seed_state.getRobotModel());
    if (!scene.processCollisionObjectMsg(cartonCollisionObject(snapshot))) {
      error = "cannot add carton to left orientation collision probe";
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(camera_pole_mutex_);
      if (camera_pole_) {
        if (!scene.processCollisionObjectMsg(*camera_pole_)) {
          error = "cannot add camera pole to left orientation collision probe";
          return false;
        }
      } else if (camera_pole_required_) {
        error = "camera pole collision object is unavailable for left orientation probe";
        return false;
      }
    }
    std::vector<Eigen::Quaterniond> mapped;
    mapped.reserve(labeled_poses.size());
    for (const auto& labeled : labeled_poses) {
      const auto& orientation = labeled.first->pose.orientation;
      mapped.emplace_back(orientation.w, orientation.x, orientation.y, orientation.z);
    }
    const Eigen::Vector3d into_surface(
      -snapshot.surface_normal.x, -snapshot.surface_normal.y,
      -snapshot.surface_normal.z);
    const Eigen::Vector3d into_unit = into_surface.normalized();
    const std::vector<double> candidates = leftAdaptiveOrientationCandidates();
    const bool detailed_diagnostics = get_parameter("left_roll_diagnostics_enabled").as_bool();
    std::string first_failure;
    std::vector<std::string> failing_categories;
    const auto record_category = [&failing_categories](const char* label) {
        const std::string name(label);
        if (std::find(failing_categories.begin(), failing_categories.end(), name) ==
            failing_categories.end()) {
          failing_categories.push_back(name);
        }
      };
    const auto apply = [&](std::size_t index, const Eigen::Quaterniond& orientation) {
        labeled_poses[index].first->pose.orientation.x = orientation.x();
        labeled_poses[index].first->pose.orientation.y = orientation.y();
        labeled_poses[index].first->pose.orientation.z = orientation.z();
        labeled_poses[index].first->pose.orientation.w = orientation.w();
      };
    const moveit::core::GroupStateValidityCallbackFn valid_state =
      [this, &scene](moveit::core::RobotState* state,
                     const moveit::core::JointModelGroup* group, const double* values) {
        state->setJointGroupPositions(group, values);
        state->update();
        return !scene.isStateColliding(*state, planning_group_);
      };
    // Diagnostic only (left_roll_diagnostics_enabled): explains why a candidate pose
    // failed without mutating the labeled poses or influencing candidate selection. Every IK probe here runs on
    // throwaway states; the ACM entry is set exactly as the search loop sets it.
    const auto diagnose_pose = [&](std::size_t index, const Eigen::Quaterniond& orientation) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(4);
        const double dot_x = (orientation * Eigen::Vector3d::UnitX()).dot(into_unit);
        const double dot_z = (orientation * Eigen::Vector3d::UnitZ()).dot(into_unit);
        out << "gate x=" << dot_x << " z=" << dot_z << " q_xyzw=(" << orientation.x() << "," <<
          orientation.y() << "," << orientation.z() << "," << orientation.w() << ")";
        if (dot_x < 0.7 && dot_z < 0.7) {
          out << " FAIL(penetration gate)";
          return out.str();
        }
        geometry_msgs::msg::Pose target = labeled_poses[index].first->pose;
        target.orientation.x = orientation.x();
        target.orientation.y = orientation.y();
        target.orientation.z = orientation.z();
        target.orientation.w = orientation.w();
        const bool contact_allowed = index % 2U != 0U && allow_tool_carton_contact_;
        scene.getAllowedCollisionMatrixNonConst().setEntry(
          tool_link_, carton_collision_id_, contact_allowed);
        out << " pos=(" << target.position.x << "," << target.position.y << "," <<
          target.position.z << ") tool/carton_allowed=" << (contact_allowed ? 1 : 0);
        moveit::core::RobotState aware(seed_state);
        if (aware.setFromIK(joint_group, target, tool_link_, 0.25, valid_state)) {
          out << " collision-aware IK OK";
          return out.str();
        }
        // Kinematic-only IK from the seed plus random restarts: separates
        // "unreachable" from "reachable but every found branch collides".
        constexpr int kRandomAttempts = 24;
        int kinematic_found = 0;
        int colliding = 0;
        int out_of_bounds = 0;
        int collision_free = 0;
        double best_depth = std::numeric_limits<double>::infinity();
        std::string best_contacts;
        std::vector<double> best_joints;
        std::map<std::string, int> pair_counts;
        for (int attempt = 0; attempt <= kRandomAttempts; ++attempt) {
          moveit::core::RobotState kin(seed_state);
          if (attempt > 0) {
            kin.setToRandomPositions(joint_group);
            kin.update();
          }
          if (!kin.setFromIK(joint_group, target, tool_link_, 0.1)) continue;
          kin.update();
          ++kinematic_found;
          if (!kin.satisfiesBounds(joint_group)) {
            ++out_of_bounds;
            continue;
          }
          collision_detection::CollisionRequest request;
          request.group_name = planning_group_;
          request.contacts = true;
          request.max_contacts = 50;
          request.max_contacts_per_pair = 1;
          collision_detection::CollisionResult result;
          scene.checkCollision(request, result, kin);
          if (!result.collision) {
            ++collision_free;
            continue;
          }
          ++colliding;
          double max_depth = 0.0;
          std::ostringstream pairs;
          pairs << std::fixed << std::setprecision(4);
          bool first = true;
          for (const auto& entry : result.contacts) {
            double depth = 0.0;
            for (const auto& contact : entry.second) depth = std::max(depth, contact.depth);
            max_depth = std::max(max_depth, depth);
            const std::string pair = entry.first.first + "<->" + entry.first.second;
            ++pair_counts[pair];
            if (!first) pairs << ", ";
            first = false;
            pairs << pair << " depth=" << depth;
          }
          if (max_depth < best_depth) {
            best_depth = max_depth;
            best_contacts = pairs.str();
            kin.copyJointGroupPositions(joint_group, best_joints);
          }
        }
        out << " FAIL(collision-aware IK): kinematic IK " << kinematic_found << "/" <<
          (kRandomAttempts + 1) << " attempts";
        if (kinematic_found == 0) {
          out << " -> UNREACHABLE (no kinematic IK solution)";
          return out.str();
        }
        out << ", out_of_bounds=" << out_of_bounds << ", colliding=" << colliding <<
          ", collision_free=" << collision_free;
        if (collision_free > 0) {
          out << " -> collision-free kinematic branch exists (collision-aware IK seed/timeout)";
        }
        if (colliding > 0) {
          out << "; pair hit counts: ";
          bool first = true;
          for (const auto& entry : pair_counts) {
            if (!first) out << ", ";
            first = false;
            out << entry.first << " x" << entry.second;
          }
          out << "; shallowest colliding branch [" << best_contacts << "] joints=(";
          for (std::size_t j = 0; j < best_joints.size(); ++j) {
            if (j > 0) out << ",";
            out << best_joints[j];
          }
          out << ")";
        }
        return out.str();
      };
    for (std::size_t candidate_index = 0; candidate_index < candidates.size();
         ++candidate_index) {
      const double theta_deg = candidates[candidate_index];
      bool feasible = true;
      std::string failure;
      for (std::size_t index = 0; index < labeled_poses.size(); ++index) {
        const Eigen::Quaterniond candidate =
          rollLeftOrientationAboutToolY(mapped[index], theta_deg);
        const double dot_x = (candidate * Eigen::Vector3d::UnitX()).dot(into_unit);
        const double dot_z = (candidate * Eigen::Vector3d::UnitZ()).dot(into_unit);
        if (dot_x < 0.7 && dot_z < 0.7) {
          std::ostringstream detail;
          detail << std::fixed << std::setprecision(4);
          detail << labeled_poses[index].second << " penetration gate (x=" << dot_x <<
            ", z=" << dot_z << " < 0.7)";
          feasible = false;
          failure = detail.str();
          record_category(labeled_poses[index].second);
          break;
        }
        apply(index, candidate);
        moveit::core::RobotState probe(seed_state);
        // Hover/approach remains free. Only cut targets may use the scoped
        // tool/carton contact allowance installed by their MTC cut stages.
        scene.getAllowedCollisionMatrixNonConst().setEntry(
          tool_link_, carton_collision_id_, index % 2U != 0U && allow_tool_carton_contact_);
        if (!probe.setFromIK(
              joint_group, labeled_poses[index].first->pose, tool_link_, 0.25, valid_state)) {
          feasible = false;
          failure = std::string(labeled_poses[index].second) +
            " collision-aware IK (no valid solution)";
          record_category(labeled_poses[index].second);
          break;
        }
      }
      if (candidate_index == 0) first_failure = failure;
      RCLCPP_INFO(
        get_logger(), "Left roll diag [%s] roll %+.1f deg: %s%s", pose_context.c_str(),
        theta_deg, feasible ? "SELECTED" : "rejected at ", feasible ? "" : failure.c_str());
      if (!feasible && detailed_diagnostics) {
        for (std::size_t index = 0; index < labeled_poses.size(); ++index) {
          RCLCPP_INFO(
            get_logger(), "Left roll diag [%s] roll %+.1f deg  %s: %s",
            pose_context.c_str(), theta_deg, labeled_poses[index].second,
            diagnose_pose(index, rollLeftOrientationAboutToolY(mapped[index], theta_deg))
            .c_str());
        }
      }
      if (feasible) {
        if (theta_deg != 0.0) {
          RCLCPP_INFO(
            get_logger(),
            "Left adaptive orientation selected TCP-local Y roll %.3f deg for %s",
            theta_deg, pose_context.c_str());
        }
        return true;
      }
    }
    for (std::size_t index = 0; index < labeled_poses.size(); ++index) {
      apply(index, mapped[index]);
    }
    std::ostringstream detail;
    detail << std::fixed << std::setprecision(3);
    detail << "left orientation adaptation failed for " << pose_context <<
      ": no TCP-local Y roll candidate passed the left X-or-Z penetration gate (>=0.700) "
      "and collision-aware IK for ";
    for (std::size_t index = 0; index < labeled_poses.size(); ++index) {
      if (index > 0) detail << "/";
      detail << labeled_poses[index].second;
    }
    detail << " (tried " << candidates.size() <<
      " candidates 0,+step,-step,...,+/-max with max=" << left_adaptive_orientation_max_deg_ <<
      " deg step=" << left_adaptive_orientation_step_deg_ << " deg)";
    if (!first_failure.empty()) detail << "; candidate 0 first failed at " << first_failure;
    if (!failing_categories.empty()) {
      detail << "; failing pose categories across search: ";
      for (std::size_t index = 0; index < failing_categories.size(); ++index) {
        if (index > 0) detail << ", ";
        detail << failing_categories[index];
      }
    }
    error = detail.str();
    return false;
  }
  // Uncorrected geometry baseline for a staged session; saved deltas reference it so
  // each completed record remains standalone.
  bool cutDebugNominalPoses(
    const Snapshot& snapshot, const std::string& cut,
    geometry_msgs::msg::PoseStamped& nominal_start,
    geometry_msgs::msg::PoseStamped& nominal_end, std::string& error) const
  {
    geometry_msgs::msg::Point start;
    geometry_msgs::msg::Point end;
    if (!cutDebugEndpoints(snapshot, cut, start, end)) {
      error = "unknown cut debug cut: " + cut;
      return false;
    }
    const bool is_middle = cut == "middle";
    const double tilt_deg = is_middle ? 0.0 :
      (cut == "right" ? right_side_cut_tilt_deg_ : left_side_cut_tilt_deg_);
    const auto targets = orientationTargets(snapshot, is_middle, start, end, start, end);
    const double cut_clearance = is_middle ? middle_cut_clearance_m_ : cut_clearance_m_;
    nominal_start = targetPose(snapshot, start, targets.first, cut_clearance, tilt_deg);
    nominal_end = targetPose(snapshot, end, targets.second, cut_clearance, tilt_deg);
    return true;
  }
  bool captureObservedTcpPose(geometry_msgs::msg::PoseStamped& observed, std::string& error)
  {
    sensor_msgs::msg::JointState joint_state;
    rclcpp::Time received;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_joint_state_ || current_joint_state_->name.empty() ||
          current_joint_state_->position.empty()) {
        error = "no /joint_states received; cannot record TCP observation";
        return false;
      }
      joint_state = *current_joint_state_;
      received = joint_state_received_;
    }
    const double receipt_age_s = (now() - received).seconds();
    if (!std::isfinite(receipt_age_s) || receipt_age_s < 0.0 || receipt_age_s > 1.0) {
      error = "no recent /joint_states received (last " + formatDouble(receipt_age_s, 3) +
        " s ago); cannot record a TCP observation from stale joint data";
      return false;
    }
    if (joint_state.name.size() > joint_state.position.size()) {
      error = "joint state has fewer positions than names; cannot record TCP observation";
      return false;
    }
    if (!cut_debug_robot_model_) {
      error = "robot model is unavailable; plan a cut debug stage before saving";
      return false;
    }
    moveit::core::RobotState state(cut_debug_robot_model_);
    state.setToDefaultValues();
    const auto& variables = cut_debug_robot_model_->getVariableNames();
    for (const auto& variable : variables) {
      const auto position = std::find(
        joint_state.name.begin(), joint_state.name.end(), variable);
      if (position == joint_state.name.end()) {
        error = "joint state is missing joint '" + variable +
          "'; refusing to record a TCP observation from default joint values";
        return false;
      }
      const double value =
        joint_state.position[static_cast<std::size_t>(position - joint_state.name.begin())];
      if (!std::isfinite(value)) {
        error = "joint state contains a non-finite position for joint '" + variable +
          "'; cannot record TCP observation";
        return false;
      }
      state.setVariablePosition(variable, value);
    }
    state.update();
    const Eigen::Isometry3d transform = state.getGlobalLinkTransform(tool_link_);
    const Eigen::Vector3d position = transform.translation();
    const Eigen::Quaterniond orientation(transform.rotation());
    if (!position.allFinite() || !orientation.coeffs().allFinite()) {
      error = "forward kinematics produced a non-finite TCP pose";
      return false;
    }
    observed.header.frame_id = planning_frame_;
    observed.header.stamp = now();
    observed.pose.position.x = position.x();
    observed.pose.position.y = position.y();
    observed.pose.position.z = position.z();
    observed.pose.orientation.x = orientation.x();
    observed.pose.orientation.y = orientation.y();
    observed.pose.orientation.z = orientation.z();
    observed.pose.orientation.w = orientation.w();
    return true;
  }
  static std::string formatDouble(double value, int precision = 9)
  {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(precision);
    stream << value;
    return stream.str();
  }
  bool cutDebugRecordPath(
    const std::string& cut, const std::string& snapshot_id, std::string& path,
    std::string& error) const
  {
    namespace fs = std::filesystem;
    fs::path directory;
    if (!cut_debug_output_dir_.empty()) {
      directory = fs::path(cut_debug_output_dir_);
    } else {
      const char* home = std::getenv("HOME");
      if (home == nullptr || std::string(home).empty()) {
        error = "cannot resolve cut debug output directory without HOME";
        return false;
      }
      directory = fs::path(home) / "cut_debug_corrections";
    }
    if (snapshot_id.empty() || snapshot_id.find('/') != std::string::npos ||
        snapshot_id.find('\\') != std::string::npos) {
      error = "snapshot ID is not safe for a record filename";
      return false;
    }
    std::error_code code;
    fs::create_directories(directory, code);
    if (code) {
      error = "cannot create cut debug record directory: " + code.message();
      return false;
    }
    path = (directory / (cut + "_" + snapshot_id + ".json")).string();
    return true;
  }
  bool writeCutDebugRecord(bool complete, std::string& error)
  {
    if (!cut_debug_snapshot_ || cut_debug_record_path_.empty()) {
      error = "cut debug session has no snapshot or record path";
      return false;
    }
    const Snapshot& snapshot = *cut_debug_snapshot_;
    std::array<double, 3> along{};
    std::array<double, 3> across{};
    std::array<double, 3> normal{};
    if (!cutDebugBasis(snapshot, cut_debug_session_cut_, along, across, normal, error)) {
      return false;
    }
    const auto world_delta = [&along, &across, &normal](
      const geometry_msgs::msg::Pose& nominal, const geometry_msgs::msg::Pose& observed) {
      return std::array<double, 3>{
        observed.position.x - nominal.position.x,
        observed.position.y - nominal.position.y,
        observed.position.z - nominal.position.z};
    };
    const auto to_local = [&along, &across, &normal](const std::array<double, 3>& world) {
      CutLocalDeltas local;
      local.along_m = world[0] * along[0] + world[1] * along[1] + world[2] * along[2];
      local.across_m = world[0] * across[0] + world[1] * across[1] + world[2] * across[2];
      local.normal_m = world[0] * normal[0] + world[1] * normal[1] + world[2] * normal[2];
      return local;
    };
    const auto orientation_deltas = [this, &along, &across, &normal](
      const geometry_msgs::msg::Pose& nominal, const geometry_msgs::msg::Pose& observed,
      geometry_msgs::msg::Quaternion& world_delta_out,
      geometry_msgs::msg::Quaternion& local_delta_out, double& angle_deg_out,
      std::string& orientation_error) {
      const Eigen::Quaterniond nominal_q(
        nominal.orientation.w, nominal.orientation.x, nominal.orientation.y,
        nominal.orientation.z);
      const Eigen::Quaterniond observed_q(
        observed.orientation.w, observed.orientation.x, observed.orientation.y,
        observed.orientation.z);
      if (!nominal_q.coeffs().allFinite() || !observed_q.coeffs().allFinite() ||
          std::abs(nominal_q.norm() - 1.0) > 0.01 ||
          std::abs(observed_q.norm() - 1.0) > 0.01) {
        orientation_error = "nominal or observed orientation is not unit length";
        return false;
      }
      const Eigen::Quaterniond world_delta_q = (observed_q * nominal_q.inverse()).normalized();
      const Eigen::Quaterniond basis_q = cutDebugBasisQuaternion(along, across, normal);
      const Eigen::Quaterniond local_q =
        (basis_q.inverse() * world_delta_q * basis_q).normalized();
      world_delta_out.x = world_delta_q.x();
      world_delta_out.y = world_delta_q.y();
      world_delta_out.z = world_delta_q.z();
      world_delta_out.w = world_delta_q.w();
      local_delta_out.x = local_q.x();
      local_delta_out.y = local_q.y();
      local_delta_out.z = local_q.z();
      local_delta_out.w = local_q.w();
      angle_deg_out = 2.0 * std::acos(std::min(1.0, std::abs(world_delta_q.w()))) * 180.0 /
        std::acos(-1.0);
      return true;
    };
    if (!cut_debug_has_observed_start_) {
      error = "cut debug start observation is missing";
      return false;
    }
    if (complete && !cut_debug_has_observed_end_) {
      error = "cut debug end observation is missing";
      return false;
    }
    const auto start_world = world_delta(
      cut_debug_nominal_start_.pose, cut_debug_observed_start_.pose);
    const CutLocalDeltas start_local = to_local(start_world);
    for (const double component : {start_local.along_m, start_local.across_m, start_local.normal_m}) {
      if (!std::isfinite(component)) {
        error = "cut debug start translation delta is non-finite; refusing to persist an unbounded correction";
        return false;
      }
    }
    geometry_msgs::msg::Quaternion start_world_q;
    geometry_msgs::msg::Quaternion start_local_q;
    double start_angle_deg{0.0};
    if (!orientation_deltas(
          cut_debug_nominal_start_.pose, cut_debug_observed_start_.pose, start_world_q,
          start_local_q, start_angle_deg, error)) {
      return false;
    }
    std::array<double, 3> end_world{0.0, 0.0, 0.0};
    CutLocalDeltas end_local{};
    geometry_msgs::msg::Quaternion end_world_q;
    geometry_msgs::msg::Quaternion end_local_q;
    double end_angle_deg{0.0};
    end_world_q.x = 0.0;
    end_world_q.y = 0.0;
    end_world_q.z = 0.0;
    end_world_q.w = 1.0;
    end_local_q = end_world_q;
    if (complete) {
      end_world = world_delta(cut_debug_nominal_end_.pose, cut_debug_observed_end_.pose);
      end_local = to_local(end_world);
      for (const double component : {end_local.along_m, end_local.across_m, end_local.normal_m}) {
        if (!std::isfinite(component)) {
          error = "cut debug end translation delta is non-finite; refusing to persist an unbounded correction";
          return false;
        }
      }
      if (!orientation_deltas(
            cut_debug_nominal_end_.pose, cut_debug_observed_end_.pose, end_world_q,
            end_local_q, end_angle_deg, error)) {
        return false;
      }
    }
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    if (cut_debug_created_stamp_ns_ == 0) cut_debug_created_stamp_ns_ = now_ns;
    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(9);
    json << "{\n";
    json << "  \"schema\": \"" << cutDebugCorrectionSchema() << "\",\n";
    json << "  \"cut\": \"" << cut_debug_session_cut_ << "\",\n";
    json << "  \"snapshot_id\": \"" << snapshot.snapshot_id << "\",\n";
    json << "  \"capture_id\": \"" << snapshot.capture_id << "\",\n";
    json << "  \"acquisition_stamp_ns\": " << snapshot.acquisition_stamp_ns << ",\n";
    json << "  \"planning_frame\": \"" << planning_frame_ << "\",\n";
    json << "  \"tool_link\": \"" << tool_link_ << "\",\n";
    json << "  \"status\": \"" << (complete ? "complete" : "start_saved") << "\",\n";
    json << "  \"created_stamp_ns\": " << cut_debug_created_stamp_ns_ << ",\n";
    json << "  \"updated_stamp_ns\": " << now_ns << ",\n";
    json << "  \"cut_basis_along_m\": [" << formatDouble(along[0]) << ", " <<
      formatDouble(along[1]) << ", " << formatDouble(along[2]) << "],\n";
    json << "  \"cut_basis_across_m\": [" << formatDouble(across[0]) << ", " <<
      formatDouble(across[1]) << ", " << formatDouble(across[2]) << "],\n";
    json << "  \"cut_basis_normal_m\": [" << formatDouble(normal[0]) << ", " <<
      formatDouble(normal[1]) << ", " << formatDouble(normal[2]) << "],\n";
    const auto pose_block = [this](
      const char* name, const geometry_msgs::msg::Pose& pose, std::ostringstream& out) {
      out << "  \"" << name << "_position_m\": [" << formatDouble(pose.position.x) << ", " <<
        formatDouble(pose.position.y) << ", " << formatDouble(pose.position.z) << "],\n";
      out << "  \"" << name << "_orientation_xyzw\": [" << formatDouble(pose.orientation.x) <<
        ", " << formatDouble(pose.orientation.y) << ", " << formatDouble(pose.orientation.z) <<
        ", " << formatDouble(pose.orientation.w) << "],\n";
    };
    pose_block("nominal_start", cut_debug_nominal_start_.pose, json);
    pose_block("observed_start", cut_debug_observed_start_.pose, json);
    json << "  \"start_delta_world_m\": [" << formatDouble(start_world[0]) << ", " <<
      formatDouble(start_world[1]) << ", " << formatDouble(start_world[2]) << "],\n";
    json << "  \"start_delta_cut_local_along_m\": " << formatDouble(start_local.along_m) << ",\n";
    json << "  \"start_delta_cut_local_across_m\": " << formatDouble(start_local.across_m) << ",\n";
    json << "  \"start_delta_cut_local_normal_m\": " << formatDouble(start_local.normal_m) << ",\n";
    json << "  \"start_orientation_delta_world_xyzw\": {\"x\": " <<
      formatDouble(start_world_q.x) << ", \"y\": " << formatDouble(start_world_q.y) <<
      ", \"z\": " << formatDouble(start_world_q.z) << ", \"w\": " <<
      formatDouble(start_world_q.w) << "},\n";
    json << "  \"start_delta_cut_local_orientation_delta_local_xyzw\": {\"x\": " <<
      formatDouble(start_local_q.x) << ", \"y\": " << formatDouble(start_local_q.y) <<
      ", \"z\": " << formatDouble(start_local_q.z) << ", \"w\": " <<
      formatDouble(start_local_q.w) << "},\n";
    json << "  \"start_orientation_delta_angle_deg\": " << formatDouble(start_angle_deg, 6) << ",\n";
    pose_block("nominal_end", cut_debug_nominal_end_.pose, json);
    if (complete) {
      pose_block("observed_end", cut_debug_observed_end_.pose, json);
      json << "  \"end_delta_world_m\": [" << formatDouble(end_world[0]) << ", " <<
        formatDouble(end_world[1]) << ", " << formatDouble(end_world[2]) << "],\n";
    } else {
      json << "  \"observed_end_position_m\": null,\n";
      json << "  \"observed_end_orientation_xyzw\": null,\n";
      json << "  \"end_delta_world_m\": null,\n";
    }
    json << "  \"end_delta_cut_local_along_m\": " << formatDouble(end_local.along_m) << ",\n";
    json << "  \"end_delta_cut_local_across_m\": " << formatDouble(end_local.across_m) << ",\n";
    json << "  \"end_delta_cut_local_normal_m\": " << formatDouble(end_local.normal_m) << ",\n";
    json << "  \"end_orientation_delta_world_xyzw\": {\"x\": " << formatDouble(end_world_q.x) <<
      ", \"y\": " << formatDouble(end_world_q.y) << ", \"z\": " << formatDouble(end_world_q.z) <<
      ", \"w\": " << formatDouble(end_world_q.w) << "},\n";
    json << "  \"end_delta_cut_local_orientation_delta_local_xyzw\": {\"x\": " <<
      formatDouble(end_local_q.x) << ", \"y\": " << formatDouble(end_local_q.y) << ", \"z\": " <<
      formatDouble(end_local_q.z) << ", \"w\": " << formatDouble(end_local_q.w) << "},\n";
    json << "  \"end_orientation_delta_angle_deg\": " << formatDouble(end_angle_deg, 6) << "\n";
    json << "}\n";
    namespace fs = std::filesystem;
    const fs::path output(cut_debug_record_path_);
    const fs::path temporary = output.string() + ".tmp";
    {
      std::ofstream file(temporary);
      if (!file) {
        error = "cannot write cut debug record: " + temporary.string();
        return false;
      }
      file << json.str();
      file.flush();
      if (!file) {
        error = "failed while writing cut debug record: " + temporary.string();
        return false;
      }
    }
    std::error_code code;
    fs::rename(temporary, output, code);
    if (code) {
      fs::remove(temporary, code);
      error = "cannot atomically persist cut debug record: " + code.message();
      return false;
    }
    cut_debug_start_local_ = start_local;
    if (complete) cut_debug_end_local_ = end_local;
    return true;
  }
  bool startCutDebugSession(std::string& message)
  {
    if (cut_debug_stage_ != CutDebugStage::kNone) {
      message = "cut debug session already active for cut " + cut_debug_session_cut_ +
        " stage " + cutDebugStageName(cut_debug_stage_) +
        "; call clear_plan before starting a new session";
      return false;
    }
    if (planned_task_ && planned_snapshot_ && !planned_task_->solutions().empty()) {
      message =
        "a normal H-cut plan is retained; call clear_plan before starting a cut debug session";
      return false;
    }
    if (cut_debug_cut_ != "left" && cut_debug_cut_ != "middle" && cut_debug_cut_ != "right") {
      message = "cut_debug_cut must be left, middle, or right";
      return false;
    }
    const auto source_path = get_parameter("external_snapshot_file").as_string();
    std::string load_error;
    const auto snapshot = loadExternalSnapshot(source_path, load_error);
    if (!snapshot) {
      message = load_error;
      return false;
    }
    std::string validation;
    if (!validateSnapshot(*snapshot, validation)) {
      message = "snapshot rejected: " + validation;
      return false;
    }
    geometry_msgs::msg::PoseStamped nominal_start;
    geometry_msgs::msg::PoseStamped nominal_end;
    if (!cutDebugNominalPoses(
          *snapshot, cut_debug_cut_, nominal_start, nominal_end, validation)) {
      message = "cut debug nominal computation failed for cut " + cut_debug_cut_ + ": " +
        validation;
      return false;
    }
    // The configured target (correction applied when opted in) is planned by the common
    // segment planner, stopping at the cut start: same dwell and wrist-guard validation
    // as full plans, with the carton added under the scoped contact allowance.
    CutSegment segment;
    segment.name = cut_debug_cut_ == "middle" ? "middle seam cut" :
      (cut_debug_cut_ == "left" ? "left side cut" : "right side cut");
    cutDebugEndpoints(*snapshot, cut_debug_cut_, segment.start, segment.end);
    segment.stop_after_start = true;
    std::unique_ptr<mtc::Task> task;
    std::vector<geometry_msgs::msg::PoseStamped> targets;
    std::string planning_message;
    bool planned = false;
    try {
      const auto corrections = loadActiveCorrections({cut_debug_cut_}, false);
      publishCartonGeometry(*snapshot);
      SegmentPlanOptions options;
      options.timeout_s = planning_timeout_s_;
      options.cut_debug_start = true;
      planned = planCutSegments(
        *snapshot, {segment}, corrections,
        get_parameter("motion_profile").as_string(), "Cut debug " + cut_debug_cut_ + " start",
        options, task, targets, planning_message);
    } catch (const std::exception& exception) {
      planning_message = exception.what();
    }
    if (!planned) {
      message = "cut debug start planning failed for cut " + cut_debug_cut_ + ": " +
        planning_message;
      return false;
    }
    if (targets.size() != 1U) {
      message = "cut debug start planner produced an unexpected target count";
      return false;
    }
    publishMarkers(targets, {segment});
    publishTargetPoses(*snapshot, targets);
    cut_debug_snapshot_ = *snapshot;
    cut_debug_session_cut_ = cut_debug_cut_;
    cut_debug_nominal_start_ = nominal_start;
    cut_debug_nominal_end_ = nominal_end;
    cut_debug_has_observed_start_ = false;
    cut_debug_has_observed_end_ = false;
    cut_debug_created_stamp_ns_ = 0;
    cut_debug_robot_model_ = task->getRobotModel();
    planned_snapshot_ = *snapshot;
    planned_task_ = std::move(task);
    cut_debug_stage_ = CutDebugStage::kStartPlanned;
    message = "cut debug session started for cut " + cut_debug_session_cut_ +
      " snapshot " + snapshot->snapshot_id +
      "; configured start target published to RViz and retained; saved deltas use the uncorrected geometry baseline; execute explicitly via execute_planned_h_cut with both execution_enabled and debug_motion_execution_enabled; this service never dispatches motion";
    return true;
  }
  bool saveCutDebugPoint(std::string& message)
  {
    if (cut_debug_stage_ == CutDebugStage::kNone) {
      message = "no cut debug session is active; call start_cut_debug_session first";
      return false;
    }
    if (cut_debug_dispatch_failed_) {
      message = "previous cut debug " + cutDebugStageName(cut_debug_stage_) + " dispatch for cut " +
        cut_debug_session_cut_ +
        " failed during execution and was NOT marked executed; the arm pose is unknown, so no observation is captured; call clear_plan then start_cut_debug_session to restart";
      return false;
    }
    if (cut_debug_stage_ == CutDebugStage::kStartPlanned) {
      message = "cut debug start planned but not yet executed for cut " + cut_debug_session_cut_ +
        "; execute via execute_planned_h_cut then teleop correct before calling save_cut_debug_point";
      return false;
    }
    if (planned_task_ && !planned_snapshot_) {
      clearRetainedPlan();
    }
    if (planned_task_) {
      clearRetainedPlan();
      message = "a stale cut debug plan is still retained for cut " + cut_debug_session_cut_ +
        "; it was cleared without saving; execute the retained stage before saving";
      return false;
    }
    if (cut_debug_stage_ == CutDebugStage::kStartExecuted) {
      geometry_msgs::msg::PoseStamped observed;
      if (!captureObservedTcpPose(observed, message)) return false;
      cut_debug_observed_start_ = observed;
      cut_debug_has_observed_start_ = true;
      std::string record_error;
      std::string record_path;
      if (!cutDebugRecordPath(
            cut_debug_session_cut_, cut_debug_snapshot_->snapshot_id, record_path,
            record_error)) {
        message = record_error;
        return false;
      }
      cut_debug_record_path_ = record_path;
      if (!writeCutDebugRecord(false, record_error)) {
        message = record_error;
        return false;
      }
      clearRetainedPlan();
      publishCartonGeometry(*cut_debug_snapshot_);
      cut_debug_stage_ = CutDebugStage::kEndExecuted;
      const auto& nominal = cut_debug_nominal_start_.pose.position;
      const auto& observed_position = observed.pose.position;
      message = "saved cut debug start observation to " + cut_debug_record_path_ +
        "; start_delta_world_m=[" + formatDouble(observed_position.x - nominal.x) + ", " +
        formatDouble(observed_position.y - nominal.y) + ", " +
        formatDouble(observed_position.z - nominal.z) +
        "] cut_local_along_m=" + formatDouble(cut_debug_start_local_.along_m) +
        " across_m=" + formatDouble(cut_debug_start_local_.across_m) +
        " normal_m=" + formatDouble(cut_debug_start_local_.normal_m) +
        "; no endpoint traverse was planned; teleop to the endpoint then call save_cut_debug_point; saved deltas use the uncorrected geometry baseline; this service never dispatches motion";
      return true;
    }
    if (cut_debug_stage_ == CutDebugStage::kEndExecuted) {
      geometry_msgs::msg::PoseStamped observed;
      if (!captureObservedTcpPose(observed, message)) return false;
      cut_debug_observed_end_ = observed;
      cut_debug_has_observed_end_ = true;
      std::string record_error;
      if (!writeCutDebugRecord(true, record_error)) {
        message = record_error;
        return false;
      }
      const auto& nominal = cut_debug_nominal_end_.pose.position;
      const auto& observed_position = observed.pose.position;
      const std::string record_path = cut_debug_record_path_;
      const std::string session_cut = cut_debug_session_cut_;
      const CutLocalDeltas end_local = cut_debug_end_local_;
      clearRetainedPlan();
      clearCutDebugSessionState();
      message = "saved cut debug end observation to " + record_path +
        "; end_delta_world_m=[" + formatDouble(observed_position.x - nominal.x) + ", " +
        formatDouble(observed_position.y - nominal.y) + ", " +
        formatDouble(observed_position.z - nominal.z) +
        "] cut_local_along_m=" + formatDouble(end_local.along_m) +
        " across_m=" + formatDouble(end_local.across_m) +
        " normal_m=" + formatDouble(end_local.normal_m) +
        "; staged session for cut " + session_cut + " complete with no lift plan; future plans apply the taught start/end poses from the record to new snapshots; this service never dispatches motion";
      return true;
    }
    message = "cut debug session for cut " + cut_debug_session_cut_ + " stage " +
      cutDebugStageName(cut_debug_stage_) + " cannot save; unexpected state";
    return false;
  }
  bool diagnoseExternalSnapshotSections(std::string& message)
  {
    const auto source_path = get_parameter("external_snapshot_file").as_string();
    const auto snapshot = loadExternalSnapshot(source_path, message);
    if (!snapshot) return false;
    std::string validation;
    const bool canonical = validateSnapshot(*snapshot, validation);
    const auto motion_profile = get_parameter("motion_profile").as_string();
    const std::array<CutSegment, 3> sections{{
      {"left side cut", snapshot->back_middle_left, snapshot->front_middle_left},
      {"middle seam cut", snapshot->seam_start, snapshot->seam_end},
      {"right side cut", snapshot->back_middle_right, snapshot->front_middle_right},
    }};
    std::ostringstream report;
    report << "diagnostic_only snapshot=" << snapshot->snapshot_id
           << " canonical=" << (canonical ? "true" : "false")
           << " validation=" << (canonical ? "ok" : validation);
    SegmentPlanOptions options;
    options.timeout_s = planning_timeout_s_;
    std::optional<std::map<std::string, CutDebugCorrection>> corrections;
    for (const auto& section : sections) {
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> targets;
      std::string result;
      bool planned = false;
      try {
        if (!corrections) corrections = loadAllActiveCorrections();
        planned = planCutSegments(
          *snapshot, {section}, *corrections, motion_profile,
          "section diagnostic " + section.name, options, task, targets, result);
      } catch (const std::exception& exception) {
        result = exception.what();
      }
      report << "; " << section.name << "=" << (planned ? "planned" : "failed")
             << " (" << result << ")";
    }
    message = "external snapshot " + source_path + ": " + report.str();
    return true;
  }

  std::optional<Snapshot> loadExternalSnapshot(
    const std::string& external_snapshot_file, std::string& message) const
  {
    namespace fs = std::filesystem;
    if (external_snapshot_file.empty()) {
      message = "external_snapshot_file is required";
      return std::nullopt;
    }
    const fs::path selected(external_snapshot_file);
    if (!selected.is_absolute() || selected.extension() != ".json") {
      message = "external_snapshot_file must be an absolute .json path";
      return std::nullopt;
    }
    std::error_code error;
    if (!fs::is_regular_file(selected, error) || error) {
      message = "external snapshot is unavailable: " + selected.string();
      return std::nullopt;
    }
    const auto modified = fs::last_write_time(selected, error);
    if (error) {
      message = "cannot read external snapshot modification time: " + selected.string();
      return std::nullopt;
    }
    const auto age_s = std::chrono::duration<double>(
      fs::file_time_type::clock::now() - modified).count();
    if (!debug_allow_stale_snapshot_ && age_s > external_snapshot_max_age_s_) {
      message = "external snapshot is stale (" + formatDouble(age_s, 3) + " s; maximum " +
        formatDouble(external_snapshot_max_age_s_, 3) + " s): " + selected.string();
      return std::nullopt;
    }
    std::ifstream input(selected);
    if (!input) {
      message = "cannot read external snapshot: " + selected.string();
      return std::nullopt;
    }
    using boost::property_tree::ptree;
    ptree document;
    try {
      boost::property_tree::read_json(input, document);
    } catch (const boost::property_tree::json_parser::json_parser_error& error) {
      message = "external snapshot is not valid JSON: " + std::string(error.what());
      return std::nullopt;
    }
    const auto required_string = [&document](const std::string& name, std::string& value) {
      const auto parsed = document.get_optional<std::string>(name);
      if (!parsed || parsed->empty()) return false;
      value = *parsed;
      return true;
    };
    const auto required_point = [&document](
      const std::string& name, geometry_msgs::msg::Point& value) {
      const auto child = document.get_child_optional(name);
      if (!child) return false;
      const auto x = child->get_optional<double>("x");
      const auto y = child->get_optional<double>("y");
      const auto z = child->get_optional<double>("z");
      if (!x || !y || !z || !std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z)) {
        return false;
      }
      value.x = *x;
      value.y = *y;
      value.z = *z;
      return true;
    };
    std::string schema;
    std::string source_frame;
    std::string source_file;
    Snapshot snapshot;
    if (!required_string("schema", schema) || schema != "box_h_cut_snapshot/v1" ||
        !required_string("snapshot_id", snapshot.snapshot_id) ||
        !required_string("capture_id", snapshot.capture_id) ||
        !required_string("frame_id", snapshot.header.frame_id) ||
        !required_string("source_frame", source_frame) ||
        !required_string("source_file", source_file)) {
      message = "external snapshot schema or required identity fields are invalid: " +
        selected.string();
      return std::nullopt;
    }
    const auto acquisition_stamp = document.get_optional<std::int64_t>("acquisition_stamp_ns");
    const auto plane_rms = document.get_optional<double>("plane_rms_m");
    geometry_msgs::msg::Point normal;
    if (!acquisition_stamp || *acquisition_stamp <= 0 || !plane_rms || !std::isfinite(*plane_rms) ||
        *plane_rms < 0.0 ||
        !required_point("seam_start", snapshot.seam_start) ||
        !required_point("seam_end", snapshot.seam_end) ||
        !required_point("seam_center", snapshot.seam_center) ||
        !required_point("back_left", snapshot.back_left) ||
        !required_point("back_right", snapshot.back_right) ||
        !required_point("front_left", snapshot.front_left) ||
        !required_point("front_right", snapshot.front_right) ||
        !required_point("back_middle_left", snapshot.back_middle_left) ||
        !required_point("front_middle_left", snapshot.front_middle_left) ||
        !required_point("back_middle_right", snapshot.back_middle_right) ||
        !required_point("front_middle_right", snapshot.front_middle_right) ||
        !required_point("surface_normal", normal)) {
      message = "external snapshot required geometry is invalid: " + selected.string();
      return std::nullopt;
    }
    snapshot.acquisition_stamp_ns = *acquisition_stamp;
    snapshot.plane_rms_m = *plane_rms;
    snapshot.surface_normal.x = normal.x;
    snapshot.surface_normal.y = normal.y;
    snapshot.surface_normal.z = normal.z;
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    const auto age_ns = now_ns - snapshot.acquisition_stamp_ns;
    const auto maximum_age_ns = static_cast<std::int64_t>(external_snapshot_max_age_s_ * 1e9);
    if (age_ns < -1'000'000'000LL ||
        (!debug_allow_stale_snapshot_ && age_ns > maximum_age_ns)) {
      message = "external snapshot acquisition is stale or invalid; maximum age " +
        formatDouble(external_snapshot_max_age_s_, 3) + " s: " + selected.string();
      return std::nullopt;
    }
    snapshot.header.stamp.sec = static_cast<std::int32_t>(snapshot.acquisition_stamp_ns / 1'000'000'000LL);
    snapshot.header.stamp.nanosec = static_cast<std::uint32_t>(snapshot.acquisition_stamp_ns % 1'000'000'000LL);
    return snapshot;
  }

  void addCameraPole(mtc::Task& task)
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
    if (!camera_pole) return;
    auto add_pole = std::make_unique<mtc::stages::ModifyPlanningScene>(
      "add fixed external camera pole collision object");
    add_pole->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    add_pole->addObject(*camera_pole);
    task.add(std::move(add_pole));
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

    const geometry_msgs::msg::Quaternion q = basisQuaternion(x_axis, y_axis, normal);

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
    const std::array<const geometry_msgs::msg::Point*, 11> points{
      &snapshot.seam_start, &snapshot.seam_end, &snapshot.back_left, &snapshot.back_right,
      &snapshot.front_left, &snapshot.front_right, &snapshot.seam_center,
      &snapshot.back_middle_left, &snapshot.front_middle_left,
      &snapshot.back_middle_right, &snapshot.front_middle_right};
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
      message = "snapshot surface_normal must be unit length; norm=" + formatDouble(normal_length, 3);
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
        !on_top_face(snapshot.seam_center) || !on_top_face(snapshot.back_middle_left) ||
        !on_top_face(snapshot.front_middle_left) || !on_top_face(snapshot.back_middle_right) ||
        !on_top_face(snapshot.front_middle_right)) {
      message = "snapshot seam or side cutting point is outside the detected carton top face";
      return false;
    }
    const auto midpoint_matches = [](
      const geometry_msgs::msg::Point& value, const geometry_msgs::msg::Point& first,
      const geometry_msgs::msg::Point& second) {
      constexpr double tolerance_m = 1e-6;
      return std::abs(value.x - (first.x + second.x) / 2.0) <= tolerance_m &&
             std::abs(value.y - (first.y + second.y) / 2.0) <= tolerance_m &&
             std::abs(value.z - (first.z + second.z) / 2.0) <= tolerance_m;
    };
    if (!midpoint_matches(snapshot.back_middle_left, snapshot.back_left, snapshot.seam_start) ||
        !midpoint_matches(snapshot.front_middle_left, snapshot.front_left, snapshot.seam_start) ||
        !midpoint_matches(snapshot.back_middle_right, snapshot.back_right, snapshot.seam_end) ||
        !midpoint_matches(snapshot.front_middle_right, snapshot.front_right, snapshot.seam_end)) {
      message = "snapshot side cutting points do not match their canonical midpoint formulas";
      return false;
    }
    return true;
  }

  bool snapshotIsFresh(const Snapshot& snapshot, std::string& message) const
  {
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    const auto age_ns = now_ns - snapshot.acquisition_stamp_ns;
    const auto maximum_age_ns = static_cast<std::int64_t>(external_snapshot_max_age_s_ * 1e9);
    if (snapshot.acquisition_stamp_ns <= 0 || age_ns < -1'000'000'000LL || age_ns > maximum_age_ns) {
      message = "planned snapshot is stale or has an invalid acquisition timestamp";
      return false;
    }
    return true;
  }

  void publishCartonGeometry(const Snapshot& snapshot)
  {
    carton_collision_pub_->publish(cartonCollisionObject(snapshot));
    const std::array<std::string, 11> labels{
      "back_left", "back_right", "front_left", "front_right", "seam_start", "seam_end",
      "seam_center", "back_middle_left", "front_middle_left", "back_middle_right",
      "front_middle_right"};
    const std::array<const geometry_msgs::msg::Point*, 11> points{
      &snapshot.back_left, &snapshot.back_right, &snapshot.front_left, &snapshot.front_right,
      &snapshot.seam_start, &snapshot.seam_end, &snapshot.seam_center,
      &snapshot.back_middle_left, &snapshot.front_middle_left,
      &snapshot.back_middle_right, &snapshot.front_middle_right};
    visualization_msgs::msg::MarkerArray markers;
    for (std::size_t index = 0; index < labels.size(); ++index) {
      visualization_msgs::msg::Marker point_marker;
      point_marker.header = snapshot.header;
      point_marker.id = static_cast<int>(index);
      point_marker.ns = "detected_carton_points";
      point_marker.type = visualization_msgs::msg::Marker::SPHERE;
      point_marker.action = visualization_msgs::msg::Marker::ADD;
      point_marker.pose.position = *points[index];
      point_marker.pose.orientation.w = 1.0;
      point_marker.scale.x = point_marker.scale.y = point_marker.scale.z = 0.015;
      point_marker.color.r = index >= 4U ? 0.0F : 1.0F;
      point_marker.color.g = index >= 4U ? 1.0F : 0.5F;
      point_marker.color.b = index >= 4U ? 1.0F : 0.0F;
      point_marker.color.a = 1.0F;
      markers.markers.push_back(point_marker);
      visualization_msgs::msg::Marker text_marker = point_marker;
      text_marker.id += 100;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.text = labels[index];
      text_marker.pose.position.z += 0.020;
      text_marker.scale.z = 0.020;
      markers.markers.push_back(text_marker);
    }
    marker_pub_->publish(markers);
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
      if (!isSupportedMotionProfile(motion_profile)) {
        throw std::invalid_argument("unsupported H-cut motion_profile: " + motion_profile);
      }
      publishCartonGeometry(snapshot);
      // Each configured correction record is loaded and validated once per request.
      const auto corrections = loadAllActiveCorrections();
      if (cut_pattern_ == "t_cut_handoff") {
        return planTCutHandoffSnapshot(snapshot, corrections, motion_profile, message);
      }
      if (cut_pattern_ == "combined" || cut_pattern_ == "t_cut") {
        return planCombinedSnapshot(
          snapshot, corrections, motion_profile, cut_pattern_ == "t_cut", message);
      }

      std::vector<CutSegment> segments;
      if (cut_pattern_ == "h") {
        segments = {
          {"left side cut", snapshot.back_middle_left, snapshot.front_middle_left},
          {"middle seam cut", snapshot.seam_start, snapshot.seam_end},
          {"right side cut", snapshot.back_middle_right, snapshot.front_middle_right},
        };
      } else if (cut_pattern_ == "left_only") {
        segments = {{"left side cut", snapshot.back_middle_left, snapshot.front_middle_left}};
      } else if (cut_pattern_ == "middle_only") {
        segments = {{"middle seam cut", snapshot.seam_start, snapshot.seam_end}};
      } else if (cut_pattern_ == "right_only") {
        segments = {{"right side cut", snapshot.back_middle_right, snapshot.front_middle_right}};
      } else if (cut_pattern_ == "middle_entry_debug") {
        segments = {{"middle seam entry debug", snapshot.seam_start, snapshot.seam_end, true}};
      } else if (cut_pattern_ == "middle_debug_short") {
        const double seam_length = distanceBetween(snapshot.seam_start, snapshot.seam_end);
        if (seam_length < debug_motion_distance_m_) {
          throw std::invalid_argument("middle seam is shorter than debug_motion_distance_m");
        }
        const double half_fraction = debug_motion_distance_m_ / (2.0 * seam_length);
        segments = {{"middle debug short move",
          interpolate(snapshot.seam_start, snapshot.seam_end, 0.5 - half_fraction),
          interpolate(snapshot.seam_start, snapshot.seam_end, 0.5 + half_fraction),
          false, true, true}};
      } else {
        throw std::invalid_argument("unsupported cut_pattern: " + cut_pattern_);
      }
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> cut_targets;
      SegmentPlanOptions options;
      options.timeout_s = planning_timeout_s_;
      options.skip_supplied_approach = cut_pattern_ != "h";
      if (!planCutSegments(
            snapshot, segments, corrections, motion_profile,
            "Snapshot-relative " + cut_pattern_ + " dry run", options, task, cut_targets,
            message)) {
        return false;
      }
      retainPlan(snapshot, segments, cut_targets, std::move(task));
      const bool includes_middle_seam =
        cut_pattern_ == "h" || cut_pattern_ == "middle_only" ||
        cut_pattern_ == "middle_entry_debug" || cut_pattern_ == "middle_debug_short";
      message = "Snapshot-relative " + cut_pattern_ + " dry-run plan [" + motion_profile +
                "] published to RViz; pre-cut stabilization_s=" +
                formatDouble(pre_cut_stabilization_duration_s_, 3) + "; dwell_count=" +
                std::to_string(pre_cut_stabilization_duration_s_ > 0.0 ? segments.size() : 0U) +
                (includes_middle_seam ? "; middle entry alignment_m=" +
                  formatDouble(middle_entry_alignment_m_, 3) : "") +
                (cut_pattern_ == "middle_entry_debug" ?
                  "; stops at seam_start before the cut traverse" : "") +
                (cut_pattern_ == "middle_debug_short" ?
                  "; fixed orientation (x=0,y=1,z=0,w=0); short stroke_m=" +
                  formatDouble(debug_motion_distance_m_, 3) + "; surface clearance_m=" +
                  formatDouble(debug_motion_clearance_m_, 3) +
                  "; manual execution additionally requires debug_motion_execution_enabled=true" : "") +
                "; execution remains disabled until execution_enabled is set";
      RCLCPP_INFO(
        get_logger(),
        "Planned snapshot %s with profile %s and pattern %s; dwell count %zu at %.3f s; middle entry alignment %.1f mm",
        snapshot.snapshot_id.c_str(), motion_profile.c_str(), cut_pattern_.c_str(),
        segments.size(), pre_cut_stabilization_duration_s_,
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
  // Publishes the plan's target markers/poses and retains it for explicit execution.
  void retainPlan(
    const Snapshot& snapshot, const std::vector<CutSegment>& segments,
    const std::vector<geometry_msgs::msg::PoseStamped>& cut_targets,
    std::unique_ptr<mtc::Task> task)
  {
    publishMarkers(cut_targets, segments);
    publishTargetPoses(snapshot, cut_targets);
    planned_snapshot_ = snapshot;
    planned_task_ = std::move(task);
  }
  // Plans the first-arm T-cut handoff candidate: the full left cut plus the middle seam
  // prefix ending at `fraction`, under min_fraction=1.0 like every other candidate.
  bool planHandoffAt(
    const Snapshot& snapshot, const std::map<std::string, CutDebugCorrection>& corrections,
    const std::string& motion_profile, double fraction, const std::string& task_name,
    std::unique_ptr<mtc::Task>& task, std::vector<geometry_msgs::msg::PoseStamped>& targets,
    std::vector<CutSegment>& segments, std::string& failure)
  {
    CutSegment handoff_segment;
    handoff_segment.name = "middle seam handoff";
    handoff_segment.start = snapshot.seam_start;
    handoff_segment.end = interpolate(snapshot.seam_start, snapshot.seam_end, fraction);
    handoff_segment.handoff_end = true;
    handoff_segment.terminal_after_cut = true;
    handoff_segment.handoff_fraction = fraction;
    segments = std::vector<CutSegment>{
      {"left side cut", snapshot.back_middle_left, snapshot.front_middle_left},
      handoff_segment,
    };
    SegmentPlanOptions options;
    options.timeout_s = planning_timeout_s_;
    return planCutSegments(
      snapshot, segments, corrections, motion_profile, task_name, options, task, targets,
      failure);
  }
  bool planTCutHandoffSnapshot(
    const Snapshot& snapshot, const std::map<std::string, CutDebugCorrection>& corrections,
    const std::string& motion_profile, std::string& message)
  {
    const double fraction = t_cut_middle_handoff_fraction_;
    if (!isValidHandoffFraction(fraction)) {
      message = kHandoffFractionRule;
      return false;
    }
    std::unique_ptr<mtc::Task> best_task;
    std::vector<geometry_msgs::msg::PoseStamped> best_targets;
    std::vector<CutSegment> best_segments;
    double best_fraction = fraction;
    constexpr double kAutoTolerance = 0.01;
    if (fraction != 0.0) {
      if (!planHandoffAt(
            snapshot, corrections, motion_profile, fraction,
            "Snapshot-relative T-cut handoff canonical waypoint plan", best_task, best_targets,
            best_segments, message)) {
        return false;
      }
    } else {
      constexpr double kAutoMin = 0.05;
      constexpr double kAutoMax = 0.95;
      // Bounded bisection over [0.05,0.95]: ceil(log2((0.95-0.05)/0.01)) = 7 probes after the
      // minimum candidate. Feasibility is monotonic along the same middle seam direction, so a
      // failed probe narrows from above and a success narrows from below. Every probe is a full
      // planCutSegments handoff candidate under min_fraction=1.0; no partial path is accepted.
      constexpr int kAutoMaxProbes = 7;
      const auto candidate_name = [](double probe) {
        return "Snapshot-relative T-cut handoff automatic candidate " + formatDouble(probe, 2);
      };
      std::string probe_failure;
      if (!planHandoffAt(
            snapshot, corrections, motion_profile, kAutoMin, candidate_name(kAutoMin),
            best_task, best_targets, best_segments, probe_failure)) {
        message =
          "Automatic T-cut handoff search failed closed: no fully planned handoff candidate at minimum "
          "middle_handoff_fraction=0.05; search_tolerance=" + formatDouble(kAutoTolerance, 2) +
          "; last_failure=" + probe_failure;
        return false;
      }
      best_fraction = kAutoMin;
      double low = kAutoMin;
      double high = kAutoMax;
      for (int probe = 0; probe < kAutoMaxProbes && (high - low) > kAutoTolerance; ++probe) {
        const double mid = 0.5 * (low + high);
        std::unique_ptr<mtc::Task> candidate_task;
        std::vector<geometry_msgs::msg::PoseStamped> candidate_targets;
        std::vector<CutSegment> candidate_segments;
        std::string candidate_failure;
        if (planHandoffAt(
              snapshot, corrections, motion_profile, mid, candidate_name(mid), candidate_task,
              candidate_targets, candidate_segments, candidate_failure)) {
          best_fraction = mid;
          best_task = std::move(candidate_task);
          best_targets = std::move(candidate_targets);
          best_segments = std::move(candidate_segments);
          low = mid;
        } else {
          high = mid;
          probe_failure = std::move(candidate_failure);
        }
      }
    }
    retainPlan(snapshot, best_segments, best_targets, std::move(best_task));
    const bool automatic = fraction == 0.0;
    message =
      "Snapshot-relative T-cut handoff " + std::string(automatic ? "automatic " : "") +
      "dry-run plan [" + motion_profile + "] published to RViz; "
      "waypoint_order=approach->back_middle_left->front_middle_left->finish->transition"
      "->approach->seam_start->middle_seam_handoff_end; " +
      (automatic ? "automatic_selection selected middle_handoff_fraction=" +
         formatDouble(best_fraction, 2) + "; search_tolerance=" +
         formatDouble(kAutoTolerance, 2) :
       "middle_handoff_fraction=" + formatDouble(best_fraction, 2)) +
      "; the retained middle traverse is an executable first-arm prefix "
      "ending at the declared handoff point and is not the complete seam; "
      "side_knife_facing=opposite_travel; pre-cut stabilization_s=" +
      formatDouble(pre_cut_stabilization_duration_s_, 3) + "; dwell_count=" +
      std::to_string(pre_cut_stabilization_duration_s_ > 0.0 ? best_segments.size() : 0U) +
      "; execution remains disabled until execution_enabled is set";
    return true;
  }


  bool planCombinedSnapshot(
    const Snapshot& snapshot, const std::map<std::string, CutDebugCorrection>& corrections,
    const std::string& motion_profile, bool omit_right, std::string& message)
  {
    std::vector<CutSegment> segments{
      {"left side cut", snapshot.back_middle_left, snapshot.front_middle_left},
      {"middle seam cut", snapshot.seam_start, snapshot.seam_end},
    };
    if (!omit_right) {
      segments.push_back({"right side cut", snapshot.back_middle_right, snapshot.front_middle_right});
    }
    std::unique_ptr<mtc::Task> task;
    std::vector<geometry_msgs::msg::PoseStamped> cut_targets;
    SegmentPlanOptions options;
    options.timeout_s = planning_timeout_s_;
    if (!planCutSegments(
          snapshot, segments, corrections, motion_profile,
          omit_right ? "Snapshot-relative T-cut canonical waypoint plan" :
            "Snapshot-relative combined canonical waypoint plan", options, task, cut_targets,
          message)) {
      return false;
    }
    retainPlan(snapshot, segments, cut_targets, std::move(task));
    message =
      "Snapshot-relative " + std::string(omit_right ? "T-cut" : "combined") +
      " dry-run plan [" + motion_profile + "] published to RViz; waypoint_order=approach"
      "->back_middle_left->front_middle_left->finish->transition->approach->seam_start"
      "->seam_end->finish" +
      (omit_right ? "" : "->transition->approach->back_middle_right->front_middle_right->finish") +
      "; side_knife_facing=opposite_travel; pre-cut stabilization_s=" +
      formatDouble(pre_cut_stabilization_duration_s_, 3) + "; dwell_count=" +
      std::to_string(pre_cut_stabilization_duration_s_ > 0.0 ? segments.size() : 0U) +
      "; execution remains disabled until execution_enabled is set";
    return true;
  }

  struct CutPlanners {
    mtc::solvers::PlannerInterfacePtr transit;
    mtc::solvers::PlannerInterfacePtr stroke;
    std::shared_ptr<mtc::solvers::CartesianPath> cartesian;
  };
  // Planner stack for one task. Free-space transits run at their own scaling so transit
  // speed can be tuned without ever raising physical cutting speed, and are wrapped by
  // makeTransitPlanner. Contact strokes use Cartesian (min_fraction=1.0, never partial)
  // or Pilz LIN at the contact scaling.
  CutPlanners makeCutPlanners(const std::string& motion_profile, double timeout_s)
  {
    const auto pipeline = [this, timeout_s](
      const std::string& pipeline_name, const std::string& planner_id, bool transit) {
      auto planner = std::make_shared<mtc::solvers::PipelinePlanner>(
        shared_from_this(), pipeline_name);
      planner->setTimeout(timeout_s);
      planner->setMaxVelocityScalingFactor(transit ? transit_velocity_scaling_ : velocity_scaling_);
      planner->setMaxAccelerationScalingFactor(
        transit ? transit_acceleration_scaling_ : acceleration_scaling_);
      if (!planner_id.empty()) planner->setPlannerId(planner_id);
      return planner;
    };
    CutPlanners planners;
    planners.cartesian = std::make_shared<mtc::solvers::CartesianPath>();
    planners.cartesian->setIKFrame(tool_link_);
    planners.cartesian->setStepSize(cartesian_step_m_);
    planners.cartesian->setJumpThreshold(0.0);
    planners.cartesian->setMinFraction(1.0);
    planners.cartesian->setMaxVelocityScalingFactor(velocity_scaling_);
    planners.cartesian->setMaxAccelerationScalingFactor(acceleration_scaling_);
    mtc::solvers::PlannerInterfacePtr transit;
    if (motion_profile == "ompl_cartesian") {
      transit = pipeline(planning_pipeline_, "", true);
      planners.stroke = planners.cartesian;
    } else if (motion_profile == "ompl_pilz") {
      transit = pipeline(planning_pipeline_, "", true);
      planners.stroke = pipeline(pilz_pipeline_, "LIN", false);
    } else if (motion_profile == "pilz_only") {
      transit = pipeline(pilz_pipeline_, "PTP", true);
      planners.stroke = pipeline(pilz_pipeline_, "LIN", false);
    } else if (motion_profile == "chomp_pilz") {
      transit = pipeline(chomp_pipeline_, "", true);
      planners.stroke = pipeline(pilz_pipeline_, "LIN", false);
    } else {
      throw std::invalid_argument("unsupported H-cut motion_profile: " + motion_profile);
    }
    planners.transit = makeTransitPlanner(transit);
    return planners;
  }

  bool planCutSegments(
    const Snapshot& snapshot, const std::vector<CutSegment>& segments,
    const std::map<std::string, CutDebugCorrection>& active_corrections,
    const std::string& motion_profile, const std::string& task_name,
    const SegmentPlanOptions& options, std::unique_ptr<mtc::Task>& task,
    std::vector<geometry_msgs::msg::PoseStamped>& cut_targets, std::string& message)
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
    // A cut-debug start may begin while the arm is still in contact from an earlier
    // session, so the carton is added with the scoped tool/carton allowance in the same
    // atomic scene change; whole plans always start free and keep add-then-allow ordering.
    const bool contact_allowed_from_start = options.cut_debug_start && allow_tool_carton_contact_;
    if (contact_allowed_from_start) {
      add_carton->allowCollisions(tool_link_, carton_collision_id_, true);
    }
    task->add(std::move(add_carton));
    addCameraPole(*task);
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

    const CutPlanners planners = makeCutPlanners(motion_profile, options.timeout_s);
    const auto& transit_planner = planners.transit;
    const auto& stroke_planner = planners.stroke;
    if (!options.skip_supplied_approach && !approach_joint_degrees_.empty()) {
      auto approach = std::make_unique<mtc::stages::MoveTo>(
        "transit to supplied link6 approach [" + motion_profile + "]", transit_planner);
      approach->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
      approach->setGroup(planning_group_);
      approach->setGoal(jointGoalDegrees(joint_group, approach_joint_degrees_));
      task->add(std::move(approach));
    }
    cut_targets.clear();
    for (std::size_t segment_index = 0; segment_index < segments.size(); ++segment_index) {
      const auto& segment = segments[segment_index];
      const bool is_middle_seam = segment.name.rfind("middle seam", 0) == 0U;
      const bool is_right_side = segment.name.rfind("right side cut", 0) == 0U;
      std::string segment_cut;
      if (segment.name.rfind("left side cut", 0) == 0U) segment_cut = "left";
      else if (is_middle_seam) segment_cut = "middle";
      else if (is_right_side) segment_cut = "right";
      const bool is_left_side = segment_cut == "left";
      geometry_msgs::msg::Point corrected_start_point = segment.start;
      geometry_msgs::msg::Point corrected_end_point = segment.end;
      const CutDebugCorrection* segment_correction = nullptr;
      if (!segment_cut.empty()) {
        const auto correction_entry = active_corrections.find(segment_cut);
        if (correction_entry != active_corrections.end()) {
          segment_correction = &correction_entry->second;
        }
      }
      if (segment_correction != nullptr) {
        std::array<double, 3> correction_along{};
        std::array<double, 3> correction_across{};
        std::array<double, 3> correction_normal{};
        std::string basis_error;
        if (!cutDebugBasis(
              snapshot, segment_cut, correction_along, correction_across, correction_normal,
              basis_error)) {
          throw std::runtime_error("cut debug correction basis failed: " + basis_error);
        }
        const CutLocalDeltas& local_start = segment_correction->start_local;
        CutLocalDeltas local_end = segment_correction->end_local;
        if (segment.handoff_end && segment_cut == "middle") {
          const double fraction = segment.handoff_fraction;
          local_end.along_m = local_start.along_m +
            fraction * (segment_correction->end_local.along_m - local_start.along_m);
          local_end.across_m = local_start.across_m +
            fraction * (segment_correction->end_local.across_m - local_start.across_m);
          local_end.normal_m = local_start.normal_m +
            fraction * (segment_correction->end_local.normal_m - local_start.normal_m);
        }
        corrected_start_point = applyCutLocalDelta(
          corrected_start_point, correction_along, correction_across, correction_normal,
          local_start);
        corrected_end_point = applyCutLocalDelta(
          corrected_end_point, correction_along, correction_across, correction_normal,
          local_end);
      }
      const double tilt_deg = is_middle_seam ? 0.0 :
        (is_right_side ? right_side_cut_tilt_deg_ : left_side_cut_tilt_deg_);
      // Side strokes travel back-to-front, but their nominal knife frame faces 180 degrees
      // away from travel. Position corrections must not rotate that nominal baseline, so
      // the direction comes from the uncorrected segment. Learned orientation records
      // bypass the nominal frame below: their complete taught TCP quaternions are mapped
      // from the recorded cut basis into the current cut basis.
      const auto orientation_targets = orientationTargets(
        snapshot, is_middle_seam, segment.start, segment.end, corrected_start_point,
        corrected_end_point);
      const double segment_cut_clearance_m =
        is_middle_seam ? middle_cut_clearance_m_ : cut_clearance_m_;
      const double target_clearance_m =
        segment.hover_only ? debug_motion_clearance_m_ : segment_cut_clearance_m;
      const double approach_clearance_m =
        segment.hover_only ? debug_motion_clearance_m_ : hover_clearance_m_;
      // Left hover poses are the cut poses withdrawn laterally (below), so they are not
      // derived from the hover clearance.
      const bool left_lateral_hover = is_left_side && !segment.hover_only;
      auto cut_start = targetPose(
        snapshot, corrected_start_point, orientation_targets.first, target_clearance_m,
        tilt_deg);
      auto cut_end = targetPose(
        snapshot, corrected_end_point, orientation_targets.second, target_clearance_m,
        tilt_deg);
      geometry_msgs::msg::PoseStamped hover_start;
      geometry_msgs::msg::PoseStamped hover_end;
      if (!left_lateral_hover) {
        hover_start = targetPose(
          snapshot, corrected_start_point, orientation_targets.first, approach_clearance_m,
          tilt_deg);
        hover_end = targetPose(
          snapshot, corrected_end_point, orientation_targets.second, approach_clearance_m,
          tilt_deg);
      }
      if (segment.force_fixed_orientation) {
        geometry_msgs::msg::Quaternion fixed;
        fixed.x = 0.0;
        fixed.y = 1.0;
        fixed.z = 0.0;
        fixed.w = 0.0;
        hover_start.pose.orientation = fixed;
        cut_start.pose.orientation = fixed;
        hover_end.pose.orientation = fixed;
        cut_end.pose.orientation = fixed;
      }
      if (segment_correction != nullptr && !segment.force_fixed_orientation) {
        geometry_msgs::msg::Quaternion handoff_end_observed =
          segment_correction->observed_end;
        if (segment.handoff_end && segment_cut == "middle") {
          handoff_end_observed = interpolateTaughtOrientation(
            segment_correction->observed_start, segment_correction->observed_end,
            segment.handoff_fraction);
        }
        const auto apply_observed = [&](
          const geometry_msgs::msg::Quaternion& observed, geometry_msgs::msg::PoseStamped& pose) {
          return applyCutDebugObservedOrientation(
            snapshot, segment_cut, *segment_correction, observed, pose, message);
        };
        if ((!left_lateral_hover &&
             !apply_observed(segment_correction->observed_start, hover_start)) ||
            !apply_observed(segment_correction->observed_start, cut_start) ||
            (!left_lateral_hover && !apply_observed(handoff_end_observed, hover_end)) ||
            !apply_observed(handoff_end_observed, cut_end)) {
          throw std::runtime_error(message);
        }
      }
      if (left_lateral_hover) {
        hover_start = cut_start;
        hover_end = cut_end;
        if (!offsetLeftPoseOutward(
            snapshot, hover_start, left_lateral_standoff_m_, message) ||
            !offsetLeftPoseOutward(
            snapshot, hover_end, left_lateral_standoff_m_, message)) {
          throw std::runtime_error(message);
        }
      }
      if (is_left_side && segment_correction != nullptr &&
          !segment.force_fixed_orientation && left_adaptive_orientation_enabled_) {
        // All probed poses are at their final positions here, so every IK probe below
        // covers the exact pose the MTC stages will target. A cut-debug start plans only
        // its start poses, so only those are probed.
        std::vector<std::pair<geometry_msgs::msg::PoseStamped*, const char*> > left_poses;
        std::string pose_context;
        if (options.cut_debug_start) {
          left_poses = {
            {&hover_start, "left cut-debug hover start"},
            {&cut_start, "left cut-debug planned start"}};
          pose_context = "cut debug " + segment_cut + " start";
        } else {
          left_poses = {
            {&hover_start, "left hover start"},
            {&cut_start, "left cut start"},
            {&hover_end, "left hover end"},
            {&cut_end, "left cut end"}};
          pose_context = "left " + segment.name + " hover/cut start+end";
        }
        if (!adaptLeftCorrectedOrientations(
              snapshot, joint_group, ik_state, left_poses, pose_context, message)) {
          if (!options.cut_debug_start) {
            message = "left orientation adaptation failed for segment " + segment.name + ": " +
              message;
          }
          return false;
        }
      }
      if (segment.force_fixed_orientation) {
        moveit::core::RobotState debug_ik_state(ik_state);
        if (!debug_ik_state.setFromIK(joint_group, cut_start.pose, tool_link_, 1.0)) {
          message =
            "fixed debug orientation (x=0,y=1,z=0,w=0) has no kinematic IK solution "
            "at the short-stroke start; collision checking was not involved";
          return false;
        }
        if (!debug_ik_state.setFromIK(joint_group, cut_end.pose, tool_link_, 1.0)) {
          message =
            "fixed debug orientation (x=0,y=1,z=0,w=0) has no kinematic IK solution "
            "at the short-stroke end; collision checking was not involved";
          return false;
        }
      }
      if (is_right_side) {
        moveit::core::RobotState approach_ik_state(ik_state);
        if (!approach_ik_state.setFromIK(joint_group, hover_start.pose, tool_link_, 0.25)) {
          message = "right-side approach IK infeasible before Cartesian stroke";
          return false;
        }
      }
      auto entry_alignment = hover_start;
      if (is_middle_seam && middle_entry_alignment_m_ > 0.0) {
        const double dx = corrected_end_point.x - corrected_start_point.x;
        const double dy = corrected_end_point.y - corrected_start_point.y;
        const double dz = corrected_end_point.z - corrected_start_point.z;
        const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (length <= 0.0) throw std::runtime_error("zero-length middle seam direction");
        entry_alignment.pose.position.x -= middle_entry_alignment_m_ * dx / length;
        entry_alignment.pose.position.y -= middle_entry_alignment_m_ * dy / length;
        entry_alignment.pose.position.z -= middle_entry_alignment_m_ * dz / length;
      }
      cut_targets.push_back(cut_start);
      if (!segment.stop_after_start) cut_targets.push_back(cut_end);
      {
        const std::string first_approach = is_left_side ?
          "approach pose outside " + segment.name : "approach pose above " + segment.name;
        const std::string transit_name = segment_index == 0U ? first_approach :
          "transition from " + segments[segment_index - 1U].name + " finish pose to " +
          segment.name + " approach pose";
        auto transit = std::make_unique<mtc::stages::MoveTo>(
          transit_name + " [" + motion_profile + "]", transit_planner);
        transit->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
        transit->setGroup(planning_group_);
        transit->setIKFrame(tool_link_);
        if (motion_profile == "chomp_pilz") {
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
      if (pre_cut_stabilization_duration_s_ > 0.0) {
        task->add(std::make_unique<StationaryDwell>(
          "stabilize blade before " + segment.name, planning_group_,
          pre_cut_stabilization_duration_s_));
      }
      if (!segment.hover_only) {
        if (allow_tool_carton_contact_ && !contact_allowed_from_start) {
          auto allow_contact = std::make_unique<mtc::stages::ModifyPlanningScene>(
            "allow knife-tip contact with carton for " + segment.name);
          allow_contact->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
          allow_contact->allowCollisions(tool_link_, carton_collision_id_, true);
          task->add(std::move(allow_contact));
        }
        const std::string entry_label = is_left_side ? "enter laterally into " :
          "descend from aligned ";
        auto descend = std::make_unique<mtc::stages::MoveTo>(
          entry_label + segment.name + " start [" + motion_profile + "]", stroke_planner);
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
      // Terminal handoff: the left arm stops at the cut-level handoff endpoint after the
      // traverse above. Skip the Cartesian hover retract and leave the planned tool/carton
      // contact allowance active; a second arm (not modeled here) resumes from this pose.
      if (!segment.hover_only && !segment.terminal_after_cut) {
        const std::string finish_label = is_left_side ? "finish pose outside " :
          "finish pose above ";
        auto ascend = std::make_unique<mtc::stages::MoveTo>(
          finish_label + segment.name + " [" + motion_profile + "]", stroke_planner);
        ascend->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
        ascend->setGroup(planning_group_);
        ascend->setIKFrame(tool_link_);
        ascend->setGoal(hover_end);
        task->add(std::move(ascend));
        const bool lift_before_middle = is_left_side &&
          (segment_index + 1U) < segments.size() &&
          segments[segment_index + 1U].name.rfind("middle seam", 0) == 0U &&
          !segments[segment_index + 1U].hover_only;
        if (lift_before_middle) {
          // Safety: pull laterally clear first (ascend above), then lift along the
          // snapshot surface normal to hover clearance while holding the exact left
          // finish orientation. Collision restore and the free-space transit/rotation
          // to the middle approach happen only after this lift.
          auto lift_target = hover_end;
          const double lift_m = hover_clearance_m_ - cut_clearance_m_;
          lift_target.pose.position.x += lift_m * snapshot.surface_normal.x;
          lift_target.pose.position.y += lift_m * snapshot.surface_normal.y;
          lift_target.pose.position.z += lift_m * snapshot.surface_normal.z;
          auto lift = std::make_unique<mtc::stages::MoveTo>(
            "lift clear of " + segment.name + " to hover [cartesian]", planners.cartesian);
          lift->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
          lift->setGroup(planning_group_);
          lift->setIKFrame(tool_link_);
          lift->setGoal(lift_target);
          task->add(std::move(lift));
        }
        if (allow_tool_carton_contact_) {
          auto forbid_contact = std::make_unique<mtc::stages::ModifyPlanningScene>(
            "restore knife-tip collision checking after " + segment.name);
          forbid_contact->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
          forbid_contact->allowCollisions(tool_link_, carton_collision_id_, false);
          task->add(std::move(forbid_contact));
        }
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
      if (options.explain_failure) {
        task->explainFailure();
      }
      message = "MTC candidate failed: task=" + task_name + " profile=" + motion_profile +
                " segments=" + std::to_string(segments.size());
      return false;
    }
    // Serialized once and shared by both fail-closed solution validators.
    const auto& solution = *task->solutions().front();
    moveit_task_constructor_msgs::msg::Solution solution_message;
    solution.toMsg(solution_message);
    const std::size_t expected_dwell_count =
      pre_cut_stabilization_duration_s_ > 0.0 ? segments.size() : 0U;
    std::string dwell_error;
    if (!validateStationaryDwells(solution_message, expected_dwell_count, dwell_error)) {
      message = "planned trajectory has invalid pre-cut stabilization: " + dwell_error;
      return false;
    }
    std::string wrist_error;
    if (!validateFreeSpaceTransitWrist(solution, solution_message, wrist_error)) {
      message = wrist_error;
      return false;
    }
    task->introspection().publishSolution(solution);
    return true;
  }

  bool validateStationaryDwells(
    const moveit_task_constructor_msgs::msg::Solution& solution_message,
    std::size_t expected_count, std::string& error) const
  {
    const auto seconds = [](const builtin_interfaces::msg::Duration& duration) {
      return static_cast<double>(duration.sec) + 1e-9 * duration.nanosec;
    };
    const auto all_zero = [](const std::vector<double>& values, std::size_t expected_size) {
      return values.size() == expected_size &&
        std::all_of(values.begin(), values.end(), [](double value) {
          return std::isfinite(value) && std::abs(value) <= 1e-12;
        });
    };

    std::size_t dwell_count = 0U;
    for (const auto& sub_trajectory : solution_message.sub_trajectory) {
      if (sub_trajectory.info.planner_id != "stationary_dwell") continue;
      ++dwell_count;
      const auto& points = sub_trajectory.trajectory.joint_trajectory.points;
      if (points.size() != 2U || points.front().positions.empty() ||
          points.back().positions.size() != points.front().positions.size()) {
        error = "dwell must contain two complete joint waypoints";
        return false;
      }
      const std::size_t variable_count = points.front().positions.size();
      for (std::size_t index = 0U; index < variable_count; ++index) {
        if (!std::isfinite(points.front().positions[index]) ||
            !std::isfinite(points.back().positions[index]) ||
            std::abs(points.front().positions[index] - points.back().positions[index]) > 1e-12) {
          error = "dwell joint positions are not stationary";
          return false;
        }
      }
      if (!all_zero(points.front().velocities, variable_count) ||
          !all_zero(points.back().velocities, variable_count) ||
          !all_zero(points.front().accelerations, variable_count) ||
          !all_zero(points.back().accelerations, variable_count)) {
        error = "dwell velocities and accelerations must be explicit zeros";
        return false;
      }
      if (std::abs(seconds(points.front().time_from_start)) > 1e-9 ||
          std::abs(seconds(points.back().time_from_start) -
            pre_cut_stabilization_duration_s_) > 1e-6) {
        error = "dwell waypoint timing does not match requested duration";
        return false;
      }
    }
    if (dwell_count != expected_count) {
      error = "expected " + std::to_string(expected_count) + " dwell trajectories, found " +
        std::to_string(dwell_count);
      return false;
    }
    return true;
  }

  // Free-space transit stack: resolve pose goals to the IK branch nearest the
  // propagated start, then try the shortest validity-checked joint-space
  // interpolation first and the profile's existing planner second. Cutting
  // strokes never use this stack.
  mtc::solvers::PlannerInterfacePtr makeTransitPlanner(
    const mtc::solvers::PlannerInterfacePtr& fallback) const
  {
    auto direct = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
    direct->setMaxVelocityScalingFactor(transit_velocity_scaling_);
    direct->setMaxAccelerationScalingFactor(transit_acceleration_scaling_);
    auto multi = std::make_shared<mtc::solvers::MultiPlanner>();
    multi->push_back(direct);
    multi->push_back(fallback);
    // Pose goals are resolved to the nearest exact IK branch before either planner runs.
    return std::make_shared<NearestBranchTransitPlanner>(multi);
  }

  // Fail-closed wrist guard for free-space transits: rejects the retained solution
  // when a free-space MoveTo transit winds joint6 through more than pi radians
  // of total travel. Travel is summed over every consecutive waypoint pair so
  // intermediate winding cannot hide behind benign endpoints. Cutting, align,
  // descend, ascend, lift, and dwell stages are never examined here.
  bool validateFreeSpaceTransitWrist(
    const mtc::SolutionBase& solution,
    const moveit_task_constructor_msgs::msg::Solution& solution_message,
    std::string& error) const
  {
    const auto* sequence = dynamic_cast<const mtc::SolutionSequence*>(&solution);
    if (sequence == nullptr) {
      error = "planned free-space transit winds the wrist: unexpected solution type; " +
        std::string("refusing to guess transit stages");
      return false;
    }
    std::vector<const mtc::SubTrajectory*> leaves;
    std::function<void(const mtc::SolutionBase*)> collect = [&](const mtc::SolutionBase* node) {
      if (node == nullptr) return;
      if (const auto* wrapped = dynamic_cast<const mtc::WrappedSolution*>(node)) {
        collect(wrapped->wrapped());
        return;
      }
      if (const auto* nested = dynamic_cast<const mtc::SolutionSequence*>(node)) {
        for (const auto* child : nested->solutions()) collect(child);
        return;
      }
      if (const auto* sub = dynamic_cast<const mtc::SubTrajectory*>(node)) {
        leaves.push_back(sub);
      }
    };
    for (const auto* child : sequence->solutions()) collect(child);
    if (leaves.size() != solution_message.sub_trajectory.size()) {
      error = "planned free-space transit winds the wrist: solution trajectory count does " +
        std::string("not match planned stages; refusing to guess");
      return false;
    }
    const auto is_free_space_transit = [](const std::string& stage_name) {
      return stage_name.rfind("transit to supplied", 0) == 0U ||
        stage_name.rfind("approach pose above ", 0) == 0U ||
        stage_name.rfind("approach pose outside ", 0) == 0U ||
        stage_name.rfind("transition from ", 0) == 0U;
    };
    constexpr double kMaxWristTravelRad = 3.14159265358979323846;
    for (std::size_t index = 0; index < leaves.size(); ++index) {
      const std::string stage_name = leaves[index]->creator() != nullptr ?
        leaves[index]->creator()->name() : std::string();
      if (!is_free_space_transit(stage_name)) continue;
      const auto& sub_msg = solution_message.sub_trajectory[index];
      const auto& joint_names = sub_msg.trajectory.joint_trajectory.joint_names;
      const auto& points = sub_msg.trajectory.joint_trajectory.points;
      const auto find_joint = [&](const std::string& name) -> std::optional<std::size_t> {
        const auto it = std::find(joint_names.begin(), joint_names.end(), name);
        if (it == joint_names.end()) return std::nullopt;
        return static_cast<std::size_t>(std::distance(joint_names.begin(), it));
      };
      const auto joint6 = find_joint("joint6");
      if (!joint6) {
        error = "planned free-space transit winds the wrist: transit '" + stage_name +
          "' has no joint6 waypoints";
        return false;
      }
      if (points.size() < 2U) {
        error = "planned free-space transit winds the wrist: transit '" + stage_name +
          "' has missing joint waypoints";
        return false;
      }
      double travel6 = 0.0;
      for (std::size_t point = 1U; point < points.size(); ++point) {
        const auto& prev = points[point - 1U].positions;
        const auto& curr = points[point].positions;
        if (prev.size() != joint_names.size() || curr.size() != joint_names.size() ||
            *joint6 >= prev.size() || *joint6 >= curr.size()) {
          error = "planned free-space transit winds the wrist: transit '" + stage_name +
            "' has incomplete joint waypoints";
          return false;
        }
        const double before6 = prev[*joint6];
        const double after6 = curr[*joint6];
        if (!std::isfinite(before6) || !std::isfinite(after6)) {
          error = "planned free-space transit winds the wrist: transit '" + stage_name +
            "' has non-finite wrist waypoints";
          return false;
        }
        travel6 += std::abs(after6 - before6);
      }
      if (travel6 > kMaxWristTravelRad) {
        std::ostringstream details;
        details << "planned free-space transit winds the wrist: transit '" << stage_name <<
          "' joint6 travel " << travel6 <<
          " rad exceeds " << kMaxWristTravelRad << " rad";
        error = details.str();
        return false;
      }
    }
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

  geometry_msgs::msg::PoseStamped targetPose(
    const Snapshot& snapshot, const geometry_msgs::msg::Point& point,
    const geometry_msgs::msg::Point& direction_target, double clearance_m,
    double tilt_deg = 0.0) const
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

    const geometry_msgs::msg::Quaternion q =
      normalizedQuaternion(basisQuaternion(x_axis, y_axis, tilted_z_axis));
    geometry_msgs::msg::PoseStamped pose;
    pose.header = snapshot.header;
    pose.pose.position.x = point.x + clearance_m * snapshot.surface_normal.x;
    pose.pose.position.y = point.y + clearance_m * snapshot.surface_normal.y;
    pose.pose.position.z = point.z + clearance_m * snapshot.surface_normal.z;
    pose.pose.orientation = q;
    if (debug_fixed_orientation_) {
      pose.pose.orientation.x = 0.0;
      pose.pose.orientation.y = 1.0;
      pose.pose.orientation.z = 0.0;
      pose.pose.orientation.w = 0.0;
    }
    return pose;
  }
  // Rotation whose columns are the given frame axes (Eigen's rotation-matrix conversion).
  static geometry_msgs::msg::Quaternion basisQuaternion(
    const std::array<double, 3>& x_axis, const std::array<double, 3>& y_axis,
    const std::array<double, 3>& z_axis)
  {
    Eigen::Matrix3d basis;
    basis.col(0) = Eigen::Vector3d(x_axis[0], x_axis[1], x_axis[2]);
    basis.col(1) = Eigen::Vector3d(y_axis[0], y_axis[1], y_axis[2]);
    basis.col(2) = Eigen::Vector3d(z_axis[0], z_axis[1], z_axis[2]);
    const Eigen::Quaterniond rotation(basis);
    geometry_msgs::msg::Quaternion q;
    q.x = rotation.x();
    q.y = rotation.y();
    q.z = rotation.z();
    q.w = rotation.w();
    return q;
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

  void publishMarkers(
    const std::vector<geometry_msgs::msg::PoseStamped>& cut_targets,
    const std::vector<CutSegment>& segments)
  {
    visualization_msgs::msg::MarkerArray array;
    std::size_t target_index = 0U;
    for (std::size_t segment_index = 0U; segment_index < segments.size(); ++segment_index) {
      const auto& segment = segments[segment_index];
      const std::size_t target_count = segment.stop_after_start ? 1U : 2U;
      if (target_index + target_count > cut_targets.size()) {
        RCLCPP_ERROR(get_logger(), "Cannot publish RViz marker: targets do not match named cut segments");
        break;
      }
      const bool middle_seam = segment.name.find("middle seam") != std::string::npos;
      visualization_msgs::msg::Marker marker;
      marker.header = cut_targets[target_index].header;
      marker.ns = "frozen_h_cut/" + segment.name;
      marker.id = static_cast<int>(segment_index);
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.color.r = middle_seam ? 0.0F : 1.0F;
      marker.color.g = middle_seam ? 1.0F : 0.4F;
      marker.color.b = middle_seam ? 1.0F : 0.0F;
      marker.color.a = 1.0F;
      if (segment.stop_after_start) {
        marker.type = visualization_msgs::msg::Marker::SPHERE;
      } else {
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.points = {
          cut_targets[target_index].pose.position,
          cut_targets[target_index + 1U].pose.position};
        marker.scale.x = 0.008;
      }
      array.markers.push_back(marker);
      target_index += target_count;
    }
    if (target_index != cut_targets.size()) {
      RCLCPP_ERROR(get_logger(), "Cannot publish RViz marker: unnamed cut targets remain");
    }
    marker_pub_->publish(array);
  }
  std::mutex plan_mutex_;
  std::mutex state_mutex_;
  std::mutex camera_pole_mutex_;
  std::optional<sensor_msgs::msg::JointState> current_joint_state_;
  std::optional<moveit_msgs::msg::CollisionObject> camera_pole_;
  rclcpp::Time joint_state_received_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr camera_pole_sub_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr target_pose_pub_;
  rclcpp::Publisher<moveit_msgs::msg::CollisionObject>::SharedPtr carton_collision_pub_;
  std::string cut_debug_cut_, cut_debug_output_dir_;
  std::string cut_debug_left_correction_file_, cut_debug_middle_correction_file_,
    cut_debug_right_correction_file_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr section_diagnostic_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr execute_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cut_debug_start_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cut_debug_save_service_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr cut_pattern_callback_;
  std::optional<Snapshot> planned_snapshot_;
  CutDebugStage cut_debug_stage_{CutDebugStage::kNone};
  std::string cut_debug_session_cut_;
  std::optional<Snapshot> cut_debug_snapshot_;
  geometry_msgs::msg::PoseStamped cut_debug_nominal_start_;
  geometry_msgs::msg::PoseStamped cut_debug_nominal_end_;
  geometry_msgs::msg::PoseStamped cut_debug_observed_start_;
  geometry_msgs::msg::PoseStamped cut_debug_observed_end_;
  bool cut_debug_has_observed_start_{false};
  bool cut_debug_has_observed_end_{false};
  bool cut_debug_dispatch_failed_{false};
  CutLocalDeltas cut_debug_start_local_{};
  CutLocalDeltas cut_debug_end_local_{};
  std::string cut_debug_record_path_;
  std::int64_t cut_debug_created_stamp_ns_{0};
  moveit::core::RobotModelConstPtr cut_debug_robot_model_;
  std::unique_ptr<mtc::Task> planned_task_;
  std::thread execution_thread_;
  std::string cut_pattern_, side_cut_orientation_mode_;
  std::string planning_frame_, planning_group_, tool_link_, planning_pipeline_, chomp_pipeline_;
  std::string pilz_pipeline_, carton_collision_id_, camera_pole_topic_;
  double external_snapshot_max_age_s_, carton_height_m_, hover_clearance_m_,
    left_lateral_standoff_m_, cut_clearance_m_, middle_cut_clearance_m_,
    debug_motion_distance_m_, debug_motion_clearance_m_,
    middle_entry_alignment_m_,
    pre_cut_stabilization_duration_s_, left_side_cut_tilt_deg_, right_side_cut_tilt_deg_,
    left_adaptive_orientation_max_deg_, left_adaptive_orientation_step_deg_,
    t_cut_middle_handoff_fraction_, cartesian_step_m_, planning_timeout_s_, velocity_scaling_, acceleration_scaling_,
    transit_velocity_scaling_, transit_acceleration_scaling_;
  std::vector<double> approach_joint_degrees_;
  int max_solutions_;
  bool debug_fixed_orientation_{false},
    debug_allow_stale_snapshot_{false}, camera_pole_required_{true},
    allow_tool_carton_contact_{false};
  bool cut_debug_correction_enabled_{false};
  bool left_adaptive_orientation_enabled_{true};
  bool execution_in_progress_{false};
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
