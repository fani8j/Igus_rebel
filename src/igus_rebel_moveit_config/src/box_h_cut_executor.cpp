#include <Eigen/Geometry>
#include <control_msgs/msg/joint_jog.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/cartesian_interpolator.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/task_constructor/container.h>
#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/joint_interpolation.h>
#include <moveit/task_constructor/solvers/multi_planner.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stage.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/stages/move_relative.h>
#include <moveit/task_constructor/task.h>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "igus_rebel_moveit_config/carton_height.hpp"
#include "igus_rebel_moveit_config/rebel_analytic_ik.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

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
// are the start state and its in-bounds spherical-wrist flip (joint4 +/- pi, -joint5,
// joint6 +/- pi); every IK result is expanded with its in-bounds wrist flip, each
// re-verified by forward kinematics, bounds and collision. (joint6 +/- 2pi variants are
// never generated: joint6 is limited to about +/-3.124 rad, so they are always out of
// bounds.) Candidates whose joint6 change would exceed pi are ranked last, then those
// with any joint within 2 deg of its URDF bound, then by L1 joint distance, so the
// transit takes the short wrist branch. If every branch fails,
// the original pose goal is retried once. All attempts share one time budget of
// min(stage timeout, max_total_s). The target pose is never altered and the trajectory
// is still produced by the wrapped planner; the solution-level wrist guard remains
// authoritative.
class NearestBranchTransitPlanner : public mtc::solvers::PlannerInterface
{
public:
  NearestBranchTransitPlanner(mtc::solvers::PlannerInterfacePtr inner, double max_total_s)
  : inner_(std::move(inner)), max_total_s_(max_total_s) {}

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
    return inner_->plan(
      from, to, jmg, std::min(timeout, max_total_s_), result, path_constraints);
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
    const auto started = std::chrono::steady_clock::now();
    const double budget_s = std::min(timeout, max_total_s_);
    const auto remaining_s = [&]() {
      return budget_s -
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };
    const auto plan_original_pose = [&]() -> Result {
      const double left_s = remaining_s();
      if (left_s <= 0.0) {
        return Result{false, "transit planning budget of " + std::to_string(budget_s) +
          " s exhausted before the original pose goal could be retried"};
      }
      return inner_->plan(from, link, offset, target, jmg, left_s, result, path_constraints);
    };
    const auto& variables = jmg->getVariableNames();
    const auto index_of = [&](const std::string& name) -> std::optional<std::size_t> {
      const auto it = std::find(variables.begin(), variables.end(), name);
      if (it == variables.end()) return std::nullopt;
      return static_cast<std::size_t>(std::distance(variables.begin(), it));
    };
    const auto j4 = index_of("joint4");
    const auto j5 = index_of("joint5");
    const auto j6 = index_of("joint6");
    if (!j6) return plan_original_pose();

    const moveit::core::RobotState& start = from->getCurrentState();
    const moveit::core::RobotModel& model = *start.getRobotModel();
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
    const auto within_bounds = [&](std::size_t index, double value) {
      const auto& bounds = model.getVariableBounds(variables[index]);
      return !bounds.position_bounded_ ||
        (value >= bounds.min_position_ && value <= bounds.max_position_);
    };
    // Equivalent spherical-wrist branch; only offsets that keep joint4 and joint6 inside
    // their position limits are generated (at most one per joint for limits below pi).
    const auto wrist_variants = [&](const std::vector<double>& values) {
      std::vector<std::vector<double>> variants;
      if (!j4 || !j5) return variants;
      for (const double turn4 : {-kPi, kPi}) {
        if (!within_bounds(*j4, values[*j4] + turn4)) continue;
        for (const double turn6 : {-kPi, kPi}) {
          if (!within_bounds(*j6, values[*j6] + turn6)) continue;
          auto flipped = values;
          flipped[*j4] += turn4;
          flipped[*j5] = -flipped[*j5];
          flipped[*j6] += turn6;
          variants.push_back(flipped);
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
      return plan_original_pose();
    }

    const auto wrist_travel = [&](const std::vector<double>& values) {
      return std::abs(values[*j6] - start_values[*j6]);
    };
    const auto joint_distance = [&](const std::vector<double>& values) {
      double sum = 0.0;
      for (std::size_t i = 0; i < values.size(); ++i) sum += std::abs(values[i] - start_values[i]);
      return sum;
    };
    // A goal branch with any joint within kNearLimitRad of its URDF bound (which is itself
    // 1 deg inside the controller limit) is ranked after branches with margin, so the
    // transit does not park a cut on a wrist branch the controller may refuse.
    constexpr double kNearLimitRad = 2.0 * kPi / 180.0;
    const auto near_limit = [&](const std::vector<double>& values) {
      for (std::size_t i = 0; i < values.size(); ++i) {
        const auto& bounds = model.getVariableBounds(variables[i]);
        if (bounds.position_bounded_ &&
            (values[i] - bounds.min_position_ < kNearLimitRad ||
             bounds.max_position_ - values[i] < kNearLimitRad)) {
          return true;
        }
      }
      return false;
    };
    std::stable_sort(
      candidates.begin(), candidates.end(),
      [&](const std::vector<double>& lhs, const std::vector<double>& rhs) {
        const bool lhs_long = wrist_travel(lhs) > kPi;
        const bool rhs_long = wrist_travel(rhs) > kPi;
        if (lhs_long != rhs_long) return !lhs_long;
        const bool lhs_near = near_limit(lhs);
        const bool rhs_near = near_limit(rhs);
        if (lhs_near != rhs_near) return !lhs_near;
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
      const double left_s = remaining_s();
      if (left_s <= 0.0) {
        last = Result{false, "transit planning budget exhausted"};
        break;
      }
      auto goal_scene = from->diff();
      goal_scene->getCurrentStateNonConst().setJointGroupPositions(jmg, candidate);
      goal_scene->getCurrentStateNonConst().update();
      last = inner_->plan(from, goal_scene, jmg, left_s, result, path_constraints);
      if (last) {
        RCLCPP_INFO(
          logger, "transit planned to IK branch joint6=%.4f (joint6 change %.4f rad)",
          candidate[*j6], wrist_travel(candidate));
        return last;
      }
    }
    RCLCPP_WARN(
      logger, "every seeded IK branch failed (%s); retrying the original pose goal once",
      last.message.c_str());
    const Result retry = plan_original_pose();
    if (retry) return retry;
    return Result{false, last.message + "; original pose goal retry: " + retry.message};
  }

  std::string getPlannerId() const override { return "NearestBranchTransit"; }

private:
  mtc::solvers::PlannerInterfacePtr inner_;
  double max_total_s_;
};

enum class CutKind : std::uint8_t { kNone, kLeft, kMiddle, kRight };
struct CutSegment {
  std::string name;
  // Which detected cut this segment belongs to; selects the correction slot, clearance,
  // tilt, orientation mode and stage shapes. kNone (middle_debug_short) takes no
  // correction and is planned like a side segment with its forced fixed orientation.
  CutKind cut{CutKind::kNone};
  geometry_msgs::msg::Point start;
  geometry_msgs::msg::Point end;
  bool stop_after_start{false};
  bool hover_only{false};
  bool force_fixed_orientation{false};
  bool handoff_end{false};
  bool terminal_after_cut{false};
  double handoff_fraction{0.0};
};
// Left-roll search result for one planning request, reused by every t_cut_handoff probe:
// the left segment, its correction, the snapshot and the seed state are identical
// across probes, so the search result is too.
struct LeftRollCache {
  bool computed{false};
  bool feasible{false};
  std::vector<geometry_msgs::msg::Quaternion> orientations;
  std::string error;
};
struct SegmentPlanOptions {
  // Skip the supplied joint-space approach (approach_joint_degrees); only "h" uses it.
  bool skip_supplied_approach{true};
  // Staged cut-debug start plan: the carton is added together with the scoped
  // tool/carton allowance in one atomic scene change (the arm may still be in contact
  // from an earlier session), and the left roll search probes only the start poses.
  bool cut_debug_start{false};
  // Optional per-request cache of the left roll search (t_cut_handoff probes).
  LeftRollCache* left_roll_cache{nullptr};
  // Side-tape hand-off standoff variant only: for a stop_after_start segment, the final
  // Cartesian entry stage targets this pose INSTEAD OF the segment's true (in-material)
  // cut_start, so a standoff approach never targets, or passes through, a pose behind the
  // outer wall plane. Every upstream stage (approach transit, hover/entry-alignment) is
  // still derived from the true geometry; only the terminal entry target is substituted.
  std::optional<geometry_msgs::msg::PoseStamped> stop_pose_override;
  // Side-tape hand-off planning: the left_parallel search is restricted to candidates within
  // side_tape_handoff_max_search_angle_deg of F0 and ordered smallest-deviation-first.
  bool side_tape_handoff{false};
  // Staged top cut approach: stop after the aligned hover above the segment start (and the
  // pre-cut dwell); the hover pose is returned through hover_start_out.
  bool stop_at_hover_start{false};
  geometry_msgs::msg::PoseStamped* hover_start_out{nullptr};
  // Taught top-cut depth (metres, + = deeper along -surface_normal): added to the MIDDLE seam
  // cut start/end targets only (hover poses unchanged). The lift/finish hover stays at
  // hover_clearance above the top, i.e. straight up along the normal from the deeper cut end.
  double middle_start_extra_depth_m{0.0};
  double middle_end_extra_depth_m{0.0};
  // Fraction of the contact scalings for the cut stroke stage only (1.0 = unchanged).
  double cut_stroke_speed_factor{1.0};
  // Top-cut only: height above the top surface of the MIDDLE seam finish/lift pose (replaces
  // hover_clearance_m for that stage; the start hover is unchanged). Unset = hover_clearance_m.
  std::optional<double> middle_finish_clearance_m;
  // continue_top_cut only: straight Cartesian lift along +Z of the planning frame (orientation
  // kept) right after the current state, BEFORE any free-space transit, so the arm leaves the
  // left-cut withdraw pose upward instead of swinging low over the table (not in the scene).
  double pre_transit_lift_m{0.0};
};
// The candidate the left_parallel search last selected (angles relative to F0).
struct LeftParallelSelection {
  bool valid{false};
  double tilt_deg{0.0}, pitch_deg{0.0}, yaw_deg{0.0}, theta_p_deg{0.0};
  double rotation_from_f0_deg{0.0};
  std::size_t candidate_index{0}, candidate_count{0};
  double maxAbsAngleDeg() const
  {
    return std::max({std::abs(tilt_deg), std::abs(pitch_deg), std::abs(yaw_deg)});
  }
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
  // Optional snapshot key support_depth_below_top_m: top face down to the surface the carton
  // stands on (hidden stand: box + stand). Absent in older snapshots.
  bool has_support_depth{false};
  double support_depth_below_top_m{std::numeric_limits<double>::quiet_NaN()};
  std::string support_depth_source;
};
// Staged cut-debug session: start planned -> start executed (explicit dispatch) ->
// start observation saved (teleop to the endpoint, then save completes the record).
enum class CutDebugStage : std::uint8_t {
  kNone = 0,
  kStartPlanned,
  kStartExecuted,
  kStartSaved,
};
struct CutLocalDeltas {
  double along_m{0.0};
  double across_m{0.0};
  double normal_m{0.0};
};
// Orthonormal right-handed cut frame: along the cut, across it, and the surface normal.
struct CutBasis {
  Eigen::Vector3d along{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d across{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
};
struct CutDebugCorrection {
  std::string cut;
  CutLocalDeltas start_local{};
  CutLocalDeltas end_local{};
  CutBasis recorded{};
  geometry_msgs::msg::Quaternion observed_start{};
  geometry_msgs::msg::Quaternion observed_end{};
};
// Side-tape hand-off (left face only, see CONTEXT.md). kActive: the truncated approach
// dispatch completed successfully; side_tape is expected to be driving the cut.
// kApproachFailed: the approach dispatch did NOT complete (aborted/exception); the arm pose
// is uncertain but the carton and hand-off geometry are kept so side_tape_retract can still
// recover. kNone/kRetracted (same meaning): no hand-off in progress. While status is kActive
// or kApproachFailed, the carton collision object and this record must survive every
// clear/replan attempt; only a successful retract, or an explicit clear_plan after verifying
// the tip is outside the carton (kApproachFailed only), returns to kNone.
// kCutting: ~/continue_side_cut is executing (cut stroke + retract); kCutFailed: that
// execution did not complete (arm somewhere along the stroke/retract), state is kept so
// side_tape_retract can still recover.
enum class SideTapeHandoffStatus : std::uint8_t {
  kNone, kActive, kApproachFailed, kRetracted, kCutting, kCutFailed};
struct SideTapeHandoffInfo {
  SideTapeHandoffStatus status{SideTapeHandoffStatus::kNone};
  bool inProgress() const
  {
    return status == SideTapeHandoffStatus::kActive ||
      status == SideTapeHandoffStatus::kApproachFailed ||
      status == SideTapeHandoffStatus::kCutting || status == SideTapeHandoffStatus::kCutFailed;
  }
  std::string face;
  std::string variant;
  double cut_length_mm{0.0};
  double cut_length_euclidean_mm{0.0};
  double penetration_mm{0.0};
  double standoff_mm{0.0};
  double min_wall_distance_mm{0.0};  // standoff only; NaN for in_material (not applicable)
  geometry_msgs::msg::PoseStamped start_pose;
  // Tool-frame axes used for the orientation check at plan time (echoed into the record).
  Eigen::Vector3d cut_axis_tool{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d lateral_axis_tool{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d depth_axis_tool{Eigen::Vector3d::UnitZ()};
  // Measured base_link directions of those axes at the start pose (unit vectors).
  Eigen::Vector3d cut_dir_base{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d lateral_dir_base{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d depth_dir_base{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d outward_normal{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d plane_point{Eigen::Vector3d::Zero()};
  LeftParallelSelection left_parallel;
  // Planned in-material cut start/end (base_link) from the full-stroke feasibility plan.
  Eigen::Vector3d planned_cut_start{Eigen::Vector3d::Zero()};
  Eigen::Vector3d planned_cut_end{Eigen::Vector3d::Zero()};
  std::string snapshot_path;
  std::string snapshot_id;
  rclcpp::Time stamp;
  // Captured at plan time: a hand-off planned/loaded under debug_allow_stale_snapshot can
  // never be executed at all, including the retract withdraw-only path.
  bool stale_override{false};
};
// Staged top (middle seam) cut. kPlanned: approach retained/dispatching; kApproached: tip at the
// aligned hover; kPenetrated: tip at the planned cut start in material; kDone: stroke + lift
// finished; kFailed: a dispatch did not complete (arm pose uncertain; top_cut_retract recovers).
enum class TopCutStagedStatus : std::uint8_t {
  kNone, kPlanned, kApproached, kPenetrated, kDone, kFailed, kCutting};
struct TopCutStagedInfo {
  TopCutStagedStatus status{TopCutStagedStatus::kNone};
  geometry_msgs::msg::PoseStamped hover_pose, cut_start_pose, cut_end_pose;
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d plane_point{Eigen::Vector3d::Zero()};  // on the top surface, under cut start
  std::string snapshot_path, snapshot_id;
  rclcpp::Time stamp;
  bool stale_override{false};
};
// Every executor parameter value. Read-only entries are fixed at startup; planning
// entries change only through the parameter callback, under plan_mutex_, and clear any
// retained plan. See kParameterTable in BoxHCutExecutor for scopes and defaults.
struct PlannerSettings {
  std::string planning_group, planning_frame, tool_link, planning_pipeline, chomp_pipeline,
    pilz_pipeline, carton_collision_id, camera_pole_topic, side_cut_orientation_mode;
  double external_snapshot_max_age_s{0.0}, carton_height_m{0.0}, carton_height_max_m{2.0}, hover_clearance_m{0.0},
    left_lateral_standoff_m{0.0}, cut_clearance_m{0.0}, middle_cut_clearance_m{0.0},
    middle_entry_alignment_m{0.0}, debug_motion_distance_m{0.0},
    debug_motion_clearance_m{0.0}, cartesian_step_m{0.0}, planning_timeout_s{0.0},
    velocity_scaling{0.0}, acceleration_scaling{0.0}, transit_velocity_scaling{0.0},
    transit_acceleration_scaling{0.0};
  bool allow_tool_carton_contact{false}, camera_pole_required{true}, carton_height_from_snapshot{true};
  std::vector<double> approach_joint_degrees;
  std::int64_t max_solutions{1};
  // Read-only side-tape hand-off/retract safety gates (numeric bounds, never live-tunable).
  double side_tape_handoff_retract_max_orientation_deg{0.0},
    side_tape_handoff_retract_max_line_offset_mm{0.0},
    side_tape_handoff_servo_quiescence_s{0.0},
    side_tape_handoff_joint_velocity_threshold{0.0};
  double side_tape_continue_max_line_offset_mm{0.0}, side_tape_continue_max_along_offset_mm{0.0},
    side_tape_continue_min_depth_mm{0.0}, side_tape_continue_max_depth_mm{0.0},
    side_tape_continue_max_orientation_deg{0.0};
  bool side_tape_continue_check_snapshot_age{true};
  double side_tape_continue_speed_factor{1.0};
  bool middle_blade_reversed{false};
  double side_tape_continue_end_extension_mm{0.0};
  double side_tape_continue_end_extra_depth_mm{0.0};
  bool side_tape_continue_check_end_depth{true};
  std::string side_tape_handoff_servo_twist_topic, side_tape_handoff_servo_joint_topic;
  std::vector<double> top_cut_start_offset_mm{0.0, 0.0, 0.0};
  std::vector<double> top_cut_end_offset_mm{0.0, 0.0, 0.0};
  double top_cut_staged_max_position_mm{10.0}, top_cut_staged_max_orientation_deg{10.0},
    top_cut_staged_speed_factor{1.0};
  // Max blade tilt (rotation about the tool X axis, pivot at the tip) the top-cut lift may use
  // when the straight lift is not reachable; 0 disables tilting.
  double top_cut_lift_max_tilt_deg{10.0};
  // Tip height above the top surface after the top-cut lift (staged / continue top cut only).
  double top_cut_lift_clearance_m{0.003};
  bool top_cut_continue_check_snapshot_age{true};
  double top_cut_pre_transit_lift_m{0.10};
  double top_cut_post_penetration_dwell_s{1.0};
  bool top_cut_allow_shortened_cut{true};
  double top_cut_lift_min_clearance_m{0.001};
  double side_tape_withdraw_min_clearance_m{0.001};
  double top_cut_pre_transit_lift_min_fraction{0.5};
  double top_cut_min_cut_fraction{0.5};
  std::string top_cut_staged_output_file;
  // Phase 1 speed-up (see rebel_analytic_ik.hpp): "analytic" replaces the left parallel
  // search's per-candidate KDL seed/wristVariants/random-restart IK with a closed-form
  // branch enumeration (falls back to "kdl" automatically, with a warning, if the analytic
  // solver's own startup self-check against MoveIt FK does not pass). "kdl" keeps the
  // original behaviour byte-for-byte and is the documented fallback.
  std::string left_search_ik_backend{"analytic"};
  // Runtime-settable planning inputs.
  std::string motion_profile, cut_pattern, cut_debug_cut, cut_debug_left_correction_file,
    cut_debug_middle_correction_file, cut_debug_right_correction_file, cut_debug_output_dir,
    side_tape_handoff_variant, side_tape_handoff_output_file;
  bool side_tape_handoff_enabled{false};
  double side_tape_handoff_standoff_mm{0.0};
  // Right-handed: depth_axis_tool = cut_axis_tool x lateral_axis_tool (validated at startup
  // and on every change). Defaults per the operator's axis decision (2026-09-29): keep the
  // existing parallel left-cut pose; side_tape adapts its own RobotCfg to match.
  std::vector<double> side_tape_handoff_cut_axis_tool{0.0, -1.0, 0.0};
  std::vector<double> side_tape_handoff_lateral_axis_tool{1.0, 0.0, 0.0};
  std::vector<double> side_tape_handoff_depth_axis_tool{0.0, 0.0, 1.0};
  std::vector<double> side_tape_handoff_box_offset_base_mm{0.0, 0.0, 0.0};
  double side_tape_handoff_max_search_angle_deg{10.0};
  double pre_cut_stabilization_duration_s{0.0}, left_side_cut_tilt_deg{0.0},
    right_side_cut_tilt_deg{0.0}, left_adaptive_orientation_max_deg{0.0},
    left_adaptive_orientation_step_deg{0.0}, t_cut_middle_handoff_fraction{0.0},
    left_parallel_max_tilt_deg{0.0}, left_parallel_max_yaw_deg{0.0},
    left_parallel_step_deg{0.0}, left_parallel_min_joint_margin_deg{0.0},
    left_parallel_search_timeout_s{0.0};
  bool debug_fixed_orientation{false}, debug_allow_stale_snapshot{false},
    cut_debug_correction_enabled{false}, left_adaptive_orientation_enabled{true},
    left_parallel_orientation_enabled{false};
};
class BoxHCutExecutor : public rclcpp::Node
{
public:
  BoxHCutExecutor() : Node("box_h_cut_executor")
  {
    // One table drives declaration, startup validation and runtime updates. Read-only
    // entries are declared read_only, so `ros2 param set` fails loudly instead of being
    // silently ignored by code that only reads them at startup.
    for (const auto& spec : parameterTable()) {
      rcl_interfaces::msg::ParameterDescriptor descriptor;
      descriptor.read_only = spec.scope == ParamScope::kReadOnly;
      rclcpp::ParameterValue default_value = spec.default_value;
      if (std::string(spec.name) == "middle_cut_clearance_m") {
        default_value = rclcpp::ParameterValue(settings_.cut_clearance_m);
      }
      const auto value = declare_parameter(spec.name, default_value, descriptor);
      const rclcpp::Parameter parameter(spec.name, value);
      if (spec.scope == ParamScope::kRuntime) {
        applyRuntimeParameter(parameter);
      } else {
        spec.apply(settings_, parameter);
      }
    }
    if (const auto error = validateSettings(settings_)) {
      throw std::invalid_argument("invalid H-cut planning safety parameters: " + *error);
    }
    const auto ompl_plugin_parameter = settings_.planning_pipeline + ".planning_plugin";
    if (!has_parameter(ompl_plugin_parameter)) {
      declare_parameter<std::string>(ompl_plugin_parameter, "ompl_interface/OMPLPlanner");
    }
    const auto ompl_adapters_parameter = settings_.planning_pipeline + ".request_adapters";
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
    const auto kinematics_prefix =
      "robot_description_kinematics." + settings_.planning_group + ".";
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
    parameter_callback_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter>& parameters) {
        return onSetParameters(parameters);
      });

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      [this](sensor_msgs::msg::JointState::SharedPtr message) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_joint_state_ = *message;
        joint_state_received_ = now();
      });
    camera_pole_sub_ = create_subscription<moveit_msgs::msg::CollisionObject>(
      settings_.camera_pole_topic, rclcpp::QoS(1).reliable().transient_local(),
      [this](moveit_msgs::msg::CollisionObject::SharedPtr message) {
        if (message->id != "camera_pole" || message->header.frame_id != settings_.planning_frame ||
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
    side_tape_handoff_pub_ = create_publisher<std_msgs::msg::String>(
      "~/side_tape_handoff", rclcpp::QoS(1).reliable().transient_local());
    top_cut_pub_ = create_publisher<std_msgs::msg::String>(
      "~/top_cut_staged", rclcpp::QoS(1).reliable().transient_local());
    // Watched only to refuse side_tape_retract while Servo looks active (never
    // paused/stopped from here). Topic names are read-only (fixed at startup).
    servo_twist_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      settings_.side_tape_handoff_servo_twist_topic, rclcpp::SensorDataQoS(),
      [this](geometry_msgs::msg::TwistStamped::SharedPtr) {
        std::lock_guard<std::mutex> lock(servo_activity_mutex_);
        servo_twist_last_received_ = now();
      });
    servo_joint_jog_sub_ = create_subscription<control_msgs::msg::JointJog>(
      settings_.side_tape_handoff_servo_joint_topic, rclcpp::SensorDataQoS(),
      [this](control_msgs::msg::JointJog::SharedPtr) {
        std::lock_guard<std::mutex> lock(servo_activity_mutex_);
        servo_joint_jog_last_received_ = now();
      });

    plan_service_ = guardedTrigger(
      "~/plan_h_cut",
      "cannot plan while another planning or execution operation is active",
      [this](std::string& message) {
        if (cut_debug_stage_ != CutDebugStage::kNone) {
          message = "cut debug session active for cut " + cut_debug_session_cut_ +
            " stage " + cutDebugStageName(cut_debug_stage_) +
            "; call clear_plan before planning a normal H-cut";
          return false;
        }
        if (side_tape_handoff_info_.inProgress()) {
          message =
            "a side-tape hand-off is active or unresolved; call side_tape_retract before planning a normal H-cut";
          return false;
        }
        if (topCutStagedInProgress()) {
          message = "a staged top cut is active; call top_cut_retract before planning a normal H-cut";
          return false;
        }
        clearRetainedPlan();
        return planExternalSnapshot(message);
      });
    side_tape_handoff_plan_service_ = guardedTrigger(
      "~/plan_side_tape_handoff",
      "cannot plan while another planning or execution operation is active",
      [this](std::string& message) { return planSideTapeHandoff(message); });
    side_tape_retract_service_ = guardedTrigger(
      "~/side_tape_retract", "another H-cut plan operation is active",
      [this](std::string& message) { return planAndExecuteSideTapeRetract(message); },
      /*reject_during_execution=*/false);
    side_tape_continue_service_ = guardedTrigger(
      "~/continue_side_cut", "another H-cut plan operation is active",
      [this](std::string& message) { return continueSideCut(message); },
      /*reject_during_execution=*/false);
    side_cut_penetrate_service_ = guardedTrigger(
      "~/side_cut_penetrate", "another H-cut plan operation is active",
      [this](std::string& message) { return sideCutPenetrate(message); },
      /*reject_during_execution=*/false);
    top_cut_plan_service_ = guardedTrigger(
      "~/plan_top_cut_staged",
      "cannot plan while another planning or execution operation is active",
      [this](std::string& message) { return planTopCutStaged(message); });
    top_cut_penetrate_service_ = guardedTrigger(
      "~/top_cut_penetrate", "another H-cut plan operation is active",
      [this](std::string& message) { return topCutPenetrate(message); },
      /*reject_during_execution=*/false);
    top_cut_cut_service_ = guardedTrigger(
      "~/top_cut_cut_and_lift", "another H-cut plan operation is active",
      [this](std::string& message) { return topCutCutAndLift(message); },
      /*reject_during_execution=*/false);
    top_cut_continue_service_ = guardedTrigger(
      "~/continue_top_cut", "another H-cut plan operation is active",
      [this](std::string& message) { return continueTopCut(message); },
      /*reject_during_execution=*/false);
    top_cut_retract_service_ = guardedTrigger(
      "~/top_cut_retract", "another H-cut plan operation is active",
      [this](std::string& message) { return topCutRetract(message); },
      /*reject_during_execution=*/false);
    cut_debug_start_service_ = guardedTrigger(
      "~/start_cut_debug_session",
      "cannot start cut debug session while another planning or execution operation is active",
      [this](std::string& message) { return startCutDebugSession(message); });
    cut_debug_save_service_ = guardedTrigger(
      "~/save_cut_debug_point",
      "cannot save cut debug point while another planning or execution operation is active",
      [this](std::string& message) { return saveCutDebugPoint(message); });
    section_diagnostic_service_ = guardedTrigger(
      "~/diagnose_external_snapshot_sections",
      "cannot diagnose while another planning or execution operation is active",
      [this](std::string& message) {
        if (cut_debug_stage_ != CutDebugStage::kNone) {
          message = "cut debug session active for cut " + cut_debug_session_cut_ +
            "; call clear_plan before running section diagnostics";
          return false;
        }
        return diagnoseExternalSnapshotSections(message);
      });
    // Execution-in-progress is reported by the execute body itself, after the gates.
    execute_service_ = guardedTrigger(
      "~/execute_planned_h_cut", "another H-cut plan operation is active",
      [this](std::string& message) { return executeRetainedPlan(message); },
      /*reject_during_execution=*/false);
    clear_service_ = guardedTrigger(
      "~/clear_plan",
      "cannot clear the plan while planning or an H-cut trajectory execution is active",
      [this](std::string& message) {
        if (topCutStagedInProgress()) {
          message = "a staged top cut is active (" +
            std::string(topCutStatusName(top_cut_info_.status)) +
            "); call top_cut_retract before clearing the plan";
          return false;
        }
        if (side_tape_handoff_info_.status == SideTapeHandoffStatus::kActive ||
            side_tape_handoff_info_.status == SideTapeHandoffStatus::kCutting ||
            side_tape_handoff_info_.status == SideTapeHandoffStatus::kCutFailed) {
          message =
            "a side-tape hand-off is active; call side_tape_retract before clearing the plan";
          return false;
        }
        if (side_tape_handoff_info_.status == SideTapeHandoffStatus::kApproachFailed) {
          std::string verify_error;
          if (!sideTapeHandoffTipVerifiedOutside(verify_error)) {
            message = "a side-tape hand-off approach failed and is unresolved; call "
              "side_tape_retract, or clear_plan is refused until the tip is verified outside "
              "the carton (" + verify_error + ")";
            return false;
          }
          side_tape_handoff_info_ = SideTapeHandoffInfo{};
          side_tape_handoff_snapshot_.reset();
        }
        clearRetainedPlan();
        clearCutDebugSessionState();
        message = "H-cut plan, cut debug session, and RViz outputs cleared";
        return true;
      });
    RCLCPP_INFO(
      get_logger(),
      "Snapshot H-cut MTC planner ready; hover %.0f mm, cut %.0f mm, pre-cut stabilization %.3f s",
      settings_.hover_clearance_m * 1000.0, settings_.cut_clearance_m * 1000.0,
      settings_.pre_cut_stabilization_duration_s);
  }

  ~BoxHCutExecutor() override
  {
    if (execution_thread_.joinable()) execution_thread_.join();
  }

private:
  // kReadOnly: startup only (declared read_only). kPlanning: changes planning inputs, so
  // it is applied under plan_mutex_ and clears any retained plan. kRuntime: read at use
  // time (execution gates, diagnostics, snapshot path); never blocked by planning.
  enum class ParamScope : std::uint8_t { kReadOnly, kPlanning, kRuntime };
  struct ParamSpec {
    const char* name;
    ParamScope scope;
    rclcpp::ParameterValue default_value;
    std::function<void(PlannerSettings&, const rclcpp::Parameter&)> apply;
  };
  template<typename T>
  static ParamSpec field(const char* name, ParamScope scope, T default_value, T PlannerSettings::*member)
  {
    return ParamSpec{
      name, scope, rclcpp::ParameterValue(default_value),
      [member](PlannerSettings& settings, const rclcpp::Parameter& parameter) {
        settings.*member = parameter.get_value<T>();
      }};
  }
  static ParamSpec runtime(const char* name, rclcpp::ParameterValue default_value)
  {
    return ParamSpec{name, ParamScope::kRuntime, std::move(default_value), nullptr};
  }
  static const std::vector<ParamSpec>& parameterTable()
  {
    using S = PlannerSettings;
    using str = std::string;
    constexpr auto kRO = ParamScope::kReadOnly;
    constexpr auto kPlan = ParamScope::kPlanning;
    // Default scalings are relative to joint_limits.yaml (0.785 rad/s, 5.24 rad/s^2) and
    // keep the physical speed of the former 0.05 default at 2.16 rad/s, 10 rad/s^2.
    static const std::vector<ParamSpec> table{
      field<str>("planning_group", kRO, "igus_rebel_arm", &S::planning_group),
      field<str>("planning_frame", kRO, "base_link", &S::planning_frame),
      field<str>("tool_link", kRO, "link6", &S::tool_link),
      field<str>("planning_pipeline", kRO, "ompl", &S::planning_pipeline),
      field<str>("chomp_pipeline", kRO, "chomp", &S::chomp_pipeline),
      field<str>("pilz_pipeline", kRO, "pilz_industrial_motion_planner", &S::pilz_pipeline),
      field<str>("motion_profile", kPlan, "ompl_pilz", &S::motion_profile),
      runtime("external_snapshot_file", rclcpp::ParameterValue(str())),
      field<double>("external_snapshot_max_age_s", kRO, 30.0, &S::external_snapshot_max_age_s),
      field<bool>("debug_allow_stale_snapshot", kPlan, false, &S::debug_allow_stale_snapshot),
      field<str>("carton_collision_id", kRO, "detected_carton", &S::carton_collision_id),
      field<bool>("allow_tool_carton_contact", kRO, false, &S::allow_tool_carton_contact),
      field<bool>("camera_pole_required", kRO, true, &S::camera_pole_required),
      field<str>(
        "camera_pole_topic", kRO, "/box_motion/camera_pole_collision", &S::camera_pole_topic),
      field<double>("carton_height_m", kRO, 0.20, &S::carton_height_m),
      field<bool>("carton_height_from_snapshot", kRO, true, &S::carton_height_from_snapshot),
      field<double>("carton_height_max_m", kRO, 2.0, &S::carton_height_max_m),
      field<double>("hover_clearance_m", kRO, 0.10, &S::hover_clearance_m),
      field<double>("left_lateral_standoff_m", kRO, 0.010, &S::left_lateral_standoff_m),
      field<double>("cut_clearance_m", kRO, 0.05, &S::cut_clearance_m),
      // Default follows cut_clearance_m (substituted at declaration).
      field<double>("middle_cut_clearance_m", kRO, 0.05, &S::middle_cut_clearance_m),
      field<double>("middle_entry_alignment_m", kRO, 0.010, &S::middle_entry_alignment_m),
      field<double>(
        "pre_cut_stabilization_duration_s", kPlan, 1.0, &S::pre_cut_stabilization_duration_s),
      field<double>("left_side_cut_tilt_deg", kPlan, 0.0, &S::left_side_cut_tilt_deg),
      field<double>("right_side_cut_tilt_deg", kPlan, 0.0, &S::right_side_cut_tilt_deg),
      field<str>(
        "side_cut_orientation_mode", kRO, "edge_aligned", &S::side_cut_orientation_mode),
      field<bool>(
        "left_adaptive_orientation_enabled", kPlan, true, &S::left_adaptive_orientation_enabled),
      field<double>(
        "left_adaptive_orientation_max_deg", kPlan, 45.0, &S::left_adaptive_orientation_max_deg),
      field<double>(
        "left_adaptive_orientation_step_deg", kPlan, 5.0, &S::left_adaptive_orientation_step_deg),
      // Opt-in geometry-derived left side-cut orientation that keeps the blade plane as
      // parallel to the detected top face as reachability allows (see
      // searchLeftParallelOrientation). Default off: plans are unchanged.
      field<bool>(
        "left_parallel_orientation_enabled", kPlan, false, &S::left_parallel_orientation_enabled),
      field<double>("left_parallel_max_tilt_deg", kPlan, 45.0, &S::left_parallel_max_tilt_deg),
      field<double>("left_parallel_max_yaw_deg", kPlan, 30.0, &S::left_parallel_max_yaw_deg),
      field<double>("left_parallel_step_deg", kPlan, 5.0, &S::left_parallel_step_deg),
      field<double>(
        "left_parallel_min_joint_margin_deg", kPlan, 5.0, &S::left_parallel_min_joint_margin_deg),
      field<double>(
        "left_parallel_search_timeout_s", kPlan, 90.0, &S::left_parallel_search_timeout_s),
      // Phase 1 speed-up: see PlannerSettings::left_search_ik_backend above. Read-only
      // (restart to change) because it selects which IK code path runs, not a planning
      // input.
      field<str>("left_search_ik_backend", kRO, "analytic", &S::left_search_ik_backend),
      // Expensive per-pose IK/collision probing that explains each rejected left roll
      // candidate. Read at use time; the roll search itself is unaffected.
      runtime("left_roll_diagnostics_enabled", rclcpp::ParameterValue(false)),
      field<double>("debug_motion_distance_m", kRO, 0.010, &S::debug_motion_distance_m),
      field<double>("debug_motion_clearance_m", kRO, 0.030, &S::debug_motion_clearance_m),
      field<bool>("debug_fixed_orientation", kPlan, false, &S::debug_fixed_orientation),
      field<str>("cut_pattern", kPlan, "t_cut", &S::cut_pattern),
      field<double>(
        "t_cut_middle_handoff_fraction", kPlan, 0.0, &S::t_cut_middle_handoff_fraction),
      // Supplied joint-space approach used only by the "h" pattern before its first cut.
      field<std::vector<double>>(
        "approach_joint_degrees", kRO, std::vector<double>{}, &S::approach_joint_degrees),
      field<double>("cartesian_step_m", kRO, 0.003, &S::cartesian_step_m),
      field<double>("planning_timeout_s", kRO, 15.0, &S::planning_timeout_s),
      field<double>("velocity_scaling", kRO, 0.1376, &S::velocity_scaling),
      field<double>("acceleration_scaling", kRO, 0.0954, &S::acceleration_scaling),
      field<double>("transit_velocity_scaling", kRO, 0.1376, &S::transit_velocity_scaling),
      field<double>(
        "transit_acceleration_scaling", kRO, 0.0954, &S::transit_acceleration_scaling),
      runtime("execution_enabled", rclcpp::ParameterValue(false)),
      runtime("debug_motion_execution_enabled", rclcpp::ParameterValue(false)),
      field<str>("cut_debug_cut", kPlan, "middle", &S::cut_debug_cut),
      field<bool>(
        "cut_debug_correction_enabled", kPlan, false, &S::cut_debug_correction_enabled),
      field<str>(
        "cut_debug_left_correction_file", kPlan, "", &S::cut_debug_left_correction_file),
      field<str>(
        "cut_debug_middle_correction_file", kPlan, "", &S::cut_debug_middle_correction_file),
      field<str>(
        "cut_debug_right_correction_file", kPlan, "", &S::cut_debug_right_correction_file),
      field<str>("cut_debug_output_dir", kPlan, "", &S::cut_debug_output_dir),
      field<std::int64_t>("max_solutions", kRO, 1, &S::max_solutions),
      // Side-tape hand-off (left face only): plans the full left cut for feasibility,
      // then retains and executes only the approach to the cut start pose, publishing the
      // hand-off record for the side_tape_ros knife-camera stack. See CONTEXT.md.
      field<bool>("side_tape_handoff_enabled", kPlan, false, &S::side_tape_handoff_enabled),
      field<str>(
        "side_tape_handoff_variant", kPlan, "in_material", &S::side_tape_handoff_variant),
      field<double>(
        "side_tape_handoff_standoff_mm", kPlan, 9.0, &S::side_tape_handoff_standoff_mm),
      field<str>(
        "side_tape_handoff_output_file", kPlan, "", &S::side_tape_handoff_output_file),
      field<double>(
        "side_tape_handoff_max_search_angle_deg", kPlan, 10.0,
        &S::side_tape_handoff_max_search_angle_deg),
      // side_tape's tool-frame convention, in xeg32_tool_tip axes. Right-handed:
      // depth_axis_tool = cut_axis_tool x lateral_axis_tool (validated). Defaults keep the
      // existing parallel left-cut pose; side_tape adapts its own RobotCfg to match
      // (operator decision, 2026-09-29). Editing these requires a restart.
      field<std::vector<double>>(
        "side_tape_handoff_cut_axis_tool", kRO, std::vector<double>{0.0, -1.0, 0.0},
        &S::side_tape_handoff_cut_axis_tool),
      field<std::vector<double>>(
        "side_tape_handoff_lateral_axis_tool", kRO, std::vector<double>{1.0, 0.0, 0.0},
        &S::side_tape_handoff_lateral_axis_tool),
      field<std::vector<double>>(
        "side_tape_handoff_depth_axis_tool", kRO, std::vector<double>{0.0, 0.0, 1.0},
        &S::side_tape_handoff_depth_axis_tool),
      field<std::vector<double>>(
        "side_tape_handoff_box_offset_base_mm", kRO, std::vector<double>{0.0, 0.0, 0.0},
        &S::side_tape_handoff_box_offset_base_mm),
      // side_tape_retract safety gates (read-only; edit the YAML and restart to change).
      field<double>(
        "side_tape_handoff_retract_max_orientation_deg", kRO, 10.0,
        &S::side_tape_handoff_retract_max_orientation_deg),
      field<double>(
        "side_tape_handoff_retract_max_line_offset_mm", kRO, 20.0,
        &S::side_tape_handoff_retract_max_line_offset_mm),
      field<str>(
        "side_tape_handoff_servo_twist_topic", kRO, "/delta_twist_cmds",
        &S::side_tape_handoff_servo_twist_topic),
      field<str>(
        "side_tape_handoff_servo_joint_topic", kRO, "/delta_joint_cmds",
        &S::side_tape_handoff_servo_joint_topic),
      field<double>(
        "side_tape_handoff_servo_quiescence_s", kRO, 0.5,
        &S::side_tape_handoff_servo_quiescence_s),
      // continue_side_cut entry gates (read-only).
      field<double>(
        "side_tape_continue_max_line_offset_mm", kRO, 20.0,
        &S::side_tape_continue_max_line_offset_mm),
      field<double>(
        "side_tape_continue_max_along_offset_mm", kRO, 20.0,
        &S::side_tape_continue_max_along_offset_mm),
      field<double>(
        "side_tape_continue_min_depth_mm", kRO, 0.5, &S::side_tape_continue_min_depth_mm),
      field<double>(
        "side_tape_continue_max_depth_mm", kRO, 20.0, &S::side_tape_continue_max_depth_mm),
      field<double>(
        "side_tape_continue_max_orientation_deg", kRO, 20.0,
        &S::side_tape_continue_max_orientation_deg),
      field<bool>(
        "side_tape_continue_check_snapshot_age", kRO, true,
        &S::side_tape_continue_check_snapshot_age),
      field<double>(
        "side_tape_continue_speed_factor", kRO, 0.5, &S::side_tape_continue_speed_factor),
      field<bool>("middle_blade_reversed", kRO, false, &S::middle_blade_reversed),
      field<double>(
        "side_tape_continue_end_extension_mm", kRO, 0.0,
        &S::side_tape_continue_end_extension_mm),
      field<double>(
        "side_tape_continue_end_extra_depth_mm", kRO, 0.0,
        &S::side_tape_continue_end_extra_depth_mm),
      field<bool>(
        "side_tape_continue_check_end_depth", kRO, true,
        &S::side_tape_continue_check_end_depth),
      field<double>(
        "side_tape_handoff_joint_velocity_threshold", kRO, 0.01,
        &S::side_tape_handoff_joint_velocity_threshold),
      // Staged top cut (~/plan_top_cut_staged, ~/top_cut_penetrate, ~/top_cut_cut_and_lift).
      field<double>(
        "top_cut_staged_max_position_mm", kRO, 10.0, &S::top_cut_staged_max_position_mm),
      field<double>(
        "top_cut_staged_max_orientation_deg", kRO, 10.0, &S::top_cut_staged_max_orientation_deg),
      field<std::vector<double>>(
        "top_cut_start_offset_mm", kRO, std::vector<double>{0.0, 0.0, 0.0},
        &S::top_cut_start_offset_mm),
      field<std::vector<double>>(
        "top_cut_end_offset_mm", kRO, std::vector<double>{0.0, 0.0, 0.0},
        &S::top_cut_end_offset_mm),
      field<double>("top_cut_staged_speed_factor", kRO, 1.0, &S::top_cut_staged_speed_factor),
      field<double>("top_cut_lift_max_tilt_deg", kRO, 10.0, &S::top_cut_lift_max_tilt_deg),
      field<double>("top_cut_lift_clearance_m", kRO, 0.003, &S::top_cut_lift_clearance_m),
      field<double>("top_cut_pre_transit_lift_m", kRO, 0.10, &S::top_cut_pre_transit_lift_m),
      field<double>(
        "top_cut_post_penetration_dwell_s", kRO, 1.0, &S::top_cut_post_penetration_dwell_s),
      field<bool>("top_cut_allow_shortened_cut", kRO, true, &S::top_cut_allow_shortened_cut),
      field<double>(
        "top_cut_lift_min_clearance_m", kRO, 0.001, &S::top_cut_lift_min_clearance_m),
      field<double>(
        "side_tape_withdraw_min_clearance_m", kRO, 0.001,
        &S::side_tape_withdraw_min_clearance_m),
      field<double>(
        "top_cut_pre_transit_lift_min_fraction", kRO, 0.5,
        &S::top_cut_pre_transit_lift_min_fraction),
      field<double>("top_cut_min_cut_fraction", kRO, 0.5, &S::top_cut_min_cut_fraction),
      field<bool>(
        "top_cut_continue_check_snapshot_age", kRO, true,
        &S::top_cut_continue_check_snapshot_age),
      field<str>("top_cut_staged_output_file", kRO, "", &S::top_cut_staged_output_file),
    };
    return table;
  }
  static const ParamSpec* findParamSpec(const std::string& name)
  {
    for (const auto& spec : parameterTable()) {
      if (name == spec.name) return &spec;
    }
    return nullptr;
  }
  static constexpr double kMaxSnapshotAgeS = 600.0;
  // Shared by startup and every runtime change: returns the first violated rule.
  static std::optional<std::string> validateSettings(const PlannerSettings& s)
  {
    const std::array<std::pair<const char*, double>, 26> doubles{{
      {"left_parallel_max_tilt_deg", s.left_parallel_max_tilt_deg},
      {"left_parallel_max_yaw_deg", s.left_parallel_max_yaw_deg},
      {"left_parallel_step_deg", s.left_parallel_step_deg},
      {"left_parallel_min_joint_margin_deg", s.left_parallel_min_joint_margin_deg},
      {"left_parallel_search_timeout_s", s.left_parallel_search_timeout_s},
      {"external_snapshot_max_age_s", s.external_snapshot_max_age_s},
      {"carton_height_m", s.carton_height_m},
      {"hover_clearance_m", s.hover_clearance_m},
      {"left_lateral_standoff_m", s.left_lateral_standoff_m},
      {"cut_clearance_m", s.cut_clearance_m}, {"middle_cut_clearance_m", s.middle_cut_clearance_m},
      {"middle_entry_alignment_m", s.middle_entry_alignment_m},
      {"debug_motion_distance_m", s.debug_motion_distance_m},
      {"debug_motion_clearance_m", s.debug_motion_clearance_m},
      {"cartesian_step_m", s.cartesian_step_m}, {"planning_timeout_s", s.planning_timeout_s},
      {"velocity_scaling", s.velocity_scaling}, {"acceleration_scaling", s.acceleration_scaling},
      {"transit_velocity_scaling", s.transit_velocity_scaling},
      {"transit_acceleration_scaling", s.transit_acceleration_scaling},
      {"pre_cut_stabilization_duration_s", s.pre_cut_stabilization_duration_s},
      {"left_side_cut_tilt_deg", s.left_side_cut_tilt_deg},
      {"right_side_cut_tilt_deg", s.right_side_cut_tilt_deg},
      {"left_adaptive_orientation_max_deg", s.left_adaptive_orientation_max_deg},
      {"left_adaptive_orientation_step_deg", s.left_adaptive_orientation_step_deg},
      {"t_cut_middle_handoff_fraction", s.t_cut_middle_handoff_fraction}}};
    for (const auto& [name, value] : doubles) {
      if (!std::isfinite(value)) return std::string(name) + " must be finite";
    }
    const auto within = [](double value, double low, double high) {
      return value >= low && value <= high;
    };
    if (s.planning_group.empty() || s.planning_frame.empty() || s.tool_link.empty() ||
        s.carton_collision_id.empty()) {
      return std::string(
        "planning_group, planning_frame, tool_link and carton_collision_id must be non-empty");
    }
    if (s.camera_pole_required && s.camera_pole_topic.empty()) {
      return std::string("camera_pole_topic is required while camera_pole_required is true");
    }
    if (s.external_snapshot_max_age_s <= 0.0 || s.external_snapshot_max_age_s > kMaxSnapshotAgeS) {
      return "external_snapshot_max_age_s must be within (0," + formatDouble(kMaxSnapshotAgeS, 0) +
        "] seconds";
    }
    if (s.carton_height_m <= 0.0) return std::string("carton_height_m must be positive");
    if (!std::isfinite(s.carton_height_max_m) || s.carton_height_max_m < s.carton_height_m) {
      return std::string("carton_height_max_m must be finite and >= carton_height_m");
    }
    if (s.hover_clearance_m < 0.005) return std::string("hover_clearance_m must be >= 0.005 m");
    if (!within(s.left_lateral_standoff_m, 0.005, 0.025)) {
      return std::string("left_lateral_standoff_m must be within [0.005,0.025] m");
    }
    if (s.cut_clearance_m < -0.003 || s.middle_cut_clearance_m < -0.005) {
      return std::string(
        "cut penetration is capped at 3 mm (cut_clearance_m) and 5 mm (middle_cut_clearance_m)");
    }
    if ((s.cut_clearance_m < 0.0 || s.middle_cut_clearance_m < 0.0) &&
        !s.allow_tool_carton_contact) {
      return std::string("negative cut clearances require allow_tool_carton_contact");
    }
    if (s.cut_clearance_m >= s.hover_clearance_m ||
        s.middle_cut_clearance_m >= s.hover_clearance_m) {
      return std::string("cut clearances must be below hover_clearance_m");
    }
    if (!within(s.middle_entry_alignment_m, 0.0, 0.10)) {
      return std::string("middle_entry_alignment_m must be within [0,0.10] m");
    }
    if (!within(s.pre_cut_stabilization_duration_s, 0.0, 10.0)) {
      return std::string(
        "pre_cut_stabilization_duration_s must be finite and within [0,10] seconds");
    }
    if (s.cartesian_step_m <= 0.0 || s.planning_timeout_s <= 0.0) {
      return std::string("cartesian_step_m and planning_timeout_s must be positive");
    }
    if (s.debug_motion_distance_m <= 0.0 || s.debug_motion_distance_m > 0.020 ||
        !within(s.debug_motion_clearance_m, 0.010, 0.100)) {
      return std::string(
        "debug_motion_distance_m must be within (0,0.020] m and debug_motion_clearance_m within [0.010,0.100] m");
    }
    if (std::abs(s.left_side_cut_tilt_deg) > 90.0 || std::abs(s.right_side_cut_tilt_deg) > 90.0) {
      return std::string("side cut tilt must be finite and within [-90,90] degrees");
    }
    if (s.left_adaptive_orientation_max_deg <= 0.0 || s.left_adaptive_orientation_max_deg > 90.0 ||
        s.left_adaptive_orientation_step_deg <= 0.0 ||
        s.left_adaptive_orientation_step_deg > s.left_adaptive_orientation_max_deg) {
      return std::string(
        "left adaptive orientation bounds must satisfy 0 < step <= max <= 90 degrees");
    }
    if (!within(s.left_parallel_max_tilt_deg, 0.0, 45.0) ||
        !within(s.left_parallel_max_yaw_deg, 0.0, 45.0) ||
        !within(s.left_parallel_step_deg, 1.0, 15.0) ||
        !within(s.left_parallel_min_joint_margin_deg, 0.0, 20.0) ||
        !within(s.left_parallel_search_timeout_s, 1.0, 600.0)) {
      return std::string(
        "left parallel orientation bounds: max_tilt and max_yaw within [0,45] deg, step within "
        "[1,15] deg, min_joint_margin within [0,20] deg, search_timeout within [1,600] s");
    }
    for (const double scaling : {s.velocity_scaling, s.acceleration_scaling,
                                 s.transit_velocity_scaling, s.transit_acceleration_scaling}) {
      if (scaling <= 0.0 || scaling > 1.0) {
        return std::string("velocity/acceleration scalings must be finite and within (0,1]");
      }
    }
    if (s.max_solutions < 1) return std::string("max_solutions must be at least 1");
    if (s.left_search_ik_backend != "analytic" && s.left_search_ik_backend != "kdl") {
      return std::string("left_search_ik_backend must be analytic or kdl");
    }
    if (!isValidHandoffFraction(s.t_cut_middle_handoff_fraction)) {
      return std::string(kHandoffFractionRule);
    }
    if (!isSupportedCutPattern(s.cut_pattern)) return std::string("unsupported cut_pattern");
    if (!isSupportedMotionProfile(s.motion_profile)) {
      return std::string(
        "unsupported motion_profile (ompl_cartesian, ompl_pilz, pilz_only, chomp_pilz)");
    }
    if (s.side_cut_orientation_mode != "edge_aligned" &&
        s.side_cut_orientation_mode != "middle_aligned") {
      return std::string(
        "unsupported side_cut_orientation_mode (edge_aligned or middle_aligned)");
    }
    if (!s.approach_joint_degrees.empty() &&
        (s.approach_joint_degrees.size() != 6U ||
         !std::all_of(s.approach_joint_degrees.begin(), s.approach_joint_degrees.end(),
                      [](double value) { return std::isfinite(value); }))) {
      return std::string("approach_joint_degrees must be empty or six finite values");
    }
    if (!cutKindFromName(s.cut_debug_cut)) {
      return std::string("cut_debug_cut must be left, middle, or right");
    }
    for (const auto& [name, file] : std::array<std::pair<const char*, const std::string*>, 3>{{
           {"cut_debug_left_correction_file", &s.cut_debug_left_correction_file},
           {"cut_debug_middle_correction_file", &s.cut_debug_middle_correction_file},
           {"cut_debug_right_correction_file", &s.cut_debug_right_correction_file}}}) {
      if (!isAbsoluteJsonOrEmpty(*file)) {
        return std::string(name) + " must be an absolute .json path or empty";
      }
    }
    if (!s.cut_debug_output_dir.empty() && s.cut_debug_output_dir.front() != '/') {
      return std::string("cut_debug_output_dir must be an absolute path or empty");
    }
    if (s.side_tape_handoff_variant != "in_material" && s.side_tape_handoff_variant != "standoff") {
      return std::string("side_tape_handoff_variant must be in_material or standoff");
    }
    if (!std::isfinite(s.side_tape_handoff_standoff_mm) || s.side_tape_handoff_standoff_mm <= 0.0 ||
        s.side_tape_handoff_standoff_mm > 50.0) {
      return std::string("side_tape_handoff_standoff_mm must be finite and within (0,50] mm");
    }
    if (!std::isfinite(s.side_tape_handoff_max_search_angle_deg) ||
        s.side_tape_handoff_max_search_angle_deg <= 0.0 ||
        s.side_tape_handoff_max_search_angle_deg > 45.0) {
      return std::string("side_tape_handoff_max_search_angle_deg must be finite and within (0,45] deg");
    }
    if (!isAbsoluteJsonOrEmpty(s.side_tape_handoff_output_file)) {
      return std::string("side_tape_handoff_output_file must be an absolute .json path or empty");
    }
    {
      const auto to_vector3 = [](const std::vector<double>& values) {
        return Eigen::Vector3d(values[0], values[1], values[2]);
      };
      const std::array<std::pair<const char*, const std::vector<double>*>, 3> axis_params{{
        {"side_tape_handoff_cut_axis_tool", &s.side_tape_handoff_cut_axis_tool},
        {"side_tape_handoff_lateral_axis_tool", &s.side_tape_handoff_lateral_axis_tool},
        {"side_tape_handoff_depth_axis_tool", &s.side_tape_handoff_depth_axis_tool}}};
      for (const auto& [name, values] : axis_params) {
        if (values->size() != 3U ||
            !std::all_of(values->begin(), values->end(),
                         [](double value) { return std::isfinite(value); })) {
          return std::string(name) + " must be three finite values";
        }
        if (std::abs(to_vector3(*values).norm() - 1.0) > 1e-3) {
          return std::string(name) + " must be a unit vector";
        }
      }
      const Eigen::Vector3d cut = to_vector3(s.side_tape_handoff_cut_axis_tool);
      const Eigen::Vector3d lateral = to_vector3(s.side_tape_handoff_lateral_axis_tool);
      const Eigen::Vector3d depth = to_vector3(s.side_tape_handoff_depth_axis_tool);
      if (std::abs(cut.dot(lateral)) > 1e-3 || std::abs(cut.dot(depth)) > 1e-3 ||
          std::abs(lateral.dot(depth)) > 1e-3) {
        return std::string(
          "side_tape_handoff_*_axis_tool must be mutually orthogonal unit vectors");
      }
      if ((cut.cross(lateral) - depth).norm() > 1e-3) {
        return std::string(
          "side_tape_handoff_*_axis_tool must be right-handed: depth_axis_tool must equal "
          "cut_axis_tool x lateral_axis_tool");
      }
    }
    if (!std::isfinite(s.side_tape_handoff_retract_max_orientation_deg) ||
        !within(s.side_tape_handoff_retract_max_orientation_deg, 0.0, 90.0)) {
      return std::string("side_tape_handoff_retract_max_orientation_deg must be within [0,90] deg");
    }
    if (!std::isfinite(s.side_tape_handoff_retract_max_line_offset_mm) ||
        s.side_tape_handoff_retract_max_line_offset_mm <= 0.0 ||
        s.side_tape_handoff_retract_max_line_offset_mm > 200.0) {
      return std::string(
        "side_tape_handoff_retract_max_line_offset_mm must be finite and within (0,200] mm");
    }
    if (!std::isfinite(s.side_tape_handoff_servo_quiescence_s) ||
        s.side_tape_handoff_servo_quiescence_s <= 0.0 ||
        s.side_tape_handoff_servo_quiescence_s > 10.0) {
      return std::string("side_tape_handoff_servo_quiescence_s must be finite and within (0,10] s");
    }
    if (!std::isfinite(s.side_tape_continue_max_line_offset_mm) ||
        s.side_tape_continue_max_line_offset_mm <= 0.0 ||
        s.side_tape_continue_max_line_offset_mm > 100.0 ||
        !std::isfinite(s.side_tape_continue_max_along_offset_mm) ||
        s.side_tape_continue_max_along_offset_mm <= 0.0 ||
        s.side_tape_continue_max_along_offset_mm > 100.0 ||
        !std::isfinite(s.side_tape_continue_min_depth_mm) ||
        !std::isfinite(s.side_tape_continue_max_depth_mm) ||
        s.side_tape_continue_min_depth_mm < 0.0 ||
        s.side_tape_continue_max_depth_mm <= s.side_tape_continue_min_depth_mm ||
        s.side_tape_continue_max_depth_mm > 50.0 ||
        !std::isfinite(s.side_tape_continue_max_orientation_deg) ||
        s.side_tape_continue_max_orientation_deg <= 0.0 ||
        s.side_tape_continue_max_orientation_deg > 45.0 ||
        !std::isfinite(s.side_tape_continue_speed_factor) ||
        s.side_tape_continue_speed_factor <= 0.0 || s.side_tape_continue_speed_factor > 1.0 ||
        !std::isfinite(s.side_tape_continue_end_extension_mm) ||
        s.side_tape_continue_end_extension_mm < -20.0 ||
        s.side_tape_continue_end_extension_mm > 20.0 ||
        !std::isfinite(s.side_tape_continue_end_extra_depth_mm) ||
        s.side_tape_continue_end_extra_depth_mm < 0.0 ||
        s.side_tape_continue_end_extra_depth_mm > 10.0) {
      return std::string(
        "side_tape_continue_* limits must be finite: line/along offsets within (0,100] mm, "
        "0 <= min_depth < max_depth <= 50 mm, orientation within (0,45] deg");
    }
    if (!std::isfinite(s.side_tape_handoff_joint_velocity_threshold) ||
        s.side_tape_handoff_joint_velocity_threshold <= 0.0 ||
        s.side_tape_handoff_joint_velocity_threshold > 1.0) {
      return std::string(
        "side_tape_handoff_joint_velocity_threshold must be finite and within (0,1] rad/s");
    }
    if (!std::isfinite(s.top_cut_staged_max_position_mm) || s.top_cut_staged_max_position_mm <= 0.0 ||
        s.top_cut_staged_max_position_mm > 50.0 ||
        !std::isfinite(s.top_cut_staged_max_orientation_deg) ||
        s.top_cut_staged_max_orientation_deg <= 0.0 || s.top_cut_staged_max_orientation_deg > 45.0 ||
        !std::isfinite(s.top_cut_staged_speed_factor) || s.top_cut_staged_speed_factor <= 0.0 ||
        s.top_cut_staged_speed_factor > 1.0 || !isAbsoluteJsonOrEmpty(s.top_cut_staged_output_file)) {
      return std::string(
        "top_cut_staged_* must be finite: position within (0,50] mm, orientation within (0,45] deg, "
        "speed factor within (0,1], output file an absolute .json path or empty");
    }
    if (!std::isfinite(s.top_cut_lift_clearance_m) || s.top_cut_lift_clearance_m < 0.001 ||
        s.top_cut_lift_clearance_m > 0.05) {
      return std::string("top_cut_lift_clearance_m must be finite and within [0.001, 0.05] m");
    }
    if (!std::isfinite(s.top_cut_lift_max_tilt_deg) || s.top_cut_lift_max_tilt_deg < 0.0 ||
        s.top_cut_lift_max_tilt_deg > 30.0) {
      return std::string("top_cut_lift_max_tilt_deg must be finite and within [0,30] degrees");
    }
    if (s.side_tape_handoff_servo_twist_topic.empty() ||
        s.side_tape_handoff_servo_joint_topic.empty()) {
      return std::string(
        "side_tape_handoff_servo_twist_topic and side_tape_handoff_servo_joint_topic must be non-empty");
    }
    return std::nullopt;
  }
  void applyRuntimeParameter(const rclcpp::Parameter& parameter)
  {
    const auto& name = parameter.get_name();
    if (name == "execution_enabled") {
      execution_enabled_.store(parameter.as_bool());
    } else if (name == "debug_motion_execution_enabled") {
      debug_motion_execution_enabled_.store(parameter.as_bool());
    } else if (name == "left_roll_diagnostics_enabled") {
      left_roll_diagnostics_enabled_.store(parameter.as_bool());
    } else if (name == "external_snapshot_file") {
      std::lock_guard<std::mutex> lock(runtime_parameter_mutex_);
      external_snapshot_file_ = parameter.as_string();
    }
  }
  std::string externalSnapshotFile()
  {
    std::lock_guard<std::mutex> lock(runtime_parameter_mutex_);
    return external_snapshot_file_;
  }
  // rclcpp holds its internal parameter mutex while this callback runs, and every
  // planning/execution service holds plan_mutex_ for its whole duration, so this
  // callback must never block on plan_mutex_ (that was a deadlock with any code reading
  // a parameter under plan_mutex_). Planning changes are rejected while busy; runtime
  // gates (execution_enabled, debug_motion_execution_enabled) always apply.
  rcl_interfaces::msg::SetParametersResult onSetParameters(
    const std::vector<rclcpp::Parameter>& parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = false;
    std::vector<std::pair<const ParamSpec*, const rclcpp::Parameter*>> planning;
    std::vector<const rclcpp::Parameter*> runtime_changes;
    for (const auto& parameter : parameters) {
      const ParamSpec* spec = findParamSpec(parameter.get_name());
      if (spec == nullptr) continue;  // MoveIt pipeline/kinematics parameters
      if (spec->scope == ParamScope::kReadOnly) {
        result.reason = "parameter '" + parameter.get_name() +
          "' is read-only: it is applied only at startup; edit box_h_cut_executor.yaml and restart";
        return result;
      }
      if (spec->scope == ParamScope::kPlanning) {
        planning.emplace_back(spec, &parameter);
      } else {
        runtime_changes.push_back(&parameter);
      }
    }
    std::unique_lock<std::mutex> lock(plan_mutex_, std::defer_lock);
    if (!planning.empty()) {
      if (!lock.try_lock()) {
        result.reason =
          "busy: planning/execution active; H-cut planning parameters were not changed; retry after the active service call returns";
        return result;
      }
      if (execution_in_progress_) {
        result.reason =
          "busy: planning/execution active; cannot change planning parameters during trajectory execution";
        return result;
      }
      if (cut_debug_stage_ != CutDebugStage::kNone) {
        result.reason = "cannot change H-cut planning or cut-debug correction parameters while a cut debug session is active for cut " +
          cut_debug_session_cut_ + " stage " + cutDebugStageName(cut_debug_stage_) +
          "; call clear_plan first; execution_enabled and debug_motion_execution_enabled remain operable gates";
        return result;
      }
      if (side_tape_handoff_info_.inProgress()) {
        result.reason =
          "cannot change H-cut planning parameters while a side-tape hand-off is active or "
          "unresolved; call side_tape_retract first";
        return result;
      }
      if (topCutStagedInProgress()) {
        result.reason =
          "cannot change H-cut planning parameters while a staged top cut is active; call "
          "top_cut_retract first";
        return result;
      }
      PlannerSettings candidate = settings_;
      for (const auto& [spec, parameter] : planning) spec->apply(candidate, *parameter);
      if (const auto error = validateSettings(candidate)) {
        result.reason = *error;
        return result;
      }
      // Commit only the planning fields; read-only fields are never rewritten.
      for (const auto& [spec, parameter] : planning) spec->apply(settings_, *parameter);
      clearRetainedPlan();
    }
    for (const auto* parameter : runtime_changes) applyRuntimeParameter(*parameter);
    result.successful = true;
    return result;
  }
  using TriggerService = rclcpp::Service<std_srvs::srv::Trigger>;
  // Every Trigger service runs under plan_mutex_ taken with try_lock, so a service never
  // queues behind a long plan; a busy request fails immediately with `busy_message`.
  TriggerService::SharedPtr guardedTrigger(
    const std::string& name, const std::string& busy_message,
    std::function<bool(std::string&)> body, bool reject_during_execution = true)
  {
    return create_service<std_srvs::srv::Trigger>(
      name,
      [this, busy_message, reject_during_execution, body = std::move(body)](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock() || (reject_during_execution && execution_in_progress_)) {
          response->success = false;
          response->message = busy_message;
          return;
        }
        response->success = body(response->message);
      },
      rmw_qos_profile_services_default, callback_group_);
  }
  bool executeRetainedPlan(std::string& message)
  {
    const bool in_debug_session = cut_debug_stage_ != CutDebugStage::kNone;
    // Checked before every other gate: a snapshot loaded under the plan-only stale
    // override (or the override being on now) can never be dispatched, on any path.
    if (settings_.debug_allow_stale_snapshot || retained_with_stale_override_) {
      message = std::string(in_debug_session ? "cut debug session" : "retained H-cut plan") +
        " execution is blocked: its snapshot was loaded with debug_allow_stale_snapshot=true "
        "(plan-only override); set debug_allow_stale_snapshot false, call clear_plan and "
        "replan from a fresh snapshot";
      return false;
    }
    if (!execution_enabled_.load()) {
      message = "execution service is disabled by execution_enabled";
      return false;
    }
    if (settings_.debug_fixed_orientation) {
      message = "execution is blocked while debug_fixed_orientation is enabled";
      return false;
    }
    if (in_debug_session && !debug_motion_execution_enabled_.load()) {
      message =
        "cut debug session execution requires both execution_enabled and debug_motion_execution_enabled";
      return false;
    }
    if (!in_debug_session && settings_.cut_pattern == "middle_debug_short" &&
        !debug_motion_execution_enabled_.load()) {
      message = "debug trajectory execution is disabled by debug_motion_execution_enabled";
      return false;
    }
    if (execution_in_progress_) {
      message = "an H-cut trajectory execution is already active";
      return false;
    }
    if (in_debug_session && cut_debug_stage_ != CutDebugStage::kStartPlanned) {
      if (cut_debug_stage_ == CutDebugStage::kStartExecuted) {
        message = "cut debug start already executed for cut " + cut_debug_session_cut_ +
          "; teleop correct the TCP then call save_cut_debug_point before any further execution";
      } else if (cut_debug_stage_ == CutDebugStage::kStartSaved) {
        message = "cut debug start observation already saved for cut " + cut_debug_session_cut_ +
          "; this session has nothing left to execute: teleop to the endpoint and call save_cut_debug_point";
      } else {
        message = "no executable cut debug plan is retained for cut " + cut_debug_session_cut_ +
          " stage " + cutDebugStageName(cut_debug_stage_);
      }
      return false;
    }
    if (in_debug_session && cut_debug_dispatch_failed_) {
      message = "previous cut debug " + cutDebugStageName(cut_debug_stage_) +
        " dispatch for cut " + cut_debug_session_cut_ +
        " failed during execution and was NOT marked executed; no retained plan remains; call clear_plan then start_cut_debug_session to restart";
      return false;
    }
    if (!planned_task_ || !planned_snapshot_ || planned_task_->solutions().empty()) {
      message = "no valid planned H-cut is available to execute";
      return false;
    }
    // Normal retained plans refuse a stale snapshot so a previous carton's result
    // can never dispatch. A staged session instead freezes its snapshot at session
    // start and each stage is planned from the live state seconds before its
    // explicit execution; teleop correction inherently takes longer than the
    // snapshot age bound, so the age check does not apply while the session owns
    // the retained plan. Both execution gates and the stale-override block above
    // still apply to every stage.
    if (!in_debug_session) {
      std::string freshness_error;
      if (!snapshotIsFresh(*planned_snapshot_, freshness_error)) {
        clearRetainedPlan();
        message = freshness_error + "; retained plan was cleared";
        return false;
      }
    }
    if (execution_thread_.joinable()) execution_thread_.join();
    execution_in_progress_ = true;
    const CutDebugStage dispatched_stage = cut_debug_stage_;
    const std::string dispatched_cut = cut_debug_session_cut_;
    const bool dispatched_is_side_tape_handoff = side_tape_handoff_pending_;
    side_tape_handoff_pending_ = false;
    const bool dispatched_is_top_cut = top_cut_pending_;
    top_cut_pending_ = false;
    auto task = std::move(planned_task_);
    planned_snapshot_.reset();
    execution_thread_ = std::thread(
      [this, task = std::move(task), dispatched_stage, dispatched_cut,
       dispatched_is_side_tape_handoff, dispatched_is_top_cut]() mutable {
        bool succeeded = false;
        std::string failure_detail;
        try {
          const auto result = task->execute(*task->solutions().front());
          succeeded = (result == moveit::core::MoveItErrorCode::SUCCESS);
          if (succeeded) {
            RCLCPP_INFO(get_logger(), "H-cut trajectory execution completed");
          } else {
            failure_detail = "controller reported failure status " + std::to_string(result.val);
            RCLCPP_ERROR(
              get_logger(), "H-cut trajectory execution failed: %s", failure_detail.c_str());
          }
        } catch (const std::exception& exception) {
          failure_detail = std::string("threw exception: ") + exception.what();
          RCLCPP_ERROR(get_logger(), "H-cut trajectory execution %s", failure_detail.c_str());
        }
        std::lock_guard<std::mutex> completion_lock(plan_mutex_);
        execution_in_progress_ = false;
        if (dispatched_is_top_cut) {
          top_cut_info_.status =
            succeeded ? TopCutStagedStatus::kApproached : TopCutStagedStatus::kFailed;
          top_cut_info_.stamp = now();
          writeTopCutStagedOutput();
          if (!succeeded) {
            RCLCPP_ERROR(
              get_logger(), "Staged top cut approach did NOT complete (%s); status 'failed'; "
              "call top_cut_retract to recover", failure_detail.c_str());
          }
        }
        if (dispatched_is_side_tape_handoff) {
          if (succeeded) {
            side_tape_handoff_info_.status = SideTapeHandoffStatus::kActive;
            side_tape_handoff_info_.stamp = now();
            writeSideTapeHandoffOutput();
          } else {
            // Do NOT drop the hand-off state: the arm pose is uncertain, but the carton and
            // recorded geometry must survive so side_tape_retract can still recover from
            // wherever execution stopped.
            side_tape_handoff_info_.status = SideTapeHandoffStatus::kApproachFailed;
            side_tape_handoff_info_.stamp = now();
            writeSideTapeHandoffOutput();
            RCLCPP_ERROR(
              get_logger(),
              "Side-tape hand-off approach dispatch did NOT complete (%s); hand-off is "
              "'approach_failed' (uncertain), NOT active for side_tape; detected_carton is "
              "kept; call side_tape_retract to recover",
              failure_detail.c_str());
          }
        }
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
      message = "cut debug start dispatch accepted for cut " + dispatched_cut +
        "; monitor controller execution, then teleop correct and call save_cut_debug_point; no further motion is automatic";
    } else {
      message = "H-cut trajectory dispatch accepted; monitor controller execution";
    }
    return true;
  }
  void clearRetainedPlan()
  {
    planned_task_.reset();
    planned_snapshot_.reset();
    retained_with_stale_override_ = false;
    // A retained-but-not-yet-dispatched hand-off plan must never survive into (be mistaken
    // for) a later, unrelated plan's dispatch: always consumed here, regardless of whether a
    // hand-off is currently in progress.
    side_tape_handoff_pending_ = false;
    top_cut_pending_ = false;
    if (!topCutStagedInProgress()) {
      top_cut_info_ = TopCutStagedInfo{};
      top_cut_snapshot_.reset();
    }
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_all);
    marker_pub_->publish(markers);
    geometry_msgs::msg::PoseArray empty_targets;
    empty_targets.header.frame_id = settings_.planning_frame;
    target_pose_pub_->publish(empty_targets);
    // A dispatched (or uncertain, approach-failed) side-tape hand-off leaves the arm engaged
    // with (or standing off from) the carton; the carton collision object must survive any
    // clear/replan until side_tape_retract (or a verified-outside clear_plan) ends it.
    if (side_tape_handoff_info_.inProgress() || topCutStagedInProgress()) return;
    moveit_msgs::msg::CollisionObject remove_carton;
    remove_carton.id = settings_.carton_collision_id;
    remove_carton.header.frame_id = settings_.planning_frame;
    remove_carton.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    carton_collision_pub_->publish(remove_carton);
  }

  void clearCutDebugSessionState()
  {
    cut_debug_stage_ = CutDebugStage::kNone;
    cut_debug_session_cut_.clear();
    cut_debug_snapshot_.reset();
    cut_debug_has_observed_start_ = false;
    cut_debug_has_observed_end_ = false;
    cut_debug_dispatch_failed_ = false;
    cut_debug_record_path_.clear();
    cut_debug_created_stamp_ns_ = 0;
  }

  bool planExternalSnapshot(std::string& message)
  {
    const auto source_path = externalSnapshotFile();
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
  static const char* cutName(CutKind cut)
  {
    switch (cut) {
      case CutKind::kLeft: return "left";
      case CutKind::kMiddle: return "middle";
      case CutKind::kRight: return "right";
      case CutKind::kNone: break;
    }
    return "";
  }
  static std::optional<CutKind> cutKindFromName(const std::string& name)
  {
    if (name == "left") return CutKind::kLeft;
    if (name == "middle") return CutKind::kMiddle;
    if (name == "right") return CutKind::kRight;
    return std::nullopt;
  }
  const std::string& correctionFileForCut(const std::string& cut) const
  {
    if (cut == "left") return settings_.cut_debug_left_correction_file;
    if (cut == "right") return settings_.cut_debug_right_correction_file;
    return settings_.cut_debug_middle_correction_file;
  }
  static std::string cutDebugStageName(CutDebugStage stage)
  {
    switch (stage) {
      case CutDebugStage::kNone: return "idle";
      case CutDebugStage::kStartPlanned: return "start_planned";
      case CutDebugStage::kStartExecuted: return "start_executed";
      case CutDebugStage::kStartSaved: return "start_saved";
    }
    return "unknown";
  }
  static Eigen::Vector3d toEigen(const geometry_msgs::msg::Point& point)
  {
    return Eigen::Vector3d(point.x, point.y, point.z);
  }
  static Eigen::Vector3d toEigen(const geometry_msgs::msg::Vector3& vector)
  {
    return Eigen::Vector3d(vector.x, vector.y, vector.z);
  }
  static geometry_msgs::msg::Point toPoint(const Eigen::Vector3d& vector)
  {
    geometry_msgs::msg::Point point;
    point.x = vector.x();
    point.y = vector.y();
    point.z = vector.z();
    return point;
  }
  static Eigen::Quaterniond toEigen(const geometry_msgs::msg::Quaternion& q)
  {
    return Eigen::Quaterniond(q.w, q.x, q.y, q.z);
  }
  static geometry_msgs::msg::Quaternion toMsg(const Eigen::Quaterniond& q)
  {
    geometry_msgs::msg::Quaternion msg;
    msg.x = q.x();
    msg.y = q.y();
    msg.z = q.z();
    msg.w = q.w();
    return msg;
  }
  // Rotation whose columns are the given frame axes (Eigen's rotation-matrix conversion).
  static Eigen::Quaterniond basisQuaternion(
    const Eigen::Vector3d& x_axis, const Eigen::Vector3d& y_axis, const Eigen::Vector3d& z_axis)
  {
    Eigen::Matrix3d basis;
    basis.col(0) = x_axis;
    basis.col(1) = y_axis;
    basis.col(2) = z_axis;
    return Eigen::Quaterniond(basis);
  }
  static Eigen::Quaterniond basisQuaternion(const CutBasis& basis)
  {
    return basisQuaternion(basis.along, basis.across, basis.normal);
  }
  // Standard detected segment for a cut (named, endpoints from the canonical snapshot).
  static CutSegment segmentFor(const Snapshot& snapshot, CutKind cut)
  {
    CutSegment segment;
    segment.cut = cut;
    switch (cut) {
      case CutKind::kLeft:
        segment.name = "left side cut";
        segment.start = snapshot.back_middle_left;
        segment.end = snapshot.front_middle_left;
        break;
      case CutKind::kMiddle:
        segment.name = "middle seam cut";
        segment.start = snapshot.seam_start;
        segment.end = snapshot.seam_end;
        break;
      case CutKind::kRight:
        segment.name = "right side cut";
        segment.start = snapshot.back_middle_right;
        segment.end = snapshot.front_middle_right;
        break;
      case CutKind::kNone:
        throw std::invalid_argument("segmentFor requires a detected cut");
    }
    return segment;
  }
  bool cutDebugBasis(
    const Snapshot& snapshot, const std::string& cut, CutBasis& basis, std::string& error) const
  {
    const auto kind = cutKindFromName(cut);
    if (!kind) {
      error = "unknown cut debug cut: " + cut;
      return false;
    }
    const CutSegment segment = segmentFor(snapshot, *kind);
    Eigen::Vector3d normal = toEigen(snapshot.surface_normal);
    const double normal_length = normal.norm();
    if (!std::isfinite(normal_length) || normal_length <= 1e-9) {
      error = "snapshot surface normal has zero length";
      return false;
    }
    normal /= normal_length;
    Eigen::Vector3d axis = toEigen(segment.end) - toEigen(segment.start);
    axis -= axis.dot(normal) * normal;
    const double axis_length = axis.norm();
    if (!std::isfinite(axis_length) || axis_length < 1e-6) {
      error = "cut debug cut has degenerate length for cut " + cut;
      return false;
    }
    basis.along = axis / axis_length;
    basis.across = normal.cross(basis.along);
    basis.normal = normal;
    return true;
  }
  bool offsetLeftPoseOutward(
    const Snapshot& snapshot, geometry_msgs::msg::PoseStamped& pose, double distance_m,
    std::string& error) const
  {
    CutBasis basis;
    if (!cutDebugBasis(snapshot, "left", basis, error)) return false;
    const Eigen::Vector3d midpoint =
      0.5 * (toEigen(snapshot.back_middle_left) + toEigen(snapshot.front_middle_left));
    const double projection = basis.across.dot(toEigen(snapshot.seam_center) - midpoint);
    if (!std::isfinite(projection) || std::abs(projection) < 1e-6) {
      error = "cannot determine outward side of left cut from carton geometry";
      return false;
    }
    const double sign = projection > 0.0 ? -1.0 : 1.0;
    pose.pose.position =
      toPoint(toEigen(pose.pose.position) + sign * distance_m * basis.across);
    return true;
  }
  template<std::size_t N>
  static bool readArray(
    const boost::property_tree::ptree& document, const std::string& field,
    std::array<double, N>& values, std::string& error)
  {
    const auto child = document.get_child_optional(field);
    if (!child || child->size() != N) {
      error = "cut debug correction file is missing " +
        std::string(N == 3U ? "three" : "four") + "-element " + field;
      return false;
    }
    std::size_t index = 0;
    try {
      for (const auto& entry : *child) values[index++] = entry.second.get_value<double>();
    } catch (const std::exception&) {
      error = "cut debug correction file has invalid " + field;
      return false;
    }
    return true;
  }
  bool loadCutDebugCorrectionFile(
    const std::string& path, CutDebugCorrection& correction, std::string& error) const
  {
    namespace fs = std::filesystem;
    if (path.empty() || !isAbsoluteJsonOrEmpty(path)) {
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
    if (!cut || !cutKindFromName(*cut)) {
      error = "cut debug correction file has an invalid cut field";
      return false;
    }
    // Frame and tool are required: a record whose frame or TCP link cannot be verified
    // against the active configuration is never applied.
    const auto frame = document.get_optional<std::string>("planning_frame");
    if (!frame || frame->empty()) {
      error = "cut debug correction file is missing planning_frame; refusing a record whose frame cannot be verified";
      return false;
    }
    if (*frame != settings_.planning_frame) {
      error = "cut debug correction planning_frame mismatch: file=" + *frame +
        " active=" + settings_.planning_frame;
      return false;
    }
    const auto tool = document.get_optional<std::string>("tool_link");
    if (!tool || tool->empty()) {
      error = "cut debug correction file is missing tool_link; refusing a record whose TCP link cannot be verified";
      return false;
    }
    if (*tool != settings_.tool_link) {
      error = "cut debug correction tool_link mismatch: file=" + *tool + " active=" +
        settings_.tool_link;
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
    const auto read_vector3 = [&document, &error](
      const std::string& field, Eigen::Vector3d& vector) {
      std::array<double, 3> values{};
      if (!readArray(document, field, values, error)) return false;
      if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
          })) {
        error = "cut debug correction file has non-finite " + field;
        return false;
      }
      vector = Eigen::Vector3d(values[0], values[1], values[2]);
      return true;
    };
    const auto read_observed_orientation = [&document, &error](
      const std::string& field, geometry_msgs::msg::Quaternion& orientation) {
      std::array<double, 4> values{};
      if (!readArray(document, field, values, error)) return false;
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
    if (!read_vector3("cut_basis_along_m", loaded.recorded.along) ||
        !read_vector3("cut_basis_across_m", loaded.recorded.across) ||
        !read_vector3("cut_basis_normal_m", loaded.recorded.normal) ||
        !read_observed_orientation("observed_start_orientation_xyzw", loaded.observed_start) ||
        !read_observed_orientation("observed_end_orientation_xyzw", loaded.observed_end)) {
      return false;
    }
    const CutBasis& recorded = loaded.recorded;
    if (std::abs(recorded.along.norm() - 1.0) > 0.01 ||
        std::abs(recorded.across.norm() - 1.0) > 0.01 ||
        std::abs(recorded.normal.norm() - 1.0) > 0.01 ||
        std::abs(recorded.along.dot(recorded.across)) > 0.01 ||
        std::abs(recorded.along.dot(recorded.normal)) > 0.01 ||
        std::abs(recorded.across.dot(recorded.normal)) > 0.01 ||
        recorded.along.cross(recorded.across).dot(recorded.normal) < 0.99) {
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
    if (!settings_.cut_debug_correction_enabled) return active;
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
    const geometry_msgs::msg::Point& point, const CutBasis& basis, const CutLocalDeltas& local)
  {
    return toPoint(
      toEigen(point) + (basis.along * local.along_m + basis.across * local.across_m +
                        basis.normal * local.normal_m));
  }
  // Nominal blade-direction targets for a cut: the blade (tool Y) follows the cut
  // segment, or the middle seam for side cuts in middle_aligned mode. Side cuts face
  // 180 degrees away from travel (facing sign -1). Returns {start target, end target}.
  std::pair<geometry_msgs::msg::Point, geometry_msgs::msg::Point> orientationTargets(
    const Snapshot& snapshot, bool is_middle, const geometry_msgs::msg::Point& segment_start,
    const geometry_msgs::msg::Point& segment_end, const geometry_msgs::msg::Point& start,
    const geometry_msgs::msg::Point& end) const
  {
    // Middle (top) seam: the knife faces along travel unless middle_blade_reversed (operator
    // 2026-10-01: the top-cut blade must be rotated 180 deg about the tool Z / surface normal).
    const double facing_sign = is_middle ? (settings_.middle_blade_reversed ? -1.0 : 1.0) : -1.0;
    const bool use_seam = !is_middle && settings_.side_cut_orientation_mode == "middle_aligned";
    const auto& from = use_seam ? snapshot.seam_start : segment_start;
    const auto& to = use_seam ? snapshot.seam_end : segment_end;
    const Eigen::Vector3d offset = facing_sign * (toEigen(to) - toEigen(from));
    return {toPoint(toEigen(start) + offset), toPoint(toEigen(end) + offset)};
  }
  static geometry_msgs::msg::Quaternion interpolateTaughtOrientation(
    const geometry_msgs::msg::Quaternion& start,
    const geometry_msgs::msg::Quaternion& end, double fraction)
  {
    const Eigen::Quaterniond start_q = toEigen(start);
    Eigen::Quaterniond end_q = toEigen(end);
    if (start_q.dot(end_q) < 0.0) end_q.coeffs() *= -1.0;
    return toMsg(start_q.slerp(fraction, end_q).normalized());
  }
  bool applyCutDebugObservedOrientation(
    const Snapshot& snapshot, const std::string& cut, const CutDebugCorrection& correction,
    const geometry_msgs::msg::Quaternion& observed, geometry_msgs::msg::PoseStamped& pose,
    std::string& error) const
  {
    CutBasis current;
    if (!cutDebugBasis(snapshot, cut, current, error)) {
      return false;
    }
    const Eigen::Quaterniond current_basis = basisQuaternion(current).normalized();
    const Eigen::Quaterniond recorded_basis = basisQuaternion(correction.recorded).normalized();
    const Eigen::Quaterniond corrected =
      (current_basis * recorded_basis.inverse() * toEigen(observed)).normalized();
    const Eigen::Vector3d into_unit = (-toEigen(snapshot.surface_normal)).normalized();
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
    pose.pose.orientation = toMsg(corrected);
    return true;
  }
  std::vector<double> leftAdaptiveOrientationCandidates() const
  {
    // Deterministic smallest-magnitude order: 0, +step, -step, ..., ±max.
    std::vector<double> candidates{0.0};
    const double max_deg = settings_.left_adaptive_orientation_max_deg;
    const double step_deg = settings_.left_adaptive_orientation_step_deg;
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
      } else if (settings_.camera_pole_required) {
        error = "camera pole collision object is unavailable for left orientation probe";
        return false;
      }
    }
    std::vector<Eigen::Quaterniond> mapped;
    mapped.reserve(labeled_poses.size());
    for (const auto& labeled : labeled_poses) mapped.push_back(toEigen(labeled.first->pose.orientation));
    const Eigen::Vector3d into_unit = (-toEigen(snapshot.surface_normal)).normalized();
    const std::vector<double> candidates = leftAdaptiveOrientationCandidates();
    const bool detailed_diagnostics = left_roll_diagnostics_enabled_.load();
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
        labeled_poses[index].first->pose.orientation = toMsg(orientation);
      };
    const moveit::core::GroupStateValidityCallbackFn valid_state =
      [this, &scene](moveit::core::RobotState* state,
                     const moveit::core::JointModelGroup* group, const double* values) {
        state->setJointGroupPositions(group, values);
        state->update();
        return !scene.isStateColliding(*state, settings_.planning_group);
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
        target.orientation = toMsg(orientation);
        const bool contact_allowed = index % 2U != 0U && settings_.allow_tool_carton_contact;
        scene.getAllowedCollisionMatrixNonConst().setEntry(
          settings_.tool_link, settings_.carton_collision_id, contact_allowed);
        out << " pos=(" << target.position.x << "," << target.position.y << "," <<
          target.position.z << ") tool/carton_allowed=" << (contact_allowed ? 1 : 0);
        moveit::core::RobotState aware(seed_state);
        if (aware.setFromIK(joint_group, target, settings_.tool_link, 0.25, valid_state)) {
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
          if (!kin.setFromIK(joint_group, target, settings_.tool_link, 0.1)) continue;
          kin.update();
          ++kinematic_found;
          if (!kin.satisfiesBounds(joint_group)) {
            ++out_of_bounds;
            continue;
          }
          collision_detection::CollisionRequest request;
          request.group_name = settings_.planning_group;
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
          settings_.tool_link, settings_.carton_collision_id, index % 2U != 0U && settings_.allow_tool_carton_contact);
        if (!probe.setFromIK(
              joint_group, labeled_poses[index].first->pose, settings_.tool_link, 0.25, valid_state)) {
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
      " candidates 0,+step,-step,...,+/-max with max=" << settings_.left_adaptive_orientation_max_deg <<
      " deg step=" << settings_.left_adaptive_orientation_step_deg << " deg)";
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
  // Opt-in (left_parallel_orientation_enabled) geometry-derived LEFT side-cut orientation.
  // Blade model: tool Y is the blade length, tool X the blade-plane normal, tool Z the edge.
  // Parallelism metric theta_p = acos(|tool X . n|): the angle between the blade plane and
  // the detected top face (n = surface normal); 0 is flat on the face, 90 the nominal
  // perpendicular knife. Reference F0: X = -n (flat, blade normal down), Y = -along (blade
  // opposite travel, as the nominal side cut), Z = X x Y (edge facing the carton side).
  // Candidates are R = Rot(n, yaw) * Rot(across, pitch) * Rot(along, tilt) * F0 on a
  // step grid within +/-max_tilt (tilt, pitch) and +/-max_yaw (yaw), ordered by theta_p,
  // then blade deviation from the cut line, then total angle, so the order (and the
  // result) is deterministic. The TCP positions and the cut line are unchanged. The first
  // candidate that passes all of the following is used for every left pose:
  //   - the existing left X-or-Z penetration gate (>= 0.7);
  //   - collision-aware IK at the first pose (hover start, fully collision-free) from the
  //     current state seed or fixed-seed random restarts;
  //   - a collision-checked Cartesian chain through the remaining poses (entry, cut, exit
  //     and lift: tool/carton contact allowed exactly as in the task) at cartesian_step_m
  //     reaching fraction 1.0, without any joint step above kMaxStrokeJointStepRad
  //     between waypoints and with every joint at least left_parallel_min_joint_margin_deg
  //     inside its URDF bounds along the whole chain.
  // It is a prefilter: MTC, min_fraction=1.0, the dwell check and the transit wrist guard
  // remain the final authority. No candidate within left_parallel_search_timeout_s fails
  // closed.
  // In-bounds spherical-wrist flips (joint4 +/- pi, -joint5, joint6 +/- pi), the same
  // variants NearestBranchTransitPlanner generates.
  static std::vector<std::vector<double> > wristVariants(
    const moveit::core::JointModelGroup* group, const std::vector<double>& values)
  {
    constexpr double kPi = 3.14159265358979323846;
    std::vector<std::vector<double> > variants;
    const auto& names = group->getVariableNames();
    const auto index_of = [&](const std::string& name) -> std::optional<std::size_t> {
        const auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end()) return std::nullopt;
        return static_cast<std::size_t>(std::distance(names.begin(), it));
      };
    const auto j4 = index_of("joint4");
    const auto j5 = index_of("joint5");
    const auto j6 = index_of("joint6");
    if (!j4 || !j5 || !j6 || values.size() != names.size()) return variants;
    const auto& model = group->getParentModel();
    const auto within = [&](std::size_t index, double value) {
        const auto& bounds = model.getVariableBounds(names[index]);
        return !bounds.position_bounded_ ||
          (value >= bounds.min_position_ && value <= bounds.max_position_);
      };
    for (const double turn4 : {-kPi, kPi}) {
      if (!within(*j4, values[*j4] + turn4)) continue;
      for (const double turn6 : {-kPi, kPi}) {
        if (!within(*j6, values[*j6] + turn6)) continue;
        auto flipped = values;
        flipped[*j4] += turn4;
        flipped[*j5] = -flipped[*j5];
        flipped[*j6] += turn6;
        if (within(*j5, flipped[*j5])) variants.push_back(flipped);
      }
    }
    return variants;
  }
  // Phase 1 speed-up (rebel_analytic_ik.hpp): joint1..joint6 bounds from the live
  // RobotModel, in the URDF's own joint order (not necessarily the JointModelGroup's
  // variable order, which analyticIkBranches maps back separately). Returns false if the
  // group's variable set is not exactly {joint1..joint6} with bounded positions, in which
  // case the analytic backend cannot be used at all (caller must fall back to kdl).
  static bool analyticJointLimitsFor(
    const moveit::core::JointModelGroup* group, std::array<rebel_analytic_ik::JointLimits, 6>& out)
  {
    static const std::array<const char*, 6> kNames{
      "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    if (group->getVariableNames().size() != 6U) return false;
    const auto& model = group->getParentModel();
    for (std::size_t j = 0; j < 6U; ++j) {
      if (std::find(
            group->getVariableNames().begin(), group->getVariableNames().end(),
            kNames[j]) == group->getVariableNames().end()) {
        return false;
      }
      const auto& bounds = model.getVariableBounds(kNames[j]);
      if (!bounds.position_bounded_) return false;
      out[j] = {bounds.min_position_, bounds.max_position_};
    }
    return true;
  }
  // Maps rebel_analytic_ik's fixed joint1..joint6 order onto `group`'s own variable order.
  // Only meaningful once analyticJointLimitsFor(group, ...) has confirmed the group is
  // exactly {joint1..joint6}.
  static std::array<std::size_t, 6> analyticVariableIndex(const moveit::core::JointModelGroup* group)
  {
    static const std::array<const char*, 6> kNames{
      "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    const auto& names = group->getVariableNames();
    std::array<std::size_t, 6> index{};
    for (std::size_t j = 0; j < 6U; ++j) {
      index[j] = static_cast<std::size_t>(
        std::distance(names.begin(), std::find(names.begin(), names.end(), kNames[j])));
    }
    return index;
  }
  // Runs two independent checks once (lazily, the first time a JointModelGroup is available:
  // robot_model_ is only set once a task has been built, not at node construction). If either
  // fails, analytic_ik_available_ stays false for the rest of the process and every
  // analytic-backend call below falls back to kdl with a one-time error, per
  // left_search_ik_backend's documented fallback behaviour:
  //   1. rebel_analytic_ik::selfCheck(): the library's own FK against its own solve(), to
  //      1e-6 m / 1e-6 rad. This only proves internal self-consistency -- it shares the
  //      library's hardcoded constants on both sides, so it cannot detect a wrong constant
  //      (see the 2026-09-28 kJ4Z transcription bug: forwardKinematics and solve() both used
  //      the same wrong value and stayed mutually consistent; this check alone would not
  //      have caught it).
  //   2. A ground-truth check against the live MoveIt RobotModel/URDF FK for
  //      settings_.tool_link, which is what actually catches a stale or wrong hardcoded
  //      constant (including a TCP recalibration or a changed xeg32 mount offset the library
  //      does not know about), because it does not go through rebel_analytic_ik at all on the
  //      "expected" side. Also fails closed immediately, before either numeric check, if
  //      tool_link is not exactly "xeg32_tool_tip" (the only TCP the library's hardcoded tool
  //      offset models) or if the robot model's root link / planning_frame is not base_link
  //      (which every rebel_analytic_ik constant is expressed relative to).
  void ensureAnalyticIkSelfCheck(const moveit::core::JointModelGroup* joint_group)
  {
    std::call_once(analytic_self_check_once_, [this, joint_group]() {
        if (settings_.tool_link != "xeg32_tool_tip") {
          RCLCPP_ERROR(
            get_logger(),
            "Analytic IK self-check refused: tool_link is '%s', not 'xeg32_tool_tip' (the "
            "only TCP rebel_analytic_ik's hardcoded link6->igus_tool0->xeg32_tool_tip offset "
            "models); disabling the analytic left-search backend for this process and "
            "falling back to kdl regardless of left_search_ik_backend",
            settings_.tool_link.c_str());
          analytic_ik_available_.store(false);
          return;
        }
        std::array<rebel_analytic_ik::JointLimits, 6> limits;
        if (!analyticJointLimitsFor(joint_group, limits)) {
          RCLCPP_WARN(
            get_logger(),
            "Analytic IK self-check skipped: planning group '%s' is not exactly "
            "{joint1..joint6} with bounded positions; left_search_ik_backend=analytic will "
            "fall back to kdl",
            settings_.planning_group.c_str());
          analytic_ik_available_.store(false);
          return;
        }
        // rebel_analytic_ik's constants are all expressed relative to a link named
        // "base_link"; that link need not be the RobotModel's actual kinematic root (SRDF
        // setups commonly add a "world" virtual-joint parent above it, which is the case
        // here -- getRootLinkName() is "world", not "base_link"). FK below is therefore taken
        // relative to the named "base_link" link explicitly, not assumed to be the model
        // frame; this only requires that link to exist and that planning_frame agrees.
        const moveit::core::LinkModel* base_link_model =
          robot_model_ ? robot_model_->getLinkModel("base_link") : nullptr;
        if (base_link_model == nullptr || settings_.planning_frame != "base_link") {
          RCLCPP_ERROR(
            get_logger(),
            "Analytic IK self-check refused: robot model has %sa link named 'base_link' and "
            "planning_frame is '%s'; rebel_analytic_ik's constants require both a 'base_link' "
            "link and planning_frame=base_link; disabling the analytic left-search backend "
            "for this process and falling back to kdl regardless of left_search_ik_backend",
            base_link_model == nullptr ? "no " : "", settings_.planning_frame.c_str());
          analytic_ik_available_.store(false);
          return;
        }
        const auto library_result =
          rebel_analytic_ik::selfCheck(2000, 1e-6, 1e-6, 20260928U, limits);
        if (!library_result.ok) {
          RCLCPP_ERROR(
            get_logger(),
            "Analytic IK library self-check FAILED (%s); disabling the analytic left-search "
            "backend for this process and falling back to kdl regardless of "
            "left_search_ik_backend",
            library_result.failure_detail.c_str());
          analytic_ik_available_.store(false);
          return;
        }
        RCLCPP_INFO(
          get_logger(),
          "Analytic IK library self-check passed: %zu/%zu samples recovered, max residual "
          "%.3e m / %.3e rad",
          library_result.samples_with_original_recovered, library_result.samples,
          library_result.max_position_residual_m, library_result.max_orientation_residual_rad);

        // Ground-truth gate: compare against the live MoveIt RobotModel/URDF FK directly, not
        // against this library's own forward kinematics (see the function comment above).
        // Tolerance 1e-6 m / 1e-6 rad. An earlier ~7e-6 residual was a real mismatch (URDF
        // J_motor1 yaw is 3.1416, not pi), not rounding; keep this tight so small geometry
        // drift is caught.
        constexpr double kGroundTruthPositionTolM = 1e-6;
        constexpr double kGroundTruthOrientationTolRad = 1e-6;
        const auto index = analyticVariableIndex(joint_group);
        const auto& names = joint_group->getVariableNames();
        moveit::core::RobotState state(robot_model_);
        random_numbers::RandomNumberGenerator rng(20260929U);
        constexpr int kGroundTruthSamples = 2000;
        double max_pos_err = 0.0;
        double max_ang_err = 0.0;
        for (int sample = 0; sample < kGroundTruthSamples; ++sample) {
          rebel_analytic_ik::JointVector q{};
          std::vector<double> values(names.size(), 0.0);
          for (std::size_t j = 0; j < 6U; ++j) {
            q[j] = rng.uniformReal(limits[j].lower, limits[j].upper);
            values[index[j]] = q[j];
          }
          state.setJointGroupPositions(joint_group, values);
          state.update();
          // tool_link's pose relative to base_link, not to whatever the model's actual root
          // is (see base_link_model's comment above).
          const Eigen::Isometry3d moveit_fk =
            state.getGlobalLinkTransform(base_link_model).inverse() *
            state.getGlobalLinkTransform(settings_.tool_link);
          const Eigen::Isometry3d analytic_fk = rebel_analytic_ik::forwardKinematics(q);
          const double pos_err = (moveit_fk.translation() - analytic_fk.translation()).norm();
          const double ang_err =
            Eigen::AngleAxisd(moveit_fk.linear().transpose() * analytic_fk.linear()).angle();
          max_pos_err = std::max(max_pos_err, pos_err);
          max_ang_err = std::max(max_ang_err, ang_err);
          if (pos_err > kGroundTruthPositionTolM || ang_err > kGroundTruthOrientationTolRad) {
            RCLCPP_ERROR(
              get_logger(),
              "Analytic IK ground-truth self-check FAILED at sample %d/%d: live MoveIt "
              "RobotModel FK for tool_link '%s' disagrees with rebel_analytic_ik::"
              "forwardKinematics by %.3e m / %.3e rad (the URDF's arm chain, tool offset or "
              "mount may have changed since the library's constants were derived); disabling "
              "the analytic left-search backend for this process and falling back to kdl "
              "regardless of left_search_ik_backend",
              sample, kGroundTruthSamples, settings_.tool_link.c_str(), pos_err, ang_err);
            analytic_ik_available_.store(false);
            return;
          }
        }
        RCLCPP_INFO(
          get_logger(),
          "Analytic IK ground-truth self-check passed: %d/%d samples matched live MoveIt "
          "RobotModel FK for tool_link '%s', max residual %.3e m / %.3e rad",
          kGroundTruthSamples, kGroundTruthSamples, settings_.tool_link.c_str(), max_pos_err,
          max_ang_err);
        analytic_ik_available_.store(true);
      });
  }
  // Every closed-form branch for `target` (base_link frame), filtered to the group's live
  // joint limits, mapped into the group's variable order. Empty (not an error) for an
  // unreachable target; check analytic_ik_available_ / analyticJointLimitsFor separately to
  // distinguish "unreachable" from "analytic backend unusable for this group".
  static std::vector<std::vector<double> > analyticIkBranches(
    const moveit::core::JointModelGroup* group, const Eigen::Isometry3d& target)
  {
    std::array<rebel_analytic_ik::JointLimits, 6> limits;
    std::vector<std::vector<double> > branches;
    if (!analyticJointLimitsFor(group, limits)) return branches;
    const auto index = analyticVariableIndex(group);
    const auto& names = group->getVariableNames();
    for (const auto& solution : rebel_analytic_ik::solve(target, limits)) {
      std::vector<double> values(names.size());
      for (std::size_t j = 0; j < 6U; ++j) values[index[j]] = solution[j];
      branches.push_back(std::move(values));
    }
    return branches;
  }
  bool searchLeftParallelOrientation(
    const Snapshot& snapshot,
    const moveit::core::JointModelGroup* joint_group,
    const moveit::core::RobotState& seed_state,
    const std::vector<std::pair<geometry_msgs::msg::PoseStamped*, const char*> >& labeled_poses,
    const std::string& pose_context, std::string& error, bool handoff_mode = false)
  {
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    last_left_parallel_selection_ = LeftParallelSelection{};
    constexpr double kMaxStrokeJointStepRad = 5.0 * kDeg;
    constexpr int kRandomBranchAttempts = 3;  // fixed-seed restarts
    constexpr double kIkTimeoutS = 0.1;
    CutBasis basis;
    if (!cutDebugBasis(snapshot, "left", basis, error)) return false;
    planning_scene::PlanningScene scene(seed_state.getRobotModel());
    if (!scene.processCollisionObjectMsg(cartonCollisionObject(snapshot))) {
      error = "cannot add carton to left parallel orientation probe";
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(camera_pole_mutex_);
      if (camera_pole_) {
        if (!scene.processCollisionObjectMsg(*camera_pole_)) {
          error = "cannot add camera pole to left parallel orientation probe";
          return false;
        }
      } else if (settings_.camera_pole_required) {
        error = "camera pole collision object is unavailable for left parallel orientation probe";
        return false;
      }
    }
    // valid_state is defined per-scene inside evaluate_candidate below (each worker thread
    // gets its own PlanningScene, so it cannot be built once here against the shared
    // top-level `scene`).
    const auto* tool = seed_state.getRobotModel()->getLinkModel(settings_.tool_link);
    if (tool == nullptr || labeled_poses.empty()) {
      error = "left parallel orientation probe has no tool link or poses";
      return false;
    }
    ensureAnalyticIkSelfCheck(joint_group);
    const bool use_analytic_backend =
      settings_.left_search_ik_backend == "analytic" && analytic_ik_available_.load();
    const Eigen::Vector3d into_unit = -basis.normal;
    Eigen::Matrix3d reference;
    reference.col(0) = -basis.normal;
    reference.col(1) = -basis.along;
    reference.col(2) = reference.col(0).cross(reference.col(1));
    struct Candidate {
      double tilt, pitch, yaw, theta_p, blade_deviation, total, rotation;
      Eigen::Quaterniond orientation;
    };
    const auto multiples = [&](double max_deg) {
      std::vector<double> values;
      const double step = settings_.left_parallel_step_deg;
      const int count = static_cast<int>(std::floor(max_deg / step + 1e-9));
      for (int k = -count; k <= count; ++k) values.push_back(k * step);
      return values;
    };
    std::vector<Candidate> candidates;
    for (const double tilt : multiples(settings_.left_parallel_max_tilt_deg)) {
      for (const double pitch : multiples(settings_.left_parallel_max_tilt_deg)) {
        for (const double yaw : multiples(settings_.left_parallel_max_yaw_deg)) {
          const Eigen::Matrix3d rotation =
            (Eigen::AngleAxisd(yaw * kDeg, basis.normal) *
             Eigen::AngleAxisd(pitch * kDeg, basis.across) *
             Eigen::AngleAxisd(tilt * kDeg, basis.along)).toRotationMatrix() * reference;
          const double dot_x = rotation.col(0).dot(into_unit);
          const double dot_z = rotation.col(2).dot(into_unit);
          if (dot_x < 0.7 && dot_z < 0.7) continue;  // existing left penetration gate
          const double rotation_deg = std::acos(std::clamp(
            (rotation * reference.transpose()).trace() * 0.5 - 0.5, -1.0, 1.0)) / kDeg;
          if (handoff_mode) {
            const double limit = settings_.side_tape_handoff_max_search_angle_deg;
            if (std::abs(tilt) > limit + 1e-9 || std::abs(pitch) > limit + 1e-9 ||
                std::abs(yaw) > limit + 1e-9 || rotation_deg > limit + 1e-6) {
              continue;
            }
          }
          Candidate candidate;
          candidate.rotation = rotation_deg;
          candidate.tilt = tilt;
          candidate.pitch = pitch;
          candidate.yaw = yaw;
          candidate.theta_p =
            std::acos(std::min(1.0, std::abs(rotation.col(0).dot(basis.normal)))) / kDeg;
          candidate.blade_deviation =
            std::acos(std::clamp(rotation.col(1).dot(-basis.along), -1.0, 1.0)) / kDeg;
          candidate.total = std::abs(tilt) + std::abs(pitch) + std::abs(yaw);
          candidate.orientation = Eigen::Quaterniond(rotation).normalized();
          candidates.push_back(candidate);
        }
      }
    }
    const auto rounded = [](double value) { return std::round(value * 1e6); };
    std::sort(candidates.begin(), candidates.end(), [&](const Candidate& a, const Candidate& b) {
        if (handoff_mode) {
          // Side-tape hand-off: smallest deviation from F0 first (the side_tape calibration
          // frame), then the usual parallelism ordering.
          const auto hkey = [&](const Candidate& c) {
            return std::make_tuple(
              rounded(c.rotation), rounded(std::max({std::abs(c.tilt), std::abs(c.pitch),
                std::abs(c.yaw)})), rounded(c.total), rounded(c.theta_p), c.tilt, c.pitch, c.yaw);
          };
          return hkey(a) < hkey(b);
        }
        const auto key = [&](const Candidate& c) {
          return std::make_tuple(
            rounded(c.theta_p), rounded(c.blade_deviation), rounded(c.total), c.tilt, c.pitch, c.yaw);
        };
        return key(a) < key(b);
      });
    const auto& bounds = joint_group->getActiveJointModelsBounds();
    const double margin = settings_.left_parallel_min_joint_margin_deg * kDeg;
    std::vector<std::pair<double, double> > limits;
    for (const auto* joint_bounds : bounds) {
      if (joint_bounds->size() != 1U || !(*joint_bounds)[0].position_bounded_) {
        error = "left parallel orientation probe expects single-variable bounded joints";
        return false;
      }
      limits.emplace_back((*joint_bounds)[0].min_position_, (*joint_bounds)[0].max_position_);
    }
    const auto inside_margin = [&](const std::vector<double>& q, double& worst) {
        for (std::size_t j = 0; j < q.size() && j < limits.size(); ++j) {
          worst = std::min(worst, std::min(q[j] - limits[j].first, limits[j].second - q[j]));
        }
        return worst >= margin;
      };
    std::vector<double> seed_positions;
    seed_state.copyJointGroupPositions(joint_group, seed_positions);
    std::vector<Eigen::Vector3d> target_positions;
    for (const auto& labeled : labeled_poses) {
      target_positions.push_back(toEigen(labeled.first->pose.position));
    }
    struct LeftCandidateEval {
      bool any_ik{false};
      bool feasible{false};
      bool chain_failed{false};
      bool margin_failed{false};
      double best_worst{std::numeric_limits<double>::infinity()};
      std::vector<double> best_start;
      std::size_t branch_count{0};
    };
    // Evaluates one orientation candidate against the caller's `scene`. Every gate is
    // identical to the pre-Phase-1 search (penetration was already checked by the caller when
    // building `candidates`; IK/collision, the Cartesian chain, jump/bounds and the joint
    // margin below); only the branch-generation source (kdl vs analytic,
    // use_analytic_backend) differs. See the comment at the analytic branch below for exactly
    // how that can change a candidate's feasibility in either direction relative to kdl.
    const auto evaluate_candidate =
      [&](const Candidate& candidate, planning_scene::PlanningScene& scene) -> LeftCandidateEval {
        LeftCandidateEval result;
        std::vector<Eigen::Isometry3d> targets;
        for (const auto& position : target_positions) {
          Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
          target.translation() = position;
          target.linear() = candidate.orientation.toRotationMatrix();
          targets.push_back(target);
        }
        const moveit::core::GroupStateValidityCallbackFn valid_state =
          [this, &scene](moveit::core::RobotState* state,
                         const moveit::core::JointModelGroup* group, const double* values) {
            state->setJointGroupPositions(group, values);
            state->update();
            return !scene.isStateColliding(*state, settings_.planning_group);
          };
        random_numbers::RandomNumberGenerator rng(1234U);
        // Every collision-free first-pose branch the free-space transit could select. Every
        // such branch must carry the whole left chain, because the transit picks the branch
        // and the contact strokes run with jump detection off.
        std::vector<std::vector<double> > branches;
        const auto add_branch = [&](const std::vector<double>& values) {
            for (const auto& other : branches) {
              double diff = 0.0;
              for (std::size_t j = 0; j < values.size(); ++j) {
                diff = std::max(diff, std::abs(values[j] - other[j]));
              }
              if (diff < 1e-3) return;
            }
            moveit::core::RobotState probe(seed_state);
            probe.setJointGroupPositions(joint_group, values);
            probe.update();
            if (!probe.satisfiesBounds(joint_group) ||
                scene.isStateColliding(probe, settings_.planning_group)) {
              return;
            }
            const Eigen::Isometry3d& reached = probe.getGlobalLinkTransform(tool);
            if ((reached.translation() - targets.front().translation()).norm() > 1e-5 ||
                Eigen::AngleAxisd(reached.linear().transpose() * targets.front().linear()).angle() >
                1e-4) {
              return;
            }
            branches.push_back(values);
          };
        scene.getAllowedCollisionMatrixNonConst().setEntry(
          settings_.tool_link, settings_.carton_collision_id, false);
        if (use_analytic_backend) {
          // Closed-form branch enumeration (rebel_analytic_ik): every algebraic IK branch
          // within the live joint limits, computed directly from targets.front() with no
          // iterative search and no per-iteration collision checking. add_branch() above
          // still runs the same bounds/collision/reached-pose check as the kdl path, once per
          // branch. Any branch the kdl path reaches is itself a valid IK solution and so must
          // be among the analytic solver's algebraically-complete set (verified by
          // ensureAnalyticIkSelfCheck's startup gate and the package's gtest): analytic's
          // branch set for a given candidate is always a superset of (or equal to) whatever
          // kdl's finite seed search (seed + wristVariants + kRandomBranchAttempts restarts,
          // each a 0.1 s IK timeout) found for that same candidate.
          //
          // Feasibility below requires EVERY branch of a candidate to pass bounds, collision,
          // the whole Cartesian chain and the joint margin (not just one) -- because the
          // free-space transit can land on any collision-free hover-start branch, and every
          // one of them must be able to carry the left chain. A strictly larger branch set
          // therefore does NOT only make a candidate more likely to pass; it can move
          // feasibility in either direction relative to kdl, for two distinct reasons:
          //   - kdl fails closed immediately when it finds zero branches (any_ik false).
          //     If analytic finds at least one branch there that passes every check, the
          //     candidate flips infeasible -> feasible: analytic rescues a candidate kdl's
          //     finite search missed entirely.
          //   - If kdl found some (but not all) branches and every one of them happened to
          //     pass, kdl accepts the candidate on that incomplete set. If analytic's
          //     complete set includes an additional branch that fails the chain or margin
          //     check, the candidate flips feasible -> infeasible: analytic is stricter
          //     because it verifies every branch the transit could actually select, not just
          //     the ones kdl's finite/random search happened to land on.
          // Both directions are a correctness improvement over kdl (rescuing a real, fully
          // verified branch, or rejecting a candidate kdl only partially verified), never a
          // weakening of any individual check.
          for (const auto& branch : analyticIkBranches(joint_group, targets.front())) {
            add_branch(branch);
          }
        } else {
          std::vector<std::vector<double> > seeds{seed_positions};
          for (const auto& variant : wristVariants(joint_group, seed_positions)) {
            seeds.push_back(variant);
          }
          for (int attempt = 0; attempt < kRandomBranchAttempts; ++attempt) {
            moveit::core::RobotState random_state(seed_state);
            random_state.setToRandomPositions(joint_group, rng);
            std::vector<double> values;
            random_state.copyJointGroupPositions(joint_group, values);
            seeds.push_back(values);
          }
          for (const auto& seed : seeds) {
            moveit::core::RobotState state(seed_state);
            state.setJointGroupPositions(joint_group, seed);
            state.update();
            if (!state.setFromIK(joint_group, targets.front(), settings_.tool_link, kIkTimeoutS,
                                 valid_state)) {
              continue;
            }
            std::vector<double> solution;
            state.copyJointGroupPositions(joint_group, solution);
            add_branch(solution);
            for (const auto& variant : wristVariants(joint_group, solution)) add_branch(variant);
          }
        }
        result.branch_count = branches.size();
        result.any_ik = !branches.empty();
        bool feasible = result.any_ik;
        double best_worst = std::numeric_limits<double>::infinity();
        std::vector<double> best_start;
        double best_distance = std::numeric_limits<double>::infinity();
        for (const auto& start : branches) {
          if (!feasible) break;
          moveit::core::RobotState state(seed_state);
          state.setJointGroupPositions(joint_group, start);
          state.update();
          double worst = std::numeric_limits<double>::infinity();
          bool ok = inside_margin(start, worst);
          std::vector<double> previous = start;
          // Entry/cut/exit/lift strokes all run under the scoped tool/carton allowance.
          scene.getAllowedCollisionMatrixNonConst().setEntry(
            settings_.tool_link, settings_.carton_collision_id, settings_.allow_tool_carton_contact);
          bool chain_ok = true;
          for (std::size_t leg = 1; leg < targets.size() && chain_ok; ++leg) {
            std::vector<moveit::core::RobotStatePtr> waypoints;
            const double fraction = moveit::core::CartesianInterpolator::computeCartesianPath(
              &state, joint_group, waypoints, tool, targets[leg], true,
              moveit::core::MaxEEFStep(settings_.cartesian_step_m, 0.0),
              moveit::core::JumpThreshold(0.0), valid_state);
            if (fraction < 1.0 - 1e-9) {
              chain_ok = false;
              break;
            }
            for (const auto& waypoint : waypoints) {
              std::vector<double> q;
              waypoint->copyJointGroupPositions(joint_group, q);
              for (std::size_t j = 0; j < q.size(); ++j) {
                if (std::abs(q[j] - previous[j]) > kMaxStrokeJointStepRad) chain_ok = false;
              }
              if (!waypoint->satisfiesBounds(joint_group)) chain_ok = false;
              ok = inside_margin(q, worst) && ok;
              previous = q;
            }
          }
          if (!chain_ok) {
            result.chain_failed = true;
            feasible = false;
            break;
          }
          if (!ok) {
            result.margin_failed = true;
            feasible = false;
            break;
          }
          best_worst = std::min(best_worst, worst);
          double distance = 0.0;
          for (std::size_t j = 0; j < start.size(); ++j) {
            distance += std::abs(start[j] - seed_positions[j]);
          }
          if (distance < best_distance) {
            best_distance = distance;
            best_start = start;
          }
        }
        result.feasible = feasible;
        result.best_worst = best_worst;
        result.best_start = std::move(best_start);
        return result;
      };

    // Serial evaluation, stopping at the first feasible candidate: identical control flow to
    // the pre-Phase-1 search (evaluate_candidate above is a refactor for readability and for
    // the analytic/kdl backend switch, not a behaviour change). An earlier version of this
    // function also had an ordered-parallel (thread-pool) path with a per-thread
    // PlanningScene/RobotState; it was removed after mock testing showed it was NOT
    // deterministic (repeated identical requests sometimes selected a worse candidate than
    // the serial search). The confirmed root cause: MoveIt's JointModelGroup owns a single
    // shared kinematics solver plugin instance per group
    // (JointModelGroup::solver_instance_, moveit/robot_model/joint_model_group.h), reused by
    // every RobotState built from that group regardless of which PlanningScene it belongs to.
    // CartesianInterpolator::computeCartesianPath calls RobotState::setFromIK once per
    // waypoint (moveit/robot_state/cartesian_interpolator.h), which by default resolves to
    // that one shared solver instance (e.g. KDLKinematicsPlugin, whose KDL FK/IK solvers hold
    // mutable per-call working state as plain member fields, not thread-local or locked --
    // see moveit/kdl_kinematics_plugin/kdl_kinematics_plugin.h). Per-thread PlanningScene and
    // RobotState clones do not isolate that shared solver, so concurrent
    // computeCartesianPath/setFromIK calls across threads race on it. See the removed code's
    // history (this branch, prior commit) for the parallel implementation if this is revisited
    // with a per-thread kinematics solver or a lock around solver use.
    const auto started = std::chrono::steady_clock::now();
    std::size_t tried = 0, ik_failures = 0, chain_failures = 0, margin_failures = 0;
    std::size_t winning_index = candidates.size();
    LeftCandidateEval winning_eval;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
      if (elapsed > settings_.left_parallel_search_timeout_s) break;
      ++tried;
      LeftCandidateEval eval = evaluate_candidate(candidates[index], scene);
      if (!eval.any_ik) ++ik_failures;
      if (eval.chain_failed) ++chain_failures;
      if (eval.margin_failed) ++margin_failures;
      if (eval.feasible) {
        winning_index = index;
        winning_eval = std::move(eval);
        break;
      }
    }
    if (winning_index < candidates.size()) {
      const auto& candidate = candidates[winning_index];
      const auto& eval = winning_eval;
      for (const auto& labeled : labeled_poses) {
        labeled.first->pose.orientation = toMsg(candidate.orientation);
      }
      last_left_parallel_selection_.valid = true;
      last_left_parallel_selection_.tilt_deg = candidate.tilt;
      last_left_parallel_selection_.pitch_deg = candidate.pitch;
      last_left_parallel_selection_.yaw_deg = candidate.yaw;
      last_left_parallel_selection_.theta_p_deg = candidate.theta_p;
      last_left_parallel_selection_.rotation_from_f0_deg = candidate.rotation;
      last_left_parallel_selection_.candidate_index = tried - 1U;
      last_left_parallel_selection_.candidate_count = candidates.size();
      std::ostringstream start_deg;
      start_deg << std::fixed << std::setprecision(1);
      for (std::size_t j = 0; j < eval.best_start.size(); ++j) {
        start_deg << (j ? "," : "") << eval.best_start[j] / kDeg;
      }
      RCLCPP_INFO(
        get_logger(),
        "Left parallel orientation selected for %s: tilt %+.1f pitch %+.1f yaw %+.1f deg, "
        "blade-plane angle to top face theta_p %.2f deg, blade %.2f deg off the cut line, "
        "gate x=%.4f z=%.4f, min joint margin %.2f deg over all %zu hover-start branches "
        "(nearest [%s] deg) (candidate %zu of %zu, %.1f s, backend=%s)",
        pose_context.c_str(), candidate.tilt, candidate.pitch, candidate.yaw, candidate.theta_p,
        candidate.blade_deviation,
        (candidate.orientation * Eigen::Vector3d::UnitX()).dot(into_unit),
        (candidate.orientation * Eigen::Vector3d::UnitZ()).dot(into_unit), eval.best_worst / kDeg,
        eval.branch_count, start_deg.str().c_str(), winning_index + 1, candidates.size(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
        use_analytic_backend ? "analytic" : "kdl");
      return true;
    }
    std::ostringstream detail;
    detail << std::fixed << std::setprecision(1);
    detail << "left parallel orientation search failed closed for " << pose_context << ": tried " <<
      tried << " of " << candidates.size() << " gate-passing candidates (max_tilt=" <<
      settings_.left_parallel_max_tilt_deg << " max_yaw=" << settings_.left_parallel_max_yaw_deg <<
      " step=" << settings_.left_parallel_step_deg << " deg, timeout " <<
      settings_.left_parallel_search_timeout_s << " s); no collision-aware IK: " <<
      ik_failures <<
      ", Cartesian chain/jump/bounds failures: " << chain_failures <<
      ", joint margin < " << settings_.left_parallel_min_joint_margin_deg << " deg: " <<
      margin_failures;
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
    const auto kind = cutKindFromName(cut);
    if (!kind) {
      error = "unknown cut debug cut: " + cut;
      return false;
    }
    const CutSegment segment = segmentFor(snapshot, *kind);
    const auto& start = segment.start;
    const auto& end = segment.end;
    const bool is_middle = *kind == CutKind::kMiddle;
    const double tilt_deg = is_middle ? 0.0 :
      (*kind == CutKind::kRight ? settings_.right_side_cut_tilt_deg :
       settings_.left_side_cut_tilt_deg);
    const auto targets = orientationTargets(snapshot, is_middle, start, end, start, end);
    const double cut_clearance =
      is_middle ? settings_.middle_cut_clearance_m : settings_.cut_clearance_m;
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
    if (!robot_model_) {
      error = "robot model is unavailable; plan a cut debug stage before saving";
      return false;
    }
    moveit::core::RobotState state(robot_model_);
    state.setToDefaultValues();
    const auto& variables = robot_model_->getVariableNames();
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
    const Eigen::Isometry3d transform = state.getGlobalLinkTransform(settings_.tool_link);
    const Eigen::Vector3d position = transform.translation();
    const Eigen::Quaterniond orientation(transform.rotation());
    if (!position.allFinite() || !orientation.coeffs().allFinite()) {
      error = "forward kinematics produced a non-finite TCP pose";
      return false;
    }
    observed.header.frame_id = settings_.planning_frame;
    observed.header.stamp = now();
    observed.pose.position = toPoint(position);
    observed.pose.orientation = toMsg(orientation);
    return true;
  }
  static std::string jsonEscape(const std::string& value)
  {
    std::ostringstream out;
    for (const char character : value) {
      switch (character) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
          if (static_cast<unsigned char>(character) < 0x20U) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<int>(static_cast<unsigned char>(character)) << std::dec;
          } else {
            out << character;
          }
      }
    }
    return out.str();
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
    if (!settings_.cut_debug_output_dir.empty()) {
      directory = fs::path(settings_.cut_debug_output_dir);
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
  // Observed-minus-nominal TCP delta for one saved cut-debug observation.
  struct PoseDelta {
    Eigen::Vector3d world{Eigen::Vector3d::Zero()};
    CutLocalDeltas local{};
    geometry_msgs::msg::Quaternion world_q{toMsg(Eigen::Quaterniond::Identity())};
    geometry_msgs::msg::Quaternion local_q{toMsg(Eigen::Quaterniond::Identity())};
    double angle_deg{0.0};
  };
  static bool computePoseDelta(
    const geometry_msgs::msg::Pose& nominal, const geometry_msgs::msg::Pose& observed,
    const CutBasis& basis, const std::string& label, PoseDelta& delta, std::string& error)
  {
    delta.world = toEigen(observed.position) - toEigen(nominal.position);
    delta.local.along_m = delta.world.dot(basis.along);
    delta.local.across_m = delta.world.dot(basis.across);
    delta.local.normal_m = delta.world.dot(basis.normal);
    if (!std::isfinite(delta.local.along_m) || !std::isfinite(delta.local.across_m) ||
        !std::isfinite(delta.local.normal_m)) {
      error = "cut debug " + label +
        " translation delta is non-finite; refusing to persist an unbounded correction";
      return false;
    }
    const Eigen::Quaterniond nominal_q = toEigen(nominal.orientation);
    const Eigen::Quaterniond observed_q = toEigen(observed.orientation);
    if (!nominal_q.coeffs().allFinite() || !observed_q.coeffs().allFinite() ||
        std::abs(nominal_q.norm() - 1.0) > 0.01 || std::abs(observed_q.norm() - 1.0) > 0.01) {
      error = "nominal or observed orientation is not unit length";
      return false;
    }
    const Eigen::Quaterniond world_delta_q = (observed_q * nominal_q.inverse()).normalized();
    const Eigen::Quaterniond basis_q = basisQuaternion(basis);
    delta.world_q = toMsg(world_delta_q);
    delta.local_q = toMsg((basis_q.inverse() * world_delta_q * basis_q).normalized());
    delta.angle_deg = 2.0 * std::acos(std::min(1.0, std::abs(world_delta_q.w()))) * 180.0 /
      std::acos(-1.0);
    return true;
  }
  bool writeCutDebugRecord(bool complete, std::string& error)
  {
    if (!cut_debug_snapshot_ || cut_debug_record_path_.empty()) {
      error = "cut debug session has no snapshot or record path";
      return false;
    }
    const Snapshot& snapshot = *cut_debug_snapshot_;
    CutBasis basis;
    if (!cutDebugBasis(snapshot, cut_debug_session_cut_, basis, error)) {
      return false;
    }
    if (!cut_debug_has_observed_start_) {
      error = "cut debug start observation is missing";
      return false;
    }
    if (complete && !cut_debug_has_observed_end_) {
      error = "cut debug end observation is missing";
      return false;
    }
    PoseDelta start;
    if (!computePoseDelta(
          cut_debug_nominal_start_.pose, cut_debug_observed_start_.pose, basis, "start", start,
          error)) {
      return false;
    }
    PoseDelta end;
    if (complete &&
        !computePoseDelta(
          cut_debug_nominal_end_.pose, cut_debug_observed_end_.pose, basis, "end", end, error)) {
      return false;
    }
    const auto now_ns = nowNs();
    if (cut_debug_created_stamp_ns_ == 0) cut_debug_created_stamp_ns_ = now_ns;
    const auto vector3 = [](const Eigen::Vector3d& value) {
      return "[" + formatDouble(value.x()) + ", " + formatDouble(value.y()) + ", " +
        formatDouble(value.z()) + "]";
    };
    const auto quaternion_object = [](const geometry_msgs::msg::Quaternion& q) {
      return "{\"x\": " + formatDouble(q.x) + ", \"y\": " + formatDouble(q.y) + ", \"z\": " +
        formatDouble(q.z) + ", \"w\": " + formatDouble(q.w) + "}";
    };
    const auto pose_block = [&vector3](
      const char* name, const geometry_msgs::msg::Pose& pose, std::ostringstream& out) {
      out << "  \"" << name << "_position_m\": " << vector3(toEigen(pose.position)) << ",\n";
      out << "  \"" << name << "_orientation_xyzw\": [" << formatDouble(pose.orientation.x) <<
        ", " << formatDouble(pose.orientation.y) << ", " << formatDouble(pose.orientation.z) <<
        ", " << formatDouble(pose.orientation.w) << "],\n";
    };
    const auto local_block = [&quaternion_object](
      const char* prefix, const PoseDelta& delta, std::ostringstream& out) {
      out << "  \"" << prefix << "_delta_cut_local_along_m\": " <<
        formatDouble(delta.local.along_m) << ",\n";
      out << "  \"" << prefix << "_delta_cut_local_across_m\": " <<
        formatDouble(delta.local.across_m) << ",\n";
      out << "  \"" << prefix << "_delta_cut_local_normal_m\": " <<
        formatDouble(delta.local.normal_m) << ",\n";
      out << "  \"" << prefix << "_orientation_delta_world_xyzw\": " <<
        quaternion_object(delta.world_q) << ",\n";
      out << "  \"" << prefix << "_delta_cut_local_orientation_delta_local_xyzw\": " <<
        quaternion_object(delta.local_q) << ",\n";
    };
    std::ostringstream json;
    json << "{\n";
    json << "  \"schema\": \"" << cutDebugCorrectionSchema() << "\",\n";
    json << "  \"cut\": \"" << jsonEscape(cut_debug_session_cut_) << "\",\n";
    json << "  \"snapshot_id\": \"" << jsonEscape(snapshot.snapshot_id) << "\",\n";
    json << "  \"capture_id\": \"" << jsonEscape(snapshot.capture_id) << "\",\n";
    json << "  \"acquisition_stamp_ns\": " << snapshot.acquisition_stamp_ns << ",\n";
    json << "  \"planning_frame\": \"" << jsonEscape(settings_.planning_frame) << "\",\n";
    json << "  \"tool_link\": \"" << jsonEscape(settings_.tool_link) << "\",\n";
    json << "  \"status\": \"" << (complete ? "complete" : "start_saved") << "\",\n";
    json << "  \"created_stamp_ns\": " << cut_debug_created_stamp_ns_ << ",\n";
    json << "  \"updated_stamp_ns\": " << now_ns << ",\n";
    json << "  \"cut_basis_along_m\": " << vector3(basis.along) << ",\n";
    json << "  \"cut_basis_across_m\": " << vector3(basis.across) << ",\n";
    json << "  \"cut_basis_normal_m\": " << vector3(basis.normal) << ",\n";
    pose_block("nominal_start", cut_debug_nominal_start_.pose, json);
    pose_block("observed_start", cut_debug_observed_start_.pose, json);
    json << "  \"start_delta_world_m\": " << vector3(start.world) << ",\n";
    local_block("start", start, json);
    json << "  \"start_orientation_delta_angle_deg\": " << formatDouble(start.angle_deg, 6) << ",\n";
    pose_block("nominal_end", cut_debug_nominal_end_.pose, json);
    if (complete) {
      pose_block("observed_end", cut_debug_observed_end_.pose, json);
      json << "  \"end_delta_world_m\": " << vector3(end.world) << ",\n";
    } else {
      json << "  \"observed_end_position_m\": null,\n";
      json << "  \"observed_end_orientation_xyzw\": null,\n";
      json << "  \"end_delta_world_m\": null,\n";
    }
    local_block("end", end, json);
    json << "  \"end_orientation_delta_angle_deg\": " << formatDouble(end.angle_deg, 6) << "\n";
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
    cut_debug_start_local_ = start.local;
    if (complete) cut_debug_end_local_ = end.local;
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
    if (side_tape_handoff_info_.inProgress()) {
      message = "a side-tape hand-off is active or unresolved; call side_tape_retract before "
        "starting a cut debug session";
      return false;
    }
    if (topCutStagedInProgress()) {
      message = "a staged top cut is active; call top_cut_retract before starting a cut debug session";
      return false;
    }
    if (planned_task_ && planned_snapshot_ && !planned_task_->solutions().empty()) {
      message =
        "a normal H-cut plan is retained; call clear_plan before starting a cut debug session";
      return false;
    }
    const auto debug_cut = cutKindFromName(settings_.cut_debug_cut);
    if (!debug_cut) {
      message = "cut_debug_cut must be left, middle, or right";
      return false;
    }
    const auto source_path = externalSnapshotFile();
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
          *snapshot, settings_.cut_debug_cut, nominal_start, nominal_end, validation)) {
      message = "cut debug nominal computation failed for cut " + settings_.cut_debug_cut + ": " +
        validation;
      return false;
    }
    // The configured target (correction applied when opted in) is planned by the common
    // segment planner, stopping at the cut start: same dwell and wrist-guard validation
    // as full plans, with the carton added under the scoped contact allowance.
    CutSegment segment = segmentFor(*snapshot, *debug_cut);
    segment.stop_after_start = true;
    std::unique_ptr<mtc::Task> task;
    std::vector<geometry_msgs::msg::PoseStamped> targets;
    std::string planning_message;
    bool planned = false;
    try {
      const auto corrections = loadActiveCorrections({settings_.cut_debug_cut}, false);
      publishCartonGeometry(*snapshot);
      SegmentPlanOptions options;
      options.cut_debug_start = true;
      planned = planCutSegments(
        *snapshot, {segment}, corrections, settings_.motion_profile,
        "Cut debug " + settings_.cut_debug_cut + " start", options, task, targets,
        planning_message);
    } catch (const std::exception& exception) {
      planning_message = exception.what();
    }
    if (!planned) {
      message = "cut debug start planning failed for cut " + settings_.cut_debug_cut + ": " +
        planning_message;
      return false;
    }
    if (targets.size() != 1U) {
      message = "cut debug start planner produced an unexpected target count";
      return false;
    }
    publishMarkers(targets, {segment});
    publishTargetPoses(*snapshot, targets);
    publishSolution(*task);
    cut_debug_snapshot_ = *snapshot;
    cut_debug_session_cut_ = settings_.cut_debug_cut;
    cut_debug_nominal_start_ = nominal_start;
    cut_debug_nominal_end_ = nominal_end;
    cut_debug_has_observed_start_ = false;
    cut_debug_has_observed_end_ = false;
    cut_debug_created_stamp_ns_ = 0;
    planned_snapshot_ = *snapshot;
    planned_task_ = std::move(task);
    retained_with_stale_override_ = settings_.debug_allow_stale_snapshot;
    cut_debug_stage_ = CutDebugStage::kStartPlanned;
    message = "cut debug session started for cut " + cut_debug_session_cut_ +
      " snapshot " + snapshot->snapshot_id +
      "; configured start target published to RViz and retained; saved deltas use the uncorrected geometry baseline; execute explicitly via execute_planned_h_cut with both execution_enabled and debug_motion_execution_enabled; this service never dispatches motion";
    return true;
  }
  // One face of the `side_tape_seed` block (docs/side_tape_seed_contract.md, v1/v1.1) that
  // side_face_seed.py adds to the planner snapshot: the outer-wall plane (point+outward unit
  // normal) and the detected top edge, both in base_link metres. Only what the hand-off needs.
  struct SideTapeSeedFace {
    Eigen::Vector3d plane_point{Eigen::Vector3d::Zero()};
    Eigen::Vector3d plane_normal{Eigen::Vector3d::UnitZ()};  // outward, away from the box centre
    Eigen::Vector3d top_edge_start{Eigen::Vector3d::Zero()};
    Eigen::Vector3d top_edge_end{Eigen::Vector3d::Zero()};
    std::string wall_source;
  };
  static Eigen::Vector3d axisParamToEigen(const std::vector<double>& values)
  {
    return Eigen::Vector3d(values[0], values[1], values[2]);
  }
  // Fail-closed check for clear_plan's escape hatch (kApproachFailed only): the tip must be
  // in front of (or exactly on) the outer wall plane, i.e. not still penetrating the carton.
  bool sideTapeHandoffTipVerifiedOutside(std::string& message)
  {
    geometry_msgs::msg::PoseStamped current_pose;
    if (!captureObservedTcpPose(current_pose, message)) return false;
    const double depth_into_face_m = -side_tape_handoff_info_.outward_normal.dot(
      toEigen(current_pose.pose.position) - side_tape_handoff_info_.plane_point);
    if (depth_into_face_m > 0.0) {
      message = "tip is " + formatDouble(depth_into_face_m * 1000.0, 2) +
        " mm behind the outer wall plane (not verified outside)";
      return false;
    }
    return true;
  }
  // Minimum signed distance (metres; negative = behind/inside the outer wall) of the
  // tool_link origin to the recorded outer-wall plane, over EVERY waypoint of every
  // sub-trajectory in the solution. Used to hard-verify a standoff approach never dips
  // behind the wall, not just at its planned endpoints.
  double minSignedWallDistanceM(const mtc::SolutionBase& solution, const SideTapeSeedFace& seed_face)
  {
    moveit_task_constructor_msgs::msg::Solution solution_message;
    solution.toMsg(solution_message);
    double min_distance = std::numeric_limits<double>::infinity();
    moveit::core::RobotState state(robot_model_);
    state.setToDefaultValues();
    Eigen::Vector3d min_tip = Eigen::Vector3d::Zero();
    std::size_t min_sub = 0U;
    std::size_t sub_index = 0U;
    for (const auto& sub : solution_message.sub_trajectory) {
      ++sub_index;
      const auto& trajectory = sub.trajectory.joint_trajectory;
      for (const auto& point : trajectory.points) {
        for (std::size_t index = 0;
             index < trajectory.joint_names.size() && index < point.positions.size(); ++index) {
          state.setVariablePosition(trajectory.joint_names[index], point.positions[index]);
        }
        state.update();
        const Eigen::Vector3d tip = state.getGlobalLinkTransform(settings_.tool_link).translation();
        const double distance = seed_face.plane_normal.dot(tip - seed_face.plane_point);
        if (distance < min_distance) {
          min_distance = distance;
          min_tip = tip;
          min_sub = sub_index;
        }
      }
    }
    RCLCPP_INFO(
      get_logger(),
      "side-tape hand-off wall-distance scan: min %.2f mm at sub-trajectory %zu of %zu, "
      "tip (%.3f, %.3f, %.3f)", min_distance * 1000.0, min_sub,
      solution_message.sub_trajectory.size(), min_tip.x(), min_tip.y(), min_tip.z());
    return min_distance;
  }
  // Side-tape hand-off (left face only). Plans the full left cut for feasibility (proving
  // IK/collisions/the full min_fraction=1.0 stroke), then retains and executes only the
  // approach to the cut start pose: in_material stops at the planned in-material start;
  // standoff's single Cartesian entry targets the standoff pose directly and NEVER targets
  // (or passes through) the in-material pose (verified over every trajectory waypoint, not
  // just the endpoints). Never applies any orientation correction of its own: if the planned
  // start pose does not satisfy the configured side_tape tool-frame convention
  // (side_tape_handoff_*_axis_tool), this fails closed and reports the measured axis mapping
  // instead of rotating anything.
  bool planSideTapeHandoff(std::string& message)
  {
    if (!settings_.side_tape_handoff_enabled) {
      message = "side_tape_handoff_enabled is false";
      return false;
    }
    if (side_tape_handoff_info_.inProgress()) {
      message = "a side-tape hand-off is already active or unresolved; call side_tape_retract "
        "first (or clear_plan, once the tip is verified outside, if approach_failed)";
      return false;
    }
    if (topCutStagedInProgress()) {
      message = "a staged top cut is active; call top_cut_retract before planning a side-tape hand-off";
      return false;
    }
    if (cut_debug_stage_ != CutDebugStage::kNone) {
      message = "cut debug session active for cut " + cut_debug_session_cut_ +
        " stage " + cutDebugStageName(cut_debug_stage_) +
        "; call clear_plan before planning a side-tape hand-off";
      return false;
    }
    const std::string variant = settings_.side_tape_handoff_variant;
    if (variant != "in_material" && variant != "standoff") {
      message = "side_tape_handoff_variant must be in_material or standoff";
      return false;
    }
    const Eigen::Vector3d cut_axis_tool = axisParamToEigen(settings_.side_tape_handoff_cut_axis_tool);
    const Eigen::Vector3d lateral_axis_tool =
      axisParamToEigen(settings_.side_tape_handoff_lateral_axis_tool);
    const Eigen::Vector3d depth_axis_tool =
      axisParamToEigen(settings_.side_tape_handoff_depth_axis_tool);
    if (!settings_.left_parallel_orientation_enabled) {
      message = "side-tape hand-off refused: left_parallel_orientation_enabled must be true. "
        "side_tape is calibrated in the left_parallel tool frame (blade normal = tool X down, "
        "tool Z into the carton side, tool -Y = travel); the non-parallel left pose does not "
        "match that calibration";
      return false;
    }
    clearRetainedPlan();
    const auto source_path = externalSnapshotFile();
    Snapshot snapshot;
    SideTapeSeedFace seed_face;
    if (!loadSnapshotAndSeedFace(source_path, "left", snapshot, seed_face, message)) return false;
    std::string validation;
    if (!validateSnapshot(snapshot, validation)) {
      message = "snapshot rejected: " + validation;
      return false;
    }
    // Operator correction (2026-09-30): the detected box is offset from the real one. Shift
    // the whole snapshot (carton, cut points, seed wall plane) horizontally by the base_link
    // (dx, dy); apply dz only to the left cut line (seam height), so the carton top stays at
    // the detected height.
    {
      const auto& off = settings_.side_tape_handoff_box_offset_base_mm;
      if (off.size() != 3U || !std::isfinite(off[0]) || !std::isfinite(off[1]) ||
          !std::isfinite(off[2]) || std::abs(off[0]) > 50.0 || std::abs(off[1]) > 50.0 ||
          std::abs(off[2]) > 50.0) {
        message = "side_tape_handoff_box_offset_base_mm must be 3 finite values within +/-50 mm";
        return false;
      }
      const Eigen::Vector3d horizontal(off[0] / 1000.0, off[1] / 1000.0, 0.0);
      const Eigen::Vector3d vertical(0.0, 0.0, off[2] / 1000.0);
      for (auto* point : {&snapshot.seam_start, &snapshot.seam_end, &snapshot.seam_center,
          &snapshot.back_left, &snapshot.back_right, &snapshot.front_left, &snapshot.front_right,
          &snapshot.back_middle_left, &snapshot.front_middle_left, &snapshot.back_middle_right,
          &snapshot.front_middle_right}) {
        *point = toPoint(toEigen(*point) + horizontal);
      }
      snapshot.back_middle_left = toPoint(toEigen(snapshot.back_middle_left) + vertical);
      snapshot.front_middle_left = toPoint(toEigen(snapshot.front_middle_left) + vertical);
      seed_face.plane_point += horizontal;
      seed_face.top_edge_start += horizontal;
      seed_face.top_edge_end += horizontal;
      if (horizontal.norm() > 0.0 || vertical.norm() > 0.0) {
        RCLCPP_WARN(
          get_logger(), "side-tape hand-off: applying box offset (%.1f, %.1f, %.1f) mm in %s "
          "(xy to the whole snapshot, z to the left cut line only)", off[0], off[1], off[2],
          settings_.planning_frame.c_str());
      }
    }

    const CutSegment full_segment = segmentFor(snapshot, CutKind::kLeft);
    const auto corrections = loadActiveCorrections({"left"}, false);
    std::unique_ptr<mtc::Task> full_task;
    std::vector<geometry_msgs::msg::PoseStamped> full_targets;
    std::string plan_error;
    bool full_ok = false;
    // One left_parallel search for the whole hand-off: the full-stroke plan proves the
    // orientation feasible for the entire cut (all four poses), and the truncated approach
    // plan reuses exactly that orientation instead of re-searching start-only poses (which
    // could pick a different, smaller-deviation orientation the full stroke cannot carry).
    LeftRollCache handoff_roll_cache;
    try {
      publishCartonGeometry(snapshot);
      SegmentPlanOptions full_options;
      full_options.side_tape_handoff = true;
      full_options.left_roll_cache = &handoff_roll_cache;
      last_left_parallel_selection_ = LeftParallelSelection{};
      full_ok = planCutSegments(
        snapshot, {full_segment}, corrections, settings_.motion_profile,
        "side-tape hand-off feasibility (left, full stroke)", full_options, full_task,
        full_targets, plan_error);
    } catch (const std::exception& exception) {
      plan_error = exception.what();
    }
    if (!full_ok) {
      message = "side-tape hand-off feasibility check failed for the full left cut: " + plan_error;
      return false;
    }
    if (full_targets.size() != 2U) {
      message = "side-tape hand-off: unexpected target count for the full left segment";
      return false;
    }
    const LeftParallelSelection selection = last_left_parallel_selection_;
    const auto& in_material_start = full_targets[0];
    const auto& cut_end = full_targets[1];
    const Eigen::Vector3d start_to_end =
      toEigen(cut_end.pose.position) - toEigen(in_material_start.pose.position);
    const Eigen::Quaterniond q_start = toEigen(in_material_start.pose.orientation).normalized();
    const Eigen::Vector3d cut_axis_base_at_start = (q_start * cut_axis_tool).normalized();
    const double cut_length_mm = std::abs(start_to_end.dot(cut_axis_base_at_start)) * 1000.0;
    const double cut_length_euclidean_mm = start_to_end.norm() * 1000.0;
    if (std::abs(cut_length_mm - cut_length_euclidean_mm) > 0.5) {
      RCLCPP_WARN(
        get_logger(),
        "side-tape hand-off: cut_length projected onto the cut axis (%.3f mm) differs from "
        "the Euclidean cut length (%.3f mm) by more than 0.5 mm; using the projection",
        cut_length_mm, cut_length_euclidean_mm);
    }
    const double depth_into_face_m = -seed_face.plane_normal.dot(
      toEigen(in_material_start.pose.position) - seed_face.plane_point);
    const double penetration_mm = depth_into_face_m * 1000.0;

    CutSegment exec_segment = full_segment;
    exec_segment.stop_after_start = true;
    SegmentPlanOptions exec_options;
    exec_options.cut_debug_start = true;
    exec_options.side_tape_handoff = true;
    exec_options.left_roll_cache = &handoff_roll_cache;
    geometry_msgs::msg::PoseStamped standoff_target;
    if (variant == "standoff") {
      const double retreat_m = settings_.side_tape_handoff_standoff_mm / 1000.0 + depth_into_face_m;
      standoff_target = in_material_start;
      standoff_target.pose.position = toPoint(
        toEigen(in_material_start.pose.position) + retreat_m * seed_face.plane_normal);
      exec_options.stop_pose_override = standoff_target;
    }
    std::unique_ptr<mtc::Task> exec_task;
    std::vector<geometry_msgs::msg::PoseStamped> exec_targets;
    std::string exec_error;
    bool exec_ok = false;
    try {
      exec_ok = planCutSegments(
        snapshot, {exec_segment}, corrections, settings_.motion_profile,
        "side-tape hand-off approach (left, " + variant + ")", exec_options, exec_task,
        exec_targets, exec_error);
    } catch (const std::exception& exception) {
      exec_error = exception.what();
    }
    if (!exec_ok) {
      message = "side-tape hand-off approach planning failed: " + exec_error;
      return false;
    }
    if (exec_targets.size() != 1U) {
      message = "side-tape hand-off: unexpected target count for the truncated approach";
      return false;
    }
    if (!selection.valid) {
      message = "side-tape hand-off refused: the left_parallel orientation search did not "
        "produce a selection for the approach plan";
      return false;
    }
    if (selection.maxAbsAngleDeg() > settings_.side_tape_handoff_max_search_angle_deg + 1e-6 ||
        selection.rotation_from_f0_deg >
        settings_.side_tape_handoff_max_search_angle_deg + 1e-6) {
      message = "side-tape hand-off refused: the selected left_parallel orientation deviates " +
        formatDouble(selection.maxAbsAngleDeg(), 1) + " deg (rotation " +
        formatDouble(selection.rotation_from_f0_deg, 1) + " deg) from F0, above "
        "side_tape_handoff_max_search_angle_deg " +
        formatDouble(settings_.side_tape_handoff_max_search_angle_deg, 1);
      return false;
    }
    {
      const double orientation_mismatch_deg = 180.0 / 3.14159265358979323846 *
        Eigen::AngleAxisd(toEigen(in_material_start.pose.orientation).normalized().conjugate() *
        toEigen(exec_targets[0].pose.orientation).normalized()).angle();
      if (orientation_mismatch_deg > 0.5) {
        message = "side-tape hand-off: orientation mismatch between the full and truncated plans (" +
          formatDouble(orientation_mismatch_deg, 2) + " deg); refusing";
        return false;
      }
    }
    if (variant == "in_material") {
      // Sanity: the truncated plan's in-material start must match the full-plan feasibility
      // geometry; both come from the same segment and corrections, so a mismatch here is a
      // planning bug, not normal variance.
      const double geometry_mismatch_m = (toEigen(in_material_start.pose.position) -
        toEigen(exec_targets[0].pose.position)).norm();
      if (geometry_mismatch_m > 0.001) {
        message = "side-tape hand-off: geometry mismatch between the full and truncated plans (" +
          formatDouble(geometry_mismatch_m * 1000.0, 3) + " mm); refusing";
        return false;
      }
    }

    // Standoff must NEVER enter material: verify over every trajectory waypoint (not just
    // the planned endpoints) that the tip stays in front of (or on) the outer wall plane.
    double min_wall_distance_mm = std::numeric_limits<double>::quiet_NaN();
    if (variant == "standoff") {
      const double min_wall_distance_m =
        minSignedWallDistanceM(*exec_task->solutions().front(), seed_face);
      min_wall_distance_mm = min_wall_distance_m * 1000.0;
      constexpr double kWallDistanceToleranceM = -0.0005;  // small numerical/FK slack only
      if (min_wall_distance_m < kWallDistanceToleranceM) {
        message = "side-tape hand-off refused: the standoff approach trajectory dips " +
          formatDouble(-min_wall_distance_mm, 2) +
          " mm behind the outer wall plane at some waypoint (must stay in front of it); "
          "refusing to plan a standoff approach that enters material";
        return false;
      }
    }

    const Eigen::Vector3d travel_dir =
      (toEigen(snapshot.front_middle_left) - toEigen(snapshot.back_middle_left)).normalized();
    const Eigen::Vector3d inward_normal = -seed_face.plane_normal;
    const auto orientation_check = checkSideTapeOrientationConvention(
      exec_targets[0].pose.orientation, travel_dir, inward_normal, cut_axis_tool,
      lateral_axis_tool, depth_axis_tool,
      std::max(kSideTapeHandoffMaxAxisAngleDeg, settings_.side_tape_handoff_max_search_angle_deg));
    if (!orientation_check.ok) {
      message = "side-tape hand-off refused: the planned start pose does not satisfy the "
        "configured side_tape tool-frame convention; " + orientation_check.message +
        "; this looks like a genuine convention mismatch, not a small calibration error -- do "
        "not force this check open, fix side_tape_handoff_*_axis_tool (or the geometry) and "
        "re-plan";
      return false;
    }

    retainPlan(snapshot, {exec_segment}, exec_targets, std::move(exec_task));
    side_tape_handoff_snapshot_ = snapshot;
    side_tape_handoff_info_ = SideTapeHandoffInfo{};
    side_tape_handoff_info_.face = "left";
    side_tape_handoff_info_.variant = variant;
    side_tape_handoff_info_.cut_length_mm = cut_length_mm;
    side_tape_handoff_info_.cut_length_euclidean_mm = cut_length_euclidean_mm;
    side_tape_handoff_info_.penetration_mm = penetration_mm;
    side_tape_handoff_info_.standoff_mm =
      variant == "standoff" ? settings_.side_tape_handoff_standoff_mm : 0.0;
    side_tape_handoff_info_.min_wall_distance_mm = min_wall_distance_mm;
    side_tape_handoff_info_.start_pose = exec_targets[0];
    side_tape_handoff_info_.cut_axis_tool = cut_axis_tool;
    side_tape_handoff_info_.lateral_axis_tool = lateral_axis_tool;
    side_tape_handoff_info_.depth_axis_tool = depth_axis_tool;
    side_tape_handoff_info_.cut_dir_base = orientation_check.cut_dir_base;
    side_tape_handoff_info_.lateral_dir_base = orientation_check.lateral_dir_base;
    side_tape_handoff_info_.depth_dir_base = orientation_check.depth_dir_base;
    side_tape_handoff_info_.outward_normal = seed_face.plane_normal;
    side_tape_handoff_info_.plane_point = seed_face.plane_point;
    side_tape_handoff_info_.left_parallel = selection;
    side_tape_handoff_info_.planned_cut_start = toEigen(in_material_start.pose.position);
    side_tape_handoff_info_.planned_cut_end = toEigen(cut_end.pose.position);
    side_tape_handoff_info_.snapshot_path = source_path;
    side_tape_handoff_info_.snapshot_id = snapshot.snapshot_id;
    side_tape_handoff_info_.stale_override = settings_.debug_allow_stale_snapshot;
    side_tape_handoff_pending_ = true;
    message = "side-tape hand-off (" + variant + ") planned for the left cut: full-stroke "
      "feasibility proven (cut_length=" + formatDouble(cut_length_mm, 1) +
      " mm [euclidean " + formatDouble(cut_length_euclidean_mm, 1) +
      " mm], in-material penetration=" + formatDouble(penetration_mm, 2) + " mm" +
      (variant == "standoff" ?
         "; min_wall_distance=" + formatDouble(min_wall_distance_mm, 2) + " mm (>=0 required)" :
         "") +
      "); " + orientation_check.message +
      "; only the approach to the start pose is retained; execute via execute_planned_h_cut "
      "with execution_enabled, then call side_tape_retract once side_tape reports done";
    return true;
  }
  // Writes the side-tape hand-off record (JSON file, atomic temp+rename, plus a latched
  // ~/side_tape_handoff topic) after the truncated approach has actually completed
  // execution. Called only from the execution-completion path, under plan_mutex_.
  static const char* sideTapeHandoffStatusName(SideTapeHandoffStatus status)
  {
    switch (status) {
      case SideTapeHandoffStatus::kNone: return "none";
      case SideTapeHandoffStatus::kActive: return "active";
      case SideTapeHandoffStatus::kApproachFailed: return "approach_failed";
      case SideTapeHandoffStatus::kRetracted: return "retracted";
      case SideTapeHandoffStatus::kCutting: return "cutting";
      case SideTapeHandoffStatus::kCutFailed: return "cut_failed";
    }
    return "unknown";
  }
  static std::string vector3Json(const Eigen::Vector3d& v)
  {
    return "{\"x\": " + formatDouble(v.x()) + ", \"y\": " + formatDouble(v.y()) + ", \"z\": " +
      formatDouble(v.z()) + "}";
  }
  // Writes the side-tape hand-off record (JSON file, atomic temp+rename, plus the latched
  // ~/side_tape_handoff topic). Called on every status change: a successful approach
  // dispatch ("active"), a failed one ("approach_failed"), and a completed retract
  // ("retracted") -- so the topic always reflects the current status.
  void writeSideTapeHandoffOutput()
  {
    const auto& info = side_tape_handoff_info_;
    std::ostringstream json;
    json << "{\n";
    json << "  \"schema\": \"box_h_cut_side_tape_handoff/v1\",\n";
    json << "  \"status\": \"" << sideTapeHandoffStatusName(info.status) << "\",\n";
    json << "  \"face\": \"" << jsonEscape(info.face) << "\",\n";
    json << "  \"variant\": \"" << jsonEscape(info.variant) << "\",\n";
    json << "  \"cut_length_mm\": " << formatDouble(info.cut_length_mm, 3) << ",\n";
    json << "  \"cut_length_euclidean_mm\": " << formatDouble(info.cut_length_euclidean_mm, 3) <<
      ",\n";
    json << "  \"penetration_mm\": " << formatDouble(info.penetration_mm, 3) << ",\n";
    json << "  \"standoff_mm\": " << formatDouble(info.standoff_mm, 3) << ",\n";
    if (std::isfinite(info.min_wall_distance_mm)) {
      json << "  \"min_wall_distance_mm\": " << formatDouble(info.min_wall_distance_mm, 3) <<
        ",\n";
    }
    json << "  \"planning_frame\": \"" << jsonEscape(settings_.planning_frame) << "\",\n";
    json << "  \"tool_link\": \"" << jsonEscape(settings_.tool_link) << "\",\n";
    json << "  \"start_pose\": {\n";
    json << "    \"position_m\": {\"x\": " << formatDouble(info.start_pose.pose.position.x) <<
      ", \"y\": " << formatDouble(info.start_pose.pose.position.y) << ", \"z\": " <<
      formatDouble(info.start_pose.pose.position.z) << "},\n";
    json << "    \"orientation_xyzw\": {\"x\": " <<
      formatDouble(info.start_pose.pose.orientation.x) << ", \"y\": " <<
      formatDouble(info.start_pose.pose.orientation.y) << ", \"z\": " <<
      formatDouble(info.start_pose.pose.orientation.z) << ", \"w\": " <<
      formatDouble(info.start_pose.pose.orientation.w) << "}\n";
    json << "  },\n";
    json << "  \"side_tape_convention\": {\n";
    json << "    \"cut_axis_tool\": " << vector3Json(info.cut_axis_tool) << ",\n";
    json << "    \"lateral_axis_tool\": " << vector3Json(info.lateral_axis_tool) << ",\n";
    json << "    \"depth_axis_tool\": " << vector3Json(info.depth_axis_tool) << ",\n";
    json << "    \"cut_dir_base\": " << vector3Json(info.cut_dir_base) << ",\n";
    json << "    \"lateral_dir_base\": " << vector3Json(info.lateral_dir_base) << ",\n";
    json << "    \"depth_dir_base\": " << vector3Json(info.depth_dir_base) << "\n";
    json << "  },\n";
    json << "  \"left_parallel_orientation\": {\"tilt_deg\": " <<
      formatDouble(info.left_parallel.tilt_deg, 3) << ", \"pitch_deg\": " <<
      formatDouble(info.left_parallel.pitch_deg, 3) << ", \"yaw_deg\": " <<
      formatDouble(info.left_parallel.yaw_deg, 3) << ", \"theta_p_deg\": " <<
      formatDouble(info.left_parallel.theta_p_deg, 3) << ", \"candidate_index\": " <<
      info.left_parallel.candidate_index << ", \"candidate_count\": " <<
      info.left_parallel.candidate_count << ", \"rotation_from_f0_deg\": " <<
      formatDouble(info.left_parallel.rotation_from_f0_deg, 3) << ", \"max_abs_angle_deg\": " <<
      formatDouble(info.left_parallel.maxAbsAngleDeg(), 3) << ", \"max_allowed_deg\": " <<
      formatDouble(settings_.side_tape_handoff_max_search_angle_deg, 3) << "},\n";
    json << "  \"planned_cut_start_m\": " << vector3Json(info.planned_cut_start) << ",\n";
    json << "  \"planned_cut_end_m\": " << vector3Json(info.planned_cut_end) << ",\n";
    json << "  \"snapshot_path\": \"" << jsonEscape(info.snapshot_path) << "\",\n";
    json << "  \"snapshot_id\": \"" << jsonEscape(info.snapshot_id) << "\",\n";
    json << "  \"stamp_ns\": " << info.stamp.nanoseconds() << "\n";
    json << "}\n";
    const std::string payload = json.str();

    std::string output_path = settings_.side_tape_handoff_output_file;
    if (output_path.empty()) {
      const char* home = std::getenv("HOME");
      output_path = (home ? std::string(home) : std::string("/tmp")) + "/.ros/side_tape_handoff.json";
    }
    std::error_code code;
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path(), code);
    const std::string temporary = output_path + ".tmp";
    bool wrote = false;
    {
      std::ofstream file(temporary);
      if (file) {
        file << payload;
        file.flush();
        wrote = static_cast<bool>(file);
      }
    }
    if (!wrote) {
      RCLCPP_ERROR(
        get_logger(), "Cannot write side-tape hand-off record: %s", temporary.c_str());
    } else {
      std::filesystem::rename(temporary, output_path, code);
      if (code) {
        RCLCPP_ERROR(
          get_logger(), "Cannot atomically persist side-tape hand-off record: %s",
          code.message().c_str());
      }
    }
    std_msgs::msg::String topic_message;
    topic_message.data = payload;
    side_tape_handoff_pub_->publish(topic_message);
    RCLCPP_INFO(
      get_logger(),
      "Side-tape hand-off status=%s: face=%s variant=%s cut_length=%.1fmm penetration=%.2fmm "
      "output=%s",
      sideTapeHandoffStatusName(info.status), info.face.c_str(), info.variant.c_str(),
      info.cut_length_mm, info.penetration_mm, output_path.c_str());
  }
  // Plans (from the CURRENT robot state, since side_tape moved the arm with Servo) and
  // executes the retract: withdraw along the face's outward normal to clear the wall, then
  // the usual free-space transit back to the canonical left approach/parking pose. Refuses
  // if no hand-off is active. Every existing gate still applies (execution_enabled, the
  // free-space transit wrist guard); nothing here loosens them.
  // Refuses if MoveIt Servo appears to be actively commanding the arm: any message on its
  // twist/joint-jog command topics within the last side_tape_handoff_servo_quiescence_s, or
  // any joint velocity above side_tape_handoff_joint_velocity_threshold. Never pauses/stops
  // Servo itself.
  bool sideTapeServoIsQuiescent(std::string& message)
  {
    {
      std::lock_guard<std::mutex> lock(servo_activity_mutex_);
      const auto recent = [this](const std::optional<rclcpp::Time>& stamp) {
        return stamp.has_value() &&
          (now() - *stamp).seconds() < settings_.side_tape_handoff_servo_quiescence_s;
      };
      if (recent(servo_twist_last_received_)) {
        message = "MoveIt Servo command activity on " +
          settings_.side_tape_handoff_servo_twist_topic + " within the last " +
          formatDouble(settings_.side_tape_handoff_servo_quiescence_s, 2) + " s";
        return false;
      }
      if (recent(servo_joint_jog_last_received_)) {
        message = "MoveIt Servo command activity on " +
          settings_.side_tape_handoff_servo_joint_topic + " within the last " +
          formatDouble(settings_.side_tape_handoff_servo_quiescence_s, 2) + " s";
        return false;
      }
    }
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (current_joint_state_) {
      for (std::size_t index = 0; index < current_joint_state_->velocity.size(); ++index) {
        const double velocity = current_joint_state_->velocity[index];
        if (!std::isfinite(velocity) ||
            std::abs(velocity) > settings_.side_tape_handoff_joint_velocity_threshold) {
          const std::string joint_name = index < current_joint_state_->name.size() ?
            current_joint_state_->name[index] : std::to_string(index);
          message = "joint '" + joint_name + "' velocity " + formatDouble(velocity, 4) +
            " rad/s exceeds side_tape_handoff_joint_velocity_threshold " +
            formatDouble(settings_.side_tape_handoff_joint_velocity_threshold, 4) + " rad/s";
          return false;
        }
      }
    }
    return true;
  }
  struct CutPlanners {
    mtc::solvers::PlannerInterfacePtr transit;
    mtc::solvers::PlannerInterfacePtr stroke;
    std::shared_ptr<mtc::solvers::CartesianPath> cartesian;
  };
  // Adds the detected carton with the scoped tool/carton contact allowance and the camera
  // pole (shared by side_tape_retract and continue_side_cut).
  void addSideTapeCartonScene(mtc::Task& task, const Snapshot& snapshot)
  {
    auto add_carton = std::make_unique<mtc::stages::ModifyPlanningScene>(
      "add detected carton collision object");
    add_carton->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    add_carton->addObject(cartonCollisionObject(snapshot));
    if (settings_.allow_tool_carton_contact) {
      add_carton->allowCollisions(settings_.tool_link, settings_.carton_collision_id, true);
    }
    task.add(std::move(add_carton));
    addCameraPole(task);
  }
  // The side_tape retract stages, from an arm pose `from_pose`: withdraw along the outward
  // face normal by withdraw_m (contact allowance still active), then -- only if `fresh` --
  // restore collision checking and free-space transit to the left approach pose.
  bool addSideTapeRetractStages(
    mtc::Task& task, const CutPlanners& planners, const Snapshot& snapshot,
    const geometry_msgs::msg::PoseStamped& from_pose, double withdraw_m, bool fresh,
    std::vector<const mtc::Stage*>& transit_stages,
    std::vector<geometry_msgs::msg::PoseStamped>& published_targets, std::string& message)
  {
    const CutSegment left_segment = segmentFor(snapshot, CutKind::kLeft);
    const auto orientation_targets = orientationTargets(
      snapshot, false, left_segment.start, left_segment.end, left_segment.start,
      left_segment.end);
    auto parked_cut_start = targetPose(
      snapshot, left_segment.start, orientation_targets.first, settings_.cut_clearance_m,
      settings_.left_side_cut_tilt_deg);
    auto parked_hover_start = parked_cut_start;
    if (fresh) {
      std::string offset_error;
      if (!offsetLeftPoseOutward(
            snapshot, parked_hover_start, settings_.left_lateral_standoff_m, offset_error)) {
        message = "side-tape retract: cannot compute the parking pose: " + offset_error;
        return false;
      }
    }
    geometry_msgs::msg::PoseStamped withdraw_target = from_pose;
    withdraw_target.header.frame_id = settings_.planning_frame;
    withdraw_target.pose.position = toPoint(
      toEigen(from_pose.pose.position) + withdraw_m * side_tape_handoff_info_.outward_normal);
    // Operator 2026-10-01: accept a partial withdraw as long as the tip ends at least
    // side_tape_withdraw_min_clearance_m outside the outer wall (withdraw_m = depth +
    // hover_clearance_m); never one that leaves the blade in the wall.
    mtc::solvers::PlannerInterfacePtr withdraw_planner = planners.stroke;
    if (withdraw_m > 1e-6 && std::dynamic_pointer_cast<mtc::solvers::CartesianPath>(planners.stroke)) {
      const double needed_m = std::max(
        0.0, withdraw_m - settings_.hover_clearance_m + settings_.side_tape_withdraw_min_clearance_m);
      const double min_fraction = std::clamp(needed_m / withdraw_m, 0.0, 1.0);
      if (min_fraction < 1.0 - 1e-9) {
        auto partial = std::make_shared<mtc::solvers::CartesianPath>();
        partial->setIKFrame(settings_.tool_link);
        partial->setStepSize(settings_.cartesian_step_m);
        partial->setJumpThreshold(0.0);
        partial->setMinFraction(min_fraction);
        partial->setMaxVelocityScalingFactor(settings_.velocity_scaling);
        partial->setMaxAccelerationScalingFactor(settings_.acceleration_scaling);
        withdraw_planner = partial;
      }
    }
    addMoveTo(
      task, "side-tape retract: withdraw along face normal [" + settings_.motion_profile + "]",
      withdraw_planner, withdraw_target);
    published_targets.push_back(withdraw_target);
    if (fresh) {
      if (settings_.allow_tool_carton_contact) {
        addCartonContactScope(
          task, "restore knife-tip collision checking after side-tape retract", false);
      }
      transit_stages.push_back(addMoveTo(
        task, "side-tape retract: transit to left approach pose [" + settings_.motion_profile +
          "]",
        planners.transit, parked_hover_start));
      published_targets.push_back(parked_hover_start);
    }
    return true;
  }
  // Plans (from the CURRENT robot state, since side_tape moved the arm with Servo) and
  // executes the retract. Refuses if no hand-off is in progress, if planned/loaded under
  // debug_allow_stale_snapshot (never executed at all), if Servo appears active, or if the
  // current pose deviates too far from the recorded hand-off (orientation or off the cut
  // line) -- side_tape may have misbehaved and the operator should recover manually. The
  // withdraw (along the outer-wall normal, by the current penetration behind the wall
  // clamped >=0, plus hover_clearance_m) always runs when everything above passes; the
  // free-space transit back to the canonical parking pose only runs if the snapshot is still
  // fresh (a stale snapshot means only the withdraw executes). Every existing gate still
  // applies (execution_enabled, the free-space transit wrist guard); nothing here loosens
  // them, and Servo is never paused/stopped.
  bool planAndExecuteSideTapeRetract(std::string& message)
  {
    if (!side_tape_handoff_info_.inProgress()) {
      message = "no side-tape hand-off is active; nothing to retract";
      return false;
    }
    if (side_tape_handoff_info_.stale_override) {
      message = "side-tape retract refused: this hand-off was planned under "
        "debug_allow_stale_snapshot=true (plan-only override); it can never be executed, "
        "including the retract withdraw; recover the arm manually (teleop)";
      return false;
    }
    if (!execution_enabled_.load()) {
      message = "execution service is disabled by execution_enabled";
      return false;
    }
    if (execution_in_progress_) {
      message = "an H-cut trajectory execution is already active";
      return false;
    }
    if (!side_tape_handoff_snapshot_) {
      message = "side-tape hand-off snapshot is unavailable; internal state error, contact the maintainer";
      return false;
    }
    std::string servo_error;
    if (!sideTapeServoIsQuiescent(servo_error)) {
      message = "side-tape retract refused: " + servo_error +
        "; Servo is left running -- wait for it to go quiet, or stop it yourself, before retracting";
      return false;
    }
    const Snapshot snapshot = *side_tape_handoff_snapshot_;
    geometry_msgs::msg::PoseStamped current_pose;
    if (!captureObservedTcpPose(current_pose, message)) return false;

    const Eigen::Quaterniond q_current = toEigen(current_pose.pose.orientation).normalized();
    const Eigen::Quaterniond q_recorded =
      toEigen(side_tape_handoff_info_.start_pose.pose.orientation).normalized();
    const double orientation_deviation_deg =
      q_current.angularDistance(q_recorded) * 180.0 / 3.14159265358979323846;
    if (orientation_deviation_deg > settings_.side_tape_handoff_retract_max_orientation_deg) {
      message = "side-tape retract refused: current tool orientation deviates " +
        formatDouble(orientation_deviation_deg, 1) +
        " deg from the recorded hand-off start orientation (max " +
        formatDouble(settings_.side_tape_handoff_retract_max_orientation_deg, 1) +
        " deg); side_tape may have misbehaved -- recover the arm manually (teleop) first";
      return false;
    }
    const Eigen::Vector3d current_position = toEigen(current_pose.pose.position);
    const Eigen::Vector3d recorded_start_position =
      toEigen(side_tape_handoff_info_.start_pose.pose.position);
    const Eigen::Vector3d delta = current_position - recorded_start_position;
    const Eigen::Vector3d cut_dir_base = side_tape_handoff_info_.cut_dir_base.normalized();
    const double line_offset_mm =
      (delta - delta.dot(cut_dir_base) * cut_dir_base).norm() * 1000.0;
    if (line_offset_mm > settings_.side_tape_handoff_retract_max_line_offset_mm) {
      message = "side-tape retract refused: current tip is " + formatDouble(line_offset_mm, 1) +
        " mm off the recorded cut line (max " +
        formatDouble(settings_.side_tape_handoff_retract_max_line_offset_mm, 1) +
        " mm); side_tape may have misbehaved -- recover the arm manually (teleop) first";
      return false;
    }

    // Withdraw distance: current penetration behind the outer wall (clamped >= 0, so an
    // already-outside tip only gets the plain hover clearance) plus hover_clearance_m.
    const double current_depth_into_face_m = -side_tape_handoff_info_.outward_normal.dot(
      current_position - side_tape_handoff_info_.plane_point);
    const double withdraw_m =
      std::max(0.0, current_depth_into_face_m) + settings_.hover_clearance_m;

    // Never refuse the withdraw for a stale snapshot (stranding the blade in the box), but
    // the free-space transit back to the parking pose follows the normal freshness rule.
    std::string freshness_error;
    const bool fresh = snapshotIsFresh(snapshot, freshness_error);

    auto task = std::make_unique<mtc::Task>();
    task->stages()->setName("side-tape hand-off retract (left)");
    if (robot_model_) {
      task->setRobotModel(robot_model_);
    } else {
      task->loadRobotModel(shared_from_this());
      robot_model_ = task->getRobotModel();
    }
    task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));
    addSideTapeCartonScene(*task, snapshot);
    const CutPlanners planners = makeCutPlanners(settings_.motion_profile);
    std::vector<const mtc::Stage*> transit_stages;
    std::vector<geometry_msgs::msg::PoseStamped> published_targets;
    if (!addSideTapeRetractStages(
          *task, planners, snapshot, current_pose, withdraw_m, fresh, transit_stages,
          published_targets, message)) {
      return false;
    }

    try {
      task->init();
    } catch (const mtc::InitStageException& exception) {
      std::ostringstream details;
      details << exception;
      message = "side-tape retract MTC stage initialization failed: " + details.str();
      return false;
    }
    const auto result = task->plan(static_cast<std::size_t>(settings_.max_solutions));
    if (result != moveit::core::MoveItErrorCode::SUCCESS || task->solutions().empty()) {
      task->explainFailure();
      message = "side-tape retract planning failed";
      return false;
    }
    if (!transit_stages.empty()) {
      const auto& solution = *task->solutions().front();
      moveit_task_constructor_msgs::msg::Solution solution_message;
      solution.toMsg(solution_message);
      std::string wrist_error;
      if (!validateFreeSpaceTransitWrist(
            solution, solution_message, transit_stages, wrist_error)) {
        message = wrist_error;
        return false;
      }
    }
    publishTargetPoses(snapshot, published_targets);
    publishSolution(*task);

    if (execution_thread_.joinable()) execution_thread_.join();
    execution_in_progress_ = true;
    execution_thread_ = std::thread(
      [this, task = std::move(task), fresh, freshness_error]() mutable {
        bool succeeded = false;
        std::string failure_detail;
        try {
          const auto exec_result = task->execute(*task->solutions().front());
          succeeded = (exec_result == moveit::core::MoveItErrorCode::SUCCESS);
          if (succeeded) {
            RCLCPP_INFO(
              get_logger(), "Side-tape hand-off retract execution completed (%s)",
              fresh ? "withdraw + transit" : "withdraw only, stale snapshot");
          } else {
            failure_detail = "controller reported failure status " + std::to_string(exec_result.val);
            RCLCPP_ERROR(
              get_logger(), "Side-tape hand-off retract execution failed: %s",
              failure_detail.c_str());
          }
        } catch (const std::exception& exception) {
          failure_detail = std::string("threw exception: ") + exception.what();
          RCLCPP_ERROR(
            get_logger(), "Side-tape hand-off retract execution %s", failure_detail.c_str());
        }
        std::lock_guard<std::mutex> completion_lock(plan_mutex_);
        execution_in_progress_ = false;
        if (succeeded) {
          side_tape_handoff_info_.status = SideTapeHandoffStatus::kRetracted;
          side_tape_handoff_info_.stamp = now();
          writeSideTapeHandoffOutput();
          side_tape_handoff_info_ = SideTapeHandoffInfo{};
          side_tape_handoff_snapshot_.reset();
          if (!fresh) {
            RCLCPP_WARN(
              get_logger(),
              "Side-tape retract withdrew the blade but skipped the return transit (%s); "
              "move away by teleop or start a fresh plan_h_cut/plan_side_tape_handoff",
              freshness_error.c_str());
          }
        } else {
          RCLCPP_ERROR(
            get_logger(),
            "Side-tape retract did NOT complete (%s); the hand-off is still unresolved; "
            "call side_tape_retract again once the arm is clear",
            failure_detail.c_str());
        }
      });
    message = fresh ?
      "side-tape retract dispatch accepted (withdraw + return transit); monitor controller execution" :
      "side-tape retract dispatch accepted: withdrawn; transit skipped (stale snapshot: " +
        freshness_error + "), move away by teleop/replan; monitor controller execution";
    return true;
  }
  // ~/continue_side_cut: after side_tape has aligned the tip into the side seam (Servo), plans
  // ONE MTC task from the CURRENT robot state -- a straight Cartesian cut stroke (contact
  // scaling, min_fraction=1.0, scoped tool/carton contact allowance) parallel to the planned
  // left cut line, keeping the current tool orientation and the alignment's lateral/depth
  // offset, ending on the plane through the planned cut end perpendicular to that line --
  // followed by exactly the side_tape_retract stages (withdraw along the outward face normal by
  // the depth at the stroke end + hover_clearance_m, then the free-space transit). Every
  // refusal happens before anything moves. Status: cutting while executing, retracted on
  // success, cut_failed on failure (hand-off state and carton kept; side_tape_retract recovers).
  // ~/side_cut_penetrate (operator 2026-10-01): staged side cut like the top cut. After the
  // standoff approach (hand-off status 'active', variant standoff), move straight from the
  // CURRENT pose into the face to the planned in-material cut start (box offset applied at
  // plan time), keeping the current orientation, and stop there so the operator can check /
  // teleop-correct the start. continue_side_cut then cuts from wherever the tip is.
  bool sideCutPenetrate(std::string& message)
  {
    const auto& info = side_tape_handoff_info_;
    if (info.status != SideTapeHandoffStatus::kActive || info.variant != "standoff") {
      message = "side_cut_penetrate refused: needs an active side-tape hand-off with variant "
        "'standoff' (plan_side_tape_handoff + execute_planned_h_cut first)";
      return false;
    }
    if (info.stale_override || settings_.debug_allow_stale_snapshot) {
      message = "side_cut_penetrate refused: planned/loaded under debug_allow_stale_snapshot";
      return false;
    }
    if (!execution_enabled_.load()) {
      message = "execution service is disabled by execution_enabled";
      return false;
    }
    if (execution_in_progress_) {
      message = "an H-cut trajectory execution is already active";
      return false;
    }
    if (!side_tape_handoff_snapshot_) {
      message = "side-tape hand-off snapshot is unavailable; internal state error";
      return false;
    }
    std::string servo_error;
    if (!sideTapeServoIsQuiescent(servo_error)) {
      message = "side_cut_penetrate refused: " + servo_error;
      return false;
    }
    geometry_msgs::msg::PoseStamped current;
    if (!captureObservedTcpPose(current, message)) return false;
    const double offset_mm =
      (toEigen(current.pose.position) - toEigen(info.start_pose.pose.position)).norm() * 1000.0;
    const double orient_deg = angleBetweenDeg(current.pose.orientation, info.start_pose.pose.orientation);
    if (offset_mm > 10.0 || orient_deg > 10.0) {
      message = "side_cut_penetrate refused: tip is " + formatDouble(offset_mm, 1) + " mm / " +
        formatDouble(orient_deg, 1) + " deg from the standoff pose (max 10 mm / 10 deg)";
      return false;
    }
    const Snapshot snapshot = *side_tape_handoff_snapshot_;
    auto task = std::make_unique<mtc::Task>();
    task->stages()->setName("side cut penetrate (left)");
    if (robot_model_) {
      task->setRobotModel(robot_model_);
    } else {
      task->loadRobotModel(shared_from_this());
      robot_model_ = task->getRobotModel();
    }
    task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));
    addSideTapeCartonScene(*task, snapshot);
    const CutPlanners planners = makeCutPlanners(settings_.motion_profile);
    geometry_msgs::msg::PoseStamped target = current;
    target.header.frame_id = settings_.planning_frame;
    target.pose.position = toPoint(info.planned_cut_start);
    addMoveTo(
      *task, "side cut: penetrate to planned start [" + settings_.motion_profile + "]",
      planners.stroke, target);
    try {
      task->init();
    } catch (const mtc::InitStageException& exception) {
      std::ostringstream details;
      details << exception;
      message = "side_cut_penetrate MTC stage initialization failed: " + details.str();
      return false;
    }
    if (task->plan(static_cast<std::size_t>(settings_.max_solutions)) !=
          moveit::core::MoveItErrorCode::SUCCESS || task->solutions().empty()) {
      task->explainFailure();
      message = "side_cut_penetrate planning failed; nothing moved";
      return false;
    }
    publishTargetPoses(snapshot, {target});
    publishSolution(*task);
    if (execution_thread_.joinable()) execution_thread_.join();
    execution_in_progress_ = true;
    const double depth_mm =
      -info.outward_normal.dot(info.planned_cut_start - info.plane_point) * 1000.0;
    execution_thread_ = std::thread([this, task = std::move(task)]() mutable {
      bool ok = false;
      try {
        ok = task->execute(*task->solutions().front()) == moveit::core::MoveItErrorCode::SUCCESS;
      } catch (const std::exception& exception) {
        RCLCPP_ERROR(get_logger(), "side_cut_penetrate execution threw: %s", exception.what());
      }
      std::lock_guard<std::mutex> completion_lock(plan_mutex_);
      execution_in_progress_ = false;
      if (ok) {
        RCLCPP_INFO(get_logger(), "side_cut_penetrate completed: tip at the planned in-material "
          "start; check/teleop, then continue_side_cut");
      } else {
        side_tape_handoff_info_.status = SideTapeHandoffStatus::kCutFailed;
        side_tape_handoff_info_.stamp = now();
        writeSideTapeHandoffOutput();
        RCLCPP_ERROR(get_logger(), "side_cut_penetrate did NOT complete; status cut_failed; "
          "recover with side_tape_retract");
      }
    });
    message = "side_cut_penetrate dispatch accepted: " + formatDouble(offset_mm, 1) +
      " mm from standoff, into the face to the planned start (" + formatDouble(depth_mm, 1) +
      " mm behind the outer wall); then check/teleop and call continue_side_cut";
    return true;
  }
  bool continueSideCut(std::string& message)
  {
    constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
    if (side_tape_handoff_info_.status != SideTapeHandoffStatus::kActive) {
      message = side_tape_handoff_info_.inProgress() ?
        "continue_side_cut refused: hand-off status is not 'active' (approach_failed/cutting/"
        "cut_failed cannot continue); recover with side_tape_retract" :
        "continue_side_cut refused: no active side-tape hand-off";
      return false;
    }
    if (side_tape_handoff_info_.stale_override || settings_.debug_allow_stale_snapshot) {
      message = "continue_side_cut refused: planned/loaded under debug_allow_stale_snapshot "
        "(plan-only override); use side_tape_retract / manual recovery";
      return false;
    }
    if (!execution_enabled_.load()) {
      message = "execution service is disabled by execution_enabled";
      return false;
    }
    if (execution_in_progress_) {
      message = "an H-cut trajectory execution is already active";
      return false;
    }
    if (!side_tape_handoff_snapshot_) {
      message = "side-tape hand-off snapshot is unavailable; internal state error";
      return false;
    }
    const Snapshot snapshot = *side_tape_handoff_snapshot_;
    std::string freshness_error;
    // The snapshot age check can be turned off for this call only (operator: the box does
    // not move and alignment takes longer than external_snapshot_max_age_s). The hand-off
    // itself was planned from a fresh snapshot, and debug_allow_stale plans stay refused above.
    if (!settings_.side_tape_continue_check_snapshot_age) {
      if (!snapshotIsFresh(snapshot, freshness_error)) {
        RCLCPP_WARN(
          get_logger(), "continue_side_cut: snapshot age not checked (%s); box assumed static",
          freshness_error.c_str());
      }
    } else if (!snapshotIsFresh(snapshot, freshness_error)) {
      message = "continue_side_cut refused: " + freshness_error +
        "; the cut is a perception-based contact motion -- use side_tape_retract instead";
      return false;
    }
    std::string servo_error;
    if (!sideTapeServoIsQuiescent(servo_error)) {
      message = "continue_side_cut refused: " + servo_error;
      return false;
    }
    geometry_msgs::msg::PoseStamped current_pose;
    if (!captureObservedTcpPose(current_pose, message)) return false;
    const auto& info = side_tape_handoff_info_;
    const Eigen::Vector3d p0 = toEigen(current_pose.pose.position);
    const Eigen::Vector3d line_vector = info.planned_cut_end - info.planned_cut_start;
    if (line_vector.norm() < 1e-3) {
      message = "continue_side_cut refused: recorded planned cut line is degenerate";
      return false;
    }
    const Eigen::Vector3d d = line_vector.normalized();
    const Eigen::Vector3d offset = p0 - info.planned_cut_start;
    const double along_mm = offset.dot(d) * 1000.0;
    const double perpendicular_mm = (offset - offset.dot(d) * d).norm() * 1000.0;
    const double depth_mm = -info.outward_normal.dot(p0 - info.plane_point) * 1000.0;
    const double orientation_deg =
      toEigen(current_pose.pose.orientation).normalized().angularDistance(
        toEigen(info.start_pose.pose.orientation).normalized()) * kRadToDeg;
    std::ostringstream measured;
    measured << std::fixed << std::setprecision(2) << " [measured: perpendicular " <<
      perpendicular_mm << " mm, along " << along_mm << " mm, depth " << depth_mm <<
      " mm, orientation " << orientation_deg << " deg]";
    if (perpendicular_mm > settings_.side_tape_continue_max_line_offset_mm) {
      message = "continue_side_cut refused: tip is off the planned cut line by more than " +
        formatDouble(settings_.side_tape_continue_max_line_offset_mm, 1) + " mm" + measured.str();
      return false;
    }
    if (std::abs(along_mm) > settings_.side_tape_continue_max_along_offset_mm) {
      message = "continue_side_cut refused: tip is more than " +
        formatDouble(settings_.side_tape_continue_max_along_offset_mm, 1) +
        " mm from the planned cut start along the line" + measured.str();
      return false;
    }
    if (depth_mm < settings_.side_tape_continue_min_depth_mm) {
      message = "continue_side_cut refused: tip is not in the material (depth behind the outer "
        "wall below " + formatDouble(settings_.side_tape_continue_min_depth_mm, 1) + " mm)" +
        measured.str();
      return false;
    }
    if (depth_mm > settings_.side_tape_continue_max_depth_mm) {
      message = "continue_side_cut refused: tip is too deep behind the outer wall (above " +
        formatDouble(settings_.side_tape_continue_max_depth_mm, 1) + " mm)" + measured.str();
      return false;
    }
    if (orientation_deg > settings_.side_tape_continue_max_orientation_deg) {
      message = "continue_side_cut refused: tool orientation deviates more than " +
        formatDouble(settings_.side_tape_continue_max_orientation_deg, 1) +
        " deg from the hand-off start orientation" + measured.str();
      return false;
    }
    // Level stroke (operator 2026-09-30): keep the z the alignment program set at the start
    // for the whole cut; run along the planned line's horizontal direction. Extend (or
    // shorten) the stroke past the planned cut end, toward the box edge.
    Eigen::Vector3d d_level(d.x(), d.y(), 0.0);
    if (d_level.norm() < 0.5) {
      message = "continue_side_cut refused: planned cut line is too steep for a level stroke";
      return false;
    }
    d_level.normalize();
    const double stroke_m = (info.planned_cut_end - p0).dot(d_level) +
      settings_.side_tape_continue_end_extension_mm / 1000.0;
    if (stroke_m < 0.005) {
      message = "continue_side_cut refused: remaining stroke to the planned cut end is only " +
        formatDouble(stroke_m * 1000.0, 2) + " mm" + measured.str();
      return false;
    }
    // Operator 2026-09-30: end the stroke side_tape_continue_end_extra_depth_mm deeper into
    // the face (horizontal inward normal, so z stays level); the stroke angles in gradually.
    Eigen::Vector3d inward_level(-info.outward_normal.x(), -info.outward_normal.y(), 0.0);
    if (inward_level.norm() < 0.5) {
      message = "continue_side_cut refused: face normal is not horizontal enough";
      return false;
    }
    inward_level.normalize();
    const Eigen::Vector3d stroke_end = p0 + d_level * stroke_m +
      inward_level * (settings_.side_tape_continue_end_extra_depth_mm / 1000.0);
    const double end_depth_m = -info.outward_normal.dot(stroke_end - info.plane_point);
    if (!settings_.side_tape_continue_check_end_depth) {
      RCLCPP_WARN(
        get_logger(), "continue_side_cut: stroke end %.2f mm behind the outer wall (end-depth "
        "check disabled)", end_depth_m * 1000.0);
    } else if (end_depth_m * 1000.0 > settings_.side_tape_continue_max_depth_mm) {
      message = "continue_side_cut refused: stroke end would be " +
        formatDouble(end_depth_m * 1000.0, 2) + " mm behind the outer wall (max " +
        formatDouble(settings_.side_tape_continue_max_depth_mm, 1) + " mm)" + measured.str();
      return false;
    }
    const double withdraw_m = std::max(0.0, end_depth_m) + settings_.hover_clearance_m;

    auto task = std::make_unique<mtc::Task>();
    task->stages()->setName("side-tape continue cut (left): stroke + retract");
    if (robot_model_) {
      task->setRobotModel(robot_model_);
    } else {
      task->loadRobotModel(shared_from_this());
      robot_model_ = task->getRobotModel();
    }
    task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));
    addSideTapeCartonScene(*task, snapshot);
    const CutPlanners planners = makeCutPlanners(settings_.motion_profile);
    // The cut stroke runs at side_tape_continue_speed_factor of the contact scalings (own
    // planner instance, so the withdraw keeps the normal contact speed).
    const CutPlanners slow_planners = makeCutPlanners(settings_.motion_profile);
    slow_planners.stroke->setMaxVelocityScalingFactor(
      settings_.velocity_scaling * settings_.side_tape_continue_speed_factor);
    slow_planners.stroke->setMaxAccelerationScalingFactor(
      settings_.acceleration_scaling * settings_.side_tape_continue_speed_factor);
    geometry_msgs::msg::PoseStamped stroke_target = current_pose;
    stroke_target.header.frame_id = settings_.planning_frame;
    stroke_target.pose.position = toPoint(stroke_end);
    addMoveTo(
      *task, "side-tape continue: cut stroke parallel to planned line [" +
      settings_.motion_profile + "]", slow_planners.stroke, stroke_target);
    geometry_msgs::msg::PoseStamped end_pose = stroke_target;
    std::vector<const mtc::Stage*> transit_stages;
    std::vector<geometry_msgs::msg::PoseStamped> published_targets{stroke_target};
    if (!addSideTapeRetractStages(
          // fresh=false: withdraw along the face normal at the cut END and stop there (no
          // transit back to the left approach pose); the operator moves on from there.
          *task, planners, snapshot, end_pose, withdraw_m, false, transit_stages,
          published_targets, message)) {
      return false;
    }
    try {
      task->init();
    } catch (const mtc::InitStageException& exception) {
      std::ostringstream details;
      details << exception;
      message = "continue_side_cut MTC stage initialization failed: " + details.str();
      return false;
    }
    const auto result = task->plan(static_cast<std::size_t>(settings_.max_solutions));
    if (result != moveit::core::MoveItErrorCode::SUCCESS || task->solutions().empty()) {
      task->explainFailure();
      message = "continue_side_cut planning failed (stroke + retract); nothing moved";
      return false;
    }
    if (!transit_stages.empty()) {
      const auto& solution = *task->solutions().front();
      moveit_task_constructor_msgs::msg::Solution solution_message;
      solution.toMsg(solution_message);
      std::string wrist_error;
      if (!validateFreeSpaceTransitWrist(
            solution, solution_message, transit_stages, wrist_error)) {
        message = "continue_side_cut refused: " + wrist_error;
        return false;
      }
    }
    publishTargetPoses(snapshot, published_targets);
    publishSolution(*task);

    if (execution_thread_.joinable()) execution_thread_.join();
    execution_in_progress_ = true;
    side_tape_handoff_info_.status = SideTapeHandoffStatus::kCutting;
    side_tape_handoff_info_.stamp = now();
    writeSideTapeHandoffOutput();
    execution_thread_ = std::thread(
      [this, task = std::move(task)]() mutable {
        bool succeeded = false;
        std::string failure_detail;
        try {
          const auto exec_result = task->execute(*task->solutions().front());
          succeeded = (exec_result == moveit::core::MoveItErrorCode::SUCCESS);
          if (succeeded) {
            RCLCPP_INFO(get_logger(), "Side-tape continue_side_cut execution completed");
          } else {
            failure_detail = "controller reported failure status " + std::to_string(exec_result.val);
            RCLCPP_ERROR(
              get_logger(), "Side-tape continue_side_cut execution failed: %s",
              failure_detail.c_str());
          }
        } catch (const std::exception& exception) {
          failure_detail = std::string("threw exception: ") + exception.what();
          RCLCPP_ERROR(get_logger(), "Side-tape continue_side_cut execution %s",
            failure_detail.c_str());
        }
        std::lock_guard<std::mutex> completion_lock(plan_mutex_);
        execution_in_progress_ = false;
        if (succeeded) {
          // Keep the finished record (status 'retracted', snapshot path/id) so
          // continue_top_cut can reload the UNSHIFTED capture; the shifted copy is dropped.
          // continue_top_cut (or the next plan_side_tape_handoff) clears the record.
          side_tape_handoff_info_.status = SideTapeHandoffStatus::kRetracted;
          side_tape_handoff_info_.stamp = now();
          writeSideTapeHandoffOutput();
          side_tape_handoff_snapshot_.reset();
        } else {
          side_tape_handoff_info_.status = SideTapeHandoffStatus::kCutFailed;
          side_tape_handoff_info_.stamp = now();
          writeSideTapeHandoffOutput();
          RCLCPP_ERROR(
            get_logger(),
            "continue_side_cut did NOT complete (%s); hand-off is 'cut_failed', detected_carton "
            "kept; call side_tape_retract to recover", failure_detail.c_str());
        }
      });
    message = "continue_side_cut dispatch accepted: stroke " + formatDouble(stroke_m * 1000.0, 1) +
      " mm parallel to the planned line, then retract (withdraw " +
      formatDouble(withdraw_m * 1000.0, 1) + " mm + return transit)" + measured.str() +
      "; monitor controller execution";
    return true;
  }
  // ---- Staged top (middle seam) cut: approach -> penetrate -> cut_and_lift (see CONTEXT.md).
  static const char* topCutStatusName(TopCutStagedStatus status)
  {
    switch (status) {
      case TopCutStagedStatus::kNone: return "none";
      case TopCutStagedStatus::kPlanned: return "planned";
      case TopCutStagedStatus::kApproached: return "approached";
      case TopCutStagedStatus::kPenetrated: return "penetrated";
      case TopCutStagedStatus::kDone: return "done";
      case TopCutStagedStatus::kFailed: return "failed";
      case TopCutStagedStatus::kCutting: return "cutting";
    }
    return "unknown";
  }
  // approached/penetrated/failed: the arm is at or in the carton; the carton collision object
  // and the record must survive clear/replan until top_cut_retract.
  bool topCutStagedInProgress() const
  {
    return top_cut_info_.status == TopCutStagedStatus::kApproached ||
      top_cut_info_.status == TopCutStagedStatus::kPenetrated ||
      top_cut_info_.status == TopCutStagedStatus::kFailed ||
      top_cut_info_.status == TopCutStagedStatus::kCutting;
  }
  void writeTopCutStagedOutput()
  {
    const auto& info = top_cut_info_;
    const auto pose_json = [](const geometry_msgs::msg::PoseStamped& pose) {
      std::ostringstream out;
      out << "{\"position_m\": {\"x\": " << formatDouble(pose.pose.position.x) << ", \"y\": " <<
        formatDouble(pose.pose.position.y) << ", \"z\": " << formatDouble(pose.pose.position.z) <<
        "}, \"orientation_xyzw\": {\"x\": " << formatDouble(pose.pose.orientation.x) <<
        ", \"y\": " << formatDouble(pose.pose.orientation.y) << ", \"z\": " <<
        formatDouble(pose.pose.orientation.z) << ", \"w\": " <<
        formatDouble(pose.pose.orientation.w) << "}}";
      return out.str();
    };
    std::ostringstream json;
    json << "{\n  \"schema\": \"box_h_cut_top_cut_staged/v1\",\n";
    json << "  \"status\": \"" << topCutStatusName(info.status) << "\",\n";
    json << "  \"planning_frame\": \"" << jsonEscape(settings_.planning_frame) << "\",\n";
    json << "  \"tool_link\": \"" << jsonEscape(settings_.tool_link) << "\",\n";
    json << "  \"planned_hover\": " << pose_json(info.hover_pose) << ",\n";
    json << "  \"planned_cut_start\": " << pose_json(info.cut_start_pose) << ",\n";
    json << "  \"planned_cut_end\": " << pose_json(info.cut_end_pose) << ",\n";
    json << "  \"top_surface_point_m\": " << vector3Json(info.plane_point) << ",\n";
    json << "  \"surface_normal\": " << vector3Json(info.normal) << ",\n";
    json << "  \"hover_clearance_m\": " << formatDouble(settings_.hover_clearance_m) << ",\n";
    json << "  \"middle_cut_clearance_m\": " << formatDouble(settings_.middle_cut_clearance_m) <<
      ",\n";
    json << "  \"snapshot_path\": \"" << jsonEscape(info.snapshot_path) << "\",\n";
    json << "  \"snapshot_id\": \"" << jsonEscape(info.snapshot_id) << "\",\n";
    json << "  \"stamp_ns\": " << info.stamp.nanoseconds() << "\n}\n";
    const std::string payload = json.str();
    const std::string output_path = settings_.top_cut_staged_output_file;
    if (!output_path.empty()) {
      std::error_code code;
      std::filesystem::create_directories(std::filesystem::path(output_path).parent_path(), code);
      const std::string temporary = output_path + ".tmp";
      bool wrote = false;
      {
        std::ofstream file(temporary);
        if (file) {
          file << payload;
          file.flush();
          wrote = static_cast<bool>(file);
        }
      }
      if (wrote) {
        std::filesystem::rename(temporary, output_path, code);
        if (code) {
          RCLCPP_ERROR(
            get_logger(), "Cannot persist top-cut staged record: %s", code.message().c_str());
        }
      } else {
        RCLCPP_ERROR(get_logger(), "Cannot write top-cut staged record: %s", temporary.c_str());
      }
    }
    std_msgs::msg::String topic_message;
    topic_message.data = payload;
    top_cut_pub_->publish(topic_message);
    RCLCPP_INFO(get_logger(), "Top-cut staged status=%s", topCutStatusName(info.status));
  }
  static double angleBetweenDeg(
    const geometry_msgs::msg::Quaternion& a, const geometry_msgs::msg::Quaternion& b)
  {
    return toEigen(a).normalized().angularDistance(toEigen(b).normalized()) * 180.0 /
      3.14159265358979323846;
  }
  // Common refusals for the three post-approach services (never moves anything).
  bool topCutCommonGates(const char* who, bool check_stale_setting, std::string& message)
  {
    if (top_cut_info_.stale_override || (check_stale_setting && settings_.debug_allow_stale_snapshot)) {
      message = std::string(who) + " refused: planned/loaded under debug_allow_stale_snapshot "
        "(plan-only override); recover manually (teleop)";
      return false;
    }
    if (!execution_enabled_.load()) {
      message = "execution service is disabled by execution_enabled";
      return false;
    }
    if (settings_.debug_fixed_orientation) {
      message = "execution is blocked while debug_fixed_orientation is enabled";
      return false;
    }
    if (execution_in_progress_) {
      message = "an H-cut trajectory execution is already active";
      return false;
    }
    if (!top_cut_snapshot_) {
      message = "top-cut staged snapshot is unavailable; internal state error";
      return false;
    }
    std::string servo_error;
    if (!sideTapeServoIsQuiescent(servo_error)) {
      message = std::string(who) + " refused: " + servo_error;
      return false;
    }
    return true;
  }
  // Builds (CurrentState, carton + scoped contact allowance, camera pole, caller stages,
  // contact restore), plans and wrist-checks nothing (no free-space transit is used).
  std::unique_ptr<mtc::Task> buildTopCutTask(
    const std::string& name, const Snapshot& snapshot,
    const std::function<void(mtc::Task&, const CutPlanners&)>& add_stages, std::string& message)
  {
    auto task = std::make_unique<mtc::Task>();
    task->stages()->setName(name);
    if (robot_model_) {
      task->setRobotModel(robot_model_);
    } else {
      task->loadRobotModel(shared_from_this());
      robot_model_ = task->getRobotModel();
    }
    task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));
    addSideTapeCartonScene(*task, snapshot);
    const CutPlanners planners = makeCutPlanners(settings_.motion_profile);
    add_stages(*task, planners);
    if (settings_.allow_tool_carton_contact) {
      addCartonContactScope(*task, "restore knife-tip collision checking after " + name, false);
    }
    try {
      task->init();
    } catch (const mtc::InitStageException& exception) {
      std::ostringstream details;
      details << exception;
      message = name + " MTC stage initialization failed: " + details.str();
      return nullptr;
    }
    const auto result = task->plan(static_cast<std::size_t>(settings_.max_solutions));
    if (result != moveit::core::MoveItErrorCode::SUCCESS || task->solutions().empty()) {
      task->explainFailure();
      message = name + " planning failed; nothing moved";
      return nullptr;
    }
    logChosenLiftTilt(*task->solutions().front(), name);
    publishSolution(*task);
    return task;
  }
  // Executes `task` on the execution thread; on_done(succeeded) runs under plan_mutex_ with
  // execution_in_progress_ already cleared.
  void dispatchTopCutTask(
    std::unique_ptr<mtc::Task> task, const std::string& label,
    std::function<void(bool)> on_done)
  {
    if (execution_thread_.joinable()) execution_thread_.join();
    execution_in_progress_ = true;
    execution_thread_ = std::thread(
      [this, task = std::move(task), label, on_done = std::move(on_done)]() mutable {
        bool succeeded = false;
        try {
          const auto result = task->execute(*task->solutions().front());
          succeeded = (result == moveit::core::MoveItErrorCode::SUCCESS);
          if (!succeeded) {
            RCLCPP_ERROR(
              get_logger(), "%s execution failed: controller status %d", label.c_str(),
              result.val);
          }
        } catch (const std::exception& exception) {
          RCLCPP_ERROR(get_logger(), "%s execution threw: %s", label.c_str(), exception.what());
        }
        std::lock_guard<std::mutex> completion_lock(plan_mutex_);
        execution_in_progress_ = false;
        on_done(succeeded);
      });
  }
  // Operator-taught top-cut correction (2026-10-01), cut-local: [along seam start->end, across
  // (normal x along), depth (+ = deeper)] mm for the start and end. along/across shift the
  // snapshot's seam points (the hover follows); depth goes into `options` and is added to the
  // cut targets only. Shared by plan_top_cut_staged and continue_top_cut so both compute
  // identical poses.
  bool applyTopCutTaughtOffsets(
    Snapshot& snapshot, SegmentPlanOptions& options, std::string& message)
  {
    const auto& so = settings_.top_cut_start_offset_mm;
    const auto& eo = settings_.top_cut_end_offset_mm;
    for (const auto* o : {&so, &eo}) {
      if (o->size() != 3U || !std::isfinite((*o)[0]) || !std::isfinite((*o)[1]) ||
          !std::isfinite((*o)[2]) || std::abs((*o)[0]) > 30.0 || std::abs((*o)[1]) > 30.0 ||
          (*o)[2] < -5.0 || (*o)[2] > 25.0) {
        message = "top_cut_start/end_offset_mm must be [along, across, depth] with |along|,"
          "|across| <= 30 mm and depth in [-5, 25] mm";
        return false;
      }
    }
    const Eigen::Vector3d top_normal = toEigen(snapshot.surface_normal).normalized();
    Eigen::Vector3d along = toEigen(snapshot.seam_end) - toEigen(snapshot.seam_start);
    along -= along.dot(top_normal) * top_normal;
    along.normalize();
    const Eigen::Vector3d across = top_normal.cross(along).normalized();
    snapshot.seam_start = toPoint(
      toEigen(snapshot.seam_start) + (so[0] * along + so[1] * across) / 1000.0);
    snapshot.seam_end = toPoint(
      toEigen(snapshot.seam_end) + (eo[0] * along + eo[1] * across) / 1000.0);
    options.middle_start_extra_depth_m = so[2] / 1000.0;
    options.middle_end_extra_depth_m = eo[2] / 1000.0;
    if (so[0] != 0.0 || so[1] != 0.0 || so[2] != 0.0 || eo[0] != 0.0 || eo[1] != 0.0 ||
        eo[2] != 0.0) {
      RCLCPP_WARN(
        get_logger(), "top cut: taught offsets applied: start [%.1f, %.1f, %.1f] end "
        "[%.1f, %.1f, %.1f] mm (along, across, depth)", so[0], so[1], so[2], eo[0], eo[1], eo[2]);
    }
    return true;
  }
  // Point on the top surface under the (depth-offset) cut start target.
  Eigen::Vector3d topCutPlanePoint(
    const geometry_msgs::msg::PoseStamped& cut_start_target, const Eigen::Vector3d& normal,
    const SegmentPlanOptions& options)
  {
    return toEigen(cut_start_target.pose.position) +
      (options.middle_start_extra_depth_m - settings_.middle_cut_clearance_m) * normal;
  }
  // ~/continue_top_cut: ONE call for the whole top (middle seam) cut from the CURRENT robot
  // state: free-space transit to the taught-offset hover, stabilization dwell, scoped
  // tool/carton contact allowance, Cartesian descend to the offset cut start, cut stroke to the
  // offset cut end (contact scalings x top_cut_staged_speed_factor), Cartesian lift straight up
  // along the top normal to hover_clearance above the top, contact restore, STOP. The poses and
  // stages come from planCutSegments with the same inputs as plan_top_cut_staged.
  // Snapshot: after a finished side-tape hand-off (status retracted) the UNSHIFTED snapshot of
  // the same capture is reloaded from the recorded file (snapshot_id must match; the box is
  // static so no age check); without a hand-off the external snapshot must be fresh.
  bool continueTopCut(std::string& message)
  {
    struct SkipAgeGuard {
      bool& flag;
      ~SkipAgeGuard() { flag = false; }
    } skip_age_guard{skip_snapshot_age_};
    skip_snapshot_age_ = !settings_.top_cut_continue_check_snapshot_age;
    if (skip_snapshot_age_) {
      RCLCPP_WARN(get_logger(), "continue_top_cut: snapshot age not checked (box assumed static)");
    }
    if (settings_.debug_allow_stale_snapshot) {
      message = "continue_top_cut refused: debug_allow_stale_snapshot is plan-only; it can never "
        "be executed";
      return false;
    }
    if (!execution_enabled_.load()) {
      message = "execution service is disabled by execution_enabled";
      return false;
    }
    if (settings_.debug_fixed_orientation) {
      message = "execution is blocked while debug_fixed_orientation is enabled";
      return false;
    }
    if (execution_in_progress_) {
      message = "an H-cut trajectory execution is already active";
      return false;
    }
    if (side_tape_handoff_info_.inProgress()) {
      message = "continue_top_cut refused: a side-tape hand-off is active or unresolved; finish "
        "it with continue_side_cut or recover with side_tape_retract";
      return false;
    }
    if (cut_debug_stage_ != CutDebugStage::kNone) {
      message = "continue_top_cut refused: cut debug session active; call clear_plan first";
      return false;
    }
    if (top_cut_info_.status == TopCutStagedStatus::kPlanned || topCutStagedInProgress()) {
      message = "continue_top_cut refused: a staged top cut is active (" +
        std::string(topCutStatusName(top_cut_info_.status)) +
        "); finish it, or call top_cut_retract / clear_plan";
      return false;
    }
    if (!isSupportedMotionProfile(settings_.motion_profile)) {
      message = "unsupported H-cut motion_profile: " + settings_.motion_profile;
      return false;
    }
    std::string servo_error;
    if (!sideTapeServoIsQuiescent(servo_error)) {
      message = "continue_top_cut refused: " + servo_error;
      return false;
    }
    try {
      std::string source_path;
      Snapshot snapshot;
      std::string snapshot_origin;
      if (side_tape_handoff_info_.status == SideTapeHandoffStatus::kRetracted &&
          !side_tape_handoff_info_.snapshot_path.empty()) {
        if (side_tape_handoff_info_.stale_override) {
          message = "continue_top_cut refused: the side-tape hand-off was planned under "
            "debug_allow_stale_snapshot";
          return false;
        }
        source_path = side_tape_handoff_info_.snapshot_path;
        const auto loaded = loadExternalSnapshot(source_path, message);
        if (!loaded) return false;
        snapshot = *loaded;
        if (snapshot.snapshot_id != side_tape_handoff_info_.snapshot_id) {
          message = "continue_top_cut refused: snapshot file " + source_path + " now has "
            "snapshot_id '" + snapshot.snapshot_id + "' but the side-tape hand-off used '" +
            side_tape_handoff_info_.snapshot_id + "'; the capture changed";
          return false;
        }
        snapshot_origin = "unshifted hand-off capture (age not checked, box assumed static)";
      } else {
        source_path = externalSnapshotFile();
        const auto loaded = loadExternalSnapshot(source_path, message);
        if (!loaded) return false;
        snapshot = *loaded;
        std::string freshness_error;
        if (settings_.top_cut_continue_check_snapshot_age &&
            !snapshotIsFresh(snapshot, freshness_error)) {
          message = "continue_top_cut refused: " + freshness_error;
          return false;
        }
        snapshot_origin = "fresh external snapshot";
      }
      std::string validation;
      if (!validateSnapshot(snapshot, validation)) {
        message = "snapshot rejected: " + validation;
        return false;
      }
      clearRetainedPlan();
      top_cut_info_ = TopCutStagedInfo{};
      top_cut_snapshot_.reset();
      SegmentPlanOptions options;
      if (!applyTopCutTaughtOffsets(snapshot, options, message)) return false;
      options.cut_stroke_speed_factor = settings_.top_cut_staged_speed_factor;
      options.middle_finish_clearance_m = settings_.top_cut_lift_clearance_m;
      options.pre_transit_lift_m = settings_.top_cut_pre_transit_lift_m;
      geometry_msgs::msg::PoseStamped hover_pose;
      options.hover_start_out = &hover_pose;
      publishCartonGeometry(snapshot);
      const auto corrections = loadAllActiveCorrections();
      const CutSegment segment = segmentFor(snapshot, CutKind::kMiddle);
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> targets;
      std::string plan_error;
      // Operator 2026-10-01 (like t_cut_handoff): if the full top cut cannot be planned
      // (e.g. the wrist runs out of range near the seam end), retry with the cut END pulled
      // back along the seam in 5 % steps down to top_cut_min_cut_fraction, then lift there.
      // Every stroke is still planned complete (min_fraction 1.0) to its shortened end; a
      // partial Cartesian path is never accepted.
      const Snapshot full_snapshot = snapshot;
      const SegmentPlanOptions full_options = options;
      double cut_fraction = 1.0;
      bool planned = false;
      std::string first_error;
      for (int step = 0; ; ++step) {
        cut_fraction = 1.0 - 0.05 * step;
        if (cut_fraction < settings_.top_cut_min_cut_fraction - 1e-9) break;
        if (step > 0 && !settings_.top_cut_allow_shortened_cut) break;
        snapshot = full_snapshot;
        options = full_options;
        snapshot.seam_end = toPoint(
          toEigen(full_snapshot.seam_start) +
          cut_fraction * (toEigen(full_snapshot.seam_end) - toEigen(full_snapshot.seam_start)));
        options.middle_end_extra_depth_m = full_options.middle_start_extra_depth_m +
          cut_fraction * (full_options.middle_end_extra_depth_m -
          full_options.middle_start_extra_depth_m);
        const CutSegment try_segment = segmentFor(snapshot, CutKind::kMiddle);
        task.reset();
        targets.clear();
        plan_error.clear();
        if (planCutSegments(
              snapshot, {try_segment}, corrections, settings_.motion_profile,
              "continue top cut (middle: transit, hover, descend, cut, lift)", options, task,
              targets, plan_error)) {
          planned = true;
          break;
        }
        if (first_error.empty()) first_error = plan_error;
        RCLCPP_WARN(
          get_logger(), "continue_top_cut: %.0f %% cut did not plan (%s); trying shorter",
          cut_fraction * 100.0, plan_error.c_str());
      }
      if (!planned) {
        message = "continue_top_cut planning failed; nothing moved (tried down to " +
          formatDouble(settings_.top_cut_min_cut_fraction * 100.0, 0) + " % of the cut): " +
          first_error;
        return false;
      }
      if (cut_fraction < 1.0) {
        RCLCPP_WARN(
          get_logger(), "continue_top_cut: SHORTENED cut, %.0f %% of the seam (the rest is not "
          "cut)", cut_fraction * 100.0);
      }
      (void)segment;
      if (targets.size() != 2U) {
        message = "continue_top_cut: unexpected target count for the middle segment";
        return false;
      }
      top_cut_snapshot_ = snapshot;
      top_cut_info_.status = TopCutStagedStatus::kCutting;
      top_cut_info_.hover_pose = hover_pose;
      top_cut_info_.cut_start_pose = targets[0];
      top_cut_info_.cut_end_pose = targets[1];
      top_cut_info_.normal = toEigen(snapshot.surface_normal).normalized();
      top_cut_info_.plane_point = topCutPlanePoint(targets[0], top_cut_info_.normal, options);
      top_cut_info_.snapshot_path = source_path;
      top_cut_info_.snapshot_id = snapshot.snapshot_id;
      top_cut_info_.stamp = now();
      top_cut_info_.stale_override = false;
      const double end_depth_m = -top_cut_info_.normal.dot(
        toEigen(targets[1].pose.position) - top_cut_info_.plane_point);
      const double lift_m = std::max(0.0, end_depth_m) + settings_.top_cut_lift_clearance_m;
      geometry_msgs::msg::PoseStamped lift_pose = targets[1];
      lift_pose.pose.position = toPoint(
        toEigen(targets[1].pose.position) + lift_m * top_cut_info_.normal);
      RCLCPP_INFO(
        get_logger(),
        "continue_top_cut targets (%s): hover [%.4f %.4f %.4f] cut_start [%.4f %.4f %.4f] "
        "cut_end [%.4f %.4f %.4f] lift [%.4f %.4f %.4f] (m, %s)", snapshot_origin.c_str(),
        hover_pose.pose.position.x, hover_pose.pose.position.y, hover_pose.pose.position.z,
        targets[0].pose.position.x, targets[0].pose.position.y, targets[0].pose.position.z,
        targets[1].pose.position.x, targets[1].pose.position.y, targets[1].pose.position.z,
        lift_pose.pose.position.x, lift_pose.pose.position.y, lift_pose.pose.position.z,
        settings_.planning_frame.c_str());
      publishTargetPoses(snapshot, {hover_pose, targets[0], targets[1], lift_pose});
      publishSolution(*task);
      writeTopCutStagedOutput();
      dispatchTopCutTask(std::move(task), "Top-cut continue", [this](bool ok) {
        top_cut_info_.status = ok ? TopCutStagedStatus::kDone : TopCutStagedStatus::kFailed;
        top_cut_info_.stamp = now();
        writeTopCutStagedOutput();
        if (ok) {
          // Top cut finished: a new cycle (plan_side_tape_handoff) may start.
          side_tape_handoff_info_ = SideTapeHandoffInfo{};
          side_tape_handoff_snapshot_.reset();
          RCLCPP_INFO(get_logger(), "continue_top_cut execution completed (status done)");
        } else {
          RCLCPP_ERROR(
            get_logger(), "continue_top_cut did NOT complete; status 'failed', detected_carton "
            "kept; call top_cut_retract to lift straight up from the current pose");
        }
      });
      message = "continue_top_cut dispatch accepted (" + snapshot_origin + ", " +
        formatDouble(cut_fraction * 100.0, 0) + " % of the seam): transit to hover, "
        "dwell, descend, cut " +
        formatDouble((toEigen(targets[1].pose.position) - toEigen(targets[0].pose.position)).norm() *
          1000.0, 1) + " mm, lift " + formatDouble(lift_m * 1000.0, 1) +
        " mm along the top normal, then stop; monitor controller execution";
      return true;
    } catch (const std::exception& exception) {
      message = std::string("continue_top_cut planning failed: ") + exception.what();
      return false;
    }
  }
  // ~/plan_top_cut_staged: plans the FULL middle cut for feasibility (exactly as the normal
  // middle cut), then retains only the approach that ends at the aligned hover above the
  // seam start. Executed through execute_planned_h_cut and its gates.
  bool planTopCutStaged(std::string& message)
  {
    if (side_tape_handoff_info_.inProgress()) {
      message = "a side-tape hand-off is active or unresolved; call side_tape_retract first";
      return false;
    }
    if (cut_debug_stage_ != CutDebugStage::kNone) {
      message = "cut debug session active; call clear_plan before planning a staged top cut";
      return false;
    }
    if (topCutStagedInProgress()) {
      message = "a staged top cut is active (" + std::string(topCutStatusName(top_cut_info_.status)) +
        "); call top_cut_retract first";
      return false;
    }
    clearRetainedPlan();
    top_cut_info_ = TopCutStagedInfo{};
    top_cut_snapshot_.reset();
    try {
      const auto source_path = externalSnapshotFile();
      const auto loaded = loadExternalSnapshot(source_path, message);
      if (!loaded) return false;
      Snapshot snapshot = *loaded;
      std::string validation;
      if (!validateSnapshot(snapshot, validation)) {
        message = "snapshot rejected: " + validation;
        return false;
      }
      SegmentPlanOptions full_options;
      if (!applyTopCutTaughtOffsets(snapshot, full_options, message)) return false;
      full_options.middle_finish_clearance_m = settings_.top_cut_lift_clearance_m;
      if (!isSupportedMotionProfile(settings_.motion_profile)) {
        message = "unsupported H-cut motion_profile: " + settings_.motion_profile;
        return false;
      }
      publishCartonGeometry(snapshot);
      const auto corrections = loadAllActiveCorrections();
      const CutSegment segment = segmentFor(snapshot, CutKind::kMiddle);
      std::unique_ptr<mtc::Task> full_task;
      std::vector<geometry_msgs::msg::PoseStamped> full_targets;
      std::string plan_error;
      if (!planCutSegments(
            snapshot, {segment}, corrections, settings_.motion_profile,
            "staged top cut feasibility (middle, full stroke)", full_options, full_task,
            full_targets, plan_error)) {
        message = "staged top cut feasibility check failed for the full middle cut: " + plan_error;
        return false;
      }
      if (full_targets.size() != 2U) {
        message = "staged top cut: unexpected target count for the full middle segment";
        return false;
      }
      SegmentPlanOptions approach_options = full_options;
      approach_options.stop_at_hover_start = true;
      geometry_msgs::msg::PoseStamped hover_pose;
      approach_options.hover_start_out = &hover_pose;
      std::unique_ptr<mtc::Task> approach_task;
      std::vector<geometry_msgs::msg::PoseStamped> approach_targets;
      if (!planCutSegments(
            snapshot, {segment}, corrections, settings_.motion_profile,
            "staged top cut approach (middle, stops at aligned hover)", approach_options,
            approach_task, approach_targets, plan_error)) {
        message = "staged top cut approach planning failed: " + plan_error;
        return false;
      }
      if (approach_targets.size() != 2U ||
          (toEigen(approach_targets[0].pose.position) - toEigen(full_targets[0].pose.position))
          .norm() > 1e-6) {
        message = "staged top cut: approach and full plans disagree on the cut start; refusing";
        return false;
      }
      auto marker_segment = segment;
      marker_segment.stop_after_start = true;
      retainPlan(snapshot, {marker_segment}, {hover_pose}, std::move(approach_task));
      top_cut_snapshot_ = snapshot;
      top_cut_info_ = TopCutStagedInfo{};
      top_cut_info_.status = TopCutStagedStatus::kPlanned;
      top_cut_info_.hover_pose = hover_pose;
      top_cut_info_.cut_start_pose = full_targets[0];
      top_cut_info_.cut_end_pose = full_targets[1];
      top_cut_info_.normal = toEigen(snapshot.surface_normal).normalized();
      top_cut_info_.plane_point = topCutPlanePoint(full_targets[0], top_cut_info_.normal, full_options);
      top_cut_info_.snapshot_path = source_path;
      top_cut_info_.snapshot_id = snapshot.snapshot_id;
      top_cut_info_.stamp = now();
      top_cut_info_.stale_override = settings_.debug_allow_stale_snapshot;
      top_cut_pending_ = true;
      writeTopCutStagedOutput();
      const double cut_length_mm =
        (toEigen(full_targets[1].pose.position) - toEigen(full_targets[0].pose.position)).norm() *
        1000.0;
      message = "staged top cut planned: full middle cut feasible (cut_length=" +
        formatDouble(cut_length_mm, 1) + " mm, cut depth " +
        formatDouble(-settings_.middle_cut_clearance_m * 1000.0, 1) +
        " mm below the top); only the approach to the aligned hover (" +
        formatDouble(settings_.hover_clearance_m * 1000.0, 1) +
        " mm above the top) is retained; execute via execute_planned_h_cut, then "
        "top_cut_penetrate, then top_cut_cut_and_lift (top_cut_retract aborts)";
      return true;
    } catch (const std::exception& exception) {
      message = std::string("staged top cut planning failed: ") + exception.what();
      return false;
    }
  }
  bool topCutPenetrate(std::string& message)
  {
    if (top_cut_info_.status != TopCutStagedStatus::kApproached) {
      message = "top_cut_penetrate refused: status is '" +
        std::string(topCutStatusName(top_cut_info_.status)) + "', must be 'approached'";
      return false;
    }
    if (!topCutCommonGates("top_cut_penetrate", true, message)) return false;
    geometry_msgs::msg::PoseStamped current;
    if (!captureObservedTcpPose(current, message)) return false;
    const auto& info = top_cut_info_;
    const double position_mm =
      (toEigen(current.pose.position) - toEigen(info.hover_pose.pose.position)).norm() * 1000.0;
    const double orientation_deg =
      angleBetweenDeg(current.pose.orientation, info.hover_pose.pose.orientation);
    if (position_mm > settings_.top_cut_staged_max_position_mm ||
        orientation_deg > settings_.top_cut_staged_max_orientation_deg) {
      message = "top_cut_penetrate refused: tip is " + formatDouble(position_mm, 1) + " mm / " +
        formatDouble(orientation_deg, 1) + " deg from the recorded hover (max " +
        formatDouble(settings_.top_cut_staged_max_position_mm, 1) + " mm / " +
        formatDouble(settings_.top_cut_staged_max_orientation_deg, 1) + " deg)";
      return false;
    }
    const Snapshot snapshot = *top_cut_snapshot_;
    geometry_msgs::msg::PoseStamped target = current;
    target.header.frame_id = settings_.planning_frame;
    target.pose.position = info.cut_start_pose.pose.position;
    const double move_mm =
      (toEigen(target.pose.position) - toEigen(current.pose.position)).norm() * 1000.0;
    auto task = buildTopCutTask(
      "staged top cut penetrate", snapshot,
      [&](mtc::Task& t, const CutPlanners& planners) {
        addMoveTo(
          t, "top cut: penetrate to planned cut start [" + settings_.motion_profile + "]",
          planners.stroke, target);
      },
      message);
    if (!task) return false;
    publishTargetPoses(snapshot, {target});
    dispatchTopCutTask(std::move(task), "Top-cut penetrate", [this](bool ok) {
      top_cut_info_.status = ok ? TopCutStagedStatus::kPenetrated : TopCutStagedStatus::kFailed;
      top_cut_info_.stamp = now();
      writeTopCutStagedOutput();
      if (!ok) {
        RCLCPP_ERROR(get_logger(), "top_cut_penetrate did NOT complete; status 'failed'; call top_cut_retract");
      }
    });
    message = "top_cut_penetrate dispatch accepted: straight move " + formatDouble(move_mm, 1) +
      " mm to the planned cut start (hover offset " + formatDouble(position_mm, 1) + " mm / " +
      formatDouble(orientation_deg, 1) + " deg); monitor controller execution";
    return true;
  }
  bool topCutCutAndLift(std::string& message)
  {
    constexpr double kMinStrokeM = 0.005;
    if (top_cut_info_.status != TopCutStagedStatus::kPenetrated) {
      message = "top_cut_cut_and_lift refused: status is '" +
        std::string(topCutStatusName(top_cut_info_.status)) + "', must be 'penetrated'";
      return false;
    }
    if (!topCutCommonGates("top_cut_cut_and_lift", true, message)) return false;
    geometry_msgs::msg::PoseStamped current;
    if (!captureObservedTcpPose(current, message)) return false;
    const auto& info = top_cut_info_;
    const Eigen::Vector3d p0 = toEigen(current.pose.position);
    const Eigen::Vector3d planned_start = toEigen(info.cut_start_pose.pose.position);
    const Eigen::Vector3d planned_end = toEigen(info.cut_end_pose.pose.position);
    const double position_mm = (p0 - planned_start).norm() * 1000.0;
    const double orientation_deg =
      angleBetweenDeg(current.pose.orientation, info.hover_pose.pose.orientation);
    if (position_mm > settings_.top_cut_staged_max_position_mm ||
        orientation_deg > settings_.top_cut_staged_max_orientation_deg) {
      message = "top_cut_cut_and_lift refused: tip is " + formatDouble(position_mm, 1) + " mm / " +
        formatDouble(orientation_deg, 1) + " deg from the planned cut start (max " +
        formatDouble(settings_.top_cut_staged_max_position_mm, 1) + " mm / " +
        formatDouble(settings_.top_cut_staged_max_orientation_deg, 1) + " deg)";
      return false;
    }
    const Eigen::Vector3d line = planned_end - planned_start;
    if (line.norm() < 1e-3) {
      message = "top_cut_cut_and_lift refused: planned cut line is degenerate";
      return false;
    }
    const Eigen::Vector3d d = line.normalized();
    const double stroke_m = (planned_end - p0).dot(d);
    if (stroke_m < kMinStrokeM) {
      message = "top_cut_cut_and_lift refused: remaining stroke to the planned cut end is only " +
        formatDouble(stroke_m * 1000.0, 2) + " mm";
      return false;
    }
    const Eigen::Vector3d stroke_end = p0 + stroke_m * d;
    const double end_depth_m = -info.normal.dot(stroke_end - info.plane_point);
    const double lift_m = std::max(0.0, end_depth_m) + settings_.top_cut_lift_clearance_m;
    const Snapshot snapshot = *top_cut_snapshot_;
    geometry_msgs::msg::PoseStamped stroke_target = current;
    stroke_target.header.frame_id = settings_.planning_frame;
    stroke_target.pose.position = toPoint(stroke_end);
    geometry_msgs::msg::PoseStamped lift_target = stroke_target;
    lift_target.pose.position = toPoint(stroke_end + lift_m * info.normal);
    auto task = buildTopCutTask(
      "staged top cut stroke and lift", snapshot,
      [&](mtc::Task& t, const CutPlanners& planners) {
        const CutPlanners slow = makeCutPlanners(settings_.motion_profile);
        slow.stroke->setMaxVelocityScalingFactor(
          settings_.velocity_scaling * settings_.top_cut_staged_speed_factor);
        slow.stroke->setMaxAccelerationScalingFactor(
          settings_.acceleration_scaling * settings_.top_cut_staged_speed_factor);
        addMoveTo(
          t, "top cut: cut stroke to planned end plane [" + settings_.motion_profile + "]",
          slow.stroke, stroke_target);
        addTiltableLift(
          t, "top cut: lift along surface normal [" + settings_.motion_profile + "]",
          planners.stroke, lift_target, topCutLiftMinFraction(end_depth_m, lift_m));
      },
      message);
    if (!task) return false;
    publishTargetPoses(snapshot, {stroke_target, lift_target});
    dispatchTopCutTask(std::move(task), "Top-cut cut_and_lift", [this](bool ok) {
      top_cut_info_.status = ok ? TopCutStagedStatus::kDone : TopCutStagedStatus::kFailed;
      top_cut_info_.stamp = now();
      writeTopCutStagedOutput();
      if (!ok) {
        RCLCPP_ERROR(get_logger(), "top_cut_cut_and_lift did NOT complete; status 'failed'; call top_cut_retract");
      }
    });
    message = "top_cut_cut_and_lift dispatch accepted: stroke " + formatDouble(stroke_m * 1000.0, 1) +
      " mm parallel to the planned line, then lift " + formatDouble(lift_m * 1000.0, 1) +
      " mm along the surface normal and stop; monitor controller execution";
    return true;
  }
  bool topCutRetract(std::string& message)
  {
    if (top_cut_info_.status == TopCutStagedStatus::kNone ||
        top_cut_info_.status == TopCutStagedStatus::kPlanned) {
      message = "top_cut_retract refused: the staged top cut has not been approached";
      return false;
    }
    if (!topCutCommonGates("top_cut_retract", false, message)) return false;
    geometry_msgs::msg::PoseStamped current;
    if (!captureObservedTcpPose(current, message)) return false;
    const Eigen::Vector3d p0 = toEigen(current.pose.position);
    const double depth_m = -top_cut_info_.normal.dot(p0 - top_cut_info_.plane_point);
    const double lift_m = std::max(0.0, depth_m) + settings_.top_cut_lift_clearance_m;
    const Snapshot snapshot = *top_cut_snapshot_;
    geometry_msgs::msg::PoseStamped lift_target = current;
    lift_target.header.frame_id = settings_.planning_frame;
    lift_target.pose.position = toPoint(p0 + lift_m * top_cut_info_.normal);
    auto task = buildTopCutTask(
      "staged top cut retract", snapshot,
      [&](mtc::Task& t, const CutPlanners& planners) {
        addTiltableLift(
          t, "top cut: retract along surface normal [" + settings_.motion_profile + "]",
          planners.stroke, lift_target, topCutLiftMinFraction(depth_m, lift_m));
      },
      message);
    if (!task) return false;
    publishTargetPoses(snapshot, {lift_target});
    dispatchTopCutTask(std::move(task), "Top-cut retract", [this](bool ok) {
      if (ok) {
        top_cut_info_.status = TopCutStagedStatus::kNone;
        top_cut_info_.stamp = now();
        writeTopCutStagedOutput();
        top_cut_info_ = TopCutStagedInfo{};
        top_cut_snapshot_.reset();
      } else {
        RCLCPP_ERROR(get_logger(), "top_cut_retract did NOT complete; call top_cut_retract again once the arm is clear");
      }
    });
    message = "top_cut_retract dispatch accepted: lift " + formatDouble(lift_m * 1000.0, 1) +
      " mm along the surface normal (depth below top " + formatDouble(depth_m * 1000.0, 1) +
      " mm), then stop; monitor controller execution";
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
    // From here the start was dispatched, which consumed the retained plan; nothing can
    // plan while a session is active, so no plan is retained in these stages.
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
      cut_debug_stage_ = CutDebugStage::kStartSaved;
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
    if (cut_debug_stage_ == CutDebugStage::kStartSaved) {
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
    const auto source_path = externalSnapshotFile();
    const auto snapshot = loadExternalSnapshot(source_path, message);
    if (!snapshot) return false;
    std::string validation;
    const bool canonical = validateSnapshot(*snapshot, validation);
    const auto& motion_profile = settings_.motion_profile;
    const std::array<CutSegment, 3> sections{
      segmentFor(*snapshot, CutKind::kLeft), segmentFor(*snapshot, CutKind::kMiddle),
      segmentFor(*snapshot, CutKind::kRight)};
    std::ostringstream report;
    report << "diagnostic_only snapshot=" << snapshot->snapshot_id
           << " canonical=" << (canonical ? "true" : "false")
           << " validation=" << (canonical ? "ok" : validation);
    const SegmentPlanOptions options;
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

  // Opens `path`, checks file-level staleness (mtime vs external_snapshot_max_age_s, unless
  // debug_allow_stale_snapshot), and parses it as JSON into `document`. Shared by every
  // reader so the snapshot is read exactly once per request (no read/parse split across two
  // separate file opens, which could otherwise observe two different files).
  bool readSnapshotDocument(
    const std::string& path, boost::property_tree::ptree& document, std::string& message) const
  {
    namespace fs = std::filesystem;
    if (path.empty()) {
      message = "external_snapshot_file is required";
      return false;
    }
    const fs::path selected(path);
    if (!selected.is_absolute() || selected.extension() != ".json") {
      message = "external_snapshot_file must be an absolute .json path";
      return false;
    }
    std::error_code error;
    if (!fs::is_regular_file(selected, error) || error) {
      message = "external snapshot is unavailable: " + selected.string();
      return false;
    }
    const auto modified = fs::last_write_time(selected, error);
    if (error) {
      message = "cannot read external snapshot modification time: " + selected.string();
      return false;
    }
    const auto age_s = std::chrono::duration<double>(
      fs::file_time_type::clock::now() - modified).count();
    if (!settings_.debug_allow_stale_snapshot && !skip_snapshot_age_ &&
        age_s > settings_.external_snapshot_max_age_s) {
      message = "external snapshot is stale (" + formatDouble(age_s, 3) + " s; maximum " +
        formatDouble(settings_.external_snapshot_max_age_s, 3) + " s): " + selected.string();
      return false;
    }
    std::ifstream input(selected);
    if (!input) {
      message = "cannot read external snapshot: " + selected.string();
      return false;
    }
    try {
      boost::property_tree::read_json(input, document);
    } catch (const boost::property_tree::json_parser::json_parser_error& error) {
      message = "external snapshot is not valid JSON: " + std::string(error.what());
      return false;
    }
    return true;
  }
  // Parses the box_h_cut_snapshot/v1 fields from an already-loaded document.
  bool parseSnapshotFromDocument(
    const boost::property_tree::ptree& document, Snapshot& snapshot, std::string& message) const
  {
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
    if (!required_string("schema", schema) || schema != "box_h_cut_snapshot/v1" ||
        !required_string("snapshot_id", snapshot.snapshot_id) ||
        !required_string("capture_id", snapshot.capture_id) ||
        !required_string("frame_id", snapshot.header.frame_id) ||
        !required_string("source_frame", source_frame) ||
        !required_string("source_file", source_file)) {
      message = "external snapshot schema or required identity fields are invalid";
      return false;
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
      message = "external snapshot required geometry is invalid";
      return false;
    }
    // plane_rms_m is validated for schema integrity only; planning does not use it.
    // Optional (additive) support_depth_below_top_m: a present-but-non-numeric value is kept as
    // NaN so the height choice rejects it with a warning instead of silently ignoring it.
    snapshot.has_support_depth = false;
    if (document.get_child_optional("support_depth_below_top_m")) {
      snapshot.has_support_depth = true;
      snapshot.support_depth_below_top_m =
        document.get_optional<double>("support_depth_below_top_m")
          .value_or(std::numeric_limits<double>::quiet_NaN());
      snapshot.support_depth_source =
        document.get_optional<std::string>("support_depth_below_top_source").value_or("");
    }
    snapshot.acquisition_stamp_ns = *acquisition_stamp;
    snapshot.surface_normal.x = normal.x;
    snapshot.surface_normal.y = normal.y;
    snapshot.surface_normal.z = normal.z;
    if (!skip_snapshot_age_ && !acquisitionIsFresh(
          snapshot.acquisition_stamp_ns, settings_.debug_allow_stale_snapshot)) {
      message = "external snapshot acquisition is stale or invalid; maximum age " +
        formatDouble(settings_.external_snapshot_max_age_s, 3) + " s";
      return false;
    }
    snapshot.header.stamp.sec = static_cast<std::int32_t>(snapshot.acquisition_stamp_ns / 1'000'000'000LL);
    snapshot.header.stamp.nanosec = static_cast<std::uint32_t>(snapshot.acquisition_stamp_ns % 1'000'000'000LL);
    return true;
  }
  std::optional<Snapshot> loadExternalSnapshot(
    const std::string& external_snapshot_file, std::string& message) const
  {
    boost::property_tree::ptree document;
    if (!readSnapshotDocument(external_snapshot_file, document, message)) return std::nullopt;
    Snapshot snapshot;
    if (!parseSnapshotFromDocument(document, snapshot, message)) return std::nullopt;
    return snapshot;
  }

  // Parses side_tape_seed from an already-loaded document (the Snapshot struct does not carry
  // it). No key means no side cut: this fails closed rather than estimating geometry. Also
  // cross-checks the seed's own `source.snapshot_id`/`source.acquisition_stamp_ns` against the
  // snapshot parsed from the SAME document, so a seed block can never be silently paired with
  // geometry from a different capture (both were read from one file, one parse).
  bool parseSideTapeSeedFaceFromDocument(
    const boost::property_tree::ptree& document, const std::string& face,
    const Snapshot& snapshot, SideTapeSeedFace& out, std::string& message) const
  {
    using boost::property_tree::ptree;
    const auto seed = document.get_child_optional("side_tape_seed");
    if (!seed) {
      message =
        "external snapshot has no side_tape_seed block; run side_face_seed.py on the snapshot "
        "before enabling side_tape_handoff_enabled";
      return false;
    }
    const auto schema = seed->get_optional<std::string>("schema");
    if (!schema || (*schema != "side_tape_seed/v1" && *schema != "side_tape_seed/v1.1")) {
      message = "side_tape_seed schema mismatch: " + (schema ? *schema : std::string("<missing>"));
      return false;
    }
    const auto frame_id = seed->get_optional<std::string>("frame_id");
    const auto units = seed->get_optional<std::string>("units");
    if (!frame_id || *frame_id != "base_link" || !units || *units != "m") {
      message = "side_tape_seed must be in base_link metres";
      return false;
    }
    const auto source = seed->get_child_optional("source");
    boost::optional<std::string> source_snapshot_id;
    boost::optional<std::int64_t> source_stamp;
    if (source) {
      source_snapshot_id = source->get_optional<std::string>("snapshot_id");
      source_stamp = source->get_optional<std::int64_t>("acquisition_stamp_ns");
    }
    if (!source || !source_snapshot_id || !source_stamp) {
      message = "side_tape_seed.source is missing snapshot_id/acquisition_stamp_ns; cannot "
        "verify it matches the snapshot it is paired with";
      return false;
    }
    if (*source_snapshot_id != snapshot.snapshot_id ||
        *source_stamp != snapshot.acquisition_stamp_ns) {
      message = "side_tape_seed.source (snapshot_id=" + *source_snapshot_id +
        ", acquisition_stamp_ns=" + std::to_string(*source_stamp) +
        ") does not match this snapshot (snapshot_id=" + snapshot.snapshot_id +
        ", acquisition_stamp_ns=" + std::to_string(snapshot.acquisition_stamp_ns) +
        "); refusing a seed that may belong to a different capture";
      return false;
    }
    const auto faces = seed->get_child_optional("faces");
    if (!faces) {
      message = "side_tape_seed has no faces block";
      return false;
    }
    const auto face_node = faces->get_child_optional(face);
    if (!face_node) {
      message = "side_tape_seed has no face '" + face + "'";
      return false;
    }
    const auto point_of = [](const ptree& node, const std::string& key, Eigen::Vector3d& value) {
      const auto child = node.get_child_optional(key);
      if (!child) return false;
      const auto x = child->get_optional<double>("x");
      const auto y = child->get_optional<double>("y");
      const auto z = child->get_optional<double>("z");
      if (!x || !y || !z || !std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z)) {
        return false;
      }
      value = Eigen::Vector3d(*x, *y, *z);
      return true;
    };
    const auto plane = face_node->get_child_optional("plane");
    Eigen::Vector3d plane_point;
    Eigen::Vector3d plane_normal;
    if (!plane || !point_of(*plane, "point", plane_point) || !point_of(*plane, "normal", plane_normal)) {
      message = "side_tape_seed face '" + face + "' plane is invalid";
      return false;
    }
    const double normal_norm = plane_normal.norm();
    if (!std::isfinite(normal_norm) || normal_norm < 1e-6) {
      message = "side_tape_seed face '" + face + "' plane normal has zero length";
      return false;
    }
    plane_normal /= normal_norm;
    // wall_top_edge (outer wall, v1.1) when present, else top_edge (v1) at the top face plane.
    auto top_edge = face_node->get_child_optional("wall_top_edge");
    if (!top_edge) top_edge = face_node->get_child_optional("top_edge");
    Eigen::Vector3d edge_start;
    Eigen::Vector3d edge_end;
    if (!top_edge || !point_of(*top_edge, "start", edge_start) || !point_of(*top_edge, "end", edge_end)) {
      message = "side_tape_seed face '" + face + "' top edge is invalid";
      return false;
    }
    out.plane_point = plane_point;
    out.plane_normal = plane_normal;
    out.top_edge_start = edge_start;
    out.top_edge_end = edge_end;
    const auto wall_source = face_node->get_optional<std::string>("wall_source");
    out.wall_source = wall_source ? *wall_source : std::string();
    return true;
  }
  // Single read of the snapshot file (one ifstream, one JSON parse): both the
  // box_h_cut_snapshot/v1 geometry and its paired side_tape_seed face come from the exact
  // same bytes, and the seed's own source identity is cross-checked against it.
  bool loadSnapshotAndSeedFace(
    const std::string& path, const std::string& face, Snapshot& snapshot,
    SideTapeSeedFace& seed_face, std::string& message) const
  {
    boost::property_tree::ptree document;
    if (!readSnapshotDocument(path, document, message)) return false;
    if (!parseSnapshotFromDocument(document, snapshot, message)) return false;
    return parseSideTapeSeedFaceFromDocument(document, face, snapshot, seed_face, message);
  }

  // Tolerance for checkSideTapeOrientationConvention: side_tape's configured tool-frame
  // convention must hold at the hand-off start pose within this many degrees, or the
  // hand-off fails closed.
  static constexpr double kSideTapeHandoffMaxAxisAngleDeg = 5.0;
  struct SideTapeOrientationCheck {
    bool ok{false};
    std::string message;
    double cut_angle_deg{0.0};
    double depth_angle_deg{0.0};
    double lateral_angle_deg{0.0};
    // Configured tool-frame axes rotated into base_link at the checked pose (unit vectors).
    Eigen::Vector3d cut_dir_base{Eigen::Vector3d::UnitX()};
    Eigen::Vector3d lateral_dir_base{Eigen::Vector3d::UnitY()};
    Eigen::Vector3d depth_dir_base{Eigen::Vector3d::UnitZ()};
  };
  // Measures (never silently corrects) whether the tool_link orientation at a pose satisfies
  // the operator-configured side_tape tool-frame convention (`side_tape_handoff_*_axis_tool`):
  // cut_axis_tool along the cut direction, depth_axis_tool into the face, lateral_axis_tool
  // on-face perpendicular to the cut (lateral = depth x cut, validated at startup). Reports
  // every measured angle plus, for each semantic role, which of the tool's six signed
  // principal axes is the closest actual match and by how much, so a genuine convention
  // mismatch is diagnosable by axis and degrees rather than just "failed".
  SideTapeOrientationCheck checkSideTapeOrientationConvention(
    const geometry_msgs::msg::Quaternion& orientation, const Eigen::Vector3d& travel_dir_unit,
    const Eigen::Vector3d& inward_normal_unit, const Eigen::Vector3d& cut_axis_tool,
    const Eigen::Vector3d& lateral_axis_tool, const Eigen::Vector3d& depth_axis_tool,
    double max_axis_angle_deg) const
  {
    SideTapeOrientationCheck result;
    const Eigen::Quaterniond q = toEigen(orientation).normalized();
    const Eigen::Vector3d measured_cut = (q * cut_axis_tool).normalized();
    const Eigen::Vector3d measured_lateral = (q * lateral_axis_tool).normalized();
    const Eigen::Vector3d measured_depth = (q * depth_axis_tool).normalized();
    result.cut_dir_base = measured_cut;
    result.lateral_dir_base = measured_lateral;
    result.depth_dir_base = measured_depth;
    const Eigen::Vector3d lateral_expected =
      inward_normal_unit.cross(travel_dir_unit).normalized();
    const auto angle_deg = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
      const double c = std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0);
      return std::acos(c) * 180.0 / 3.14159265358979323846;
    };
    result.cut_angle_deg = angle_deg(measured_cut, travel_dir_unit);
    result.depth_angle_deg = angle_deg(measured_depth, inward_normal_unit);
    result.lateral_angle_deg = angle_deg(measured_lateral, lateral_expected);
    // Diagnostics only: which of the tool's raw +/-X/Y/Z axes best matches each semantic
    // role, independent of the configured mapping (helps spot a mis-configured mapping).
    const Eigen::Vector3d tool_x = (q * Eigen::Vector3d::UnitX()).normalized();
    const Eigen::Vector3d tool_y = (q * Eigen::Vector3d::UnitY()).normalized();
    const Eigen::Vector3d tool_z = (q * Eigen::Vector3d::UnitZ()).normalized();
    const std::array<std::pair<const char*, Eigen::Vector3d>, 6> axes{{
      {"+X", tool_x}, {"-X", Eigen::Vector3d(-tool_x)}, {"+Y", tool_y},
      {"-Y", Eigen::Vector3d(-tool_y)}, {"+Z", tool_z}, {"-Z", Eigen::Vector3d(-tool_z)}}};
    const auto best_match = [&](const Eigen::Vector3d& target) {
      std::size_t best_index = 0U;
      double best_angle = angle_deg(axes.front().second, target);
      for (std::size_t index = 1U; index < axes.size(); ++index) {
        const double candidate_angle = angle_deg(axes[index].second, target);
        if (candidate_angle < best_angle) {
          best_angle = candidate_angle;
          best_index = index;
        }
      }
      return std::make_pair(std::string(axes[best_index].first), best_angle);
    };
    const auto cut_best = best_match(travel_dir_unit);
    const auto depth_best = best_match(inward_normal_unit);
    const auto lateral_best = best_match(lateral_expected);
    std::ostringstream diagnostic;
    diagnostic << "side_tape axis check (tool_frame, tolerance " <<
      formatDouble(max_axis_angle_deg, 1) <<
      " deg): cut_axis_tool measured angle to travel direction " <<
      formatDouble(result.cut_angle_deg, 1) << " deg (closest raw tool axis " << cut_best.first <<
      " at " << formatDouble(cut_best.second, 1) <<
      " deg); depth_axis_tool measured angle to inward face normal " <<
      formatDouble(result.depth_angle_deg, 1) << " deg (closest raw tool axis " <<
      depth_best.first << " at " << formatDouble(depth_best.second, 1) <<
      " deg); lateral_axis_tool measured angle to depth x cut " <<
      formatDouble(result.lateral_angle_deg, 1) << " deg (closest raw tool axis " <<
      lateral_best.first << " at " << formatDouble(lateral_best.second, 1) << " deg)";
    result.message = diagnostic.str();
    result.ok = result.cut_angle_deg <= max_axis_angle_deg &&
      result.depth_angle_deg <= max_axis_angle_deg && result.lateral_angle_deg <= max_axis_angle_deg;
    return result;
  }

  void addCameraPole(mtc::Task& task)
  {
    std::optional<moveit_msgs::msg::CollisionObject> camera_pole;
    {
      std::lock_guard<std::mutex> lock(camera_pole_mutex_);
      camera_pole = camera_pole_;
    }
    if (settings_.camera_pole_required && !camera_pole) {
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

  mutable std::string last_carton_height_log_;
  moveit_msgs::msg::CollisionObject cartonCollisionObject(const Snapshot& snapshot) const
  {
    const Eigen::Vector3d normal = toEigen(snapshot.surface_normal).normalized();
    const Eigen::Vector3d back_left = toEigen(snapshot.back_left);
    const Eigen::Vector3d back_right = toEigen(snapshot.back_right);
    const Eigen::Vector3d front_left = toEigen(snapshot.front_left);
    const Eigen::Vector3d front_right = toEigen(snapshot.front_right);
    const Eigen::Vector3d top_center = (back_left + back_right + front_left + front_right) / 4.0;
    Eigen::Vector3d x_axis = toEigen(snapshot.seam_end) - toEigen(snapshot.seam_start);
    x_axis -= x_axis.dot(normal) * normal;
    x_axis.normalize();
    const Eigen::Vector3d y_axis = normal.cross(x_axis).normalized();
    const double width = ((back_right - back_left).norm() + (front_right - front_left).norm()) / 2.0;
    const double depth = ((front_left - back_left).norm() + (front_right - back_right).norm()) / 2.0;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    const auto choice = carton_height::choose(
      settings_.carton_height_from_snapshot, snapshot.support_depth_below_top_m,
      snapshot.has_support_depth, settings_.carton_height_m, settings_.carton_height_max_m);
    const double carton_height = choice.height_m;
    if (last_carton_height_log_ != snapshot.snapshot_id + "/" + choice.source) {
      last_carton_height_log_ = snapshot.snapshot_id + "/" + choice.source;
      if (choice.warn) {
        RCLCPP_WARN(
          get_logger(),
          "carton collision height %.3f m from %s (snapshot support_depth_below_top_m=%.4f, "
          "allowed %.3f..%.3f m)",
          carton_height, choice.source.c_str(), snapshot.support_depth_below_top_m,
          settings_.carton_height_m, settings_.carton_height_max_m);
      } else {
        RCLCPP_INFO(
          get_logger(), "carton collision height %.3f m from %s%s%s", carton_height,
          choice.source.c_str(), snapshot.support_depth_source.empty() ? "" : " via ",
          snapshot.support_depth_source.c_str());
      }
    }
    primitive.dimensions = {width, depth, carton_height};
    geometry_msgs::msg::Pose pose;
    pose.position = toPoint(top_center - normal * carton_height / 2.0);
    pose.orientation = toMsg(basisQuaternion(x_axis, y_axis, normal));
    moveit_msgs::msg::CollisionObject object;
    object.id = settings_.carton_collision_id;
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
    if (snapshot.snapshot_id.empty() || snapshot.header.frame_id != settings_.planning_frame) {
      message = "snapshot must have an ID and use planning frame " + settings_.planning_frame +
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

  static std::int64_t nowNs()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  }
  // Shared acquisition-age rule: a positive stamp, at most 1 s in the future, and (unless
  // the plan-only stale override applies) no older than external_snapshot_max_age_s.
  bool acquisitionIsFresh(std::int64_t stamp_ns, bool ignore_max_age) const
  {
    const auto age_ns = nowNs() - stamp_ns;
    const auto maximum_age_ns =
      static_cast<std::int64_t>(settings_.external_snapshot_max_age_s * 1e9);
    return stamp_ns > 0 && age_ns >= -1'000'000'000LL &&
      (ignore_max_age || age_ns <= maximum_age_ns);
  }
  bool snapshotIsFresh(const Snapshot& snapshot, std::string& message) const
  {
    if (!acquisitionIsFresh(snapshot.acquisition_stamp_ns, false)) {
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

  // Segments for every cut pattern except t_cut_handoff (see planTCutHandoffSnapshot).
  std::vector<CutSegment> segmentsForPattern(
    const Snapshot& snapshot, const std::string& pattern) const
  {
    const auto left = segmentFor(snapshot, CutKind::kLeft);
    const auto middle = segmentFor(snapshot, CutKind::kMiddle);
    const auto right = segmentFor(snapshot, CutKind::kRight);
    if (pattern == "h" || pattern == "combined") return {left, middle, right};
    if (pattern == "t_cut") return {left, middle};
    if (pattern == "left_only") return {left};
    if (pattern == "middle_only") return {middle};
    if (pattern == "right_only") return {right};
    if (pattern == "middle_entry_debug") {
      auto entry = middle;
      entry.name = "middle seam entry debug";
      entry.stop_after_start = true;
      return {entry};
    }
    if (pattern == "middle_debug_short") {
      const double seam_length = distanceBetween(snapshot.seam_start, snapshot.seam_end);
      if (seam_length < settings_.debug_motion_distance_m) {
        throw std::invalid_argument("middle seam is shorter than debug_motion_distance_m");
      }
      const double half_fraction = settings_.debug_motion_distance_m / (2.0 * seam_length);
      // No cut kind: never corrected, no middle entry alignment, fixed orientation.
      CutSegment short_move;
      short_move.name = "middle debug short move";
      short_move.start = interpolate(snapshot.seam_start, snapshot.seam_end, 0.5 - half_fraction);
      short_move.end = interpolate(snapshot.seam_start, snapshot.seam_end, 0.5 + half_fraction);
      short_move.hover_only = true;
      short_move.force_fixed_orientation = true;
      return {short_move};
    }
    throw std::invalid_argument("unsupported cut_pattern: " + pattern);
  }
  bool planSnapshot(const Snapshot& snapshot, std::string& message)
  {
    const std::string pattern = settings_.cut_pattern;
    try {
      std::string validation;
      if (!validateSnapshot(snapshot, validation)) {
        message = "snapshot rejected: " + validation;
        return false;
      }
      const auto& motion_profile = settings_.motion_profile;
      if (!isSupportedMotionProfile(motion_profile)) {
        throw std::invalid_argument("unsupported H-cut motion_profile: " + motion_profile);
      }
      publishCartonGeometry(snapshot);
      // Each configured correction record is loaded and validated once per request.
      const auto corrections = loadAllActiveCorrections();
      if (pattern == "t_cut_handoff") {
        return planTCutHandoffSnapshot(snapshot, corrections, motion_profile, message);
      }
      const bool waypoint_pattern = pattern == "combined" || pattern == "t_cut";
      const std::vector<CutSegment> segments = segmentsForPattern(snapshot, pattern);
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> cut_targets;
      SegmentPlanOptions options;
      options.skip_supplied_approach = pattern != "h";
      const std::string task_name = !waypoint_pattern ?
        "Snapshot-relative " + pattern + " dry run" :
        (pattern == "t_cut" ? "Snapshot-relative T-cut canonical waypoint plan" :
         "Snapshot-relative combined canonical waypoint plan");
      if (!planCutSegments(
            snapshot, segments, corrections, motion_profile, task_name, options, task,
            cut_targets, message)) {
        return false;
      }
      retainPlan(snapshot, segments, cut_targets, std::move(task));
      const std::string dwell_summary = "pre-cut stabilization_s=" +
        formatDouble(settings_.pre_cut_stabilization_duration_s, 3) + "; dwell_count=" +
        std::to_string(settings_.pre_cut_stabilization_duration_s > 0.0 ? segments.size() : 0U);
      if (waypoint_pattern) {
        const bool t_cut = pattern == "t_cut";
        message =
          "Snapshot-relative " + std::string(t_cut ? "T-cut" : "combined") +
          " dry-run plan [" + motion_profile + "] published to RViz; waypoint_order=approach"
          "->back_middle_left->front_middle_left->finish->transition->approach->seam_start"
          "->seam_end->finish" +
          (t_cut ? "" : "->transition->approach->back_middle_right->front_middle_right->finish") +
          "; side_knife_facing=opposite_travel; " + dwell_summary +
          "; execution remains disabled until execution_enabled is set";
        return true;
      }
      // Only segments of the middle seam get the entry-alignment stage.
      const bool includes_middle_seam =
        std::any_of(segments.begin(), segments.end(), [](const CutSegment& segment) {
          return segment.cut == CutKind::kMiddle;
        });
      message = "Snapshot-relative " + pattern + " dry-run plan [" + motion_profile +
                "] published to RViz; " + dwell_summary +
                (includes_middle_seam ? "; middle entry alignment_m=" +
                  formatDouble(settings_.middle_entry_alignment_m, 3) : "") +
                (pattern == "middle_entry_debug" ?
                  "; stops at seam_start before the cut traverse" : "") +
                (pattern == "middle_debug_short" ?
                  "; fixed orientation (x=0,y=1,z=0,w=0); short stroke_m=" +
                  formatDouble(settings_.debug_motion_distance_m, 3) + "; surface clearance_m=" +
                  formatDouble(settings_.debug_motion_clearance_m, 3) +
                  "; manual execution additionally requires debug_motion_execution_enabled=true" : "") +
                "; execution remains disabled until execution_enabled is set";
      RCLCPP_INFO(
        get_logger(),
        "Planned snapshot %s with profile %s and pattern %s; dwell count %zu at %.3f s; middle entry alignment %.1f mm",
        snapshot.snapshot_id.c_str(), motion_profile.c_str(), pattern.c_str(),
        segments.size(), settings_.pre_cut_stabilization_duration_s,
        includes_middle_seam ? settings_.middle_entry_alignment_m * 1000.0 : 0.0);
      return true;
    } catch (const std::exception& exception) {
      RCLCPP_ERROR(
        get_logger(), "H-cut %s planning initialization failed: %s",
        pattern.c_str(), exception.what());
      message = pattern + " planning failed: " + exception.what();
      return false;
    }
  }
  void publishSolution(mtc::Task& task)
  {
    task.introspection().publishSolution(*task.solutions().front());
  }
  // Publishes the plan's target markers/poses and solution, and retains it for explicit
  // execution. Only retained plans are published (never rejected handoff probes).
  void retainPlan(
    const Snapshot& snapshot, const std::vector<CutSegment>& segments,
    const std::vector<geometry_msgs::msg::PoseStamped>& cut_targets,
    std::unique_ptr<mtc::Task> task)
  {
    publishMarkers(cut_targets, segments);
    publishTargetPoses(snapshot, cut_targets);
    publishSolution(*task);
    planned_snapshot_ = snapshot;
    planned_task_ = std::move(task);
    retained_with_stale_override_ = settings_.debug_allow_stale_snapshot;
  }
  // Plans the first-arm T-cut handoff candidate: the full left cut plus the middle seam
  // prefix ending at `fraction`, under min_fraction=1.0 like every other candidate.
  bool planHandoffAt(
    const Snapshot& snapshot, const std::map<std::string, CutDebugCorrection>& corrections,
    const std::string& motion_profile, double fraction, const std::string& task_name,
    LeftRollCache& left_roll_cache, std::unique_ptr<mtc::Task>& task,
    std::vector<geometry_msgs::msg::PoseStamped>& targets,
    std::vector<CutSegment>& segments, std::string& failure)
  {
    CutSegment handoff_segment = segmentFor(snapshot, CutKind::kMiddle);
    handoff_segment.name = "middle seam handoff";
    handoff_segment.end = interpolate(snapshot.seam_start, snapshot.seam_end, fraction);
    handoff_segment.handoff_end = true;
    handoff_segment.terminal_after_cut = true;
    handoff_segment.handoff_fraction = fraction;
    segments = {segmentFor(snapshot, CutKind::kLeft), handoff_segment};
    SegmentPlanOptions options;
    options.left_roll_cache = &left_roll_cache;
    return planCutSegments(
      snapshot, segments, corrections, motion_profile, task_name, options, task, targets,
      failure);
  }
  bool planTCutHandoffSnapshot(
    const Snapshot& snapshot, const std::map<std::string, CutDebugCorrection>& corrections,
    const std::string& motion_profile, std::string& message)
  {
    const double fraction = settings_.t_cut_middle_handoff_fraction;
    if (!isValidHandoffFraction(fraction)) {
      message = kHandoffFractionRule;
      return false;
    }
    std::unique_ptr<mtc::Task> best_task;
    std::vector<geometry_msgs::msg::PoseStamped> best_targets;
    std::vector<CutSegment> best_segments;
    double best_fraction = fraction;
    constexpr double kAutoTolerance = 0.01;
    // The left segment is identical for every probe of this request: its roll search
    // runs once and the selected (or failed) result is reused.
    LeftRollCache left_roll_cache;
    if (fraction != 0.0) {
      if (!planHandoffAt(
            snapshot, corrections, motion_profile, fraction,
            "Snapshot-relative T-cut handoff canonical waypoint plan", left_roll_cache,
            best_task, best_targets, best_segments, message)) {
        return false;
      }
    } else {
      constexpr double kAutoMin = 0.05;
      constexpr double kAutoMax = 0.95;
      const auto candidate_name = [](double probe) {
        return "Snapshot-relative T-cut handoff automatic candidate " + formatDouble(probe, 2);
      };
      // Every probe is a full planCutSegments handoff candidate under min_fraction=1.0;
      // no partial path is accepted. The maximum is probed first: when the complete
      // declared prefix up to 0.95 plans, no search is needed.
      std::string probe_failure;
      if (planHandoffAt(
            snapshot, corrections, motion_profile, kAutoMax, candidate_name(kAutoMax),
            left_roll_cache, best_task, best_targets, best_segments, probe_failure)) {
        best_fraction = kAutoMax;
      } else {
        const std::string max_failure = probe_failure;
        if (!planHandoffAt(
              snapshot, corrections, motion_profile, kAutoMin, candidate_name(kAutoMin),
              left_roll_cache, best_task, best_targets, best_segments, probe_failure)) {
          message =
            "Automatic T-cut handoff search failed closed: no fully planned handoff candidate at minimum "
            "middle_handoff_fraction=0.05; search_tolerance=" + formatDouble(kAutoTolerance, 2) +
            "; last_failure=" + probe_failure;
          return false;
        }
        best_fraction = kAutoMin;
        // Bounded bisection over [0.05,0.95): ceil(log2((0.95-0.05)/0.01)) = 7 probes.
        // Feasibility is monotonic along the same middle seam direction, so a failed probe
        // narrows from above and a success narrows from below.
        constexpr int kAutoMaxProbes = 7;
        double low = kAutoMin;
        double high = kAutoMax;
        for (int probe = 0; probe < kAutoMaxProbes && (high - low) > kAutoTolerance; ++probe) {
          const double mid = 0.5 * (low + high);
          std::unique_ptr<mtc::Task> candidate_task;
          std::vector<geometry_msgs::msg::PoseStamped> candidate_targets;
          std::vector<CutSegment> candidate_segments;
          std::string candidate_failure;
          if (planHandoffAt(
                snapshot, corrections, motion_profile, mid, candidate_name(mid),
                left_roll_cache, candidate_task, candidate_targets, candidate_segments,
                candidate_failure)) {
            best_fraction = mid;
            best_task = std::move(candidate_task);
            best_targets = std::move(candidate_targets);
            best_segments = std::move(candidate_segments);
            low = mid;
          } else {
            high = mid;
          }
        }
        RCLCPP_INFO(
          get_logger(), "T-cut handoff: maximum fraction 0.95 did not plan (%s); selected %.4f",
          max_failure.c_str(), best_fraction);
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
      formatDouble(settings_.pre_cut_stabilization_duration_s, 3) + "; dwell_count=" +
      std::to_string(settings_.pre_cut_stabilization_duration_s > 0.0 ? best_segments.size() : 0U) +
      "; execution remains disabled until execution_enabled is set";
    return true;
  }

  // Planner stack for one task. Free-space transits run at their own scaling so transit
  // speed can be tuned without ever raising physical cutting speed, and are wrapped by
  // makeTransitPlanner. Contact strokes use Cartesian (min_fraction=1.0, never partial)
  // or Pilz LIN at the contact scaling. Every pipeline stage is bounded by
  // planning_timeout_s.
  CutPlanners makeCutPlanners(const std::string& motion_profile)
  {
    const auto pipeline = [this](
      const std::string& pipeline_name, const std::string& planner_id, bool transit) {
      auto planner = std::make_shared<mtc::solvers::PipelinePlanner>(
        shared_from_this(), pipeline_name);
      planner->setTimeout(settings_.planning_timeout_s);
      planner->setMaxVelocityScalingFactor(
        transit ? settings_.transit_velocity_scaling : settings_.velocity_scaling);
      planner->setMaxAccelerationScalingFactor(
        transit ? settings_.transit_acceleration_scaling : settings_.acceleration_scaling);
      if (!planner_id.empty()) planner->setPlannerId(planner_id);
      return planner;
    };
    CutPlanners planners;
    planners.cartesian = std::make_shared<mtc::solvers::CartesianPath>();
    planners.cartesian->setIKFrame(settings_.tool_link);
    planners.cartesian->setStepSize(settings_.cartesian_step_m);
    planners.cartesian->setJumpThreshold(0.0);
    planners.cartesian->setMinFraction(1.0);
    planners.cartesian->setMaxVelocityScalingFactor(settings_.velocity_scaling);
    planners.cartesian->setMaxAccelerationScalingFactor(settings_.acceleration_scaling);
    mtc::solvers::PlannerInterfacePtr transit;
    if (motion_profile == "ompl_cartesian") {
      transit = pipeline(settings_.planning_pipeline, "", true);
      planners.stroke = planners.cartesian;
    } else if (motion_profile == "ompl_pilz") {
      transit = pipeline(settings_.planning_pipeline, "", true);
      planners.stroke = pipeline(settings_.pilz_pipeline, "LIN", false);
    } else if (motion_profile == "pilz_only") {
      transit = pipeline(settings_.pilz_pipeline, "PTP", true);
      planners.stroke = pipeline(settings_.pilz_pipeline, "LIN", false);
    } else if (motion_profile == "chomp_pilz") {
      transit = pipeline(settings_.chomp_pipeline, "", true);
      planners.stroke = pipeline(settings_.pilz_pipeline, "LIN", false);
    } else {
      throw std::invalid_argument("unsupported H-cut motion_profile: " + motion_profile);
    }
    planners.transit = makeTransitPlanner(transit);
    return planners;
  }
  // Forward-only MoveTo of the planning group; returns the stage for bookkeeping.
  template<typename Goal>
  const mtc::Stage* addMoveTo(
    mtc::Task& task, const std::string& name,
    const mtc::solvers::PlannerInterfacePtr& planner, const Goal& goal,
    bool set_ik_frame = true)
  {
    auto stage = std::make_unique<mtc::stages::MoveTo>(name, planner);
    stage->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    stage->setGroup(settings_.planning_group);
    if (set_ik_frame) stage->setIKFrame(settings_.tool_link);
    stage->setGoal(goal);
    const mtc::Stage* raw = stage.get();
    task.add(std::move(stage));
    return raw;
  }
  // Top-cut lift candidates, in order: 0 deg, then +/-step, +/-2*step ... up to
  // top_cut_lift_max_tilt_deg (step 2 deg, smallest |angle| first). Each tilts about the TOOL X
  // axis (the blade's flat-face normal, so the blade stays in its slit plane) pivoting at the
  // tip, which is raised exactly as in the straight lift.
  std::vector<double> liftTiltCandidatesDeg() const
  {
    constexpr double kStepDeg = 2.0;
    std::vector<double> angles{0.0};
    for (double a = kStepDeg; a <= settings_.top_cut_lift_max_tilt_deg + 1e-9; a += kStepDeg) {
      angles.push_back(a);
      angles.push_back(-a);
    }
    return angles;
  }
  static geometry_msgs::msg::PoseStamped tiltedAboutToolX(
    const geometry_msgs::msg::PoseStamped& pose, double angle_deg)
  {
    auto tilted = pose;
    const Eigen::Quaterniond q = toEigen(pose.pose.orientation).normalized();
    const Eigen::Vector3d tool_x_base = q * Eigen::Vector3d::UnitX();
    const Eigen::Quaterniond rotated =
      Eigen::Quaterniond(Eigen::AngleAxisd(angle_deg * 3.14159265358979323846 / 180.0, tool_x_base)) * q;
    tilted.pose.orientation = toMsg(rotated.normalized());
    return tilted;
  }
  // Adds the lift to `lift_target` (orientation = the cut-end orientation). The straight lift is
  // tried first; if its Cartesian path (min_fraction 1.0) fails, the tilted candidates follow
  // inside an MTC Fallbacks container (first one that plans wins). Stage names carry
  // "[lift tilt X deg]" so the chosen tilt can be read back from the solution.
  // Fraction of a top-cut lift that must be achieved so the tip ends at least
  // top_cut_lift_min_clearance_m above the top surface (operator 2026-10-01: accept a partial
  // lift, but never one that leaves the blade in the slit). depth_m = tip depth below the top
  // at the lift start, lift_m = planned lift distance.
  double topCutLiftMinFraction(double depth_m, double lift_m) const
  {
    if (lift_m <= 1e-6) return 1.0;
    const double needed = std::max(0.0, depth_m) + settings_.top_cut_lift_min_clearance_m;
    return std::clamp(needed / lift_m, 0.0, 1.0);
  }
  void addTiltableLift(
    mtc::Task& task, const std::string& name, mtc::solvers::PlannerInterfacePtr planner,
    const geometry_msgs::msg::PoseStamped& lift_target, double min_fraction = 1.0)
  {
    if (min_fraction < 1.0 - 1e-9 &&
        std::dynamic_pointer_cast<mtc::solvers::CartesianPath>(planner)) {
      auto partial = std::make_shared<mtc::solvers::CartesianPath>();
      partial->setIKFrame(settings_.tool_link);
      partial->setStepSize(settings_.cartesian_step_m);
      partial->setJumpThreshold(0.0);
      partial->setMinFraction(min_fraction);
      partial->setMaxVelocityScalingFactor(settings_.velocity_scaling);
      partial->setMaxAccelerationScalingFactor(settings_.acceleration_scaling);
      planner = partial;
    }
    const auto angles = liftTiltCandidatesDeg();
    if (angles.size() == 1U) {
      addMoveTo(task, name + " [lift tilt 0.0 deg]", planner, lift_target);
      return;
    }
    auto fallbacks = std::make_unique<mtc::Fallbacks>(name + " (straight, then tilted)");
    for (const double angle : angles) {
      auto stage = std::make_unique<mtc::stages::MoveTo>(
        name + " [lift tilt " + formatDouble(angle, 1) + " deg]", planner);
      stage->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
      stage->setGroup(settings_.planning_group);
      stage->setIKFrame(settings_.tool_link);
      stage->setGoal(tiltedAboutToolX(lift_target, angle));
      fallbacks->add(std::move(stage));
    }
    task.add(std::move(fallbacks));
  }
  // Logs the lift tilt(s) the planned solution actually uses.
  void logChosenLiftTilt(const mtc::SolutionBase& solution, const std::string& who) const
  {
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
        const auto* creator = sub->creator();
        const std::string name = creator != nullptr ? creator->name() : std::string();
        const auto pos = name.find("[lift tilt ");
        if (pos != std::string::npos) {
          RCLCPP_INFO(get_logger(), "%s: lift stage '%s' (chosen tilt)", who.c_str(), name.c_str());
        }
      }
    };
    collect(&solution);
  }
  // Scoped tool/carton contact allowance (true) or its restore (false).
  void addCartonContactScope(mtc::Task& task, const std::string& name, bool allow)
  {
    auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(name);
    stage->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    stage->allowCollisions(settings_.tool_link, settings_.carton_collision_id, allow);
    task.add(std::move(stage));
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
    // The robot model (and with it MTC's cached planning pipelines) is loaded once.
    if (robot_model_) {
      task->setRobotModel(robot_model_);
    } else {
      task->loadRobotModel(shared_from_this());
      robot_model_ = task->getRobotModel();
    }
    task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));
    auto add_carton = std::make_unique<mtc::stages::ModifyPlanningScene>(
      "add detected carton collision object");
    add_carton->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    add_carton->addObject(cartonCollisionObject(snapshot));
    // A cut-debug start may begin while the arm is still in contact from an earlier
    // session, so the carton is added with the scoped tool/carton allowance in the same
    // atomic scene change; whole plans always start free and keep add-then-allow ordering.
    const bool contact_allowed_from_start =
      options.cut_debug_start && settings_.allow_tool_carton_contact;
    if (contact_allowed_from_start) {
      add_carton->allowCollisions(settings_.tool_link, settings_.carton_collision_id, true);
    }
    task->add(std::move(add_carton));
    addCameraPole(*task);
    if (options.pre_transit_lift_m > 0.0) {
      auto lift_planner = std::make_shared<mtc::solvers::CartesianPath>();
      lift_planner->setIKFrame(settings_.tool_link);
      lift_planner->setStepSize(settings_.cartesian_step_m);
      lift_planner->setJumpThreshold(0.0);
      lift_planner->setMinFraction(settings_.top_cut_pre_transit_lift_min_fraction);
      lift_planner->setMaxVelocityScalingFactor(settings_.velocity_scaling);
      lift_planner->setMaxAccelerationScalingFactor(settings_.acceleration_scaling);
      auto lift = std::make_unique<mtc::stages::MoveRelative>(
        "lift straight up before transit (table clearance)", lift_planner);
      lift->setGroup(settings_.planning_group);
      lift->setIKFrame(settings_.tool_link);
      geometry_msgs::msg::Vector3Stamped up;
      up.header.frame_id = settings_.planning_frame;
      up.vector.z = options.pre_transit_lift_m;
      lift->setDirection(up);
      task->add(std::move(lift));
    }
    moveit::core::RobotState ik_state(robot_model_);
    ik_state.setToDefaultValues();
    const auto& model_variables = robot_model_->getVariableNames();
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
    const auto* joint_group = robot_model_->getJointModelGroup(settings_.planning_group);
    if (joint_group == nullptr) {
      throw std::runtime_error("planning group is missing from robot model");
    }

    const CutPlanners planners = makeCutPlanners(motion_profile);
    const auto& transit_planner = planners.transit;
    const auto& stroke_planner = planners.stroke;
    // Every free-space transit stage built below, recorded for the wrist guard.
    std::vector<const mtc::Stage*> transit_stages;
    if (!options.skip_supplied_approach && !settings_.approach_joint_degrees.empty()) {
      transit_stages.push_back(addMoveTo(
        *task, "transit to supplied link6 approach [" + motion_profile + "]", transit_planner,
        jointGoalDegrees(joint_group, settings_.approach_joint_degrees),
        /*set_ik_frame=*/false));
    }
    cut_targets.clear();
    for (std::size_t segment_index = 0; segment_index < segments.size(); ++segment_index) {
      const auto& segment = segments[segment_index];
      const bool is_middle_seam = segment.cut == CutKind::kMiddle;
      const bool is_right_side = segment.cut == CutKind::kRight;
      const bool is_left_side = segment.cut == CutKind::kLeft;
      const std::string segment_cut = cutName(segment.cut);
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
        CutBasis correction_basis;
        std::string basis_error;
        if (!cutDebugBasis(snapshot, segment_cut, correction_basis, basis_error)) {
          throw std::runtime_error("cut debug correction basis failed: " + basis_error);
        }
        const CutLocalDeltas& local_start = segment_correction->start_local;
        CutLocalDeltas local_end = segment_correction->end_local;
        if (segment.handoff_end && is_middle_seam) {
          const double fraction = segment.handoff_fraction;
          local_end.along_m = local_start.along_m +
            fraction * (segment_correction->end_local.along_m - local_start.along_m);
          local_end.across_m = local_start.across_m +
            fraction * (segment_correction->end_local.across_m - local_start.across_m);
          local_end.normal_m = local_start.normal_m +
            fraction * (segment_correction->end_local.normal_m - local_start.normal_m);
        }
        corrected_start_point =
          applyCutLocalDelta(corrected_start_point, correction_basis, local_start);
        corrected_end_point = applyCutLocalDelta(corrected_end_point, correction_basis, local_end);
      }
      const double tilt_deg = is_middle_seam ? 0.0 :
        (is_right_side ? settings_.right_side_cut_tilt_deg : settings_.left_side_cut_tilt_deg);
      // Side strokes travel back-to-front, but their nominal knife frame faces 180 degrees
      // away from travel. Position corrections must not rotate that nominal baseline, so
      // the direction comes from the uncorrected segment. Learned orientation records
      // bypass the nominal frame below: their complete taught TCP quaternions are mapped
      // from the recorded cut basis into the current cut basis.
      const auto orientation_targets = orientationTargets(
        snapshot, is_middle_seam, segment.start, segment.end, corrected_start_point,
        corrected_end_point);
      const double segment_cut_clearance_m =
        is_middle_seam ? settings_.middle_cut_clearance_m : settings_.cut_clearance_m;
      const double target_clearance_m =
        segment.hover_only ? settings_.debug_motion_clearance_m : segment_cut_clearance_m;
      const double approach_clearance_m =
        segment.hover_only ? settings_.debug_motion_clearance_m : settings_.hover_clearance_m;
      // Left hover poses are the cut poses withdrawn laterally (below), so they are not
      // derived from the hover clearance.
      const bool left_lateral_hover = is_left_side && !segment.hover_only;
      auto cut_start = targetPose(
        snapshot, corrected_start_point, orientation_targets.first, target_clearance_m,
        tilt_deg);
      auto cut_end = targetPose(
        snapshot, corrected_end_point, orientation_targets.second, target_clearance_m,
        tilt_deg);
      if (is_middle_seam &&
          (options.middle_start_extra_depth_m != 0.0 || options.middle_end_extra_depth_m != 0.0)) {
        const Eigen::Vector3d deeper = -toEigen(snapshot.surface_normal).normalized();
        cut_start.pose.position = toPoint(
          toEigen(cut_start.pose.position) + options.middle_start_extra_depth_m * deeper);
        cut_end.pose.position = toPoint(
          toEigen(cut_end.pose.position) + options.middle_end_extra_depth_m * deeper);
      }
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
        const auto fixed = fixedDebugOrientation();
        hover_start.pose.orientation = fixed;
        cut_start.pose.orientation = fixed;
        hover_end.pose.orientation = fixed;
        cut_end.pose.orientation = fixed;
      }
      if (segment_correction != nullptr && !segment.force_fixed_orientation) {
        geometry_msgs::msg::Quaternion handoff_end_observed = segment_correction->observed_end;
        if (segment.handoff_end && is_middle_seam) {
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
            snapshot, hover_start, settings_.left_lateral_standoff_m, message) ||
            !offsetLeftPoseOutward(
            snapshot, hover_end, settings_.left_lateral_standoff_m, message)) {
          throw std::runtime_error(message);
        }
        // Side-tape standoff: the entry ends at the standoff pose (further out than the
        // in-material start), so start it left_lateral_standoff_m beyond that pose; offsetting
        // from cut_start would put the hover inside the standoff and the entry would back away.
        if (segment.stop_after_start && options.stop_pose_override) {
          hover_start.pose.position = options.stop_pose_override->pose.position;
          if (!offsetLeftPoseOutward(
              snapshot, hover_start, settings_.left_lateral_standoff_m, message)) {
            throw std::runtime_error(message);
          }
        }
      }
      const bool left_parallel = is_left_side && !segment.force_fixed_orientation &&
        !segment.hover_only && settings_.left_parallel_orientation_enabled;
      if (left_parallel) {
        if (segment_correction != nullptr) {
          message =
            "left_parallel_orientation_enabled cannot be combined with an active left correction "
            "record; clear cut_debug_left_correction_file or disable cut_debug_correction_enabled";
          return false;
        }
        // Same poses the MTC stages target; the lift exists only before a middle segment.
        const bool lift_follows = !options.cut_debug_start &&
          (segment_index + 1U) < segments.size() &&
          segments[segment_index + 1U].cut == CutKind::kMiddle &&
          !segments[segment_index + 1U].hover_only;
        auto lift_probe = hover_end;
        const double lift_m = settings_.hover_clearance_m - settings_.cut_clearance_m;
        lift_probe.pose.position.x += lift_m * snapshot.surface_normal.x;
        lift_probe.pose.position.y += lift_m * snapshot.surface_normal.y;
        lift_probe.pose.position.z += lift_m * snapshot.surface_normal.z;
        std::vector<std::pair<geometry_msgs::msg::PoseStamped*, const char*> > left_poses;
        if (options.cut_debug_start) {
          left_poses = {{&hover_start, "left hover start"}, {&cut_start, "left cut start"}};
        } else {
          left_poses = {
            {&hover_start, "left hover start"}, {&cut_start, "left cut start"},
            {&cut_end, "left cut end"}, {&hover_end, "left hover end"}};
          if (lift_follows) left_poses.push_back({&lift_probe, "left lift"});
        }
        const std::string pose_context = options.cut_debug_start ?
          "cut debug left start" : "left " + segment.name;
        LeftRollCache* cache = options.left_roll_cache;
        if (cache != nullptr && cache->computed && cache->orientations.size() == 1U) {
          if (!cache->feasible) {
            message = cache->error;
            return false;
          }
          for (const auto& labeled : left_poses) {
            labeled.first->pose.orientation = cache->orientations.front();
          }
        } else {
          const bool feasible = searchLeftParallelOrientation(
            snapshot, joint_group, ik_state, left_poses, pose_context, message,
            options.side_tape_handoff);
          if (cache != nullptr) {
            cache->computed = true;
            cache->feasible = feasible;
            cache->error = feasible ? std::string() : message;
            cache->orientations = {hover_start.pose.orientation};
          }
          if (!feasible) return false;
        }
      }
      if (is_left_side && segment_correction != nullptr && !left_parallel &&
          !segment.force_fixed_orientation && settings_.left_adaptive_orientation_enabled) {
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
        LeftRollCache* cache = options.left_roll_cache;
        if (cache != nullptr && cache->computed &&
            (!cache->feasible || cache->orientations.size() == left_poses.size())) {
          if (!cache->feasible) {
            message = cache->error;
            return false;
          }
          for (std::size_t index = 0; index < left_poses.size(); ++index) {
            left_poses[index].first->pose.orientation = cache->orientations[index];
          }
        } else {
          const bool feasible = adaptLeftCorrectedOrientations(
            snapshot, joint_group, ik_state, left_poses, pose_context, message);
          if (!feasible && !options.cut_debug_start) {
            message = "left orientation adaptation failed for segment " + segment.name + ": " +
              message;
          }
          if (cache != nullptr) {
            cache->computed = true;
            cache->feasible = feasible;
            cache->error = feasible ? std::string() : message;
            cache->orientations.clear();
            for (const auto& labeled : left_poses) {
              cache->orientations.push_back(labeled.first->pose.orientation);
            }
          }
          if (!feasible) return false;
        }
      }
      if (segment.force_fixed_orientation) {
        moveit::core::RobotState debug_ik_state(ik_state);
        if (!debug_ik_state.setFromIK(joint_group, cut_start.pose, settings_.tool_link, 1.0)) {
          message =
            "fixed debug orientation (x=0,y=1,z=0,w=0) has no kinematic IK solution "
            "at the short-stroke start; collision checking was not involved";
          return false;
        }
        if (!debug_ik_state.setFromIK(joint_group, cut_end.pose, settings_.tool_link, 1.0)) {
          message =
            "fixed debug orientation (x=0,y=1,z=0,w=0) has no kinematic IK solution "
            "at the short-stroke end; collision checking was not involved";
          return false;
        }
      }
      if (is_right_side) {
        moveit::core::RobotState approach_ik_state(ik_state);
        if (!approach_ik_state.setFromIK(
              joint_group, hover_start.pose, settings_.tool_link, 0.25)) {
          message = "right-side approach IK infeasible before Cartesian stroke";
          return false;
        }
      }
      const bool middle_entry_alignment =
        is_middle_seam && settings_.middle_entry_alignment_m > 0.0;
      auto entry_alignment = hover_start;
      if (middle_entry_alignment) {
        const Eigen::Vector3d direction =
          toEigen(corrected_end_point) - toEigen(corrected_start_point);
        const double length = direction.norm();
        if (length <= 0.0) throw std::runtime_error("zero-length middle seam direction");
        entry_alignment.pose.position = toPoint(
          toEigen(entry_alignment.pose.position) -
          settings_.middle_entry_alignment_m * direction / length);
      }
      cut_targets.push_back(cut_start);
      if (!segment.stop_after_start) cut_targets.push_back(cut_end);
      if (options.hover_start_out != nullptr) *options.hover_start_out = hover_start;
      {
        const std::string first_approach = is_left_side ?
          "approach pose outside " + segment.name : "approach pose above " + segment.name;
        const std::string transit_name = segment_index == 0U ? first_approach :
          "transition from " + segments[segment_index - 1U].name + " finish pose to " +
          segment.name + " approach pose";
        const std::string stage_name = transit_name + " [" + motion_profile + "]";
        if (motion_profile == "chomp_pilz") {
          transit_stages.push_back(addMoveTo(
            *task, stage_name, transit_planner, jointGoal(ik_state, joint_group, entry_alignment)));
        } else {
          transit_stages.push_back(addMoveTo(*task, stage_name, transit_planner, entry_alignment));
        }
      }
      if (middle_entry_alignment) {
        addMoveTo(
          *task, "align above middle seam start [" + motion_profile + "]", stroke_planner,
          hover_start);
      }
      if (settings_.pre_cut_stabilization_duration_s > 0.0) {
        task->add(std::make_unique<StationaryDwell>(
          "stabilize blade before " + segment.name, settings_.planning_group,
          settings_.pre_cut_stabilization_duration_s));
      }
      if (options.stop_at_hover_start) continue;
      if (!segment.hover_only) {
        if (settings_.allow_tool_carton_contact && !contact_allowed_from_start) {
          addCartonContactScope(
            *task, "allow knife-tip contact with carton for " + segment.name, true);
        }
        const std::string entry_label = is_left_side ? "enter laterally into " :
          "descend from aligned ";
        // Side-tape hand-off standoff variant only: the single Cartesian entry stroke
        // targets the standoff override pose directly, so it NEVER targets, or interpolates
        // through, the true in-material cut_start -- the approach/hover geometry above is
        // still derived from the true cut_start, only this terminal target is substituted.
        const bool use_override = segment.stop_after_start && options.stop_pose_override;
        const auto& entry_target = use_override ? *options.stop_pose_override : cut_start;
        addMoveTo(
          *task, entry_label + segment.name + " start [" + motion_profile + "]", stroke_planner,
          entry_target);
        if (use_override) cut_targets.back() = entry_target;
        // Operator 2026-10-01: hold at the top-cut penetration pose before the cut stroke.
        if (is_middle_seam && settings_.top_cut_post_penetration_dwell_s > 0.0) {
          task->add(std::make_unique<StationaryDwell>(
            "hold at penetration before " + segment.name, settings_.planning_group,
            settings_.top_cut_post_penetration_dwell_s));
        }
      }
      if (segment.stop_after_start) {
        continue;
      }

      auto cut_stroke_planner = stroke_planner;
      CutPlanners slow_cut_planners;
      if (options.cut_stroke_speed_factor != 1.0) {
        slow_cut_planners = makeCutPlanners(motion_profile);
        slow_cut_planners.stroke->setMaxVelocityScalingFactor(
          settings_.velocity_scaling * options.cut_stroke_speed_factor);
        slow_cut_planners.stroke->setMaxAccelerationScalingFactor(
          settings_.acceleration_scaling * options.cut_stroke_speed_factor);
        cut_stroke_planner = slow_cut_planners.stroke;
      }
      addMoveTo(
        *task, "cut " + segment.name + " [" + motion_profile + "]", cut_stroke_planner, cut_end);
      // Terminal handoff: the left arm stops at the cut-level handoff endpoint after the
      // traverse above. Skip the Cartesian hover retract and leave the planned tool/carton
      // contact allowance active; a second arm (not modeled here) resumes from this pose.
      if (!segment.hover_only && !segment.terminal_after_cut) {
        const std::string finish_label = is_left_side ? "finish pose outside " :
          "finish pose above ";
        if (is_middle_seam) {
          // Top cut only: finish at top_cut_lift_clearance_m above the top, not the hover height.
          auto finish_pose = hover_end;
          if (options.middle_finish_clearance_m) {
            const Eigen::Vector3d up = toEigen(snapshot.surface_normal).normalized();
            finish_pose.pose.position = toPoint(
              toEigen(finish_pose.pose.position) +
              (*options.middle_finish_clearance_m - approach_clearance_m) * up);
          }
          // Straight lift first; if it fails the blade may tilt about tool X (see addTiltableLift).
          const double finish_clear_m = options.middle_finish_clearance_m ?
            *options.middle_finish_clearance_m : approach_clearance_m;
          const double end_depth_m = -settings_.middle_cut_clearance_m +
            options.middle_end_extra_depth_m;
          addTiltableLift(
            *task, finish_label + segment.name + " [" + motion_profile + "]", stroke_planner,
            finish_pose, options.middle_finish_clearance_m ?
            topCutLiftMinFraction(end_depth_m, end_depth_m + finish_clear_m) : 1.0);
        } else {
          addMoveTo(
            *task, finish_label + segment.name + " [" + motion_profile + "]", stroke_planner,
            hover_end);
        }
        const bool lift_before_middle = is_left_side &&
          (segment_index + 1U) < segments.size() &&
          segments[segment_index + 1U].cut == CutKind::kMiddle &&
          !segments[segment_index + 1U].hover_only;
        if (lift_before_middle) {
          // Safety: pull laterally clear first (ascend above), then lift along the
          // snapshot surface normal to hover clearance while holding the exact left
          // finish orientation. Collision restore and the free-space transit/rotation
          // to the middle approach happen only after this lift.
          auto lift_target = hover_end;
          const double lift_m = settings_.hover_clearance_m - settings_.cut_clearance_m;
          lift_target.pose.position.x += lift_m * snapshot.surface_normal.x;
          lift_target.pose.position.y += lift_m * snapshot.surface_normal.y;
          lift_target.pose.position.z += lift_m * snapshot.surface_normal.z;
          addMoveTo(
            *task, "lift clear of " + segment.name + " to hover [cartesian]",
            planners.cartesian, lift_target);
        }
        if (settings_.allow_tool_carton_contact) {
          addCartonContactScope(
            *task, "restore knife-tip collision checking after " + segment.name, false);
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
    const auto result = task->plan(static_cast<std::size_t>(settings_.max_solutions));
    if (result != moveit::core::MoveItErrorCode::SUCCESS || task->solutions().empty()) {
      task->explainFailure();
      message = "MTC candidate failed: task=" + task_name + " profile=" + motion_profile +
                " segments=" + std::to_string(segments.size());
      return false;
    }
    // Serialized once and shared by both fail-closed solution validators. Publication
    // happens only for the retained plan (retainPlan / cut-debug start).
    const auto& solution = *task->solutions().front();
    logChosenLiftTilt(solution, "planCutSegments(" + task_name + ")");
    moveit_task_constructor_msgs::msg::Solution solution_message;
    solution.toMsg(solution_message);
    std::size_t expected_dwell_count =
      settings_.pre_cut_stabilization_duration_s > 0.0 ? segments.size() : 0U;
    // Plus one top-cut penetration hold per middle segment that descends (see planCutSegments).
    if (settings_.top_cut_post_penetration_dwell_s > 0.0 && !options.stop_at_hover_start) {
      for (const auto& planned_segment : segments) {
        if (planned_segment.cut == CutKind::kMiddle && !planned_segment.hover_only) {
          ++expected_dwell_count;
        }
      }
    }
    std::string dwell_error;
    if (!validateStationaryDwells(solution_message, expected_dwell_count, dwell_error)) {
      message = "planned trajectory has invalid pre-cut stabilization: " + dwell_error;
      return false;
    }
    std::string wrist_error;
    if (!validateFreeSpaceTransitWrist(solution, solution_message, transit_stages, wrist_error)) {
      message = wrist_error;
      return false;
    }
    if (settings_.left_parallel_orientation_enabled &&
        !validateLeftContactJointTravel(solution, wrist_error)) {
      message = wrist_error;
      return false;
    }
    return true;
  }

  // Parallel-left mode only: the Cartesian contact strokes run with jump detection off,
  // so a wrist flip through the joint5 singularity inside a left entry/cut/exit/lift stroke
  // would otherwise be accepted. Rejects any such stroke whose joint travel exceeds
  // kMaxLeftContactJointTravelRad for any joint (a constant-orientation 25-60 mm stroke
  // needs far less).
  bool validateLeftContactJointTravel(const mtc::SolutionBase& solution, std::string& error) const
  {
    constexpr double kMaxLeftContactJointTravelRad = 1.0;
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
      if (const auto* sub = dynamic_cast<const mtc::SubTrajectory*>(node)) leaves.push_back(sub);
    };
    collect(&solution);
    std::size_t examined = 0U;
    for (const auto* leaf : leaves) {
      const auto* creator = leaf->creator();
      const std::string name = creator != nullptr ? creator->name() : std::string();
      const bool left_contact = name.rfind("enter laterally into left side cut", 0) == 0U ||
        name.rfind("cut left side cut", 0) == 0U ||
        name.rfind("finish pose outside left side cut", 0) == 0U ||
        name.rfind("lift clear of left side cut", 0) == 0U;
      if (!left_contact) continue;
      ++examined;
      const auto& trajectory = leaf->trajectory();
      if (!trajectory || trajectory->getWayPointCount() == 0U) {
        error = "left contact stroke '" + name + "' has no trajectory; refusing to guess";
        return false;
      }
      std::vector<double> previous;
      std::vector<double> travel;
      for (std::size_t index = 0; index < trajectory->getWayPointCount(); ++index) {
        std::vector<double> values;
        trajectory->getWayPoint(index).copyJointGroupPositions(settings_.planning_group, values);
        if (previous.empty()) travel.assign(values.size(), 0.0);
        for (std::size_t j = 0; j < values.size() && !previous.empty(); ++j) {
          travel[j] += std::abs(values[j] - previous[j]);
        }
        previous = values;
      }
      for (std::size_t j = 0; j < travel.size(); ++j) {
        if (travel[j] > kMaxLeftContactJointTravelRad) {
          error = "left contact stroke '" + name + "' moves joint" + std::to_string(j + 1) + " " +
            formatDouble(travel[j], 3) + " rad (> " +
            formatDouble(kMaxLeftContactJointTravelRad, 1) +
            " rad): wrist branch change during contact; refusing the parallel left plan";
          return false;
        }
      }
    }
    (void)examined;  // plans without a left segment have nothing to check
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
      const double dwell_end_s = seconds(points.back().time_from_start);
      if (std::abs(seconds(points.front().time_from_start)) > 1e-9 ||
          (std::abs(dwell_end_s - settings_.pre_cut_stabilization_duration_s) > 1e-6 &&
           std::abs(dwell_end_s - settings_.top_cut_post_penetration_dwell_s) > 1e-6)) {
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
  // interpolation first and the profile's existing planner second, all within one
  // planning_timeout_s budget per transit. Cutting strokes never use this stack.
  mtc::solvers::PlannerInterfacePtr makeTransitPlanner(
    const mtc::solvers::PlannerInterfacePtr& fallback) const
  {
    auto direct = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
    direct->setMaxVelocityScalingFactor(settings_.transit_velocity_scaling);
    direct->setMaxAccelerationScalingFactor(settings_.transit_acceleration_scaling);
    auto multi = std::make_shared<mtc::solvers::MultiPlanner>();
    multi->push_back(direct);
    multi->push_back(fallback);
    // Pose goals are resolved to the nearest exact IK branch before either planner runs.
    return std::make_shared<NearestBranchTransitPlanner>(multi, settings_.planning_timeout_s);
  }

  // Fail-closed wrist guard for free-space transits: rejects the retained solution
  // when a free-space MoveTo transit winds joint6 through more than pi radians
  // of total travel. Travel is summed over every consecutive waypoint pair so
  // intermediate winding cannot hide behind benign endpoints. Cutting, align,
  // descend, ascend, lift, and dwell stages are never examined here. Transit stages
  // are the ones recorded while the task was built (`transit_stages`); the check fails
  // closed unless every recorded transit is examined exactly once and no other stage
  // carries a transit name.
  bool validateFreeSpaceTransitWrist(
    const mtc::SolutionBase& solution,
    const moveit_task_constructor_msgs::msg::Solution& solution_message,
    const std::vector<const mtc::Stage*>& transit_stages, std::string& error) const
  {
    const std::string prefix = "planned free-space transit winds the wrist: ";
    const auto* sequence = dynamic_cast<const mtc::SolutionSequence*>(&solution);
    if (sequence == nullptr) {
      error = prefix + "unexpected solution type; refusing to guess transit stages";
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
      error = prefix + "solution trajectory count does not match planned stages; refusing to guess";
      return false;
    }
    const std::set<const mtc::Stage*> recorded(transit_stages.begin(), transit_stages.end());
    if (recorded.size() != transit_stages.size() || recorded.count(nullptr) != 0U) {
      error = prefix + "transit stage bookkeeping is inconsistent; refusing to guess";
      return false;
    }
    const auto has_transit_name = [](const std::string& stage_name) {
      return stage_name.rfind("transit to supplied", 0) == 0U ||
        stage_name.rfind("approach pose above ", 0) == 0U ||
        stage_name.rfind("approach pose outside ", 0) == 0U ||
        stage_name.rfind("transition from ", 0) == 0U;
    };
    constexpr double kMaxWristTravelRad = 3.14159265358979323846;
    std::set<const mtc::Stage*> examined;
    for (std::size_t index = 0; index < leaves.size(); ++index) {
      const mtc::Stage* creator = leaves[index]->creator();
      const std::string stage_name = creator != nullptr ? creator->name() : std::string();
      const bool is_recorded = recorded.count(creator) != 0U;
      if (!is_recorded) {
        if (has_transit_name(stage_name)) {
          error = prefix + "stage '" + stage_name +
            "' is named as a transit but was not recorded as one; refusing to guess";
          return false;
        }
        continue;
      }
      if (!examined.insert(creator).second) {
        error = prefix + "transit '" + stage_name +
          "' produced more than one trajectory; refusing to guess";
        return false;
      }
      const auto& sub_msg = solution_message.sub_trajectory[index];
      const auto& joint_names = sub_msg.trajectory.joint_trajectory.joint_names;
      const auto& points = sub_msg.trajectory.joint_trajectory.points;
      const auto joint6_it = std::find(joint_names.begin(), joint_names.end(), "joint6");
      if (joint6_it == joint_names.end()) {
        error = prefix + "transit '" + stage_name + "' has no joint6 waypoints";
        return false;
      }
      const auto joint6 = static_cast<std::size_t>(std::distance(joint_names.begin(), joint6_it));
      if (points.size() < 2U) {
        error = prefix + "transit '" + stage_name + "' has missing joint waypoints";
        return false;
      }
      double travel6 = 0.0;
      for (std::size_t point = 1U; point < points.size(); ++point) {
        const auto& prev = points[point - 1U].positions;
        const auto& curr = points[point].positions;
        if (prev.size() != joint_names.size() || curr.size() != joint_names.size()) {
          error = prefix + "transit '" + stage_name + "' has incomplete joint waypoints";
          return false;
        }
        if (!std::isfinite(prev[joint6]) || !std::isfinite(curr[joint6])) {
          error = prefix + "transit '" + stage_name + "' has non-finite wrist waypoints";
          return false;
        }
        travel6 += std::abs(curr[joint6] - prev[joint6]);
      }
      if (travel6 > kMaxWristTravelRad) {
        std::ostringstream details;
        details << prefix << "transit '" << stage_name << "' joint6 travel " << travel6 <<
          " rad exceeds " << kMaxWristTravelRad << " rad";
        error = details.str();
        return false;
      }
    }
    if (examined.size() != recorded.size()) {
      error = prefix + "examined " + std::to_string(examined.size()) + " of " +
        std::to_string(recorded.size()) +
        " planned free-space transits; refusing to accept an unchecked transit";
      return false;
    }
    return true;
  }

  static geometry_msgs::msg::Point interpolate(
    const geometry_msgs::msg::Point& start, const geometry_msgs::msg::Point& end,
    double fraction)
  {
    geometry_msgs::msg::Point point;
    point.x = start.x + fraction * (end.x - start.x);
    point.y = start.y + fraction * (end.y - start.y);
    point.z = start.z + fraction * (end.z - start.z);
    return point;
  }

  static double distanceBetween(
    const geometry_msgs::msg::Point& start, const geometry_msgs::msg::Point& end)
  {
    return (toEigen(end) - toEigen(start)).norm();
  }

  // middle_debug_short / debug_fixed_orientation tool orientation, XYZW=(0,1,0,0).
  static geometry_msgs::msg::Quaternion fixedDebugOrientation()
  {
    geometry_msgs::msg::Quaternion fixed;
    fixed.x = 0.0;
    fixed.y = 1.0;
    fixed.z = 0.0;
    fixed.w = 0.0;
    return fixed;
  }

  geometry_msgs::msg::PoseStamped targetPose(
    const Snapshot& snapshot, const geometry_msgs::msg::Point& point,
    const geometry_msgs::msg::Point& direction_target, double clearance_m,
    double tilt_deg = 0.0) const
  {
    const Eigen::Vector3d z_axis = (-toEigen(snapshot.surface_normal)).normalized();
    // The cutter's local Y axis is its blade length. Keep it along the seam,
    // rather than using local X as was appropriate only for the top-seam trial.
    Eigen::Vector3d y_axis = toEigen(direction_target) - toEigen(point);
    y_axis -= y_axis.dot(z_axis) * z_axis;
    y_axis.normalize();
    const Eigen::Vector3d x_axis = y_axis.cross(z_axis).normalized();
    // Side tilt is a roll about the blade, so it cannot rotate the blade off seam.
    const double radians = tilt_deg * std::acos(-1.0) / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    const Eigen::Vector3d tilted_x_axis = cosine * x_axis - sine * z_axis;
    const Eigen::Vector3d tilted_z_axis = sine * x_axis + cosine * z_axis;

    geometry_msgs::msg::PoseStamped pose;
    pose.header = snapshot.header;
    pose.pose.position = toPoint(toEigen(point) + clearance_m * toEigen(snapshot.surface_normal));
    pose.pose.orientation =
      toMsg(basisQuaternion(tilted_x_axis, y_axis, tilted_z_axis).normalized());
    if (settings_.debug_fixed_orientation) pose.pose.orientation = fixedDebugOrientation();
    return pose;
  }

  std::map<std::string, double> jointGoal(
    moveit::core::RobotState& state,
    const moveit::core::JointModelGroup* group,
    const geometry_msgs::msg::PoseStamped& pose) const
  {
    if (!state.setFromIK(group, pose.pose, settings_.tool_link, 0.25)) {
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
      const bool middle_seam = segment.cut == CutKind::kMiddle;
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
  // Held for the whole duration of every Trigger service (try_lock) and by the
  // parameter callback (try_lock only) when applying planning parameters.
  std::mutex plan_mutex_;
  std::mutex state_mutex_;
  std::mutex camera_pole_mutex_;
  std::mutex runtime_parameter_mutex_;
  PlannerSettings settings_;
  // Runtime parameters: readable without plan_mutex_ or rclcpp's parameter mutex.
  std::atomic<bool> execution_enabled_{false};
  std::atomic<bool> debug_motion_execution_enabled_{false};
  std::atomic<bool> left_roll_diagnostics_enabled_{false};
  // Phase 1 analytic-IK startup gate (see ensureAnalyticIkSelfCheck): run once, lazily, the
  // first time a joint model group is available, against the live RobotModel's own joint
  // limits. If the self-check fails, the analytic backend is disabled for the rest of the
  // process (falls back to "kdl") regardless of left_search_ik_backend, and a warning is
  // logged once.
  std::once_flag analytic_self_check_once_;
  std::atomic<bool> analytic_ik_available_{false};
  std::string external_snapshot_file_;
  std::optional<sensor_msgs::msg::JointState> current_joint_state_;
  std::optional<moveit_msgs::msg::CollisionObject> camera_pole_;
  rclcpp::Time joint_state_received_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr camera_pole_sub_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr target_pose_pub_;
  rclcpp::Publisher<moveit_msgs::msg::CollisionObject>::SharedPtr carton_collision_pub_;
  TriggerService::SharedPtr plan_service_;
  TriggerService::SharedPtr section_diagnostic_service_;
  TriggerService::SharedPtr execute_service_;
  TriggerService::SharedPtr clear_service_;
  TriggerService::SharedPtr cut_debug_start_service_;
  TriggerService::SharedPtr cut_debug_save_service_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
  // Loaded once, reused by every task (Task::setRobotModel) and by the cut-debug FK.
  moveit::core::RobotModelConstPtr robot_model_;
  std::optional<Snapshot> planned_snapshot_;
  // True when the retained plan's snapshot was loaded under debug_allow_stale_snapshot.
  bool retained_with_stale_override_{false};
  // Set only while continue_top_cut loads its snapshot with
  // top_cut_continue_check_snapshot_age=false (operator: box static, alignment is slow).
  mutable bool skip_snapshot_age_{false};
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
  std::unique_ptr<mtc::Task> planned_task_;
  std::thread execution_thread_;
  bool execution_in_progress_{false};
  // Side-tape hand-off (left only). side_tape_handoff_pending_ marks a retained plan (in
  // planned_task_/planned_snapshot_) as a hand-off approach; it is consumed (cleared) the
  // moment execution is dispatched, and also unconditionally by clearRetainedPlan() so it
  // can never leak into an unrelated later plan. side_tape_handoff_info_.inProgress()
  // instead spans dispatch completion (active or approach_failed) -> side_tape_retract
  // completion, and is what guards the carton object and busy-rejects other
  // planning/clear operations.
  LeftParallelSelection last_left_parallel_selection_;
  bool side_tape_handoff_pending_{false};
  SideTapeHandoffInfo side_tape_handoff_info_;
  std::optional<Snapshot> side_tape_handoff_snapshot_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr side_tape_handoff_pub_;
  TriggerService::SharedPtr side_tape_handoff_plan_service_;
  TriggerService::SharedPtr side_tape_retract_service_;
  TriggerService::SharedPtr side_tape_continue_service_;
  TriggerService::SharedPtr side_cut_penetrate_service_;
  TopCutStagedInfo top_cut_info_;
  std::optional<Snapshot> top_cut_snapshot_;
  bool top_cut_pending_{false};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr top_cut_pub_;
  TriggerService::SharedPtr top_cut_plan_service_, top_cut_penetrate_service_, top_cut_cut_service_,
    top_cut_retract_service_, top_cut_continue_service_;
  // Servo activity, watched only to refuse side_tape_retract while Servo looks active;
  // Servo itself is never paused/stopped from here.
  std::mutex servo_activity_mutex_;
  std::optional<rclcpp::Time> servo_twist_last_received_;
  std::optional<rclcpp::Time> servo_joint_jog_last_received_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr servo_twist_sub_;
  rclcpp::Subscription<control_msgs::msg::JointJog>::SharedPtr servo_joint_jog_sub_;
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
