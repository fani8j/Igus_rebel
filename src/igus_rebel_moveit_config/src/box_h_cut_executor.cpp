#include <Eigen/Geometry>
#include <control_msgs/msg/joint_jog.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/cartesian_interpolator.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_parameterization.h>
#include <moveit/task_constructor/container.h>
#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/joint_interpolation.h>
#include <moveit/task_constructor/solvers/multi_planner.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stage.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/fixed_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/stages/move_relative.h>
#include <moveit/task_constructor/task.h>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <moveit_task_constructor_msgs/msg/task_description.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <dual_cut_msgs/msg/planned_segment.hpp>
#include <dual_cut_msgs/srv/get_planned_solution.hpp>
#include <dual_cut_msgs/srv/plan_from_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "igus_rebel_moveit_config/rebel_analytic_ik.hpp"
#include "igus_rebel_moveit_config/left_varying_chain.hpp"
#include "igus_rebel_moveit_config/snapshot_frame.hpp"
#include "igus_rebel_moveit_config/plan_export.hpp"
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
#include <tuple>
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
    return guardResult(
      inner_->plan(from, to, jmg, std::min(timeout, max_total_s_), result, path_constraints),
      result);
  }

  Result plan(
    const planning_scene::PlanningSceneConstPtr& from, const moveit::core::LinkModel& link,
    const Eigen::Isometry3d& offset, const Eigen::Isometry3d& target,
    const moveit::core::JointModelGroup* jmg, double timeout,
    robot_trajectory::RobotTrajectoryPtr& result,
    const moveit_msgs::msg::Constraints& path_constraints) override
  {
    return guardResult(
      planToPose(from, link, offset, target, jmg, timeout, result, path_constraints), result);
  }

  std::string getPlannerId() const override { return "NearestBranchTransit"; }

private:
  // MTC's MoveTo::compute() calls result->getLastWayPoint() whenever `result` is non-null, even
  // when planning FAILED. Attempts that fail leave a stale or empty trajectory behind:
  // PipelinePlanner keeps the invalid path of an INVALID_MOTION_PLAN result, and the retry's
  // MultiPlanner::plan() then clear()s that trajectory and may return {false, "timeout"} without
  // planning anything. getLastWayPoint() on the emptied trajectory is undefined behaviour (seen as
  // SIGSEGV in MoveTo::compute). So a failed transit never hands a trajectory back, and a
  // "successful" one must contain waypoints.
  static Result guardResult(Result outcome, robot_trajectory::RobotTrajectoryPtr& result)
  {
    if (outcome && (!result || result->empty())) {
      outcome = Result{false, "transit planner reported success without a trajectory"};
    }
    if (!outcome) result.reset();
    return outcome;
  }

  Result planToPose(
    const planning_scene::PlanningSceneConstPtr& from, const moveit::core::LinkModel& link,
    const Eigen::Isometry3d& offset, const Eigen::Isometry3d& target,
    const moveit::core::JointModelGroup* jmg, double timeout,
    robot_trajectory::RobotTrajectoryPtr& result,
    const moveit_msgs::msg::Constraints& path_constraints)
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

  mtc::solvers::PlannerInterfacePtr inner_;
  double max_total_s_;
};

// Varying-orientation left cut: ONE contact stroke through a list of knot poses. Behaves like
// mtc::solvers::CartesianPath (same properties, scalings, collision validity on the stage's
// planning scene, min_fraction 1.0 semantics) but runs
// CartesianInterpolator::computeCartesianPath leg by leg over the knot list, carrying the robot
// state from leg to leg (orientation slerped inside each leg, TCP straight between knots),
// requires fraction 1.0 for EVERY leg (the whole path fails closed otherwise), additionally
// rejects any joint step above `max_joint_step_rad` between consecutive waypoints, and
// time-parameterises the whole path ONCE with the planner's velocity/acceleration scalings.
// The knots are given in the planning frame; the final MoveTo target (model frame) fixes the
// rigid transform between the two. The first knot is the stroke's start pose and must already be
// reached by the incoming state (it is checked, not driven to).
class WaypointCartesianPath : public mtc::solvers::CartesianPath
{
public:
  using mtc::solvers::CartesianPath::plan;
  struct Config {
    double step_m{0.003};
    double rotation_step_rad{0.05};
    double max_joint_step_rad{0.0873};
  };
  WaypointCartesianPath(std::vector<Eigen::Isometry3d> knots, const Config& config)
  : knots_(std::move(knots)), config_(config) {}

  // Shared by the planner and the pre-planning replay so both run identical geometry. `targets`
  // are link poses in the state's model frame; the state's current link pose is the start.
  // On success `waypoints` holds the whole path including the start state. On failure
  // `failed_leg` is the 0-based index into `targets` of the leg that did not complete and
  // `detail` says why.
  static bool computeLegs(
    moveit::core::RobotState& state, const moveit::core::JointModelGroup* group,
    const moveit::core::LinkModel* link, const std::vector<Eigen::Isometry3d>& targets,
    const Config& config, const moveit::core::GroupStateValidityCallbackFn& valid,
    std::vector<moveit::core::RobotStatePtr>& waypoints, std::size_t& failed_leg,
    std::string& detail)
  {
    waypoints.clear();
    failed_leg = 0U;
    std::vector<double> previous;
    state.copyJointGroupPositions(group, previous);
    for (std::size_t leg = 0; leg < targets.size(); ++leg) {
      std::vector<moveit::core::RobotStatePtr> segment;
      const double fraction = moveit::core::CartesianInterpolator::computeCartesianPath(
        &state, group, segment, link, targets[leg], true,
        moveit::core::MaxEEFStep(config.step_m, config.rotation_step_rad),
        moveit::core::JumpThreshold(0.0), valid);
      if (fraction < 1.0 - 1e-9 || segment.empty()) {
        failed_leg = leg;
        detail = "leg " + std::to_string(leg) + " reached Cartesian fraction " +
          std::to_string(fraction) + " < 1.0";
        return false;
      }
      for (std::size_t index = 0; index < segment.size(); ++index) {
        if (leg > 0U && index == 0U) continue;  // duplicate of the previous leg's last state
        std::vector<double> q;
        segment[index]->copyJointGroupPositions(group, q);
        for (std::size_t j = 0; j < q.size() && j < previous.size(); ++j) {
          if (std::abs(q[j] - previous[j]) > config.max_joint_step_rad) {
            failed_leg = leg;
            detail = "leg " + std::to_string(leg) + " moves joint" + std::to_string(j + 1) + " " +
              std::to_string(std::abs(q[j] - previous[j])) + " rad between two waypoints";
            return false;
          }
        }
        previous = q;
        waypoints.push_back(segment[index]);
      }
      state.setJointGroupPositions(group, previous);
      state.update();
    }
    return true;
  }

  Result plan(
    const planning_scene::PlanningSceneConstPtr& from, const moveit::core::LinkModel& link,
    const Eigen::Isometry3d& offset, const Eigen::Isometry3d& target,
    const moveit::core::JointModelGroup* jmg, double /*timeout*/,
    robot_trajectory::RobotTrajectoryPtr& result,
    const moveit_msgs::msg::Constraints& /*path_constraints*/) override
  {
    if (knots_.size() < 2U) return Result{false, "waypoint stroke needs at least two knots"};
    const Eigen::Isometry3d frame = target * knots_.back().inverse();  // planning -> model frame
    std::vector<Eigen::Isometry3d> targets;
    for (std::size_t k = 1; k < knots_.size(); ++k) {
      targets.push_back(frame * knots_[k] * offset.inverse());
    }
    moveit::core::RobotState state(from->getCurrentState());
    state.update();
    const Eigen::Isometry3d start = frame * knots_.front() * offset.inverse();
    const Eigen::Isometry3d& reached = state.getGlobalLinkTransform(&link);
    if ((reached.translation() - start.translation()).norm() > 1e-3 ||
        Eigen::AngleAxisd(reached.linear().transpose() * start.linear()).angle() > 0.01) {
      return Result{false, "waypoint stroke does not start at its first knot"};
    }
    const auto scene = from->diff();
    const moveit::core::GroupStateValidityCallbackFn valid =
      [scene](moveit::core::RobotState* s, const moveit::core::JointModelGroup* group,
              const double* values) {
        s->setJointGroupPositions(group, values);
        s->update();
        return !scene->isStateColliding(
          const_cast<const moveit::core::RobotState&>(*s), group->getName());
      };
    std::vector<moveit::core::RobotStatePtr> waypoints;
    std::size_t failed_leg = 0U;
    std::string detail;
    if (!computeLegs(state, jmg, &link, targets, config_, valid, waypoints, failed_leg, detail)) {
      return Result{false, "waypoint stroke failed closed: " + detail};
    }
    result = std::make_shared<robot_trajectory::RobotTrajectory>(state.getRobotModel(), jmg);
    for (const auto& waypoint : waypoints) result->addSuffixWayPoint(waypoint, 0.0);
    const auto& props = properties();
    const auto timing =
      props.get<trajectory_processing::TimeParameterizationPtr>("time_parameterization");
    if (!timing || !timing->computeTimeStamps(
        *result, props.get<double>("max_velocity_scaling_factor"),
        props.get<double>("max_acceleration_scaling_factor"))) {
      return Result{false, "waypoint stroke time parameterisation failed"};
    }
    return Result{true, ""};
  }

  std::string getPlannerId() const override { return "WaypointCartesianPath"; }

private:
  std::vector<Eigen::Isometry3d> knots_;
  Config config_;
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
  // Left side cut built with start/end swapped (left_cut_reverse_travel); re-checked in
  // planCutSegments against every unsupported combination.
  bool reversed_travel{false};
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
struct LeftVaryingPlan;
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
  // Set only by planSnapshot for left_only/t_cut/h/combined: the sole path that may plan the
  // varying-orientation left cut (left_parallel_varying_orientation_enabled). Every other
  // caller (side-tape hand-off, cut-debug start, t_cut_handoff probes, staged top cut, ...)
  // leaves it false and planCutSegments then refuses the setting instead of ignoring it.
  bool allow_left_varying{false};
  // left_debug_stop_at_approach (planSnapshot only): the first (full) run returns the verified
  // varying plan here; the second, stop_at_hover_start run is handed it back through
  // reuse_varying (no second search) and reports the approach joints it must reach through
  // hover_start_joints_out.
  LeftVaryingPlan* varying_plan_out{nullptr};
  const LeftVaryingPlan* reuse_varying{nullptr};
  std::vector<double>* hover_start_joints_out{nullptr};
  // Receives the joint state at the end of the planned trajectory (last waypoint of the last
  // sub-trajectory) once every validator has passed.
  std::map<std::string, double>* final_joints_out{nullptr};
  // plan_h_cut only (return_home_after_cut): end the task with a free-space transit to the SRDF
  // "home" state (recorded for the wrist guard; collision-gated like every stage).
  bool return_home{false};
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
// One station of a varying-orientation left cut (searchLeftVaryingOrientation).
struct LeftVaryingStation {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};  // on the planned (cut-depth) cut line
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};  // the smoothed profile
  bool is_knot{false};  // a stroke knot (orientation = the DP candidate; tilt/pitch/yaw valid)
  double tilt_deg{0.0}, pitch_deg{0.0}, yaw_deg{0.0}, theta_p_deg{0.0}, blade_deviation_deg{0.0};
  std::size_t station_index{0};  // index within the FULL station list s_0..s_N
  std::vector<double> joints;    // the replayed stroke's IK branch at this station (group order)
};
struct LeftVaryingPlan {
  std::vector<LeftVaryingStation> stations;  // the planned interval [s_i, s_j] only
  std::size_t full_station_count{0};         // N + 1
  double full_length_m{0.0}, planned_length_m{0.0};
  // left_cut_extend_to_corners only: trim of the kept interval against the extended span.
  bool extended{false};
  double trim_back_mm{0.0}, trim_front_mm{0.0};
  double max_theta_p_deg{0.0}, max_blade_deviation_deg{0.0};
  // left_varying_start_extra_inward_m: horizontal inward offset (mm) applied to the first and
  // the last kept station (already contained in LeftVaryingStation::position).
  double inward_start_mm{0.0}, inward_finish_mm{0.0};
  std::vector<double> hover_start_joints;  // transit goal (group order): verified hover state
  std::string summary;  // one line for the plan response
};
// left_cut_extend_to_corners geometry handed to searchLeftVaryingOrientation. Arc positions are
// metres along the candidate line measured from its START (the back end of the extended span);
// the nominal segment starts nominal_start_m along it and is nominal_length_m long.
struct LeftExtension {
  double nominal_length_m{0.0};
  double nominal_start_m{0.0};     // arc position of back_middle_left on the extended line
  double seam_m{0.0};              // arc position of the seam crossing
  double band_lo_m{0.0}, band_hi_m{0.0};  // required seam band, clamped to the span
  double corner_margin_m{0.0}, seam_cover_m{0.0};
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
  double external_snapshot_max_age_s{0.0}, carton_height_m{0.0}, hover_clearance_m{0.0},
    left_lateral_standoff_m{0.0}, cut_clearance_m{0.0}, middle_cut_clearance_m{0.0},
    middle_entry_alignment_m{0.0}, debug_motion_distance_m{0.0},
    debug_motion_clearance_m{0.0}, cartesian_step_m{0.0}, planning_timeout_s{0.0},
    velocity_scaling{0.0}, acceleration_scaling{0.0}, transit_velocity_scaling{0.0},
    transit_acceleration_scaling{0.0};
  bool allow_tool_carton_contact{false}, camera_pole_required{true};
  // Links that get the scoped carton contact allowance during cut strokes (read-only). Empty (the
  // default) means {tool_link}: today's behaviour. A blade collision link of a per-arm model is listed
  // here so the exposed blade may touch the carton ONLY while a cut stage's allowance is installed.
  std::vector<std::string> carton_contact_links;
  std::vector<std::string> cartonContactLinks() const
  {
    return carton_contact_links.empty() ? std::vector<std::string>{tool_link} : carton_contact_links;
  }
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
  double t_cut_settle_timeout_s{5.0};
  double side_cut_post_penetration_dwell_s{1.0};
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
    left_parallel_search_timeout_s{0.0}, left_varying_station_spacing_m{0.0},
    left_varying_max_step_deg{0.0}, left_varying_min_cut_fraction{0.0},
    left_cut_corner_margin_m{0.0}, left_cut_seam_cover_m{0.0},
    left_varying_start_extra_inward_m{0.0}, left_cut_inward_m{0.0};
  bool debug_fixed_orientation{false}, debug_allow_stale_snapshot{false},
    cut_debug_correction_enabled{false}, left_adaptive_orientation_enabled{true},
    left_parallel_orientation_enabled{false}, left_cut_reverse_travel{false},
    left_parallel_varying_orientation_enabled{false}, left_cut_extend_to_corners{false},
    left_debug_stop_at_approach{false}, middle_cut_reverse_travel{false},
    return_home_after_cut{false};
  double middle_cut_tilt_deg{0.0};
  double middle_transit_extra_lift_m{0.0};
  // Dual-arm planning stack (all default to today's single-arm behaviour; see CONTEXT.md).
  // snapshot_frame_*: how the perception snapshot is brought into planning_frame (read-only).
  // middle_seam_stop / middle_seam_center_overlap_m: half-seam top cut (planning, middle_only).
  std::string snapshot_frame_mode{"planning_frame"}, snapshot_frame_id{"base_link"},
    snapshot_relabel{"none"}, middle_seam_stop{"end"};
  std::vector<double> snapshot_planning_base_xyz{0.0, 0.0, 0.0},
    snapshot_planning_base_rpy{0.0, 0.0, 0.0};
  double middle_seam_center_overlap_m{0.0};
  bool plan_export_only{false};
};
// ~/plan_h_cut_from_state: the planning group's joint values the plan starts from, instead of the
// live /joint_states. Set (RAII) only for the duration of one planExternalSnapshot call.
struct StartStateOverride {
  std::vector<std::string> names;    // the planning group's variable names, group order
  std::vector<double> positions;
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
    t_cut_status_pub_ = create_publisher<std_msgs::msg::String>(
      "~/t_cut_status", rclcpp::QoS(1).reliable().transient_local());
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
    // plan_export_only: only plan_h_cut, plan_h_cut_from_state, get_planned_solution, clear_plan and
    // the section diagnostics are advertised. The services below (side-tape, T cut, staged top cut,
    // cut debug) and execute_planned_h_cut are not created at all. The block body is deliberately
    // not re-indented, to keep the diff of this safety-relevant gate reviewable.
    if (!settings_.plan_export_only) {
    side_tape_handoff_plan_service_ = guardedTrigger(
      "~/plan_side_tape_handoff",
      "cannot plan while another planning or execution operation is active",
      [this](std::string& message) { return planSideTapeHandoff(message); });
    t_cut_plan_service_ = guardedTrigger(
      "~/plan_t_cut", "cannot plan while another planning or execution operation is active",
      [this](std::string& message) { return planTCut(message); });
    t_cut_execute_service_ = guardedTrigger(
      "~/execute_t_cut", "cannot start while another planning or execution operation is active",
      [this](std::string& message) { return executeTCut(message); });
    // Not guarded: the sequencer holds plan_mutex_ while it plans a step, and abort must never
    // wait for that. It only sets an atomic flag.
    t_cut_abort_service_ = create_service<std_srvs::srv::Trigger>(
      "~/abort_t_cut",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        if (!t_cut_running_.load()) {
          response->success = false;
          response->message = "no T cut sequence is running";
          return;
        }
        t_cut_abort_ = true;
        response->success = true;
        response->message = "T cut abort requested: no further step will start. A trajectory "
          "that is already executing is NOT cancelled by this node (no cancel path exists): "
          "use the e-stop if it must stop now, and side_tape_retract / top_cut_retract to "
          "recover once the sequence has ended";
      },
      rmw_qos_profile_services_default, callback_group_);
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
    }  // !plan_export_only
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
    if (!settings_.plan_export_only) {
      execute_service_ = guardedTrigger(
        "~/execute_planned_h_cut", "another H-cut plan operation is active",
        [this](std::string& message) { return executeRetainedPlan(message); },
        /*reject_during_execution=*/false);
    }
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
    // Plan export / plan from an explicit start state (dual-arm planning stack). Both are advertised
    // in plan_export_only mode as well. get_planned_solution only reads the retained plan.
    get_planned_solution_service_ = create_service<dual_cut_msgs::srv::GetPlannedSolution>(
      "~/get_planned_solution",
      [this](
        const std::shared_ptr<dual_cut_msgs::srv::GetPlannedSolution::Request>,
        std::shared_ptr<dual_cut_msgs::srv::GetPlannedSolution::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
          response->success = false;
          response->message = "busy: a planning or execution call holds the plan lock";
          return;
        }
        try {
          response->success = exportPlannedSolution(*response);
        } catch (const std::exception& exception) {
          response->success = false;
          response->message = std::string("export failed: ") + exception.what();
        }
      },
      rmw_qos_profile_services_default, callback_group_);
    plan_from_state_service_ = create_service<dual_cut_msgs::srv::PlanFromState>(
      "~/plan_h_cut_from_state",
      [this](
        const std::shared_ptr<dual_cut_msgs::srv::PlanFromState::Request> request,
        std::shared_ptr<dual_cut_msgs::srv::PlanFromState::Response> response) {
        std::unique_lock<std::mutex> lock(plan_mutex_, std::try_to_lock);
        response->success = false;
        response->plan_seq = 0;
        if (!lock.owns_lock() || execution_in_progress_) {
          response->message = "cannot plan while another planning or execution operation is active";
          return;
        }
        if (t_cut_running_.load()) {
          response->message = "a T cut sequence is running (execute_t_cut); only abort_t_cut is "
            "accepted until it finishes or stops";
          return;
        }
        try {
          response->success = planHCutFromState(request->start_state, response->message);
        } catch (const std::exception& exception) {
          response->success = false;
          response->message = std::string("plan_h_cut_from_state failed: ") + exception.what();
        }
        if (response->success) response->plan_seq = plan_seq_;
      },
      rmw_qos_profile_services_default, callback_group_);
    RCLCPP_INFO(
      get_logger(),
      "Snapshot H-cut MTC planner ready; hover %.0f mm, cut %.0f mm, pre-cut stabilization %.3f s",
      settings_.hover_clearance_m * 1000.0, settings_.cut_clearance_m * 1000.0,
      settings_.pre_cut_stabilization_duration_s);
  }

  // Startup refusal for a non-default carton_contact_links: every listed link must exist in the robot
  // model the planner loads (a misspelt blade link would otherwise silently keep the blade checked
  // against the carton during the cut strokes, or never be allowed). Returns the error text, or an
  // empty string if the list is the default (nothing to check: behaviour and start-up are unchanged)
  // or all links exist. Called from main() before the node spins (needs shared_from_this()).
  std::string checkCartonContactLinksInModel()
  {
    if (settings_.carton_contact_links.empty()) return std::string();
    mtc::Task probe;
    try {
      probe.loadRobotModel(shared_from_this());
    } catch (const std::exception& exception) {
      return std::string("carton_contact_links: cannot load the robot model to check the links: ") +
        exception.what();
    }
    const auto model = probe.getRobotModel();
    if (!model) return "carton_contact_links: the robot model did not load";
    for (const auto& link : settings_.carton_contact_links) {
      if (!model->hasLinkModel(link)) {
        return "carton_contact_links: link '" + link + "' is not in the robot model";
      }
    }
    return std::string();
  }

  ~BoxHCutExecutor() override
  {
    t_cut_abort_ = true;
    if (t_cut_thread_.joinable()) t_cut_thread_.join();
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
      field<std::vector<std::string>>(
        "carton_contact_links", kRO, std::vector<std::string>{}, &S::carton_contact_links),
      field<bool>("camera_pole_required", kRO, true, &S::camera_pole_required),
      field<str>(
        "camera_pole_topic", kRO, "/box_motion/camera_pole_collision", &S::camera_pole_topic),
      field<double>("carton_height_m", kRO, 0.20, &S::carton_height_m),
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
      // Opt-in (operator-approved): the planned LEFT side cut travels front_middle_left ->
      // back_middle_left instead of back -> front (blade still faces away from travel, so
      // blade +Y points toward the front). Default off: plans are unchanged. Fails closed
      // with left corrections, left_parallel, staged/hand-off paths, middle_aligned mode.
      field<bool>("left_cut_reverse_travel", kPlan, false, &S::left_cut_reverse_travel),
      // middle_only only (operator 2026-10-13, knife_c left arm): the top stroke runs
      // seam_end -> seam_start; with middle_blade_reversed the sharp edge leads. Shortening
      // pulls the START (the far seam_end side) back. Refused by every other middle path.
      field<bool>("middle_cut_reverse_travel", kPlan, false, &S::middle_cut_reverse_travel),
      // plan_h_cut: after the last cut, a free-space transit back to the SRDF "home" state
      // (operator 2026-10-15). Default false: plans are unchanged.
      field<bool>("return_home_after_cut", kPlan, false, &S::return_home_after_cut),
      // Top (middle seam) side lean: roll about the blade, same convention as
      // left/right_side_cut_tilt_deg (operator 2026-10-14: knife_c on the left arm cuts
      // seam_start -> seam_end with the sharp edge leading only with a side lean).
      field<double>("middle_cut_tilt_deg", kPlan, 0.0, &S::middle_cut_tilt_deg),
      // Half-seam top cut (middle_only only; refused by every other pattern): middle_seam_stop
      // "center" ends the stroke at the seam centre plus middle_seam_center_overlap_m ([0,0.03] m)
      // and disables the shortening loop (the half either plans or the plan fails). "end" (default)
      // is the unchanged full seam.
      field<str>("middle_seam_stop", kPlan, "end", &S::middle_seam_stop),
      field<double>(
        "middle_seam_center_overlap_m", kPlan, 0.0, &S::middle_seam_center_overlap_m),
      // Snapshot frame handling for an executor that plans for a DIFFERENT arm than the one whose
      // base frame the perception snapshot is written in (see snapshot_frame.hpp). Read-only; the
      // non-default values need plan_export_only=true (validateSettings).
      field<str>("snapshot_frame_mode", kRO, "planning_frame", &S::snapshot_frame_mode),
      field<str>("snapshot_frame_id", kRO, "base_link", &S::snapshot_frame_id),
      field<std::vector<double>>(
        "snapshot_planning_base_xyz", kRO, std::vector<double>{0.0, 0.0, 0.0},
        &S::snapshot_planning_base_xyz),
      field<std::vector<double>>(
        "snapshot_planning_base_rpy", kRO, std::vector<double>{0.0, 0.0, 0.0},
        &S::snapshot_planning_base_rpy),
      field<str>("snapshot_relabel", kRO, "none", &S::snapshot_relabel),
      // Advertise only the planning/export services (no execute, side-tape, staged, T-cut or
      // cut-debug service). Read-only.
      field<bool>("plan_export_only", kRO, false, &S::plan_export_only),
      // Left -> middle (t_cut/h/combined): after the left lift and the collision restore, raise
      // this much further straight up along the surface normal (Cartesian, collision-checked)
      // before the free-space transit, so the transit starts away from the carton and base.
      field<double>(
        "middle_transit_extra_lift_m", kPlan, 0.0, &S::middle_transit_extra_lift_m),
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
      // Opt-in (operator-approved): the left cut orientation may vary along the stroke
      // (see searchLeftVaryingOrientation). Requires left_parallel_orientation_enabled and
      // plan-only left_only/t_cut/h/combined planning; refused on every other path. Default
      // off: plans are unchanged.
      field<bool>(
        "left_parallel_varying_orientation_enabled", kPlan, false,
        &S::left_parallel_varying_orientation_enabled),
      field<double>(
        "left_varying_station_spacing_m", kPlan, 0.010, &S::left_varying_station_spacing_m),
      field<double>("left_varying_max_step_deg", kPlan, 5.0, &S::left_varying_max_step_deg),
      field<double>(
        "left_varying_min_cut_fraction", kPlan, 0.5, &S::left_varying_min_cut_fraction),
      // Operator-approved: push the blade this far further INTO the box (horizontally inward,
      // opposite to the lateral standoff direction) at the START of the kept varying left
      // stroke, tapering linearly to 0 at its finish. [0, 0.005] m; > 0 needs
      // left_parallel_varying_orientation_enabled (planning is refused otherwise). Default 0:
      // plans are unchanged.
      field<double>(
        "left_varying_start_extra_inward_m", kPlan, 0.0, &S::left_varying_start_extra_inward_m),
      // Constant horizontal inward offset of the WHOLE varying left stroke (same direction as
      // left_varying_start_extra_inward_m, added to it). [0, 0.008] m; > 0 needs
      // left_parallel_varying_orientation_enabled. Operator 2026-10-15: with the hover-corrected
      // calibration the blade ran exactly on the detected face and no longer cut the side.
      field<double>("left_cut_inward_m", kPlan, 0.0, &S::left_cut_inward_m),
      // Opt-in (operator-approved): the varying search runs over the whole left edge (back_left
      // + margin -> front_left - margin, on the nominal cut line) and keeps the longest
      // reachable interval that covers the seam crossing +/- left_cut_seam_cover_m. Requires
      // left_parallel_varying_orientation_enabled. Default off: plans are unchanged.
      field<bool>(
        "left_cut_extend_to_corners", kPlan, false, &S::left_cut_extend_to_corners),
      field<double>("left_cut_corner_margin_m", kPlan, 0.020, &S::left_cut_corner_margin_m),
      field<double>("left_cut_seam_cover_m", kPlan, 0.030, &S::left_cut_seam_cover_m),
      // Debug (operator-approved): with cut_pattern left_only, plan the full left cut as usual
      // but retain only the approach to the left hover pose (after the pre-cut dwell). Executing
      // that plan additionally requires debug_motion_execution_enabled. Default off.
      field<bool>(
        "left_debug_stop_at_approach", kPlan, false, &S::left_debug_stop_at_approach),
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
      field<double>("t_cut_settle_timeout_s", kRO, 5.0, &S::t_cut_settle_timeout_s),
      field<double>(
        "side_cut_post_penetration_dwell_s", kRO, 1.0, &S::side_cut_post_penetration_dwell_s),
      field<bool>("top_cut_allow_shortened_cut", kRO, true, &S::top_cut_allow_shortened_cut),
      field<double>(
        "top_cut_lift_min_clearance_m", kRO, 0.001, &S::top_cut_lift_min_clearance_m),
      field<double>(
        "side_tape_withdraw_min_clearance_m", kRO, 0.001,
        &S::side_tape_withdraw_min_clearance_m),
      field<double>(
        "top_cut_pre_transit_lift_min_fraction", kRO, 0.5,
        &S::top_cut_pre_transit_lift_min_fraction),
      // Planning parameter (was read-only) [0.05, 1.0]: operator 2026-10-15, the right arm's top cut
      // (tools/right_arm_single) keeps whatever it reaches from the right end, down to 5 %.
      field<double>("top_cut_min_cut_fraction", kPlan, 0.5, &S::top_cut_min_cut_fraction),
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
    const std::array<std::pair<const char*, double>, 33> doubles{{
      {"left_varying_start_extra_inward_m", s.left_varying_start_extra_inward_m},
      {"left_cut_inward_m", s.left_cut_inward_m},
      {"left_cut_corner_margin_m", s.left_cut_corner_margin_m},
      {"left_cut_seam_cover_m", s.left_cut_seam_cover_m},
      {"left_varying_station_spacing_m", s.left_varying_station_spacing_m},
      {"left_varying_max_step_deg", s.left_varying_max_step_deg},
      {"left_varying_min_cut_fraction", s.left_varying_min_cut_fraction},
      {"left_parallel_max_tilt_deg", s.left_parallel_max_tilt_deg},
      {"left_parallel_max_yaw_deg", s.left_parallel_max_yaw_deg},
      {"left_parallel_step_deg", s.left_parallel_step_deg},
      {"left_parallel_min_joint_margin_deg", s.left_parallel_min_joint_margin_deg},
      {"left_parallel_search_timeout_s", s.left_parallel_search_timeout_s},
      {"external_snapshot_max_age_s", s.external_snapshot_max_age_s},
      {"carton_height_m", s.carton_height_m}, {"hover_clearance_m", s.hover_clearance_m},
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
    for (std::size_t i = 0; i < s.carton_contact_links.size(); ++i) {
      const std::string& link = s.carton_contact_links[i];
      if (link.empty()) return std::string("carton_contact_links must not contain an empty name");
      if (link == s.carton_collision_id) {
        return std::string("carton_contact_links must not contain carton_collision_id '") + link + "'";
      }
      for (std::size_t j = 0; j < i; ++j) {
        if (s.carton_contact_links[j] == link) {
          return std::string("carton_contact_links lists '") + link + "' twice";
        }
      }
    }
    if (s.external_snapshot_max_age_s <= 0.0 || s.external_snapshot_max_age_s > kMaxSnapshotAgeS) {
      return "external_snapshot_max_age_s must be within (0," + formatDouble(kMaxSnapshotAgeS, 0) +
        "] seconds";
    }
    if (s.carton_height_m <= 0.0) return std::string("carton_height_m must be positive");
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
    if (!std::isfinite(s.middle_transit_extra_lift_m) || s.middle_transit_extra_lift_m < 0.0 ||
        s.middle_transit_extra_lift_m > 0.25) {
      return std::string("middle_transit_extra_lift_m must be finite and within [0,0.25] m");
    }
    if (!std::isfinite(s.middle_cut_tilt_deg) || std::abs(s.middle_cut_tilt_deg) > 30.0) {
      return std::string("middle_cut_tilt_deg must be finite and within [-30,30] degrees");
    }
    if (std::abs(s.left_side_cut_tilt_deg) > 90.0 || std::abs(s.right_side_cut_tilt_deg) > 90.0) {
      return std::string("side cut tilt must be finite and within [-90,90] degrees");
    }
    if (s.middle_seam_stop != "end" && s.middle_seam_stop != "center") {
      return std::string("middle_seam_stop must be end or center");
    }
    if (!std::isfinite(s.middle_seam_center_overlap_m) || !within(s.middle_seam_center_overlap_m, 0.0, 0.03)) {
      return std::string("middle_seam_center_overlap_m must be finite and within [0,0.03] m");
    }
    {
      const auto frame_mode = snapshot_frame::modeFromString(s.snapshot_frame_mode);
      const auto relabel = snapshot_frame::relabelFromString(s.snapshot_relabel);
      if (!frame_mode) {
        return std::string("snapshot_frame_mode must be planning_frame or fixed_transform");
      }
      if (!relabel) return std::string("snapshot_relabel must be none or rotate_180");
      if (s.snapshot_frame_id.empty()) return std::string("snapshot_frame_id must be non-empty");
      for (const auto* values : {&s.snapshot_planning_base_xyz, &s.snapshot_planning_base_rpy}) {
        if (values->size() != 3U ||
            !std::all_of(values->begin(), values->end(),
                         [](double value) { return std::isfinite(value); })) {
          return std::string("snapshot_planning_base_xyz and _rpy must be three finite values");
        }
      }
      const bool identity_base =
        std::all_of(s.snapshot_planning_base_xyz.begin(), s.snapshot_planning_base_xyz.end(),
                    [](double value) { return value == 0.0; }) &&
        std::all_of(s.snapshot_planning_base_rpy.begin(), s.snapshot_planning_base_rpy.end(),
                    [](double value) { return value == 0.0; });
      if (*frame_mode == snapshot_frame::Mode::kPlanningFrame && !identity_base) {
        return std::string(
          "snapshot_planning_base_xyz/_rpy are only used with snapshot_frame_mode fixed_transform; "
          "refusing a non-zero value that would be silently ignored");
      }
      const bool non_default_snapshot = *frame_mode != snapshot_frame::Mode::kPlanningFrame ||
        *relabel != snapshot_frame::Relabel::kNone;
      if (non_default_snapshot) {
        // The transformed/relabelled snapshot is for planning and export only: nothing that
        // moves a robot, and nothing that carries left-arm data, may be reachable.
        if (!s.plan_export_only) {
          return std::string(
            "snapshot_frame_mode fixed_transform / snapshot_relabel rotate_180 require "
            "plan_export_only=true");
        }
        if (s.cut_debug_correction_enabled) {
          return std::string(
            "taught cut-debug corrections are left-arm data: cut_debug_correction_enabled must be "
            "false with snapshot_frame_mode fixed_transform / snapshot_relabel rotate_180");
        }
        if (s.side_tape_handoff_enabled) {
          return std::string(
            "side_tape_handoff_enabled must be false with snapshot_frame_mode fixed_transform / "
            "snapshot_relabel rotate_180 (the side-tape seed is not transformed)");
        }
      }
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
    if (!within(s.left_varying_station_spacing_m, 0.005, 0.030) ||
        !within(s.left_varying_max_step_deg, 1.0, 15.0) ||
        !within(s.left_varying_min_cut_fraction, 0.2, 1.0)) {
      return std::string(
        "left varying orientation bounds: station_spacing within [0.005,0.030] m, max_step "
        "within [1,15] deg, min_cut_fraction within [0.2,1.0]");
    }
    if (!within(s.left_varying_start_extra_inward_m, 0.0, 0.005)) {
      return std::string("left_varying_start_extra_inward_m must be within [0,0.005] m");
    }
    if (!within(s.left_cut_inward_m, 0.0, 0.008)) {
      return std::string("left_cut_inward_m must be within [0,0.008] m");
    }
    if (!within(s.top_cut_min_cut_fraction, 0.05, 1.0)) {
      return std::string("top_cut_min_cut_fraction must be within [0.05,1.0]");
    }
    if (!within(s.left_cut_corner_margin_m, 0.005, 0.10) ||
        !within(s.left_cut_seam_cover_m, 0.0, 0.10)) {
      return std::string(
        "left cut extension bounds: left_cut_corner_margin_m within [0.005,0.10] m, "
        "left_cut_seam_cover_m within [0,0.10] m");
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
      if (execution_in_progress_ || t_cut_running_.load()) {
        result.reason =
          "busy: planning/execution active; cannot change planning parameters during trajectory execution or a T cut sequence";
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
        // While a T cut sequence runs (it releases plan_mutex_ between steps) every other
        // service refuses; only abort_t_cut is accepted.
        if (t_cut_running_.load()) {
          response->success = false;
          response->message = "a T cut sequence is running (execute_t_cut); only abort_t_cut is "
            "accepted until it finishes or stops";
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
    if (retained_from_override_) {
      message = std::string(in_debug_session ? "cut debug session" : "retained H-cut plan") +
        " execution is blocked: it was planned from an explicit start state "
        "(plan_h_cut_from_state), not from the robot's current state; call clear_plan and plan "
        "with plan_h_cut";
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
    if (retained_debug_motion_ && !debug_motion_execution_enabled_.load()) {
      message = "the retained plan is an approach-only left debug plan "
        "(left_debug_stop_at_approach): its execution requires both execution_enabled and "
        "debug_motion_execution_enabled";
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
    const bool dispatched_is_left_debug_approach = retained_debug_motion_;
    retained_debug_motion_ = false;
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
      message = dispatched_is_left_debug_approach ?
        "left debug APPROACH-ONLY trajectory dispatch accepted (stops at the left hover pose); "
        "monitor controller execution" :
        "H-cut trajectory dispatch accepted; monitor controller execution";
    }
    return true;
  }
  void clearRetainedPlan()
  {
    planned_task_.reset();
    planned_snapshot_.reset();
    retained_with_stale_override_ = false;
    retained_from_override_ = false;
    exportable_task_ = nullptr;
    planned_cut_targets_.clear();
    planned_segments_.clear();
    retained_summary_.clear();
    retained_cut_pattern_.clear();
    retained_middle_fraction_ = 1.0;
    // A retained-but-not-yet-dispatched hand-off plan must never survive into (be mistaken
    // for) a later, unrelated plan's dispatch: always consumed here, regardless of whether a
    // hand-off is currently in progress.
    side_tape_handoff_pending_ = false;
    top_cut_pending_ = false;
    retained_debug_motion_ = false;
    t_cut_plan_ready_ = false;
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
    if (planned) retained_summary_ = message;
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
  // Left side cut as used by the normal planning path (segmentsForPattern, planHandoffAt).
  // With left_cut_reverse_travel the stroke runs front_middle_left -> back_middle_left.
  // segmentFor() itself stays back -> front: cutDebugBasis() (correction-record bases),
  // the side-tape hand-off and its retract/parking pose all depend on that.
  CutSegment leftPlanningSegment(const Snapshot& snapshot) const
  {
    CutSegment segment = segmentFor(snapshot, CutKind::kLeft);
    if (settings_.left_cut_reverse_travel) {
      std::swap(segment.start, segment.end);
      segment.reversed_travel = true;
    }
    return segment;
  }
  // Empty when left_cut_reverse_travel can be combined with the current settings/state,
  // otherwise the reason it is refused. Every combination listed here assumes back -> front
  // left travel (or has not been reviewed for reversed travel).
  std::string leftReverseTravelConflict() const
  {
    if (!settings_.left_cut_reverse_travel) return std::string();
    if (settings_.cut_debug_correction_enabled &&
        !settings_.cut_debug_left_correction_file.empty()) {
      return "left_cut_reverse_travel cannot be combined with an active left correction record "
        "(cut_debug_correction_enabled with cut_debug_left_correction_file); taught corrections "
        "assume back->front travel";
    }
    if (settings_.left_parallel_orientation_enabled) {
      return "left_cut_reverse_travel cannot be combined with left_parallel_orientation_enabled "
        "(the left_parallel search and its tool-frame convention assume back->front travel)";
    }
    if (settings_.side_cut_orientation_mode != "edge_aligned") {
      return "left_cut_reverse_travel requires side_cut_orientation_mode edge_aligned";
    }
    if (cut_debug_stage_ != CutDebugStage::kNone) {
      return "left_cut_reverse_travel cannot be combined with a staged cut-debug session";
    }
    if (side_tape_handoff_info_.inProgress()) {
      return "left_cut_reverse_travel cannot be combined with an active side-tape hand-off";
    }
    return std::string();
  }
  // middle_seam_stop=center (half-seam top cut) is implemented only by plan_h_cut with cut_pattern
  // middle_only (planSnapshot). Every other planning entry point would silently plan the full seam,
  // so each one calls this and refuses instead.
  bool refuseHalfSeam(const char* what, std::string& message) const
  {
    if (settings_.middle_seam_stop == "end") return false;
    message = std::string(what) + " refused: middle_seam_stop is '" + settings_.middle_seam_stop +
      "' (half-seam top cut), which only plan_h_cut with cut_pattern middle_only supports; set "
      "middle_seam_stop end first";
    return true;
  }
  // For the staged / side-tape hand-off entry points, which are all back -> front only.
  bool refuseLeftReverseTravel(const char* what, std::string& message) const
  {
    if (!settings_.left_cut_reverse_travel) return false;
    message = std::string(what) + " refused: left_cut_reverse_travel is true and this path "
      "assumes back_middle_left -> front_middle_left travel; set left_cut_reverse_travel false "
      "first";
    return true;
  }
  // Empty when left_parallel_varying_orientation_enabled can plan `pattern` with the current
  // settings/state, otherwise the reason it is refused. Only left_only/t_cut/h/combined
  // planned through planSnapshot may use it.
  std::string leftVaryingConflict(const std::string& pattern) const
  {
    if (settings_.left_cut_extend_to_corners &&
        !settings_.left_parallel_varying_orientation_enabled) {
      return "left_cut_extend_to_corners requires left_parallel_varying_orientation_enabled=true "
        "(it extends the varying-orientation left search to the corners); set both or neither";
    }
    if (!settings_.left_parallel_varying_orientation_enabled) {
      if (settings_.left_varying_start_extra_inward_m > 0.0) {
        return "left_varying_start_extra_inward_m is " +
          formatDouble(settings_.left_varying_start_extra_inward_m * 1000.0, 1) +
          " mm but left_parallel_varying_orientation_enabled is false (the inward start offset "
          "exists only for the varying-orientation left cut); enable it or set "
          "left_varying_start_extra_inward_m to 0";
      }
      if (settings_.left_cut_inward_m > 0.0) {
        return "left_cut_inward_m is " + formatDouble(settings_.left_cut_inward_m * 1000.0, 1) +
          " mm but left_parallel_varying_orientation_enabled is false (the inward offset exists "
          "only for the varying-orientation left cut); enable it or set left_cut_inward_m to 0";
      }
      return std::string();
    }
    if (!settings_.left_parallel_orientation_enabled) {
      return "left_parallel_varying_orientation_enabled requires "
        "left_parallel_orientation_enabled=true (it varies the left_parallel orientation along "
        "the stroke); set both or neither";
    }
    if (settings_.left_cut_reverse_travel) {
      return "left_parallel_varying_orientation_enabled cannot be combined with "
        "left_cut_reverse_travel (the varying profile assumes back_middle_left -> "
        "front_middle_left travel); set left_cut_reverse_travel false";
    }
    if (settings_.cut_debug_correction_enabled &&
        !settings_.cut_debug_left_correction_file.empty()) {
      return "left_parallel_varying_orientation_enabled cannot be combined with an active left "
        "correction record (cut_debug_correction_enabled with cut_debug_left_correction_file)";
    }
    if (settings_.side_cut_orientation_mode != "edge_aligned") {
      return "left_parallel_varying_orientation_enabled requires side_cut_orientation_mode "
        "edge_aligned";
    }
    if (settings_.left_search_ik_backend != "analytic") {
      return "left_parallel_varying_orientation_enabled requires left_search_ik_backend analytic";
    }
    if (settings_.motion_profile != "ompl_cartesian") {
      return "left_parallel_varying_orientation_enabled supports only motion_profile "
        "ompl_cartesian (its single contact stroke is a CartesianInterpolator path)";
    }
    if (cut_debug_stage_ != CutDebugStage::kNone) {
      return "left_parallel_varying_orientation_enabled cannot be combined with a staged "
        "cut-debug session";
    }
    if (side_tape_handoff_info_.inProgress()) {
      return "left_parallel_varying_orientation_enabled cannot be combined with an active "
        "side-tape hand-off";
    }
    if (pattern == "t_cut_handoff") {
      return "left_parallel_varying_orientation_enabled is not supported by cut_pattern "
        "t_cut_handoff (use left_only, t_cut, h or combined)";
    }
    return std::string();
  }
  // left_cut_extend_to_corners geometry. `segment` is the NOMINAL left segment
  // (back_middle_left -> front_middle_left). With A = segment.start, B = segment.end,
  // L = |B - A|, u = (B - A) / L and s(P) = (P - A) . u (arc position on the nominal line):
  //   s_start = s(back_left) + left_cut_corner_margin_m      (back corner moved toward front)
  //   s_end   = s(front_left) - left_cut_corner_margin_m     (front corner moved toward back)
  //   extended start/end points = A + s_start * u, A + s_end * u   (the SAME line as the nominal
  //   segment, extended in both directions; the caller then applies the usual cut_clearance
  //   offset along the surface normal via targetPose()).
  // The seam band is s(seam_start) +/- left_cut_seam_cover_m clamped to [s_start, s_end].
  // Fails closed on a degenerate/short span (see left_varying::computeExtendedSpan) and when a
  // corner or the seam crossing is more than 30 mm off the nominal line.
  bool computeLeftExtension(
    const Snapshot& snapshot, const CutSegment& segment, LeftExtension& geometry,
    geometry_msgs::msg::Point& extended_start_point, geometry_msgs::msg::Point& extended_end_point,
    std::string& error) const
  {
    const Eigen::Vector3d origin = toEigen(segment.start);
    const Eigen::Vector3d nominal = toEigen(segment.end) - origin;
    const double nominal_length = nominal.norm();
    if (!std::isfinite(nominal_length) || nominal_length < 1e-6) {
      error = "left_cut_extend_to_corners: the nominal left segment is degenerate";
      return false;
    }
    const Eigen::Vector3d direction = nominal / nominal_length;
    constexpr double kMaxOffLineM = 0.030;
    const auto arc = [&](const geometry_msgs::msg::Point& point, const char* label, double& s) {
        const Eigen::Vector3d relative = toEigen(point) - origin;
        s = relative.dot(direction);
        const double off_line = (relative - s * direction).norm();
        if (!std::isfinite(s) || !std::isfinite(off_line) || off_line > kMaxOffLineM) {
          error = std::string("left_cut_extend_to_corners: ") + label + " is " +
            formatDouble(off_line * 1000.0, 1) + " mm off the nominal left cut line (max " +
            formatDouble(kMaxOffLineM * 1000.0, 0) + " mm)";
          return false;
        }
        return true;
      };
    double s_back = 0.0, s_front = 0.0, s_seam = 0.0;
    if (!arc(snapshot.back_left, "back_left", s_back) ||
        !arc(snapshot.front_left, "front_left", s_front) ||
        !arc(snapshot.seam_start, "seam_start", s_seam)) {
      return false;
    }
    const auto span = left_varying::computeExtendedSpan(
      nominal_length, s_back, s_front, s_seam, settings_.left_cut_corner_margin_m,
      settings_.left_cut_seam_cover_m);
    if (!span.ok) {
      error = "left_cut_extend_to_corners refused: " + span.error;
      return false;
    }
    extended_start_point = toPoint(origin + span.s_start * direction);
    extended_end_point = toPoint(origin + span.s_end * direction);
    geometry.nominal_length_m = nominal_length;
    geometry.nominal_start_m = -span.s_start;
    geometry.seam_m = s_seam - span.s_start;
    geometry.band_lo_m = span.band_lo - span.s_start;
    geometry.band_hi_m = span.band_hi - span.s_start;
    geometry.corner_margin_m = settings_.left_cut_corner_margin_m;
    geometry.seam_cover_m = settings_.left_cut_seam_cover_m;
    return true;
  }
  // For the staged / side-tape hand-off / continue entry points, which keep their fixed-
  // orientation behaviour and must not be silently affected by the varying-orientation mode.
  bool refuseLeftVarying(const char* what, std::string& message) const
  {
    if (settings_.left_cut_extend_to_corners) {
      message = std::string(what) + " refused: left_cut_extend_to_corners is true and this path "
        "keeps the nominal left cut segment; set it false first";
      return true;
    }
    if (settings_.left_debug_stop_at_approach) {
      message = std::string(what) + " refused: left_debug_stop_at_approach is true and this path "
        "does not implement the approach-only debug plan (only plan_h_cut with cut_pattern "
        "left_only does); set it false first";
      return true;
    }
    if (!settings_.left_parallel_varying_orientation_enabled) {
      if (settings_.left_varying_start_extra_inward_m > 0.0) {
        message = std::string(what) + " refused: left_varying_start_extra_inward_m is " +
          formatDouble(settings_.left_varying_start_extra_inward_m * 1000.0, 1) +
          " mm and this path does not apply it (it exists only for the varying-orientation left "
          "cut); set left_varying_start_extra_inward_m to 0 first";
        return true;
      }
      if (settings_.left_cut_inward_m > 0.0) {
        message = std::string(what) + " refused: left_cut_inward_m is " +
          formatDouble(settings_.left_cut_inward_m * 1000.0, 1) +
          " mm and this path does not apply it (it exists only for the varying-orientation left "
          "cut); set left_cut_inward_m to 0 first";
        return true;
      }
      return false;
    }
    message = std::string(what) + " refused: left_parallel_varying_orientation_enabled is true "
      "and this path keeps its fixed left orientation; set it false first";
    return true;
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
        setCartonContactEntries(scene.getAllowedCollisionMatrixNonConst(), contact_allowed);
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
        setCartonContactEntries(
          scene.getAllowedCollisionMatrixNonConst(),
          index % 2U != 0U && settings_.allow_tool_carton_contact);
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
  //      constant (including a changed arm chain or mount the library does not know about),
  //      because it does not go through rebel_analytic_ik at all on the "expected" side. The
  //      link6 -> tool_link offset fed to the library is derived from the live RobotModel (all
  //      joints between the two must be fixed), so both tool variants are supported. Also
  //      fails closed immediately, before either numeric check, if that chain is not all-fixed
  //      or if the robot model has no base_link / planning_frame is not base_link (which every
  //      rebel_analytic_ik arm constant is expressed relative to).
  void ensureAnalyticIkSelfCheck(const moveit::core::JointModelGroup* joint_group)
  {
    std::call_once(analytic_self_check_once_, [this, joint_group]() {
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
        // link6 -> tool_link, derived from the live RobotModel (the tool variant -- e.g.
        // knife_c with its -30 deg pitch, or xeg32_knife_v1 -- is chosen by the URDF xacro
        // arg, so nothing about the tool is hardcoded in rebel_analytic_ik). Every joint
        // between link6 and tool_link must be FIXED, else the offset is not a constant.
        {
          const moveit::core::LinkModel* link6_model = robot_model_->getLinkModel("link6", nullptr);
          const moveit::core::LinkModel* tool_model =
            robot_model_->getLinkModel(settings_.tool_link, nullptr);
          if (link6_model == nullptr || tool_model == nullptr) {
            RCLCPP_ERROR(
              get_logger(),
              "Analytic IK refused: robot model has no link '%s' and/or 'link6'; disabling the "
              "analytic left-search backend for this process and falling back to kdl "
              "regardless of left_search_ik_backend",
              settings_.tool_link.c_str());
            analytic_ik_available_.store(false);
            return;
          }
          const moveit::core::LinkModel* walk = tool_model;
          bool chain_ok = true;
          std::string chain_detail;
          while (walk != link6_model) {
            const moveit::core::JointModel* parent_joint = walk->getParentJointModel();
            const moveit::core::LinkModel* parent_link =
              parent_joint != nullptr ? parent_joint->getParentLinkModel() : nullptr;
            if (parent_joint == nullptr || parent_link == nullptr) {
              chain_ok = false;
              chain_detail = "link6 is not an ancestor of '" + settings_.tool_link + "'";
              break;
            }
            if (parent_joint->getType() != moveit::core::JointModel::FIXED) {
              chain_ok = false;
              chain_detail = "joint '" + parent_joint->getName() + "' between link6 and '" +
                settings_.tool_link + "' is not fixed";
              break;
            }
            walk = parent_link;
          }
          if (!chain_ok) {
            RCLCPP_ERROR(
              get_logger(),
              "Analytic IK refused: %s; disabling the analytic left-search backend for this "
              "process and falling back to kdl regardless of left_search_ik_backend",
              chain_detail.c_str());
            analytic_ik_available_.store(false);
            return;
          }
          moveit::core::RobotState offset_state(robot_model_);
          offset_state.setToDefaultValues();
          offset_state.update();
          analytic_tool_offset_ = offset_state.getGlobalLinkTransform(link6_model).inverse() *
                                  offset_state.getGlobalLinkTransform(tool_model);
          // URDF rpy convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
          const Eigen::Matrix3d& r_tool = analytic_tool_offset_.linear();
          const Eigen::Vector3d rpy_deg =
            Eigen::Vector3d(
              std::atan2(r_tool(2, 1), r_tool(2, 2)),
              std::atan2(-r_tool(2, 0), std::hypot(r_tool(2, 1), r_tool(2, 2))),
              std::atan2(r_tool(1, 0), r_tool(0, 0))) *
            (180.0 / M_PI);
          RCLCPP_INFO(
            get_logger(),
            "Analytic IK tool offset link6 -> %s (derived from live RobotModel): xyz = "
            "(%.3f, %.3f, %.3f) mm, rpy = (%.3f, %.3f, %.3f) deg",
            settings_.tool_link.c_str(), analytic_tool_offset_.translation().x() * 1000.0,
            analytic_tool_offset_.translation().y() * 1000.0,
            analytic_tool_offset_.translation().z() * 1000.0, rpy_deg.x(), rpy_deg.y(),
            rpy_deg.z());
        }
        const auto library_result = rebel_analytic_ik::selfCheck(
          2000, 1e-6, 1e-6, 20260928U, analytic_tool_offset_, limits);
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
          const Eigen::Isometry3d analytic_fk = rebel_analytic_ik::forwardKinematics(q, analytic_tool_offset_);
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
  std::vector<std::vector<double> > analyticIkBranches(
    const moveit::core::JointModelGroup* group, const Eigen::Isometry3d& target) const
  {
    std::array<rebel_analytic_ik::JointLimits, 6> limits;
    std::vector<std::vector<double> > branches;
    if (!analyticJointLimitsFor(group, limits)) return branches;
    const auto index = analyticVariableIndex(group);
    const auto& names = group->getVariableNames();
    for (const auto& solution : rebel_analytic_ik::solve(target, analytic_tool_offset_, limits)) {
      std::vector<double> values(names.size());
      for (std::size_t j = 0; j < 6U; ++j) values[index[j]] = solution[j];
      branches.push_back(std::move(values));
    }
    return branches;
  }
  // Orientation candidates of the left_parallel search (see the comment above
  // searchLeftParallelOrientation): R = Rot(n, yaw) * Rot(across, pitch) * Rot(along, tilt) * F0
  // on the step grid, penetration-gated, in the deterministic search order (theta_p, blade
  // deviation, total angle; or smallest rotation from F0 first in side-tape hand-off mode).
  // Shared by the fixed search and the varying-orientation search so both use the SAME set.
  struct LeftParallelCandidate {
    double tilt, pitch, yaw, theta_p, blade_deviation, total, rotation;
    Eigen::Quaterniond orientation;
  };
  std::vector<LeftParallelCandidate> buildLeftParallelCandidates(
    const CutBasis& basis, bool handoff_mode) const
  {
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    using Candidate = LeftParallelCandidate;
    const Eigen::Vector3d into_unit = -basis.normal;
    Eigen::Matrix3d reference;
    reference.col(0) = -basis.normal;
    reference.col(1) = -basis.along;
    reference.col(2) = reference.col(0).cross(reference.col(1));
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
    return candidates;
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
    using Candidate = LeftParallelCandidate;
    const std::vector<Candidate> candidates = buildLeftParallelCandidates(basis, handoff_mode);
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
        setCartonContactEntries(scene.getAllowedCollisionMatrixNonConst(), false);
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
          setCartonContactEntries(
            scene.getAllowedCollisionMatrixNonConst(), settings_.allow_tool_carton_contact);
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
  // Existing aggregate joint-travel bound for ONE left contact stroke (entry, cut, finish, lift);
  // the varying cut is a single stroke, so it applies unchanged (validateLeftContactJointTravel).
  static constexpr double kMaxLeftContactJointTravelRad = 1.0;
  // Grid points the varying search accepts (its neighbour table is quadratic in this).
  static constexpr std::size_t kMaxVaryingCandidates = 12000;
  // Opt-in (left_parallel_varying_orientation_enabled) LEFT side cut whose tool orientation
  // may change along the stroke. The fixed left_parallel search (above) needs ONE orientation
  // that carries the whole stroke; where the holder collides at one end that forces a large
  // blade-plane angle theta_p. Here the cut line is sampled at stations s_0..s_N
  // (left_varying_station_spacing_m, both end points included, at the cut depth already in
  // `cut_start`/`cut_end`) and, per station, the SAME orientation candidates as the fixed
  // search (buildLeftParallelCandidates: penetration gate included) are evaluated with the
  // analytic IK (all branches), the bounds, the left_parallel_min_joint_margin_deg joint
  // margin and the collision check under the scoped tool/carton allowance. A dynamic program
  // (left_varying::ChainDp) then picks, per station of one CONTIGUOUS interval [s_i, s_j], one
  // (candidate, IK branch) such that consecutive stations differ by at most
  // left_varying_max_step_deg of tool rotation and by at most kMaxStationJointStepRad on every
  // joint (a joint4/joint6 wrap is therefore impossible). It maximises the interval length,
  // then minimises the summed per-candidate score (theta_p dominant, as in the fixed search's
  // ordering) plus a small smoothness term. Both ends must also be enterable/leavable:
  //   - the interval start needs the hover (lateral standoff) pose to have a collision-free IK
  //     branch that continues to the station branch (joint6 change from `transit_seed` at most
  //     pi); the caller then gives the transit that branch as a JOINT goal (see hover_for);
  //   - the end needs the hover-end pose (and the lift pose when a middle cut follows) to have
  //     a continuing collision-free branch.
  // The chosen profile is finally replayed with the Cartesian interpolator exactly as the MTC
  // strokes will run (entry, one slerp leg per station, finish, lift; fraction 1.0, no joint
  // step above 5 deg between waypoints, bounds, margin, collisions); a failing leg is banned
  // and the DP re-solved (bounded). An interval shorter than left_varying_min_cut_fraction
  // of the cut fails closed. MTC, min_fraction=1.0, the dwell check, the contact joint-travel
  // validators and the transit wrist guard remain the final authority.
  //
  // left_varying_start_extra_inward_m (> 0 only): every station position is moved horizontally
  // INTO the box (opposite to offsetLeftPoseOutward) by extra * (1 - (s - s_lo) / (s_hi - s_lo))
  // (clamped to [0, extra]), where [s_lo, s_hi] is the taper window of one search pass. The
  // offset is applied to the station positions themselves, so the IK/collision feasibility
  // search, the DP, the hover/lift poses, the Cartesian replay, the plan's stations and
  // validateLeftVaryingRetainedStroke all use the SAME offset positions. The intended window is
  // the KEPT interval, which is only known after the DP: searchLeftVaryingOrientation therefore
  // runs a first pass over the whole searched span and, if the kept interval is shorter, re-runs
  // the complete search with the window set to that kept interval, until a pass keeps exactly the
  // interval its window was defined on (bounded; fails closed otherwise). Nothing is ever
  // executed at positions that were not themselves IK/collision/replay verified.
  bool searchLeftVaryingOrientation(
    const Snapshot& snapshot, const moveit::core::JointModelGroup* joint_group,
    const moveit::core::RobotState& seed_state,
    const geometry_msgs::msg::PoseStamped& cut_start,
    const geometry_msgs::msg::PoseStamped& cut_end, const std::vector<double>& transit_seed,
    bool lift_follows, LeftVaryingPlan& plan, std::string& error,
    const LeftExtension* extension = nullptr)
  {
    const double extra_m = settings_.left_varying_start_extra_inward_m;
    if (!(extra_m > 0.0)) {
      return searchLeftVaryingOrientationPass(
        snapshot, joint_group, seed_state, cut_start, cut_end, transit_seed, lift_follows, plan,
        error, extension, 0.0, 0.0, 0.0);
    }
    constexpr int kMaxTaperPasses = 3;
    const double full_length_m =
      (toEigen(cut_end.pose.position) - toEigen(cut_start.pose.position)).norm();
    double window_lo_m = 0.0;
    double window_hi_m = full_length_m;
    std::string history;
    for (int pass = 1; pass <= kMaxTaperPasses; ++pass) {
      if (!searchLeftVaryingOrientationPass(
          snapshot, joint_group, seed_state, cut_start, cut_end, transit_seed, lift_follows, plan,
          error, extension, extra_m, window_lo_m, window_hi_m)) {
        if (pass > 1) {
          error += " (taper pass " + std::to_string(pass) + " of left_varying_start_extra_inward_m)";
        }
        return false;
      }
      const double spacing_m = plan.full_length_m / static_cast<double>(plan.full_station_count - 1U);
      const double kept_lo_m = static_cast<double>(plan.stations.front().station_index) * spacing_m;
      const double kept_hi_m = static_cast<double>(plan.stations.back().station_index) * spacing_m;
      if (std::abs(kept_lo_m - window_lo_m) < 1e-6 && std::abs(kept_hi_m - window_hi_m) < 1e-6) {
        return true;
      }
      history += (history.empty() ? "" : ", ") + formatDouble(kept_lo_m * 1000.0, 1) + ".." +
        formatDouble(kept_hi_m * 1000.0, 1) + " mm";
      RCLCPP_INFO(
        get_logger(),
        "left_varying_start_extra_inward_m: pass %d kept %.1f..%.1f mm but the taper window was "
        "%.1f..%.1f mm; re-planning with the window on the kept interval",
        pass, kept_lo_m * 1000.0, kept_hi_m * 1000.0, window_lo_m * 1000.0, window_hi_m * 1000.0);
      window_lo_m = kept_lo_m;
      window_hi_m = kept_hi_m;
    }
    plan = LeftVaryingPlan{};
    error = "left_varying_start_extra_inward_m: the kept left interval did not settle on the "
      "interval the inward taper is defined over after " + std::to_string(kMaxTaperPasses) +
      " passes (kept " + history + "); failing closed. Set left_varying_start_extra_inward_m 0 "
      "or retry";
    return false;
  }
  // One search pass; see searchLeftVaryingOrientation. `inward_extra_m` 0 = no inward offset.
  bool searchLeftVaryingOrientationPass(
    const Snapshot& snapshot, const moveit::core::JointModelGroup* joint_group,
    const moveit::core::RobotState& seed_state,
    const geometry_msgs::msg::PoseStamped& cut_start,
    const geometry_msgs::msg::PoseStamped& cut_end, const std::vector<double>& transit_seed,
    bool lift_follows, LeftVaryingPlan& plan, std::string& error,
    const LeftExtension* extension, double inward_extra_m, double taper_lo_m, double taper_hi_m)
  {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kDeg = kPi / 180.0;
    constexpr double kMaxStrokeJointStepRad = 5.0 * kDeg;  // same gate as the fixed search
    constexpr double kMaxStationJointStepRad = 0.2;
    constexpr double kDeviationWeight = 0.01;
    constexpr double kTotalWeight = 1e-4;
    constexpr double kSmoothWeight = 0.01;
    constexpr int kMaxRepairs = 500;  // ~45 ms each; bounded by the search timeout
    constexpr double kStrokeRotationStepRad = 0.05;  // finer than MoveIt's default 0.1 rad
    plan = LeftVaryingPlan{};
    const auto started = std::chrono::steady_clock::now();
    const auto elapsed_s = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      };
    const auto out_of_time = [&]() { return elapsed_s() > settings_.left_parallel_search_timeout_s; };
    std::string timeout_hint;  // extended span only: the search covers more stations
    const auto timeout_error = [&](const std::string& where) {
        error = "left varying orientation search timed out " + where + " after " +
          formatDouble(elapsed_s(), 1) + " s (left_parallel_search_timeout_s " +
          formatDouble(settings_.left_parallel_search_timeout_s, 1) + ")" + timeout_hint +
          "; failing closed";
        return false;
      };
    CutBasis basis;
    if (!cutDebugBasis(snapshot, "left", basis, error)) return false;
    const Eigen::Vector3d into_unit = -basis.normal;
    const auto* tool = seed_state.getRobotModel()->getLinkModel(settings_.tool_link);
    if (tool == nullptr) {
      error = "left varying orientation search has no tool link";
      return false;
    }
    ensureAnalyticIkSelfCheck(joint_group);
    if (settings_.left_search_ik_backend != "analytic" || !analytic_ik_available_.load()) {
      error = "left_parallel_varying_orientation_enabled requires left_search_ik_backend "
        "'analytic' with a passing analytic IK self-check (it enumerates every IK branch)";
      return false;
    }
    // Two scenes: `free_scene` forbids tool/carton contact (hover poses, like the fixed
    // search's first pose), `contact_scene` applies the scoped allowance (cut strokes, finish
    // and lift: exactly the ACM state of the MTC stages).
    const auto make_scene = [&](bool allow_contact) -> std::unique_ptr<planning_scene::PlanningScene> {
        auto scene = std::make_unique<planning_scene::PlanningScene>(seed_state.getRobotModel());
        if (!scene->processCollisionObjectMsg(cartonCollisionObject(snapshot))) {
          error = "cannot add carton to left varying orientation probe";
          return nullptr;
        }
        {
          std::lock_guard<std::mutex> lock(camera_pole_mutex_);
          if (camera_pole_) {
            if (!scene->processCollisionObjectMsg(*camera_pole_)) {
              error = "cannot add camera pole to left varying orientation probe";
              return nullptr;
            }
          } else if (settings_.camera_pole_required) {
            error = "camera pole collision object is unavailable for left varying orientation probe";
            return nullptr;
          }
        }
        setCartonContactEntries(
          scene->getAllowedCollisionMatrixNonConst(), allow_contact && settings_.allow_tool_carton_contact);
        return scene;
      };
    const auto free_scene = make_scene(false);
    if (!free_scene) return false;
    const auto contact_scene = make_scene(true);
    if (!contact_scene) return false;

    const auto& bounds = joint_group->getActiveJointModelsBounds();
    const double margin = settings_.left_parallel_min_joint_margin_deg * kDeg;
    std::vector<std::pair<double, double> > limits;
    for (const auto* joint_bounds : bounds) {
      if (joint_bounds->size() != 1U || !(*joint_bounds)[0].position_bounded_) {
        error = "left varying orientation probe expects single-variable bounded joints";
        return false;
      }
      limits.emplace_back((*joint_bounds)[0].min_position_, (*joint_bounds)[0].max_position_);
    }
    const auto inside_margin = [&](const std::vector<double>& q) {
        for (std::size_t j = 0; j < q.size() && j < limits.size(); ++j) {
          if (std::min(q[j] - limits[j].first, limits[j].second - q[j]) < margin) return false;
        }
        return true;
      };
    const auto& variable_names = joint_group->getVariableNames();
    const auto j6_it = std::find(variable_names.begin(), variable_names.end(), "joint6");
    if (j6_it == variable_names.end() || transit_seed.size() != variable_names.size()) {
      error = "left varying orientation probe needs joint6 and a full transit seed";
      return false;
    }
    const std::size_t j6 = static_cast<std::size_t>(std::distance(variable_names.begin(), j6_it));

    // ---- candidates (the same set and order as the fixed search) and their neighbour table
    {
      // The neighbour table is quadratic in the candidate count: cap the grid before building it.
      const auto grid_points = [&](double max_deg) {
          return 2.0 * std::floor(max_deg / settings_.left_parallel_step_deg + 1e-9) + 1.0;
        };
      const double grid_size = grid_points(settings_.left_parallel_max_tilt_deg) *
        grid_points(settings_.left_parallel_max_tilt_deg) *
        grid_points(settings_.left_parallel_max_yaw_deg);
      if (grid_size > static_cast<double>(kMaxVaryingCandidates)) {
        error = "left varying orientation search: " + formatDouble(grid_size, 0) +
          " orientation grid points exceed the cap of " + std::to_string(kMaxVaryingCandidates) +
          "; raise left_parallel_step_deg or lower left_parallel_max_tilt_deg / "
          "left_parallel_max_yaw_deg";
        return false;
      }
    }
    const std::vector<LeftParallelCandidate> candidates = buildLeftParallelCandidates(basis, false);
    if (out_of_time()) return timeout_error("building the orientation candidates");
    if (candidates.empty()) {
      error = "left varying orientation search: no orientation candidate passes the penetration gate";
      return false;
    }
    std::vector<double> candidate_score(candidates.size());
    std::vector<std::array<double, 4> > candidate_quat(candidates.size());
    std::vector<Eigen::Matrix3d> candidate_rotation(candidates.size());
    for (std::size_t c = 0; c < candidates.size(); ++c) {
      const auto& cand = candidates[c];
      candidate_score[c] = cand.theta_p + kDeviationWeight * cand.blade_deviation +
        kTotalWeight * cand.total;
      candidate_quat[c] = {cand.orientation.x(), cand.orientation.y(), cand.orientation.z(),
        cand.orientation.w()};
      candidate_rotation[c] = cand.orientation.toRotationMatrix();
    }
    left_varying::NeighborTable neighbors;
    if (!left_varying::buildNeighborTableBounded(
        candidate_quat, settings_.left_varying_max_step_deg, neighbors, out_of_time)) {
      return timeout_error("building the orientation neighbour table");
    }

    // ---- stations
    const Eigen::Vector3d line_start = toEigen(cut_start.pose.position);
    const Eigen::Vector3d line_end = toEigen(cut_end.pose.position);
    const double full_length_m = (line_end - line_start).norm();
    if (!std::isfinite(full_length_m) || full_length_m < 0.010) {
      error = "left varying orientation search: the left cut line is shorter than 10 mm";
      return false;
    }
    const std::size_t intervals =
      left_varying::stationIntervals(full_length_m, settings_.left_varying_station_spacing_m);
    if (extension != nullptr) {
      timeout_hint = "; the extended left cut (left_cut_extend_to_corners) searches " +
        std::to_string(intervals + 1U) + " stations over " + formatDouble(full_length_m * 1000.0, 1) +
        " mm, consider a larger left_parallel_search_timeout_s or left_varying_station_spacing_m";
    }
    Eigen::Vector3d outward = Eigen::Vector3d::Zero();
    {
      geometry_msgs::msg::PoseStamped probe;
      probe.pose.orientation.w = 1.0;
      if (!offsetLeftPoseOutward(snapshot, probe, settings_.left_lateral_standoff_m, error)) {
        return false;
      }
      outward = toEigen(probe.pose.position);
    }
    // Station positions on the cut line, with the inward (opposite to `outward`, horizontal,
    // perpendicular to the cut line and the surface normal) taper offset when requested.
    const Eigen::Vector3d inward_unit =
      outward.norm() > 1e-9 ? Eigen::Vector3d(-outward / outward.norm()) : Eigen::Vector3d::Zero();
    const auto inward_at = [&](std::size_t k) {
        return left_varying::inwardTaperM(
          inward_extra_m, full_length_m * (static_cast<double>(k) / intervals), taper_lo_m,
          taper_hi_m);
      };
    std::vector<Eigen::Vector3d> positions;
    for (std::size_t k = 0; k <= intervals; ++k) {
      positions.push_back(
        line_start + (line_end - line_start) * (static_cast<double>(k) / intervals));
      if (inward_extra_m > 0.0) positions.back() += inward_unit * inward_at(k);
      // left_cut_inward_m: constant over the whole line, so every station, the entry, the
      // hover/lift poses and the replay use the shifted positions.
      if (settings_.left_cut_inward_m > 0.0) {
        positions.back() += inward_unit * settings_.left_cut_inward_m;
      }
    }
    const Eigen::Vector3d normal = toEigen(snapshot.surface_normal).normalized();
    const Eigen::Vector3d lift_vector =
      (settings_.hover_clearance_m - settings_.cut_clearance_m) * normal;

    const auto pose_at = [&](std::size_t station, std::size_t candidate, const Eigen::Vector3d& offset) {
        Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
        target.translation() = positions[station] + offset;
        target.linear() = candidate_rotation[candidate];
        return target;
      };
    // Joint-space distance (max abs) between two branches.
    const auto max_diff = [](const std::vector<double>& a, const std::vector<double>& b) {
        double diff = 0.0;
        for (std::size_t j = 0; j < a.size() && j < b.size(); ++j) {
          diff = std::max(diff, std::abs(a[j] - b[j]));
        }
        return diff;
      };
    moveit::core::RobotState probe_state(seed_state);
    // Branch passes bounds, exact-pose check and (optionally) is collision-free in `scene`.
    const auto branch_valid = [&](
      const std::vector<double>& values, const Eigen::Isometry3d& target,
      planning_scene::PlanningScene& scene) {
        probe_state.setJointGroupPositions(joint_group, values);
        probe_state.update();
        if (!probe_state.satisfiesBounds(joint_group)) return false;
        const Eigen::Isometry3d& reached = probe_state.getGlobalLinkTransform(tool);
        if ((reached.translation() - target.translation()).norm() > 1e-5 ||
            Eigen::AngleAxisd(reached.linear().transpose() * target.linear()).angle() > 1e-4) {
          return false;
        }
        return !scene.isStateColliding(probe_state, settings_.planning_group);
      };

    // ---- per-(station, candidate) evaluation
    std::size_t evaluated_pairs = 0;
    const auto evaluate_station = [&](
      std::size_t station, const std::vector<std::size_t>& candidate_ids,
      std::vector<left_varying::Node>& nodes) {
        for (const std::size_t c : candidate_ids) {
          if (out_of_time()) return false;
          ++evaluated_pairs;
          const Eigen::Isometry3d target = pose_at(station, c, Eigen::Vector3d::Zero());
          for (const auto& values : analyticIkBranches(joint_group, target)) {
            if (!inside_margin(values)) continue;
            if (!branch_valid(values, target, *contact_scene)) continue;
            nodes.push_back(left_varying::Node{c, values, candidate_score[c]});
          }
        }
        return true;
      };

    // Hover (lateral standoff) state the transit will be sent to for a chain that STARTS at
    // this node: the collision-free (no tool/carton contact) analytic branch of the hover pose
    // nearest to the node's own branch (within kMaxStationJointStepRad, inside the joint margin,
    // joint6 change from the transit start at most pi like the wrist guard). The plan gives the
    // transit this JOINT goal instead of the pose goal, so the branch the profile was verified
    // on is the branch that is executed (a pose goal lets NearestBranchTransitPlanner pick a
    // different, unverified branch, on which the jump-detection-free strokes can flip).
    std::map<std::pair<std::size_t, std::size_t>, std::optional<std::vector<double> > > hover_cache;
    const auto hover_for = [&](std::size_t station, std::size_t index, const left_varying::Node& node)
      -> const std::optional<std::vector<double> >& {
        const auto key = std::make_pair(station, index);
        const auto found = hover_cache.find(key);
        if (found != hover_cache.end()) return found->second;
        std::optional<std::vector<double> > best;
        double best_diff = std::numeric_limits<double>::infinity();
        const Eigen::Isometry3d target = pose_at(station, node.candidate, outward);
        for (const auto& values : analyticIkBranches(joint_group, target)) {
          const double diff = max_diff(values, node.q);
          if (diff > kMaxStationJointStepRad || diff >= best_diff) continue;
          if (std::abs(values[j6] - transit_seed[j6]) > kPi) continue;
          if (!inside_margin(values) || !branch_valid(values, target, *free_scene)) continue;
          best = values;
          best_diff = diff;
        }
        return hover_cache.emplace(key, std::move(best)).first->second;
      };
    // Branch of `target` within kMaxStationJointStepRad of `reference` that is valid in
    // `scene` and inside the joint margin.
    const auto continuing_branch = [&](
      const Eigen::Isometry3d& target, const std::vector<double>& reference,
      planning_scene::PlanningScene& scene, std::vector<double>& found) {
        for (const auto& values : analyticIkBranches(joint_group, target)) {
          if (max_diff(values, reference) > kMaxStationJointStepRad) continue;
          if (!inside_margin(values) || !branch_valid(values, target, scene)) continue;
          found = values;
          return true;
        }
        return false;
      };

    // ---- the dynamic program, with lazy start/end gates and repair bans. Every station is fed
    // with new starts allowed and every candidate evaluated: there is no length-based pruning,
    // because a longer chain that later fails its exit/Cartesian verification would otherwise
    // suppress a shorter, fully verifiable one elsewhere on the cut.
    std::set<std::pair<std::size_t, std::size_t> > banned_starts, banned_ends;
    std::set<std::tuple<std::size_t, std::size_t, std::size_t> > banned_edges;
    std::vector<std::vector<left_varying::Node> > station_nodes(intervals + 1U);
    std::map<std::pair<std::size_t, std::size_t>, bool> end_cache;
    const auto start_ok = [&](std::size_t station, std::size_t index, const left_varying::Node& node) {
        if (banned_starts.count({station, index}) != 0U) return false;
        return hover_for(station, index, node).has_value();
      };
    const auto edge_ok = [&](std::size_t station, std::size_t from, std::size_t to) {
        return banned_edges.count({station, from, to}) == 0U;
      };
    const auto end_ok = [&](std::size_t station, std::size_t index) {
        if (banned_ends.count({station, index}) != 0U) return false;
        const auto key = std::make_pair(station, index);
        const auto cached = end_cache.find(key);
        if (cached != end_cache.end()) return cached->second;
        const auto& node = station_nodes[station][index];
        std::vector<double> hover_end_q;
        bool ok = continuing_branch(
          pose_at(station, node.candidate, outward), node.q, *contact_scene, hover_end_q);
        if (ok && lift_follows) {
          std::vector<double> lift_q;
          ok = continuing_branch(
            pose_at(station, node.candidate, outward + lift_vector), hover_end_q, *contact_scene,
            lift_q);
        }
        end_cache[key] = ok;
        return ok;
      };
    const left_varying::ChainDp::Config dp_config{kMaxStationJointStepRad, kSmoothWeight};
    const auto make_dp = [&]() {
        return left_varying::ChainDp(dp_config, neighbors, start_ok, edge_ok);
      };
    left_varying::ChainDp dp = make_dp();
    std::vector<std::size_t> all_candidates(candidates.size());
    for (std::size_t c = 0; c < all_candidates.size(); ++c) all_candidates[c] = c;
    for (std::size_t station = 0; station <= intervals; ++station) {
      std::vector<left_varying::Node> nodes;
      if (!evaluate_station(station, all_candidates, nodes)) {
        return timeout_error("evaluating station " + std::to_string(station) + " of " +
          std::to_string(intervals));
      }
      station_nodes[station] = nodes;
      const std::size_t node_count = nodes.size();
      dp.addStation(std::move(nodes), true);
      RCLCPP_DEBUG(
        get_logger(), "left varying station %zu/%zu: %zu feasible nodes, best chain %zu stations "
        "(%.1f s)", station, intervals, node_count, dp.bestLength(), elapsed_s());
      if (out_of_time()) {
        return timeout_error("after station " + std::to_string(station) + " of " +
          std::to_string(intervals));
      }
    }

    // ---- choose the best verifiable chain, smooth its orientation profile, replay the ONE
    // contact stroke exactly as the MTC stage will run it, repair on failure
    const double spacing_step_m = full_length_m / static_cast<double>(intervals);
    // Minimum kept length: left_varying_min_cut_fraction of the cut. In the extended mode that is
    // of the NOMINAL segment (back_middle_left -> front_middle_left), not of the extended span.
    const double min_kept_length_m = settings_.left_varying_min_cut_fraction *
      (extension != nullptr ? extension->nominal_length_m : full_length_m);
    const std::size_t min_stations_needed = extension == nullptr ?
      static_cast<std::size_t>(std::ceil(
        settings_.left_varying_min_cut_fraction * static_cast<double>(intervals) - 1e-9)) + 1U :
      static_cast<std::size_t>(std::ceil(min_kept_length_m / spacing_step_m - 1e-9)) + 1U;
    // Extended mode: the kept interval must cover the seam band, i.e. start at or before
    // required.first_max and end at or after required.last_min.
    left_varying::RequiredStations required;
    if (extension != nullptr) {
      required = left_varying::requiredStationRange(
        extension->band_lo_m, extension->band_hi_m, full_length_m, intervals);
    }
    const auto best_end = [&](const left_varying::ChainDp& active_dp,
                              left_varying::ChainDp::End& chosen, std::size_t min_length) {
        return extension == nullptr ?
          left_varying::bestVerifiedEnd(active_dp, end_ok, chosen, min_length) :
          left_varying::bestVerifiedEndCovering(
            active_dp, end_ok, chosen, min_length, required.first_max, required.last_min);
      };
    const auto fraction_failure = [&](std::size_t stations_found) {
        if (extension != nullptr) {
          const double kept_mm = static_cast<double>(stations_found > 0U ? stations_found - 1U : 0U) *
            spacing_step_m * 1000.0;
          error = "left varying orientation search failed closed (left_cut_extend_to_corners): " +
            std::string(stations_found == 0U ?
              "no feasible contiguous interval that covers the seam crossing +/- " +
              formatDouble(extension->seam_cover_m * 1000.0, 1) + " mm (band " +
              formatDouble((extension->band_lo_m - extension->nominal_start_m) * 1000.0, 1) + ".." +
              formatDouble((extension->band_hi_m - extension->nominal_start_m) * 1000.0, 1) +
              " mm from back_middle_left) exists" :
              "the longest feasible interval covering the seam band is " + formatDouble(kept_mm, 1) +
              " mm") +
            ", required at least " + formatDouble(min_kept_length_m * 1000.0, 1) + " mm (" +
            formatDouble(settings_.left_varying_min_cut_fraction * 100.0, 1) +
            "% of the nominal " + formatDouble(extension->nominal_length_m * 1000.0, 1) +
            " mm segment); extended span " + formatDouble(full_length_m * 1000.0, 1) + " mm (corner margin " +
            formatDouble(extension->corner_margin_m * 1000.0, 1) + " mm, " +
            std::to_string(candidates.size()) + " candidates, " + std::to_string(intervals + 1U) +
            " stations at " + formatDouble(spacing_step_m * 1000.0, 1) + " mm, " +
            std::to_string(evaluated_pairs) + " station/candidate pairs evaluated)";
          return false;
        }
        const double fraction = static_cast<double>(stations_found > 0U ? stations_found - 1U : 0U) /
          static_cast<double>(intervals);
        error = "left varying orientation search failed closed: the best contiguous feasible "
          "interval covers " + formatDouble(fraction * 100.0, 1) + "% (" +
          formatDouble(fraction * full_length_m * 1000.0, 1) + " mm of " +
          formatDouble(full_length_m * 1000.0, 1) + " mm) of the left cut, below "
          "left_varying_min_cut_fraction " +
          formatDouble(settings_.left_varying_min_cut_fraction * 100.0, 1) + "% (" +
          std::to_string(candidates.size()) + " candidates, " + std::to_string(intervals + 1U) +
          " stations at " + formatDouble(full_length_m / intervals * 1000.0, 1) + " mm, " +
          std::to_string(evaluated_pairs) + " station/candidate pairs evaluated)";
        return false;
      };
    struct Failure {
      // kTravel: the stroke exceeds kMaxLeftContactJointTravelRad. Banning one node does not
      // help (the interval is too long), so the whole first or last station is dropped.
      enum class Kind { kNone, kEntry, kEdge, kExit, kTravel } kind{Kind::kNone};
      std::size_t station{0}, from{0}, to{0};
      std::string detail;
      bool trim_start{false};
    };
    const WaypointCartesianPath::Config stroke_config{
      settings_.cartesian_step_m, kStrokeRotationStepRad, kMaxStrokeJointStepRad};
    const moveit::core::GroupStateValidityCallbackFn contact_valid =
      [this, &contact_scene](moveit::core::RobotState* s,
                             const moveit::core::JointModelGroup* group, const double* values) {
        s->setJointGroupPositions(group, values);
        s->update();
        return !contact_scene->isStateColliding(*s, settings_.planning_group);
      };
    const auto pose_with = [&](std::size_t station, const Eigen::Matrix3d& rotation,
                               const Eigen::Vector3d& offset) {
        Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
        target.translation() = positions[station] + offset;
        target.linear() = rotation;
        return target;
      };
    // Results of the accepted replay.
    std::size_t final_first = 0;
    std::vector<std::size_t> final_nodes;
    std::vector<bool> final_knots;
    std::vector<Eigen::Quaterniond> final_profile;
    std::vector<std::vector<double> > final_joints;
    std::vector<double> final_hover;
    std::string last_repair_detail;
    bool found_profile = false;
    // Joint-travel repairs (kTravel): first try many other branches at the offending end before
    // shortening (keeps the longest cuts); if that strategy fails, retry from scratch shortening
    // after 2 (the greedy side choice can trim itself out of the seam band). Both fail closed.
    constexpr std::array<int, 2> kTravelNodeBanStrategies{30, 2};
    std::map<std::pair<std::size_t, bool>, int> travel_failures;
    for (std::size_t strategy = 0; strategy < kTravelNodeBanStrategies.size() && !found_profile;
         ++strategy) {
      const int kTravelNodeBans = kTravelNodeBanStrategies[strategy];
      const bool last_strategy = strategy + 1U == kTravelNodeBanStrategies.size();
      banned_starts.clear();
      banned_ends.clear();
      banned_edges.clear();
      travel_failures.clear();
      last_repair_detail.clear();
      for (int attempt = 0; attempt <= kMaxRepairs && !found_profile; ++attempt) {
        if (out_of_time()) return timeout_error("repairing the station profile");
        left_varying::ChainDp replay_dp = make_dp();
        left_varying::ChainDp* active = &dp;
        if (attempt > 0) {
          for (std::size_t station = 0; station <= intervals; ++station) {
            replay_dp.addStation(station_nodes[station], true);
          }
          active = &replay_dp;
        }
        left_varying::ChainDp::End chosen_end;
        if (!best_end(*active, chosen_end, min_stations_needed)) {
          if (!last_strategy) {
            RCLCPP_INFO(get_logger(), "left varying travel repair strategy %d ran out of intervals; "
              "retrying with %d", kTravelNodeBans, kTravelNodeBanStrategies[strategy + 1U]);
            break;
          }
          // Report the longest interval that can at least be entered and left.
          left_varying::ChainDp::End any_end;
          const std::size_t best_len = best_end(*active, any_end, 1U) ? any_end.length : 0U;
          const bool result = fraction_failure(best_len);
          if (extension != nullptr) {
            left_varying::ChainDp::End unconstrained_end;
            const std::size_t unconstrained_len =
              left_varying::bestVerifiedEnd(*active, end_ok, unconstrained_end, 1U) ?
              unconstrained_end.length : 0U;
            error += "; longest enterable/leavable interval anywhere on the span: " +
              formatDouble(static_cast<double>(unconstrained_len > 0U ? unconstrained_len - 1U : 0U) *
              spacing_step_m * 1000.0, 1) + " mm";
          }
          if (!last_repair_detail.empty()) {
            error += "; last Cartesian-replay rejection: " + last_repair_detail;
          }
          return result;
        }
        const auto chain = active->extract(chosen_end);
        const std::size_t first = chain.first_station;
        const std::size_t count = chain.nodes.size();
        const std::size_t last_station = first + count - 1U;
        std::vector<const left_varying::Node*> dp_node(count);
        std::vector<const LeftParallelCandidate*> dp_cand(count);
        std::vector<Eigen::Quaterniond> dp_quat(count);
        for (std::size_t t = 0; t < count; ++t) {
          dp_node[t] = &station_nodes[first + t][chain.nodes[t]];
          dp_cand[t] = &candidates[dp_node[t]->candidate];
          dp_quat[t] = dp_cand[t]->orientation;
        }

        // -- orientation profile: keyframes, piecewise slerp. Start with the two end orientations
        // only; every station is validated (penetration gate, theta_p / blade deviation within the
        // DP station's value + left_varying_max_step_deg, a collision-free in-margin IK branch
        // continuing from the previous station); the worst failing station becomes a knot with
        // its DP orientation until the whole profile passes. All knots = the DP profile.
        std::vector<bool> knots(count, false);
        knots.front() = true;
        knots.back() = true;
        std::vector<Eigen::Quaterniond> profile(count);
        const auto fill_profile = [&]() {
            std::size_t a = 0;
            for (std::size_t b = 1; b < count; ++b) {
              if (!knots[b]) continue;
              for (std::size_t t = a; t <= b; ++t) {
                profile[t] = dp_quat[a].slerp(
                  static_cast<double>(t - a) / static_cast<double>(b - a), dp_quat[b]);
              }
              a = b;
            }
          };
        // true = profile acceptable. `blocked` = a knot itself could not continue (use all knots).
        const auto validate_profile = [&](std::vector<std::size_t>& failing, bool& blocked) {
            failing.clear();
            blocked = false;
            std::vector<double> previous = dp_node[0]->q;
            for (std::size_t t = 0; t < count; ++t) {
              const Eigen::Matrix3d rotation = profile[t].toRotationMatrix();
              const double theta_p = std::acos(std::min(1.0, std::abs(rotation.col(0).dot(basis.normal)))) / kDeg;
              const double deviation =
                std::acos(std::clamp(rotation.col(1).dot(-basis.along), -1.0, 1.0)) / kDeg;
              bool ok = (rotation.col(0).dot(into_unit) >= 0.7 || rotation.col(2).dot(into_unit) >= 0.7) &&
                theta_p <= dp_cand[t]->theta_p + settings_.left_varying_max_step_deg + 1e-6 &&
                deviation <= dp_cand[t]->blade_deviation + settings_.left_varying_max_step_deg + 1e-6;
              std::vector<double> found;
              if (ok) {
                const Eigen::Isometry3d target = pose_with(first + t, rotation, Eigen::Vector3d::Zero());
                double best = std::numeric_limits<double>::infinity();
                ok = false;
                for (const auto& values : analyticIkBranches(joint_group, target)) {
                  const double diff = max_diff(values, previous);
                  if (diff > kMaxStationJointStepRad || diff >= best) continue;
                  if (!inside_margin(values) || !branch_valid(values, target, *contact_scene)) continue;
                  found = values;
                  best = diff;
                  ok = true;
                }
              }
              if (ok) {
                previous = found;
              } else {
                if (knots[t]) {
                  blocked = true;
                  return false;
                }
                failing.push_back(t);
                previous = dp_node[t]->q;  // keep walking on the DP branch
              }
            }
            return failing.empty();
          };
        for (std::size_t round = 0; round <= count; ++round) {
          fill_profile();
          std::vector<std::size_t> failing;
          bool blocked = false;
          if (validate_profile(failing, blocked)) break;
          if (blocked || failing.empty()) {
            std::fill(knots.begin(), knots.end(), true);
            fill_profile();
            break;
          }
          std::size_t worst = failing.front();
          double worst_angle = -1.0;
          for (const std::size_t t : failing) {
            const double angle = left_varying::quaternionAngleDeg(
              {profile[t].x(), profile[t].y(), profile[t].z(), profile[t].w()},
              {dp_quat[t].x(), dp_quat[t].y(), dp_quat[t].z(), dp_quat[t].w()});
            if (angle > worst_angle) {
              worst_angle = angle;
              worst = t;
            }
          }
          knots[worst] = true;
        }

        // -- replay: hover -> entry, ONE stroke through the knots (the planner's own geometry),
        // finish, lift. Fails closed on fraction < 1, joint step, bounds, margin, collisions and
        // the aggregate contact joint travel the validators apply to the retained stroke.
        const auto& first_hover = hover_for(first, chain.nodes[0], *dp_node[0]);
        std::vector<std::vector<double> > replay_joints;
        const auto replay = [&](const std::vector<bool>& knot_flags, Failure& failure) {
            failure = Failure{};
            replay_joints.clear();
            moveit::core::RobotState state(seed_state);
            state.setJointGroupPositions(joint_group, *first_hover);
            state.update();
            std::string detail;
            const auto single_leg = [&](const Eigen::Isometry3d& target) {
                std::vector<moveit::core::RobotStatePtr> waypoints;
                std::size_t failed_leg = 0;
                return WaypointCartesianPath::computeLegs(
                  state, joint_group, tool, {target}, stroke_config, contact_valid, waypoints,
                  failed_leg, detail) && [&]() {
                    for (const auto& waypoint : waypoints) {
                      std::vector<double> q;
                      waypoint->copyJointGroupPositions(joint_group, q);
                      if (!waypoint->satisfiesBounds(joint_group)) {
                        detail = "joint bounds violated";
                        return false;
                      }
                      if (!inside_margin(q)) {
                        detail = "joint margin below " +
                          formatDouble(settings_.left_parallel_min_joint_margin_deg, 1) + " deg";
                        return false;
                      }
                    }
                    return true;
                  }();
              };
            if (!single_leg(pose_with(first, dp_quat[0].toRotationMatrix(), Eigen::Vector3d::Zero()))) {
              failure = {Failure::Kind::kEntry, first, chain.nodes[0], 0U, "lateral entry: " + detail};
              return false;
            }
            std::vector<Eigen::Isometry3d> targets;
            std::vector<std::size_t> target_t;
            for (std::size_t t = 1; t < count; ++t) {
              if (!knot_flags[t]) continue;
              targets.push_back(pose_with(first + t, dp_quat[t].toRotationMatrix(), Eigen::Vector3d::Zero()));
              target_t.push_back(t);
            }
            std::vector<moveit::core::RobotStatePtr> waypoints;
            std::size_t failed_leg = 0;
            if (!WaypointCartesianPath::computeLegs(
                state, joint_group, tool, targets, stroke_config, contact_valid, waypoints,
                failed_leg, detail)) {
              const std::size_t t_to = target_t[failed_leg];
              failure = {Failure::Kind::kEdge, first + t_to - 1U, chain.nodes[t_to - 1U],
                chain.nodes[t_to], "contact stroke " + detail};
              return false;
            }
            std::vector<double> previous;
            std::vector<double> travel;
            for (const auto& waypoint : waypoints) {
              std::vector<double> q;
              waypoint->copyJointGroupPositions(joint_group, q);
              if (travel.empty()) travel.assign(q.size(), 0.0);
              for (std::size_t j = 0; j < q.size() && !previous.empty(); ++j) {
                travel[j] += std::abs(q[j] - previous[j]);
              }
              if (!waypoint->satisfiesBounds(joint_group) || !inside_margin(q)) {
                failure = {Failure::Kind::kExit, last_station, chain.nodes.back(), 0U,
                  "contact stroke joint bounds/margin (" +
                  formatDouble(settings_.left_parallel_min_joint_margin_deg, 1) + " deg)"};
                return false;
              }
              previous = q;
              replay_joints.push_back(q);
            }
            for (std::size_t j = 0; j < travel.size(); ++j) {
              if (travel[j] > kMaxLeftContactJointTravelRad) {
                // Trim the end whose last station-length of stroke moves joint j more.
                const std::size_t n = replay_joints.size();
                const std::size_t k = std::max<std::size_t>(1U, n / std::max<std::size_t>(count, 1U));
                double head = 0.0, tail = 0.0;
                for (std::size_t w = 1; w < n && w <= k; ++w) {
                  head += std::abs(replay_joints[w][j] - replay_joints[w - 1U][j]);
                }
                for (std::size_t w = n > k ? n - k : 1U; w < n; ++w) {
                  tail += std::abs(replay_joints[w][j] - replay_joints[w - 1U][j]);
                }
                failure = {Failure::Kind::kTravel, head > tail ? first : last_station,
                  head > tail ? chain.nodes.front() : chain.nodes.back(), 0U,
                  "contact stroke moves joint" + std::to_string(j + 1) + " " +
                  formatDouble(travel[j], 3) + " rad (> " +
                  formatDouble(kMaxLeftContactJointTravelRad, 1) + " rad aggregate)", head > tail};
                return false;
              }
            }
            if (!single_leg(pose_with(last_station, dp_quat.back().toRotationMatrix(), outward))) {
              failure = {Failure::Kind::kExit, last_station, chain.nodes.back(), 0U,
                "finish (lateral withdrawal): " + detail};
              return false;
            }
            if (lift_follows &&
                !single_leg(pose_with(last_station, dp_quat.back().toRotationMatrix(), outward + lift_vector))) {
              failure = {Failure::Kind::kExit, last_station, chain.nodes.back(), 0U, "lift: " + detail};
              return false;
            }
            return true;
          };
        Failure failure;
        std::vector<bool> used_knots = knots;
        bool replayed = replay(used_knots, failure);
        const bool smoothed = std::count(knots.begin(), knots.end(), false) > 0;
        if (!replayed && smoothed) {
          RCLCPP_INFO(
            get_logger(), "left varying smoothed profile rejected (%s); using the full DP profile",
            failure.detail.c_str());
          used_knots.assign(count, true);
          for (std::size_t t = 0; t < count; ++t) profile[t] = dp_quat[t];
          replayed = replay(used_knots, failure);
        }
        if (replayed) {
          final_first = first;
          final_nodes = chain.nodes;
          final_knots = used_knots;
          final_profile = profile;
          final_hover = *first_hover;
          found_profile = true;
          // Station joints = the replayed stroke's own branch at each station.
          std::vector<Eigen::Vector3d> stroke_points;
          {
            moveit::core::RobotState fk(seed_state);
            for (const auto& q : replay_joints) {
              fk.setJointGroupPositions(joint_group, q);
              fk.update();
              stroke_points.push_back(fk.getGlobalLinkTransform(tool).translation());
            }
          }
          final_joints.clear();
          for (std::size_t t = 0; t < count; ++t) {
            // nearest replayed waypoint to the station position, joints interpolated on its segment
            const Eigen::Vector3d point = positions[first + t];
            double best = std::numeric_limits<double>::infinity();
            std::vector<double> joints = replay_joints.front();
            for (std::size_t w = 0; w + 1U < replay_joints.size(); ++w) {
              const Eigen::Vector3d seg = stroke_points[w + 1U] - stroke_points[w];
              const double len2 = seg.squaredNorm();
              const double u = len2 > 1e-18 ?
                std::clamp((point - stroke_points[w]).dot(seg) / len2, 0.0, 1.0) : 0.0;
              const double dist = (stroke_points[w] + u * seg - point).norm();
              if (dist < best) {
                best = dist;
                for (std::size_t j = 0; j < joints.size(); ++j) {
                  joints[j] = replay_joints[w][j] + u * (replay_joints[w + 1U][j] - replay_joints[w][j]);
                }
              }
            }
            final_joints.push_back(joints);
          }
          break;
        }
        last_repair_detail = failure.detail;
        RCLCPP_INFO(
          get_logger(), "left varying profile attempt %d rejected (%s); banning and re-solving",
          attempt, failure.detail.c_str());
        switch (failure.kind) {
          case Failure::Kind::kEntry: banned_starts.insert({failure.station, failure.from}); break;
          case Failure::Kind::kEdge: banned_edges.insert({failure.station, failure.from, failure.to}); break;
          case Failure::Kind::kExit: banned_ends.insert({failure.station, failure.from}); break;
          case Failure::Kind::kTravel: {
            // First try other branches at that end; after kTravelNodeBans failures there, shorten:
            // no chain may start (trim_start) / end at this station any more.
            auto& banned = failure.trim_start ? banned_starts : banned_ends;
            if (++travel_failures[{failure.station, failure.trim_start}] <= kTravelNodeBans) {
              banned.insert({failure.station, failure.from});
            } else {
              for (std::size_t index = 0; index < station_nodes[failure.station].size(); ++index) {
                banned.insert({failure.station, index});
              }
            }
            break;
          }
          case Failure::Kind::kNone: break;
        }
      }
    }
    if (!found_profile) {
      error = "left varying orientation search failed closed: no station profile survived the "
        "Cartesian replay after " + std::to_string(kMaxRepairs) + " repairs (last: " +
        last_repair_detail + ")";
      return false;
    }
    if (extension != nullptr) {
      // Defence in depth: the constrained selection above guarantees it; verify on the result.
      const double kept_lo_m = static_cast<double>(final_first) * spacing_step_m;
      const double kept_hi_m =
        static_cast<double>(final_first + final_nodes.size() - 1U) * spacing_step_m;
      if (kept_lo_m > extension->band_lo_m + 1e-9 || kept_hi_m < extension->band_hi_m - 1e-9) {
        error = "left varying orientation search failed closed: the kept extended interval does "
          "not cover the seam band (internal consistency check)";
        return false;
      }
    }

    // ---- assemble the plan
    plan.full_station_count = intervals + 1U;
    plan.full_length_m = full_length_m;
    plan.hover_start_joints = final_hover;
    const double spacing_m = full_length_m / static_cast<double>(intervals);
    std::size_t knot_count = 0;
    for (std::size_t t = 0; t < final_nodes.size(); ++t) {
      const std::size_t station = final_first + t;
      const auto& node = station_nodes[station][final_nodes[t]];
      const auto& cand = candidates[node.candidate];
      const Eigen::Matrix3d rotation = final_profile[t].toRotationMatrix();
      LeftVaryingStation entry;
      entry.position = positions[station];
      entry.orientation = final_profile[t];
      entry.is_knot = final_knots[t];
      if (entry.is_knot) {
        ++knot_count;
        entry.tilt_deg = cand.tilt;
        entry.pitch_deg = cand.pitch;
        entry.yaw_deg = cand.yaw;
      }
      entry.theta_p_deg = std::acos(std::min(1.0, std::abs(rotation.col(0).dot(basis.normal)))) / kDeg;
      entry.blade_deviation_deg =
        std::acos(std::clamp(rotation.col(1).dot(-basis.along), -1.0, 1.0)) / kDeg;
      entry.station_index = station;
      entry.joints = final_joints[t];
      plan.max_theta_p_deg = std::max(plan.max_theta_p_deg, entry.theta_p_deg);
      plan.max_blade_deviation_deg = std::max(plan.max_blade_deviation_deg, entry.blade_deviation_deg);
      plan.stations.push_back(entry);
    }
    const std::size_t last_index = final_first + final_nodes.size() - 1U;
    plan.planned_length_m = static_cast<double>(final_nodes.size() - 1U) * spacing_m;
    const double trim_back_mm = static_cast<double>(final_first) * spacing_m * 1000.0;
    const double trim_front_mm = static_cast<double>(intervals - last_index) * spacing_m * 1000.0;
    plan.extended = extension != nullptr;
    plan.trim_back_mm = trim_back_mm;
    plan.trim_front_mm = trim_front_mm;
    std::ostringstream trimmed;
    trimmed << std::fixed << std::setprecision(1);
    if (extension != nullptr) {
      trimmed << "extended span";
    } else if (final_first == 0U && last_index == intervals) {
      trimmed << "full cut";
    } else {
      trimmed << "TRIMMED";
      if (final_first > 0U) trimmed << " " << trim_back_mm << " mm at the start (back_middle_left) end";
      if (final_first > 0U && last_index < intervals) trimmed << " and";
      if (last_index < intervals) trimmed << " " << trim_front_mm << " mm at the end (front_middle_left) end";
    }
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(1);
    if (extension != nullptr) {
      // Arc positions of the kept interval relative to back_middle_left (negative = beyond it
      // toward the back corner) and of the seam band.
      const double kept_from_mm =
        (static_cast<double>(final_first) * spacing_m - extension->nominal_start_m) * 1000.0;
      const double kept_to_mm =
        (static_cast<double>(last_index) * spacing_m - extension->nominal_start_m) * 1000.0;
      summary << "left varying orientation (left_cut_extend_to_corners): extended span "
              << full_length_m * 1000.0 << " mm (corner margin "
              << extension->corner_margin_m * 1000.0 << " mm, nominal segment "
              << extension->nominal_length_m * 1000.0 << " mm), kept " << plan.planned_length_m * 1000.0
              << " mm (stations " << final_first << ".." << last_index << " of 0.." << intervals
              << ", " << plan.planned_length_m / extension->nominal_length_m * 100.0
              << "% of nominal, " << kept_from_mm << " .. " << kept_to_mm
              << " mm from back_middle_left), trim " << trim_back_mm
              << " mm at the back end and " << trim_front_mm
              << " mm at the front end of the span; seam band covered (seam "
              << (extension->seam_m - extension->nominal_start_m) * 1000.0
              << " mm from back_middle_left +/- " << extension->seam_cover_m * 1000.0
              << " mm, band " << (extension->band_lo_m - extension->nominal_start_m) * 1000.0
              << " .. " << (extension->band_hi_m - extension->nominal_start_m) * 1000.0
              << " mm); one continuous stroke, " << knot_count << " orientation knots; max theta_p "
              << std::setprecision(2) << plan.max_theta_p_deg
              << " deg, max blade deviation from cut line " << plan.max_blade_deviation_deg << " deg";
    } else {
      summary << "left varying orientation: planned " << plan.planned_length_m * 1000.0 << " mm of "
              << full_length_m * 1000.0 << " mm (" << plan.planned_length_m / full_length_m * 100.0
              << "%, stations " << final_first << ".." << last_index << " of 0.." << intervals
              << ", " << trimmed.str() << ", one continuous stroke, " << knot_count
              << " orientation knots); max theta_p " << std::setprecision(2)
              << plan.max_theta_p_deg << " deg, max blade deviation from cut line "
              << plan.max_blade_deviation_deg << " deg";
    }
    plan.summary = summary.str();
    if (settings_.left_cut_inward_m > 0.0) {
      plan.summary += "; whole left stroke " + formatDouble(settings_.left_cut_inward_m * 1000.0, 1) +
        " mm inside the detected face (left_cut_inward_m)";
    }
    if (inward_extra_m > 0.0) {
      plan.inward_start_mm = inward_at(final_first) * 1000.0;
      plan.inward_finish_mm = inward_at(last_index) * 1000.0;
      plan.summary += "; blade pushed inward (left_varying_start_extra_inward_m " +
        formatDouble(inward_extra_m * 1000.0, 1) + " mm): " + formatDouble(plan.inward_start_mm, 2) +
        " mm at the start -> " + formatDouble(plan.inward_finish_mm, 2) +
        " mm at the finish, linear taper over the kept stroke";
    }
    for (const auto& station : plan.stations) {
      std::ostringstream joints_deg;
      joints_deg << std::fixed << std::setprecision(1);
      for (std::size_t j = 0; j < station.joints.size(); ++j) {
        joints_deg << (j ? "," : "") << station.joints[j] / kDeg;
      }
      std::ostringstream knot_text;
      if (station.is_knot) {
        knot_text << std::fixed << std::setprecision(1) << " KNOT tilt " << std::showpos
                  << station.tilt_deg << " pitch " << station.pitch_deg << " yaw " << station.yaw_deg
                  << " deg";
      }
      RCLCPP_INFO(
        get_logger(),
        "Left varying orientation station %zu/%zu: s=%.1f mm theta_p %.2f deg, blade deviation "
        "%.2f deg%s, joints [%s] deg",
        station.station_index, intervals,
        static_cast<double>(station.station_index) * spacing_m * 1000.0, station.theta_p_deg,
        station.blade_deviation_deg, knot_text.str().c_str(), joints_deg.str().c_str());
    }
    if (final_first > 0U || last_index < intervals) {
      RCLCPP_WARN(
        get_logger(), "%s; planning the left cut over the trimmed interval only", plan.summary.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "%s", plan.summary.c_str());
    }
    RCLCPP_INFO(
      get_logger(),
      "Left varying orientation search done in %.1f s: %zu candidates, %zu stations, %zu "
      "station/candidate pairs evaluated, %zu repairs, %zu knots",
      elapsed_s(), candidates.size(), intervals + 1U, evaluated_pairs,
      banned_starts.size() + banned_edges.size() + banned_ends.size(), knot_count);
    return true;
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
    const double tilt_deg = is_middle ? settings_.middle_cut_tilt_deg :
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
    if (refuseLeftReverseTravel("start_cut_debug_session", message)) return false;
    if (refuseHalfSeam("start_cut_debug_session", message)) return false;
    if (refuseLeftVarying("start_cut_debug_session", message)) return false;
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
    exportable_task_ = nullptr;
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
  bool planSideTapeHandoff(std::string& message, const std::string& variant_override = "")
  {
    if (refuseLeftReverseTravel("plan_side_tape_handoff", message)) return false;
    if (refuseHalfSeam("plan_side_tape_handoff", message)) return false;
    if (refuseLeftVarying("plan_side_tape_handoff", message)) return false;
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
    const std::string variant =
      variant_override.empty() ? settings_.side_tape_handoff_variant : variant_override;
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
      allowCartonContact(*add_carton, true);
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
    if (refuseLeftReverseTravel("side_cut_penetrate", message)) return false;
    if (refuseHalfSeam("side_cut_penetrate", message)) return false;
    if (refuseLeftVarying("side_cut_penetrate", message)) return false;
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
    if (refuseLeftReverseTravel("continue_side_cut", message)) return false;
    if (refuseHalfSeam("continue_side_cut", message)) return false;
    if (refuseLeftVarying("continue_side_cut", message)) return false;
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
    // Operator 2026-10-02: hold in the material for side_cut_post_penetration_dwell_s before
    // the left cut stroke so the blade/arm settles after penetrating.
    if (settings_.side_cut_post_penetration_dwell_s > 0.0) {
      task->add(std::make_unique<StationaryDwell>(
        "hold at penetration before left cut", settings_.planning_group,
        settings_.side_cut_post_penetration_dwell_s));
    }
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
  // ---- One-call T cut: plan_t_cut / execute_t_cut / abort_t_cut / t_cut_status.
  // plan_t_cut = plan_side_tape_handoff with variant standoff (retains the left approach plan)
  // PLUS a top-cut feasibility check of the SAME capture (unshifted snapshot, taught offsets,
  // reversed blade, shortening allowed per the top_cut_* parameters), planned into a separate,
  // discarded task: top_cut_info_/top_cut_snapshot_/the retained plan are not touched. It
  // fails closed (never changes parameters) unless side_tape_handoff_enabled and
  // left_parallel_orientation_enabled are already true, because silently flipping planning
  // parameters would clear plans and hide a mis-configuration; set them with `ros2 param set`.
  bool planTCut(std::string& message)
  {
    t_cut_plan_ready_ = false;
    if (refuseLeftVarying("plan_t_cut", message)) return false;
    if (!settings_.side_tape_handoff_enabled || !settings_.left_parallel_orientation_enabled) {
      message = "plan_t_cut refused: side_tape_handoff_enabled and "
        "left_parallel_orientation_enabled must both be true (set them with ros2 param set; "
        "this call never changes parameters)";
      return false;
    }
    std::string handoff_message;
    if (!planSideTapeHandoff(handoff_message, "standoff")) {
      message = "plan_t_cut refused, left approach (standoff) not planned: " + handoff_message;
      return false;
    }
    const auto abandon = [this]() {
      // Nothing may stay retained after a refusal: drop the hand-off plan and its record.
      side_tape_handoff_info_ = SideTapeHandoffInfo{};
      side_tape_handoff_snapshot_.reset();
      clearRetainedPlan();
    };
    std::string top_message;
    double top_cut_fraction = 1.0;
    double top_cut_length_mm = 0.0;
    try {
      const auto loaded = loadExternalSnapshot(externalSnapshotFile(), top_message);
      if (!loaded) {
        abandon();
        message = "plan_t_cut refused, top cut snapshot: " + top_message;
        return false;
      }
      Snapshot snapshot = *loaded;
      std::string validation;
      if (!validateSnapshot(snapshot, validation)) {
        abandon();
        message = "plan_t_cut refused, top cut snapshot rejected: " + validation;
        return false;
      }
      if (snapshot.snapshot_id != side_tape_handoff_info_.snapshot_id) {
        abandon();
        message = "plan_t_cut refused: snapshot_id changed while planning";
        return false;
      }
      SegmentPlanOptions options;
      geometry_msgs::msg::PoseStamped hover_pose;
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> targets;
      const bool ok = planTopCutWithShortening(
        snapshot, options, hover_pose, task, targets, top_cut_fraction,
        "plan_t_cut top-cut feasibility", top_message, /*skip_pre_transit_lift=*/true);
      // planTopCutWithShortening published the unshifted carton; put the hand-off's
      // (shifted) carton back so the scene matches the retained approach plan.
      if (side_tape_handoff_snapshot_) publishCartonGeometry(*side_tape_handoff_snapshot_);
      if (!ok) {
        abandon();
        message = "plan_t_cut refused, top cut not feasible: " + top_message;
        return false;
      }
      top_cut_length_mm =
        (toEigen(targets[1].pose.position) - toEigen(targets[0].pose.position)).norm() * 1000.0;
      task.reset();  // feasibility only: discarded
    } catch (const std::exception& exception) {
      abandon();
      message = std::string("plan_t_cut refused, top cut feasibility threw: ") + exception.what();
      return false;
    }
    const auto& sel = side_tape_handoff_info_.left_parallel;
    t_cut_plan_ready_ = true;
    message = "T cut planned. LEFT: standoff approach retained (cut_length=" +
      formatDouble(side_tape_handoff_info_.cut_length_mm, 1) + " mm, left angles tilt/pitch/yaw=" +
      formatDouble(sel.tilt_deg, 1) + "/" + formatDouble(sel.pitch_deg, 1) + "/" +
      formatDouble(sel.yaw_deg, 1) + " deg, rotation_from_F0=" +
      formatDouble(sel.rotation_from_f0_deg, 1) + " deg). TOP: feasible (cut_length=" +
      formatDouble(top_cut_length_mm, 1) + " mm, " + formatDouble(top_cut_fraction * 100.0, 0) +
      " % of the seam, " + (top_cut_fraction < 1.0 ? "SHORTENED" : "not shortened") +
      "); the top plan is discarded and is re-planned from the live state at its step. "
      "Execute via execute_t_cut with execution_enabled";
    return true;
  }
  void publishTCutStatus(
    int step_index, const std::string& step, const std::string& status,
    const std::string& message)
  {
    std::ostringstream json;
    json << "{\"step_index\": " << step_index << ", \"step\": \"" << jsonEscape(step) <<
      "\", \"status\": \"" << status << "\", \"message\": \"" << jsonEscape(message) <<
      "\", \"stamp_ns\": " << now().nanoseconds() << "}";
    std_msgs::msg::String topic_message;
    topic_message.data = json.str();
    t_cut_status_pub_->publish(topic_message);
    RCLCPP_INFO(
      get_logger(), "T cut step %d '%s': %s%s%s", step_index, step.c_str(), status.c_str(),
      message.empty() ? "" : " - ", message.c_str());
  }
  bool executeTCut(std::string& message)
  {
    if (!t_cut_plan_ready_ || !planned_task_ || !planned_snapshot_ || !side_tape_handoff_pending_ ||
        planned_task_->solutions().empty()) {
      t_cut_plan_ready_ = false;
      message = "execute_t_cut refused: no retained plan from plan_t_cut (call plan_t_cut first)";
      return false;
    }
    if (settings_.debug_allow_stale_snapshot || retained_with_stale_override_ ||
        side_tape_handoff_info_.stale_override) {
      message = "execute_t_cut refused: planned/loaded under debug_allow_stale_snapshot "
        "(plan-only override)";
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
    std::string servo_error;
    if (!sideTapeServoIsQuiescent(servo_error)) {
      message = "execute_t_cut refused: " + servo_error;
      return false;
    }
    // t_cut_running_ is false and plan_mutex_ is held: the previous sequencer (if any) has
    // already left its last locked section, so joining cannot deadlock.
    if (t_cut_thread_.joinable()) t_cut_thread_.join();
    t_cut_plan_ready_ = false;
    t_cut_abort_ = false;
    t_cut_running_ = true;
    publishTCutStatus(0, "sequence", "running", "T cut sequence dispatched (4 steps)");
    t_cut_thread_ = std::thread([this]() { runTCutSequence(); });
    message = "T cut dispatch accepted: approach_standoff -> side_cut_penetrate -> "
      "continue_side_cut -> continue_top_cut; watch ~/t_cut_status; abort_t_cut stops further steps";
    return true;
  }
  // Runs on t_cut_thread_. Every step calls the SAME function the individual service calls,
  // under plan_mutex_ (so all its gates apply), then waits with the mutex released for the
  // dispatched execution to finish (its completion handler needs plan_mutex_). Success is read
  // from the state the completion handler leaves. Any failure stops the sequence at once and
  // leaves the state exactly as the failing individual service would.
  void runTCutSequence()
  {
    struct Step {
      const char* name;
      std::function<bool(std::string&)> body;
      std::function<bool()> succeeded;
      const char* expectation;
    };
    const std::vector<Step> steps = {
      {"approach_standoff", [this](std::string& m) { return executeRetainedPlan(m); },
       [this]() { return side_tape_handoff_info_.status == SideTapeHandoffStatus::kActive; },
       "hand-off status active"},
      {"side_cut_penetrate", [this](std::string& m) { return sideCutPenetrate(m); },
       [this]() { return side_tape_handoff_info_.status == SideTapeHandoffStatus::kActive; },
       "hand-off status still active (not cut_failed)"},
      {"continue_side_cut", [this](std::string& m) { return continueSideCut(m); },
       [this]() { return side_tape_handoff_info_.status == SideTapeHandoffStatus::kRetracted; },
       "hand-off status retracted"},
      {"continue_top_cut", [this](std::string& m) { return continueTopCut(m); },
       [this]() { return top_cut_info_.status == TopCutStagedStatus::kDone; },
       "top cut status done"},
    };
    std::string failure;
    int failed_index = 0;
    std::string failed_name = "sequence";
    for (std::size_t i = 0; i < steps.size() && failure.empty(); ++i) {
      const int index = static_cast<int>(i) + 1;
      const Step& step = steps[i];
      publishTCutStatus(index, step.name, "running", "");
      // The arm settles slowly after each move (JTC reports success while it still creeps
      // in): wait up to t_cut_settle_timeout_s for joints at rest / Servo quiet before the
      // next step, whose own quiescence gate would otherwise refuse it.
      if (i > 0) {
        const auto settle_deadline = std::chrono::steady_clock::now() +
          std::chrono::duration<double>(settings_.t_cut_settle_timeout_s);
        std::string settle_message;
        while (!t_cut_abort_ && rclcpp::ok() && !sideTapeServoIsQuiescent(settle_message) &&
          std::chrono::steady_clock::now() < settle_deadline)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
      std::string step_message;
      bool dispatched = false;
      {
        std::unique_lock<std::mutex> lock(plan_mutex_);
        if (t_cut_abort_ || !rclcpp::ok()) {
          failure = "aborted before this step started";
        } else {
          try {
            dispatched = step.body(step_message);
          } catch (const std::exception& exception) {
            step_message = std::string("threw: ") + exception.what();
          }
          if (!dispatched) failure = "refused/planning failed: " + step_message;
        }
      }
      if (failure.empty()) {
        bool done = false;
        while (!done) {
          {
            std::lock_guard<std::mutex> lock(plan_mutex_);
            if (!execution_in_progress_) {
              done = true;
              if (!step.succeeded()) {
                failure = std::string("execution did not complete (expected ") +
                  step.expectation + ")";
              }
            }
          }
          if (!done) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      }
      if (!failure.empty()) {
        failed_index = index;
        failed_name = step.name;
        break;
      }
      publishTCutStatus(index, step.name, "done", step_message);
    }
    if (failure.empty()) {
      publishTCutStatus(5, "sequence", "done", "all 4 steps completed");
    } else {
      publishTCutStatus(
        failed_index, failed_name, "failed",
        failure + "; state left as the individual service leaves it (recover with "
        "side_tape_retract or top_cut_retract)");
    }
    std::lock_guard<std::mutex> lock(plan_mutex_);
    t_cut_running_ = false;
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
  // Plans the whole top (middle seam) cut from the CURRENT robot state with the taught
  // offsets, the staged speed factor and the pre-transit lift; shared by continue_top_cut and
  // plan_t_cut's feasibility check so both compute identical poses. `snapshot` is modified in
  // place (offsets, shortened seam end). Operator 2026-10-01 (like t_cut_handoff): if the full
  // top cut cannot be planned (e.g. the wrist runs out of range near the seam end), retry with
  // the cut END pulled back along the seam in 5 % steps down to top_cut_min_cut_fraction. Every
  // stroke is still planned complete (min_fraction 1.0) to its shortened end; a partial
  // Cartesian path is never accepted. Plans only; retains and dispatches nothing.
  bool planTopCutWithShortening(
    Snapshot& snapshot, SegmentPlanOptions& options, geometry_msgs::msg::PoseStamped& hover_pose,
    std::unique_ptr<mtc::Task>& task, std::vector<geometry_msgs::msg::PoseStamped>& targets,
    double& cut_fraction, const std::string& who, std::string& message,
    bool skip_pre_transit_lift = false)
  {
    if (settings_.middle_cut_reverse_travel) {
      message = who + " refused: middle_cut_reverse_travel is middle_only only; set it false";
      return false;
    }
    if (refuseHalfSeam(who.c_str(), message)) return false;
    if (!applyTopCutTaughtOffsets(snapshot, options, message)) return false;
    options.cut_stroke_speed_factor = settings_.top_cut_staged_speed_factor;
    options.middle_finish_clearance_m = settings_.top_cut_lift_clearance_m;
    // plan_t_cut checks feasibility from the CURRENT (e.g. home) pose, where the pre-transit
    // lift may be impossible (arm already near full height); at run time continue_top_cut
    // starts from the left-cut withdraw pose and applies the lift.
    options.pre_transit_lift_m = skip_pre_transit_lift ? 0.0 : settings_.top_cut_pre_transit_lift_m;
    options.hover_start_out = &hover_pose;
    publishCartonGeometry(snapshot);
    const auto corrections = loadAllActiveCorrections();
    std::string plan_error;
    const Snapshot full_snapshot = snapshot;
    const SegmentPlanOptions full_options = options;
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
        get_logger(), "%s: %.0f %% cut did not plan (%s); trying shorter", who.c_str(),
        cut_fraction * 100.0, plan_error.c_str());
    }
    if (!planned) {
      message = who + " planning failed; nothing moved (tried down to " +
        formatDouble(settings_.top_cut_min_cut_fraction * 100.0, 0) + " % of the cut): " +
        first_error;
      return false;
    }
    if (cut_fraction < 1.0) {
      RCLCPP_WARN(
        get_logger(), "%s: SHORTENED cut, %.0f %% of the seam (the rest is not cut)",
        who.c_str(), cut_fraction * 100.0);
    }
    if (targets.size() != 2U) {
      message = who + ": unexpected target count for the middle segment";
      return false;
    }
    return true;
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
      geometry_msgs::msg::PoseStamped hover_pose;
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> targets;
      double cut_fraction = 1.0;
      if (!planTopCutWithShortening(
            snapshot, options, hover_pose, task, targets, cut_fraction, "continue_top_cut",
            message)) {
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
    if (settings_.middle_cut_reverse_travel) {
      message = "plan_top_cut_staged refused: middle_cut_reverse_travel is middle_only only; set it false";
      return false;
    }
    if (refuseHalfSeam("plan_top_cut_staged", message)) return false;
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
    return applySnapshotFrameHandling(snapshot, message);
  }
  // True when the snapshot is NOT used as written (fixed transform and/or relabel).
  bool snapshotFrameHandlingIsActive() const
  {
    return settings_.snapshot_frame_mode != "planning_frame" || settings_.snapshot_relabel != "none";
  }
  // Brings a freshly parsed snapshot into planning_frame (snapshot_frame_mode fixed_transform:
  // p_plan = B^-1 p_snapshot, B = snapshot_planning_base_xyz/_rpy) and then relabels the corners
  // (snapshot_relabel rotate_180). Applied exactly once, at the end of parseSnapshotFromDocument,
  // so every caller sees the same snapshot; validateSnapshot then runs unchanged on the result.
  // With the defaults this returns immediately without touching the snapshot.
  bool applySnapshotFrameHandling(Snapshot& snapshot, std::string& message) const
  {
    const auto mode = snapshot_frame::modeFromString(settings_.snapshot_frame_mode);
    const auto relabel = snapshot_frame::relabelFromString(settings_.snapshot_relabel);
    if (!mode || !relabel) {
      message = "invalid snapshot_frame_mode or snapshot_relabel";
      return false;
    }
    if (*mode == snapshot_frame::Mode::kPlanningFrame && *relabel == snapshot_frame::Relabel::kNone) {
      return true;
    }
    if (!settings_.plan_export_only) {
      message = "a transformed or relabelled snapshot is only allowed with plan_export_only=true";
      return false;
    }
    snapshot_frame::Points points;
    const auto to_vec = [](const geometry_msgs::msg::Point& point) {
      return Eigen::Vector3d(point.x, point.y, point.z);
    };
    points.seam_start = to_vec(snapshot.seam_start);
    points.seam_end = to_vec(snapshot.seam_end);
    points.seam_center = to_vec(snapshot.seam_center);
    points.back_left = to_vec(snapshot.back_left);
    points.back_right = to_vec(snapshot.back_right);
    points.front_left = to_vec(snapshot.front_left);
    points.front_right = to_vec(snapshot.front_right);
    points.back_middle_left = to_vec(snapshot.back_middle_left);
    points.front_middle_left = to_vec(snapshot.front_middle_left);
    points.back_middle_right = to_vec(snapshot.back_middle_right);
    points.front_middle_right = to_vec(snapshot.front_middle_right);
    points.surface_normal = toEigen(snapshot.surface_normal);
    if (*mode == snapshot_frame::Mode::kFixedTransform) {
      if (snapshot.header.frame_id != settings_.snapshot_frame_id) {
        message = "snapshot frame_id '" + snapshot.header.frame_id + "' != snapshot_frame_id '" +
          settings_.snapshot_frame_id + "' (snapshot_frame_mode fixed_transform)";
        return false;
      }
      const Eigen::Isometry3d base = snapshot_frame::poseFromXyzRpy(
        {settings_.snapshot_planning_base_xyz[0], settings_.snapshot_planning_base_xyz[1],
         settings_.snapshot_planning_base_xyz[2]},
        {settings_.snapshot_planning_base_rpy[0], settings_.snapshot_planning_base_rpy[1],
         settings_.snapshot_planning_base_rpy[2]});
      points = snapshot_frame::toPlanningFrame(points, base);
      snapshot.header.frame_id = settings_.planning_frame;
    }
    if (*relabel == snapshot_frame::Relabel::kRotate180) {
      const snapshot_frame::Points relabelled = snapshot_frame::relabelRotate180(points);
      if (!snapshot_frame::sameHandedness(points, relabelled)) {
        message = "snapshot_relabel rotate_180 would change the handedness of the carton labels "
          "(mirror); refusing";
        return false;
      }
      points = relabelled;
    }
    const auto from_vec = [](const Eigen::Vector3d& value) { return toPoint(value); };
    snapshot.seam_start = from_vec(points.seam_start);
    snapshot.seam_end = from_vec(points.seam_end);
    snapshot.seam_center = from_vec(points.seam_center);
    snapshot.back_left = from_vec(points.back_left);
    snapshot.back_right = from_vec(points.back_right);
    snapshot.front_left = from_vec(points.front_left);
    snapshot.front_right = from_vec(points.front_right);
    snapshot.back_middle_left = from_vec(points.back_middle_left);
    snapshot.front_middle_left = from_vec(points.front_middle_left);
    snapshot.back_middle_right = from_vec(points.back_middle_right);
    snapshot.front_middle_right = from_vec(points.front_middle_right);
    snapshot.surface_normal.x = points.surface_normal.x();
    snapshot.surface_normal.y = points.surface_normal.y();
    snapshot.surface_normal.z = points.surface_normal.z();
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
    if (snapshotFrameHandlingIsActive()) {
      // The seed block (plane, wall edge) is written in the snapshot frame with the snapshot's own
      // labels and is NOT transformed or relabelled: using it would silently mix two frames.
      message = "side_tape_seed is refused while snapshot_frame_mode is fixed_transform or "
        "snapshot_relabel is not none (the seed is not transformed)";
      return false;
    }
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
    primitive.dimensions = {width, depth, settings_.carton_height_m};
    geometry_msgs::msg::Pose pose;
    pose.position = toPoint(top_center - normal * settings_.carton_height_m / 2.0);
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
    const auto left = leftPlanningSegment(snapshot);
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
      if (settings_.left_debug_stop_at_approach && pattern != "left_only") {
        message = "left_debug_stop_at_approach refused: it is only valid with cut_pattern "
          "left_only (current cut_pattern '" + pattern + "'); set left_debug_stop_at_approach "
          "false or cut_pattern left_only";
        return false;
      }
      if (settings_.middle_seam_stop != "end" && pattern != "middle_only") {
        message = "middle_seam_stop refused: '" + settings_.middle_seam_stop + "' (half-seam top "
          "cut) is only valid with cut_pattern middle_only (current cut_pattern '" + pattern +
          "'); set middle_seam_stop end";
        return false;
      }
      if (snapshotFrameHandlingIsActive() && pattern == "h" &&
          !settings_.approach_joint_degrees.empty()) {
        message = "cut_pattern h refused with a transformed/relabelled snapshot: "
          "approach_joint_degrees is a joint-space approach taught for the LEFT arm";
        return false;
      }
      if (settings_.middle_cut_reverse_travel && pattern != "middle_only" &&
          pattern != "left_only" && pattern != "right_only") {
        message = "middle_cut_reverse_travel refused: it is only valid with cut_pattern "
          "middle_only (current cut_pattern '" + pattern + "'); set it false";
        return false;
      }
      std::string validation;
      if (!validateSnapshot(snapshot, validation)) {
        message = "snapshot rejected: " + validation;
        return false;
      }
      const auto& motion_profile = settings_.motion_profile;
      if (!isSupportedMotionProfile(motion_profile)) {
        throw std::invalid_argument("unsupported H-cut motion_profile: " + motion_profile);
      }
      const bool pattern_has_left = pattern == "left_only" || pattern == "t_cut" ||
        pattern == "t_cut_handoff" || pattern == "h" || pattern == "combined";
      last_left_varying_summary_.clear();
      if (pattern_has_left) {
        const std::string varying_conflict = leftVaryingConflict(pattern);
        if (!varying_conflict.empty()) {
          message = pattern + " planning refused: " + varying_conflict;
          return false;
        }
        if (settings_.left_parallel_varying_orientation_enabled) {
          RCLCPP_WARN(
            get_logger(), "left_parallel_varying_orientation_enabled=true: the LEFT cut orientation "
            "varies along the stroke and the cut may be trimmed (pattern %s)", pattern.c_str());
        }
        if (settings_.left_cut_extend_to_corners) {
          RCLCPP_WARN(
            get_logger(), "left_cut_extend_to_corners=true: the LEFT cut is searched over the whole "
            "edge (corner margin %.1f mm) and keeps the longest reachable interval covering the "
            "seam crossing +/- %.1f mm (pattern %s)", settings_.left_cut_corner_margin_m * 1000.0,
            settings_.left_cut_seam_cover_m * 1000.0, pattern.c_str());
        }
        const std::string reverse_conflict = leftReverseTravelConflict();
        if (!reverse_conflict.empty()) {
          message = pattern + " planning refused: " + reverse_conflict;
          return false;
        }
        if (settings_.left_cut_reverse_travel) {
          RCLCPP_WARN(
            get_logger(), "left_cut_reverse_travel=true: planning the LEFT side cut front_middle_left"
            "->back_middle_left (blade +Y toward the front); pattern %s", pattern.c_str());
        }
      }
      publishCartonGeometry(snapshot);
      // Each configured correction record is loaded and validated once per request.
      const auto corrections = loadAllActiveCorrections();
      if (pattern == "t_cut_handoff") {
        return planTCutHandoffSnapshot(snapshot, corrections, motion_profile, message);
      }
      const bool waypoint_pattern = pattern == "combined" || pattern == "t_cut";
      std::vector<CutSegment> segments = segmentsForPattern(snapshot, pattern);
      std::unique_ptr<mtc::Task> task;
      std::vector<geometry_msgs::msg::PoseStamped> cut_targets;
      SegmentPlanOptions options;
      options.skip_supplied_approach = pattern != "h";
      options.allow_left_varying = pattern_has_left;
      options.return_home = settings_.return_home_after_cut;
      // left_debug_stop_at_approach (left_only only, checked above): the full run is planned
      // exactly as usual; these hooks hand its results to the approach-only second run.
      const bool debug_stop_at_approach = settings_.left_debug_stop_at_approach;
      LeftRollCache debug_roll_cache;
      LeftVaryingPlan debug_varying_plan;
      std::vector<double> debug_hover_joints;
      geometry_msgs::msg::PoseStamped debug_full_hover;
      if (debug_stop_at_approach) {
        options.left_roll_cache = &debug_roll_cache;
        options.varying_plan_out = &debug_varying_plan;
        options.hover_start_joints_out = &debug_hover_joints;
        options.hover_start_out = &debug_full_hover;
        last_left_parallel_selection_ = LeftParallelSelection{};
      }
      const std::string task_name = !waypoint_pattern ?
        "Snapshot-relative " + pattern + " dry run" :
        (pattern == "t_cut" ? "Snapshot-relative T-cut canonical waypoint plan" :
         "Snapshot-relative combined canonical waypoint plan");
      // middle_only (operator 2026-10-13, like continue_top_cut) and t_cut (operator 2026-10-14):
      // if the full seam does not plan, retry with the cut END pulled back along the seam in 5 % steps down to
      // top_cut_min_cut_fraction (top_cut_allow_shortened_cut). Every candidate is a complete
      // plan (min_fraction 1.0); a partial Cartesian path is never accepted.
      double middle_only_fraction = 1.0;
      const bool middle_reverse = pattern == "middle_only" && settings_.middle_cut_reverse_travel;
      Snapshot base_snapshot = snapshot;
      if (middle_reverse) {
        std::swap(base_snapshot.seam_start, base_snapshot.seam_end);
        segments = segmentsForPattern(base_snapshot, pattern);
      }
      // Half-seam top cut (middle_seam_stop=center, middle_only only): the stroke ends at the seam
      // centre (projected onto the seam from seam_start) plus middle_seam_center_overlap_m beyond
      // it in the travel direction. The carton box keeps its axis direction. The shortening loop
      // below is disabled in this mode: shortening would leave tape uncut at the centre.
      const bool half_seam = pattern == "middle_only" && settings_.middle_seam_stop == "center";
      double half_seam_fraction = 1.0;
      if (half_seam) {
        const Eigen::Vector3d seam_from = toEigen(base_snapshot.seam_start);
        const Eigen::Vector3d seam_axis = toEigen(base_snapshot.seam_end) - seam_from;
        const double seam_length = seam_axis.norm();
        if (!(seam_length > 0.05)) {
          message = "middle_seam_stop center refused: the middle seam is shorter than 50 mm";
          return false;
        }
        const double center_fraction =
          seam_axis.dot(toEigen(base_snapshot.seam_center) - seam_from) / (seam_length * seam_length);
        half_seam_fraction = center_fraction + settings_.middle_seam_center_overlap_m / seam_length;
        if (!(center_fraction > 0.1 && center_fraction < 0.9) || !(half_seam_fraction > 0.1) ||
            !(half_seam_fraction <= 1.0)) {
          message = "middle_seam_stop center refused: seam_center projects to " +
            formatDouble(center_fraction * 100.0, 1) + " % of the seam (stop at " +
            formatDouble(half_seam_fraction * 100.0, 1) + " % with overlap " +
            formatDouble(settings_.middle_seam_center_overlap_m * 1000.0, 1) +
            " mm); the centre must lie within 10..90 % of the seam";
          return false;
        }
        base_snapshot.seam_end = toPoint(seam_from + half_seam_fraction * seam_axis);
        segments = segmentsForPattern(base_snapshot, pattern);
      }
      Snapshot planned_snapshot = base_snapshot;
      // Left -> middle transit height (t_cut/h/combined): no single extra lift works for every
      // carton pose (2026-10-15 mock: +40 deg needs it, -10 deg fails with it), so each seam
      // length tries the configured lift, then none, then 0.10 m before shortening. Each try
      // replans every stage, so a randomly unlucky OMPL transit gets another chance too.
      std::vector<double> lift_candidates{settings_.middle_transit_extra_lift_m};
      if (pattern == "t_cut" || pattern == "h" || pattern == "combined") {
        for (const double lift : {0.0, 0.10}) {
          if (std::abs(lift - lift_candidates.front()) > 1e-9) lift_candidates.push_back(lift);
        }
      }
      double used_lift = lift_candidates.front();
      // The varying left search depends only on the left edge, carton and start state, none of
      // which change across the retries below: search once, reuse the verified plan after.
      const bool varying_left =
        pattern_has_left && settings_.left_parallel_varying_orientation_enabled;
      if (varying_left) options.varying_plan_out = &debug_varying_plan;
      for (int step = 0; ; ++step) {
        bool planned = false;
        for (const double lift : lift_candidates) {
          task.reset();
          cut_targets.clear();
          message.clear();
          transit_extra_lift_override_ = lift;
          planned = planCutSegments(
            planned_snapshot, segments, corrections, motion_profile, task_name, options, task,
            cut_targets, message);
          transit_extra_lift_override_.reset();
          if (planned) {
            used_lift = lift;
            break;
          }
          if (varying_left && options.reuse_varying == nullptr) {
            // The left search itself failed: no lift or top-cut length can change that.
            if (debug_varying_plan.stations.empty()) return false;
            options.reuse_varying = &debug_varying_plan;
          }
          RCLCPP_WARN(
            get_logger(), "%s: plan with transit extra lift %.0f mm failed (%s)", pattern.c_str(),
            lift * 1000.0, message.c_str());
        }
        if (planned) break;
        const double next_fraction = 1.0 - 0.05 * (step + 1);
        if ((pattern != "middle_only" && pattern != "t_cut") ||
            !settings_.top_cut_allow_shortened_cut || half_seam ||
            next_fraction < settings_.top_cut_min_cut_fraction - 1e-9) {
          if (half_seam) {
            message = "middle_only half seam (middle_seam_stop center, stop at " +
              formatDouble(half_seam_fraction * 100.0, 1) + " % of the seam) did not plan; "
              "shortening is disabled in this mode: " + message;
          } else if (middle_only_fraction < 1.0) {
            message = pattern + " planning failed down to " +
              formatDouble(settings_.top_cut_min_cut_fraction * 100.0, 0) +
              " % of the seam: " + message;
          }
          return false;
        }
        RCLCPP_WARN(
          get_logger(), "%s: %.0f %% top cut did not plan (%s); trying shorter",
          pattern.c_str(), middle_only_fraction * 100.0, message.c_str());
        middle_only_fraction = next_fraction;
        planned_snapshot = base_snapshot;
        if (middle_reverse) {
          // Reversed travel: the unreachable far side is the stroke START; pull it back.
          planned_snapshot.seam_start = toPoint(
            toEigen(base_snapshot.seam_end) + middle_only_fraction *
            (toEigen(base_snapshot.seam_start) - toEigen(base_snapshot.seam_end)));
        } else {
          planned_snapshot.seam_end = toPoint(
            toEigen(base_snapshot.seam_start) + middle_only_fraction *
            (toEigen(base_snapshot.seam_end) - toEigen(base_snapshot.seam_start)));
        }
        segments = segmentsForPattern(planned_snapshot, pattern);
        task.reset();
        cut_targets.clear();
        message.clear();
      }
      if (middle_only_fraction < 1.0) {
        RCLCPP_WARN(
          get_logger(), "%s: SHORTENED top cut, %.0f %% of the seam (the rest is not cut)",
          pattern.c_str(), middle_only_fraction * 100.0);
      }
      if (debug_stop_at_approach) {
        return retainLeftApproachOnly(
          snapshot, segments, corrections, motion_profile, options, cut_targets,
          debug_full_hover, debug_hover_joints, debug_varying_plan, message);
      }
      retainPlan(planned_snapshot, segments, cut_targets, std::move(task));
      retained_middle_fraction_ = half_seam ? half_seam_fraction : middle_only_fraction;
      const std::string dwell_summary = "pre-cut stabilization_s=" +
        formatDouble(settings_.pre_cut_stabilization_duration_s, 3) + "; dwell_count=" +
        std::to_string(settings_.pre_cut_stabilization_duration_s > 0.0 ? segments.size() : 0U);
      if (waypoint_pattern) {
        const bool t_cut = pattern == "t_cut";
        message =
          "Snapshot-relative " + std::string(t_cut ? "T-cut" : "combined") +
          " dry-run plan [" + motion_profile + "] published to RViz; waypoint_order=approach" +
          (settings_.left_cut_reverse_travel ?
           "->front_middle_left->back_middle_left" : "->back_middle_left->front_middle_left") +
          "->finish->transition->approach->seam_start"
          "->seam_end->finish" +
          (t_cut ? "" : "->transition->approach->back_middle_right->front_middle_right->finish") +
          "; side_knife_facing=opposite_travel; " + dwell_summary +
          (last_left_varying_summary_.empty() ? std::string() : "; " + last_left_varying_summary_) +
          "; transit extra lift " + formatDouble(used_lift * 1000.0, 0) + " mm" +
          (t_cut ? "; top cut_fraction=" + formatDouble(middle_only_fraction * 100.0, 0) +
            " % of the seam" + (middle_only_fraction < 1.0 ? " (SHORTENED, the rest is not cut)" : "") :
            std::string()) +
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
                (settings_.left_cut_reverse_travel && pattern == "left_only" ?
                  "; left_cut_reverse_travel=true (front_middle_left->back_middle_left)" : "") +
                (last_left_varying_summary_.empty() ? std::string() :
                  "; " + last_left_varying_summary_) +
                (includes_middle_seam ? "; middle entry alignment_m=" +
                  formatDouble(settings_.middle_entry_alignment_m, 3) : "") +
                (middle_reverse ? "; middle_cut_reverse_travel=true (seam_end->seam_start)" : "") +
                (settings_.return_home_after_cut ? "; returns to the SRDF home pose after the cut" : "") +
                (pattern == "middle_only" ? "; middle_only cut_fraction=" +
                  formatDouble(retained_middle_fraction_ * 100.0, 0) + " % of the seam" +
                  (middle_only_fraction < 1.0 ? " (SHORTENED, the rest is not cut)" : "") : "") +
                (half_seam ? "; middle_seam_stop=center (half seam, stroke ends " +
                  formatDouble(settings_.middle_seam_center_overlap_m * 1000.0, 1) +
                  " mm past the seam centre; the rest is not cut)" : "") +
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
  // left_debug_stop_at_approach: `full_options` produced the full left_only plan (already
  // validated end to end); this plans a SECOND task over the same segment that stops at the
  // left approach/hover pose after the pre-cut stabilization dwell (stop_at_hover_start: no
  // carton contact scope, lateral entry, stroke or lift), retains THAT task, and publishes the
  // full cut's markers/targets for inspection. In varying mode the second run is handed the full
  // run's verified plan (no second search) and its trajectory must end on the verified hover
  // branch; otherwise its hover pose must equal the full plan's. Fails closed on any mismatch.
  // The retained plan is flagged so execute_planned_h_cut additionally needs
  // debug_motion_execution_enabled.
  bool retainLeftApproachOnly(
    const Snapshot& snapshot, const std::vector<CutSegment>& segments,
    const std::map<std::string, CutDebugCorrection>& corrections,
    const std::string& motion_profile, const SegmentPlanOptions& full_options,
    const std::vector<geometry_msgs::msg::PoseStamped>& full_targets,
    const geometry_msgs::msg::PoseStamped& full_hover, const std::vector<double>& full_hover_joints,
    const LeftVaryingPlan& full_varying, std::string& message)
  {
    const std::string full_summary_varying = last_left_varying_summary_;
    const LeftParallelSelection selection = last_left_parallel_selection_;
    if (segments.size() != 1U || full_targets.size() != 2U) {
      message = "left_debug_stop_at_approach: unexpected segment/target count for the left_only plan";
      return false;
    }
    const bool varying_used = !full_hover_joints.empty();
    SegmentPlanOptions approach_options = full_options;
    approach_options.stop_at_hover_start = true;
    geometry_msgs::msg::PoseStamped approach_hover;
    approach_options.hover_start_out = &approach_hover;
    approach_options.varying_plan_out = nullptr;
    approach_options.hover_start_joints_out = nullptr;
    approach_options.reuse_varying = varying_used ? &full_varying : nullptr;
    std::map<std::string, double> final_joints;
    approach_options.final_joints_out = &final_joints;
    std::unique_ptr<mtc::Task> approach_task;
    std::vector<geometry_msgs::msg::PoseStamped> approach_targets;
    std::string plan_error;
    if (!planCutSegments(
          snapshot, segments, corrections, motion_profile,
          "Snapshot-relative left_only APPROACH-ONLY debug plan (stops at the left hover pose)",
          approach_options, approach_task, approach_targets, plan_error)) {
      message = "left_debug_stop_at_approach: the full left cut planned, but the approach-only "
        "task failed: " + plan_error;
      return false;
    }
    last_left_varying_summary_ = full_summary_varying;  // keep the full run's summary
    if (approach_targets.size() != full_targets.size() ||
        (toEigen(approach_targets[0].pose.position) - toEigen(full_targets[0].pose.position))
        .norm() > 1e-6 ||
        (toEigen(approach_targets[1].pose.position) - toEigen(full_targets[1].pose.position))
        .norm() > 1e-6) {
      message = "left_debug_stop_at_approach: the approach-only and full plans disagree on the cut "
        "targets; refusing";
      return false;
    }
    if ((toEigen(approach_hover.pose.position) - toEigen(full_hover.pose.position)).norm() > 1e-6 ||
        toEigen(approach_hover.pose.orientation).normalized().angularDistance(
          toEigen(full_hover.pose.orientation).normalized()) > 1e-6) {
      message = "left_debug_stop_at_approach: the approach-only and full plans disagree on the "
        "left hover pose; refusing";
      return false;
    }
    const auto* joint_group = robot_model_ ?
      robot_model_->getJointModelGroup(settings_.planning_group) : nullptr;
    if (joint_group == nullptr) {
      message = "left_debug_stop_at_approach: planning group is missing from the robot model";
      return false;
    }
    const auto& group_names = joint_group->getVariableNames();
    std::vector<double> end_joints;
    for (const auto& name : group_names) {
      const auto found = final_joints.find(name);
      if (found == final_joints.end()) {
        message = "left_debug_stop_at_approach: the approach-only trajectory has no end value for " +
          name + "; refusing";
        return false;
      }
      end_joints.push_back(found->second);
    }
    if (varying_used) {
      constexpr double kHoverJointTolRad = 1e-3;
      if (full_hover_joints.size() != end_joints.size()) {
        message = "left_debug_stop_at_approach: hover joint count mismatch; refusing";
        return false;
      }
      for (std::size_t index = 0; index < end_joints.size(); ++index) {
        if (!(std::abs(end_joints[index] - full_hover_joints[index]) <= kHoverJointTolRad)) {
          message = "left_debug_stop_at_approach: the approach-only trajectory ends on " +
            group_names[index] + "=" + formatDouble(end_joints[index], 4) +
            " rad, not on the full plan's verified hover branch (" +
            formatDouble(full_hover_joints[index], 4) + " rad); refusing";
          return false;
        }
      }
    }
    // Full-cut markers/targets, the approach-only task's solution.
    retainPlan(snapshot, segments, full_targets, std::move(approach_task));
    retained_debug_motion_ = true;
    std::ostringstream joints_text;
    joints_text << std::fixed << std::setprecision(1);
    for (std::size_t index = 0; index < end_joints.size(); ++index) {
      joints_text << (index ? ", " : "") << group_names[index] << "=" <<
        end_joints[index] * 180.0 / 3.14159265358979323846;
    }
    const double cut_length_mm =
      (toEigen(full_targets[1].pose.position) - toEigen(full_targets[0].pose.position)).norm() * 1000.0;
    std::string cut_summary;
    if (!full_summary_varying.empty()) {
      cut_summary = full_summary_varying;
    } else {
      cut_summary = "left cut length " + formatDouble(cut_length_mm, 1) + " mm";
      if (selection.valid) {
        cut_summary += ", left_parallel theta_p " + formatDouble(selection.theta_p_deg, 2) +
          " deg (tilt " + formatDouble(selection.tilt_deg, 1) + ", pitch " +
          formatDouble(selection.pitch_deg, 1) + ", yaw " + formatDouble(selection.yaw_deg, 1) + " deg)";
      }
    }
    const auto& q = approach_hover.pose.orientation;
    message = "Snapshot-relative left_only APPROACH-ONLY DEBUG plan [" + motion_profile +
      "] (left_debug_stop_at_approach): the full left cut was planned and validated, its "
      "markers/targets are published to RViz, but ONLY the approach to the left hover pose "
      "(after the pre-cut stabilization dwell, s=" +
      formatDouble(settings_.pre_cut_stabilization_duration_s, 3) +
      ") is retained: no carton contact scope, lateral entry, stroke or lift; full-cut summary: " +
      cut_summary + "; final hover pose position=(" +
      formatDouble(approach_hover.pose.position.x, 4) + ", " +
      formatDouble(approach_hover.pose.position.y, 4) + ", " +
      formatDouble(approach_hover.pose.position.z, 4) + ") m, orientation xyzw=(" +
      formatDouble(q.x, 4) + ", " + formatDouble(q.y, 4) + ", " + formatDouble(q.z, 4) + ", " +
      formatDouble(q.w, 4) + ") in " + approach_hover.header.frame_id + "; final hover joints [deg] " +
      joints_text.str() + (varying_used ? " (the verified varying hover branch)" : "") +
      "; executing it requires BOTH execution_enabled=true AND debug_motion_execution_enabled=true "
      "(execute_planned_h_cut), and the arm stops at the hover pose";
    return true;
  }
  void publishSolution(mtc::Task& task)
  {
    task.introspection().publishSolution(*task.solutions().front());
  }
  // The RobotModel is normally loaded by the first planCutSegments; plan_h_cut_from_state needs it
  // earlier to validate its input.
  void ensureRobotModel()
  {
    if (robot_model_) return;
    mtc::Task probe;
    probe.loadRobotModel(shared_from_this());
    robot_model_ = probe.getRobotModel();
    if (!robot_model_) throw std::runtime_error("robot model could not be loaded");
  }
  // Exactly the planning group's variables, each once, finite, inside the joint bounds.
  bool validateStartState(
    const sensor_msgs::msg::JointState& state, StartStateOverride& out, std::string& message)
  {
    ensureRobotModel();
    const auto* group = robot_model_->getJointModelGroup(settings_.planning_group);
    if (group == nullptr) {
      message = "planning group is missing from the robot model";
      return false;
    }
    const auto& variables = group->getVariableNames();
    if (state.name.size() != variables.size() || state.position.size() != variables.size()) {
      message = "start_state must carry exactly the " + std::to_string(variables.size()) +
        " planning-group joints (name and position arrays of that length); got " +
        std::to_string(state.name.size()) + " names / " + std::to_string(state.position.size()) +
        " positions";
      return false;
    }
    out.names = variables;
    out.positions.assign(variables.size(), 0.0);
    std::vector<bool> seen(variables.size(), false);
    for (std::size_t index = 0; index < state.name.size(); ++index) {
      const auto found = std::find(variables.begin(), variables.end(), state.name[index]);
      if (found == variables.end()) {
        message = "start_state joint '" + state.name[index] + "' is not a planning-group joint";
        return false;
      }
      const auto slot = static_cast<std::size_t>(std::distance(variables.begin(), found));
      if (seen[slot]) {
        message = "start_state joint '" + state.name[index] + "' appears twice";
        return false;
      }
      seen[slot] = true;
      const double position = state.position[index];
      if (!std::isfinite(position)) {
        message = "start_state joint '" + state.name[index] + "' is not finite";
        return false;
      }
      const auto& bounds = robot_model_->getVariableBounds(state.name[index]);
      if (bounds.position_bounded_ &&
          (position < bounds.min_position_ || position > bounds.max_position_)) {
        message = "start_state joint '" + state.name[index] + "' = " + formatDouble(position, 6) +
          " rad is outside its limits [" + formatDouble(bounds.min_position_, 6) + ", " +
          formatDouble(bounds.max_position_, 6) + "]";
        return false;
      }
      out.positions[slot] = position;
    }
    return true;
  }
  // The planning scene MTC's CurrentState would start from (move_group's get_planning_scene with
  // every component, fetched through a temporary node), with the planning group's joints replaced
  // by the override. Throws on any failure.
  planning_scene::PlanningScenePtr startStateOverrideScene(const StartStateOverride& override_state)
  {
    ensureRobotModel();
    static std::atomic<int> fetch_counter{0};
    auto node = std::make_shared<rclcpp::Node>(
      "box_h_cut_executor_start_state_" + std::to_string(fetch_counter.fetch_add(1)),
      get_namespace());
    auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
    if (!client->wait_for_service(std::chrono::seconds(5))) {
      throw std::runtime_error("get_planning_scene is not available (is move_group running?)");
    }
    auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    using Components = moveit_msgs::msg::PlanningSceneComponents;
    request->components.components =
      Components::SCENE_SETTINGS | Components::ROBOT_STATE |
      Components::ROBOT_STATE_ATTACHED_OBJECTS | Components::WORLD_OBJECT_NAMES |
      Components::WORLD_OBJECT_GEOMETRY | Components::OCTOMAP | Components::TRANSFORMS |
      Components::ALLOWED_COLLISION_MATRIX | Components::LINK_PADDING_AND_SCALING |
      Components::OBJECT_COLORS;
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    auto future = client->async_send_request(request);
    if (executor.spin_until_future_complete(future, std::chrono::seconds(10)) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      throw std::runtime_error("get_planning_scene did not answer within 10 s");
    }
    const auto response = future.get();
    auto scene = std::make_shared<planning_scene::PlanningScene>(robot_model_);
    scene->setPlanningSceneMsg(response->scene);
    auto& state = scene->getCurrentStateNonConst();
    state.setVariablePositions(override_state.names, override_state.positions);
    state.zeroVelocities();
    state.update();
    return scene;
  }
  // ~/plan_h_cut_from_state: the normal plan_h_cut flow (same gates, same snapshot, same pattern)
  // with the plan starting from `state` instead of the live joint states. The retained plan is
  // marked retained_from_override_ and can never be executed.
  bool planHCutFromState(const sensor_msgs::msg::JointState& state, std::string& message)
  {
    if (cut_debug_stage_ != CutDebugStage::kNone) {
      message = "cut debug session active for cut " + cut_debug_session_cut_ +
        " stage " + cutDebugStageName(cut_debug_stage_) +
        "; call clear_plan before planning a normal H-cut";
      return false;
    }
    if (side_tape_handoff_info_.inProgress()) {
      message = "a side-tape hand-off is active or unresolved; call side_tape_retract before "
        "planning a normal H-cut";
      return false;
    }
    if (topCutStagedInProgress()) {
      message = "a staged top cut is active; call top_cut_retract before planning a normal H-cut";
      return false;
    }
    const std::string& pattern = settings_.cut_pattern;
    if (pattern == "t_cut_handoff" || pattern == "middle_entry_debug" ||
        pattern == "middle_debug_short" || settings_.left_debug_stop_at_approach) {
      message = "plan_h_cut_from_state refused for cut_pattern '" + pattern +
        (settings_.left_debug_stop_at_approach ? "' with left_debug_stop_at_approach" : "'") +
        " (handoff, debug and approach-only plans are not supported from an explicit start state)";
      return false;
    }
    StartStateOverride requested;
    if (!validateStartState(state, requested, message)) {
      message = "plan_h_cut_from_state refused: " + message;
      return false;
    }
    clearRetainedPlan();
    struct OverrideReset {
      std::optional<StartStateOverride>& slot;
      ~OverrideReset() { slot.reset(); }
    } reset{start_state_override_};
    start_state_override_ = std::move(requested);
    return planExternalSnapshot(message);
  }
  // ~/get_planned_solution (plan_mutex_ held by the caller). Read only: the retained task is neither
  // consumed nor changed. Every leaf stage with a trajectory must have a known name (plan_export.hpp).
  bool exportPlannedSolution(dual_cut_msgs::srv::GetPlannedSolution::Response& response)
  {
    if (!planned_task_ || !planned_snapshot_ || planned_task_->solutions().empty()) {
      response.message = "no retained H-cut plan to export";
      return false;
    }
    if (planned_task_.get() != exportable_task_) {
      response.message = "the retained plan was not produced by plan_h_cut (for example a staged "
        "cut-debug start); refusing to export it";
      return false;
    }
    const mtc::SolutionBase& solution = *planned_task_->solutions().front();
    moveit_task_constructor_msgs::msg::Solution raw;
    solution.toMsg(raw, &planned_task_->introspection());
    {
      // Introspection::publishSolution also stamps the task id; reproduce it so `raw` equals the
      // message published on the introspection solution topic.
      moveit_task_constructor_msgs::msg::TaskDescription description;
      planned_task_->introspection().fillTaskDescription(description);
      raw.task_id = description.task_id;
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
      if (const auto* sub = dynamic_cast<const mtc::SubTrajectory*>(node)) leaves.push_back(sub);
    };
    collect(&solution);
    if (leaves.size() != raw.sub_trajectory.size()) {
      response.message = "export refused: solution leaf count (" + std::to_string(leaves.size()) +
        ") does not match its message (" + std::to_string(raw.sub_trajectory.size()) + ")";
      return false;
    }
    std::vector<std::string> segment_names;
    for (const auto& segment : planned_segments_) segment_names.push_back(segment.name);
    const bool rotate_180 = settings_.snapshot_relabel == "rotate_180";
    std::vector<dual_cut_msgs::msg::PlannedSegment> exported;
    std::size_t cut_stages = 0U;
    std::vector<std::string> joint_names;
    std::vector<double> start_positions;
    std::vector<double> end_positions;
    for (std::size_t index = 0; index < leaves.size(); ++index) {
      const auto& trajectory = raw.sub_trajectory[index].trajectory;
      const auto& joint_trajectory = trajectory.joint_trajectory;
      if (joint_trajectory.points.empty()) {
        if (!trajectory.multi_dof_joint_trajectory.points.empty()) {
          response.message = "export refused: a stage has a multi-DOF trajectory only";
          return false;
        }
        continue;  // scene-only stage
      }
      const mtc::Stage* creator = leaves[index]->creator();
      const std::string name = creator != nullptr ? creator->name() : std::string();
      const auto kind = plan_export::classifyStageName(name);
      if (!kind) {
        response.message = "export refused: stage '" + name +
          "' has a trajectory but an unknown name (plan_export.hpp); not guessing its kind";
        return false;
      }
      if (!trajectory.multi_dof_joint_trajectory.points.empty()) {
        response.message = "export refused: stage '" + name + "' has a multi-DOF trajectory";
        return false;
      }
      if (joint_names.empty()) {
        joint_names = joint_trajectory.joint_names;
        start_positions = joint_trajectory.points.front().positions;
      } else if (joint_trajectory.joint_names != joint_names) {
        response.message = "export refused: stage '" + name + "' has different joint names";
        return false;
      }
      end_positions = joint_trajectory.points.back().positions;
      dual_cut_msgs::msg::PlannedSegment segment;
      segment.kind = static_cast<std::uint8_t>(*kind);
      segment.stage_name = name;
      if (const auto owner = plan_export::segmentIndexForStage(name, segment_names)) {
        const CutSegment& owner_segment = planned_segments_[*owner];
        segment.cut_id = owner_segment.cut != CutKind::kNone ? std::string(cutName(owner_segment.cut)) :
          owner_segment.name;
      }
      segment.physical_cut = plan_export::physicalCut(segment.cut_id, rotate_180);
      if (leaves[index]->start() != nullptr && leaves[index]->start()->scene()) {
        const auto& matrix = leaves[index]->start()->scene()->getAllowedCollisionMatrix();
        // True only if EVERY carton_contact_links entry is allowed at the start of the stage (default:
        // the tool_link entry alone).
        bool all_allowed = true;
        for (const auto& link : settings_.cartonContactLinks()) {
          collision_detection::AllowedCollision::Type allowed;
          all_allowed = all_allowed && matrix.getEntry(link, settings_.carton_collision_id, allowed) &&
            allowed == collision_detection::AllowedCollision::ALWAYS;
        }
        segment.tool_carton_contact_allowed = all_allowed;
      }
      segment.trajectory = trajectory;
      if (*kind == plan_export::Kind::kCut) ++cut_stages;
      exported.push_back(std::move(segment));
    }
    if (exported.empty()) {
      response.message = "export refused: the retained plan has no trajectory";
      return false;
    }
    if (planned_cut_targets_.size() != 2U * cut_stages) {
      response.message = "export refused: " + std::to_string(cut_stages) + " CUT stage(s) but " +
        std::to_string(planned_cut_targets_.size()) + " cut targets (two per CUT stage expected)";
      return false;
    }
    response.plan_seq = plan_seq_;
    response.snapshot_id = planned_snapshot_->snapshot_id;
    response.capture_id = planned_snapshot_->capture_id;
    response.acquisition_stamp_ns = static_cast<std::uint64_t>(planned_snapshot_->acquisition_stamp_ns);
    response.stale_override = retained_with_stale_override_;
    response.planning_frame = settings_.planning_frame;
    response.tool_link = settings_.tool_link;
    response.cut_pattern = retained_cut_pattern_;
    response.snapshot_frame_mode = settings_.snapshot_frame_mode;
    response.snapshot_frame_id = settings_.snapshot_frame_id;
    response.snapshot_relabel = settings_.snapshot_relabel;
    for (std::size_t i = 0; i < 3U; ++i) {
      response.snapshot_planning_base_xyz[i] = settings_.snapshot_planning_base_xyz[i];
      response.snapshot_planning_base_rpy[i] = settings_.snapshot_planning_base_rpy[i];
    }
    response.middle_cut_fraction = retained_middle_fraction_;
    response.cut_targets = planned_cut_targets_;
    response.start_state.name = joint_names;
    response.start_state.position = start_positions;
    response.end_state.name = joint_names;
    response.end_state.position = end_positions;
    response.segments = std::move(exported);
    response.summary = retained_summary_;
    response.raw = std::move(raw);
    response.message = "exported " + std::to_string(response.segments.size()) + " segments (" +
      std::to_string(cut_stages) + " CUT) of plan " + std::to_string(plan_seq_) +
      (retained_from_override_ ? " (planned from an explicit start state; not executable)" : "");
    return true;
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
    // Export bookkeeping (get_planned_solution); nothing here feeds back into planning.
    exportable_task_ = planned_task_.get();
    planned_cut_targets_ = cut_targets;
    planned_segments_ = segments;
    retained_cut_pattern_ = settings_.cut_pattern;
    retained_middle_fraction_ = 1.0;
    retained_summary_.clear();
    retained_from_override_ = start_state_override_.has_value();
    ++plan_seq_;
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
    segments = {leftPlanningSegment(snapshot), handoff_segment};
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
      "waypoint_order=approach" +
      std::string(settings_.left_cut_reverse_travel ?
        "->front_middle_left->back_middle_left" : "->back_middle_left->front_middle_left") +
      "->finish->transition->approach->seam_start->middle_seam_handoff_end; " +
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
  // Scoped tool/carton contact allowance (true) or its restore (false), for every link of
  // carton_contact_links (default: tool_link only, one call, exactly as before).
  void addCartonContactScope(mtc::Task& task, const std::string& name, bool allow)
  {
    auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(name);
    stage->restrictDirection(mtc::PropagatingEitherWay::FORWARD);
    allowCartonContact(*stage, allow);
    task.add(std::move(stage));
  }
  void allowCartonContact(mtc::stages::ModifyPlanningScene& stage, bool allow) const
  {
    for (const auto& link : settings_.cartonContactLinks()) {
      stage.allowCollisions(link, settings_.carton_collision_id, allow);
    }
  }
  // Same allowance written straight into a planning scene's ACM (analytic-search scenes).
  void setCartonContactEntries(collision_detection::AllowedCollisionMatrix& matrix, bool allowed) const
  {
    for (const auto& link : settings_.cartonContactLinks()) {
      matrix.setEntry(link, settings_.carton_collision_id, allowed);
    }
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
    if (start_state_override_) {
      // plan_h_cut_from_state: start from the supplied joint values instead of /joint_states.
      auto fixed_state = std::make_unique<mtc::stages::FixedState>("start state (override)");
      fixed_state->setState(startStateOverrideScene(*start_state_override_));
      task->add(std::move(fixed_state));
    } else {
      task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));
    }
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
      allowCartonContact(*add_carton, true);
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
    if (start_state_override_) {
      // Every IK seed below (left searches, transit seed) derives from ik_state.
      for (std::size_t index = 0; index < start_state_override_->names.size(); ++index) {
        ik_state.setVariablePosition(
          start_state_override_->names[index], start_state_override_->positions[index]);
      }
      ik_state.update();
    } else
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
    std::optional<LeftVaryingPlan> retained_varying;  // validated on the retained trajectory
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
      if (segment.reversed_travel) {
        // Defense in depth: every caller that builds a reversed left segment is gated
        // already; this re-checks the combinations that cannot be reviewed per caller.
        if (segment_correction != nullptr) {
          message = "left_cut_reverse_travel cannot be combined with an active left correction "
            "record; taught corrections assume back->front travel";
          return false;
        }
        const std::string reverse_conflict = leftReverseTravelConflict();
        if (!reverse_conflict.empty() || options.side_tape_handoff || options.cut_debug_start) {
          message = reverse_conflict.empty() ?
            std::string("left_cut_reverse_travel is not supported in side-tape hand-off or cut-debug planning") :
            reverse_conflict;
          return false;
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
      const double tilt_deg = is_middle_seam ? settings_.middle_cut_tilt_deg :
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
      // Varying-orientation left cut (opt-in): the profile replaces the single fixed
      // orientation; the stage structure below is the fixed parallel mode's, built from the
      // first/last station poses, with one Cartesian segment per station pair.
      std::optional<LeftVaryingPlan> varying_plan;
      const bool left_varying = left_parallel && settings_.left_parallel_varying_orientation_enabled;
      if (left_varying) {
        if (motion_profile != "ompl_cartesian") {
          message =
            "left_parallel_varying_orientation_enabled supports only motion_profile "
            "ompl_cartesian (its single contact stroke is a CartesianInterpolator path; the "
            "Pilz/CHOMP profiles plan strokes differently)";
          return false;
        }
        if (segment_correction != nullptr || segment.reversed_travel || options.side_tape_handoff ||
            options.cut_debug_start || !options.allow_left_varying || segment_index != 0U) {
          message =
            "left_parallel_varying_orientation_enabled is only supported for the first left "
            "segment of left_only/t_cut/h/combined planning (no left correction, reversed "
            "travel, staged cut-debug, side-tape hand-off or t_cut_handoff)";
          return false;
        }
        const bool lift_follows_varying = (segment_index + 1U) < segments.size() &&
          segments[segment_index + 1U].cut == CutKind::kMiddle &&
          !segments[segment_index + 1U].hover_only;
        // The transit into the left approach starts at the current state, or at the supplied
        // joint-space approach when this plan begins with it (cut_pattern h).
        std::vector<double> transit_seed;
        ik_state.copyJointGroupPositions(joint_group, transit_seed);
        if (!options.skip_supplied_approach && !settings_.approach_joint_degrees.empty()) {
          const auto approach = jointGoalDegrees(joint_group, settings_.approach_joint_degrees);
          const auto& group_names = joint_group->getVariableNames();
          for (std::size_t index = 0; index < group_names.size(); ++index) {
            transit_seed[index] = approach.at(group_names[index]);
          }
        }
        // left_cut_extend_to_corners: the candidate line is the nominal back_middle_left ->
        // front_middle_left line extended in both directions to the corners (minus the
        // margin). Its end points go through the SAME targetPose() (surface-normal clearance
        // of the cut, same orientation direction) as the nominal cut_start/cut_end, so the
        // line stays at the nominal cut height; only the position along the line changes.
        std::optional<LeftExtension> extension;
        if (settings_.left_cut_extend_to_corners) {
          LeftExtension geometry;
          geometry_msgs::msg::Point extended_start_point;
          geometry_msgs::msg::Point extended_end_point;
          if (!computeLeftExtension(
              snapshot, segment, geometry, extended_start_point, extended_end_point, message)) {
            return false;
          }
          const auto extended_orientation_targets = orientationTargets(
            snapshot, false, segment.start, segment.end, extended_start_point, extended_end_point);
          cut_start = targetPose(
            snapshot, extended_start_point, extended_orientation_targets.first,
            target_clearance_m, tilt_deg);
          cut_end = targetPose(
            snapshot, extended_end_point, extended_orientation_targets.second,
            target_clearance_m, tilt_deg);
          extension = geometry;
        }
        LeftVaryingPlan found;
        if (options.reuse_varying != nullptr) {
          // left_debug_stop_at_approach second run: the full run's verified plan, not a re-search.
          found = *options.reuse_varying;
        } else if (!searchLeftVaryingOrientation(
            snapshot, joint_group, ik_state, cut_start, cut_end, transit_seed,
            lift_follows_varying, found, message, extension ? &*extension : nullptr)) {
          return false;
        }
        if (options.varying_plan_out != nullptr) *options.varying_plan_out = found;
        if (options.hover_start_joints_out != nullptr) {
          *options.hover_start_joints_out = found.hover_start_joints;
        }
        const auto station_pose = [&](const LeftVaryingStation& station) {
            auto pose = cut_start;
            pose.pose.position = toPoint(station.position);
            pose.pose.orientation = toMsg(station.orientation);
            return pose;
          };
        cut_start = station_pose(found.stations.front());
        cut_end = station_pose(found.stations.back());
        hover_start = cut_start;
        hover_end = cut_end;
        if (!offsetLeftPoseOutward(
            snapshot, hover_start, settings_.left_lateral_standoff_m, message) ||
            !offsetLeftPoseOutward(
            snapshot, hover_end, settings_.left_lateral_standoff_m, message)) {
          return false;
        }
        if (options.reuse_varying == nullptr) last_left_varying_summary_ = found.summary;
        retained_varying = found;
        varying_plan = std::move(found);
      }
      if (left_parallel && !left_varying) {
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
        if (varying_plan) {
          // Verified hover branch (see searchLeftVaryingOrientation::hover_for) as a joint goal.
          std::map<std::string, double> hover_goal;
          const auto& group_names = joint_group->getVariableNames();
          for (std::size_t index = 0; index < group_names.size(); ++index) {
            hover_goal.emplace(group_names[index], varying_plan->hover_start_joints.at(index));
          }
          transit_stages.push_back(addMoveTo(*task, stage_name, transit_planner, hover_goal));
        } else if (motion_profile == "chomp_pilz") {
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
      if (varying_plan) {
        // ONE contact stroke through the orientation knots: a single trajectory, time-
        // parameterised once with the contact scalings, fraction 1.0 for the whole path. The
        // stage keeps the fixed mode's name, so the contact validators and the stage list are
        // unchanged.
        std::vector<Eigen::Isometry3d> knot_poses;
        for (const auto& station : varying_plan->stations) {
          if (!station.is_knot) continue;
          Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
          pose.translation() = station.position;
          pose.linear() = station.orientation.toRotationMatrix();
          knot_poses.push_back(pose);
        }
        auto varying_stroke = std::make_shared<WaypointCartesianPath>(
          knot_poses, WaypointCartesianPath::Config{
            settings_.cartesian_step_m, 0.05, 5.0 * 3.14159265358979323846 / 180.0});
        varying_stroke->setIKFrame(settings_.tool_link);
        varying_stroke->setStepSize(settings_.cartesian_step_m);
        varying_stroke->setJumpThreshold(0.0);
        varying_stroke->setMinFraction(1.0);
        varying_stroke->setMaxVelocityScalingFactor(
          settings_.velocity_scaling * options.cut_stroke_speed_factor);
        varying_stroke->setMaxAccelerationScalingFactor(
          settings_.acceleration_scaling * options.cut_stroke_speed_factor);
        addMoveTo(
          *task, "cut " + segment.name + " [" + motion_profile + "]", varying_stroke, cut_end);
      } else {
        addMoveTo(
          *task, "cut " + segment.name + " [" + motion_profile + "]", cut_stroke_planner, cut_end);
      }
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
        const double extra_lift_m =
          transit_extra_lift_override_.value_or(settings_.middle_transit_extra_lift_m);
        if (lift_before_middle && extra_lift_m > 0.0) {
          auto raise_target = hover_end;
          const double raise_m = settings_.hover_clearance_m - settings_.cut_clearance_m +
            extra_lift_m;
          raise_target.pose.position.x += raise_m * snapshot.surface_normal.x;
          raise_target.pose.position.y += raise_m * snapshot.surface_normal.y;
          raise_target.pose.position.z += raise_m * snapshot.surface_normal.z;
          addMoveTo(
            *task, "raise to safe transit height after " + segment.name + " [cartesian]",
            planners.cartesian, raise_target);
        }
      }
    }

    if (options.return_home && !options.stop_at_hover_start) {
      std::map<std::string, double> home;
      if (!joint_group->getVariableDefaultPositions("home", home) ||
          home.size() != joint_group->getVariableCount()) {
        message = "return_home_after_cut: the SRDF has no complete 'home' state for " +
          settings_.planning_group;
        return false;
      }
      transit_stages.push_back(addMoveTo(
        *task, "transit to supplied home pose after the cut [" + motion_profile + "]",
        transit_planner, home, /*set_ik_frame=*/false));
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
    if (!validateCutStrokesStraight(solution, wrist_error)) {
      message = wrist_error;
      return false;
    }
    if (!validateTrajectoriesCollisionFree(solution, wrist_error)) {
      message = wrist_error;
      return false;
    }
    // The approach-only debug task (stop_at_hover_start) has no stroke; the full task it was cut
    // from passed this check before it was derived.
    if (retained_varying && !options.stop_at_hover_start &&
        !validateLeftVaryingRetainedStroke(solution, *retained_varying, wrist_error)) {
      message = wrist_error;
      return false;
    }
    if (options.final_joints_out != nullptr) {
      options.final_joints_out->clear();
      for (auto it = solution_message.sub_trajectory.rbegin();
           it != solution_message.sub_trajectory.rend(); ++it) {
        const auto& joint_trajectory = it->trajectory.joint_trajectory;
        if (joint_trajectory.points.empty()) continue;
        const auto& names = joint_trajectory.joint_names;
        const auto& positions = joint_trajectory.points.back().positions;
        if (names.size() != positions.size()) {
          message = "planned trajectory end state has mismatched joint names and positions";
          return false;
        }
        for (std::size_t index = 0; index < names.size(); ++index) {
          options.final_joints_out->emplace(names[index], positions[index]);
        }
        break;
      }
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

  // Every cut stroke (stage name "cut ...", all patterns) is a straight Cartesian line, but the
  // strokes run with jump detection off, so a wrist flip between two Cartesian waypoints (e.g.
  // joint5 crossing 0 along the top seam) is joint-interpolated into a large off-line swing
  // (2026-10-13 mock: middle_only reversed travel, tip 300 mm off the seam, tool tilted 106 deg;
  // only move_group's execution monitor stopped it). Fails closed when any retained waypoint's
  // tool_link position is more than kMaxCutStrokeDeviationM from the chord between the stroke's
  // first and last waypoint, or when a cut stroke has no trajectory.
  // Fail-closed: every waypoint of every retained stage, and states interpolated between
  // consecutive waypoints at <= kStepRad joint distance, must be collision-free in that stage's own
  // start scene (so the scoped knife/carton allowance applies only where the task set it). Catches
  // collisions a planner skipped between its checked states, which MoveIt's execution monitor
  // would otherwise only find while the arm is moving.
  bool validateTrajectoriesCollisionFree(const mtc::SolutionBase& solution, std::string& error) const
  {
    constexpr double kStepRad = 0.01;
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
    for (const auto* leaf : leaves) {
      const auto& trajectory = leaf->trajectory();
      if (!trajectory || trajectory->getWayPointCount() == 0U) continue;
      const auto* creator = leaf->creator();
      const std::string name = creator != nullptr ? creator->name() : std::string("unnamed stage");
      if (leaf->start() == nullptr || !leaf->start()->scene()) {
        error = "collision gate: stage '" + name + "' has no start scene; refusing the plan";
        return false;
      }
      const auto& scene = leaf->start()->scene();
      const auto* group = trajectory->getGroup();
      const std::string group_name = group != nullptr ? group->getName() : settings_.planning_group;
      moveit::core::RobotState probe(trajectory->getWayPoint(0));
      const auto colliding = [&](const moveit::core::RobotState& state) {
          collision_detection::CollisionRequest request;
          collision_detection::CollisionResult result;
          request.group_name = group_name;
          request.contacts = true;
          request.max_contacts = 1;
          scene->checkCollision(request, result, state);
          if (!result.collision) return std::string();
          for (const auto& contact : result.contacts) {
            return contact.first.first + " / " + contact.first.second;
          }
          return std::string("unknown bodies");
        };
      const std::size_t count = trajectory->getWayPointCount();
      for (std::size_t index = 0; index < count; ++index) {
        const auto& to = trajectory->getWayPoint(index);
        std::size_t steps = 1U;
        if (index > 0U) {
          const auto& from = trajectory->getWayPoint(index - 1U);
          const double distance = from.distance(to, group != nullptr ? group : to.getJointModelGroup(settings_.planning_group));
          steps = std::max<std::size_t>(1U, static_cast<std::size_t>(std::ceil(distance / kStepRad)));
          for (std::size_t step = 1; step < steps; ++step) {
            from.interpolate(to, static_cast<double>(step) / static_cast<double>(steps), probe);
            probe.update();
            const std::string bodies = colliding(probe);
            if (!bodies.empty()) {
              error = "collision gate: stage '" + name + "' collides (" + bodies + ") between waypoints " +
                std::to_string(index - 1U) + " and " + std::to_string(index) + "/" + std::to_string(count) +
                "; refusing the plan";
              return false;
            }
          }
        }
        probe = to;
        probe.update();
        const std::string bodies = colliding(probe);
        if (!bodies.empty()) {
          error = "collision gate: stage '" + name + "' collides (" + bodies + ") at waypoint " +
            std::to_string(index) + "/" + std::to_string(count) + "; refusing the plan";
          return false;
        }
      }
    }
    return true;
  }

  bool validateCutStrokesStraight(const mtc::SolutionBase& solution, std::string& error) const
  {
    constexpr double kMaxCutStrokeDeviationM = 0.003;
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
    for (const auto* leaf : leaves) {
      const auto* creator = leaf->creator();
      const std::string name = creator != nullptr ? creator->name() : std::string();
      if (name.rfind("cut ", 0) != 0U) continue;
      const auto& trajectory = leaf->trajectory();
      if (!trajectory || trajectory->getWayPointCount() == 0U) {
        error = "cut stroke '" + name + "' has no trajectory; refusing to guess";
        return false;
      }
      const std::size_t count = trajectory->getWayPointCount();
      const Eigen::Vector3d first =
        trajectory->getWayPoint(0).getGlobalLinkTransform(settings_.tool_link).translation();
      const Eigen::Vector3d last =
        trajectory->getWayPoint(count - 1).getGlobalLinkTransform(settings_.tool_link).translation();
      const Eigen::Vector3d chord = last - first;
      const double chord_sq = chord.squaredNorm();
      double worst = 0.0;
      std::size_t worst_index = 0U;
      for (std::size_t index = 0; index < count; ++index) {
        const Eigen::Vector3d point =
          trajectory->getWayPoint(index).getGlobalLinkTransform(settings_.tool_link).translation();
        const double t = chord_sq > 1e-12 ?
          std::min(1.0, std::max(0.0, (point - first).dot(chord) / chord_sq)) : 0.0;
        const double deviation = (point - (first + t * chord)).norm();
        if (deviation > worst) {
          worst = deviation;
          worst_index = index;
        }
      }
      if (!std::isfinite(worst) || worst > kMaxCutStrokeDeviationM) {
        error = "cut stroke '" + name + "' leaves its straight line by " +
          formatDouble(worst * 1000.0, 1) + " mm at waypoint " + std::to_string(worst_index) +
          "/" + std::to_string(count) + " (> " +
          formatDouble(kMaxCutStrokeDeviationM * 1000.0, 1) +
          " mm): joint-space jump (wrist flip) inside the contact stroke; refusing the plan";
        return false;
      }
    }
    return true;
  }

  // Varying-orientation left cut: checks the RETAINED contact-stroke trajectory itself (not just
  // the pre-planning replay). It must be the single "cut left side cut" leaf, keep every
  // waypoint inside the joint bounds and the left_parallel joint margin with no joint step above
  // 5 deg between waypoints, and pass through every planned station in order: the TCP polyline
  // within 1 mm and 1 deg of the station waypoint, on the replayed IK branch (every joint within
  // 0.2 rad of the station's recorded joints). Logs the stroke duration and path length.
  bool validateLeftVaryingRetainedStroke(
    const mtc::SolutionBase& solution, const LeftVaryingPlan& plan, std::string& error)
  {
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    constexpr double kMaxStepRad = 5.0 * kDeg + 1e-6;
    constexpr double kStationPositionTolM = 1e-3;
    constexpr double kStationAngleTolRad = 1.0 * kDeg;
    constexpr double kBranchTolRad = 0.2;
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
    const mtc::SubTrajectory* stroke = nullptr;
    for (const auto* leaf : leaves) {
      const auto* creator = leaf->creator();
      const std::string name = creator != nullptr ? creator->name() : std::string();
      if (name.rfind("cut left side cut", 0) != 0U) continue;
      if (stroke != nullptr) {
        error = "left varying plan has more than one left cut stroke stage; refusing";
        return false;
      }
      stroke = leaf;
    }
    if (stroke == nullptr || !stroke->trajectory() || stroke->trajectory()->getWayPointCount() < 2U) {
      error = "left varying plan has no usable single left cut stroke trajectory; refusing";
      return false;
    }
    if (!robot_model_ || plan.stations.size() < 2U) {
      error = "left varying retained-stroke validation lacks a robot model or stations";
      return false;
    }
    const auto* group = robot_model_->getJointModelGroup(settings_.planning_group);
    const auto* tool = robot_model_->getLinkModel(settings_.tool_link);
    if (group == nullptr || tool == nullptr) {
      error = "left varying retained-stroke validation: unknown planning group or tool link";
      return false;
    }
    const double margin = settings_.left_parallel_min_joint_margin_deg * kDeg;
    const auto& trajectory = *stroke->trajectory();
    const std::size_t count = trajectory.getWayPointCount();
    std::vector<Eigen::Isometry3d> tcp;
    std::vector<std::vector<double> > joints;
    std::vector<double> arc{0.0};
    for (std::size_t index = 0; index < count; ++index) {
      moveit::core::RobotState state(trajectory.getWayPoint(index));
      state.update();
      std::vector<double> q;
      state.copyJointGroupPositions(group, q);
      if (!state.satisfiesBounds(group)) {
        error = "left varying stroke waypoint " + std::to_string(index) + " violates joint bounds";
        return false;
      }
      const auto& bounds = group->getActiveJointModelsBounds();
      for (std::size_t j = 0; j < q.size() && j < bounds.size(); ++j) {
        const auto& b = (*bounds[j])[0];
        if (std::min(q[j] - b.min_position_, b.max_position_ - q[j]) < margin) {
          error = "left varying stroke waypoint " + std::to_string(index) + " has joint" +
            std::to_string(j + 1) + " inside the " +
            formatDouble(settings_.left_parallel_min_joint_margin_deg, 1) + " deg joint margin";
          return false;
        }
      }
      if (!joints.empty()) {
        for (std::size_t j = 0; j < q.size(); ++j) {
          if (std::abs(q[j] - joints.back()[j]) > kMaxStepRad) {
            error = "left varying stroke moves joint" + std::to_string(j + 1) + " " +
              formatDouble(std::abs(q[j] - joints.back()[j]) / kDeg, 1) +
              " deg between two waypoints (> 5 deg): branch change during contact; refusing";
            return false;
          }
        }
        arc.push_back(arc.back() +
          (state.getGlobalLinkTransform(tool).translation() - tcp.back().translation()).norm());
      }
      tcp.push_back(state.getGlobalLinkTransform(tool));
      joints.push_back(q);
    }
    double previous_arc = -1.0;
    for (const auto& station : plan.stations) {
      double best = std::numeric_limits<double>::infinity();
      double best_arc = 0.0;
      Eigen::Quaterniond best_orientation = Eigen::Quaterniond::Identity();
      std::vector<double> best_joints = joints.front();
      for (std::size_t w = 0; w + 1U < count; ++w) {
        const Eigen::Vector3d a = tcp[w].translation();
        const Eigen::Vector3d seg = tcp[w + 1U].translation() - a;
        const double len2 = seg.squaredNorm();
        const double u = len2 > 1e-18 ? std::clamp((station.position - a).dot(seg) / len2, 0.0, 1.0) : 0.0;
        const double dist = (a + u * seg - station.position).norm();
        if (dist < best) {
          best = dist;
          best_arc = arc[w] + u * std::sqrt(len2);
          best_orientation = Eigen::Quaterniond(tcp[w].linear()).slerp(
            u, Eigen::Quaterniond(tcp[w + 1U].linear()));
          for (std::size_t j = 0; j < best_joints.size(); ++j) {
            best_joints[j] = joints[w][j] + u * (joints[w + 1U][j] - joints[w][j]);
          }
        }
      }
      const std::string label = "station " + std::to_string(station.station_index);
      if (best > kStationPositionTolM) {
        error = "left varying stroke misses " + label + " by " + formatDouble(best * 1000.0, 2) +
          " mm (> 1 mm); refusing";
        return false;
      }
      const double angle = best_orientation.angularDistance(station.orientation);
      if (angle > kStationAngleTolRad) {
        error = "left varying stroke orientation at " + label + " is " +
          formatDouble(angle / kDeg, 2) + " deg off the planned profile (> 1 deg); refusing";
        return false;
      }
      for (std::size_t j = 0; j < best_joints.size() && j < station.joints.size(); ++j) {
        if (std::abs(best_joints[j] - station.joints[j]) > kBranchTolRad) {
          error = "left varying stroke leaves the verified IK branch at " + label + " (joint" +
            std::to_string(j + 1) + " differs by " +
            formatDouble(std::abs(best_joints[j] - station.joints[j]), 3) + " rad); refusing";
          return false;
        }
      }
      if (best_arc < previous_arc - 1e-6) {
        error = "left varying stroke passes " + label + " out of order; refusing";
        return false;
      }
      previous_arc = best_arc;
    }
    const double duration = trajectory.getDuration();
    RCLCPP_INFO(
      get_logger(),
      "Left varying retained stroke OK: %zu waypoints, TCP path %.1f mm, duration %.2f s; %zu "
      "stations within 1 mm / 1 deg on the verified branch",
      count, arc.back() * 1000.0, duration, plan.stations.size());
    last_left_varying_summary_ += "; contact stroke " + formatDouble(duration, 1) + " s (" +
      std::to_string(count) + " waypoints, one trajectory)";
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
    // MTC checks the interpolated path only every max_step (default 0.1 rad): on a long transit the
    // handpiece moved through base_link between two checked states (mock 2026-10-14, t_cut +40 deg).
    direct->properties().set("max_step", 0.01);
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
  // Set only by planSnapshot's transit-lift retry loop around planCutSegments.
  std::optional<double> transit_extra_lift_override_;
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
  // link6 -> settings_.tool_link, derived from the live RobotModel by ensureAnalyticIkSelfCheck
  // (written once inside the call_once, read-only afterwards).
  Eigen::Isometry3d analytic_tool_offset_{Eigen::Isometry3d::Identity()};
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
  // Fraction of the middle seam the retained plan cuts (1.0 unless shortened or a half seam).
  double retained_middle_fraction_{1.0};
  // Plan export / plan from state (retainPlan fills these; clearRetainedPlan resets them).
  std::uint64_t plan_seq_{0};
  const mtc::Task* exportable_task_{nullptr};  // == planned_task_.get() only for a retainPlan() plan
  std::vector<geometry_msgs::msg::PoseStamped> planned_cut_targets_;
  std::vector<CutSegment> planned_segments_;
  std::string retained_summary_, retained_cut_pattern_;
  bool retained_from_override_{false};
  std::optional<StartStateOverride> start_state_override_;  // set only inside planHCutFromState
  rclcpp::Service<dual_cut_msgs::srv::GetPlannedSolution>::SharedPtr get_planned_solution_service_;
  rclcpp::Service<dual_cut_msgs::srv::PlanFromState>::SharedPtr plan_from_state_service_;
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
  // One-line summary of the last varying-orientation left plan (empty when not used).
  std::string last_left_varying_summary_;
  bool side_tape_handoff_pending_{false};
  SideTapeHandoffInfo side_tape_handoff_info_;
  std::optional<Snapshot> side_tape_handoff_snapshot_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr side_tape_handoff_pub_;
  TriggerService::SharedPtr side_tape_handoff_plan_service_;
  TriggerService::SharedPtr side_tape_retract_service_;
  TriggerService::SharedPtr side_tape_continue_service_;
  TriggerService::SharedPtr side_cut_penetrate_service_;
  // One-call T cut (plan_t_cut / execute_t_cut). t_cut_plan_ready_: the retained plan is a
  // plan_t_cut result (cleared by clearRetainedPlan and consumed by execute_t_cut).
  // t_cut_running_ is written under plan_mutex_ only; the sequencer thread releases
  // plan_mutex_ while it waits for a trajectory so completion handlers can take it.
  bool t_cut_plan_ready_{false};
  std::atomic<bool> t_cut_running_{false};
  std::atomic<bool> t_cut_abort_{false};
  std::thread t_cut_thread_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr t_cut_status_pub_;
  TriggerService::SharedPtr t_cut_plan_service_, t_cut_execute_service_, t_cut_abort_service_;
  TopCutStagedInfo top_cut_info_;
  std::optional<Snapshot> top_cut_snapshot_;
  bool top_cut_pending_{false};
  // The retained plan is a left_debug_stop_at_approach approach-only debug plan: executing it
  // needs debug_motion_execution_enabled in addition to execution_enabled. Reset by
  // clearRetainedPlan() and consumed at dispatch.
  bool retained_debug_motion_{false};
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
  if (const std::string error = node->checkCartonContactLinksInModel(); !error.empty()) {
    RCLCPP_FATAL(node->get_logger(), "refusing to start: %s", error.c_str());
    node.reset();
    rclcpp::shutdown();
    return 2;
  }
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
