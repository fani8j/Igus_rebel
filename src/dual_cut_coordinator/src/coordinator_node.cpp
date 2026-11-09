// dual_cut_coordinator: parameters, model / scene access, ~/plan, main(). Execution, stop and rearm are
// in execution.cpp, the monitor / status / joint-state handling in monitor.cpp.
#include "dual_cut_coordinator/coordinator.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <iomanip>
#include <regex>
#include <sstream>

#include <geometric_shapes/shapes.h>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <srdfdom/model.h>
#include <urdf/model.h>

namespace dual_cut
{
using namespace std::chrono_literals;

namespace
{
std::string jsonEscape(const std::string & in)
{
  std::string out;
  for (char c : in)
  {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) out += ' ';
    else out += c;
  }
  return out;
}

rclcpp::ParameterValue inferValue(const std::string & text)
{
  static const std::regex int_re("^[-+]?[0-9]+$");
  static const std::regex dbl_re("^[-+]?([0-9]+\\.?[0-9]*|\\.[0-9]+)([eE][-+]?[0-9]+)?$");
  if (text == "true") return rclcpp::ParameterValue(true);
  if (text == "false") return rclcpp::ParameterValue(false);
  if (std::regex_match(text, int_re)) return rclcpp::ParameterValue(static_cast<int64_t>(std::stoll(text)));
  if (std::regex_match(text, dbl_re)) return rclcpp::ParameterValue(std::stod(text));
  return rclcpp::ParameterValue(text);
}
}  // namespace

// ---------------------------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------------------------

void CoordinatorNode::declareParameters()
{
  // --- planning / validation (rejected while planning or executing) ---
  declare_parameter<double>("swept_padding_m", 0.05);
  declare_parameter<double>("exact_distance_tolerance_m", SweptVolumeConfig{}.exact_distance_tolerance_m);
  declare_parameter<double>("tip_on_line_tolerance_m", 0.002);
  // 0 = any contact outside the allowed tool/carton contact is a collision (default). Model-coverage escape hatch only.
  declare_parameter<double>("world_penetration_tolerance_m", 0.0);
  // Table clearance gate of the world check (2026-10-22): every sample must keep every link of the checked arm, except
  // the exempt ones, at least this far from the table link of the dual URDF. 0 = off. The exempt links
  // (<arm prefix> + suffix) are the base links, which sit 1 mm / 3.9 mm above the table by design.
  declare_parameter<double>("table_min_clearance_m", 0.010);
  declare_parameter<std::string>("table_link_name", "table");
  declare_parameter<std::vector<std::string>>("table_exempt_link_suffixes", {"base_link"});
  declare_parameter<double>("frame_tolerance", 1e-6);
  declare_parameter<double>("external_snapshot_max_age_s", 240.0);
  declare_parameter<double>("sample_dt_s", 0.05);
  declare_parameter<double>("max_joint_step_rad", 0.01);
  declare_parameter<double>("executor_call_timeout_s", 200.0);
  declare_parameter<bool>("allow_serial_fallback", true);
  declare_parameter<bool>("require_planning_scene", true);
  declare_parameter<std::string>("carton_object_id", "detected_carton");
  declare_parameter<bool>("require_carton", true);
  declare_parameter<double>("carton_consistency_tol_m", 0.002);
  declare_parameter<double>("carton_consistency_tol_rad", 0.005);
  declare_parameter<std::vector<std::string>>("contact_object_ids", {"detected_carton"});
  declare_parameter<std::vector<std::string>>(
    "contact_link_suffixes", {"knife_c_holder", "knife_c_handpiece", "knife_c_wrist_camera", "knife_c_blade", "xeg32_tool_tip"});
  declare_parameter<std::vector<std::string>>(
    "left_phase_a_params", {"cut_pattern=middle_only", "middle_seam_stop=center"});
  declare_parameter<std::vector<std::string>>(
    "left_phase_b_params", {"cut_pattern=left_only", "middle_seam_stop=end"});
  declare_parameter<std::vector<std::string>>(
    "right_phase_a_params", {"cut_pattern=left_only", "middle_seam_stop=end"});
  declare_parameter<std::vector<std::string>>(
    "right_phase_b_params", {"cut_pattern=middle_only", "middle_seam_stop=center"});
  // --- top-seam split between the arms (planning scope; see top_split.hpp) ---
  declare_parameter<std::string>("phase_order", "top_first");
  // automatic re-plan after a random executor plan-stage failure (never after a validator refusal)
  declare_parameter<int>("plan_retries", 2);
  // The collision-gate entries match only free-space transits (OMPL paths, random; fallback_speed 2026-10-24 t074923_a:
  // a right approach hit the gate once, its 4 repeats passed); a colliding Cartesian stroke is deterministic, not retried.
  declare_parameter<std::vector<std::string>>(
    "plan_retry_signatures", {"winds the wrist", "MTC candidate failed", "no planner solution",
                              "collision gate: stage 'approach pose", "collision gate: stage 'transition from",
                              "collision gate: stage 'transit to"});
  // Order of the retries (2026-10-24): breadth first = every plan_fallback_chain entry is tried once before any entry is
  // retried; only a refusal whose text contains one of plan_inplace_retry_labels is retried at once, in place (data
  // 2026-10-22: the retry of "right job 2 (right face)" succeeded 4/4, the retries of the left top descent / right top /
  // left face after the top 0/70). If nothing passed, the entries are retried in chain order. false = every random-looking
  // refusal is retried in place first (the former behaviour).
  declare_parameter<bool>("plan_retry_breadth_first", true);
  declare_parameter<std::vector<std::string>>("plan_inplace_retry_labels", {"right job 2 (right face)"});
  // order fallback: when the requested plan is refused, try the next entry of this chain (only for refusals an order
  // change can fix, see classifyRefusal); the first plan that passes every check is kept. Entries: sequence,
  // sequence_left_top_first, phases_top_first, phases_side_first. The chain starts at the entry of the configured
  // schedule_mode / phase_order. plan_fallback_budget_s: no further attempt starts after this much total planning time, and no
  // random-failure retry (plan_retries) after half of it.
  declare_parameter<std::vector<std::string>>(
    "plan_fallback_chain", std::vector<std::string>{"sequence", "sequence_left_top_first", "phases_top_first", "phases_side_first"});
  declare_parameter<double>("plan_fallback_budget_s", 150.0);
  // detection permission (snapshot cut_permission): a blocked detection plans but ~/execute is refused unless
  // allow_blocked_detection (runtime) is true or the detector's own override is set and honored.
  declare_parameter<bool>("allow_blocked_detection", false);
  declare_parameter<bool>("honor_detection_override", false);
  declare_parameter<std::string>("top_split_mode", "left_max");
  declare_parameter<double>("top_split_overlap_m", TopSplitParams{}.overlap_m);
  declare_parameter<double>("top_split_right_min_tail", TopSplitParams{}.right_min_tail);
  declare_parameter<double>("top_split_left_min_fraction", TopSplitParams{}.left_min_fraction);
  declare_parameter<double>("top_split_min_overlap_m", TopSplitParams{}.min_overlap_m);
  declare_parameter<double>("top_split_end_tolerance_m", TopSplitParams{}.end_tolerance_m);
  declare_parameter<double>("top_split_skip_right_top_fraction", TopSplitParams{}.skip_right_top_fraction);
  declare_parameter<double>("top_split_right_end_extension_m", TopSplitParams{}.right_end_extension_m);
  declare_parameter<std::string>("top_split_snapshot_file", TopSplitParams{}.snapshot_file);
  // --- schedule mode (planning scope) ---
  declare_parameter<std::string>("schedule_mode", "phases");
  declare_parameter<double>("sequence_skew_margin_s", 2.0);
  declare_parameter<double>("sequence_delay_step_s", 1.0);
  declare_parameter<double>("sequence_max_delay_s", 60.0);
  declare_parameter<double>("sequence_dispatch_lead_s", 0.05);
  // --- monitor / barriers (runtime) ---
  declare_parameter<double>("monitor_stop_distance_m", 0.03);
  declare_parameter<double>("joint_state_stale_s", 0.4);
  declare_parameter<double>("tracking_stop_rad", 0.2);
  declare_parameter<double>("tracking_warn_rad", 0.1);
  declare_parameter<double>("barrier_joint_state_age_s", 0.1);
  declare_parameter<double>("barrier_start_tolerance_rad", 0.01);
  declare_parameter<double>("barrier_rest_velocity", 0.03);
  declare_parameter<double>("barrier_rest_duration_s", 0.5);
  declare_parameter<double>("barrier_timeout_s", 8.0);
  declare_parameter<double>("start_delay_s", 0.25);
  declare_parameter<int>("expected_hardware_components", 2);
  // --- execution enables (all default false) ---
  declare_parameter<bool>("execution_enabled", false);
  declare_parameter<bool>("execute_phase_a_enabled", false);
  declare_parameter<bool>("execute_phase_b_enabled", false);
  // --- wiring (read only) ---
  declare_parameter<std::string>("left_executor_ns", "/arm_left/box_h_cut_executor");
  declare_parameter<std::string>("right_executor_ns", "/arm_right/box_h_cut_executor");
  declare_parameter<std::string>("left_scene_ns", "/arm_left");   // namespace of the per-arm move_group
  declare_parameter<std::string>("right_scene_ns", "/arm_right");
  declare_parameter<std::string>("left_controller", "arm_left_trajectory_controller");
  declare_parameter<std::string>("right_controller", "arm_right_trajectory_controller");
  declare_parameter<std::string>("controller_manager", "/controller_manager");
  declare_parameter<std::string>("model_node", "/move_group");
  declare_parameter<std::string>("planning_scene_service", "/get_planning_scene");
  declare_parameter<std::string>("display_topic", "/display_planned_path");
  declare_parameter<std::string>("expected_planning_frame", "base_link");
  declare_parameter<std::string>("expected_tool_link", "xeg32_tool_tip");
  declare_parameter<std::string>("tip_link_suffix", "xeg32_tool_tip");
  declare_parameter<std::string>("robot_description", "");
  declare_parameter<std::string>("robot_description_semantic", "");
  declare_parameter<double>("monitor_rate_hz", 20.0);

  // --- robot-side joint move (CRI MoveJoint) for eligible free-space transit legs: OPT-IN, default OFF ---
  // Planning scope (rejected while planning / executing), out-of-range values are rejected, never clamped. Phases mode,
  // SERIAL phases only (stage 1). See cri_transit.hpp and the dual_cut_bringup README.
  declare_parameter<bool>("cri_transit_enabled", false);
  declare_parameter<double>("cri_transit_velocity_percent", 20.0);       // (0, 25]
  declare_parameter<double>("cri_transit_acceleration_percent", 30.0);   // (0, 40]
  declare_parameter<double>("cri_transit_min_length_rad", 0.05);
  declare_parameter<double>("cri_transit_path_tolerance_rad", 0.005);   // measured real deviation <= 0.9 mrad
  declare_parameter<double>("cri_transit_goal_tolerance_rad", 0.005);
  declare_parameter<double>("cri_transit_start_tolerance_rad", 0.01);
  declare_parameter<int>("cri_transit_max_runs_per_arm_phase", 1);
  declare_parameter<double>("cri_driver_max_velocity_cap_percent", 25.0);
  declare_parameter<double>("cri_driver_max_acceleration_cap_percent", 40.0);
  declare_parameter<bool>("cri_assume_driver_cap_ok", false);           // until HardwareState carries the effective caps
  declare_parameter<double>("cri_transit_reach_m", 0.0);                 // 0 = derive the rigorous bound from the model (default)
  // JTC sub-chords either side of a partial CRI stretch: rest-to-rest quintics, leading joint within these limits
  // (= transit_velocity_scaling 0.20 * 0.785 rad/s and transit_acceleration_scaling 0.10 * 5.24 rad/s^2 of box_h_cut_executor.yaml)
  declare_parameter<double>("cri_subchord_velocity_rad_s", 0.157);
  declare_parameter<double>("cri_subchord_acceleration_rad_s2", 0.400);
  // wiring (read only). TODO-confirm: node_name igus_rebel_left / igus_rebel_right from rebel_dual_table.urdf.xacro
  declare_parameter<std::string>("cri_left_move_joint_action", "/igus_rebel_left/move_joint");
  declare_parameter<std::string>("cri_right_move_joint_action", "/igus_rebel_right/move_joint");
  declare_parameter<std::string>("cri_left_hardware_state_topic", "/igus_rebel_left/hardware_state");
  declare_parameter<std::string>("cri_right_hardware_state_topic", "/igus_rebel_right/hardware_state");
  declare_parameter<std::string>("cri_left_robot_joint_states_topic", "/igus_rebel_left/robot_joint_states");
  declare_parameter<std::string>("cri_right_robot_joint_states_topic", "/igus_rebel_right/robot_joint_states");
}

rcl_interfaces::msg::SetParametersResult CoordinatorNode::onSetParameters(const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult r;
  r.successful = true;
  static const std::vector<std::string> read_only = {
    "left_executor_ns", "right_executor_ns", "left_scene_ns", "right_scene_ns", "carton_object_id", "left_controller", "right_controller", "controller_manager",
    "model_node", "planning_scene_service", "display_topic", "expected_planning_frame", "expected_tool_link",
    "tip_link_suffix", "robot_description", "robot_description_semantic", "monitor_rate_hz",
    "expected_hardware_components", "table_link_name", "table_exempt_link_suffixes",
    "cri_left_move_joint_action", "cri_right_move_joint_action", "cri_left_hardware_state_topic", "cri_right_hardware_state_topic",
    "cri_left_robot_joint_states_topic", "cri_right_robot_joint_states_topic"};
  static const std::vector<std::string> runtime = {
    "execution_enabled", "execute_phase_a_enabled", "execute_phase_b_enabled", "monitor_stop_distance_m",
    "joint_state_stale_s", "tracking_stop_rad", "tracking_warn_rad", "barrier_joint_state_age_s",
    "barrier_start_tolerance_rad", "barrier_rest_velocity", "barrier_rest_duration_s", "barrier_timeout_s",
    "start_delay_s", "external_snapshot_max_age_s", "allow_blocked_detection"};
  {
    // top_split_*: validate the resulting set as a whole (ranges, mode, absolute .json path)
    TopSplitParams cand = topSplitParams();
    bool touched = false;
    for (const auto & p : params)
    {
      const std::string & n = p.get_name();
      if (n.rfind("top_split_", 0) != 0) continue;
      touched = true;
      if (n == "top_split_mode")
      {
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_STRING) { r.successful = false; r.reason = n + " must be a string"; return r; }
        const Status ms = topSplitModeFromString(p.as_string(), cand.mode);
        if (!ms) { r.successful = false; r.reason = ms.message; return r; }
      }
      else if (n == "top_split_snapshot_file")
      {
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_STRING) { r.successful = false; r.reason = n + " must be a string"; return r; }
        cand.snapshot_file = p.as_string();
      }
      else if (p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
      {
        const double v = p.as_double();
        if (n == "top_split_overlap_m") cand.overlap_m = v;
        else if (n == "top_split_right_min_tail") cand.right_min_tail = v;
        else if (n == "top_split_left_min_fraction") cand.left_min_fraction = v;
        else if (n == "top_split_min_overlap_m") cand.min_overlap_m = v;
        else if (n == "top_split_end_tolerance_m") cand.end_tolerance_m = v;
        else if (n == "top_split_skip_right_top_fraction") cand.skip_right_top_fraction = v;
        else if (n == "top_split_right_end_extension_m") cand.right_end_extension_m = v;
      }
    }
    if (touched && (planning_.load() || executing_.load()))
    {
      r.successful = false;
      r.reason = "top_split_* rejected: busy (planning or executing)";
      return r;
    }
    if (touched)
    {
      const Status vs = validateTopSplitParams(cand);
      if (!vs) { r.successful = false; r.reason = vs.message; return r; }
    }
  }
  for (const auto & p : params)
  {
    const std::string & n = p.get_name();
    if (n == "plan_retries")
    {
      if (p.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER || p.as_int() < 0 || p.as_int() > 5)
      {
        r.successful = false;
        r.reason = "plan_retries must be an integer within [0, 5]";
        return r;
      }
    }
    else if (n == "phase_order")
    {
      if (p.get_type() != rclcpp::ParameterType::PARAMETER_STRING || (p.as_string() != "top_first" && p.as_string() != "side_first"))
      {
        r.successful = false;
        r.reason = "phase_order must be 'top_first' or 'side_first'";
        return r;
      }
    }
    else if (n == "plan_fallback_chain")
    {
      std::vector<PlanVariant> probe;
      if (p.get_type() != rclcpp::ParameterType::PARAMETER_STRING_ARRAY)
      {
        r.successful = false;
        r.reason = "plan_fallback_chain must be a string array";
        return r;
      }
      const Status cs = buildPlanChain(p.as_string_array(), p.as_string_array().empty() ? std::string("sequence") : p.as_string_array().front(), probe);
      if (!cs) { r.successful = false; r.reason = cs.message; return r; }
    }
    else if (n == "plan_fallback_budget_s")
    {
      if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE || !std::isfinite(p.as_double()) || p.as_double() < 10.0 || p.as_double() > 3000.0)
      {
        r.successful = false;
        r.reason = "plan_fallback_budget_s must be a number within [10, 3000]";
        return r;
      }
    }
    else if (n == "schedule_mode")
    {
      if (p.get_type() != rclcpp::ParameterType::PARAMETER_STRING || (p.as_string() != "phases" && p.as_string() != "sequence"))
      {
        r.successful = false;
        r.reason = "schedule_mode must be 'phases' or 'sequence'";
        return r;
      }
    }
    else if (n.rfind("sequence_", 0) == 0 && p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
    {
      const double v = p.as_double();
      const bool ok = std::isfinite(v) && (n == "sequence_skew_margin_s" ? (v >= 0.5 && v <= 10.0)
                                           : n == "sequence_delay_step_s" ? (v >= 0.1 && v <= 10.0)
                                           : n == "sequence_max_delay_s" ? (v >= 0.0 && v <= 600.0)
                                           : n == "sequence_dispatch_lead_s" ? (v >= 0.0 && v <= 1.0) : true);
      if (!ok)
      {
        r.successful = false;
        r.reason = n + " out of range (skew margin [0.5, 10] s, delay step [0.1, 10] s, max delay [0, 600] s, dispatch lead [0, 1] s)";
        return r;
      }
    }
    if (n.rfind("cri_", 0) == 0)
    {
      // typed range check; out of range is REJECTED (never clamped)
      Status cs = Status::Ok();
      if (n == "cri_transit_enabled" || n == "cri_assume_driver_cap_ok")
      {
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) cs = Status::Error(n + " must be a bool");
      }
      else if (n == "cri_transit_max_runs_per_arm_phase")
      {
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) cs = Status::Error(n + " must be an integer");
        else cs = validateCriParameter(n, static_cast<int64_t>(p.as_int()));
      }
      else if (n.rfind("cri_transit_", 0) == 0 || n.rfind("cri_driver_", 0) == 0 || n.rfind("cri_subchord_", 0) == 0)
      {
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) cs = Status::Error(n + " must be a double");
        else cs = validateCriParameter(n, p.as_double());
      }
      if (!cs)
      {
        r.successful = false;
        r.reason = cs.message;
        return r;
      }
    }
    if (std::find(read_only.begin(), read_only.end(), n) != read_only.end())
    {
      r.successful = false;
      r.reason = n + " is read-only (edit the config and restart)";
      return r;
    }
    const bool is_runtime = std::find(runtime.begin(), runtime.end(), n) != runtime.end();
    if (!is_runtime && (planning_.load() || executing_.load()))
    {
      r.successful = false;
      r.reason = n + " rejected: busy (planning or executing)";
      return r;
    }
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
    {
      const double v = p.as_double();
      const bool positive = n == "exact_distance_tolerance_m" || n == "sample_dt_s" || n == "max_joint_step_rad" ||
                            n == "external_snapshot_max_age_s" || n == "tip_on_line_tolerance_m" ||
                            n == "monitor_stop_distance_m" || n == "joint_state_stale_s" || n == "tracking_stop_rad" ||
                            n == "barrier_joint_state_age_s" || n == "barrier_rest_duration_s";
      if (!std::isfinite(v) || v < 0.0 || (positive && v <= 0.0))
      {
        r.successful = false;
        r.reason = n + " must be finite and " + (positive ? "> 0" : ">= 0");
        return r;
      }
    }
  }
  return r;
}

CoordinatorNode::CoordinatorNode(const rclcpp::NodeOptions & options) : rclcpp::Node("dual_cut_coordinator", options)
{
  declareParameters();
  param_cb_ = add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & p) { return onSetParameters(p); });

  cbg_plan_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbg_exec_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbg_stop_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);   // own group: stop never waits for planning
  cbg_monitor_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbg_js_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbg_io_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  for (auto * a : {&left_, &right_})
  {
    a->side = a == &left_ ? "left" : "right";
    a->prefix = armPrefix(a->side);
    a->executor_ns = ps(a == &left_ ? "left_executor_ns" : "right_executor_ns");
    a->controller = ps(a == &left_ ? "left_controller" : "right_controller");
    a->scene_ns = ps(a == &left_ ? "left_scene_ns" : "right_scene_ns");
    a->set_params = create_client<rcl_interfaces::srv::SetParameters>(
      a->executor_ns + "/set_parameters", rmw_qos_profile_services_default, cbg_io_);
    a->get_params = create_client<rcl_interfaces::srv::GetParameters>(
      a->executor_ns + "/get_parameters", rmw_qos_profile_services_default, cbg_io_);
    a->scene_client = create_client<moveit_msgs::srv::GetPlanningScene>(
      a->scene_ns + "/get_planning_scene", rmw_qos_profile_services_default, cbg_io_);
    a->plan_from_state = create_client<dual_cut_msgs::srv::PlanFromState>(
      a->executor_ns + "/plan_h_cut_from_state", rmw_qos_profile_services_default, cbg_io_);
    a->get_plan = create_client<dual_cut_msgs::srv::GetPlannedSolution>(
      a->executor_ns + "/get_planned_solution", rmw_qos_profile_services_default, cbg_io_);
    a->action = rclcpp_action::create_client<FJT>(this, "/" + a->controller + "/follow_joint_trajectory", cbg_io_);
    a->hold_pub = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/" + a->controller + "/joint_trajectory", rclcpp::QoS(10).reliable());
    // Robot-side joint move: the action client and the two telemetry subscriptions are created always but are passive:
    // nothing reads them, no gate depends on them and no goal is sent unless cri_transit_enabled is true.
    const char * side = a == &left_ ? "left" : "right";
    const size_t idx = a == &left_ ? 0 : 1;
    a->cri_action = rclcpp_action::create_client<MoveJoint>(this, ps(("cri_" + std::string(side) + "_move_joint_action").c_str()), cbg_io_);
    rclcpp::SubscriptionOptions telemetry_opts;
    telemetry_opts.callback_group = cbg_js_;
    a->cri_hw_sub = create_subscription<HardwareState>(
      ps(("cri_" + std::string(side) + "_hardware_state_topic").c_str()), rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
      [this, idx](const HardwareState::SharedPtr m) { onHardwareState(idx, m); }, telemetry_opts);
    a->cri_js_sub = create_subscription<sensor_msgs::msg::JointState>(
      ps(("cri_" + std::string(side) + "_robot_joint_states_topic").c_str()), rclcpp::SensorDataQoS().keep_last(10),
      [this, idx](const sensor_msgs::msg::JointState::SharedPtr m) { onRobotJointStates(idx, m); }, telemetry_opts);
  }
  scene_client_ = create_client<moveit_msgs::srv::GetPlanningScene>(
    ps("planning_scene_service"), rmw_qos_profile_services_default, cbg_io_);
  list_controllers_ = create_client<controller_manager_msgs::srv::ListControllers>(
    ps("controller_manager") + "/list_controllers", rmw_qos_profile_services_default, cbg_io_);
  list_hardware_ = create_client<controller_manager_msgs::srv::ListHardwareComponents>(
    ps("controller_manager") + "/list_hardware_components", rmw_qos_profile_services_default, cbg_io_);
  switch_controller_ = create_client<controller_manager_msgs::srv::SwitchController>(
    ps("controller_manager") + "/switch_controller", rmw_qos_profile_services_default, cbg_io_);

  srv_plan_ = create_service<Trigger>(
    "~/plan", [this](const std::shared_ptr<Trigger::Request> q, std::shared_ptr<Trigger::Response> r) { handlePlan(q, r); },
    rmw_qos_profile_services_default, cbg_plan_);
  srv_execute_ = create_service<Trigger>(
    "~/execute", [this](const std::shared_ptr<Trigger::Request> q, std::shared_ptr<Trigger::Response> r) { handleExecute(q, r); },
    rmw_qos_profile_services_default, cbg_exec_);
  srv_stop_ = create_service<Trigger>(
    "~/stop", [this](const std::shared_ptr<Trigger::Request> q, std::shared_ptr<Trigger::Response> r) { handleStop(q, r); },
    rmw_qos_profile_services_default, cbg_stop_);
  srv_rearm_ = create_service<Trigger>(
    "~/rearm", [this](const std::shared_ptr<Trigger::Request> q, std::shared_ptr<Trigger::Response> r) { handleRearm(q, r); },
    rmw_qos_profile_services_default, cbg_exec_);

  report_pub_ = create_publisher<std_msgs::msg::String>("~/plan_report", rclcpp::QoS(1).reliable().transient_local());
  exec_report_pub_ = create_publisher<std_msgs::msg::String>("~/exec_report", rclcpp::QoS(20).reliable().transient_local());
  status_pub_ = create_publisher<Status_>("~/status", rclcpp::QoS(10).reliable());
  display_pub_ = create_publisher<moveit_msgs::msg::DisplayTrajectory>(ps("display_topic"), rclcpp::QoS(10).reliable());

  rclcpp::SubscriptionOptions sub_opts;
  sub_opts.callback_group = cbg_js_;
  js_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", rclcpp::QoS(10), [this](const sensor_msgs::msg::JointState::SharedPtr m) { onJointState(m); }, sub_opts);

  const double rate = std::max(1.0, pd("monitor_rate_hz"));
  monitor_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this]() { monitorTick(); }, cbg_monitor_);
  poll_timer_ = create_wall_timer(200ms, [this]() { pollTick(); }, cbg_io_);

  stop_worker_ = std::thread([this]() { stopWorkerLoop(); });

  RCLCPP_INFO(
    get_logger(), "dual_cut_coordinator ready (executors %s, %s; controllers %s, %s). execution_enabled=%s",
    left_.executor_ns.c_str(), right_.executor_ns.c_str(), left_.controller.c_str(), right_.controller.c_str(),
    pb("execution_enabled") ? "true" : "false");
}

CoordinatorNode::~CoordinatorNode()
{
  stop_latched_ = true;   // let a running execution thread leave
  if (exec_thread_.joinable()) exec_thread_.join();
  {
    std::lock_guard<std::mutex> lk(stop_req_mutex_);
    stop_worker_quit_ = true;
  }
  stop_req_cv_.notify_all();
  if (stop_worker_.joinable()) stop_worker_.join();
}

// ---------------------------------------------------------------------------------------------
// model and scene
// ---------------------------------------------------------------------------------------------

Status CoordinatorNode::ensureModel()
{
  std::lock_guard<std::mutex> lk(model_mutex_);
  if (model_) return Status::Ok();
  std::string urdf_xml = ps("robot_description");
  std::string srdf_xml = ps("robot_description_semantic");
  if (urdf_xml.empty() || srdf_xml.empty())
  {
    const std::string node = ps("model_node");
    auto client = std::make_shared<rclcpp::AsyncParametersClient>(
      shared_from_this(), node, rmw_qos_profile_parameters, cbg_io_);
    if (!client->wait_for_service(10s)) return Status::Error("parameter service of " + node + " not available");
    auto fut = client->get_parameters({"robot_description", "robot_description_semantic"});
    if (fut.wait_for(15s) != std::future_status::ready)
      return Status::Error("timeout reading robot_description from " + node);
    const auto values = fut.get();
    if (values.size() != 2 || values[0].as_string().empty() || values[1].as_string().empty())
      return Status::Error(node + " has no robot_description / robot_description_semantic");
    urdf_xml = values[0].as_string();
    srdf_xml = values[1].as_string();
  }
  auto u = std::make_shared<urdf::Model>();
  if (!u->initString(urdf_xml)) return Status::Error("cannot parse the URDF");
  auto s = std::make_shared<srdf::Model>();
  if (!s->initString(*u, srdf_xml)) return Status::Error("cannot parse the SRDF");
  auto m = std::make_shared<moveit::core::RobotModel>(urdf::ModelInterfaceSharedPtr(u), s);
  Status st = makeArmSpec(*m, "left", left_.spec);
  if (!st) return st;
  st = makeArmSpec(*m, "right", right_.spec);
  if (!st) return st;
  model_ = m;
  monitor_scene_ = std::make_shared<planning_scene::PlanningScene>(model_);
  SweptVolumeConfig cfg;
  cfg.padding_m = 0.0;
  cfg.exact_distance_tolerance_m = pd("exact_distance_tolerance_m");
  st = SweptVolumeChecker::create(monitor_scene_, left_.spec, right_.spec, cfg, monitor_checker_);
  if (!st)
  {
    model_.reset();
    return Status::Error("monitor checker: " + st.message);
  }
  model_ready_ = true;
  RCLCPP_INFO(get_logger(), "robot model '%s' loaded", model_->getName().c_str());
  return Status::Ok();
}

Status CoordinatorNode::refreshScene(planning_scene::PlanningScenePtr & scene)
{
  scene = std::make_shared<planning_scene::PlanningScene>(model_);
  if (!pb("require_planning_scene")) return Status::Ok();
  auto req = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  req->components.components = moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES |
                               moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY |
                               moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
  std::string err;
  auto resp = callService<moveit_msgs::srv::GetPlanningScene>(scene_client_, req, 10.0, err);
  if (!resp) return Status::Error("planning scene: " + err);
  moveit_msgs::msg::PlanningScene msg = (*resp)->scene;
  msg.is_diff = false;
  msg.robot_state = moveit_msgs::msg::RobotState();       // keep the default robot state
  msg.robot_state.is_diff = true;
  scene->setPlanningSceneMsg(msg);
  return Status::Ok();
}

Status CoordinatorNode::makeValidator(const planning_scene::PlanningScenePtr & scene, std::unique_ptr<PhaseValidator> & out, double extra_padding_m)
{
  PhaseValidatorConfig cfg;
  cfg.swept.padding_m = pd("swept_padding_m") + extra_padding_m;
  cfg.swept.exact_distance_tolerance_m = pd("exact_distance_tolerance_m");
  cfg.swept.stop_at_first_violation = true;   // a verdict needs one violation, not the exact minimum
  cfg.tip_tolerance_m = pd("tip_on_line_tolerance_m");
  cfg.world_penetration_tolerance_m = pd("world_penetration_tolerance_m");
  cfg.table_min_clearance_m = pd("table_min_clearance_m") + extra_padding_m;
  cfg.world_object_clearance_m = extra_padding_m;
  cfg.table_link_name = ps("table_link_name");
  // (a copy: as_string_array() refers into the temporary Parameter)
  const std::vector<std::string> table_exempt_suffixes = get_parameter("table_exempt_link_suffixes").as_string_array();
  for (const auto & suffix : table_exempt_suffixes)
    for (auto * a : {&left_, &right_})
    {
      const std::string name = a->prefix + suffix;
      if (model_->hasLinkModel(name))
        (a == &left_ ? cfg.table_exempt_links_left : cfg.table_exempt_links_right).push_back(name);
    }
  cfg.tip_link_suffix = ps("tip_link_suffix");
  cfg.sample.dt_s = pd("sample_dt_s");
  cfg.sample.max_joint_step_rad = pd("max_joint_step_rad");
  cfg.allow_serial_fallback = pb("allow_serial_fallback");
  cfg.contact_object_ids = get_parameter("contact_object_ids").as_string_array();
  const std::vector<std::string> suffixes = get_parameter("contact_link_suffixes").as_string_array();
  for (const auto & suffix : suffixes)
    for (auto * a : {&left_, &right_})
    {
      const std::string name = a->prefix + suffix;
      if (model_->hasLinkModel(name))
        (a == &left_ ? cfg.contact_links_left : cfg.contact_links_right).push_back(name);
    }
  return PhaseValidator::create(scene, cfg, out);
}

Status CoordinatorNode::addCartonFromArmScenes(planning_scene::PlanningScenePtr & scene, double & center_diff_m, double & rot_diff_rad)
{
  center_diff_m = rot_diff_rad = 0.0;
  const std::string id = ps("carton_object_id");
  struct Seen
  {
    bool found = false;
    Eigen::Isometry3d world = Eigen::Isometry3d::Identity();
    Eigen::Vector3d size = Eigen::Vector3d::Zero();
  } seen[2];
  int k = 0;
  for (ArmIo * io : {&left_, &right_})
  {
    auto req = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    req->components.components = moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES |
                                 moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY;
    std::string err;
    auto resp = callService<moveit_msgs::srv::GetPlanningScene>(io->scene_client, req, 10.0, err);
    if (!resp) return Status::Error(io->side + " scene: " + err);
    for (const auto & o : (*resp)->scene.world.collision_objects)
    {
      if (o.id != id) continue;
      if (o.primitives.size() != 1 || o.primitives[0].type != shape_msgs::msg::SolidPrimitive::BOX ||
          o.primitives[0].dimensions.size() != 3 || o.primitive_poses.size() != 1)
        return Status::Error(io->side + ": " + id + " is not a single box");
      if (o.header.frame_id != "world" && o.header.frame_id != ps("expected_planning_frame"))
        return Status::Error(io->side + ": " + id + " is in frame '" + o.header.frame_id + "', expected the arm's base frame");
      auto toIso = [](const geometry_msgs::msg::Pose & p)
      {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
        T.linear() = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z).normalized().toRotationMatrix();
        return T;
      };
      // The per-arm scene's `world` is that arm's base frame: bring the box into the dual world.
      seen[k].world = urdfBaseToWorld(*io) * toIso(o.pose) * toIso(o.primitive_poses[0]);
      seen[k].size = Eigen::Vector3d(o.primitives[0].dimensions[0], o.primitives[0].dimensions[1], o.primitives[0].dimensions[2]);
      seen[k].found = true;
    }
    ++k;
  }
  if (!seen[0].found || !seen[1].found)
  {
    if (pb("require_carton"))
      return Status::Error(std::string("carton '") + id + "' missing in the " + (seen[0].found ? "right" : "left") + " arm's scene");
    return Status::Ok();
  }
  Status s = checkBoxConsistency(
    seen[0].world, seen[0].size, seen[1].world, seen[1].size, pd("carton_consistency_tol_m"),
    pd("carton_consistency_tol_rad"), &center_diff_m, &rot_diff_rad);
  if (!s) return s;
  // The left arm's scene IS the dual world; the consistency check above ties the right arm's to it.
  scene->getWorldNonConst()->removeObject(id);
  scene->getWorldNonConst()->addToObject(
    id, std::make_shared<shapes::Box>(seen[0].size.x(), seen[0].size.y(), seen[0].size.z()), seen[0].world);
  return Status::Ok();
}

Eigen::Isometry3d CoordinatorNode::urdfBaseToWorld(const ArmIo & io) const
{
  moveit::core::RobotState st(model_);
  st.setToDefaultValues();
  st.updateLinkTransforms();
  return st.getGlobalLinkTransform(io.prefix + "base_link");
}

// ---------------------------------------------------------------------------------------------
// service helper
// ---------------------------------------------------------------------------------------------

template<typename SrvT>
std::optional<typename SrvT::Response::SharedPtr> CoordinatorNode::callService(
  typename rclcpp::Client<SrvT>::SharedPtr client, typename SrvT::Request::SharedPtr req, double timeout_s,
  std::string & error)
{
  if (!client->wait_for_service(20s))
  {
    error = "service " + std::string(client->get_service_name()) + " is not available";
    return std::nullopt;
  }
  auto pending = client->async_send_request(req);
  if (pending.future.wait_for(std::chrono::duration<double>(timeout_s)) != std::future_status::ready)
  {
    client->remove_pending_request(pending.request_id);
    error = "timeout calling " + std::string(client->get_service_name());
    return std::nullopt;
  }
  return pending.future.get();
}

// used from sequence_mode.cpp too
template std::optional<dual_cut_msgs::srv::PlanFromState::Response::SharedPtr> CoordinatorNode::callService<dual_cut_msgs::srv::PlanFromState>(
  rclcpp::Client<dual_cut_msgs::srv::PlanFromState>::SharedPtr, dual_cut_msgs::srv::PlanFromState::Request::SharedPtr, double, std::string &);
template std::optional<dual_cut_msgs::srv::GetPlannedSolution::Response::SharedPtr> CoordinatorNode::callService<dual_cut_msgs::srv::GetPlannedSolution>(
  rclcpp::Client<dual_cut_msgs::srv::GetPlannedSolution>::SharedPtr, dual_cut_msgs::srv::GetPlannedSolution::Request::SharedPtr, double, std::string &);

Status CoordinatorNode::setExecutorParams(ArmIo & io, const std::vector<std::string> & kv)
{
  auto req = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();
  for (const auto & item : kv)
  {
    const auto eq = item.find('=');
    if (eq == std::string::npos || eq == 0) return Status::Error("bad executor parameter '" + item + "' (need name=value)");
    rclcpp::Parameter p(item.substr(0, eq), inferValue(item.substr(eq + 1)));
    req->parameters.push_back(p.to_parameter_msg());
  }
  if (req->parameters.empty()) return Status::Ok();
  std::string err;
  auto resp = callService<rcl_interfaces::srv::SetParameters>(io.set_params, req, 15.0, err);
  if (!resp) return Status::Error(io.side + ": " + err);
  for (size_t i = 0; i < (*resp)->results.size(); ++i)
    if (!(*resp)->results[i].successful)
      return Status::Error(io.side + ": executor rejected parameter " + kv[i] + ": " + (*resp)->results[i].reason);
  return Status::Ok();
}

// ---------------------------------------------------------------------------------------------
// planning
// ---------------------------------------------------------------------------------------------

Status CoordinatorNode::gatePlan(const ArmIo & io, const ArmPlan & plan, double & snapshot_age_s)
{
  if (plan.planning_frame != ps("expected_planning_frame"))
    return Status::Error(io.side + ": planning_frame '" + plan.planning_frame + "' != '" + ps("expected_planning_frame") + "'");
  if (plan.tool_link != ps("expected_tool_link"))
    return Status::Error(io.side + ": tool_link '" + plan.tool_link + "' != '" + ps("expected_tool_link") + "'");
  if (plan.snapshot.stale_override) return Status::Error(io.side + ": plan was made with a stale-snapshot override");
  // Frame-consistency gate: the exported transform must be the dual URDF's world -> arm base.
  const Eigen::Isometry3d exported = poseFromXyzRpy(plan.base_xyz, plan.base_rpy);
  Status s = checkFrameConsistency(exported, urdfBaseToWorld(io), pd("frame_tolerance"));
  if (!s) return Status::Error(io.side + ": frame-consistency gate: " + s.message);
  // Snapshot freshness (executors stamp with the wall clock).
  const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  snapshot_age_s = 1e-9 * (static_cast<double>(now_ns) - static_cast<double>(plan.snapshot.acquisition_stamp_ns));
  if (plan.snapshot.acquisition_stamp_ns == 0 || snapshot_age_s < -1.0)
    return Status::Error(io.side + ": snapshot has no valid acquisition stamp");
  if (snapshot_age_s > pd("external_snapshot_max_age_s"))
    return Status::Error(
      io.side + ": snapshot is stale (" + std::to_string(snapshot_age_s) + " s old, limit " +
      std::to_string(pd("external_snapshot_max_age_s")) + " s)");
  return Status::Ok();
}

Status CoordinatorNode::topSagFor(ArmIo & io, const std::string & source_text, const Eigen::Isometry3d & snapshot_to_world, CutSag & out)
{
  out = CutSag{};
  auto req = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
  req->names = {"middle_cut_sag_m"};
  std::string err;
  auto resp = callService<rcl_interfaces::srv::GetParameters>(io.get_params, req, 10.0, err);
  if (!resp) return Status::Error(io.side + " executor middle_cut_sag_m: " + err);
  double sag = 0.0;
  // an executor without the parameter answers with an empty list or a NOT_SET value: no sag
  if ((*resp)->values.size() > 1) return Status::Error(io.side + " executor middle_cut_sag_m: bad reply");
  rcl_interfaces::msg::ParameterValue v;
  if ((*resp)->values.size() == 1) v = (*resp)->values[0];
  if (v.type == rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE) sag = v.double_value;
  else if (v.type == rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER) sag = static_cast<double>(v.integer_value);
  else if (v.type != rcl_interfaces::msg::ParameterType::PARAMETER_NOT_SET)
    return Status::Error(io.side + " executor middle_cut_sag_m has an unexpected type");
  if (sag == 0.0) return Status::Ok();   // missing or 0: today's straight-line check
  SnapshotSagGeometry g;
  Status s = sagGeometryFromSnapshotText(source_text, g);
  if (!s) return Status::Error(io.side + " sag geometry: " + s.message);
  s = makeCutSag(g, snapshot_to_world, sag, out);
  if (!s) return Status::Error(io.side + " sag geometry: " + s.message);
  return Status::Ok();
}

TopSplitParams CoordinatorNode::topSplitParams() const
{
  TopSplitParams p;
  Status s = topSplitModeFromString(get_parameter("top_split_mode").as_string(), p.mode);
  if (!s) p.mode = TopSplitMode::LeftMax;   // validateTopSplitParams cannot see this: planImpl re-checks the string
  p.overlap_m = pd("top_split_overlap_m");
  p.right_min_tail = pd("top_split_right_min_tail");
  p.left_min_fraction = pd("top_split_left_min_fraction");
  p.min_overlap_m = pd("top_split_min_overlap_m");
  p.end_tolerance_m = pd("top_split_end_tolerance_m");
  p.skip_right_top_fraction = pd("top_split_skip_right_top_fraction");
  p.right_end_extension_m = pd("top_split_right_end_extension_m");
  p.snapshot_file = ps("top_split_snapshot_file");
  return p;
}

Status CoordinatorNode::leftSnapshotPath(std::string & path)
{
  auto req = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
  req->names = {"external_snapshot_file"};
  std::string err;
  auto resp = callService<rcl_interfaces::srv::GetParameters>(left_.get_params, req, 10.0, err);
  if (!resp) return Status::Error("left executor external_snapshot_file: " + err);
  if ((*resp)->values.size() != 1 || (*resp)->values[0].type != rcl_interfaces::msg::ParameterType::PARAMETER_STRING ||
      (*resp)->values[0].string_value.empty())
    return Status::Error("the left executor has no external_snapshot_file set (the top-split source snapshot)");
  path = (*resp)->values[0].string_value;
  return Status::Ok();
}

CoordinatorNode::ArmPlanPair CoordinatorNode::planArm(
  ArmIo & io, const sensor_msgs::msg::JointState & start, const std::vector<std::string> & params_a,
  const std::function<Status(std::vector<std::string> &)> & params_b,
  const std::function<void(const Status &, const ArmPlan &)> & after_a,
  const std::function<Status(std::vector<std::string> &)> & params_a_fn,
  const std::function<void(const Status &, const ArmPlan &)> & after_b)
{
  ArmPlanPair out;
  const double timeout = pd("executor_call_timeout_s");
  const auto t0 = std::chrono::steady_clock::now();
  auto planPhase = [&](const char * phase, const std::vector<std::string> & params,
                       const sensor_msgs::msg::JointState & from, ArmPlan & plan) -> Status
  {
    if (stop_latched_) return Status::Error("stop latched during planning");
    Status s = setExecutorParams(io, params);
    if (!s) return s;
    auto req = std::make_shared<dual_cut_msgs::srv::PlanFromState::Request>();
    req->start_state = from;
    std::string err;
    auto r1 = callService<dual_cut_msgs::srv::PlanFromState>(io.plan_from_state, req, timeout, err);
    if (!r1) return Status::Error(io.side + " phase " + phase + ": " + err);
    if (!(*r1)->success)
      return Status::Error(io.side + " phase " + phase + ": plan_h_cut_from_state failed: " + (*r1)->message);
    auto r2 = callService<dual_cut_msgs::srv::GetPlannedSolution>(
      io.get_plan, std::make_shared<dual_cut_msgs::srv::GetPlannedSolution::Request>(), 30.0, err);
    if (!r2) return Status::Error(io.side + " phase " + phase + ": " + err);
    if (!(*r2)->success)
      return Status::Error(io.side + " phase " + phase + ": get_planned_solution failed: " + (*r2)->message);
    if ((*r2)->plan_seq != (*r1)->plan_seq)
      return Status::Error(
        io.side + " phase " + phase + ": exported plan_seq " + std::to_string((*r2)->plan_seq) +
        " != planned " + std::to_string((*r1)->plan_seq) + " (the executor plan changed underneath us)");
    s = fromResponse(io.side, **r2, plan);
    if (!s) return Status::Error(io.side + " phase " + phase + ": " + s.message);
    return Status::Ok();
  };
  if (params_a_fn)
  {
    std::vector<std::string> pa_list;
    out.status = params_a_fn(pa_list);
    if (out.status) out.status = planPhase("A", pa_list, start, out.a);
  }
  else
  {
    out.status = planPhase("A", params_a, start, out.a);
  }
  out.a_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (after_a) after_a(out.status, out.a);
  if (!out.status)
  {
    if (after_b) after_b(out.status, out.b);
    return out;
  }
  sensor_msgs::msg::JointState from_a;
  from_a.name = out.a.joint_names;
  from_a.position = out.a.end_q;
  std::vector<std::string> pb_list;
  out.status = params_b(pb_list);
  if (out.status) out.status = planPhase("B", pb_list, from_a, out.b);
  out.b_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - out.a_seconds;
  if (after_b) after_b(out.status, out.b);
  return out;
}

bool CoordinatorNode::executorBool(ArmIo & io, const std::string & name)
{
  auto req = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
  req->names = {name};
  std::string err;
  auto resp = callService<rcl_interfaces::srv::GetParameters>(io.get_params, req, 10.0, err);
  if (!resp || (*resp)->values.size() != 1) return false;
  const auto & v = (*resp)->values[0];
  return v.type == rcl_interfaces::msg::ParameterType::PARAMETER_BOOL && v.bool_value;
}

// Left side cut reverse patch (executor parameter left_cut_patch_enabled): the executor already drops a patch it cannot
// plan or that fails one of its own gates. The patch must never make the whole dual plan fail, so if every attempt of the
// chain was refused and the refusal names a patch stage ('left side cut patch', e.g. a world-check collision of the patch
// transit that only this coordinator sees), the chain is run once more with the patch switched off on the arms that had
// it on (restored afterwards). Without a patch-related refusal this is exactly the old handlePlan.
void CoordinatorNode::handlePlan(const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> res)
{
  handlePlanChain(req, res);
  if (res->success || res->message.find("left side cut patch") == std::string::npos) return;
  std::vector<ArmIo *> flipped;
  for (ArmIo * io : {&left_, &right_})
    if (executorBool(*io, "left_cut_patch_enabled") && setExecutorParams(*io, {"left_cut_patch_enabled=false"})) flipped.push_back(io);
  if (flipped.empty()) return;
  const std::string first_refusal = res->message;
  RCLCPP_WARN(
    get_logger(), "plan refused because of the left reverse patch; planning again with the patch switched off: %s",
    first_refusal.size() > 400 ? (first_refusal.substr(0, 400) + "...").c_str() : first_refusal.c_str());
  handlePlanChain(req, res);
  for (ArmIo * io : flipped)
  {
    const Status back = setExecutorParams(*io, {"left_cut_patch_enabled=true"});
    if (!back) RCLCPP_ERROR(get_logger(), "could not restore left_cut_patch_enabled=true on the %s executor: %s", io->side.c_str(), back.message.c_str());
  }
  const std::string head = first_refusal.size() > 300 ? first_refusal.substr(0, 300) + "..." : first_refusal;
  res->message = res->success ? "LEFT PATCH DROPPED by the coordinator (the plan with the reverse patch was refused: " + head + "); " + res->message
                              : res->message + " [also refused without the patch; first refusal: " + head + "]";
}

void CoordinatorNode::handlePlanChain(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res)
{
  std::unique_lock<std::mutex> plan_lock(plan_mutex_, std::try_to_lock);
  if (!plan_lock.owns_lock())
  {
    res->success = false;
    res->message = "busy: another plan / rearm call holds the plan lock";
    return;
  }
  if (executing_)
  {
    res->success = false;
    res->message = "refused: an execution is running";
    return;
  }
  if (stop_latched_)
  {
    res->success = false;
    res->message = "refused: stop is latched (" + stopReason() + "); call ~/rearm";
    return;
  }
  planning_ = true;
  state_ = Status_::STATE_PLANNING;
  last_plan_report_.clear();   // amendPlanReport() only ever amends a report of THIS call
  const auto t_call0 = std::chrono::steady_clock::now();
  auto elapsedS = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_call0).count(); };
  const int max_retries = static_cast<int>(get_parameter("plan_retries").as_int());
  const std::vector<std::string> signatures = get_parameter("plan_retry_signatures").as_string_array();
  const double budget_s = pd("plan_fallback_budget_s");

  // the order fallback chain, starting at the configured mode (invalid configuration: nothing is planned, fail closed)
  std::vector<PlanVariant> chain;
  const std::string configured = configuredVariantName(ps("schedule_mode"), ps("phase_order"));
  Status s = buildPlanChain(get_parameter("plan_fallback_chain").as_string_array(), configured, chain);
  std::string summary;
  int total_retries = 0;
  std::vector<PlanAttemptRecord> failed;   // the refused attempts, one record per chain entry (latest refusal), in chain order
  size_t used = 0;
  bool used_second_pass = false;
  std::string used_first_refusal;          // second pass only: the refusal that made the entry be retried
  // the same snapshot must still be fresh enough (the executors check it too)
  auto snapshotFresh = [&]() {
    std::string path, text;
    uint64_t stamp_ns = 0;
    if (!(leftSnapshotPath(path) && readTextFile(path, text) && snapshotStampFromText(text, stamp_ns))) return false;
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return 1e-9 * (static_cast<double>(now_ns) - static_cast<double>(stamp_ns)) <= pd("external_snapshot_max_age_s");
  };
  if (s)
  {
    // Order of the attempts (plan_policy.hpp, PlanChainDriver): breadth first - every chain entry once, only a refusal on
    // plan_inplace_retry_labels is retried at once - then, if nothing passed, the entries are retried in chain order.
    PlanRetryPolicy policy;
    policy.max_retries = max_retries;
    policy.signatures = signatures;
    policy.inplace_labels = get_parameter("plan_inplace_retry_labels").as_string_array();
    policy.breadth_first = pb("plan_retry_breadth_first");
    PlanChainDriver driver(chain.size(), policy);
    std::vector<double> entry_wall_s(chain.size(), 0.0);
    std::vector<int> record_of(chain.size(), -1);   // index into `failed`
    auto previousFor = [&](size_t k) {
      std::vector<PlanAttemptRecord> prev;
      for (size_t i = 0; i < failed.size(); ++i)
        if (record_of[k] < 0 || static_cast<int>(i) != record_of[k]) prev.push_back(failed[i]);
      return prev;
    };
    while (true)
    {
      const PlanChainDriver::Next nx = driver.next();
      if (nx.done) break;
      const size_t k = nx.entry;
      const PlanVariant & variant = chain[k];
      if (!nx.retry && k > 0)
      {
        if (elapsedS() >= budget_s)
        {
          RCLCPP_WARN(get_logger(), "plan fallback: budget %.0f s used up (%.1f s), '%s' and the rest of the chain are not tried", budget_s, elapsedS(), variant.name.c_str());
          driver.stop();
          break;
        }
        if (!snapshotFresh())
        {
          RCLCPP_WARN(get_logger(), "plan fallback: skipped, the snapshot is no longer fresh enough (or unreadable)");
          driver.stop();
          break;
        }
      }
      if (nx.retry)
      {
        // random-failure retries may use at most half of the budget, so that the (cheaper) alternatives of the chain still get their turn
        if (elapsedS() >= 0.5 * budget_s)
        {
          RCLCPP_WARN(get_logger(), "plan retry skipped: half of plan_fallback_budget_s (%.0f s) is used up (%.1f s)", budget_s, elapsedS());
          driver.declineRetry();
          continue;
        }
        if (!snapshotFresh())
        {
          RCLCPP_WARN(get_logger(), "plan retry skipped: the snapshot is no longer fresh enough (or unreadable)");
          driver.declineRetry();
          continue;
        }
        const int failed_index = driver.retriesUsed(k) - 1;   // the execution that was just refused
        RCLCPP_WARN(
          get_logger(), "plan attempt %d of '%s' refused by a random executor failure, retrying%s (%d left): %s", failed_index, variant.name.c_str(),
          nx.second_pass ? " in the second pass" : "", max_retries - failed_index, driver.reasons(k).back().c_str());
      }
      attempt_index_ = k;
      const double t_attempt0 = elapsedS();
      {
        // every attempt starts clean (rearm-equivalent reset of the retained plan state), same snapshot
        std::lock_guard<std::mutex> lk(data_mutex_);
        retained_ = Retained{};
        phase_validated_ = false;
      }
      {
        std::ostringstream o;
        o << "{\"attempt\":" << driver.retriesUsed(k) << ",\"max_retries\":" << max_retries << ",\"previous_refusals\":[";
        const auto & earlier = driver.reasons(k);
        for (size_t i = 0; i < earlier.size(); ++i) o << (i ? "," : "") << "\"" << jsonEscape(earlier[i]) << "\"";
        o << "]}";
        retry_json_ = o.str();
      }
      fallback_json_ = planFallbackJson(chain, k, previousFor(k), budget_s, elapsedS(), nx.second_pass ? 2 : 1);
      summary.clear();
      s = planImpl(variant, summary);
      entry_wall_s[k] += elapsedS() - t_attempt0;
      if (s)
      {
        driver.succeeded();
        break;
      }
      const RefusalClass cls = classifyRefusal(s.message);
      driver.failed(s.message, cls);
      PlanAttemptRecord rec;
      rec.variant = variant.name;
      rec.reason = s.message;
      rec.cls = cls;
      rec.wall_s = entry_wall_s[k];
      rec.retries = driver.retriesUsed(k);
      if (record_of[k] < 0)
      {
        record_of[k] = static_cast<int>(failed.size());
        failed.push_back(rec);
      }
      else
        failed[static_cast<size_t>(record_of[k])] = rec;
      if (stop_latched_)
      {
        driver.stop();
        break;
      }
      if (driver.inplacePending() || nx.second_pass) continue;   // an in-place retry follows / second pass: the retry WARN says it
      if (cls != RefusalClass::OrderFixable)
      {
        RCLCPP_WARN(
          get_logger(), "plan attempt %zu/%zu '%s' refused (%s, no order change can fix it): %s", k + 1, chain.size(), variant.name.c_str(),
          toString(cls), s.message.c_str());
        continue;   // the driver ends the chain
      }
      if (k + 1 < chain.size())
        RCLCPP_WARN(
          get_logger(), "plan attempt %zu/%zu '%s' refused (%s) after %.1f s: %s -> trying '%s'", k + 1, chain.size(), variant.name.c_str(),
          toString(cls), rec.wall_s, s.message.c_str(), chain[k + 1].name.c_str());
    }
    total_retries = driver.totalRetries();
    if (driver.ok())
    {
      used = driver.used();
      used_second_pass = driver.usedInSecondPass();
      if (used_second_pass && !driver.reasons(used).empty()) used_first_refusal = driver.reasons(used).front();
      if (record_of[used] >= 0) failed.erase(failed.begin() + record_of[used]);   // it passed after all: not a failed attempt any more
    }
  }
  retry_json_.clear();
  fallback_json_.clear();
  planning_ = false;
  if (!stop_latched_) state_ = s ? Status_::STATE_PLANNED : Status_::STATE_IDLE;
  const double total_s = elapsedS();
  res->success = s.ok;
  if (s.ok)
  {
    res->message = summary;
    if (total_retries > 0) res->message += " [plan ok after " + std::to_string(total_retries) + " retries]";
    if (used > 0 || used_second_pass)
    {
      std::string how = used_second_pass ? " on a retry in the second pass (breadth first: every chain entry was tried once first; its own first refusal: " +
                                             (used_first_refusal.size() > 200 ? used_first_refusal.substr(0, 200) + "..." : used_first_refusal) + ")"
                                         : std::string();
      res->message = "PLAN ORDER FALLBACK: attempt " + std::to_string(used + 1) + "/" + std::to_string(chain.size()) + " '" + chain[used].name +
                     "' used" + how + " (" + describeAttempts(failed) + "); " + res->message;
      RCLCPP_WARN(get_logger(), "plan order fallback: '%s' used after %zu refused attempt(s): %s", chain[used].name.c_str(), failed.size(), describeAttempts(failed, 400).c_str());
    }
    std::ostringstream pt;
    pt << std::fixed << std::setprecision(1) << " [planning " << total_s << " s]";
    res->message += pt.str();
  }
  else if (failed.empty())
  {
    res->message = s.message;
  }
  else if (failed.size() == 1)
  {
    res->message = total_retries > 0 ? "refused after " + std::to_string(total_retries) + " retries: " + s.message : s.message;
    if (chain.size() > 1)
      res->message += failed[0].cls == RefusalClass::NotOrderFixable ? " [no order fallback tried: a different order cannot fix this refusal]"
                                                                     : " [order fallback chain ended after this attempt (budget, stale snapshot or stop)]";
  }
  else
  {
    res->message = "refused, " + std::to_string(failed.size()) + " attempts of the order fallback chain: " + describeAttempts(failed);
  }
  // totals into the last published report
  {
    std::ostringstream o;
    o << std::setprecision(6) << "\"plan_total_s\":" << total_s << ",\"plan_variant_used\":"
      << (s.ok && used < chain.size() ? "\"" + jsonEscape(chain[used].name) + "\"" : std::string("null")) << ",\"plan_attempts_failed\":"
      << failed.size() << ",\"plan_attempts_summary\":\"" << jsonEscape(describeAttempts(failed, 400)) << "\"";
    amendPlanReport(o.str());
  }
  if (detection_.state == DetectionPermission::State::Blocked)
  {
    std::string why = detection_.malformed ? detection_.note : std::string();
    for (const auto & r : detection_.reasons) why += (why.empty() ? "" : "; ") + r;
    RCLCPP_WARN(
      get_logger(), "DETECTION BLOCKED by the detector%s: %s. The plan is made, but ~/execute is refused unless allow_blocked_detection is true%s.",
      detection_.malformed ? " (cut_permission unusable)" : "", why.c_str(),
      detection_.has_override && pb("honor_detection_override") ? " or the detector's own override (set and honored here)" : "");
  }
  else if (detection_.state == DetectionPermission::State::Unknown && s)
    RCLCPP_WARN(get_logger(), "detection permission unknown: the snapshot has no cut_permission field (old snapshot), treated as allowed");
  if (s) RCLCPP_INFO(get_logger(), "plan ok: %s", res->message.c_str());
  else RCLCPP_WARN(get_logger(), "plan refused: %s", res->message.c_str());
  publishStatus();
}

void CoordinatorNode::amendPlanReport(const std::string & members)
{
  if (last_plan_report_.size() < 2 || last_plan_report_.back() != '}') return;
  last_plan_report_.pop_back();
  last_plan_report_ += "," + members + "}";
  std_msgs::msg::String m;
  m.data = last_plan_report_;
  report_pub_->publish(m);
}

Status CoordinatorNode::verifyExecutorPlans(uint64_t left_seq, uint64_t right_seq)
{
  for (auto [io, want] : {std::pair<ArmIo *, uint64_t>{&left_, left_seq}, std::pair<ArmIo *, uint64_t>{&right_, right_seq}})
  {
    std::string err;
    auto r = callService<dual_cut_msgs::srv::GetPlannedSolution>(
      io->get_plan, std::make_shared<dual_cut_msgs::srv::GetPlannedSolution::Request>(), 30.0, err);
    if (!r) return Status::Error(io->side + " executor plan mismatch: cannot read the executor's retained plan: " + err);
    if (!(*r)->success)
      return Status::Error(io->side + " retained executor plan is missing (expected plan_seq " + std::to_string(want) + "): " + (*r)->message);
    if ((*r)->plan_seq != want)
      return Status::Error(
        io->side + " executor plan mismatch: the executor holds plan_seq " + std::to_string((*r)->plan_seq) + ", the chosen attempt's last job is " +
        std::to_string(want));
  }
  return Status::Ok();
}

Status CoordinatorNode::planImpl(const PlanVariant & variant, std::string & summary)
{
  current_variant_ = variant.name;
  detection_ = DetectionPermission{};
  detection_json_.clear();
  attempt_detail_json_.clear();
  Status s = ensureModel();
  if (!s) return s;

  // --- 1. preconditions: fresh joint states, both arms at rest, scene ---
  std::vector<double> ql, qr;
  double age_l = 0, age_r = 0;
  if (!armQ(left_, ql, &age_l) || !armQ(right_, qr, &age_r))
    return Status::Error("precondition: no /joint_states for both arms");
  if (std::max(age_l, age_r) > 0.5) return Status::Error("precondition: /joint_states is stale");
  double rest;
  {
    std::lock_guard<std::mutex> lk(js_mutex_);
    rest = rest_tracker_.restDuration(std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
  }
  if (rest < pd("barrier_rest_duration_s"))
    return Status::Error("precondition: the arms are not at rest (" + std::to_string(rest) + " s)");

  auto jointState = [&](const ArmIo & io, const std::vector<double> & q)
  {
    sensor_msgs::msg::JointState js;
    for (size_t i = 0; i < io.spec.joint_names.size(); ++i)
    {
      std::string plain;
      stripJointPrefix(io.prefix, io.spec.joint_names[i], plain);
      js.name.push_back(plain);
      js.position.push_back(q[i]);
    }
    return js;
  };

  // --- 1b. top-seam split: the source snapshot is the LEFT executor's external_snapshot_file ---
  TopSplitParams tsp = topSplitParams();
  s = topSplitModeFromString(ps("top_split_mode"), tsp.mode);
  if (!s) return s;
  s = validateTopSplitParams(tsp);
  if (!s) return s;
  std::string source_path, source_text;
  s = leftSnapshotPath(source_path);
  if (!s) return s;
  s = readTextFile(source_path, source_text);
  if (!s) return Status::Error("top split source snapshot: " + s.message);
  Eigen::Vector3d seam_a, seam_b;   // seam_start / seam_end of the source, in the snapshot frame
  s = seamFromSnapshotText(source_text, seam_a, seam_b);
  if (!s) return Status::Error("top split source snapshot " + source_path + ": " + s.message);
  const double seam_length_m = (seam_b - seam_a).norm();
  if (!(seam_length_m > 0.05)) return Status::Error("top split: the source snapshot's top seam is shorter than 50 mm");
  const std::string source_sha = sha256Hex(source_text);
  // the detector's cut_permission of the source snapshot: the plan runs, the report and ~/execute carry the verdict
  detection_ = parseDetectionPermission(source_text);
  detection_json_ = detectionPermissionJson(detection_, pb("honor_detection_override"));
  const bool left_max = tsp.mode == TopSplitMode::LeftMax;
  // sequence variants never reach the phases code below; the phase order of a phases variant comes from the chain entry
  const std::string phase_order = variant.sequence ? ps("phase_order") : variant.phase_order;
  if (phase_order != "top_first" && phase_order != "side_first")
    return Status::Error("phase_order must be 'top_first' or 'side_first', got '" + phase_order + "'");
  const bool side_first = !variant.sequence && phase_order == "side_first";
  // phase that holds the top seam of each arm (top_first: A for the left, B for the right; side_first: B and A)
  const char * const left_top_phase = side_first ? "B" : "A";
  const char * const right_top_phase = side_first ? "A" : "B";

  // Executor parameter lists. Every list ends with middle_cut_reverse_travel=false unless it is the right arm's
  // phase B of left_max (a previous left_max run must never leak "true" into another pattern).
  const std::string kReverseOff = "middle_cut_reverse_travel=false";
  auto fmt = [](double v)
  {
    std::ostringstream o;
    o << std::setprecision(12) << v;
    std::string text = o.str();
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";   // the executor parameter is a double
    return text;
  };
  auto configured = [&](const char * key) { return get_parameter(key).as_string_array(); };
  auto with = [](std::vector<std::string> base, std::initializer_list<std::string> extra)
  {
    base.insert(base.end(), extra.begin(), extra.end());
    return base;
  };
  const std::string src_param = "external_snapshot_file=" + source_path;

  // shared state of the left_max hand-off (left phase A export -> right phase B derived snapshot)
  struct LeftReach
  {
    bool ok = false;
    double fraction = 0.0;
    std::string message;
  };
  std::promise<LeftReach> left_a_promise;
  std::shared_future<LeftReach> left_a_future = left_a_promise.get_future().share();
  struct SplitInfo
  {
    bool derived = false, skipped = false;
    double left_reach = 0.0, start_fraction = 0.0, right_end_extension_m = 0.0;
  } split_info;
  const double wait_s = 2.0 * pd("executor_call_timeout_s") + 60.0;

  std::vector<std::string> left_a, right_a;
  std::function<Status(std::vector<std::string> &)> left_b, right_b;
  std::function<void(const Status &, const ArmPlan &)> left_after_a;
  if (left_max)
  {
    left_a = with(configured("left_phase_a_params"),
                  {"cut_pattern=middle_only", "middle_seam_stop=end", kReverseOff,
                   "top_cut_min_cut_fraction=" + fmt(tsp.left_min_fraction)});
    left_b = [&](std::vector<std::string> & out) { out = with(configured("left_phase_b_params"), {kReverseOff}); return Status::Ok(); };
    left_after_a = [&](const Status & st, const ArmPlan & plan)
    {
      LeftReach r;
      r.ok = st.ok;
      r.fraction = plan.middle_cut_fraction;
      r.message = st.message;
      left_a_promise.set_value(r);
    };
    right_a = with(configured("right_phase_a_params"), {kReverseOff, src_param});
    right_b = [&](std::vector<std::string> & out) -> Status
    {
      if (left_a_future.wait_for(std::chrono::duration<double>(wait_s)) != std::future_status::ready)
        return Status::Error(std::string("right phase ") + right_top_phase + ": timeout waiting for the left arm's phase " + left_top_phase + " export");
      const LeftReach lr = left_a_future.get();
      if (!lr.ok) return Status::Error(std::string("right phase ") + right_top_phase + " not planned: the left arm's phase " + left_top_phase + " failed (" + lr.message + ")");
      double f = 0.0;
      Status st = topSplitStartFraction(lr.fraction, seam_length_m, tsp, f);
      if (!st) return Status::Error(std::string("right phase ") + right_top_phase + ": " + st.message);
      st = writeDerivedTopSnapshot(source_path, tsp.snapshot_file, f, lr.fraction, source_sha, tsp.right_end_extension_m);
      if (!st) return Status::Error(std::string("right phase ") + right_top_phase + ": derived snapshot: " + st.message);
      split_info.derived = true;
      split_info.left_reach = lr.fraction;
      split_info.start_fraction = f;
      split_info.right_end_extension_m = tsp.right_end_extension_m;
      out = with(configured("right_phase_b_params"),
                 {"cut_pattern=middle_only", "middle_seam_stop=end", "middle_cut_reverse_travel=true", "top_cut_min_cut_fraction=1.0",
                  "external_snapshot_file=" + tsp.snapshot_file});
      return Status::Ok();
    };
  }
  else
  {
    left_a = with(configured("left_phase_a_params"), {kReverseOff});
    left_b = [&](std::vector<std::string> & out) { out = with(configured("left_phase_b_params"), {kReverseOff}); return Status::Ok(); };
    right_a = with(configured("right_phase_a_params"), {kReverseOff, src_param});
    right_b = [&](std::vector<std::string> & out) { out = with(configured("right_phase_b_params"), {kReverseOff, src_param}); return Status::Ok(); };
  }

  // --- 1c. schedule mode: sequence (each arm back to back, see sequence_mode.cpp) or today's phases ---
  if (variant.sequence)
  {
    const std::string kHomeOff = "return_home_after_cut=false", kHomeOn = "return_home_after_cut=true";
    std::promise<LeftReach> seq_promise;
    std::shared_future<LeftReach> seq_future = seq_promise.get_future().share();
    SeqInputs in;
    in.left_start = jointState(left_, ql);
    in.right_start = jointState(right_, qr);
    in.ql = ql;
    in.qr = qr;
    in.tsp = tsp;
    in.source_path = source_path;
    in.source_sha = source_sha;
    in.source_text = source_text;
    in.src_param = src_param;
    in.seam_a = seam_a;
    in.seam_b = seam_b;
    in.seam_length_m = seam_length_m;
    in.margin_s = pd("sequence_skew_margin_s");
    in.step_s = pd("sequence_delay_step_s");
    in.max_delay_s = pd("sequence_max_delay_s");
    // the LEFT arm's job order: face then top (today), or top then face (variant sequence_left_top_first). The right
    // arm keeps its order (top remainder, then face).
    const std::vector<std::string> left_face_list = with(configured("left_phase_b_params"), {kReverseOff});
    const std::vector<std::string> left_top_list =
      left_max ? with(configured("left_phase_a_params"),
                      {"cut_pattern=middle_only", "middle_seam_stop=end", kReverseOff, "top_cut_min_cut_fraction=" + fmt(tsp.left_min_fraction)})
               : with(configured("left_phase_a_params"), {kReverseOff});
    const LeftJobOrder left_order = orderLeftJobs(variant.left_top_first, left_face_list, left_top_list, kHomeOff, kHomeOn);
    in.left_job1 = left_order.job1;
    in.left_job2 = left_order.job2;
    in.left_job1_label = left_order.job1_label;
    in.left_job2_label = left_order.job2_label;
    in.left_top_first = left_order.top_first;
    in.right_job2 = with(configured("right_phase_a_params"), {kReverseOff, src_param, kHomeOn});
    if (left_max)
    {
      in.left_after_top = [&](const Status & st, const ArmPlan & plan)
      {
        LeftReach r;
        r.ok = st.ok;
        r.fraction = plan.middle_cut_fraction;
        r.message = st.message;
        seq_promise.set_value(r);
      };
      in.right_job1 = [&](std::vector<std::string> & out, bool & skip) -> Status
      {
        skip = false;
        if (seq_future.wait_for(std::chrono::duration<double>(wait_s)) != std::future_status::ready)
          return Status::Error("right job 1: timeout waiting for the left arm's top job export");
        const LeftReach lr = seq_future.get();
        if (!lr.ok) return Status::Error("right job 1 not planned: the left arm's top job failed (" + lr.message + ")");
        if (skipRightTop(lr.fraction, tsp.skip_right_top_fraction))
        {
          // the left top covers the seam: no right top, no derived snapshot
          skip = true;
          split_info.skipped = true;
          split_info.left_reach = lr.fraction;
          out.clear();
          return Status::Ok();
        }
        double f = 0.0;
        Status st = topSplitStartFraction(lr.fraction, seam_length_m, tsp, f);
        if (!st) return Status::Error("right job 1: " + st.message);
        st = writeDerivedTopSnapshot(source_path, tsp.snapshot_file, f, lr.fraction, source_sha, tsp.right_end_extension_m);
        if (!st) return Status::Error("right job 1: derived snapshot: " + st.message);
        split_info.derived = true;
        split_info.left_reach = lr.fraction;
        split_info.start_fraction = f;
        split_info.right_end_extension_m = tsp.right_end_extension_m;
        out = with(configured("right_phase_b_params"),
                   {"cut_pattern=middle_only", "middle_seam_stop=end", "middle_cut_reverse_travel=true", "top_cut_min_cut_fraction=1.0",
                    "external_snapshot_file=" + tsp.snapshot_file, kHomeOff});
        return Status::Ok();
      };
    }
    else
    {
      in.right_job1 = [&](std::vector<std::string> & out, bool & skip) -> Status
      {
        skip = false;
        out = with(configured("right_phase_b_params"), {kReverseOff, src_param, kHomeOff});
        return Status::Ok();
      };
    }
    in.split_json = [&]() -> std::string
    {
      if (split_info.skipped)
      {
        std::ostringstream o;
        o << std::setprecision(9) << "\"right_top_skipped\":true,\"left_reach_fraction\":" << split_info.left_reach;
        return o.str();
      }
      if (!split_info.derived) return std::string();
      std::ostringstream o;
      o << std::setprecision(9) << "\"left_reach_fraction\":" << split_info.left_reach << ",\"start_fraction\":" << split_info.start_fraction
        << ",\"derived_file\":\"" << jsonEscape(tsp.snapshot_file) << "\"";
      if (split_info.right_end_extension_m > 0.0) o << ",\"right_end_extension_m\":" << split_info.right_end_extension_m;
      return o.str();
    };
    const SeqOutcome so = planSequence(in, summary);
    attempt_detail_json_ = so.fallback_json;
    return so.status;   // a refused sequence attempt (also "no safe schedule") is handled by the plan fallback chain
  }

  // From here on: schedule_mode phases. A previous (sequence) attempt may have left return_home_after_cut=false on an
  // executor, and the phases lists do not set it: put the default back first.
  if (attempt_index_ > 0)
    for (ArmIo * io : {&left_, &right_})
    {
      s = setExecutorParams(*io, {"return_home_after_cut=true"});
      if (!s) return Status::Error("order fallback: " + s.message);
    }

  // --- 1d. phase_order side_first: A = left face + right top, B = left top + right face. The lists above are the
  // top_first ones (A = left top + right face, B = left face + right top): swap which phase each comes from.
  // No return home between the phases: both phase A lists get return_home_after_cut=false, both B lists true (each
  // arm ends at home only after its last cut); the right top phase (now A) waits for the left top phase (now B).
  std::function<Status(std::vector<std::string> &)> right_a_fn;
  std::function<void(const Status &, const ArmPlan &)> left_after_b;
  if (side_first)
  {
    const char * const kHomeOff = "return_home_after_cut=false";
    const char * const kHomeOn = "return_home_after_cut=true";
    std::vector<std::string> left_face_list;
    s = left_b(left_face_list);
    if (!s) return s;
    const std::vector<std::string> left_top_list = left_a, right_face_list = right_a;
    const auto right_top_fn = right_b;
    left_a = with(left_face_list, {kHomeOff});
    left_b = [left_top_list, kHomeOn](std::vector<std::string> & out)
    {
      out = left_top_list;
      out.push_back(kHomeOn);
      return Status::Ok();
    };
    left_after_b = left_after_a;
    left_after_a = nullptr;
    right_a.clear();
    right_a_fn = [right_top_fn, kHomeOff](std::vector<std::string> & out) -> Status
    {
      Status st = right_top_fn(out);
      if (st) out.push_back(kHomeOff);
      return st;
    };
    right_b = [right_face_list, kHomeOn](std::vector<std::string> & out)
    {
      out = right_face_list;
      out.push_back(kHomeOn);
      return Status::Ok();
    };
  }

  // --- 2. plan both arms, phases A and B, in parallel (left_max: right B waits for left A's export) ---
  const auto t_plan0 = std::chrono::steady_clock::now();
  auto fl = std::async(std::launch::async, [&]() { return planArm(left_, jointState(left_, ql), left_a, left_b, left_after_a, nullptr, left_after_b); });
  auto fr = std::async(std::launch::async, [&]() { return planArm(right_, jointState(right_, qr), right_a, right_b, nullptr, right_a_fn); });
  ArmPlanPair pl = fl.get();
  ArmPlanPair pr = fr.get();
  const double plan_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_plan0).count();
  if (side_first && !(pl.status && pr.status))
  {
    // return_home_after_cut=false must not leak into a later plan (best effort). Only after a FAILED plan: on
    // success both phase B lists already set it back to true, and changing a planning parameter after the plan
    // clears the executor's plan and its detected_carton collision object (the coordinator needs it).
    for (ArmIo * io : {&left_, &right_})
    {
      const Status hs = setExecutorParams(*io, {"return_home_after_cut=true"});
      if (!hs) RCLCPP_WARN(get_logger(), "side_first: could not restore return_home_after_cut on %s: %s", io->side.c_str(), hs.message.c_str());
    }
  }
  if (left_max)
  {
    // The right executor goes back to the source snapshot whatever happened (best effort).
    const Status rs = setExecutorParams(right_, {src_param});
    if (!rs) RCLCPP_WARN(get_logger(), "top split: could not point the right executor back at the source snapshot: %s", rs.message.c_str());
  }
  std::ostringstream rep;   // JSON report, published on ~/plan_report whatever the outcome
  rep << std::setprecision(9);
  rep << "{\"executor_calls\":{\"wall_s\":" << plan_wall_s << ",\"left\":{\"A_s\":" << pl.a_seconds << ",\"B_s\":" << pl.b_seconds
      << "},\"right\":{\"A_s\":" << pr.a_seconds << ",\"B_s\":" << pr.b_seconds << "}}";
  rep << ",\"phase_order\":\"" << phase_order << "\"";
  rep << ",\"plan_retry\":" << (retry_json_.empty() ? std::string("null") : retry_json_);
  rep << ",\"plan_fallback\":" << (fallback_json_.empty() ? std::string("null") : fallback_json_);
  rep << ",\"detection\":" << (detection_json_.empty() ? std::string("null") : detection_json_);
  rep << ",\"top_split\":{\"mode\":\"" << toString(tsp.mode) << "\",\"source_file\":\"" << jsonEscape(source_path)
      << "\",\"source_sha256\":\"" << source_sha << "\",\"seam_length_m\":" << seam_length_m;
  if (split_info.derived)
    rep << ",\"left_reach_fraction\":" << split_info.left_reach << ",\"start_fraction\":" << split_info.start_fraction
        << ",\"derived_file\":\"" << jsonEscape(tsp.snapshot_file) << "\"";
  if (split_info.derived && split_info.right_end_extension_m > 0.0) rep << ",\"right_end_extension_m\":" << split_info.right_end_extension_m;
  rep << "}";
  auto finishReport = [&](bool ok, const std::string & msg)
  {
    rep << ",\"ok\":" << (ok ? "true" : "false") << ",\"message\":\"" << jsonEscape(msg) << "\"}";
    last_plan_report_ = rep.str();
    std_msgs::msg::String m;
    m.data = last_plan_report_;
    report_pub_->publish(m);
  };
  auto fail = [&](const Status & st)
  {
    finishReport(false, st.message);
    return st;
  };
  if (!pl.status) return fail(pl.status);
  if (!pr.status) return fail(pr.status);
  if (stop_latched_) return fail(Status::Error("stop latched during planning"));

  // --- gates on every exported plan ---
  double age = 0.0, age_max = 0.0;
  for (auto * item : {&pl.a, &pl.b, &pr.a, &pr.b})
  {
    s = gatePlan(item->side == "left" ? left_ : right_, *item, age);
    if (!s) return fail(s);
    age_max = std::max(age_max, age);
  }
  s = checkPermissionMatchesPlan(detection_, pl.a.snapshot.snapshot_id, pl.a.snapshot.acquisition_stamp_ns);
  if (!s) return fail(s);
  auto segs_json = [&](const ArmPlan & p)
  {
    std::ostringstream o;
    o << std::setprecision(9) << "{\"pattern\":\"" << p.cut_pattern << "\",\"plan_seq\":" << p.plan_seq << ",\"duration_s\":" << p.duration()
      << ",\"cut_targets\":" << p.cut_targets.size() << ",\"segments\":[";
    for (size_t i = 0; i < p.segments.size(); ++i)
    {
      const auto & g = p.segments[i];
      o << (i ? "," : "") << "{\"kind\":\"" << toString(g.kind) << "\",\"name\":\"" << jsonEscape(g.stage_name) << "\",\"s\":"
        << g.trajectory.duration() << ",\"contact\":" << (g.tool_carton_contact_allowed ? "true" : "false") << "}";
    }
    o << "]}";
    return o.str();
  };
  rep << ",\"plans\":{\"left\":{\"A\":" << segs_json(pl.a) << ",\"B\":" << segs_json(pl.b) << "},\"right\":{\"A\":" << segs_json(pr.a)
      << ",\"B\":" << segs_json(pr.b) << "}}";
  rep << ",\"snapshot\":{\"id\":\"" << jsonEscape(pl.a.snapshot.snapshot_id) << "\",\"capture\":\"" << jsonEscape(pl.a.snapshot.capture_id)
      << "\",\"acquisition_stamp_ns\":" << pl.a.snapshot.acquisition_stamp_ns << ",\"age_at_plan_end_s\":" << age_max << "}";

  // --- top-seam coverage (fail closed, both modes): the two top cut lines, in world, on the source seam axis ---
  std::string split_summary;
  {
    const Eigen::Isometry3d snap_to_world =
      pl.a.snapshot_frame_mode == "planning_frame" ? urdfBaseToWorld(left_) : Eigen::Isometry3d::Identity();
    const Eigen::Vector3d w0 = snap_to_world * seam_a, w1 = snap_to_world * seam_b;
    auto topInterval = [&](const ArmPlan & p, ArmIo & io, const char * what, SeamInterval & iv) -> Status  // (phases)
    {
      size_t cuts = 0;
      for (const auto & seg : p.segments) cuts += seg.kind == SegmentKind::Cut;
      if (cuts != 1)
        return Status::Error(std::string("top split coverage: ") + what + " has " + std::to_string(cuts) + " CUT segments, expected exactly one (middle_only)");
      std::vector<CutLine> lines;
      Status st = cutLinesFromTargets(p.cut_targets, cuts, p.planning_frame, urdfBaseToWorld(io), lines);
      if (!st) return Status::Error(std::string("top split coverage: ") + what + ": " + st.message);
      return projectCutLineOnSeam(lines[0], w0, w1, iv);
    };
    SeamInterval li, ri;
    Status st = topInterval(side_first ? pl.b : pl.a, left_, side_first ? "left phase B" : "left phase A", li);
    if (st) st = topInterval(side_first ? pr.a : pr.b, right_, side_first ? "right phase A" : "right phase B", ri);
    if (!st) return fail(st);
    TopCoverage cov;
    st = checkTopCoverage(li, ri, (w1 - w0).norm(), tsp.end_tolerance_m, tsp.min_overlap_m, &cov, effectiveRightEndExtension(tsp));
    rep << ",\"top_coverage\":" << topCoverageJson(cov) << ",\"top_coverage_ok\":" << (st.ok ? "true" : "false");
    if (!st) return fail(st);
    split_summary = std::string("top split ") + toString(tsp.mode) + ": " + describeTopCoverage(cov) + "; ";
  }

  // --- 3/4. B's leading transits into A ---
  PhaseSet set;
  s = buildPhases(pl.a, pl.b, pr.a, pr.b, PlanTolerances{}, set);
  if (!s) return fail(s);
  rep << ",\"moved_transits\":{\"left\":" << set.moved_left << ",\"right\":" << set.moved_right << "}";

  // --- the dual scene: pole / head from the dual move_group, the carton from the per-arm scenes ---
  planning_scene::PlanningScenePtr scene;
  s = refreshScene(scene);
  if (!s) return fail(s);
  double carton_dc = 0.0, carton_dr = 0.0;
  s = addCartonFromArmScenes(scene, carton_dc, carton_dr);
  rep << ",\"carton\":{\"center_diff_m\":" << carton_dc << ",\"rotation_diff_rad\":" << carton_dr << "}";
  if (!s) return fail(s);
  std::unique_ptr<PhaseValidator> validator;
  s = makeValidator(scene, validator);
  if (!s) return fail(Status::Error("validator: " + s.message));

  // expected tip curve of the top cut per arm (middle_cut_sag_m; 0 / missing = the straight line as always)
  CutSag sag_left, sag_right;
  {
    const Eigen::Isometry3d snap_to_world =
      pl.a.snapshot_frame_mode == "planning_frame" ? urdfBaseToWorld(left_) : Eigen::Isometry3d::Identity();
    s = topSagFor(left_, source_text, snap_to_world, sag_left);
    if (s) s = topSagFor(right_, source_text, snap_to_world, sag_right);
    if (!s) return fail(s);
    rep << ",\"top_sag_m\":{\"left\":" << sag_left.sag_m << ",\"right\":" << sag_right.sag_m << "}";
  }

  // --- 5/6. validate each phase (world, tip-on-line, cross-arm, serial fallback) ---
  std::ostringstream sum;
  sum << (side_first ? "phase order side_first: " : "") << split_summary << "moved " << set.moved_left << "/" << set.moved_right << " leading transit(s) of B into A (left/right); ";
  rep << ",\"phases\":{";
  bool all_ok = true;
  std::string refusal;
  for (Phase * ph : {&set.a, &set.b})
  {
    std::vector<CutLine> ll, rl;
    for (auto [arm, lines] : {std::pair<const ArmPhase *, std::vector<CutLine> *>{&ph->left(), &ll},
                              std::pair<const ArmPhase *, std::vector<CutLine> *>{&ph->right(), &rl}})
    {
      size_t cuts = 0;
      for (const auto & seg : arm->segments) cuts += seg.kind == SegmentKind::Cut;
      ArmIo & io = this->arm(arm->side);
      s = cutLinesFromTargets(arm->cut_targets, cuts, arm->planning_frame, urdfBaseToWorld(io), *lines);
      if (!s) return fail(Status::Error("phase " + ph->name() + " " + arm->side + ": " + s.message));
      applyTopSag(*arm, *lines, arm->side == "left" ? sag_left : sag_right);
    }
    const auto t_val0 = std::chrono::steady_clock::now();
    const PhaseValidation v = validator->validate(ph->left(), ph->right(), ll, rl);
    const double val_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_val0).count();
    ph->recordValidation(PhaseValidator::toRecord(v), v.ok);
    sum << "phase " << ph->name() << ": " << v.message;
    if (std::isfinite(v.min_cross_distance_m)) sum << " [min cross " << std::fixed << std::setprecision(3) << v.min_cross_distance_m << " m]";
    sum << "; ";
    auto world_json = [](const WorldCheckResult & w)
    {
      std::ostringstream o;
      o << "{\"ok\":" << (w.safe() ? "true" : "false") << ",\"message\":\"" << jsonEscape(w.message) << "\",\"samples\":" << w.samples_checked
        << ",\"sample_index\":" << w.sample_index << ",\"max_tolerated_depth_m\":" << w.max_tolerated_depth_m << ",\"tolerated_samples\":"
        << w.tolerated_samples << ",\"max_tolerated_sample\":" << w.max_tolerated_sample << ",\"depth_m\":" << w.depth_m << ",\"link\":\"" << jsonEscape(w.link) << "\",\"object\":\"" << jsonEscape(w.object) << "\"";
      if (w.table_measured)
        o << ",\"table_min_m\":" << w.table_min_distance_m << ",\"table_min_link\":\"" << jsonEscape(w.table_min_link)
          << "\",\"table_min_time_s\":" << w.table_min_time_s;
      o << "}";
      return o.str();
    };
    auto tip_json = [](const TipGateResult & t)
    {
      std::ostringstream o;
      o << std::setprecision(9) << "{\"ok\":" << (t.ok ? "true" : "false") << ",\"cut_segments\":" << t.cut_segments << ",\"samples\":" << t.samples
        << ",\"max_deviation_m\":" << t.max_deviation_m << ",\"max_endpoint_error_m\":" << t.max_endpoint_error_m << ",\"message\":\""
        << jsonEscape(t.message) << "\"}";
      return o.str();
    };
    auto cross_json = [](const SweptVolumeResult & c)
    {
      std::ostringstream o;
      o << std::setprecision(9) << "{\"safe\":" << (c.safe() ? "true" : "false") << ",\"min_distance_m\":" << c.min_distance
        << ",\"min_distance_exact\":" << (c.min_distance_exact ? "true" : "false") << ",\"effective_min_m\":" << c.effective_min_distance
        << ",\"sampling_gap_m\":" << c.sampling_gap_m << ",\"link_a\":\"" << c.link_a << "\",\"link_b\":\"" << c.link_b
        << "\",\"sample_pairs\":" << c.sample_pairs << ",\"candidates\":" << c.candidate_pairs << ",\"exact_checks\":" << c.exact_checks
        << ",\"certified\":" << c.certified_pairs << ",\"message\":\"" << jsonEscape(c.message) << "\"}";
      return o.str();
    };
    rep << (ph->name() == "A" ? "" : ",") << "\"" << ph->name() << "\":{\"ok\":" << (v.ok ? "true" : "false") << ",\"mode\":\""
        << (v.serial ? "serial" : "parallel") << "\",\"serial_first\":\"" << v.serial_first << "\",\"message\":\"" << jsonEscape(v.message)
        << "\",\"validation_s\":" << val_s << ",\"duration_left_s\":" << ph->left().duration() << ",\"duration_right_s\":"
        << ph->right().duration() << ",\"samples_left\":" << v.samples_left << ",\"samples_right\":" << v.samples_right
        << ",\"cross_parallel\":" << cross_json(v.cross);
    if (v.serial)
      rep << ",\"cross_serial_first\":" << cross_json(v.serial_first_result) << ",\"cross_serial_second\":" << cross_json(v.serial_second_result);
    rep << ",\"min_cross_effective_m\":" << (std::isfinite(v.min_cross_distance_m) ? v.min_cross_distance_m : -1.0)
        << ",\"world_left\":" << world_json(v.world_left) << ",\"world_right\":" << world_json(v.world_right)
        << ",\"tip_left\":" << tip_json(v.tip_left) << ",\"tip_right\":" << tip_json(v.tip_right) << "}";
    if (!v.ok)
    {
      all_ok = false;
      if (refusal.empty()) refusal = "phase " + ph->name() + " refused: " + v.message;
    }
  }
  rep << "}";
  if (!all_ok) return fail(Status::Error(refusal));

  // --- 6b. the executors must hold exactly the plans of THIS attempt's last jobs (fail closed) ---
  rep << ",\"executor_plans\":{\"left_plan_seq\":" << pl.b.plan_seq << ",\"right_plan_seq\":" << pr.b.plan_seq << "}";
  s = verifyExecutorPlans(pl.b.plan_seq, pr.b.plan_seq);
  if (!s) return fail(s);

  // --- 6c. robot-side joint move legs of the serial phases (opt-in, default off; never refuses the plan) ---
  CriPlan cri;
  if (criEnabled())
  {
    std::string cri_json;
    cri = planCri(set, scene, cri_json);
    rep << ",\"cri_transit\":" << cri_json;
  }

  // --- 7. retain and display ---
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    retained_.valid = true;
    retained_.id = next_plan_id_++;
    retained_.set = set;
    retained_.variant = current_variant_;
    retained_.detection = detection_;
    retained_.executor_plan_seq_left = pl.b.plan_seq;
    retained_.executor_plan_seq_right = pr.b.plan_seq;
    retained_.cri = cri;
    retained_.measured_left = ql;
    retained_.measured_right = qr;
    retained_.snapshot_age_at_plan_s = age_max;
    retained_.summary = sum.str();
    phase_validated_ = true;
  }
  publishDisplay(set, ql, qr);
  summary = sum.str();
  finishReport(true, summary);
  return Status::Ok();
}

void CoordinatorNode::publishDisplay(const PhaseSet & set, const std::vector<double> & left0, const std::vector<double> & right0)
{
  moveit_msgs::msg::DisplayTrajectory disp;
  disp.model_id = model_->getName();
  disp.trajectory_start.joint_state.header.stamp = now();
  disp.trajectory_start.joint_state.name = left_.spec.joint_names;
  disp.trajectory_start.joint_state.name.insert(
    disp.trajectory_start.joint_state.name.end(), right_.spec.joint_names.begin(), right_.spec.joint_names.end());
  disp.trajectory_start.joint_state.position = left0;
  disp.trajectory_start.joint_state.position.insert(disp.trajectory_start.joint_state.position.end(), right0.begin(), right0.end());
  for (const Phase * ph : {&set.a, &set.b})
  {
    if (ph->left().segments.empty()) continue;   // sequence schedule: phase B is unused
    moveit_msgs::msg::RobotTrajectory rt;
    Status s = bothArmsTrajectory(*ph, 0.05, rt.joint_trajectory);
    if (!s)
    {
      RCLCPP_WARN(get_logger(), "display trajectory of phase %s: %s", ph->name().c_str(), s.message.c_str());
      continue;
    }
    disp.trajectory.push_back(std::move(rt));
  }
  display_pub_->publish(disp);
}

}  // namespace dual_cut

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<dual_cut::CoordinatorNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 10);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
