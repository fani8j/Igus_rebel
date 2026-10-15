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
}

rcl_interfaces::msg::SetParametersResult CoordinatorNode::onSetParameters(const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult r;
  r.successful = true;
  static const std::vector<std::string> read_only = {
    "left_executor_ns", "right_executor_ns", "left_scene_ns", "right_scene_ns", "carton_object_id", "left_controller", "right_controller", "controller_manager",
    "model_node", "planning_scene_service", "display_topic", "expected_planning_frame", "expected_tool_link",
    "tip_link_suffix", "robot_description", "robot_description_semantic", "monitor_rate_hz",
    "expected_hardware_components"};
  static const std::vector<std::string> runtime = {
    "execution_enabled", "execute_phase_a_enabled", "execute_phase_b_enabled", "monitor_stop_distance_m",
    "joint_state_stale_s", "tracking_stop_rad", "tracking_warn_rad", "barrier_joint_state_age_s",
    "barrier_start_tolerance_rad", "barrier_rest_velocity", "barrier_rest_duration_s", "barrier_timeout_s",
    "start_delay_s", "external_snapshot_max_age_s"};
  for (const auto & p : params)
  {
    const std::string & n = p.get_name();
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
    a->scene_client = create_client<moveit_msgs::srv::GetPlanningScene>(
      a->scene_ns + "/get_planning_scene", rmw_qos_profile_services_default, cbg_io_);
    a->plan_from_state = create_client<dual_cut_msgs::srv::PlanFromState>(
      a->executor_ns + "/plan_h_cut_from_state", rmw_qos_profile_services_default, cbg_io_);
    a->get_plan = create_client<dual_cut_msgs::srv::GetPlannedSolution>(
      a->executor_ns + "/get_planned_solution", rmw_qos_profile_services_default, cbg_io_);
    a->action = rclcpp_action::create_client<FJT>(this, "/" + a->controller + "/follow_joint_trajectory", cbg_io_);
    a->hold_pub = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/" + a->controller + "/joint_trajectory", rclcpp::QoS(10).reliable());
  }
  scene_client_ = create_client<moveit_msgs::srv::GetPlanningScene>(
    ps("planning_scene_service"), rmw_qos_profile_services_default, cbg_io_);
  list_controllers_ = create_client<controller_manager_msgs::srv::ListControllers>(
    ps("controller_manager") + "/list_controllers", rmw_qos_profile_services_default, cbg_io_);
  list_hardware_ = create_client<controller_manager_msgs::srv::ListHardwareComponents>(
    ps("controller_manager") + "/list_hardware_components", rmw_qos_profile_services_default, cbg_io_);

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

Status CoordinatorNode::makeValidator(const planning_scene::PlanningScenePtr & scene, std::unique_ptr<PhaseValidator> & out)
{
  PhaseValidatorConfig cfg;
  cfg.swept.padding_m = pd("swept_padding_m");
  cfg.swept.exact_distance_tolerance_m = pd("exact_distance_tolerance_m");
  cfg.swept.stop_at_first_violation = true;   // a verdict needs one violation, not the exact minimum
  cfg.tip_tolerance_m = pd("tip_on_line_tolerance_m");
  cfg.world_penetration_tolerance_m = pd("world_penetration_tolerance_m");
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

CoordinatorNode::ArmPlanPair CoordinatorNode::planArm(ArmIo & io, const sensor_msgs::msg::JointState & start)
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
  const std::string key = io.side == "left" ? "left" : "right";
  out.status = planPhase("A", get_parameter(key + "_phase_a_params").as_string_array(), start, out.a);
  out.a_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!out.status) return out;
  sensor_msgs::msg::JointState from_a;
  from_a.name = out.a.joint_names;
  from_a.position = out.a.end_q;
  out.status = planPhase("B", get_parameter(key + "_phase_b_params").as_string_array(), from_a, out.b);
  out.b_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - out.a_seconds;
  return out;
}

void CoordinatorNode::handlePlan(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res)
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
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    retained_ = Retained{};
    phase_validated_ = false;
  }
  std::string summary;
  const Status s = planImpl(summary);
  planning_ = false;
  if (!stop_latched_) state_ = s ? Status_::STATE_PLANNED : Status_::STATE_IDLE;
  res->success = s.ok;
  res->message = s.ok ? summary : s.message;
  if (s) RCLCPP_INFO(get_logger(), "plan ok: %s", summary.c_str());
  else RCLCPP_WARN(get_logger(), "plan refused: %s", s.message.c_str());
  publishStatus();
}

Status CoordinatorNode::planImpl(std::string & summary)
{
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

  // --- 2. plan both arms, phases A and B, in parallel ---
  const auto t_plan0 = std::chrono::steady_clock::now();
  auto fl = std::async(std::launch::async, [&]() { return planArm(left_, jointState(left_, ql)); });
  auto fr = std::async(std::launch::async, [&]() { return planArm(right_, jointState(right_, qr)); });
  ArmPlanPair pl = fl.get();
  ArmPlanPair pr = fr.get();
  const double plan_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_plan0).count();
  std::ostringstream rep;   // JSON report, published on ~/plan_report whatever the outcome
  rep << std::setprecision(9);
  rep << "{\"executor_calls\":{\"wall_s\":" << plan_wall_s << ",\"left\":{\"A_s\":" << pl.a_seconds << ",\"B_s\":" << pl.b_seconds
      << "},\"right\":{\"A_s\":" << pr.a_seconds << ",\"B_s\":" << pr.b_seconds << "}}";
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

  // --- 5/6. validate each phase (world, tip-on-line, cross-arm, serial fallback) ---
  std::ostringstream sum;
  sum << "moved " << set.moved_left << "/" << set.moved_right << " leading transit(s) of B into A (left/right); ";
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
        << w.tolerated_samples << ",\"max_tolerated_sample\":" << w.max_tolerated_sample << ",\"depth_m\":" << w.depth_m << ",\"link\":\"" << jsonEscape(w.link) << "\",\"object\":\"" << jsonEscape(w.object) << "\"}";
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

  // --- 7. retain and display ---
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    retained_.valid = true;
    retained_.id = next_plan_id_++;
    retained_.set = set;
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
