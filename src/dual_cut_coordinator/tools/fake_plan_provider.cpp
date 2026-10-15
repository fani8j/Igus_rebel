// Fake plan provider: serves the interface of the (not yet existing) per-arm box_h_cut_executor
// export, i.e. under <namespace>/box_h_cut_executor
//     ~/plan_h_cut_from_state   (dual_cut_msgs/PlanFromState)
//     ~/get_planned_solution    (dual_cut_msgs/GetPlannedSolution)
// plus a settable parameter `cut_pattern` (the coordinator sets it before each planning call). The
// plans are SYNTHESISED from a scenario YAML (joint-space waypoints, straight-line cut strokes by
// numeric IK on the dual model), so the coordinator cannot tell them from real exports.
//
// Parameters: arm (left|right), scenario_file, calibration_file (dual_arm_calibration YAML: the right
// base pose exported as snapshot_planning_base_*), robot_description, robot_description_semantic,
// snapshot_stamp_ns (acquisition stamp, identical for both arms; set by the launch file) or
// snapshot_age_s (age of the snapshot at provider start; it keeps growing), lie_transform (export a
// base transform that is 5 mm off), cut_line_offset_m (shift the exported cut targets sideways),
// fail_pattern (a cut_pattern for which planning fails), plan_delay_s.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include <srdfdom/model.h>
#include <urdf/model.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include "dual_cut_coordinator/gates.hpp"
#include "dual_cut_coordinator/trajectory_tools.hpp"
#include "dual_cut_msgs/msg/planned_segment.hpp"
#include "dual_cut_msgs/srv/get_planned_solution.hpp"
#include "dual_cut_msgs/srv/plan_from_state.hpp"
#include "rclcpp/rclcpp.hpp"

using dual_cut_msgs::msg::PlannedSegment;

namespace
{

struct SegSpec
{
  uint8_t kind = PlannedSegment::TRANSIT;
  std::string name;
  std::vector<double> to;       // joint target (absolute) or empty
  std::vector<double> line;     // dx dy dz length (world), straight-line tool motion via IK
  double duration = 2.0;
  bool contact = false;
  std::string cut_id = "left", physical_cut = "left";
};

uint8_t kindFrom(const std::string & k)
{
  static const std::map<std::string, uint8_t> m = {
    {"TRANSIT", PlannedSegment::TRANSIT}, {"ALIGN", PlannedSegment::ALIGN}, {"DWELL", PlannedSegment::DWELL},
    {"ENTRY", PlannedSegment::ENTRY}, {"CUT", PlannedSegment::CUT}, {"FINISH", PlannedSegment::FINISH},
    {"LIFT", PlannedSegment::LIFT}};
  return m.at(k);
}

double quinticS(double s) { return 10 * s * s * s - 15 * s * s * s * s + 6 * s * s * s * s * s; }
double quinticDs(double s) { return 30 * s * s - 60 * s * s * s + 30 * s * s * s * s; }
double quinticDds(double s) { return 60 * s - 180 * s * s + 120 * s * s * s; }

}  // namespace

class FakePlanProvider : public rclcpp::Node
{
public:
  FakePlanProvider()
  : rclcpp::Node(
      "box_h_cut_executor",
      rclcpp::NodeOptions().allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true))
  {
    side_ = get_parameter_or<std::string>("arm", "left");
    prefix_ = dual_cut::armPrefix(side_);
    const std::string urdf_xml = get_parameter("robot_description").as_string();
    const std::string srdf_xml = get_parameter("robot_description_semantic").as_string();
    auto u = std::make_shared<urdf::Model>();
    if (!u->initString(urdf_xml)) throw std::runtime_error("fake provider: bad URDF");
    auto s = std::make_shared<srdf::Model>();
    if (!s->initString(*u, srdf_xml)) throw std::runtime_error("fake provider: bad SRDF");
    model_ = std::make_shared<moveit::core::RobotModel>(urdf::ModelInterfaceSharedPtr(u), s);
    jmg_ = model_->getJointModelGroup("arm_" + side_);
    if (!jmg_) throw std::runtime_error("fake provider: no group arm_" + side_);
    tip_ = model_->getLinkModel(prefix_ + "xeg32_tool_tip");

    // scenario
    YAML::Node root = YAML::LoadFile(get_parameter("scenario_file").as_string());
    const YAML::Node arm = root["arm_" + side_];
    for (auto it = arm.begin(); it != arm.end(); ++it)
    {
      std::vector<SegSpec> segs;
      for (const auto & n : it->second)
      {
        SegSpec sp;
        sp.kind = kindFrom(n["kind"].as<std::string>());
        sp.name = n["name"].as<std::string>();
        if (n["to"]) sp.to = n["to"].as<std::vector<double>>();
        if (n["line"]) sp.line = n["line"].as<std::vector<double>>();
        sp.duration = n["duration"] ? n["duration"].as<double>() : 2.0;
        sp.contact = n["contact"] ? n["contact"].as<bool>() : false;
        if (n["cut_id"]) sp.cut_id = n["cut_id"].as<std::string>();
        if (n["physical_cut"]) sp.physical_cut = n["physical_cut"].as<std::string>();
        segs.push_back(sp);
      }
      plans_[it->first.as<std::string>()] = segs;
    }

    // exported base transform: the right arm's from the calibration YAML (like the real executor's
    // launch), the left arm is the identity (planning_frame mode)
    base_xyz_ = {0, 0, 0};
    base_rpy_ = {0, 0, 0};
    if (side_ == "right")
    {
      YAML::Node cal = YAML::LoadFile(get_parameter("calibration_file").as_string());
      base_xyz_ = cal["dual_arm_calibration"]["arm_right_base"]["xyz"].as<std::array<double, 3>>();
      base_rpy_ = cal["dual_arm_calibration"]["arm_right_base"]["rpy"].as<std::array<double, 3>>();
    }
    if (get_parameter_or<bool>("lie_transform", false)) base_xyz_[0] += 0.005;
    cut_offset_ = get_parameter_or<double>("cut_line_offset_m", 0.0);
    fail_pattern_ = get_parameter_or<std::string>("fail_pattern", "");
    plan_delay_s_ = get_parameter_or<double>("plan_delay_s", 0.0);
    const double age = get_parameter_or<double>("snapshot_age_s", 5.0);
    const int64_t fixed_stamp = get_parameter_or<int64_t>("snapshot_stamp_ns", 0);   // shared by both arms
    stamp_ns_ = fixed_stamp > 0 ? fixed_stamp : std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() -
                static_cast<int64_t>(age * 1e9);

    plan_srv_ = create_service<dual_cut_msgs::srv::PlanFromState>(
      "~/plan_h_cut_from_state",
      [this](const std::shared_ptr<dual_cut_msgs::srv::PlanFromState::Request> q,
             std::shared_ptr<dual_cut_msgs::srv::PlanFromState::Response> r) { onPlan(q, r); });
    get_srv_ = create_service<dual_cut_msgs::srv::GetPlannedSolution>(
      "~/get_planned_solution",
      [this](const std::shared_ptr<dual_cut_msgs::srv::GetPlannedSolution::Request>,
             std::shared_ptr<dual_cut_msgs::srv::GetPlannedSolution::Response> r) { onGet(r); });
    RCLCPP_INFO(get_logger(), "fake plan provider for the %s arm: %zu pattern(s), base xyz %.4f %.4f %.4f", side_.c_str(),
                plans_.size(), base_xyz_[0], base_xyz_[1], base_xyz_[2]);
  }

private:
  using JS = sensor_msgs::msg::JointState;

  Eigen::Isometry3d baseToWorld() const { return dual_cut::poseFromXyzRpy(base_xyz_, base_rpy_); }
  // The URDF value (without any lie) for FK-consistent cut targets.
  Eigen::Isometry3d urdfBase() const
  {
    moveit::core::RobotState st(model_);
    st.setToDefaultValues();
    st.updateLinkTransforms();
    return st.getGlobalLinkTransform(prefix_ + "base_link");
  }

  Eigen::Isometry3d tipPose(const std::vector<double> & q) const
  {
    moveit::core::RobotState st(model_);
    st.setToDefaultValues();
    st.setJointGroupPositions(jmg_, q);
    st.updateLinkTransforms();
    return st.getGlobalLinkTransform(tip_);
  }

  // Damped-least-squares IK tracking a straight world line with the start orientation held.
  bool ikLine(const std::vector<double> & q0, const Eigen::Vector3d & dir, double len, int steps,
              std::vector<std::vector<double>> & chain, std::string & err) const
  {
    moveit::core::RobotState st(model_);
    st.setToDefaultValues();
    st.setJointGroupPositions(jmg_, q0);
    st.updateLinkTransforms();
    const Eigen::Isometry3d T0 = st.getGlobalLinkTransform(tip_);
    chain.assign(1, q0);
    std::vector<double> q = q0;
    for (int k = 1; k <= steps; ++k)
    {
      const Eigen::Vector3d target = T0.translation() + dir.normalized() * (len * k / steps);
      bool converged = false;
      for (int it = 0; it < 200; ++it)
      {
        st.setJointGroupPositions(jmg_, q);
        st.updateLinkTransforms();
        const Eigen::Isometry3d T = st.getGlobalLinkTransform(tip_);
        Eigen::Matrix<double, 6, 1> e;
        e.head<3>() = target - T.translation();
        const Eigen::AngleAxisd aa(T0.linear() * T.linear().transpose());
        e.tail<3>() = aa.axis() * aa.angle();
        if (e.head<3>().norm() < 2e-6 && e.tail<3>().norm() < 1e-5)
        {
          converged = true;
          break;
        }
        Eigen::MatrixXd J;
        st.getJacobian(jmg_, tip_, Eigen::Vector3d::Zero(), J);
        const double lambda = 0.02;
        Eigen::Matrix<double, 6, 6> JJt = J * J.transpose() + lambda * lambda * Eigen::Matrix<double, 6, 6>::Identity();
        const Eigen::VectorXd dq = J.transpose() * JJt.ldlt().solve(e);
        for (size_t i = 0; i < q.size(); ++i) q[i] += dq[i];
      }
      if (!converged)
      {
        err = "IK did not converge on the cut line at step " + std::to_string(k);
        return false;
      }
      chain.push_back(q);
    }
    return true;
  }

  dual_cut::JointTrajectory restToRest(const std::vector<double> & from, const std::vector<double> & to, double T) const
  {
    dual_cut::JointTrajectory t;
    for (int i = 1; i <= 6; ++i) t.joint_names.push_back("joint" + std::to_string(i));
    const int n = 10;
    for (int k = 0; k <= n; ++k)
    {
      const double s = static_cast<double>(k) / n;
      dual_cut::TrajPoint p;
      p.time_s = s * T;
      for (size_t j = 0; j < 6; ++j)
      {
        const double d = to[j] - from[j];
        p.positions.push_back(k == n ? to[j] : from[j] + d * quinticS(s));
        p.velocities.push_back(d * quinticDs(s) / T);
        p.accelerations.push_back(d * quinticDds(s) / (T * T));
      }
      if (k == 0 || k == n)
      {
        std::fill(p.velocities.begin(), p.velocities.end(), 0.0);
        std::fill(p.accelerations.begin(), p.accelerations.end(), 0.0);
      }
      t.points.push_back(p);
    }
    return t;
  }

  // Straight-line chain with a rest-to-rest time law (positions from the IK chain, velocities by
  // central differences, no accelerations -> cubic splines in the controller).
  dual_cut::JointTrajectory chainTrajectory(const std::vector<std::vector<double>> & chain, double T) const
  {
    dual_cut::JointTrajectory t;
    for (int i = 1; i <= 6; ++i) t.joint_names.push_back("joint" + std::to_string(i));
    const size_t n = chain.size() - 1;
    std::vector<double> times;
    for (size_t k = 0; k <= n; ++k) times.push_back(T * quinticS(static_cast<double>(k) / n));
    for (size_t k = 0; k <= n; ++k)
    {
      dual_cut::TrajPoint p;
      p.time_s = times[k];
      p.positions = chain[k];
      for (size_t j = 0; j < 6; ++j)
      {
        double v = 0.0;
        if (k > 0 && k < n) v = (chain[k + 1][j] - chain[k - 1][j]) / (times[k + 1] - times[k - 1]);
        p.velocities.push_back(v);
      }
      t.points.push_back(p);
    }
    // times must be strictly increasing; quinticS makes the first/last steps tiny but positive
    return t;
  }

  void onPlan(const std::shared_ptr<dual_cut_msgs::srv::PlanFromState::Request> req,
              std::shared_ptr<dual_cut_msgs::srv::PlanFromState::Response> res)
  {
    std::lock_guard<std::mutex> lk(m_);
    const std::string pattern = get_parameter_or<std::string>("cut_pattern", "none");
    auto fail = [&](const std::string & msg)
    {
      res->success = false;
      res->message = msg;
      RCLCPP_WARN(get_logger(), "plan_h_cut_from_state failed: %s", msg.c_str());
    };
    // input validation like the real handler: exactly joint1..6, finite
    std::vector<double> q0(6, 0.0);
    if (req->start_state.name.size() != 6 || req->start_state.position.size() != 6) return fail("start_state must have exactly 6 joints");
    for (int i = 1; i <= 6; ++i)
    {
      auto it = std::find(req->start_state.name.begin(), req->start_state.name.end(), "joint" + std::to_string(i));
      if (it == req->start_state.name.end()) return fail("start_state lacks joint" + std::to_string(i));
      const double v = req->start_state.position[static_cast<size_t>(it - req->start_state.name.begin())];
      if (!std::isfinite(v)) return fail("non-finite start_state");
      q0[static_cast<size_t>(i - 1)] = v;
    }
    if (plan_delay_s_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(plan_delay_s_));
    if (!fail_pattern_.empty() && pattern == fail_pattern_) return fail("scenario: planning fails for pattern " + pattern);
    auto it = plans_.find(pattern);
    if (it == plans_.end()) return fail("scenario has no plan for cut_pattern '" + pattern + "'");

    Retained r;
    r.pattern = pattern;
    r.start = q0;
    std::vector<double> q = q0;
    const Eigen::Isometry3d urdf_T = urdfBase();
    for (const SegSpec & sp : it->second)
    {
      dual_cut::JointTrajectory traj;
      std::vector<double> qe;
      if (!sp.line.empty())
      {
        std::vector<std::vector<double>> chain;
        std::string err;
        const int steps = std::max(8, static_cast<int>(sp.line[3] / 0.004));
        if (!ikLine(q, Eigen::Vector3d(sp.line[0], sp.line[1], sp.line[2]), sp.line[3], steps, chain, err))
          return fail("segment '" + sp.name + "': " + err);
        traj = chainTrajectory(chain, sp.duration);
        qe = chain.back();
      }
      else
      {
        qe = sp.to;
        if (qe.size() != 6) return fail("segment '" + sp.name + "' needs 'to' or 'line'");
        traj = restToRest(q, qe, sp.duration);
      }
      if (sp.kind == PlannedSegment::CUT)
      {
        // world tip positions of the stroke ends, exported in the planning frame (URDF value: the
        // transform lie, if any, only affects the exported snapshot_planning_base_*)
        for (const auto & qq : {q, qe})
        {
          Eigen::Vector3d p = urdf_T.inverse() * tipPose(qq).translation();
          p.y() += cut_offset_;
          r.cut_targets.push_back(p);
        }
      }
      r.segments.push_back({sp, traj});
      q = qe;
    }
    r.end = q;
    r.seq = ++seq_;
    retained_ = r;
    have_ = true;
    res->success = true;
    res->message = "fake plan for pattern " + pattern;
    res->plan_seq = r.seq;
    RCLCPP_INFO(get_logger(), "planned pattern '%s' (%zu segments), plan_seq %lu", pattern.c_str(), r.segments.size(),
                static_cast<unsigned long>(r.seq));
  }

  void onGet(std::shared_ptr<dual_cut_msgs::srv::GetPlannedSolution::Response> res)
  {
    std::lock_guard<std::mutex> lk(m_);
    if (!have_)
    {
      res->success = false;
      res->message = "no plan retained";
      return;
    }
    const Retained & r = retained_;
    res->success = true;
    res->message = "fake export";
    res->plan_seq = r.seq;
    res->snapshot_id = "fake-snapshot";
    res->capture_id = "fake-capture-1";
    res->acquisition_stamp_ns = static_cast<uint64_t>(stamp_ns_);
    res->stale_override = false;
    res->planning_frame = "base_link";
    res->tool_link = "xeg32_tool_tip";
    res->cut_pattern = r.pattern;
    res->snapshot_frame_mode = side_ == "left" ? "planning_frame" : "fixed_transform";
    res->snapshot_frame_id = "base_link";
    res->snapshot_relabel = side_ == "left" ? "none" : "rotate_180";
    for (size_t i = 0; i < 3; ++i)
    {
      res->snapshot_planning_base_xyz[i] = base_xyz_[i];
      res->snapshot_planning_base_rpy[i] = base_rpy_[i];
    }
    res->middle_cut_fraction = 1.0;
    for (const auto & p : r.cut_targets)
    {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = "base_link";
      ps.pose.position.x = p.x();
      ps.pose.position.y = p.y();
      ps.pose.position.z = p.z();
      ps.pose.orientation.w = 1.0;
      res->cut_targets.push_back(ps);
    }
    auto js = [&](const std::vector<double> & q)
    {
      sensor_msgs::msg::JointState j;
      for (int i = 1; i <= 6; ++i) j.name.push_back("joint" + std::to_string(i));
      j.position = q;
      return j;
    };
    res->start_state = js(r.start);
    res->end_state = js(r.end);
    for (const auto & [sp, traj] : r.segments)
    {
      PlannedSegment seg;
      seg.kind = sp.kind;
      seg.stage_name = sp.name;
      seg.cut_id = sp.cut_id;
      seg.physical_cut = sp.physical_cut;
      seg.tool_carton_contact_allowed = sp.contact;
      seg.trajectory.joint_trajectory = dual_cut::toMsg(traj);
      res->segments.push_back(seg);
    }
    res->summary = "fake plan, " + std::to_string(r.segments.size()) + " segments";
  }

  struct Retained
  {
    std::string pattern;
    std::vector<double> start, end;
    std::vector<std::pair<SegSpec, dual_cut::JointTrajectory>> segments;
    std::vector<Eigen::Vector3d> cut_targets;
    uint64_t seq = 0;
  };

  std::string side_, prefix_, fail_pattern_;
  moveit::core::RobotModelPtr model_;
  const moveit::core::JointModelGroup * jmg_ = nullptr;
  const moveit::core::LinkModel * tip_ = nullptr;
  std::map<std::string, std::vector<SegSpec>> plans_;
  std::array<double, 3> base_xyz_{}, base_rpy_{};
  double cut_offset_ = 0.0, plan_delay_s_ = 0.0;
  int64_t stamp_ns_ = 0;
  std::mutex m_;
  Retained retained_;
  bool have_ = false;
  uint64_t seq_ = 0;
  rclcpp::Service<dual_cut_msgs::srv::PlanFromState>::SharedPtr plan_srv_;
  rclcpp::Service<dual_cut_msgs::srv::GetPlannedSolution>::SharedPtr get_srv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakePlanProvider>());   // single-threaded: requests are served one at a time
  rclcpp::shutdown();
  return 0;
}
