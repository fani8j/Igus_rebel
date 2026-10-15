// dual_cut_coordinator node (WP6), design section 6.
//
//   ~/plan      plan phases A and B on both per-arm executors, move B's leading transits into A,
//               validate each phase (frame gate, world, tip-on-line, cross-arm swept volume, serial
//               fallback), retain the plan, publish a both_arms DisplayTrajectory
//   ~/execute   (execution_enabled + execute_phase_{a,b}_enabled, all default false) barriers, then two
//               FollowJointTrajectory goals with one common start stamp
//   ~/stop      own callback group, never takes the plan lock: hold + cancel + latch
//   ~/rearm     only at rest; clears the retained plan and the latch
//   ~/status    dual_cut_msgs/CoordinatorStatus
//
// Threads: services and timers run on a MultiThreadedExecutor. Callback groups: plan, exec, stop (own,
// lock-free w.r.t. planning), monitor (20 Hz), joint states, io (clients / action clients, reentrant).
#pragma once

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "controller_manager_msgs/srv/list_controllers.hpp"
#include "controller_manager_msgs/srv/list_hardware_components.hpp"
#include "dual_cut_coordinator/gates.hpp"
#include "dual_cut_coordinator/phase_plan.hpp"
#include "dual_cut_coordinator/phase_validator.hpp"
#include "dual_cut_coordinator/swept_volume.hpp"
#include "dual_cut_msgs/msg/coordinator_status.hpp"
#include "dual_cut_msgs/srv/get_planned_solution.hpp"
#include "dual_cut_msgs/srv/plan_from_state.hpp"
#include "moveit_msgs/msg/display_trajectory.hpp"
#include "moveit_msgs/srv/get_planning_scene.hpp"
#include "rcl_interfaces/srv/set_parameters.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace dual_cut
{

class CoordinatorNode : public rclcpp::Node
{
public:
  using Trigger = std_srvs::srv::Trigger;
  using FJT = control_msgs::action::FollowJointTrajectory;
  using GoalHandle = rclcpp_action::ClientGoalHandle<FJT>;
  using Status_ = dual_cut_msgs::msg::CoordinatorStatus;

  explicit CoordinatorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~CoordinatorNode() override;

  /// Callback groups are created in the constructor; main() adds the node to a multi-threaded executor.
  // Test / diagnostics hooks
  int stateValue() const { return state_.load(); }
  bool stopLatched() const { return stop_latched_.load(); }

private:
  // ---- per-arm I/O ----
  enum GoalStateValue { kNone = 0, kSent, kAccepted, kRejected, kSucceeded, kAborted, kCanceled, kUnknown };
  struct GoalTrack
  {
    std::mutex m;
    GoalHandle::SharedPtr handle;
    int state = kNone;
    std::string text;
    bool unexpected = false;   // ended without us asking for it
    void reset()
    {
      std::lock_guard<std::mutex> lk(m);
      handle.reset();
      state = kNone;
      text.clear();
      unexpected = false;
    }
  };
  struct ArmIo
  {
    std::string side;
    std::string prefix;
    std::string executor_ns;
    std::string controller;
    std::string scene_ns;   // namespace of the per-arm move_group (scene with the executor's carton)
    rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr scene_client;
    ArmSpec spec;
    rclcpp::Client<rcl_interfaces::srv::SetParameters>::SharedPtr set_params;
    rclcpp::Client<dual_cut_msgs::srv::PlanFromState>::SharedPtr plan_from_state;
    rclcpp::Client<dual_cut_msgs::srv::GetPlannedSolution>::SharedPtr get_plan;
    rclcpp_action::Client<FJT>::SharedPtr action;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr hold_pub;
    GoalTrack goal;
  };
  struct ArmPlanPair
  {
    Status status;
    ArmPlan a, b;
    double a_seconds = 0.0, b_seconds = 0.0;   // wall time of the executor calls per phase
  };

  // ---- retained plan ----
  struct Retained
  {
    bool valid = false;
    uint64_t id = 0;
    PhaseSet set;
    std::vector<double> measured_left, measured_right;   // at plan time (spec order)
    double snapshot_age_at_plan_s = 0.0;
    std::string summary;
  };

  // ---- expected motion for the tracking monitor ----
  struct Expected
  {
    bool valid = false;
    JointTrajectory traj;                                    // prefixed names, concatenated
    std::vector<double> q_start, q_end;                      // spec order
    std::chrono::steady_clock::time_point t_start;           // goal start (stamp) in steady time
    bool started = false;
    bool finished = false;
  };

  // ---- parameters ----
  void declareParameters();
  rcl_interfaces::msg::SetParametersResult onSetParameters(const std::vector<rclcpp::Parameter> & params);
  double pd(const char * name) const { return get_parameter(name).as_double(); }
  bool pb(const char * name) const { return get_parameter(name).as_bool(); }
  std::string ps(const char * name) const { return get_parameter(name).as_string(); }

  // ---- model / scene ----
  Status ensureModel();
  Status refreshScene(planning_scene::PlanningScenePtr & scene);
  Status makeValidator(const planning_scene::PlanningScenePtr & scene, std::unique_ptr<PhaseValidator> & out);

  // ---- planning ----
  void handlePlan(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res);
  Status planImpl(std::string & summary);
  Status addCartonFromArmScenes(planning_scene::PlanningScenePtr & scene, double & center_diff_m, double & rot_diff_rad);
  ArmPlanPair planArm(ArmIo & io, const sensor_msgs::msg::JointState & start);
  Status gatePlan(const ArmIo & io, const ArmPlan & plan, double & snapshot_age_s);
  Status setExecutorParams(ArmIo & io, const std::vector<std::string> & kv);
  template<typename SrvT>
  std::optional<typename SrvT::Response::SharedPtr> callService(
    typename rclcpp::Client<SrvT>::SharedPtr client, typename SrvT::Request::SharedPtr req, double timeout_s,
    std::string & error);
  Eigen::Isometry3d urdfBaseToWorld(const ArmIo & io) const;
  void publishDisplay(const PhaseSet & set, const std::vector<double> & left0, const std::vector<double> & right0);

  // ---- execution ----
  void handleExecute(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res);
  void executionThread(std::vector<char> phases);
  Status waitBarriers(const Phase & phase, double snapshot_stamp_age_s_at_plan);
  Status buildGoal(const ArmPhase & arm, Expected & expected, FJT::Goal & goal);
  void sendGoal(ArmIo & io, const FJT::Goal & goal);
  bool waitGoalsDone(const std::vector<ArmIo *> & arms, double timeout_s);
  void finishExecution(int new_state);
  double snapshotAgeS() const;

  // ---- stop / rearm ----
  void handleStop(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res);
  void handleRearm(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res);
  void doStop(const std::string & reason);
  /// doStop() from inside an rclcpp_action callback would self-deadlock (the client invokes result
  /// callbacks while holding its goal-handle mutex, and cancelling takes it again): hand it to the
  /// stop worker thread instead.
  void requestStopAsync(const std::string & reason);
  void stopWorkerLoop();
  std::string stopReason() { std::lock_guard<std::mutex> lk(stop_mutex_); return stop_reason_; }

  // ---- monitor / status ----
  void monitorTick();
  void pollTick();
  void publishStatus();
  void resetExecStats();
  void publishExecReport(const std::string & json);
  void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg);
  bool armQ(const ArmIo & io, std::vector<double> & q, double * age_s = nullptr) const;
  double jointStateAgeS() const;

  ArmIo & arm(const std::string & side) { return side == "left" ? left_ : right_; }

  // members
  ArmIo left_, right_;
  rclcpp::CallbackGroup::SharedPtr cbg_plan_, cbg_exec_, cbg_stop_, cbg_monitor_, cbg_js_, cbg_io_;
  rclcpp::Service<Trigger>::SharedPtr srv_plan_, srv_execute_, srv_stop_, srv_rearm_;
  rclcpp::Publisher<Status_>::SharedPtr status_pub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr report_pub_, exec_report_pub_;
  std::string last_plan_report_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  rclcpp::TimerBase::SharedPtr monitor_timer_, poll_timer_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr scene_client_;
  rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedPtr list_controllers_;
  rclcpp::Client<controller_manager_msgs::srv::ListHardwareComponents>::SharedPtr list_hardware_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_;

  // model (guarded by model_mutex_)
  std::mutex model_mutex_;
  std::atomic<bool> model_ready_{false};   // set after model_ / monitor_checker_ are complete
  moveit::core::RobotModelPtr model_;
  planning_scene::PlanningScenePtr monitor_scene_;                // scene without world for the monitor
  std::unique_ptr<SweptVolumeChecker> monitor_checker_;           // monitor thread only

  // plan lock and retained plan
  std::mutex plan_mutex_;
  std::mutex data_mutex_;
  Retained retained_;
  uint64_t next_plan_id_ = 1;

  // state
  std::atomic<int> state_{Status_::STATE_IDLE};
  std::atomic<bool> planning_{false}, executing_{false}, stop_latched_{false};
  std::mutex stop_mutex_;
  std::string stop_reason_;
  std::string current_phase_;
  std::atomic<uint64_t> status_seq_{0};

  // joint states
  mutable std::mutex js_mutex_;
  struct JointSample
  {
    double pos = 0.0, vel = 0.0;
    std::chrono::steady_clock::time_point t;
    bool have_prev = false;
  };
  std::map<std::string, JointSample> joints_;
  std::chrono::steady_clock::time_point js_last_rx_;
  bool js_have_ = false;
  RestTracker rest_tracker_{0.03};

  // monitor results
  mutable std::mutex mon_mutex_;
  std::chrono::steady_clock::time_point mon_last_tick_;
  bool mon_healthy_ = false;
  bool mon_distance_valid_ = false;
  double mon_distance_m_ = std::numeric_limits<double>::quiet_NaN();
  std::string mon_link_a_, mon_link_b_;
  double mon_tracking_err_ = 0.0;
  // statistics over one executing phase (reset by resetExecStats())
  double exec_min_distance_m_ = std::numeric_limits<double>::infinity();
  double exec_max_tracking_rad_ = 0.0;
  size_t exec_monitor_ticks_ = 0;

  // controller / hardware poll results
  mutable std::mutex poll_mutex_;
  std::chrono::steady_clock::time_point poll_ctrl_time_, poll_hw_time_;
  bool poll_ctrl_valid_ = false, poll_hw_valid_ = false;
  bool ctrl_active_ = false, hw_active_ = false;
  std::string poll_text_;

  // execution
  std::thread exec_thread_;
  std::thread stop_worker_;
  std::mutex stop_req_mutex_;
  std::condition_variable stop_req_cv_;
  std::string stop_req_reason_;
  bool stop_req_pending_ = false;
  bool stop_worker_quit_ = false;
  std::mutex expected_mutex_;
  Expected expected_left_, expected_right_;
  std::string barrier_message_;
  bool barrier_ok_ = false;
  std::atomic<bool> phase_validated_{false};
};

}  // namespace dual_cut
