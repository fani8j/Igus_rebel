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

#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <functional>
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
#include "controller_manager_msgs/srv/switch_controller.hpp"
#include "dual_cut_coordinator/cri_transit.hpp"
#include "dual_cut_coordinator/gates.hpp"
#include "dual_cut_coordinator/phase_plan.hpp"
#include "dual_cut_coordinator/phase_validator.hpp"
#include "dual_cut_coordinator/plan_policy.hpp"
#include "dual_cut_coordinator/sequence_plan.hpp"
#include "dual_cut_coordinator/swept_volume.hpp"
#include "dual_cut_coordinator/top_split.hpp"
#include "dual_cut_msgs/msg/coordinator_status.hpp"
#include "dual_cut_msgs/srv/get_planned_solution.hpp"
#include "dual_cut_msgs/srv/plan_from_state.hpp"
#include "igus_rebel_msgs/action/move_joint.hpp"
#include "igus_rebel_msgs/msg/hardware_state.hpp"
#include "moveit_msgs/msg/display_trajectory.hpp"
#include "moveit_msgs/srv/get_planning_scene.hpp"
#include "rcl_interfaces/srv/get_parameters.hpp"
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
  using MoveJoint = igus_rebel_msgs::action::MoveJoint;
  using MoveJointHandle = rclcpp_action::ClientGoalHandle<MoveJoint>;
  using HardwareState = igus_rebel_msgs::msg::HardwareState;

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
    rclcpp::Client<rcl_interfaces::srv::GetParameters>::SharedPtr get_params;
    rclcpp::Client<dual_cut_msgs::srv::PlanFromState>::SharedPtr plan_from_state;
    rclcpp::Client<dual_cut_msgs::srv::GetPlannedSolution>::SharedPtr get_plan;
    rclcpp_action::Client<FJT>::SharedPtr action;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr hold_pub;
    GoalTrack goal;
    // robot-side joint move (cri_transit_enabled; all of it is passive while that parameter is false)
    rclcpp_action::Client<MoveJoint>::SharedPtr cri_action;
    rclcpp::Subscription<HardwareState>::SharedPtr cri_hw_sub;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr cri_js_sub;
  };
  /// Everything the coordinator knows about one arm's CRI side, guarded by cri_mutex_. Index 0 = left, 1 = right.
  struct CriArmState
  {
    // ~/hardware_state
    CriHwView hw;                                          // age_s is recomputed on read
    std::chrono::steady_clock::time_point hw_rx;
    int64_t hw_stamp_ns = 0;                               // header stamp (system clock), 0 = none
    std::string latch_reason;
    // ~/robot_joint_states (spec joint order)
    bool live_have = false;
    std::vector<double> live_q;
    std::chrono::steady_clock::time_point live_rx;
    int64_t live_stamp_ns = 0;
    WindowVelocity window{0.15, 0.5};
    RestTracker rest{0.03};
    // JTC expectation (controller gate)
    CtrlGateMode mode = CtrlGateMode::Normal;
    std::chrono::steady_clock::time_point mode_since;
    // the run in flight (monitor)
    bool run_active = false;
    std::vector<double> run_start, run_goal;               // spec order
    double run_path_tol = 0.01;
    std::chrono::steady_clock::time_point run_sent;
    double run_timeout_s = 0.0;
    bool run_result_received = false;
    // MoveJoint goal tracking
    bool sending = false;                                  // goal sent, response not yet seen
    MoveJointHandle::SharedPtr handle;
    bool accepted = false, rejected = false, have_result = false;
    MoveJointResultView result;
    double max_path_deviation = 0.0;
    double progress = 0.0;
  };
  /// One run of an arm-phase for the exec report.
  struct CriRunReport
  {
    std::string arm, type, note;
    size_t first_segment = 0, end_segment = 0;
    double planned_s = 0.0, wall_s = 0.0;
    bool ok = false;
    // MoveJoint (type cri)
    double model_s = 0.0;               // the driver's duration model (the timeout is 2 x this + 10 s)
    int action_outcome = -1, error_code = -1;
    std::string end_reason, message;
    bool stop_sent = false, stop_confirmed = false;
    double max_path_deviation_rad = 0.0;
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
    double sequence_margin_s = 2.0;
    bool sequence = false;   // schedule_mode sequence: set.a holds both arms' whole timelines, set.b is unused
    std::string variant;     // plan_fallback_chain entry that produced this plan
    DetectionPermission detection;   // cut_permission of the plan's source snapshot (execute gate)
    uint64_t executor_plan_seq_left = 0, executor_plan_seq_right = 0;   // plan the executors retain (their last job)
    CriPlan cri;             // robot-side joint move legs (cri_transit_enabled at plan time; serial phases only)
  };

  // ---- sequence schedule (sequence_mode.cpp) ----
  struct SeqInputs
  {
    sensor_msgs::msg::JointState left_start, right_start;
    std::vector<double> ql, qr;
    TopSplitParams tsp;
    std::string source_path, source_sha, src_param, source_text;
    Eigen::Vector3d seam_a = Eigen::Vector3d::Zero(), seam_b = Eigen::Vector3d::Zero();
    double seam_length_m = 0.0;
    /// The LEFT arm's two jobs in the order they are planned and run (orderLeftJobs): job 1 never returns home.
    std::vector<std::string> left_job1, left_job2, right_job2;
    std::string left_job1_label = "left face", left_job2_label = "left top";
    bool left_top_first = false;   // job 1 is the left TOP cut (job 2 the face cut)
    /// May block (left_max: waits for the left reach of job 2) and fills the executor parameters of right job 1.
    /// skip is set when the left top reaches top_split_skip_right_top_fraction: then there is no right job 1.
    std::function<Status(std::vector<std::string> &, bool & skip)> right_job1;
    /// Called exactly once when the left TOP job is finished or will never be planned (success or not): after job 2
    /// (face first) or right after job 1 (top first).
    std::function<void(const Status &, const ArmPlan &)> left_after_top;
    std::function<std::string()> split_json;   // JSON members (no braces) with the derived top split numbers
    double margin_s = 2.0, step_s = 1.0, max_delay_s = 60.0;
  };
  struct SeqOutcome
  {
    Status status;
    bool fallback = false;         // sequence planned fine but no schedule passed: plan the normal phases
    std::string fallback_reason;
    std::string fallback_json;     // diagnostics of the failed schedule search (goes into the attempt record)
    uint64_t executor_seq_left = 0, executor_seq_right = 0;   // plan_seq of the plans the executors must hold
  };
  std::string retry_json_;   // "plan_retry" object of the current attempt (planning thread only)
  std::string fallback_json_;     // "plan_fallback" object of the current attempt (planning thread only)
  std::string detection_json_;    // "detection" object of the current plan (planning thread only)
  DetectionPermission detection_; // cut_permission of the current plan's source snapshot (planning thread only)
  std::string current_variant_;       // plan_fallback_chain entry being planned (planning thread only)
  size_t attempt_index_ = 0;          // index of the current attempt in the chain (planning thread only)
  std::string attempt_detail_json_;   // extra diagnostics of the current attempt (sequence schedule search)
  SeqOutcome planSequence(const SeqInputs & in, std::string & summary);
  void runSequence(const Retained & plan);

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
  /// extra_padding_m > 0 (the CRI certificate): cross-arm padding, table clearance and object clearance all grow by it.
  Status makeValidator(const planning_scene::PlanningScenePtr & scene, std::unique_ptr<PhaseValidator> & out, double extra_padding_m = 0.0);

  // ---- planning ----
  void handlePlan(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res);
  /// The whole plan chain (the former handlePlan). handlePlan calls it once and, if the refusal is caused by the left
  /// side cut reverse patch of an executor (stage names 'left side cut patch'), a second time with the patch switched off.
  void handlePlanChain(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res);
  /// Reads a bool planning parameter of an executor (false = not set / unreachable).
  bool executorBool(ArmIo & io, const std::string & name);
  Status planImpl(const PlanVariant & variant, std::string & summary);
  /// Fail closed: both executors must still hold exactly the plans (plan_seq) of the chosen attempt's last jobs.
  Status verifyExecutorPlans(uint64_t left_seq, uint64_t right_seq);
  /// Adds `members` (JSON members without braces) to the last published plan report and publishes it again.
  void amendPlanReport(const std::string & members);
  Status addCartonFromArmScenes(planning_scene::PlanningScenePtr & scene, double & center_diff_m, double & rot_diff_rad);
  /// Plans phase A with params_a from `start`, then phase B from A's end with the list params_b() returns
  /// (called after A; it may block, e.g. for the other arm's export). after_a (optional) is called exactly
  /// once when phase A is finished, successfully or not. params_a_fn (optional) replaces params_a and may
  /// block (phase_order side_first: the right arm's top phase waits for the left reach). after_b (optional)
  /// is called exactly once when phase B is finished or will never be planned (A or the B list failed).
  ArmPlanPair planArm(
    ArmIo & io, const sensor_msgs::msg::JointState & start, const std::vector<std::string> & params_a,
    const std::function<Status(std::vector<std::string> &)> & params_b,
    const std::function<void(const Status &, const ArmPlan &)> & after_a = nullptr,
    const std::function<Status(std::vector<std::string> &)> & params_a_fn = nullptr,
    const std::function<void(const Status &, const ArmPlan &)> & after_b = nullptr);
  TopSplitParams topSplitParams() const;
  /// external_snapshot_file of the LEFT executor (single source of truth for the top-split source).
  Status leftSnapshotPath(std::string & path);
  Status gatePlan(const ArmIo & io, const ArmPlan & plan, double & snapshot_age_s);
  /// Expected tip curve of the top (middle_only) cut of this arm: reads the executor's planning parameter
  /// middle_cut_sag_m (0 or missing = inactive sag, i.e. today's straight line check) and, if > 0, builds the sag
  /// geometry from the SOURCE snapshot text in the world (snapshot_to_world as the coverage check uses it).
  Status topSagFor(ArmIo & io, const std::string & source_text, const Eigen::Isometry3d & snapshot_to_world, CutSag & out);
  Status setExecutorParams(ArmIo & io, const std::vector<std::string> & kv);
  template<typename SrvT>
  std::optional<typename SrvT::Response::SharedPtr> callService(
    typename rclcpp::Client<SrvT>::SharedPtr client, typename SrvT::Request::SharedPtr req, double timeout_s,
    std::string & error);
  Eigen::Isometry3d urdfBaseToWorld(const ArmIo & io) const;
  void publishDisplay(const PhaseSet & set, const std::vector<double> & left0, const std::vector<double> & right0);

  // ---- robot-side joint move (cri_execution.cpp) ----
  bool criEnabled() const { return get_parameter("cri_transit_enabled").as_bool(); }
  CriParams criParams() const;
  void onHardwareState(size_t arm, const HardwareState::SharedPtr msg);
  void onRobotJointStates(size_t arm, const sensor_msgs::msg::JointState::SharedPtr msg);
  /// Snapshots under cri_mutex_ (ages are computed now).
  CriHwView criHwView(size_t arm) const;
  CriLiveView criLiveView(size_t arm) const;
  /// Barrier / next-phase / rearm gate of BOTH arms (cri enabled): hardware idle+healthy, live measurement fresh.
  Status criBothArmsGate(double live_max_age_s);
  /// Live (robot_joint_states) joint values of an arm, spec order; false if none.
  bool criQ(const ArmIo & io, std::vector<double> & q, double * age_s = nullptr) const;
  /// armQ() when CRI is off (exactly as before); the live measurement when it is on.
  bool armQEff(const ArmIo & io, std::vector<double> & q, double * age_s = nullptr) const;
  /// Seconds both arms have been at rest: rest_tracker_ (/joint_states) when CRI is off, the live per-arm trackers when on.
  double restDurationS();
  size_t armIndex(const ArmIo & io) const { return &io == &left_ ? 0 : 1; }
  void criSetMode(size_t arm, CtrlGateMode mode);
  /// Plan time (after the phase validation, phases mode): eligibility + certificate + selection per serial arm-phase.
  CriPlan planCri(const PhaseSet & set, const planning_scene::PlanningScenePtr & scene, std::string & json);
  /// Strict deactivate / activate of one arm's trajectory controller and the confirmation via list_controllers.
  Status criSwitchController(ArmIo & io, bool activate);
  Status criControllerState(ArmIo & io, std::string & state);
  /// One CRI run: pre-dispatch gate, strict deactivate, MoveJoint, supervision, result classification, hardware idle,
  /// strict reactivate, hold check. Returns false on any fault (the caller stops; doStop has been requested).
  bool runCriLeg(ArmIo & io, const ArmPhase & arm, const CriLegRecord & leg, const CriParams & params, CriRunReport & rep);
  bool waitAtRest(const std::vector<ArmIo *> & arms, double timeout_s);
  /// Rearm: strict reactivation of a JTC that a CRI run (or its fault) left inactive, after the hardware is clean.
  Status criRecoverControllers();
  std::string criRunReportJson(const std::vector<CriRunReport> & runs, const std::string & note) const;

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
  rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr switch_controller_;
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
  std::array<std::string, 2> ctrl_state_;                 // list_controllers state of each arm's trajectory controller ("" = not listed)
  std::chrono::steady_clock::time_point poll_ctrl_sent_;  // when the request of the cached answer was SENT

  // robot-side joint move state (see CriArmState); touched from the telemetry callbacks, the monitor and the exec thread
  mutable std::mutex cri_mutex_;
  std::array<CriArmState, 2> cri_;

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
  /// A robot-side joint move was started (or is being set up) since the last rearm: doStop() takes the CRI steps. false = today's doStop.
  std::atomic<bool> cri_used_{false};
};

}  // namespace dual_cut
