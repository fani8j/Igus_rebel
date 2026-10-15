// Phase A/B data structures and bookkeeping (WP5), design section 6 steps 2-4.
//
// Each arm is planned twice by its per-arm executor: plan A (from the measured state) and plan B
// (from A's end state). Phase B as exported starts with the left arm still above the seam centre,
// which is exactly where the right arm's half ends. So the LEADING TRANSIT segments of each B plan
// are moved to the END of the same arm's A plan (shiftLeadingTransits):
//     A' = A + leading transits of B        B' = B without them
// Afterwards phase A is  L(top half + transit to the left face) || R(right face + transit above the
// right end)  and B is the rest of both. The two arms of one phase run in parallel, each from its own
// time 0.
//
// A Phase remembers whether it has been validated. Any mutable access resets the validation, and the
// record stores the revision it was made for, so a stale "validated" can never survive an edit.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "dual_cut_coordinator/trajectory_tools.hpp"
#include "dual_cut_msgs/msg/planned_segment.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "dual_cut_msgs/srv/get_planned_solution.hpp"

namespace dual_cut
{

/// Mirrors PlannedSegment.msg kind constants.
enum class SegmentKind : uint8_t
{
  Transit = 0,
  Align = 1,
  Dwell = 2,
  Entry = 3,
  Cut = 4,
  Finish = 5,
  Lift = 6
};
const char * toString(SegmentKind kind);
Status segmentKindFromUint8(uint8_t value, SegmentKind & out);

struct PlannedSegmentData
{
  SegmentKind kind = SegmentKind::Transit;
  std::string stage_name;
  std::string cut_id;
  std::string physical_cut;
  bool tool_carton_contact_allowed = false;
  JointTrajectory trajectory;  // executor (unprefixed) joint names, time from 0
};

/// Identity of the snapshot a plan was made from; both arms and both plans of a cell must agree.
struct SnapshotIdentity
{
  std::string snapshot_id;
  std::string capture_id;
  uint64_t acquisition_stamp_ns = 0;
  bool stale_override = false;
  bool operator==(const SnapshotIdentity & o) const
  {
    return snapshot_id == o.snapshot_id && capture_id == o.capture_id &&
           acquisition_stamp_ns == o.acquisition_stamp_ns && stale_override == o.stale_override;
  }
};

/// One exported executor plan for one arm (unprefixed names).
struct ArmPlan
{
  std::string side;                      // "left" / "right"
  uint64_t plan_seq = 0;
  SnapshotIdentity snapshot;
  std::vector<std::string> joint_names;  // order of start_q / end_q
  std::vector<double> start_q;
  std::vector<double> end_q;
  std::vector<PlannedSegmentData> segments;

  // Export metadata used by the coordinator's gates (frame consistency, tip-on-line).
  std::string planning_frame;
  std::string tool_link;
  std::string cut_pattern;
  std::string snapshot_frame_mode;
  std::string snapshot_frame_id;
  std::string snapshot_relabel;
  std::array<double, 3> base_xyz{{0.0, 0.0, 0.0}};   // pose of this arm's base in the snapshot frame
  std::array<double, 3> base_rpy{{0.0, 0.0, 0.0}};   // URDF fixed-axis rpy
  double middle_cut_fraction = 1.0;
  /// Cut targets in planning_frame. Convention used by the coordinator: for the i-th CUT segment of
  /// this plan (in segment order) targets[2*i] is the cut start and targets[2*i+1] the cut end.
  std::vector<geometry_msgs::msg::PoseStamped> cut_targets;

  double duration() const;
};

Status fromMsg(const dual_cut_msgs::msg::PlannedSegment & msg, PlannedSegmentData & out);
/// side is "left" or "right". Fails (fail closed) on an unsuccessful response, unknown segment kind
/// or any inconsistency found by validateArmPlan().
Status fromResponse(
  const std::string & side, const dual_cut_msgs::srv::GetPlannedSolution::Response & resp,
  ArmPlan & out);

struct PlanTolerances
{
  double chain_tol = 1e-6;     // inside one plan: segment ends vs next starts, start_q / end_q
  double handoff_tol = 1e-9;   // B.start vs A.end (design section 6, step 2)
};

/// Joint names consistent everywhere, at least one segment, every segment valid, the chain of
/// segments continuous, start_q equals the first segment's start and end_q the last segment's end.
Status validateArmPlan(const ArmPlan & plan, const PlanTolerances & tol);

/// Move the leading TRANSIT segments of b to the end of a. Requires b.start_q == a.end_q within
/// tol.handoff_tol, the same arm and the same snapshot. Fails (leaving both unchanged) if b would
/// be left without any segment (all transits). moved receives the count.
Status shiftLeadingTransits(
  ArmPlan & a, ArmPlan & b, const PlanTolerances & tol, size_t * moved = nullptr);

/// What one arm does in one phase (a view with the same segment type).
struct ArmPhase
{
  std::string side;
  std::vector<std::string> joint_names;  // order of start_q / end_q (executor, unprefixed)
  std::string planning_frame;
  std::vector<geometry_msgs::msg::PoseStamped> cut_targets;  // of the CUT segments in this phase
  std::vector<double> start_q;
  std::vector<double> end_q;
  std::vector<PlannedSegmentData> segments;

  double duration() const;
  /// Segment trajectories with prefixed joint names (arm_X_jointN), ready for buildSampledPath() and
  /// for a FollowJointTrajectory goal after concatenate().
  Status prefixedSegments(std::vector<SegmentInput> & out) const;
};

struct ValidationRecord
{
  bool cross_arm_ok = false;
  bool world_ok_left = false;
  bool world_ok_right = false;
  bool tip_on_line_ok = false;
  bool serial_fallback = false;     // phase only valid when the arms run one after the other
  std::string serial_first;         // "left" or "right": the arm that moves first in serial mode
  double min_cross_distance_m = 0.0;
  std::string note;
  uint64_t revision = 0;            // revision of the Phase this record was made for
};

class Phase
{
public:
  Phase() = default;
  Phase(std::string name, ArmPhase left, ArmPhase right);

  const std::string & name() const { return name_; }
  const ArmPhase & left() const { return left_; }
  const ArmPhase & right() const { return right_; }
  const ArmPhase & arm(const std::string & side) const;
  /// Mutable access: bumps the revision and drops any validation.
  ArmPhase & mutableLeft();
  ArmPhase & mutableRight();

  uint64_t revision() const { return revision_; }
  double duration() const;

  /// Record the outcome of the validator for the CURRENT revision. all_ok is the validator's overall
  /// verdict (cross-arm, both world checks and the tip-on-line gate, possibly in serial mode).
  void recordValidation(const ValidationRecord & rec, bool all_ok);
  /// True only if the last recorded validation passed and the phase was not edited since.
  bool validated() const { return validated_ && validation_.revision == revision_; }
  const ValidationRecord & validation() const { return validation_; }
  /// Whether a validation was recorded and it failed.
  bool rejected() const { return recorded_ && !validated_ && validation_.revision == revision_; }

private:
  std::string name_;
  ArmPhase left_, right_;
  uint64_t revision_ = 1;
  bool validated_ = false;
  bool recorded_ = false;
  ValidationRecord validation_;
};

/// The retained plan: phases A and B built from the four exported plans.
struct PhaseSet
{
  Phase a;
  Phase b;
  size_t moved_left = 0;    // leading transits moved from B to A
  size_t moved_right = 0;
  SnapshotIdentity snapshot;
};

/// Sampled joint trajectory of BOTH arms for display (names arm_left_*, then arm_right_*), time from 0.
/// Parallel phases: both arms from t = 0, the shorter one held at its end. Serial phases (validation
/// serial_fallback, serial_first): the first arm moves while the other is held at its start, then the
/// second moves while the first is held at its end.
Status bothArmsTrajectory(const Phase & phase, double dt_s, trajectory_msgs::msg::JointTrajectory & out);

/// Build phases A and B from the four plans (left A/B, right A/B): all four validated, same snapshot,
/// A/B handoff checked, leading transits of both B plans moved into A.
Status buildPhases(
  ArmPlan left_a, ArmPlan left_b, ArmPlan right_a, ArmPlan right_b, const PlanTolerances & tol,
  PhaseSet & out);

}  // namespace dual_cut
