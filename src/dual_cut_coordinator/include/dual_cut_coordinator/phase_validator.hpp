// Validation of one phase (design section 6, step 5): world check per arm, tip-on-line gate, cross-arm
// swept volume, and the per-phase serial fallback ("one arm after the other"). Library class with
// plain gtests on the synthetic cell; the node only supplies the scene and the exported cut lines.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dual_cut_coordinator/gates.hpp"
#include "dual_cut_coordinator/phase_plan.hpp"
#include "dual_cut_coordinator/swept_volume.hpp"
#include "dual_cut_coordinator/trajectory_tools.hpp"

namespace dual_cut
{

/// Memo for the segment-window check: paths and pair results by content key (a candidate schedule that only
/// shifts or delays segments reuses all earlier pair results).
struct SegmentPairCache
{
  std::map<uint64_t, SampledPath> paths;
  std::map<std::pair<uint64_t, uint64_t>, SweptVolumeResult> pairs;
};

struct WindowPair
{
  size_t left_index = 0, right_index = 0;
  std::string left_name, right_name;
  bool safe = false;
  double effective_min_m = 0.0;
};

struct WindowCrossResult
{
  SweptVolumeResult worst;          // the unsafe / smallest-distance pair (an error pair if there is one)
  std::string worst_left, worst_right;
  size_t overlapping_pairs = 0;
  std::vector<WindowPair> pairs;    // every overlapping pair
  bool safe() const { return worst.safe(); }
};

struct PhaseValidatorConfig
{
  SweptVolumeConfig swept;                         // padding, exact tolerance, caps
  std::vector<std::string> contact_object_ids{"detected_carton"};
  std::vector<std::string> contact_links_left;     // tool links allowed to touch the carton in CUT samples
  std::vector<std::string> contact_links_right;
  size_t world_max_samples = 200000;
  double world_penetration_tolerance_m = 0.0;     // see WorldCheckConfig; 0 = strict
  std::string table_link_name = "table";           // table clearance gate (WorldCheckConfig); 0 clearance = off
  double table_min_clearance_m = 0.0;
  double world_object_clearance_m = 0.0;              // WorldCheckConfig::object_clearance_m (0 = off; the CRI certificate sets it)
  std::vector<std::string> table_exempt_links_left;   // full names of the arm links exempt from the gate
  std::vector<std::string> table_exempt_links_right;
  double tip_tolerance_m = 0.002;                  // tip-on-line gate
  std::string tip_link_suffix = "xeg32_tool_tip";  // arm_X_<suffix>
  SampleOptions sample;                            // dt 0.05 s, 0.01 rad
  ConcatOptions concat;
  bool allow_serial_fallback = true;
};

struct PhaseValidation
{
  bool ok = false;               // phase may be executed (in parallel, or serially if serial)
  std::string message;
  bool serial = false;           // only valid with the arms running one after the other
  std::string serial_first;      // "left" / "right"
  SweptVolumeResult cross;       // the parallel check
  SweptVolumeResult serial_first_result, serial_second_result;
  WorldCheckResult world_left, world_right;
  TipGateResult tip_left, tip_right;
  size_t samples_left = 0, samples_right = 0;

  /// min cross-arm distance (effective) of the mode that is used; NaN if none passed.
  double min_cross_distance_m = std::numeric_limits<double>::quiet_NaN();
};

/// Result of PhaseValidator::checkChord(): the straight joint-space chord of ONE arm checked against the world and against
/// the other arm PARKED (serial phase), with whatever extra clearances this validator was configured with.
struct ChordCheck
{
  bool ok = false;
  std::string reason;                 // first failure (empty when ok)
  size_t samples = 0;
  WorldCheckResult world;
  SweptVolumeResult cross;
  double cross_required_m = std::numeric_limits<double>::quiet_NaN();
  double cross_effective_min_m = std::numeric_limits<double>::quiet_NaN();
  double table_required_m = std::numeric_limits<double>::quiet_NaN();
  double table_min_m = std::numeric_limits<double>::quiet_NaN();
  double object_required_m = std::numeric_limits<double>::quiet_NaN();
  double object_min_m = std::numeric_limits<double>::quiet_NaN();
};

/// Result of PhaseValidator::certifyChord(): the full chord, or its maximal contiguous middle stretch.
struct ChordStretch
{
  bool found = false;
  std::string reason;                 // why nothing was found
  bool full = false;                  // the whole chord passed
  size_t first_sample = 0, last_sample = 0, total_steps = 0;   // stretch = samples [first, last] of a chord of total_steps steps
  double s0 = 0.0, s1 = 1.0;          // progress along the chord
  std::vector<double> start, goal;    // the chord points at s0 / s1 (plan joint order)
  double length_rad = 0.0;            // leading joint change of the stretch
  ChordCheck check;                   // the final check of exactly this stretch (clearances reported from it)
};

class PhaseValidator
{
public:
  /// scene: the dual planning scene (robot model + world objects + ACM).
  static Status create(
    const planning_scene::PlanningSceneConstPtr & scene, const PhaseValidatorConfig & config,
    std::unique_ptr<PhaseValidator> & out);

  /// left_lines / right_lines: world cut lines of each arm's CUT segments (see cutLinesFromTargets).
  PhaseValidation validate(
    const ArmPhase & left, const ArmPhase & right, const std::vector<CutLine> & left_lines,
    const std::vector<CutLine> & right_lines) const;

  /// Sequence schedule cross-arm check: SEGMENT-WINDOW ALL-PAIRS (see sequence_plan.hpp). For every left and
  /// right segment (and each arm's tail hold pose) whose windows [planned start - margin, planned end +
  /// margin] overlap, the ordinary all-pairs swept check of those two segments (no time assumption).
  WindowCrossResult crossWindows(
    const ArmPhase & left, const ArmPhase & right, double margin_s, SegmentPairCache * cache = nullptr) const;

  /// Like validate() (world check, tip-on-line) but the cross-arm check is crossWindows() and there is NO
  /// serial fallback. v.cross is the worst pair.
  PhaseValidation validateSequence(
    const ArmPhase & left, const ArmPhase & right, const std::vector<CutLine> & left_lines,
    const std::vector<CutLine> & right_lines, double margin_s, SegmentPairCache * cache,
    WindowCrossResult * windows = nullptr) const;

  /// Certificate of a robot-side joint move: the chord start->goal of arm `side` (plan joint names/order), re-sampled with at
  /// most max_step_rad per joint, against the world (collision, table clearance, object clearance as configured) with
  /// the other arm held at other_parked_q (other_joint_names order), and against that parked arm with the cross-arm swept
  /// check. The padding is the one of THIS validator's config, so create a validator with the extra padding for it.
  ChordCheck checkChord(
    const std::string & side, const std::vector<std::string> & joint_names, const std::vector<double> & start,
    const std::vector<double> & goal, const std::vector<std::string> & other_joint_names,
    const std::vector<double> & other_parked_q, double max_step_rad) const;

  /// checkChord() of the whole chord; if only its ends fail (close to an object, the table or the parked arm), the maximal contiguous
  /// run of samples that pass the same checks one by one, re-checked as a chord of its own (the sampling gap of the swept check
  /// included, trimmed one sample at both ends until it passes), must change the leading joint by at least min_length_rad.
  ChordStretch certifyChord(
    const std::string & side, const std::vector<std::string> & joint_names, const std::vector<double> & start,
    const std::vector<double> & goal, const std::vector<std::string> & other_joint_names,
    const std::vector<double> & other_parked_q, double max_step_rad, double min_length_rad) const;

  /// Fill a ValidationRecord from a result.
  static ValidationRecord toRecord(const PhaseValidation & v);

  const ArmSpec & leftSpec() const { return left_; }
  const ArmSpec & rightSpec() const { return right_; }
  const SweptVolumeChecker & crossChecker() const { return *swept_; }
  const PhaseValidatorConfig & config() const { return config_; }
  const planning_scene::PlanningSceneConstPtr & scene() const { return scene_; }

  /// Joint values of a plan-order state in the spec's joint order.
  Status orderedQ(const ArmSpec & arm, const std::vector<std::string> & plan_joint_names,
                  const std::vector<double> & q, std::vector<double> & out) const;

private:
  PhaseValidator() = default;
  Status armPath(const ArmPhase & arm, SampledPath & out) const;
  Status parkedPath(const ArmSpec & spec, const ArmPhase & arm, const std::vector<double> & q, SampledPath & out) const;
  TipFk tipFk(const ArmSpec & spec) const;

  planning_scene::PlanningSceneConstPtr scene_;
  PhaseValidatorConfig config_;
  ArmSpec left_, right_;
  std::unique_ptr<SweptVolumeChecker> swept_;
  std::unique_ptr<WorldChecker> world_left_, world_right_;
};

}  // namespace dual_cut
