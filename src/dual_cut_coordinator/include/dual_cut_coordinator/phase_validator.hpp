// Validation of one phase (design section 6, step 5): world check per arm, tip-on-line gate, cross-arm
// swept volume, and the per-phase serial fallback ("one arm after the other"). Library class with
// plain gtests on the synthetic cell; the node only supplies the scene and the exported cut lines.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dual_cut_coordinator/gates.hpp"
#include "dual_cut_coordinator/phase_plan.hpp"
#include "dual_cut_coordinator/swept_volume.hpp"
#include "dual_cut_coordinator/trajectory_tools.hpp"

namespace dual_cut
{

struct PhaseValidatorConfig
{
  SweptVolumeConfig swept;                         // padding, exact tolerance, caps
  std::vector<std::string> contact_object_ids{"detected_carton"};
  std::vector<std::string> contact_links_left;     // tool links allowed to touch the carton in CUT samples
  std::vector<std::string> contact_links_right;
  size_t world_max_samples = 200000;
  double world_penetration_tolerance_m = 0.0;     // see WorldCheckConfig; 0 = strict
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
