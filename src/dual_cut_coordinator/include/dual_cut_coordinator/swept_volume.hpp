// Cross-arm swept-volume check and per-arm world check on the dual MoveIt model (WP5).
//
// Cross-arm check (SweptVolumeChecker::check), design section 6, step 5:
//   * Each cross-arm link gets ONE bounding sphere in its link frame (union of the spheres of all its
//     collision shapes, via the collision origin transforms). A sphere lower bound on the distance of
//     link a (arm A) and link b (arm B) at samples i, j is |ca(i) - cb(j)| - ra - rb.
//   * The checker compares EVERY sample of arm A's whole phase with EVERY sample of arm B's whole
//     phase (no time alignment is assumed: it is a swept-volume test, not a simultaneous one).
//     A per-sample whole-arm bounding sphere rejects most sample pairs before the per-link spheres.
//   * A sample pair (i, j) is a candidate if any link-pair lower bound is < padding. Candidates are
//     sorted by their lower bound and checked exactly: one dual RobotState (qA(i), qB(j)) and the
//     planning scene's FCL distanceSelf with an ACM in which ONLY cross-arm link pairs count.
//   * Certificate skipping (config.use_certificates, on by default): an exact distance d at (i, j)
//     with d >= padding certifies every other pair (i', j') with
//         d - |U_A[i'] - U_A[i]| - |U_B[j'] - U_B[j]| >= padding
//     where U_X is the cumulative sample-to-sample displacement bound of any point of any link of
//     arm X (translation of the link origin plus rotation angle times the link's extent). Such pairs
//     need no query. It is a rigorous Lipschitz bound on the sample states, it does not change the
//     verdict, and it is what makes dense phases tractable (an exact query costs ~3.5 ms on the real
//     model).
//   * Result: pass iff the minimum distance >= padding. FAIL CLOSED (safe() == false) if the number of
//     sample pairs or candidates exceeds its cap, if any considered link has unsupported collision
//     geometry (plane, octomap, empty or non-finite mesh), if the link padding of the collision env is
//     not zero, if a path does not match the arm's joints, or if a distance query returns nothing.
//
// Reported min distance (conservative): exact (FCL) distances have exact_distance_tolerance_m
// subtracted; if candidates exist and all were checked it is the minimum over them (the true minimum
// whenever it is below the threshold, because every pair below it is a candidate), otherwise a
// certified lower bound from the spheres / certificates (min_distance_exact == false).
// The verdict is on effective_min = min_distance - 0.5 * (max_step_a + max_step_b) >= padding; every
// threshold inside check() is therefore padding + that sampling gap.
//
// FCL accuracy (scan_results/dual_arm_stack_20261014/wp5b_fcl_accuracy/): against a brute-force
// triangle-triangle reference on ~9000 random and axis-aligned two-arm states of the real dual model
// (all-mesh hulls, and the earlier box/cylinder/mesh version) the largest overestimate was 0.07 mm
// (tessellation noise of the reference), no penetration was missed. BUT with exactly coplanar
// axis-aligned PRIMITIVE boxes FCL's GJK distance overestimates (0.316 m for a true 0.300 m; 0.412
// for 0.400 in the synthetic coplanar cell, test_swept_volume). That is measure-zero in random
// states, so the default tolerance (2 mm) does not cover it; re-run the measurement script whenever
// the collision geometry changes, and keep primitives out of the cross-arm links.
//
// Limits: only the SAMPLED states are checked (the sampling gap above accounts for the motion between
// them). Distances are between the unpadded collision geometry of the model. Attached bodies are
// ignored. A dual model whose collision primitives under-cover the real links (HANDOFF :83, up to
// 85-88 mm) gives no proof until the padding covers it.
#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include "dual_cut_coordinator/trajectory_tools.hpp"

namespace shapes
{
class Shape;
}

namespace dual_cut
{

struct Sphere
{
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  double radius = 0.0;
};

/// Bounding sphere of one collision shape in the shape's own frame. Fails (fail closed) for planes,
/// octomaps, unknown types, non-finite or empty meshes.
Status boundingSphereOfShape(const shapes::Shape & shape, Sphere & out);
/// Smallest sphere enclosing both spheres.
Sphere mergeSpheres(const Sphere & a, const Sphere & b);
/// Bounding sphere of all collision shapes of a link, in the link frame. has_geometry is false (and
/// the status ok) if the link has no collision shape at all; fails on unsupported geometry.
Status linkBoundingSphere(const moveit::core::LinkModel & link, Sphere & out, bool & has_geometry);

/// One arm of the dual model: name prefix of its links and its joints in controller order.
struct ArmSpec
{
  std::string side;                      // "left" / "right"
  std::string prefix;                    // "arm_left_"
  std::vector<std::string> joint_names;  // prefixed active joints, model order
};
/// Builds the spec of the arm whose links and joints start with armPrefix(side). Fails if the model
/// has no active joint with that prefix.
Status makeArmSpec(const moveit::core::RobotModel & model, const std::string & side, ArmSpec & out);

struct SweptVolumeConfig
{
  /// Pass iff effective_min = min_distance - 0.5 * (max_step_a + max_step_b) >= padding.
  double padding_m = 0.05;
  /// Subtracted from EVERY exact (FCL) distance, in the verdict and in the certificates. FCL's
  /// cross-arm distance was measured against a brute-force triangle reference on the real dual model
  /// (scan_results/dual_arm_stack_20261014/wp5b_fcl_accuracy/): max overestimate 0.07 mm over ~9000
  /// states (tessellation noise only; the installed all-mesh model is exact). 2 mm leaves a ~30x
  /// margin. It does NOT cover FCL's GJK failure on exactly coplanar axis-aligned PRIMITIVE boxes
  /// (up to 16 mm in the synthetic coplanar cell); never zero (create() rejects <= 0).
  double exact_distance_tolerance_m = 0.002;
  size_t max_sample_pairs = 100000000;     // |A| * |B| above this -> fail closed
  size_t max_candidate_pairs = 5000000;    // sample pairs surviving the sphere prefilter (memory cap)
  size_t max_exact_checks = 20000;         // exact distance queries; above this -> fail closed
  bool use_certificates = true;            // skip pairs certified by an earlier exact distance
  bool stop_at_first_violation = false;    // return as soon as an exact distance < padding is found
};

enum class SweptCode
{
  Ok,                  // min distance >= padding
  Violation,           // min distance < padding
  PairCapExceeded,     // too many sample pairs / candidates (fail closed)
  UnsupportedGeometry, // (reported by create(); kept for completeness)
  InvalidInput,        // path does not match the arm, empty path, non-finite values
  QueryFailed          // distance query returned nothing (fail closed)
};

struct SweptVolumeResult
{
  SweptCode code = SweptCode::InvalidInput;
  std::string message;
  double min_distance = std::numeric_limits<double>::quiet_NaN();
  bool min_distance_exact = false;
  std::string link_a;            // link of arm A at the minimum
  std::string link_b;            // link of arm B at the minimum
  size_t sample_index_a = 0;
  size_t sample_index_b = 0;
  size_t sample_pairs = 0;       // |A| * |B|
  size_t candidate_pairs = 0;    // sample pairs that passed the sphere prefilter
  size_t exact_checks = 0;       // exact distance queries made
  size_t certified_pairs = 0;    // candidates skipped because an exact distance certified them
  double max_step_a_m = 0.0;     // largest sample-to-sample displacement bound of arm A / B
  double max_step_b_m = 0.0;
  double sampling_gap_m = 0.0;   // 0.5 * (max_step_a_m + max_step_b_m)
  /// min_distance - sampling_gap_m: what the verdict is based on (pass iff >= padding).
  double effective_min_distance = std::numeric_limits<double>::quiet_NaN();
  /// min_distance is already CONSERVATIVE: exact (FCL) values have exact_distance_tolerance_m
  /// subtracted. min_distance_exact therefore means "from an exact query", not "true distance".

  /// True only for a definite pass. Anything else (violation, cap, errors) is not safe.
  bool safe() const { return code == SweptCode::Ok; }
};

/// The sampling-gap rule used by check(): the continuous swept volume between samples can come
/// closer than any sampled pair by up to 0.5 * (max_step_a + max_step_b).
double effectiveMinDistance(double min_distance, double max_step_a_m, double max_step_b_m);

class SweptVolumeChecker
{
public:
  /// Fails (fail closed) on unsupported geometry, on non-zero link padding/scale in the collision env,
  /// or when the two arms share no cross pair with geometry.
  static Status create(
    const planning_scene::PlanningSceneConstPtr & scene, const ArmSpec & arm_a,
    const ArmSpec & arm_b, const SweptVolumeConfig & config,
    std::unique_ptr<SweptVolumeChecker> & out);

  SweptVolumeResult check(const SampledPath & path_a, const SampledPath & path_b) const;

  /// Exact cross-arm-only distance of one dual configuration (arm joints in the given spec order),
  /// e.g. for the 20 Hz monitor. RAW FCL value: callers subtract config().exact_distance_tolerance_m.
  /// Negative or zero means contact/penetration. Other arm joints in the state are left at the model
  /// default.
  Status exactDistance(
    const std::vector<double> & q_a, const std::vector<double> & q_b, double & distance,
    std::string & link_a, std::string & link_b) const;

  /// Smallest per-link-pair sphere lower bound of one configuration pair (never larger than the exact
  /// distance of the same pair of links). Test hook for the prefilter-soundness property.
  Status sphereLowerBound(
    const std::vector<double> & q_a, const std::vector<double> & q_b, double & lower_bound,
    std::string & link_a, std::string & link_b) const;

  /// Per link-pair bound and exact value for one configuration pair, same link order for both
  /// vectors: entry k belongs to (links_a()[ia], links_b()[ib]). Exact values come from a SINGLE
  /// type distance request. Test hook.
  struct PairReport
  {
    std::string link_a, link_b;
    double sphere_lower_bound;
    double exact_distance;   // NaN if the query returned no entry for this pair
  };
  Status pairReport(
    const std::vector<double> & q_a, const std::vector<double> & q_b,
    std::vector<PairReport> & out) const;

  const std::vector<std::string> & linksA() const { return names_a_; }
  const std::vector<std::string> & linksB() const { return names_b_; }
  const ArmSpec & armA() const { return arm_a_; }
  const ArmSpec & armB() const { return arm_b_; }
  const SweptVolumeConfig & config() const { return config_; }

private:
  struct LinkSphere
  {
    const moveit::core::LinkModel * link = nullptr;
    Sphere sphere;  // link frame
  };
  struct Prepared  // sphere centres in the world frame for every sample of one path
  {
    std::vector<Eigen::Vector3d> centers;  // [sample * L + link]
    std::vector<Eigen::Vector3d> arm_center;
    std::vector<double> arm_radius;
    std::vector<double> cum_disp;  // U: cumulative displacement bound, U[0] = 0
    double max_step = 0.0;
  };

  SweptVolumeChecker() = default;
  Status mapJoints(const SampledPath & path, const ArmSpec & arm, std::vector<int> & var_index) const;
  Status mapJointVector(const ArmSpec & arm, std::vector<int> & var_index) const;
  Status prepare(
    const SampledPath & path, const std::vector<int> & var_index,
    const std::vector<LinkSphere> & spheres, Prepared & out) const;
  void centersFor(
    moveit::core::RobotState & state, const std::vector<LinkSphere> & spheres,
    std::vector<Eigen::Vector3d> & centers) const;
  Status exactAt(
    moveit::core::RobotState & state, double & distance, std::string & la, std::string & lb) const;

  planning_scene::PlanningSceneConstPtr scene_;
  moveit::core::RobotModelConstPtr model_;
  ArmSpec arm_a_, arm_b_;
  SweptVolumeConfig config_;
  std::vector<LinkSphere> spheres_a_, spheres_b_;
  std::vector<std::string> names_a_, names_b_;
  std::vector<int> vars_a_, vars_b_;  // variable indices in arm spec order
  collision_detection::AllowedCollisionMatrix cross_only_acm_;
};

// ---- world check -----------------------------------------------------------------------------

struct WorldCheckConfig
{
  /// Scene objects the tool may touch while a sample is contact_allowed.
  std::vector<std::string> contact_object_ids{"detected_carton"};
  /// Links of the checked arm that may touch them (the tool links). Must be non-empty if the path
  /// has any contact_allowed sample, and every name must be a link of the arm.
  std::vector<std::string> contact_link_names;
  size_t max_samples = 200000;  // fail closed above this many samples
  /// Contacts whose penetration depth is at most this are ignored (default 0 = any contact is a collision).
  /// For model-coverage mismatches only: e.g. the dual model's blade box is not in the per-arm planner's model,
  /// so approach poses planned for the TCP can graze the carton by a fraction of a millimetre.
  double penetration_tolerance_m = 0.0;
};

enum class WorldCode
{
  Ok,
  Collision,
  InvalidInput,
  TooManySamples,
  QueryFailed
};

struct WorldCheckResult
{
  WorldCode code = WorldCode::InvalidInput;
  std::string message;
  size_t sample_index = 0;      // first colliding sample
  double time_s = 0.0;
  bool sample_contact_allowed = false;
  size_t segment_index = 0;     // segment of the path the sample belongs to
  double depth_m = 0.0;         // penetration depth of the reported contact
  Eigen::Vector3d contact_pos = Eigen::Vector3d::Zero();     // world frame
  Eigen::Vector3d contact_normal = Eigen::Vector3d::Zero();
  std::string link;             // colliding pair, as reported by the collision env
  std::string object;
  size_t samples_checked = 0;
  /// With a penetration tolerance: the deepest contact that was ignored (always reported, also on success).
  double max_tolerated_depth_m = 0.0;
  size_t tolerated_samples = 0;
  size_t max_tolerated_sample = 0;

  bool safe() const { return code == WorldCode::Ok; }
};

/// Checks the sampled states of one arm on the dual scene: the arm's own links against the scene
/// objects (pole, head, carton ...), against its own links and against the non-arm robot links such
/// as the table, using the scene's ACM. Padding is 0: pure FCL collision. All pairs involving the
/// OTHER arm are allowed (it is held at the given configuration; the cross-arm check covers it), and
/// so are pairs of a non-arm robot link (the table) with a world object (a pole standing on it).
/// During contact_allowed samples the configured tool links may touch the configured objects.
class WorldChecker
{
public:
  static Status create(
    const planning_scene::PlanningSceneConstPtr & scene, const ArmSpec & arm,
    const ArmSpec & other_arm, const WorldCheckConfig & config,
    std::unique_ptr<WorldChecker> & out);

  /// other_arm_q is in other_arm.joint_names order (the configuration the other arm is held at).
  WorldCheckResult check(const SampledPath & path, const std::vector<double> & other_arm_q) const;

private:
  WorldChecker() = default;

  planning_scene::PlanningSceneConstPtr scene_;
  moveit::core::RobotModelConstPtr model_;
  ArmSpec arm_, other_;
  WorldCheckConfig config_;
  std::vector<std::string> arm_links_, other_links_;
};

}  // namespace dual_cut
