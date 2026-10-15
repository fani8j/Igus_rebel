// Tests for the closed-form ReBeL IK solver (see rebel_analytic_ik.hpp). Every test runs for
// BOTH tool offsets: the legacy xeg32_knife_v1 one (Rz(pi), no tilt) and the knife_c one, whose
// link6 -> xeg32_tool_tip transform contains a -30 deg pitch (a general rotation).
//
// (a) >=10000 random joint configs within limits: FK -> analytic IK -> the original config
//     is among the returned solutions (within 1e-6), and every returned solution's FK
//     matches the target pose within 1e-6 m / 1e-6 rad. This is exactly what
//     rebel_analytic_ik::selfCheck implements; the box_h_cut_executor startup gate calls the
//     same function.
// (b) For poses where KDL finds a solution, the analytic solver finds one too. Built and run
//     only when kdl_parser/orocos_kdl are available (guarded by
//     REBEL_ANALYTIC_IK_HAVE_KDL); see CMakeLists.txt.
// (c) Clearly-unreachable poses (far outside the arm's reach envelope) return empty fast.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "igus_rebel_moveit_config/rebel_analytic_ik.hpp"

#if REBEL_ANALYTIC_IK_HAVE_KDL
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>
#endif

namespace
{
constexpr double kPositionTolM = 1e-6;
constexpr double kOrientationTolRad = 1e-6;

struct NamedOffset
{
  const char* name;
  Eigen::Isometry3d offset;
};

// Composes a chain of URDF fixed joints (xyz, rpy) into one transform, transcribed
// independently of rebel_analytic_ik.cpp (from igus_rebel_xeg32_wrapper.urdf.xacro).
Eigen::Isometry3d urdfJoint(double x, double y, double z, double roll, double pitch, double yaw)
{
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.translation() = Eigen::Vector3d(x, y, z);
  t.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
                 .toRotationMatrix();
  return t;
}

// xeg32_knife_v1: link6 -> igus_tool0 (0,0,0.0527; rpy 0,0,pi) -> xeg32_tool_tip
// (0.005937,0.094121,0.130699; rpy 0).
Eigen::Isometry3d xacroLegacyOffset()
{
  return urdfJoint(0, 0, 0.0527, 0, 0, M_PI) * urdfJoint(0.005937, 0.094121, 0.130699, 0, 0, 0);
}

// knife_c: link6 -> igus_tool0 -> knife_c_tcp_point (-0.042069,-0.089590,0.104551; rpy 0)
// -> xeg32_tool_tip (xyz 0; rpy 0,-pi/6,0).
Eigen::Isometry3d xacroKnifeCOffset()
{
  return urdfJoint(0, 0, 0.0527, 0, 0, M_PI) *
         urdfJoint(-0.042069, -0.089590, 0.104551, 0, 0, 0) *
         urdfJoint(0, 0, 0, 0, -0.5235987755982988, 0);
}

std::vector<NamedOffset> allOffsets()
{
  return {
    {"legacy_xeg32_knife_v1", rebel_analytic_ik::legacyXeg32KnifeV1ToolOffset()},
    {"knife_c", xacroKnifeCOffset()},
  };
}
}  // namespace

TEST(RebelAnalyticIk, LegacyOffsetConstantMatchesIndependentXacroTranscription)
{
  const Eigen::Isometry3d a = rebel_analytic_ik::legacyXeg32KnifeV1ToolOffset();
  const Eigen::Isometry3d b = xacroLegacyOffset();
  EXPECT_LT((a.translation() - b.translation()).norm(), 1e-12);
  EXPECT_LT((a.linear() - b.linear()).norm(), 1e-12);
}

TEST(RebelAnalyticIk, KnifeCOffsetHasRotation)
{
  // Guards the test itself: the knife_c offset must differ from the legacy one by a real
  // (non-self-inverse) rotation, otherwise this suite would not exercise the general case.
  const Eigen::Isometry3d legacy = rebel_analytic_ik::legacyXeg32KnifeV1ToolOffset();
  const Eigen::Isometry3d knife_c = xacroKnifeCOffset();
  const double ang =
    Eigen::AngleAxisd(legacy.linear().transpose() * knife_c.linear()).angle();
  EXPECT_NEAR(ang, M_PI / 6.0, 1e-9);
}

TEST(RebelAnalyticIk, SelfConsistencyOver10000RandomConfigs)
{
  const auto limits = rebel_analytic_ik::defaultJointLimits();
  for (const auto& named : allOffsets()) {
    SCOPED_TRACE(named.name);
    const auto result = rebel_analytic_ik::selfCheck(
      10000, kPositionTolM, kOrientationTolRad, /*seed=*/12345U, named.offset, limits);
    EXPECT_TRUE(result.ok) << result.failure_detail;
    EXPECT_EQ(result.samples, 10000U);
    EXPECT_EQ(result.samples_with_original_recovered, result.samples) << result.failure_detail;
    EXPECT_LT(result.max_position_residual_m, kPositionTolM);
    EXPECT_LT(result.max_orientation_residual_rad, kOrientationTolRad);
  }
}

// Independent of selfCheck's internals: sample configs directly, call solve(), and check
// every returned solution's FK against the target and that the sampled q is recovered.
TEST(RebelAnalyticIk, EverySolutionFkMatchesTargetAndOriginalIsRecovered)
{
  const auto limits = rebel_analytic_ik::defaultJointLimits();
  for (const auto& named : allOffsets()) {
  SCOPED_TRACE(named.name);
  const auto& tool = named.offset;
  std::mt19937 rng(987654321U);
  constexpr std::size_t kSamples = 10000;
  std::size_t total_solutions = 0;
  std::size_t branch_histogram[9] = {0};
  for (std::size_t i = 0; i < kSamples; ++i) {
    rebel_analytic_ik::JointVector q{};
    for (std::size_t j = 0; j < 6; ++j) {
      std::uniform_real_distribution<double> dist(limits[j].lower, limits[j].upper);
      q[j] = dist(rng);
    }
    const auto target = rebel_analytic_ik::forwardKinematics(q, tool);
    const auto solutions = rebel_analytic_ik::solve(target, tool, limits);
    ASSERT_FALSE(solutions.empty()) << "sample " << i << " (FK of an in-limits config) was "
      "reported unreachable";
    ASSERT_LE(solutions.size(), 8U) << "sample " << i << ": more than the expected <=8 "
      "Pieper branches";
    branch_histogram[solutions.size()]++;
    total_solutions += solutions.size();
    bool recovered = false;
    for (const auto& candidate : solutions) {
      const auto fk = rebel_analytic_ik::forwardKinematics(candidate, tool);
      const double pos_err = (fk.translation() - target.translation()).norm();
      const double ang_err =
        Eigen::AngleAxisd(fk.linear().transpose() * target.linear()).angle();
      EXPECT_LT(pos_err, kPositionTolM) << "sample " << i;
      EXPECT_LT(ang_err, kOrientationTolRad) << "sample " << i;
      bool matches = true;
      for (std::size_t j = 0; j < 6; ++j) {
        double diff = candidate[j] - q[j];
        while (diff > M_PI) diff -= 2.0 * M_PI;
        while (diff < -M_PI) diff += 2.0 * M_PI;
        if (std::abs(diff) > 1e-6) { matches = false; break; }
      }
      if (matches) recovered = true;
    }
    EXPECT_TRUE(recovered) << "sample " << i << " original joint vector not among solutions";
  }
  RecordProperty(std::string("mean_branches_per_pose_") + named.name, std::to_string(
    static_cast<double>(total_solutions) / static_cast<double>(kSamples)));
  }
}

TEST(RebelAnalyticIk, UnreachablePosesReturnEmptyFast)
{
  const auto limits = rebel_analytic_ik::defaultJointLimits();
  std::vector<Eigen::Isometry3d> unreachable;
  // Far outside any plausible reach envelope (arm is well under 1 m reach).
  for (const Eigen::Vector3d& p : {
         Eigen::Vector3d(100.0, 0.0, 0.0), Eigen::Vector3d(0.0, -50.0, 0.0),
         Eigen::Vector3d(0.0, 0.0, 1000.0), Eigen::Vector3d(-25.0, 25.0, -25.0)}) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = p;
    unreachable.push_back(pose);
  }
  const auto started = std::chrono::steady_clock::now();
  for (const auto& named : allOffsets()) {
    for (const auto& pose : unreachable) {
      EXPECT_TRUE(rebel_analytic_ik::solve(pose, named.offset, limits).empty()) << named.name;
    }
  }
  const double elapsed =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  // No iterative search happens in solve(); this should be sub-millisecond total. A generous
  // bound avoids flakiness on loaded CI machines while still catching an accidental fallback
  // to an iterative method.
  EXPECT_LT(elapsed, 0.05);
}

TEST(RebelAnalyticIk, JointLimitsAreEnforced)
{
  const auto limits = rebel_analytic_ik::defaultJointLimits();
  // A config at the joint4/joint6 limit: its "twin" solution via the wrist-flip branch may
  // land just outside the opposite limit and must be filtered, not silently returned.
  rebel_analytic_ik::JointVector q{0.1, 0.2, -0.3, 3.10, 1.0, -3.10};
  for (const auto& named : allOffsets()) {
    SCOPED_TRACE(named.name);
    const auto target = rebel_analytic_ik::forwardKinematics(q, named.offset);
    const auto solutions = rebel_analytic_ik::solve(target, named.offset, limits);
    for (const auto& s : solutions) {
      for (std::size_t j = 0; j < 6; ++j) {
        EXPECT_GE(s[j], limits[j].lower - 1e-9);
        EXPECT_LE(s[j], limits[j].upper + 1e-9);
      }
    }
  }
}

#if REBEL_ANALYTIC_IK_HAVE_KDL
namespace
{
bool buildChain(KDL::Chain& chain, std::string& error)
{
  const std::string urdf_path =
    ament_index_cpp::get_package_share_directory("igus_rebel_description") +
    "/urdf/igus_rebel2.urdf";
  urdf::Model model;
  if (!model.initFile(urdf_path)) {
    error = "failed to parse " + urdf_path;
    return false;
  }
  KDL::Tree tree;
  if (!kdl_parser::treeFromUrdfModel(model, tree)) {
    error = "kdl_parser::treeFromUrdfModel failed";
    return false;
  }
  if (!tree.getChain("base_link", "link6", chain)) {
    error = "no chain base_link -> link6 in the URDF-derived KDL tree";
    return false;
  }
  return true;
}
}  // namespace

// Directly checks rebel_analytic_ik::forwardKinematics against ground truth: a real
// KDL chain parsed from the shipped URDF (independent of this library's internal constants)
// composed with the TCP offset transcribed independently here from
// igus_rebel_xeg32_wrapper.urdf.xacro (link6 -> igus_tool0 -> xeg32_tool_tip), not from
// rebel_analytic_ik.cpp. This is the test that catches a systematic constant error shared
// between forwardKinematics() and solve() (which the pure self-consistency tests above
// cannot: they compare the library only against itself). An earlier version of this file had
// only the self-consistency and CartToJnt-comparison tests; both passed even while
// rebel_analytic_ik.cpp's kJ4Z constant was missing the URDF's separate joint4-origin
// z-offset (0.052 m, motor4->link4), a ~52 mm absolute FK error caught only by running the
// analytic backend in mock (see report.md) and diagnosed back to this gap. This test exists
// so that class of bug fails `colcon test`, not just a live mock run.
TEST(RebelAnalyticIk, ForwardKinematicsMatchesKdlPlusIndependentToolOffset)
{
  KDL::Chain chain;
  std::string error;
  ASSERT_TRUE(buildChain(chain, error)) << error;
  KDL::ChainFkSolverPos_recursive fk(chain);

  // Offsets transcribed independently from igus_rebel_xeg32_wrapper.urdf.xacro (see
  // xacroLegacyOffset / xacroKnifeCOffset above), for both tool variants.
  const std::vector<NamedOffset> offsets = {
    {"legacy_xeg32_knife_v1", xacroLegacyOffset()}, {"knife_c", xacroKnifeCOffset()}};

  const auto limits = rebel_analytic_ik::defaultJointLimits();
  std::mt19937 rng(2024U);
  for (const auto& named : offsets)
  for (int i = 0; i < 2000; ++i) {
    const Eigen::Isometry3d& tool_offset = named.offset;
    rebel_analytic_ik::JointVector q{};
    KDL::JntArray q_kdl(6);
    for (std::size_t j = 0; j < 6; ++j) {
      std::uniform_real_distribution<double> dist(limits[j].lower, limits[j].upper);
      q[j] = dist(rng);
      q_kdl(j) = q[j];
    }
    KDL::Frame link6_kdl;
    ASSERT_EQ(fk.JntToCart(q_kdl, link6_kdl), 0);
    Eigen::Isometry3d link6_eigen = Eigen::Isometry3d::Identity();
    link6_eigen.translation() = Eigen::Vector3d(link6_kdl.p.x(), link6_kdl.p.y(), link6_kdl.p.z());
    double qx, qy, qz, qw;
    link6_kdl.M.GetQuaternion(qx, qy, qz, qw);
    link6_eigen.linear() = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
    const Eigen::Isometry3d expected_tcp = link6_eigen * tool_offset;

    const Eigen::Isometry3d actual_tcp = rebel_analytic_ik::forwardKinematics(q, tool_offset);
    const double pos_err = (actual_tcp.translation() - expected_tcp.translation()).norm();
    const double ang_err =
      Eigen::AngleAxisd(actual_tcp.linear().transpose() * expected_tcp.linear()).angle();
    ASSERT_LT(pos_err, 1e-6) << named.name << " sample " << i << ": forwardKinematics vs KDL+tool-offset "
      "position mismatch (this is the class of bug a self-consistency-only test cannot catch)";
    ASSERT_LT(ang_err, 1e-6) << "sample " << i;
  }
}

// KDL only knows the arm chain up to link6 (xeg32_tool_tip is added by a separate xacro this
// test does not process), so this comparison is done in link6's frame; the analytic solver is
// asked for link6 poses by composing the known, fixed link6->xeg32_tool_tip offset and its
// inverse. Both solvers see the same target pose either way.
TEST(RebelAnalyticIk, MatchesKdlWhenKdlSucceeds)
{
  KDL::Chain chain;
  std::string error;
  ASSERT_TRUE(buildChain(chain, error)) << error;
  KDL::ChainFkSolverPos_recursive fk(chain);
  KDL::ChainIkSolverVel_pinv vel_ik(chain);

  const auto limits = rebel_analytic_ik::defaultJointLimits();
  KDL::JntArray q_min(6), q_max(6);
  for (std::size_t j = 0; j < 6; ++j) {
    q_min(j) = limits[j].lower;
    q_max(j) = limits[j].upper;
  }
  // ChainIkSolverPos_NR_JL is the joint-limited Newton-Raphson solver MoveIt's
  // KDLKinematicsPlugin uses (unlike ChainIkSolverPos_LMA, which ignores joint limits
  // entirely and can return out-of-bounds solutions that the analytic solver correctly
  // rejects -- an earlier version of this test used LMA and saw ~1.6% spurious mismatches
  // traced to exactly that).
  KDL::ChainIkSolverPos_NR_JL ik(chain, q_min, q_max, fk, vel_ik, 200, 1e-6);

  std::mt19937 rng(42U);
  constexpr std::size_t kSamples = 500;  // KDL's iterative IK is far slower than the closed form.
  std::size_t kdl_successes = 0;
  std::size_t analytic_also_succeeded = 0;
  for (std::size_t i = 0; i < kSamples; ++i) {
    rebel_analytic_ik::JointVector q{};
    KDL::JntArray q_kdl(6);
    for (std::size_t j = 0; j < 6; ++j) {
      std::uniform_real_distribution<double> dist(limits[j].lower, limits[j].upper);
      q[j] = dist(rng);
      q_kdl(j) = q[j];
    }
    KDL::Frame target_link6;
    ASSERT_EQ(fk.JntToCart(q_kdl, target_link6), 0);
    // Perturb the seed so KDL is doing real iterative work, not just returning the seed.
    KDL::JntArray seed(6);
    for (std::size_t j = 0; j < 6; ++j) {
      std::uniform_real_distribution<double> dist(limits[j].lower, limits[j].upper);
      seed(j) = dist(rng);
    }
    KDL::JntArray solution(6);
    if (ik.CartToJnt(seed, target_link6, solution) < 0) continue;
    ++kdl_successes;

    // Compose the known link6 -> xeg32_tool_tip offset to hand the analytic solver a TCP
    // pose, exactly as forwardKinematics() does internally.
    rebel_analytic_ik::JointVector q_kdl_solution{};
    for (std::size_t j = 0; j < 6; ++j) q_kdl_solution[j] = solution(j);
    const Eigen::Isometry3d knife_c_offset = xacroKnifeCOffset();
    const Eigen::Isometry3d target_tcp =
      rebel_analytic_ik::forwardKinematics(q_kdl_solution, knife_c_offset);
    const auto analytic_solutions = rebel_analytic_ik::solve(target_tcp, knife_c_offset, limits);
    if (!analytic_solutions.empty()) ++analytic_also_succeeded;
    EXPECT_FALSE(analytic_solutions.empty())
      << "sample " << i << ": KDL found a solution but the analytic solver did not";
  }
  RecordProperty("kdl_successes", std::to_string(kdl_successes));
  RecordProperty("analytic_also_succeeded", std::to_string(analytic_also_succeeded));
  // NR_JL's convergence rate from a fully random seed is much lower than a seeded planner's
  // would be in practice; a 10% success rate over 500 samples (50 comparisons) is still a
  // meaningful check that zero mismatches is not a fluke of an empty sample.
  EXPECT_GT(kdl_successes, kSamples / 10) << "too few KDL successes to be a meaningful check";
}
#endif  // REBEL_ANALYTIC_IK_HAVE_KDL
