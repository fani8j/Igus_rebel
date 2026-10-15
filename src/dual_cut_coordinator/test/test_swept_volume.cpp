#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <geometric_shapes/shapes.h>

#include "dual_cut_coordinator/swept_volume.hpp"
#include "test_common.hpp"

using namespace dual_cut;
using dual_cut_test::pathOf;

namespace
{

// Synthetic cell (test/data/synthetic_dual.urdf): both arms are two yaw joints about world z at
// height 0.1. Left base at the origin, right base at y = 0.5 (and z = 0.013, see the URDF). link1 = 0.4 m box along +x, link2 =
// 0.2 m box along +x, all 0.04 m thick; base = 0.1 m cube; tool = 0.03 m sphere.
class SweptVolumeTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model = dual_cut_test::syntheticModel();
    ASSERT_NE(model, nullptr);
    scene = std::make_shared<planning_scene::PlanningScene>(model);
    ASSERT_TRUE(makeArmSpec(*model, "left", left));
    ASSERT_TRUE(makeArmSpec(*model, "right", right));
  }

  std::unique_ptr<SweptVolumeChecker> makeChecker(const SweptVolumeConfig & cfg = SweptVolumeConfig{})
  {
    std::unique_ptr<SweptVolumeChecker> c;
    const Status s = SweptVolumeChecker::create(scene, left, right, cfg, c);
    EXPECT_TRUE(s) << s.message;
    return c;
  }

  moveit::core::RobotModelPtr model;
  std::shared_ptr<planning_scene::PlanningScene> scene;
  ArmSpec left, right;
};

}  // namespace

TEST(BoundingSphere, PrimitivesAndFailClosed)
{
  Sphere s;
  ASSERT_TRUE(boundingSphereOfShape(shapes::Box(0.2, 0.4, 0.4), s));
  EXPECT_NEAR(s.radius, 0.3, 1e-12);
  ASSERT_TRUE(boundingSphereOfShape(shapes::Sphere(0.05), s));
  EXPECT_NEAR(s.radius, 0.05, 1e-12);
  ASSERT_TRUE(boundingSphereOfShape(shapes::Cylinder(0.03, 0.8), s));
  EXPECT_NEAR(s.radius, std::hypot(0.03, 0.4), 1e-12);
  ASSERT_TRUE(boundingSphereOfShape(shapes::Cone(0.03, 0.08), s));
  EXPECT_NEAR(s.radius, 0.05, 1e-12);

  // Unsupported geometry fails closed.
  EXPECT_FALSE(boundingSphereOfShape(shapes::Plane(0, 0, 1, 0), s));
  EXPECT_FALSE(boundingSphereOfShape(shapes::Mesh(0, 0), s));  // no vertices
  shapes::Mesh m(1, 0);
  m.vertices[0] = NAN;
  m.vertices[1] = 0.0;
  m.vertices[2] = 0.0;
  EXPECT_FALSE(boundingSphereOfShape(m, s));

  // Mesh: sphere around the AABB centre covers all vertices.
  shapes::Mesh m2(3, 0);
  const double v[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0};
  std::copy(v, v + 9, m2.vertices);
  ASSERT_TRUE(boundingSphereOfShape(m2, s));
  for (int i = 0; i < 3; ++i)
    EXPECT_LE((Eigen::Vector3d(v[3 * i], v[3 * i + 1], v[3 * i + 2]) - s.center).norm(), s.radius + 1e-12);
}

TEST(BoundingSphere, MergeContainsBoth)
{
  Sphere a{Eigen::Vector3d(0, 0, 0), 1.0};
  Sphere b{Eigen::Vector3d(3, 0, 0), 0.5};
  const Sphere m = mergeSpheres(a, b);
  EXPECT_LE((m.center - a.center).norm() + a.radius, m.radius + 1e-12);
  EXPECT_LE((m.center - b.center).norm() + b.radius, m.radius + 1e-12);
  EXPECT_NEAR(m.radius, 2.25, 1e-12);
  // one inside the other
  const Sphere c = mergeSpheres(a, Sphere{Eigen::Vector3d(0.1, 0, 0), 0.2});
  EXPECT_NEAR(c.radius, 1.0, 1e-12);
}

TEST_F(SweptVolumeTest, ArmSpecsAndCreateFailClosed)
{
  EXPECT_EQ(left.joint_names, (std::vector<std::string>{"arm_left_joint1", "arm_left_joint2"}));
  EXPECT_EQ(right.prefix, "arm_right_");
  ArmSpec bad;
  EXPECT_FALSE(makeArmSpec(*model, "middle", bad));

  auto c = makeChecker();
  ASSERT_NE(c, nullptr);
  // base, link1, link2, tool of each arm
  EXPECT_EQ(c->linksA().size(), 4u);
  EXPECT_EQ(c->linksB().size(), 4u);

  std::unique_ptr<SweptVolumeChecker> d;
  SweptVolumeConfig cfg;
  cfg.padding_m = -1.0;
  EXPECT_FALSE(SweptVolumeChecker::create(scene, left, right, cfg, d));
  EXPECT_FALSE(SweptVolumeChecker::create(scene, left, left, SweptVolumeConfig{}, d));
  EXPECT_FALSE(SweptVolumeChecker::create(nullptr, left, right, SweptVolumeConfig{}, d));
}

TEST_F(SweptVolumeTest, MinDistanceMatchesHandComputedCases)
{
  auto c = makeChecker();
  ASSERT_NE(c, nullptr);
  double d;
  std::string la, lb;

  // Both arms stretched along +x. Nearest: the two bases (y in [-0.05, 0.05] and [0.45, 0.55], x
  // and z ranges overlap): 0.45 - 0.05 = 0.40.
  ASSERT_TRUE(c->exactDistance({0, 0}, {0, 0}, d, la, lb));
  EXPECT_NEAR(d, 0.40, 1e-5);
  EXPECT_EQ(la, "arm_left_base_link");
  EXPECT_EQ(lb, "arm_right_base_link");

  // Left: joint1 = +90 deg (link1 along +y to y = 0.4), joint2 = -90 deg (link2 along +x at
  // y in [0.38, 0.42], x in [0, 0.2]). Nearest: link2 to the right base (x <= 0.05, y >= 0.45):
  // 0.45 - 0.42 = 0.03.
  ASSERT_TRUE(c->exactDistance({M_PI / 2, -M_PI / 2}, {0, 0}, d, la, lb));
  EXPECT_NEAR(d, 0.03, 1e-5);
  EXPECT_EQ(la, "arm_left_link2");
  EXPECT_EQ(lb, "arm_right_base_link");

  // Arm B and arm A order in the result is always (arm A link, arm B link).
  ASSERT_TRUE(c->exactDistance({0, 0}, {0, 0}, d, la, lb));
  EXPECT_EQ(la.rfind("arm_left_", 0), 0u);
  EXPECT_EQ(lb.rfind("arm_right_", 0), 0u);

  // Penetration: left link2 pointing +y into the right base -> distance <= 0.
  ASSERT_TRUE(c->exactDistance({M_PI / 2, 0}, {0, 0}, d, la, lb));
  EXPECT_LE(d, 0.0);

  // Wrong sizes / non-finite fail.
  EXPECT_FALSE(c->exactDistance({0}, {0, 0}, d, la, lb));
  EXPECT_FALSE(c->exactDistance({NAN, 0}, {0, 0}, d, la, lb));
}

TEST_F(SweptVolumeTest, SinglePairPathsAgainstThePadding)
{
  const std::vector<double> qa{M_PI / 2, -M_PI / 2};
  const std::vector<double> qb{0, 0};
  {
    SweptVolumeConfig cfg;
    cfg.padding_m = 0.05;
    auto c = makeChecker(cfg);
    const SweptVolumeResult r = c->check(pathOf(left, {qa}), pathOf(right, {qb}));
    EXPECT_EQ(r.code, SweptCode::Violation) << r.message;
    EXPECT_FALSE(r.safe());
    EXPECT_NEAR(r.min_distance, 0.03 - cfg.exact_distance_tolerance_m, 1e-5);  // tolerance subtracted
    EXPECT_NEAR(r.effective_min_distance, r.min_distance, 1e-12);              // one sample: no gap
    EXPECT_TRUE(r.min_distance_exact);
    EXPECT_EQ(r.link_a, "arm_left_link2");
    EXPECT_EQ(r.link_b, "arm_right_base_link");
    EXPECT_EQ(r.sample_index_a, 0u);
    EXPECT_EQ(r.sample_index_b, 0u);
    EXPECT_EQ(r.exact_checks, 1u);
  }
  {
    SweptVolumeConfig cfg;
    cfg.padding_m = 0.02;
    auto c = makeChecker(cfg);
    const SweptVolumeResult r = c->check(pathOf(left, {qa}), pathOf(right, {qb}));
    EXPECT_EQ(r.code, SweptCode::Ok) << r.message;
    EXPECT_TRUE(r.safe());
    EXPECT_GE(r.min_distance, 0.02);
    EXPECT_LE(r.min_distance, 0.03 - cfg.exact_distance_tolerance_m + 1e-5);  // never above truth - tol
  }
  {
    // Far apart: no candidates, no exact queries, the reported value is a lower bound >= padding.
    SweptVolumeConfig cfg;
    cfg.padding_m = 0.05;
    auto c = makeChecker(cfg);
    const SweptVolumeResult r = c->check(pathOf(left, {{0, 0}}), pathOf(right, {{0, 0}}));
    EXPECT_TRUE(r.safe());
    EXPECT_EQ(r.exact_checks, 0u);
    EXPECT_FALSE(r.min_distance_exact);
    EXPECT_GE(r.min_distance, 0.05);
    EXPECT_LE(r.min_distance, 0.40 + 1e-5);
  }
}

TEST_F(SweptVolumeTest, SpherePrefilterIsSoundAgainstBruteForce)
{
  auto c = makeChecker();
  ASSERT_NE(c, nullptr);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> U(-M_PI, M_PI);
  size_t pairs = 0, near_pairs = 0;
  for (int it = 0; it < 150; ++it)
  {
    const std::vector<double> qa{U(rng), U(rng)};
    const std::vector<double> qb{U(rng), U(rng)};
    std::vector<SweptVolumeChecker::PairReport> rep;
    ASSERT_TRUE(c->pairReport(qa, qb, rep));
    for (const auto & r : rep)
    {
      if (std::isnan(r.exact_distance)) continue;  // pair not evaluated by FCL (distance skipped)
      ++pairs;
      // The sphere bound may never exceed the exact distance of the same link pair.
      const double slack = r.exact_distance < 0.0 ? 1e-3 : 1e-6;
      EXPECT_LE(r.sphere_lower_bound, r.exact_distance + slack)
        << r.link_a << " / " << r.link_b << " it=" << it;
      if (r.exact_distance < 0.05) ++near_pairs;
    }
    // Global bound vs global exact distance.
    double lb, d;
    std::string a1, b1, a2, b2;
    ASSERT_TRUE(c->sphereLowerBound(qa, qb, lb, a1, b1));
    ASSERT_TRUE(c->exactDistance(qa, qb, d, a2, b2));
    EXPECT_LE(lb, d + (d < 0 ? 1e-3 : 1e-6));
  }
  EXPECT_GT(pairs, 1000u);
  EXPECT_GT(near_pairs, 10u);  // the random sweep does produce close and colliding pairs
}

TEST_F(SweptVolumeTest, CheckAgreesWithBruteForceOnRandomPaths)
{
  std::mt19937 rng(99);
  std::uniform_real_distribution<double> U(-M_PI, M_PI);
  std::uniform_real_distribution<double> W(-0.04, 0.04);
  std::uniform_real_distribution<double> P(0.01, 0.30);
  std::uniform_int_distribution<int> N(1, 7);
  int violations = 0, oks = 0;
  for (int it = 0; it < 60; ++it)
  {
    SweptVolumeConfig cfg;
    cfg.padding_m = P(rng);
    auto c = makeChecker(cfg);
    ASSERT_NE(c, nullptr);
    const double tol = cfg.exact_distance_tolerance_m;
    // short random walks (small steps) so that the sampling gap is small and outcomes are mixed
    auto walk = [&]()
    {
      std::vector<std::vector<double>> q;
      std::vector<double> cur{U(rng), U(rng)};
      for (int i = N(rng); i > 0; --i)
      {
        q.push_back(cur);
        cur[0] += W(rng);
        cur[1] += W(rng);
      }
      return q;
    };
    const auto qa = walk();
    const auto qb = walk();

    double best = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < qa.size(); ++i)
      for (size_t j = 0; j < qb.size(); ++j)
      {
        double d;
        std::string x, y;
        ASSERT_TRUE(c->exactDistance(qa[i], qb[j], d, x, y));
        best = std::min(best, d - tol);  // conservative sampled minimum
      }
    const SweptVolumeResult r = c->check(pathOf(left, qa), pathOf(right, qb));
    ASSERT_TRUE(r.code == SweptCode::Ok || r.code == SweptCode::Violation) << r.message;
    const double gap = r.sampling_gap_m;
    EXPECT_NEAR(gap, 0.5 * (r.max_step_a_m + r.max_step_b_m), 1e-12);
    EXPECT_EQ(r.safe(), best - gap >= cfg.padding_m)
      << "brute sampled min " << best << " gap " << gap << " padding " << cfg.padding_m;
    if (best - gap < cfg.padding_m)
    {
      ++violations;
      EXPECT_NEAR(r.min_distance, best, 1e-9);
      EXPECT_NEAR(r.effective_min_distance, best - gap, 1e-9);
      EXPECT_TRUE(r.min_distance_exact);
      // The reported sample pair attains the minimum (ties are legitimate: a pair such as left
      // base / right link2 does not depend on the other arm's joints).
      ASSERT_LT(r.sample_index_a, qa.size());
      ASSERT_LT(r.sample_index_b, qb.size());
      double at;
      std::string x, y;
      ASSERT_TRUE(c->exactDistance(qa[r.sample_index_a], qb[r.sample_index_b], at, x, y));
      EXPECT_NEAR(at - tol, best, 1e-9);
      EXPECT_EQ(x, r.link_a);
      EXPECT_EQ(y, r.link_b);
    }
    else
    {
      ++oks;
      EXPECT_GE(r.effective_min_distance, cfg.padding_m);
      EXPECT_LE(r.min_distance, best + 1e-6);  // never claims more than the truth
    }
  }
  EXPECT_GT(violations, 3);
  EXPECT_GT(oks, 3);
}

TEST_F(SweptVolumeTest, FailClosedPaths)
{
  const std::vector<double> qa{M_PI / 2, -M_PI / 2};
  const std::vector<double> qb{0, 0};
  {
    // More candidate sample pairs than the exact-check cap.
    SweptVolumeConfig cfg;
    cfg.padding_m = 0.05;
    cfg.max_exact_checks = 1;
    auto c = makeChecker(cfg);
    const SweptVolumeResult r =
      c->check(pathOf(left, {qa, qa, qa}), pathOf(right, {qb, qb, qb}));
    EXPECT_EQ(r.code, SweptCode::PairCapExceeded) << r.message;
    EXPECT_FALSE(r.safe());
  }
  {
    SweptVolumeConfig cfg;
    cfg.padding_m = 0.05;
    cfg.max_candidate_pairs = 2;
    auto c = makeChecker(cfg);
    const SweptVolumeResult r =
      c->check(pathOf(left, {qa, qa, qa}), pathOf(right, {qb, qb, qb}));
    EXPECT_EQ(r.code, SweptCode::PairCapExceeded) << r.message;
    EXPECT_FALSE(r.safe());
  }
  {
    SweptVolumeConfig cfg;
    cfg.max_sample_pairs = 5;
    auto c = makeChecker(cfg);
    const SweptVolumeResult r = c->check(pathOf(left, {qa, qa}), pathOf(right, {qb, qb, qb}));
    EXPECT_EQ(r.code, SweptCode::PairCapExceeded);
    EXPECT_FALSE(r.safe());
  }
  {
    auto c = makeChecker();
    // empty path
    EXPECT_EQ(c->check(SampledPath{}, pathOf(right, {qb})).code, SweptCode::InvalidInput);
    // path joints of the wrong arm
    EXPECT_EQ(c->check(pathOf(right, {qb}), pathOf(right, {qb})).code, SweptCode::InvalidInput);
    // wrong joint count in a sample
    SampledPath bad = pathOf(left, {qa});
    bad.samples[0].q.push_back(0.0);
    EXPECT_EQ(c->check(bad, pathOf(right, {qb})).code, SweptCode::InvalidInput);
    // non-finite sample
    SampledPath nan = pathOf(left, {qa});
    nan.samples[0].q[0] = NAN;
    const SweptVolumeResult r = c->check(nan, pathOf(right, {qb}));
    EXPECT_EQ(r.code, SweptCode::InvalidInput);
    EXPECT_FALSE(r.safe());
    // joint names permuted is fine (mapped by name), a duplicated name is not
    SampledPath dup = pathOf(left, {qa});
    dup.joint_names = {left.joint_names[0], left.joint_names[0]};
    EXPECT_EQ(c->check(dup, pathOf(right, {qb})).code, SweptCode::InvalidInput);
    SampledPath perm = pathOf(left, {{-M_PI / 2, M_PI / 2}});  // (q2, q1) = (-90, +90) deg
    perm.joint_names = {left.joint_names[1], left.joint_names[0]};
    const SweptVolumeResult rp = c->check(perm, pathOf(right, {qb}));
    EXPECT_TRUE(rp.code == SweptCode::Ok || rp.code == SweptCode::Violation);
    EXPECT_NEAR(rp.min_distance, 0.03 - SweptVolumeConfig{}.exact_distance_tolerance_m, 1e-5);
  }
}

TEST_F(SweptVolumeTest, CertificatesSkipQueriesWithoutChangingTheVerdict)
{
  // Left sweeps joint1 densely while joint2 stays folded back, right arm static at the hand case.
  std::vector<std::vector<double>> qa, qb;
  for (int i = 0; i <= 150; ++i) qa.push_back({0.01 * i, -M_PI / 2});
  for (int i = 0; i < 40; ++i) qb.push_back({0.0, 0.0});
  for (double padding : {0.01, 0.05, 0.10, 0.30})
  {
    SweptVolumeConfig on;
    on.padding_m = padding;
    SweptVolumeConfig off = on;
    off.use_certificates = false;
    const SweptVolumeResult a = makeChecker(on)->check(pathOf(left, qa), pathOf(right, qb));
    const SweptVolumeResult b = makeChecker(off)->check(pathOf(left, qa), pathOf(right, qb));
    EXPECT_EQ(a.code, b.code) << "padding " << padding;
    EXPECT_EQ(a.candidate_pairs, b.candidate_pairs);
    EXPECT_LE(a.exact_checks, b.exact_checks);
    EXPECT_EQ(a.exact_checks + a.certified_pairs, b.exact_checks) << "padding " << padding;
    if (a.code == SweptCode::Violation)
    {
      EXPECT_NEAR(a.min_distance, b.min_distance, 1e-9);
    }
    else
    {
      EXPECT_LE(a.min_distance, b.min_distance + 1e-9);  // a certified bound is never above the exact one
    }
    if (a.code == SweptCode::Ok)
    {
      EXPECT_GE(a.min_distance, padding);
    }
    EXPECT_GT(a.max_step_a_m, 0.0);
    EXPECT_EQ(a.max_step_b_m, 0.0);  // the static arm does not move
    std::printf("[cert] padding %.2f: candidates %zu, exact with %zu / without %zu, certified %zu, code %d\n",
                padding, a.candidate_pairs, a.exact_checks, b.exact_checks, a.certified_pairs, int(a.code));
    if (padding == 0.01)
    {
      EXPECT_EQ(a.code, SweptCode::Ok);
    }
    if (padding >= 0.05)
    {
      EXPECT_EQ(a.code, SweptCode::Violation);
    }
    if (padding <= 0.05)
    {
      EXPECT_LT(a.exact_checks, b.exact_checks);
    }
  }
}

TEST_F(SweptVolumeTest, StepDisplacementBoundIsRigorous)
{
  // One joint moves 0.01 rad: the farthest point of the left arm (tool sphere end, ~0.63 m from
  // joint1) moves by about 0.0063 m; the bound must be at least that and not absurdly larger.
  SweptVolumeConfig cfg;
  auto c = makeChecker(cfg);
  const SweptVolumeResult r = c->check(pathOf(left, {{0.0, 0.0}, {0.01, 0.0}}), pathOf(right, {{0, 0}}));
  EXPECT_GE(r.max_step_a_m, 0.01 * 0.63 - 1e-9);
  EXPECT_LE(r.max_step_a_m, 0.01 * 0.63 * 3);
}

TEST_F(SweptVolumeTest, ToleranceIsNeverZeroAndIsAppliedToTheVerdict)
{
  SweptVolumeConfig cfg;
  EXPECT_GT(cfg.exact_distance_tolerance_m, 0.0);  // default is never zero
  EXPECT_GE(cfg.exact_distance_tolerance_m, 0.0007 * 0.1);  // >= measured max overestimate (0.07 mm)
  std::unique_ptr<SweptVolumeChecker> bad;
  cfg.exact_distance_tolerance_m = 0.0;
  EXPECT_FALSE(SweptVolumeChecker::create(scene, left, right, cfg, bad));
  cfg.exact_distance_tolerance_m = -0.001;
  EXPECT_FALSE(SweptVolumeChecker::create(scene, left, right, cfg, bad));

  // Hand case: exact 0.030. Padding 0.0295 passes with a 0.1 mm tolerance and fails with 1 mm.
  const std::vector<double> qa{M_PI / 2, -M_PI / 2}, qb{0, 0};
  SweptVolumeConfig small;
  small.padding_m = 0.0295;
  small.exact_distance_tolerance_m = 0.0001;
  EXPECT_TRUE(makeChecker(small)->check(pathOf(left, {qa}), pathOf(right, {qb})).safe());
  SweptVolumeConfig big = small;
  big.exact_distance_tolerance_m = 0.001;
  const SweptVolumeResult r = makeChecker(big)->check(pathOf(left, {qa}), pathOf(right, {qb}));
  EXPECT_EQ(r.code, SweptCode::Violation);
  EXPECT_NEAR(r.min_distance, 0.030 - 0.001, 1e-5);
}

TEST_F(SweptVolumeTest, ToleranceIsAppliedToTheCertificates)
{
  // Two left samples a hair apart, right arm static, all near the 0.030 m hand case. Pair 0 gets an
  // exact query with distance d0; pair 1 is certified iff d0 - tol >= padding + 1.5 * step (the
  // sampling-gap threshold is padding + 0.5 * step and the certificate subtracts one more step).
  const std::vector<double> qa0{M_PI / 2, -M_PI / 2}, qa1{M_PI / 2 + 0.002, -M_PI / 2}, qb{0, 0};
  double d0;
  std::string x, y;
  {
    auto c = makeChecker();
    ASSERT_TRUE(c->exactDistance(qa0, qb, d0, x, y));
  }
  SweptVolumeConfig probe;
  probe.padding_m = 0.0;
  const double step = makeChecker(probe)->check(pathOf(left, {qa0, qa1}), pathOf(right, {qb})).max_step_a_m;
  ASSERT_GT(step, 0.0005);
  const double padding = d0 - 1.5 * step - 0.0015;

  auto run = [&](double tol)
  {
    SweptVolumeConfig cfg;
    cfg.padding_m = padding;
    cfg.exact_distance_tolerance_m = tol;
    return makeChecker(cfg)->check(pathOf(left, {qa0, qa1}), pathOf(right, {qb}));
  };
  const SweptVolumeResult small = run(0.001);   // d0 - 0.001 >= padding + 1.5 step: pair 1 certified
  const SweptVolumeResult large = run(0.0025);  // d0 - 0.0025 <  padding + 1.5 step: needs its own query
  EXPECT_EQ(small.candidate_pairs, 2u);
  EXPECT_EQ(large.candidate_pairs, 2u);
  EXPECT_EQ(small.certified_pairs, 1u);
  EXPECT_EQ(small.exact_checks, 1u);
  EXPECT_EQ(large.certified_pairs, 0u);
  EXPECT_EQ(large.exact_checks, 2u);
  EXPECT_TRUE(small.safe()) << small.message;
}

TEST(CoplanarBoxes, FclOverestimateIsJudgedSafelyWithAToleranceButNotWithout)
{
  const std::string dir = DUAL_CUT_TEST_DATA_DIR;
  auto model = dual_cut_test::buildModel(
    dual_cut_test::readFile(dir + "/synthetic_dual_coplanar.urdf"),
    dual_cut_test::readFile(dir + "/synthetic_dual.srdf"));
  ASSERT_NE(model, nullptr);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  ArmSpec left, right;
  ASSERT_TRUE(makeArmSpec(*model, "left", left));
  ASSERT_TRUE(makeArmSpec(*model, "right", right));
  const std::vector<std::vector<double>> q{{0, 0}};

  auto run = [&](double padding, double tol)
  {
    SweptVolumeConfig cfg;
    cfg.padding_m = padding;
    cfg.exact_distance_tolerance_m = tol;
    std::unique_ptr<SweptVolumeChecker> c;
    EXPECT_TRUE(SweptVolumeChecker::create(scene, left, right, cfg, c));
    return c->check(pathOf(left, q), pathOf(right, q));
  };
  // True base-to-base distance is 0.400 m. Documented FCL behaviour: it reports ~0.4123 here.
  std::unique_ptr<SweptVolumeChecker> probe;
  ASSERT_TRUE(SweptVolumeChecker::create(scene, left, right, SweptVolumeConfig{}, probe));
  double raw;
  std::string la, lb;
  ASSERT_TRUE(probe->exactDistance(q[0], q[0], raw, la, lb));
  if (raw < 0.405) GTEST_SKIP() << "FCL is exact on coplanar boxes in this build (raw " << raw << ")";
  EXPECT_NEAR(raw, 0.4123, 2e-3);

  // Padding 0.405 is above the true 0.400: the correct verdict is "unsafe".
  // Without a real tolerance the overestimate wrongly passes (this is the failure being guarded).
  EXPECT_TRUE(run(0.405, 1e-6).safe());
  // With a tolerance covering the FCL error the case is judged safely (refused).
  const SweptVolumeResult r = run(0.405, 0.02);
  EXPECT_EQ(r.code, SweptCode::Violation) << r.message;
  EXPECT_FALSE(r.safe());
  EXPECT_NEAR(r.min_distance, raw - 0.02, 1e-6);
}

TEST_F(SweptVolumeTest, EffectiveMinSubtractsTheSamplingGap)
{
  EXPECT_DOUBLE_EQ(effectiveMinDistance(0.50, 0.10, 0.30), 0.30);
  EXPECT_DOUBLE_EQ(effectiveMinDistance(0.50, 0.0, 0.0), 0.50);

  // Two left samples 0.1 rad apart, right arm static: the arm sweeps ~0.06 m between the samples.
  // Pick a padding between (sampled min - gap) and the sampled min: the sampled minimum alone would
  // pass, the effective one must not.
  const auto pa = pathOf(left, {{0.0, 0.0}, {0.1, 0.0}});
  const auto pb = pathOf(right, {{0, 0}});
  SweptVolumeConfig cfg;
  cfg.padding_m = 1.0;  // huge: forces the exact minimum to be computed (reported exactly on violation)
  const SweptVolumeResult base = makeChecker(cfg)->check(pa, pb);
  ASSERT_TRUE(base.min_distance_exact);
  ASSERT_GT(base.sampling_gap_m, 0.02);
  ASSERT_GT(base.min_distance, 2.0 * base.sampling_gap_m);
  cfg.padding_m = base.min_distance - 0.5 * base.sampling_gap_m;
  const SweptVolumeResult r = makeChecker(cfg)->check(pa, pb);
  EXPECT_GE(r.min_distance, cfg.padding_m) << "sampled minimum alone would pass";
  EXPECT_NEAR(r.effective_min_distance,
              effectiveMinDistance(r.min_distance, r.max_step_a_m, r.max_step_b_m), 1e-12);
  EXPECT_LT(r.effective_min_distance, cfg.padding_m);
  EXPECT_EQ(r.code, SweptCode::Violation) << r.message;
  EXPECT_FALSE(r.safe());

  // Padding below the effective minimum passes.
  cfg.padding_m = base.min_distance - 1.5 * base.sampling_gap_m;
  const SweptVolumeResult ok = makeChecker(cfg)->check(pa, pb);
  EXPECT_TRUE(ok.safe()) << ok.message;
  EXPECT_GE(ok.effective_min_distance, cfg.padding_m);
}

TEST_F(SweptVolumeTest, StopAtFirstViolation)
{
  SweptVolumeConfig cfg;
  cfg.padding_m = 0.05;
  cfg.stop_at_first_violation = true;
  auto c = makeChecker(cfg);
  const std::vector<double> qa{M_PI / 2, -M_PI / 2};
  const SweptVolumeResult r = c->check(pathOf(left, {qa, qa, qa}), pathOf(right, {{0, 0}, {0, 0}}));
  EXPECT_EQ(r.code, SweptCode::Violation);
  EXPECT_EQ(r.exact_checks, 1u);
}

// ---- world check -------------------------------------------------------------------------------

class WorldCheckTest : public SweptVolumeTest
{
protected:
  void SetUp() override
  {
    SweptVolumeTest::SetUp();
    // Obstacle: 0.05 m cube centred at (0, 0.3, 0.1). Left link1 sweeps into it near joint1 = 90 deg.
    scene->getWorldNonConst()->addToObject(
      "obs", std::make_shared<shapes::Box>(0.05, 0.05, 0.05),
      Eigen::Isometry3d(Eigen::Translation3d(0.0, 0.3, 0.1)));
  }

  static std::vector<std::vector<double>> sweep()
  {
    std::vector<std::vector<double>> qs;
    for (double deg : {0.0, 30.0, 60.0, 90.0}) qs.push_back({deg * M_PI / 180.0, 0.0});
    return qs;
  }
  std::unique_ptr<WorldChecker> makeWorld(const WorldCheckConfig & cfg)
  {
    std::unique_ptr<WorldChecker> w;
    const Status s = WorldChecker::create(scene, left, right, cfg, w);
    EXPECT_TRUE(s) << s.message;
    return w;
  }
};

TEST_F(WorldCheckTest, FindsTheFirstCollidingSample)
{
  auto w = makeWorld(WorldCheckConfig{});
  ASSERT_NE(w, nullptr);
  const SampledPath path = pathOf(left, sweep());
  const WorldCheckResult r = w->check(path, {0, 0});
  EXPECT_EQ(r.code, WorldCode::Collision) << r.message;
  EXPECT_FALSE(r.safe());
  EXPECT_EQ(r.sample_index, 3u);
  EXPECT_EQ(r.samples_checked, 4u);
  EXPECT_TRUE((r.link == "arm_left_link1" && r.object == "obs") ||
              (r.link == "obs" && r.object == "arm_left_link1"))
    << r.link << " / " << r.object;
}

TEST_F(WorldCheckTest, CleanPathAndOtherArmIsIgnored)
{
  auto w = makeWorld(WorldCheckConfig{});
  std::vector<std::vector<double>> qs{{0, 0}, {20 * M_PI / 180, 0}, {40 * M_PI / 180, 0}};
  EXPECT_TRUE(w->check(pathOf(left, qs), {0, 0}).safe());

  // Left link2 pointing +y penetrates the right base: that is the cross-arm check's business, the
  // world check must not report it (the other arm is held and all its pairs are allowed).
  // (Use a config where the obstacle is not touched: joint1 = 90 deg touches "obs", so move obs.)
  scene->getWorldNonConst()->removeObject("obs");
  const WorldCheckResult r = w->check(pathOf(left, {{M_PI / 2, 0}}), {0, 0});
  EXPECT_TRUE(r.safe()) << r.message;
}

TEST_F(WorldCheckTest, ContactAllowedOnlyForConfiguredLinksAndObjects)
{
  scene->getWorldNonConst()->addToObject(
    "detected_carton", std::make_shared<shapes::Box>(0.05, 0.05, 0.05),
    Eigen::Isometry3d(Eigen::Translation3d(0.0, 0.3, 0.1)));
  scene->getWorldNonConst()->removeObject("obs");

  WorldCheckConfig cfg;
  cfg.contact_link_names = {"arm_left_link1"};
  auto w = makeWorld(cfg);
  ASSERT_NE(w, nullptr);

  SampledPath path = pathOf(left, sweep());
  EXPECT_EQ(w->check(path, {0, 0}).code, WorldCode::Collision);  // no flag -> collision
  path.samples[3].contact_allowed = true;
  EXPECT_TRUE(w->check(path, {0, 0}).safe());                    // flagged sample -> allowed

  // A flagged sample colliding with a DIFFERENT object is still a collision.
  scene->getWorldNonConst()->removeObject("detected_carton");
  scene->getWorldNonConst()->addToObject(
    "pole", std::make_shared<shapes::Box>(0.05, 0.05, 0.05),
    Eigen::Isometry3d(Eigen::Translation3d(0.0, 0.3, 0.1)));
  const WorldCheckResult r = w->check(path, {0, 0});
  EXPECT_EQ(r.code, WorldCode::Collision);
  EXPECT_TRUE(r.sample_contact_allowed);

  // Flagged samples without any configured contact link are a configuration error.
  auto w2 = makeWorld(WorldCheckConfig{});
  EXPECT_EQ(w2->check(path, {0, 0}).code, WorldCode::InvalidInput);

  // Contact link that is not an arm link -> create fails.
  WorldCheckConfig bad;
  bad.contact_link_names = {"arm_right_link1"};
  std::unique_ptr<WorldChecker> w3;
  EXPECT_FALSE(WorldChecker::create(scene, left, right, bad, w3));
}

TEST_F(WorldCheckTest, FailClosedInputs)
{
  WorldCheckConfig cfg;
  cfg.max_samples = 3;
  auto w = makeWorld(cfg);
  EXPECT_EQ(w->check(pathOf(left, sweep()), {0, 0}).code, WorldCode::TooManySamples);
  EXPECT_EQ(w->check(SampledPath{}, {0, 0}).code, WorldCode::InvalidInput);
  EXPECT_EQ(w->check(pathOf(left, {{0, 0}}), {0}).code, WorldCode::InvalidInput);
  EXPECT_EQ(w->check(pathOf(left, {{0, 0}}), {NAN, 0}).code, WorldCode::InvalidInput);
  EXPECT_EQ(w->check(pathOf(right, {{0, 0}}), {0, 0}).code, WorldCode::InvalidInput);
}

TEST_F(WorldCheckTest, PenetrationToleranceIgnoresOnlyShallowGrazes)
{
  // A thin obstacle that the sweep touches with a known depth at its last sample. Strict: collision; with a
  // tolerance larger than the depth: ok; smaller: still a collision.
  scene->getWorldNonConst()->removeObject("obs");
  // Link1 at joint1 = 90 deg occupies x in [-0.02, 0.02]; a 0.05 box centred at x = 0.045 overlaps it by 0.005.
  scene->getWorldNonConst()->addToObject(
    "graze", std::make_shared<shapes::Box>(0.05, 0.05, 0.05), Eigen::Isometry3d(Eigen::Translation3d(0.045, 0.3, 0.1)));
  const SampledPath path = pathOf(left, {{M_PI / 2, 0.0}});
  WorldCheckConfig strict;
  auto w0 = makeWorld(strict);
  const WorldCheckResult r0 = w0->check(path, {0, 0});
  ASSERT_EQ(r0.code, WorldCode::Collision);
  EXPECT_GT(r0.depth_m, 0.0);
  EXPECT_LT(r0.depth_m, 0.006);
  WorldCheckConfig loose;
  loose.penetration_tolerance_m = 0.01;
  EXPECT_TRUE(makeWorld(loose)->check(path, {0, 0}).safe());
  WorldCheckConfig tight;
  tight.penetration_tolerance_m = r0.depth_m * 0.5;
  EXPECT_EQ(makeWorld(tight)->check(path, {0, 0}).code, WorldCode::Collision);
}
