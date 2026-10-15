#include <gtest/gtest.h>

#include <cmath>

#include <geometric_shapes/shapes.h>

#include "dual_cut_coordinator/phase_validator.hpp"
#include "test_common.hpp"

using namespace dual_cut;

namespace
{

JointTrajectory rest2rest(const std::vector<double> & from, const std::vector<double> & to, double T)
{
  JointTrajectory t;
  t.joint_names = {"joint1", "joint2"};
  // Quintic rest-to-rest polynomial sampled at 8 points, with the analytic derivatives, so that the
  // controller-style spline reproduces it exactly.
  const int n = 8;
  for (int k = 0; k <= n; ++k)
  {
    const double s = static_cast<double>(k) / n;
    const double b = 10 * s * s * s - 15 * s * s * s * s + 6 * s * s * s * s * s;
    const double db = (30 * s * s - 60 * s * s * s + 30 * s * s * s * s) / T;
    const double ddb = (60 * s - 180 * s * s + 120 * s * s * s) / (T * T);
    TrajPoint p;
    p.time_s = s * T;
    for (size_t j = 0; j < 2; ++j)
    {
      p.positions.push_back(from[j] + (to[j] - from[j]) * b);
      p.velocities.push_back((to[j] - from[j]) * db);
      p.accelerations.push_back((to[j] - from[j]) * ddb);
    }
    t.points.push_back(p);
  }
  return t;
}

PlannedSegmentData seg(SegmentKind k, const std::string & name, std::vector<double> from, std::vector<double> to)
{
  PlannedSegmentData s;
  s.kind = k;
  s.stage_name = name;
  s.trajectory = rest2rest(from, to, 1.5);
  return s;
}

ArmPhase phase(const std::string & side, std::vector<std::pair<std::vector<double>, std::vector<double>>> moves,
               std::vector<SegmentKind> kinds = {})
{
  ArmPhase ph;
  ph.side = side;
  ph.joint_names = {"joint1", "joint2"};
  for (size_t i = 0; i < moves.size(); ++i)
    ph.segments.push_back(seg(i < kinds.size() ? kinds[i] : SegmentKind::Transit, "seg" + std::to_string(i),
                              moves[i].first, moves[i].second));
  ph.start_q = moves.front().first;
  ph.end_q = moves.back().second;
  return ph;
}

class PhaseValidatorTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model = dual_cut_test::syntheticModel();
    ASSERT_NE(model, nullptr);
    scene = std::make_shared<planning_scene::PlanningScene>(model);
    PhaseValidatorConfig cfg;
    cfg.tip_link_suffix = "tool";
    cfg.swept.padding_m = 0.05;
    cfg.contact_links_left = {"arm_left_tool"};
    cfg.contact_links_right = {"arm_right_tool"};
    cfg.sample.dt_s = 0.05;
    cfg.sample.max_joint_step_rad = 0.01;
    const Status s = PhaseValidator::create(scene, cfg, validator);
    ASSERT_TRUE(s) << s.message;
  }
  moveit::core::RobotModelPtr model;
  std::shared_ptr<planning_scene::PlanningScene> scene;
  std::unique_ptr<PhaseValidator> validator;
};

}  // namespace

TEST_F(PhaseValidatorTest, ClearScenarioRunsInParallel)
{
  // Each arm swings away from the other (left to -y, right to +y).
  const auto left = phase("left", {{{0, 0}, {-0.6, 0.3}}});
  const auto right = phase("right", {{{0, 0}, {0.6, -0.3}}});
  const PhaseValidation v = validator->validate(left, right, {}, {});
  EXPECT_TRUE(v.ok) << v.message;
  EXPECT_FALSE(v.serial);
  EXPECT_TRUE(v.cross.safe());
  EXPECT_GT(v.samples_left, 20u);
  EXPECT_GT(v.min_cross_distance_m, 0.05);
  const ValidationRecord rec = PhaseValidator::toRecord(v);
  EXPECT_TRUE(rec.cross_arm_ok && rec.world_ok_left && rec.world_ok_right && rec.tip_on_line_ok);
  EXPECT_FALSE(rec.serial_fallback);
}

TEST_F(PhaseValidatorTest, CollidingScenarioIsRefusedEvenSerially)
{
  // Both arms end up reaching toward each other and stay: parallel collides, and the second mover
  // would drive into the first mover's parked end pose.
  const auto left = phase("left", {{{0, 0}, {1.2, 0}}});
  const auto right = phase("right", {{{0, 0}, {-1.2, 0}}});
  const PhaseValidation v = validator->validate(left, right, {}, {});
  EXPECT_FALSE(v.ok);
  EXPECT_FALSE(v.serial);
  EXPECT_FALSE(v.cross.safe());
  EXPECT_NE(v.message.find("cross-arm"), std::string::npos) << v.message;
}

TEST_F(PhaseValidatorTest, SerialOnlyScenarioFallsBackToOneArmAfterTheOther)
{
  // Each arm swings out toward the middle and back; the swept volumes cross, but each arm alone is
  // clear of the other one parked at home.
  const auto left = phase("left", {{{0, 0}, {0.9, -1.6}}, {{0.9, -1.6}, {0, 0}}});
  const auto right = phase("right", {{{0, 0}, {-0.9, 1.6}}, {{-0.9, 1.6}, {0, 0}}});
  const PhaseValidation v = validator->validate(left, right, {}, {});
  EXPECT_FALSE(v.cross.safe()) << "the parallel swept volumes must cross";
  EXPECT_TRUE(v.ok) << v.message;
  EXPECT_TRUE(v.serial);
  EXPECT_EQ(v.serial_first, "left");
  EXPECT_TRUE(v.serial_first_result.safe());
  EXPECT_TRUE(v.serial_second_result.safe());
  EXPECT_TRUE(PhaseValidator::toRecord(v).serial_fallback);

  // With the fallback disabled the same phase is refused.
  PhaseValidatorConfig cfg = validator->config();
  cfg.allow_serial_fallback = false;
  std::unique_ptr<PhaseValidator> strict;
  ASSERT_TRUE(PhaseValidator::create(scene, cfg, strict));
  EXPECT_FALSE(strict->validate(left, right, {}, {}).ok);
}

TEST_F(PhaseValidatorTest, SerialOrderCanBeRightFirst)
{
  // Left parks (end pose) in the middle, right comes from the side where left's END pose would block
  // it if left went first: left moves to the middle and stays; right swings out and back.
  // right-first works (left still at its start), left-first does not (right would hit left's end).
  const auto left = phase("left", {{{0, 0}, {0.9, -1.6}}});
  const auto right = phase("right", {{{0, 0}, {-0.9, 1.6}}, {{-0.9, 1.6}, {0, 0}}});
  const PhaseValidation v = validator->validate(left, right, {}, {});
  ASSERT_TRUE(v.ok) << v.message;
  ASSERT_TRUE(v.serial);
  EXPECT_EQ(v.serial_first, "right");
}

TEST_F(PhaseValidatorTest, WorldCollisionRefuses)
{
  scene->getWorldNonConst()->addToObject(
    "camera_pole", std::make_shared<shapes::Box>(0.1, 0.1, 0.4),
    Eigen::Isometry3d(Eigen::Translation3d(0.2, -0.2, 0.1)));  // in the left arm's swing at -y
  std::unique_ptr<PhaseValidator> v2;
  ASSERT_TRUE(PhaseValidator::create(scene, validator->config(), v2));
  const auto left = phase("left", {{{0, 0}, {-0.9, 0.0}}});
  const auto right = phase("right", {{{0, 0}, {0.6, -0.3}}});
  const PhaseValidation v = v2->validate(left, right, {}, {});
  EXPECT_FALSE(v.ok);
  EXPECT_FALSE(v.world_left.safe());
  EXPECT_TRUE(v.world_right.safe()) << v.world_right.message;
  EXPECT_NE(v.message.find("left world check"), std::string::npos) << v.message;
}

TEST_F(PhaseValidatorTest, TipOnLineGateUsesTheDualModelFk)
{
  // A CUT segment moving joint1 from 0.3 to 0.4 rad. The tool of the left arm sits at
  // FK(arm_left_tool); build the cut line from the FK of the first/last point, then lie about it.
  const auto left = phase("left", {{{0, 0}, {0.3, 0}}, {{0.3, 0}, {0.4, 0}}}, {SegmentKind::Transit, SegmentKind::Cut});
  const auto right = phase("right", {{{0, 0}, {0.6, -0.3}}});
  auto fk = [&](double q1)
  {
    moveit::core::RobotState st(model);
    st.setToDefaultValues();
    st.setVariablePosition("arm_left_joint1", q1);
    st.update();
    return st.getGlobalLinkTransform("arm_left_tool").translation();
  };
  // The tool traces an arc of radius ~0.63 m: its chord deviates by sagitta = r (1 - cos(0.05)) ~ 0.8 mm.
  const CutLine good{fk(0.3), fk(0.4)};
  PhaseValidation v = validator->validate(left, right, {good}, {});
  EXPECT_TRUE(v.tip_left.safe()) << v.tip_left.message;
  EXPECT_LT(v.tip_left.max_deviation_m, 0.002);
  EXPECT_TRUE(v.ok) << v.message;

  CutLine shifted = good;
  shifted.start.y() += 0.01;
  shifted.end.y() += 0.01;
  v = validator->validate(left, right, {shifted}, {});
  EXPECT_FALSE(v.ok);
  EXPECT_FALSE(v.tip_left.safe());
  EXPECT_NE(v.message.find("tip-on-line"), std::string::npos) << v.message;

  // wrong number of lines
  v = validator->validate(left, right, {}, {});
  EXPECT_FALSE(v.ok);
}

TEST_F(PhaseValidatorTest, FailClosedOnBadInputs)
{
  const auto left = phase("left", {{{0, 0}, {-0.6, 0.3}}});
  auto right = phase("right", {{{0, 0}, {0.6, -0.3}}});
  right.segments[0].trajectory.joint_names[0] = "elbow";   // not jointN
  EXPECT_FALSE(validator->validate(left, right, {}, {}).ok);
  EXPECT_FALSE(validator->validate(right, left, {}, {}).ok);   // sides swapped
  // missing tip link
  PhaseValidatorConfig cfg = validator->config();
  cfg.tip_link_suffix = "no_such_link";
  std::unique_ptr<PhaseValidator> bad;
  EXPECT_FALSE(PhaseValidator::create(scene, cfg, bad));
}

TEST(DisplayTrajectory, ParallelHoldsTheShorterAndSerialSequences)
{
  auto mk = [](const std::string & side, double to, double T)
  {
    ArmPhase ph;
    ph.side = side;
    ph.joint_names = {"joint1", "joint2"};
    PlannedSegmentData s;
    s.kind = SegmentKind::Transit;
    s.trajectory = rest2rest({0, 0}, {to, 0}, T);
    ph.segments = {s};
    ph.start_q = {0, 0};
    ph.end_q = {to, 0};
    return ph;
  };
  Phase p("A", mk("left", 1.0, 1.0), mk("right", -0.5, 2.0));
  trajectory_msgs::msg::JointTrajectory d;
  ASSERT_TRUE(bothArmsTrajectory(p, 0.1, d));
  ASSERT_EQ(d.joint_names.size(), 4u);
  EXPECT_EQ(d.joint_names[0], "arm_left_joint1");
  EXPECT_EQ(d.joint_names[2], "arm_right_joint1");
  EXPECT_EQ(d.points.size(), 21u);                       // 0 .. 2.0 s every 0.1 s
  EXPECT_NEAR(d.points.back().positions[0], 1.0, 1e-9);  // left held at its end
  EXPECT_NEAR(d.points.back().positions[2], -0.5, 1e-9);
  EXPECT_NEAR(d.points[10].positions[0], 1.0, 1e-9);     // left done at t = 1.0
  const double s25 = 0.25, b25 = 10 * std::pow(s25, 3) - 15 * std::pow(s25, 4) + 6 * std::pow(s25, 5);
  EXPECT_NEAR(d.points[5].positions[2], -0.5 * b25, 1e-9);   // right at t = 0.5 s (s = 0.25 of 2 s)

  // Serial, right arm first: the left arm waits at its start for 2 s, then moves for 1 s.
  ValidationRecord rec;
  rec.serial_fallback = true;
  rec.serial_first = "right";
  p.recordValidation(rec, true);
  ASSERT_TRUE(bothArmsTrajectory(p, 0.5, d));
  EXPECT_NEAR(d.points.back().time_from_start.sec + 1e-9 * d.points.back().time_from_start.nanosec, 3.0, 1e-9);
  EXPECT_NEAR(d.points[3].positions[0], 0.0, 1e-12);   // t = 1.5: left not started
  EXPECT_NEAR(d.points[3].positions[2], -0.5 * (10 * std::pow(0.75, 3) - 15 * std::pow(0.75, 4) + 6 * std::pow(0.75, 5)), 1e-9);
  EXPECT_NEAR(d.points.back().positions[0], 1.0, 1e-9);
  EXPECT_NEAR(d.points.back().positions[2], -0.5, 1e-9);   // right held at its end
  EXPECT_NEAR(d.points[4].positions[0], 0.0, 1e-12);       // t = 2.0: left starts now
}
