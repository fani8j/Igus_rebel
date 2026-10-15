#include <gtest/gtest.h>

#include <cmath>

#include "dual_cut_coordinator/gates.hpp"

using namespace dual_cut;

namespace
{
geometry_msgs::msg::PoseStamped pose(const std::string & frame, double x, double y, double z)
{
  geometry_msgs::msg::PoseStamped p;
  p.header.frame_id = frame;
  p.pose.position.x = x;
  p.pose.position.y = y;
  p.pose.position.z = z;
  p.pose.orientation.w = 1.0;
  return p;
}

// A CUT segment where joint1 goes from a to b (single joint), tip = (joint1, 0, 0.1).
ArmPhase cutPhase(double a, double b)
{
  ArmPhase ph;
  ph.side = "left";
  ph.joint_names = {"joint1"};
  PlannedSegmentData lift;
  lift.kind = SegmentKind::Lift;
  lift.trajectory.joint_names = {"joint1"};
  TrajPoint p0;
  p0.time_s = 0.0;
  p0.positions = {a - 0.1};
  TrajPoint p1;
  p1.time_s = 1.0;
  p1.positions = {a};
  lift.trajectory.points = {p0, p1};
  PlannedSegmentData cut;
  cut.kind = SegmentKind::Cut;
  cut.stage_name = "cut left";
  cut.trajectory.joint_names = {"joint1"};
  p0.positions = {a};
  p1.positions = {b};
  cut.trajectory.points = {p0, p1};
  ph.segments = {lift, cut};
  return ph;
}
const TipFk kLineFk = [](const std::vector<std::string> &, const std::vector<double> & q, Eigen::Vector3d & tip) -> Status
{
  tip = Eigen::Vector3d(q[0], 0.0, 0.1);
  return Status::Ok();
};
}  // namespace

TEST(FrameGate, EqualAndOffTransforms)
{
  const std::array<double, 3> xyz{{-0.0013202340013850325, -0.4542609819306861, 0.004780591203829147}};
  const std::array<double, 3> rpy{{-0.01822495461019838, 0.008429085689667826, 0.04908320092978558}};
  const Eigen::Isometry3d T = poseFromXyzRpy(xyz, rpy);
  EXPECT_TRUE(checkFrameConsistency(T, T));
  EXPECT_TRUE(checkFrameConsistency(T, poseFromXyzRpy({{xyz[0] + 5e-7, xyz[1], xyz[2]}}, rpy)));
  EXPECT_FALSE(checkFrameConsistency(T, poseFromXyzRpy({{xyz[0] + 5e-6, xyz[1], xyz[2]}}, rpy)));
  EXPECT_FALSE(checkFrameConsistency(T, poseFromXyzRpy(xyz, {{rpy[0], rpy[1], rpy[2] + 5e-6}})));
  EXPECT_FALSE(checkFrameConsistency(T, poseFromXyzRpy({{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}})));  // the 454 mm lie
  Eigen::Isometry3d nan = T;
  nan.translation().x() = NAN;
  EXPECT_FALSE(checkFrameConsistency(nan, T));
}

TEST(FrameGate, RpyConventionIsFixedAxisZyx)
{
  // yaw 90 deg: x axis maps to y
  const Eigen::Isometry3d T = poseFromXyzRpy({{0, 0, 0}}, {{0, 0, M_PI / 2}});
  EXPECT_NEAR((T.linear() * Eigen::Vector3d::UnitX() - Eigen::Vector3d::UnitY()).norm(), 0.0, 1e-12);
  // roll then yaw composition: R = Rz Ry Rx
  const Eigen::Isometry3d U = poseFromXyzRpy({{1, 2, 3}}, {{0.3, 0.2, 0.1}});
  const Eigen::Matrix3d R = (Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX())).toRotationMatrix();
  EXPECT_LT((U.linear() - R).norm(), 1e-12);
  EXPECT_EQ(U.translation(), Eigen::Vector3d(1, 2, 3));
}

TEST(CutLines, FromTargetsAndFailClosed)
{
  const Eigen::Isometry3d T = poseFromXyzRpy({{0, -0.45, 0}}, {{0, 0, M_PI / 2}});
  std::vector<CutLine> lines;
  ASSERT_TRUE(cutLinesFromTargets({pose("base_link", 1, 0, 0), pose("base_link", 2, 0, 0)}, 1, "base_link", T, lines));
  ASSERT_EQ(lines.size(), 1u);
  // x axis -> y axis (yaw 90), plus the base offset
  EXPECT_NEAR((lines[0].start - Eigen::Vector3d(0, -0.45 + 1, 0)).norm(), 0.0, 1e-12);
  EXPECT_NEAR((lines[0].end - Eigen::Vector3d(0, -0.45 + 2, 0)).norm(), 0.0, 1e-12);
  EXPECT_FALSE(cutLinesFromTargets({pose("base_link", 1, 0, 0)}, 1, "base_link", T, lines));          // count
  EXPECT_FALSE(cutLinesFromTargets({pose("base_link", 1, 0, 0), pose("world", 2, 0, 0)}, 1, "base_link", T, lines));
  EXPECT_FALSE(cutLinesFromTargets({pose("base_link", NAN, 0, 0), pose("base_link", 2, 0, 0)}, 1, "base_link", T, lines));
  EXPECT_TRUE(cutLinesFromTargets({}, 0, "base_link", T, lines));
  EXPECT_TRUE(lines.empty());
}

TEST(CutLines, PointToSegment)
{
  const Eigen::Vector3d a(0, 0, 0), b(1, 0, 0);
  EXPECT_NEAR(distancePointToSegment({0.5, 0.3, 0}, a, b), 0.3, 1e-12);
  EXPECT_NEAR(distancePointToSegment({-0.5, 0, 0}, a, b), 0.5, 1e-12);   // beyond the start
  EXPECT_NEAR(distancePointToSegment({2.0, 0, 0}, a, b), 1.0, 1e-12);    // beyond the end
  EXPECT_NEAR(distancePointToSegment({0.2, 0, 0.4}, a, a), std::hypot(0.2, 0.4), 1e-12);  // degenerate
}

TEST(TipGate, PassesOnTheLineAndFailsOffIt)
{
  SampleOptions so;
  so.dt_s = 0.1;
  so.max_joint_step_rad = 0.01;
  const ArmPhase ph = cutPhase(0.2, 0.5);
  EXPECT_TRUE(checkTipOnLine(ph, {CutLine{{0.2, 0, 0.1}, {0.5, 0, 0.1}}}, kLineFk, 0.002, so).ok);

  // line shifted by 3 mm sideways: deviation of every sample is 3 mm > 2 mm
  const TipGateResult off = checkTipOnLine(ph, {CutLine{{0.2, 0.003, 0.1}, {0.5, 0.003, 0.1}}}, kLineFk, 0.002, so);
  EXPECT_FALSE(off.ok);
  EXPECT_NEAR(off.max_deviation_m, 0.003, 1e-9);
  // shifted by 1 mm: inside the 2 mm limit
  EXPECT_TRUE(checkTipOnLine(ph, {CutLine{{0.2, 0.001, 0.1}, {0.5, 0.001, 0.1}}}, kLineFk, 0.002, so).ok);
  // along-line error at an end: the cut stops 5 mm short of the exported end
  const TipGateResult short_end = checkTipOnLine(ph, {CutLine{{0.2, 0, 0.1}, {0.505, 0, 0.1}}}, kLineFk, 0.002, so);
  EXPECT_FALSE(short_end.ok);
  EXPECT_GT(short_end.max_endpoint_error_m, 0.004);
  // count mismatch
  EXPECT_FALSE(checkTipOnLine(ph, {}, kLineFk, 0.002, so).ok);
  // no CUT segment and no line is fine
  ArmPhase none = ph;
  none.segments.pop_back();
  EXPECT_TRUE(checkTipOnLine(none, {}, kLineFk, 0.002, so).ok);
  // FK failure fails closed
  const TipFk bad = [](const std::vector<std::string> &, const std::vector<double> &, Eigen::Vector3d &) -> Status
  { return Status::Error("no fk"); };
  EXPECT_FALSE(checkTipOnLine(ph, {CutLine{{0.2, 0, 0.1}, {0.5, 0, 0.1}}}, bad, 0.002, so).ok);
}

TEST(RestTracker, NeedsAnUninterruptedStretch)
{
  RestTracker r(0.03);
  EXPECT_DOUBLE_EQ(r.restDuration(1.0), 0.0);  // no sample yet
  r.update(1.0, 0.0);
  r.update(1.2, 0.01);
  EXPECT_NEAR(r.restDuration(1.4), 0.4, 1e-12);
  r.update(1.4, 0.05);  // moves
  EXPECT_DOUBLE_EQ(r.restDuration(1.4), 0.0);
  r.update(1.5, 0.0);
  r.update(1.9, 0.0);
  EXPECT_NEAR(r.restDuration(2.0), 0.5, 1e-12);
  r.update(2.1, NAN);   // garbage counts as moving
  EXPECT_DOUBLE_EQ(r.restDuration(2.1), 0.0);
  r.reset();
  EXPECT_DOUBLE_EQ(r.restDuration(3.0), 0.0);
}

TEST(Barriers, AllHoldAndEachOneFailsAlone)
{
  BarrierInputs ok;
  ok.joint_state_age_s = 0.02;
  ok.max_joint_error_to_start_rad = 0.002;
  ok.rest_duration_s = 0.7;
  ok.controllers_active = ok.hardware_active = true;
  ok.snapshot_age_s = 30.0;
  ok.stop_latched = false;
  ok.monitor_age_s = 0.05;
  ok.monitor_healthy = true;
  const BarrierLimits lim;
  ASSERT_TRUE(evaluateBarriers(ok, lim));

  auto fails = [&](auto mutate)
  {
    BarrierInputs b = ok;
    mutate(b);
    std::vector<std::string> f;
    const Status s = evaluateBarriers(b, lim, &f);
    EXPECT_FALSE(s.ok);
    EXPECT_EQ(f.size(), 1u) << s.message;
    return !s.ok && f.size() == 1u;
  };
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.joint_state_age_s = 0.1; }));            // not < 0.1
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.max_joint_error_to_start_rad = 0.0101; }));
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.rest_duration_s = 0.49; }));
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.controllers_active = false; }));
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.hardware_active = false; }));
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.snapshot_age_s = 240.5; }));
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.stop_latched = true; }));
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.monitor_age_s = 0.3; }));
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.monitor_healthy = false; }));
  EXPECT_TRUE(fails([](BarrierInputs & b) { b.joint_state_age_s = NAN; }));            // NaN never passes
  // The default-constructed inputs fail closed.
  EXPECT_FALSE(evaluateBarriers(BarrierInputs{}, lim));
}

TEST(Monitor, EachStopConditionAndTheOrderOfEvaluation)
{
  MonitorInputs ok;
  ok.have_joint_state = true;
  ok.joint_state_age_s = 0.05;
  ok.distance_valid = true;
  ok.cross_distance_m = 0.2;
  ok.poll_valid = true;
  ok.poll_age_s = 0.2;
  ok.controllers_active = ok.hardware_active = true;
  ok.tracking_error_rad = 0.01;
  const MonitorLimits lim;
  EXPECT_FALSE(evaluateMonitor(ok, lim).stop);
  EXPECT_FALSE(evaluateMonitor(ok, lim).warn);

  auto stops = [&](auto mutate)
  {
    MonitorInputs m = ok;
    mutate(m);
    const MonitorDecision d = evaluateMonitor(m, lim);
    EXPECT_TRUE(d.stop);
    EXPECT_FALSE(d.reason.empty());
    return d.stop;
  };
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.cross_distance_m = 0.0299; }));
  EXPECT_FALSE(evaluateMonitor([&] { MonitorInputs m = ok; m.cross_distance_m = 0.03; return m; }(), lim).stop);  // at the limit
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.joint_state_age_s = 0.41; }));
  EXPECT_FALSE(evaluateMonitor([&] { MonitorInputs m = ok; m.joint_state_age_s = 0.4; return m; }(), lim).stop);
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.have_joint_state = false; }));
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.unexpected_goal_end = true; m.goal_end_reason = "arm_right ABORTED"; }));
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.controllers_active = false; }));
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.hardware_active = false; }));
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.tracking_error_rad = 0.2001; }));
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.distance_valid = false; }));
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.cross_distance_m = NAN; }));
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.poll_valid = false; }));
  EXPECT_TRUE(stops([](MonitorInputs & m) { m.poll_age_s = 1.5; }));
  // warning band
  MonitorInputs w = ok;
  w.tracking_error_rad = 0.15;
  const MonitorDecision d = evaluateMonitor(w, lim);
  EXPECT_FALSE(d.stop);
  EXPECT_TRUE(d.warn);
  // an unexpected goal end wins over everything else
  MonitorInputs both = ok;
  both.unexpected_goal_end = true;
  both.goal_end_reason = "x";
  both.cross_distance_m = 0.0;
  EXPECT_NE(evaluateMonitor(both, lim).reason.find("goal ended"), std::string::npos);
}

TEST(BoxConsistency, SameBoxUpToItsSymmetries)
{
  const Eigen::Vector3d size(0.39, 0.28, 0.2);
  Eigen::Isometry3d a = Eigen::Isometry3d::Identity();
  a.translate(Eigen::Vector3d(0.43, -0.14, -0.06));
  a.rotate(Eigen::AngleAxisd(-1.57, Eigen::Vector3d::UnitZ()));
  double dc = 0, dr = 0;
  EXPECT_TRUE(checkBoxConsistency(a, size, a, size, 1e-3, 1e-3, &dc, &dr));
  EXPECT_DOUBLE_EQ(dc, 0.0);
  // rotated 180 degrees about its own vertical axis: the same box
  Eigen::Isometry3d flipped = a;
  flipped.rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
  EXPECT_TRUE(checkBoxConsistency(a, size, flipped, size, 1e-3, 1e-3, &dc, &dr));
  EXPECT_LT(dr, 1e-12);
  // rotated 90 degrees: another box
  Eigen::Isometry3d quarter = a;
  quarter.rotate(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
  EXPECT_FALSE(checkBoxConsistency(a, size, quarter, size, 1e-3, 1e-3));
  // shifted by 9 mm in z (the kind of mismatch two arms with a wrong frame would show)
  Eigen::Isometry3d shifted = a;
  shifted.translation().z() += 0.009;
  EXPECT_FALSE(checkBoxConsistency(a, size, shifted, size, 0.002, 1e-3, &dc));
  EXPECT_NEAR(dc, 0.009, 1e-12);
  EXPECT_TRUE(checkBoxConsistency(a, size, shifted, size, 0.01, 1e-3));
  // other dimensions
  EXPECT_FALSE(checkBoxConsistency(a, size, a, Eigen::Vector3d(0.39, 0.28, 0.25), 1e-3, 1e-3));
  // tilted about x by 0.01 rad
  Eigen::Isometry3d tilted = a;
  tilted.rotate(Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitX()));
  EXPECT_FALSE(checkBoxConsistency(a, size, tilted, size, 1e-3, 1e-3));
  EXPECT_TRUE(checkBoxConsistency(a, size, tilted, size, 1e-3, 0.02));
}
