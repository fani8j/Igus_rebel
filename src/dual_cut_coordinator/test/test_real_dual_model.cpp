// The one test that uses the real dual model (rebel_dual_table_description + moveit_config SRDF).
// Skipped when those packages are not installed or xacro cannot be run.
#include <gtest/gtest.h>

#include <cstdio>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <geometric_shapes/shapes.h>
#include "dual_cut_coordinator/swept_volume.hpp"
#include "test_common.hpp"

using namespace dual_cut;

namespace
{
std::string runCommand(const std::string & cmd)
{
  std::string out;
  FILE * p = popen(cmd.c_str(), "r");
  if (!p) return out;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
  if (pclose(p) != 0) return std::string();
  return out;
}
}  // namespace

TEST(RealDualModel, ArmSpecsSphereBoundAndCrossArmDistance)
{
  std::string desc, moveit;
  try
  {
    desc = ament_index_cpp::get_package_share_directory("rebel_dual_table_description");
    moveit = ament_index_cpp::get_package_share_directory("rebel_dual_table_moveit_config");
  }
  catch (const std::exception & e)
  {
    GTEST_SKIP() << "real dual model packages not installed: " << e.what();
  }
  const std::string urdf =
    runCommand("xacro " + desc + "/urdf/rebel_dual_table.urdf.xacro include_ros2_control:=false 2>/dev/null");
  if (urdf.empty()) GTEST_SKIP() << "xacro failed";
  const std::string srdf = dual_cut_test::readFile(moveit + "/config/rebel_dual_table.srdf");
  ASSERT_FALSE(srdf.empty());
  auto model = dual_cut_test::buildModel(urdf, srdf);
  ASSERT_NE(model, nullptr);

  ArmSpec left, right;
  ASSERT_TRUE(makeArmSpec(*model, "left", left));
  ASSERT_TRUE(makeArmSpec(*model, "right", right));
  ASSERT_EQ(left.joint_names.size(), 6u);
  ASSERT_EQ(right.joint_names.size(), 6u);
  EXPECT_EQ(left.joint_names[0], "arm_left_joint1");
  EXPECT_EQ(right.joint_names[5], "arm_right_joint6");

  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  SweptVolumeConfig cfg;
  cfg.padding_m = 0.05;
  std::unique_ptr<SweptVolumeChecker> c;
  const Status s = SweptVolumeChecker::create(scene, left, right, cfg, c);
  ASSERT_TRUE(s) << s.message;  // all collision geometry of both arms is supported
  EXPECT_GT(c->linksA().size(), 5u);
  EXPECT_GT(c->linksB().size(), 5u);

  const std::vector<double> home(6, 0.0);
  double d = 0.0, lb = 0.0;
  std::string a1, b1, a2, b2;
  ASSERT_TRUE(c->exactDistance(home, home, d, a1, b1));
  ASSERT_TRUE(c->sphereLowerBound(home, home, lb, a2, b2));
  RecordProperty("home_exact_distance_m", std::to_string(d));
  RecordProperty("home_sphere_lower_bound_m", std::to_string(lb));
  std::printf("[real model] home/home: exact %.4f m (%s / %s), sphere bound %.4f m\n", d, a1.c_str(),
              b1.c_str(), lb);
  EXPECT_LE(lb, d + 1e-6);
  EXPECT_TRUE(std::isfinite(d));

  // The full checker on single-sample paths is consistent with the exact distance.
  const SweptVolumeResult r = c->check(
    dual_cut_test::pathOf(left, {home}), dual_cut_test::pathOf(right, {home}));
  ASSERT_TRUE(r.code == SweptCode::Ok || r.code == SweptCode::Violation) << r.message;
  EXPECT_EQ(r.safe(), d >= cfg.padding_m);

  // A dense two-arm sweep (300 x 300 samples, 90000 sample pairs) stays tractable on the real model:
  // the sphere prefilter is loose there (one sphere per mesh link), so exact queries are limited by
  // the certificates, not by the pair count.
  std::vector<std::vector<double>> qa, qb;
  for (int i = 0; i < 300; ++i)
  {
    const double f = static_cast<double>(i) / 300.0;
    qa.push_back({0.8 * f, 0.5 * f, -0.4 * f, 0.0, 0.3 * f, 0.0});
    qb.push_back({-0.8 * f, 0.5 * f, -0.4 * f, 0.0, 0.3 * f, 0.0});
  }
  const SweptVolumeResult sweep = c->check(
    dual_cut_test::pathOf(left, qa), dual_cut_test::pathOf(right, qb));
  std::printf("[real model] dense sweep: code %d, min %.4f m (exact=%d), candidates %zu, exact queries %zu, certified %zu\n",
              static_cast<int>(sweep.code), sweep.min_distance, sweep.min_distance_exact,
              sweep.candidate_pairs, sweep.exact_checks, sweep.certified_pairs);
  EXPECT_TRUE(sweep.code == SweptCode::Ok || sweep.code == SweptCode::Violation) << sweep.message;
  EXPECT_LT(sweep.exact_checks, 2000u);
}

// The cut-contact allowance: the tool links allowed to touch detected_carton in contact-flagged samples
// must include the blade box (<prefix>knife_c_blade) and the TCP link; everything else, including the
// blade against the camera pole / head / table, stays fully checked.
TEST(RealDualModel, BladeContactAllowanceOnlyForTheCartonOnlyWhenFlagged)
{
  std::string desc, moveit;
  try
  {
    desc = ament_index_cpp::get_package_share_directory("rebel_dual_table_description");
    moveit = ament_index_cpp::get_package_share_directory("rebel_dual_table_moveit_config");
  }
  catch (const std::exception & e)
  {
    GTEST_SKIP() << "real dual model packages not installed: " << e.what();
  }
  const std::string urdf =
    runCommand("xacro " + desc + "/urdf/rebel_dual_table.urdf.xacro include_ros2_control:=false 2>/dev/null");
  if (urdf.empty()) GTEST_SKIP() << "xacro failed";
  auto model = dual_cut_test::buildModel(urdf, dual_cut_test::readFile(moveit + "/config/rebel_dual_table.srdf"));
  ASSERT_NE(model, nullptr);
  if (!model->hasLinkModel("arm_left_knife_c_blade")) GTEST_SKIP() << "installed model has no knife_c_blade link";
  ASSERT_FALSE(model->getLinkModel("arm_left_knife_c_blade")->getShapes().empty());

  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  ArmSpec left, right;
  ASSERT_TRUE(makeArmSpec(*model, "left", left));
  ASSERT_TRUE(makeArmSpec(*model, "right", right));

  // A small box centred on the blade at the home pose: the blade is inside it.
  moveit::core::RobotState st(model);
  st.setToDefaultValues();
  st.update();
  const Eigen::Isometry3d blade = st.getGlobalLinkTransform("arm_left_knife_c_blade");
  auto addBox = [&](const std::string & id)
  {
    scene->getWorldNonConst()->addToObject(id, std::make_shared<shapes::Box>(0.02, 0.02, 0.02), blade);
  };
  const std::vector<double> home(6, 0.0);
  const std::vector<double> right_home(6, 0.0);
  auto path = [&](bool flag)
  {
    SampledPath p = dual_cut_test::pathOf(left, {home});
    p.samples[0].contact_allowed = flag;
    return p;
  };
  auto run = [&](const std::string & /*object_id*/, bool flag, const std::vector<std::string> & links)
  {
    WorldCheckConfig cfg;
    cfg.contact_object_ids = {"detected_carton"};
    cfg.contact_link_names = links;
    std::unique_ptr<WorldChecker> w;
    EXPECT_TRUE(WorldChecker::create(scene, left, right, cfg, w));
    return w->check(path(flag), right_home);
  };
  const std::vector<std::string> contact_links = {
    "arm_left_knife_c_blade", "arm_left_xeg32_tool_tip", "arm_left_knife_c_holder", "arm_left_knife_c_handpiece"};

  addBox("detected_carton");
  EXPECT_EQ(run("detected_carton", false, contact_links).code, WorldCode::Collision);   // unflagged: checked
  const WorldCheckResult flagged = run("detected_carton", true, contact_links);
  // Flagged: the blade (and the tool links) may touch the carton. Other links of the arm may still hit it
  // only if the box reaches them; the box is 2 cm around the blade, so those are not involved.
  EXPECT_TRUE(flagged.safe()) << flagged.message;
  // Without the blade in the allowance list the same flagged stroke is refused (this is the failure that
  // the allowance is there to avoid).
  EXPECT_EQ(run("detected_carton", true, {"arm_left_xeg32_tool_tip"}).code, WorldCode::Collision);

  // The same box as the camera pole is never allowed, flagged or not.
  scene->getWorldNonConst()->removeObject("detected_carton");
  addBox("camera_pole");
  EXPECT_EQ(run("camera_pole", true, contact_links).code, WorldCode::Collision);
  scene->getWorldNonConst()->removeObject("camera_pole");
  addBox("camera_head");
  EXPECT_EQ(run("camera_head", true, contact_links).code, WorldCode::Collision);
}
