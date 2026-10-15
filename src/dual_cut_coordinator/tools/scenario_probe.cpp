// Design aid for the fake-plan scenarios: tool-tip FK of both arms for given joint values and the
// cross-arm distance on the installed real dual model.
//   scenario_probe l1 .. l6 r1 .. r6
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <srdfdom/model.h>
#include <urdf/model.h>

#include "dual_cut_coordinator/swept_volume.hpp"

int main(int argc, char ** argv)
{
  if (argc < 13) { std::fprintf(stderr, "usage: scenario_probe l1..l6 r1..r6\n"); return 2; }
  const std::string desc = ament_index_cpp::get_package_share_directory("rebel_dual_table_description");
  const std::string cfg = ament_index_cpp::get_package_share_directory("rebel_dual_table_moveit_config");
  std::string xml;
  FILE * p = popen(("xacro " + desc + "/urdf/rebel_dual_table.urdf.xacro include_ros2_control:=false 2>/dev/null").c_str(), "r");
  char buf[4096]; size_t n;
  while (p && (n = fread(buf, 1, sizeof(buf), p)) > 0) xml.append(buf, n);
  if (p) pclose(p);
  std::ifstream sf(cfg + "/config/rebel_dual_table.srdf");
  std::stringstream ss; ss << sf.rdbuf();
  auto u = std::make_shared<urdf::Model>(); u->initString(xml);
  auto s = std::make_shared<srdf::Model>(); s->initString(*u, ss.str());
  auto model = std::make_shared<moveit::core::RobotModel>(urdf::ModelInterfaceSharedPtr(u), s);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  dual_cut::ArmSpec l, r;
  dual_cut::makeArmSpec(*model, "left", l); dual_cut::makeArmSpec(*model, "right", r);
  std::vector<double> ql, qr;
  for (int i = 0; i < 6; ++i) ql.push_back(std::atof(argv[1 + i]));
  for (int i = 0; i < 6; ++i) qr.push_back(std::atof(argv[7 + i]));
  moveit::core::RobotState st(model); st.setToDefaultValues();
  for (int i = 0; i < 6; ++i) { st.setVariablePosition(l.joint_names[i], ql[i]); st.setVariablePosition(r.joint_names[i], qr[i]); }
  st.update();
  for (auto * a : {&l, &r})
  {
    const auto T = st.getGlobalLinkTransform(a->prefix + "xeg32_tool_tip");
    std::printf("%s tool_tip pos %.4f %.4f %.4f  z-axis %.3f %.3f %.3f  x-axis %.3f %.3f %.3f\n", a->side.c_str(), T.translation().x(), T.translation().y(),
                T.translation().z(), T.linear()(0, 2), T.linear()(1, 2), T.linear()(2, 2), T.linear()(0, 0), T.linear()(1, 0), T.linear()(2, 0));
  }
  std::unique_ptr<dual_cut::SweptVolumeChecker> c;
  dual_cut::SweptVolumeConfig sc;
  if (dual_cut::SweptVolumeChecker::create(scene, l, r, sc, c))
  {
    double d; std::string a, b;
    c->exactDistance(ql, qr, d, a, b);
    std::printf("cross-arm distance %.4f m (%s / %s)\n", d, a.c_str(), b.c_str());
  }
  return 0;
}
