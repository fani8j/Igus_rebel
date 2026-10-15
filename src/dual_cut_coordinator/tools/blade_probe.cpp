// WP7 diagnostic: where is the dual model's blade box relative to the carton at a given arm configuration?
//   blade_probe <plan_dump.json> <left|right> <segment_index> <first|last>
// plan_dump.json: {"arm_left": {"segs": [{"first": [6], "last": [6]}...]}, "carton": {"pos":[3],"quat":[4],"dims":[3]}}
// (carton given in the LEFT arm's scene = dual world). Prints the TCP in the carton frame, the blade box, and the
// FCL distance blade <-> carton (negative = penetration).
#include <cstdio>
#include <fstream>
#include <sstream>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometric_shapes/shapes.h>
#include <moveit/planning_scene/planning_scene.h>
#include <srdfdom/model.h>
#include <urdf/model.h>
#include <yaml-cpp/yaml.h>

#include "dual_cut_coordinator/swept_volume.hpp"

int main(int argc, char ** argv)
{
  if (argc < 5) return 2;
  YAML::Node d = YAML::LoadFile(argv[1]);
  const std::string side = argv[2];
  const size_t idx = std::stoul(argv[3]);
  const bool first = std::string(argv[4]) == "first";
  const std::string desc = ament_index_cpp::get_package_share_directory("rebel_dual_table_description");
  const std::string cfg = ament_index_cpp::get_package_share_directory("rebel_dual_table_moveit_config");
  std::string xml;
  FILE * p = popen(("xacro " + desc + "/urdf/rebel_dual_table.urdf.xacro include_ros2_control:=false 2>/dev/null").c_str(), "r");
  char buf[4096];
  size_t n;
  while (p && (n = fread(buf, 1, sizeof(buf), p)) > 0) xml.append(buf, n);
  if (p) pclose(p);
  std::ifstream sf(cfg + "/config/rebel_dual_table.srdf");
  std::stringstream ss;
  ss << sf.rdbuf();
  auto u = std::make_shared<urdf::Model>();
  u->initString(xml);
  auto s = std::make_shared<srdf::Model>();
  s->initString(*u, ss.str());
  auto model = std::make_shared<moveit::core::RobotModel>(urdf::ModelInterfaceSharedPtr(u), s);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  const auto c = d["carton"];
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(c["pos"][0].as<double>(), c["pos"][1].as<double>(), c["pos"][2].as<double>());
  T.linear() = Eigen::Quaterniond(c["quat"][3].as<double>(), c["quat"][0].as<double>(), c["quat"][1].as<double>(), c["quat"][2].as<double>()).toRotationMatrix();
  const Eigen::Vector3d dims(c["dims"][0].as<double>(), c["dims"][1].as<double>(), c["dims"][2].as<double>());
  scene->getWorldNonConst()->addToObject("detected_carton", std::make_shared<shapes::Box>(dims.x(), dims.y(), dims.z()), T);
  const auto q = d["arm_" + side]["segs"][idx][first ? "first" : "last"].as<std::vector<double>>();
  moveit::core::RobotState st(model);
  st.setToDefaultValues();
  for (size_t i = 0; i < 6; ++i) st.setVariablePosition("arm_" + side + "_joint" + std::to_string(i + 1), q[i]);
  st.update();
  const std::string blade = "arm_" + side + "_knife_c_blade";
  const Eigen::Isometry3d B = st.getGlobalLinkTransform(blade);
  const Eigen::Isometry3d tip = st.getGlobalLinkTransform("arm_" + side + "_xeg32_tool_tip");
  const auto * lm = model->getLinkModel(blade);
  const auto box = std::static_pointer_cast<const shapes::Box>(lm->getShapes()[0]);
  const Eigen::Isometry3d Cinv = T.inverse();
  const Eigen::Vector3d tc = Cinv * tip.translation();
  std::printf("TCP in carton frame: %.4f %.4f %.4f  (half extents %.4f %.4f %.4f) -> margin to faces x:%.4f y:%.4f z(top):%.4f\n", tc.x(), tc.y(), tc.z(),
              dims.x() / 2, dims.y() / 2, dims.z() / 2, dims.x() / 2 - std::abs(tc.x()), dims.y() / 2 - std::abs(tc.y()), dims.z() / 2 - tc.z());
  std::printf("blade box dims %.4f %.4f %.4f; origin offset in link %.4f %.4f %.4f\n", box->size[0], box->size[1], box->size[2],
              lm->getCollisionOriginTransforms()[0].translation().x(), lm->getCollisionOriginTransforms()[0].translation().y(), lm->getCollisionOriginTransforms()[0].translation().z());
  const Eigen::Isometry3d Bbox = B * lm->getCollisionOriginTransforms()[0];
  double min_z = 1e9;
  for (int i = 0; i < 8; ++i)
  {
    const Eigen::Vector3d corner((i & 1 ? 0.5 : -0.5) * box->size[0], (i & 2 ? 0.5 : -0.5) * box->size[1], (i & 4 ? 0.5 : -0.5) * box->size[2]);
    const Eigen::Vector3d cc = Cinv * (Bbox * corner);
    std::printf("  blade corner %d in carton frame: %.4f %.4f %.4f  (inside x:%d y:%d, above top by %.4f)\n", i, cc.x(), cc.y(), cc.z(),
                std::abs(cc.x()) < dims.x() / 2, std::abs(cc.y()) < dims.y() / 2, cc.z() - dims.z() / 2);
    min_z = std::min(min_z, cc.z());
  }
  collision_detection::AllowedCollisionMatrix acm;
  std::vector<std::string> all;
  for (auto * l : model->getLinkModels()) all.push_back(l->getName());
  acm.setEntry(all, all, true);
  acm.setEntry(blade, "detected_carton", false);
  collision_detection::DistanceRequest req;
  req.enable_nearest_points = true;
  req.enable_signed_distance = true;
  req.acm = &acm;
  collision_detection::DistanceResult res;
  scene->getCollisionEnvUnpadded()->distanceRobot(req, res, st);
  std::printf("FCL distance blade<->carton: %.6f m (%s / %s)\n", res.minimum_distance.distance, res.minimum_distance.link_names[0].c_str(), res.minimum_distance.link_names[1].c_str());
  return 0;
}
