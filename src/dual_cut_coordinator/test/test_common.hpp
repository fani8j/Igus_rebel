// Shared helpers for the dual_cut_coordinator gtests.
#pragma once

#include <memory>
#include <string>

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <srdfdom/model.h>
#include <urdf/model.h>

#include "dual_cut_coordinator/swept_volume.hpp"

namespace dual_cut_test
{

inline moveit::core::RobotModelPtr buildModel(const std::string & urdf_xml, const std::string & srdf_xml)
{
  auto u = std::make_shared<urdf::Model>();
  if (!u->initString(urdf_xml)) return nullptr;
  auto s = std::make_shared<srdf::Model>();
  if (!s->initString(*u, srdf_xml)) return nullptr;
  return std::make_shared<moveit::core::RobotModel>(urdf::ModelInterfaceSharedPtr(u), s);
}

inline std::string readFile(const std::string & path)
{
  std::FILE * f = std::fopen(path.c_str(), "rb");
  if (!f) return std::string();
  std::string out;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  std::fclose(f);
  return out;
}

inline moveit::core::RobotModelPtr syntheticModel()
{
  const std::string dir = DUAL_CUT_TEST_DATA_DIR;
  return buildModel(readFile(dir + "/synthetic_dual.urdf"), readFile(dir + "/synthetic_dual.srdf"));
}

/// Single configuration as a one-sample path for an arm (joint order = arm.joint_names).
inline dual_cut::SampledPath pathOf(
  const dual_cut::ArmSpec & arm, const std::vector<std::vector<double>> & qs)
{
  dual_cut::SampledPath p;
  p.joint_names = arm.joint_names;
  double t = 0.0;
  for (const auto & q : qs)
  {
    dual_cut::PathSample s;
    s.time_s = t;
    t += 0.05;
    s.q = q;
    p.samples.push_back(s);
  }
  return p;
}

}  // namespace dual_cut_test
