// Prints the 11 points + normal of a snapshot (read as "name x y z" lines on stdin) after
// toPlanningFrame(B^-1) + relabelRotate180, using the executor's own header (snapshot_frame.hpp).
// usage: snapshot_frame_check bx by bz roll pitch yaw < points.txt   (%.17g output)
#include <cstdio>
#include <iostream>
#include <map>
#include <string>

#include "igus_rebel_moveit_config/snapshot_frame.hpp"

int main(int argc, char ** argv)
{
  if (argc != 7) return 2;
  std::array<double, 3> xyz{std::stod(argv[1]), std::stod(argv[2]), std::stod(argv[3])};
  std::array<double, 3> rpy{std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6])};
  std::map<std::string, Eigen::Vector3d> in;
  std::string name;
  double x, y, z;
  while (std::cin >> name >> x >> y >> z) in[name] = Eigen::Vector3d(x, y, z);
  snapshot_frame::Points p;
  p.seam_start = in["seam_start"]; p.seam_end = in["seam_end"]; p.seam_center = in["seam_center"];
  p.back_left = in["back_left"]; p.back_right = in["back_right"];
  p.front_left = in["front_left"]; p.front_right = in["front_right"];
  p.back_middle_left = in["back_middle_left"]; p.front_middle_left = in["front_middle_left"];
  p.back_middle_right = in["back_middle_right"]; p.front_middle_right = in["front_middle_right"];
  p.surface_normal = in["surface_normal"];
  const auto planning = snapshot_frame::toPlanningFrame(p, snapshot_frame::poseFromXyzRpy(xyz, rpy));
  const auto out = snapshot_frame::relabelRotate180(planning);
  if (!snapshot_frame::sameHandedness(planning, out)) return 3;
  const auto emit = [](const char * n, const Eigen::Vector3d & v) {
    std::printf("%s %.17g %.17g %.17g\n", n, v.x(), v.y(), v.z());
  };
  emit("seam_start", out.seam_start); emit("seam_end", out.seam_end); emit("seam_center", out.seam_center);
  emit("back_left", out.back_left); emit("back_right", out.back_right);
  emit("front_left", out.front_left); emit("front_right", out.front_right);
  emit("back_middle_left", out.back_middle_left); emit("front_middle_left", out.front_middle_left);
  emit("back_middle_right", out.back_middle_right); emit("front_middle_right", out.front_middle_right);
  emit("surface_normal", out.surface_normal);
  return 0;
}
