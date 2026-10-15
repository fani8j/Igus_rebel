// Measures the error of MoveIt's FCL cross-arm distance on the installed real dual model against an
// independent brute-force reference: the exact triangle-triangle distance on the collision geometry
// (meshes as stored, primitives tessellated finely).
//
//   fcl_accuracy [--samples N] [--seed S] [--csv out.csv] [--tess-mm 0.05] [--near-fraction 0.5]
//                [--urdf plain.urdf --srdf model.srdf] [--inventory] [--grid]
//
// Tessellation: boxes are exact (12 triangles). Spheres, cylinders and cones are tessellated with
// vertices ON the true surface (inscribed polytope), so the polytope is inside the true shape; the
// per-shape sagitta bound is accumulated and printed ("tessellation_bound_mm"). The reference is
// therefore >= the true distance by at most the sum of the two shapes' sagittas.
//
// Error convention: err = max(fcl, 0) - max(ref, 0). Positive = FCL says "farther than the reference"
// (the unsafe direction). A pair counts as "penetration missed" if ref <= 0 < fcl.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometric_shapes/shapes.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <srdfdom/model.h>
#include <urdf/model.h>

#include "dual_cut_coordinator/swept_volume.hpp"

using Vec = Eigen::Vector3d;

namespace
{

struct Tri
{
  Vec a, b, c;
};

struct LocalShape
{
  std::string type;
  std::vector<Vec> verts;           // link-frame vertices (after the collision origin)
  std::vector<std::array<int, 3>> tris;
  double sagitta = 0.0;             // tessellation bound of this shape
};

struct LinkGeom
{
  const moveit::core::LinkModel * link = nullptr;
  std::vector<LocalShape> shapes;
};

// ---- tessellation -------------------------------------------------------------------------

void addBox(const shapes::Box & b, LocalShape & out)
{
  const double x = 0.5 * b.size[0], y = 0.5 * b.size[1], z = 0.5 * b.size[2];
  for (int i = 0; i < 8; ++i)
    out.verts.emplace_back((i & 1 ? x : -x), (i & 2 ? y : -y), (i & 4 ? z : -z));
  const int f[12][3] = {{0, 1, 3}, {0, 3, 2}, {4, 6, 7}, {4, 7, 5}, {0, 4, 5}, {0, 5, 1},
                        {2, 3, 7}, {2, 7, 6}, {0, 2, 6}, {0, 6, 4}, {1, 5, 7}, {1, 7, 3}};
  for (auto & t : f) out.tris.push_back({t[0], t[1], t[2]});
}

// Segments so that the sagitta r (1 - cos(pi / n)) <= tol.
int segmentsFor(double radius, double tol)
{
  if (radius <= tol) return 8;
  const int n = static_cast<int>(std::ceil(M_PI / std::acos(std::max(0.0, 1.0 - tol / radius))));
  return std::max(8, std::min(n, 4000));
}

void addPrism(double r, double h, bool cone, double tol, LocalShape & out)
{
  const int n = segmentsFor(r, tol);
  out.sagitta = r * (1.0 - std::cos(M_PI / n));
  const int base = static_cast<int>(out.verts.size());
  for (int i = 0; i < n; ++i)
  {
    const double a = 2.0 * M_PI * i / n;
    out.verts.emplace_back(r * std::cos(a), r * std::sin(a), -0.5 * h);
    if (!cone) out.verts.emplace_back(r * std::cos(a), r * std::sin(a), 0.5 * h);
  }
  const int bc = static_cast<int>(out.verts.size());
  out.verts.emplace_back(0, 0, -0.5 * h);
  const int tc = static_cast<int>(out.verts.size());
  out.verts.emplace_back(0, 0, 0.5 * h);
  const int stride = cone ? 1 : 2;
  for (int i = 0; i < n; ++i)
  {
    const int j = (i + 1) % n;
    const int b0 = base + i * stride, b1 = base + j * stride;
    out.tris.push_back({bc, b1, b0});
    if (cone)
    {
      out.tris.push_back({b0, b1, tc});
    }
    else
    {
      const int t0 = b0 + 1, t1 = b1 + 1;
      out.tris.push_back({tc, t0, t1});
      out.tris.push_back({b0, b1, t1});
      out.tris.push_back({b0, t1, t0});
    }
  }
}

void addSphere(double r, double tol, LocalShape & out)
{
  // Icosphere, subdivided until the edge sagitta bound r (1 - cos(theta/2)) <= tol.
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  std::vector<Vec> v{{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                     {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (auto & p : v) p.normalize();
  std::vector<std::array<int, 3>> f{{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
                                    {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                    {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
                                    {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
  auto sag = [&]()
  {
    double e = 0.0;
    for (auto & t3 : f)
      for (int k = 0; k < 3; ++k) e = std::max(e, (v[t3[k]] - v[t3[(k + 1) % 3]]).norm());
    return r * (1.0 - std::sqrt(std::max(0.0, 1.0 - 0.25 * e * e)));  // chord e on a unit sphere
  };
  while (sag() > tol)
  {
    std::map<std::pair<int, int>, int> mid;
    auto midpoint = [&](int a, int b)
    {
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      auto it = mid.find(key);
      if (it != mid.end()) return it->second;
      v.push_back(((v[a] + v[b]) * 0.5).normalized());
      return mid[key] = static_cast<int>(v.size()) - 1;
    };
    std::vector<std::array<int, 3>> g;
    for (auto & t3 : f)
    {
      const int a = midpoint(t3[0], t3[1]), b = midpoint(t3[1], t3[2]), c = midpoint(t3[2], t3[0]);
      g.push_back({t3[0], a, c});
      g.push_back({t3[1], b, a});
      g.push_back({t3[2], c, b});
      g.push_back({a, b, c});
    }
    f = g;
  }
  out.sagitta = sag();
  for (auto & p : v) out.verts.push_back(p * r);
  out.tris = f;
}

bool tessellate(const shapes::Shape & s, double tol, LocalShape & out)
{
  switch (s.type)
  {
    case shapes::BOX:
      out.type = "box";
      addBox(static_cast<const shapes::Box &>(s), out);
      return true;
    case shapes::SPHERE:
      out.type = "sphere";
      addSphere(static_cast<const shapes::Sphere &>(s).radius, tol, out);
      return true;
    case shapes::CYLINDER:
    {
      out.type = "cylinder";
      const auto & c = static_cast<const shapes::Cylinder &>(s);
      addPrism(c.radius, c.length, false, tol, out);
      return true;
    }
    case shapes::CONE:
    {
      out.type = "cone";
      const auto & c = static_cast<const shapes::Cone &>(s);
      addPrism(c.radius, c.length, true, tol, out);
      return true;
    }
    case shapes::MESH:
    {
      out.type = "mesh";
      const auto & m = static_cast<const shapes::Mesh &>(s);
      for (unsigned i = 0; i < m.vertex_count; ++i)
        out.verts.emplace_back(m.vertices[3 * i], m.vertices[3 * i + 1], m.vertices[3 * i + 2]);
      for (unsigned i = 0; i < m.triangle_count; ++i)
        out.tris.push_back({static_cast<int>(m.triangles[3 * i]), static_cast<int>(m.triangles[3 * i + 1]),
                            static_cast<int>(m.triangles[3 * i + 2])});
      return true;
    }
    default:
      return false;
  }
}

// ---- exact triangle-triangle distance ------------------------------------------------------

Vec closestPointOnTriangle(const Vec & p, const Vec & a, const Vec & b, const Vec & c)
{
  // Ericson, Real-Time Collision Detection 5.1.5
  const Vec ab = b - a, ac = c - a, ap = p - a;
  const double d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const Vec bp = p - b;
  const double d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
  const Vec cp = p - c;
  const double d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  const double denom = 1.0 / (va + vb + vc);
  return a + ab * (vb * denom) + ac * (vc * denom);
}

double segSegDist2(const Vec & p1, const Vec & q1, const Vec & p2, const Vec & q2)
{
  const Vec d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
  const double a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
  double s, t;
  const double eps = 1e-18;
  if (a <= eps && e <= eps) return r.squaredNorm();
  if (a <= eps)
  {
    s = 0;
    t = std::clamp(f / e, 0.0, 1.0);
  }
  else
  {
    const double c = d1.dot(r);
    if (e <= eps)
    {
      t = 0;
      s = std::clamp(-c / a, 0.0, 1.0);
    }
    else
    {
      const double b = d1.dot(d2), denom = a * e - b * b;
      s = denom > eps ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
      t = (b * s + f) / e;
      if (t < 0)
      {
        t = 0;
        s = std::clamp(-c / a, 0.0, 1.0);
      }
      else if (t > 1)
      {
        t = 1;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }
  return ((p1 + d1 * s) - (p2 + d2 * t)).squaredNorm();
}

bool segTriIntersect(const Vec & p, const Vec & q, const Tri & t)
{
  const Vec e1 = t.b - t.a, e2 = t.c - t.a, d = q - p;
  const Vec h = d.cross(e2);
  const double det = e1.dot(h);
  if (std::abs(det) < 1e-15) return false;  // parallel: edge cases are caught by the distances
  const double inv = 1.0 / det;
  const Vec s = p - t.a;
  const double u = inv * s.dot(h);
  if (u < 0 || u > 1) return false;
  const Vec qv = s.cross(e1);
  const double v = inv * d.dot(qv);
  if (v < 0 || u + v > 1) return false;
  const double w = inv * e2.dot(qv);
  return w >= 0 && w <= 1;
}

double triTriDist2(const Tri & x, const Tri & y)
{
  const Vec xv[3] = {x.a, x.b, x.c}, yv[3] = {y.a, y.b, y.c};
  for (int i = 0; i < 3; ++i)
    if (segTriIntersect(xv[i], xv[(i + 1) % 3], y) || segTriIntersect(yv[i], yv[(i + 1) % 3], x)) return 0.0;
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 3; ++i)
  {
    best = std::min(best, (xv[i] - closestPointOnTriangle(xv[i], y.a, y.b, y.c)).squaredNorm());
    best = std::min(best, (yv[i] - closestPointOnTriangle(yv[i], x.a, x.b, x.c)).squaredNorm());
    for (int j = 0; j < 3; ++j)
      best = std::min(best, segSegDist2(xv[i], xv[(i + 1) % 3], yv[j], yv[(j + 1) % 3]));
  }
  return best;
}

struct WorldShape
{
  std::string link, type;
  std::vector<Tri> tris;
  std::vector<Vec> tri_lo, tri_hi;
  Vec lo, hi;
};

double boxDist(const Vec & alo, const Vec & ahi, const Vec & blo, const Vec & bhi)
{
  Vec d = Vec::Zero();
  for (int k = 0; k < 3; ++k) d[k] = std::max({0.0, alo[k] - bhi[k], blo[k] - ahi[k]});
  return d.norm();
}

}  // namespace

int main(int argc, char ** argv)
{
  int samples = 2000;
  unsigned seed = 1;
  std::string csv;
  double tess_mm = 0.05, near_fraction = 0.5;
  bool inventory_only = false, grid = false;
  std::string urdf_path, srdf_path;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--samples") samples = std::atoi(next().c_str());
    else if (a == "--seed") seed = static_cast<unsigned>(std::atoi(next().c_str()));
    else if (a == "--csv") csv = next();
    else if (a == "--tess-mm") tess_mm = std::atof(next().c_str());
    else if (a == "--near-fraction") near_fraction = std::atof(next().c_str());
    else if (a == "--inventory") inventory_only = true;
    else if (a == "--grid") grid = true;  // joints from {-pi/2, 0, pi/2, pi}: axis-aligned, degeneracy-prone
    else if (a == "--urdf") urdf_path = next();   // plain URDF instead of the installed xacro
    else if (a == "--srdf") srdf_path = next();
  }

  // ---- model (the installed real dual model) ----
  const std::string desc = ament_index_cpp::get_package_share_directory("rebel_dual_table_description");
  const std::string moveit_cfg = ament_index_cpp::get_package_share_directory("rebel_dual_table_moveit_config");
  std::string urdf_xml;
  if (!urdf_path.empty())
  {
    std::ifstream f(urdf_path);
    std::stringstream b;
    b << f.rdbuf();
    urdf_xml = b.str();
  }
  else
  {
    const std::string cmd =
      "xacro " + desc + "/urdf/rebel_dual_table.urdf.xacro include_ros2_control:=false 2>/dev/null";
    FILE * p = popen(cmd.c_str(), "r");
    char buf[4096];
    size_t n;
    while (p && (n = fread(buf, 1, sizeof(buf), p)) > 0) urdf_xml.append(buf, n);
    if (p) pclose(p);
  }
  std::ifstream sf(srdf_path.empty() ? moveit_cfg + "/config/rebel_dual_table.srdf" : srdf_path);
  std::stringstream ss;
  ss << sf.rdbuf();
  auto u = std::make_shared<urdf::Model>();
  if (urdf_xml.empty() || !u->initString(urdf_xml))
  {
    std::fprintf(stderr, "cannot load the URDF\n");
    return 2;
  }
  auto sm = std::make_shared<srdf::Model>();
  sm->initString(*u, ss.str());
  auto model = std::make_shared<moveit::core::RobotModel>(urdf::ModelInterfaceSharedPtr(u), sm);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);

  dual_cut::ArmSpec left, right;
  if (!dual_cut::makeArmSpec(*model, "left", left) || !dual_cut::makeArmSpec(*model, "right", right)) return 2;
  std::unique_ptr<dual_cut::SweptVolumeChecker> checker;
  dual_cut::SweptVolumeConfig cfg;
  const auto st = dual_cut::SweptVolumeChecker::create(scene, left, right, cfg, checker);
  if (!st)
  {
    std::fprintf(stderr, "checker: %s\n", st.message.c_str());
    return 2;
  }

  // ---- tessellated cross-arm geometry ----
  const double tol = tess_mm * 1e-3;
  std::vector<LinkGeom> geoms[2];
  std::map<std::string, int> inventory;
  double max_sagitta = 0.0;
  for (int arm = 0; arm < 2; ++arm)
  {
    const auto & names = arm == 0 ? checker->linksA() : checker->linksB();
    for (const auto & name : names)
    {
      LinkGeom g;
      g.link = model->getLinkModel(name);
      const auto & shapes_v = g.link->getShapes();
      const auto & origins = g.link->getCollisionOriginTransforms();
      for (size_t k = 0; k < shapes_v.size(); ++k)
      {
        LocalShape ls;
        if (!tessellate(*shapes_v[k], tol, ls))
        {
          std::fprintf(stderr, "unsupported shape on %s\n", name.c_str());
          return 2;
        }
        for (auto & v : ls.verts) v = origins[k] * v;
        inventory[ls.type]++;
        max_sagitta = std::max(max_sagitta, ls.sagitta);
        std::printf("shape %-40s %-9s verts=%zu tris=%zu sagitta=%.4f mm\n", name.c_str(), ls.type.c_str(),
                    ls.verts.size(), ls.tris.size(), ls.sagitta * 1e3);
        g.shapes.push_back(std::move(ls));
      }
      geoms[arm].push_back(std::move(g));
    }
  }
  std::printf("inventory:");
  for (auto & kv : inventory) std::printf(" %s=%d", kv.first.c_str(), kv.second);
  std::printf("\ntessellation_bound_mm (sum of the two largest sagittas) = %.4f\n", 2e3 * max_sagitta);
  if (inventory_only) return 0;

  // joint limits
  std::vector<std::pair<double, double>> lim[2];
  std::vector<int> vidx[2];
  for (int arm = 0; arm < 2; ++arm)
    for (const auto & j : (arm == 0 ? left : right).joint_names)
    {
      const auto & b = model->getVariableBounds(j);
      lim[arm].push_back({std::max(b.min_position_, -M_PI), std::min(b.max_position_, M_PI)});
    }

  std::mt19937 rng(seed);
  auto randq = [&](int arm)
  {
    std::vector<double> q;
    for (auto & l : lim[arm])
    {
      if (grid)
      {
        static const double g[4] = {-M_PI / 2, 0.0, M_PI / 2, M_PI};
        double v = g[rng() % 4];
        q.push_back(std::clamp(v, l.first, l.second));
      }
      else
      {
        q.push_back(std::uniform_real_distribution<double>(l.first, l.second)(rng));
      }
    }
    return q;
  };

  struct Row
  {
    double fcl, ref, err;
    std::string fcl_a, fcl_b, ref_a, ref_b, ref_ta, ref_tb;
    std::string category;
  };
  std::vector<Row> rows;
  const int want_near = static_cast<int>(samples * near_fraction);
  int near = 0, far = 0, tried = 0;
  moveit::core::RobotState state(model);
  auto t0 = std::chrono::steady_clock::now();
  while (near + far < samples && tried < samples * 400)
  {
    ++tried;
    const auto qa = randq(0), qb = randq(1);
    double d;
    std::string fa, fb;
    if (!checker->exactDistance(qa, qb, d, fa, fb)) continue;
    // bucket by FCL distance (bounds the cost; the reference decides the actual numbers)
    const bool is_near = d < 0.15;
    if (is_near && near >= want_near) continue;
    if (!is_near && far >= samples - want_near) continue;

    // world geometry for this state
    state.setToDefaultValues();
    for (size_t k = 0; k < qa.size(); ++k) state.setVariablePosition(left.joint_names[k], qa[k]);
    for (size_t k = 0; k < qb.size(); ++k) state.setVariablePosition(right.joint_names[k], qb[k]);
    state.update();
    std::vector<WorldShape> w[2];
    for (int arm = 0; arm < 2; ++arm)
      for (auto & g : geoms[arm])
      {
        const Eigen::Isometry3d T = state.getGlobalLinkTransform(g.link);
        for (auto & ls : g.shapes)
        {
          WorldShape ws;
          ws.link = g.link->getName();
          ws.type = ls.type;
          std::vector<Vec> wv;
          for (auto & v : ls.verts) wv.push_back(T * v);
          ws.lo = Vec::Constant(1e9);
          ws.hi = Vec::Constant(-1e9);
          for (auto & t3 : ls.tris)
          {
            Tri t{wv[t3[0]], wv[t3[1]], wv[t3[2]]};
            ws.tris.push_back(t);
            ws.tri_lo.push_back(t.a.cwiseMin(t.b).cwiseMin(t.c));
            ws.tri_hi.push_back(t.a.cwiseMax(t.b).cwiseMax(t.c));
            ws.lo = ws.lo.cwiseMin(ws.tri_lo.back());
            ws.hi = ws.hi.cwiseMax(ws.tri_hi.back());
          }
          w[arm].push_back(std::move(ws));
        }
      }
    double best2 = std::numeric_limits<double>::infinity();
    std::string ra, rb, rta, rtb;
    for (auto & sa : w[0])
      for (auto & sb : w[1])
      {
        const double bd = boxDist(sa.lo, sa.hi, sb.lo, sb.hi);
        if (bd * bd >= best2) continue;
        for (size_t i = 0; i < sa.tris.size(); ++i)
        {
          for (size_t j = 0; j < sb.tris.size(); ++j)
          {
            const double tb = boxDist(sa.tri_lo[i], sa.tri_hi[i], sb.tri_lo[j], sb.tri_hi[j]);
            if (tb * tb >= best2) continue;
            const double d2 = triTriDist2(sa.tris[i], sb.tris[j]);
            if (d2 < best2)
            {
              best2 = d2;
              ra = sa.link;
              rb = sb.link;
              rta = sa.type;
              rtb = sb.type;
            }
          }
        }
      }
    Row r;
    r.fcl = d;
    r.ref = std::sqrt(best2);
    r.err = std::max(d, 0.0) - std::max(r.ref, 0.0);
    r.fcl_a = fa;
    r.fcl_b = fb;
    r.ref_a = ra;
    r.ref_b = rb;
    r.ref_ta = rta;
    r.ref_tb = rtb;
    r.category = r.ref < 0.02 ? "contact_or_<2cm" : (r.ref < 0.15 ? "near_2_15cm" : "far");
    (is_near ? near : far)++;
    rows.push_back(std::move(r));
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  if (!csv.empty())
  {
    std::ofstream o(csv);
    o << "category,fcl_m,ref_m,err_m,fcl_link_a,fcl_link_b,ref_link_a,ref_link_b,ref_type_a,ref_type_b\n";
    for (auto & r : rows)
      o << r.category << ',' << r.fcl << ',' << r.ref << ',' << r.err << ',' << r.fcl_a << ',' << r.fcl_b << ','
        << r.ref_a << ',' << r.ref_b << ',' << r.ref_ta << ',' << r.ref_tb << '\n';
  }

  auto stats = [&](const char * label, const std::vector<const Row *> & sel)
  {
    if (sel.empty()) return;
    std::vector<double> e;
    for (auto * r : sel) e.push_back(r->err);
    std::sort(e.begin(), e.end());
    auto pct = [&](double p) { return e[std::min(e.size() - 1, static_cast<size_t>(p * e.size()))]; };
    int over5 = 0, over1 = 0, missed = 0;
    for (auto * r : sel)
    {
      over5 += r->err > 0.005;
      over1 += r->err > 0.001;
      missed += (r->ref <= 0.0 && r->fcl > 0.0);
    }
    std::printf("%-18s n=%5zu  err[mm]: min %+8.3f  p50 %+8.3f  p95 %+8.3f  p99 %+8.3f  max %+8.3f   >1mm: %d  >5mm: %d  penetration-missed: %d\n",
                label, sel.size(), 1e3 * e.front(), 1e3 * pct(0.5), 1e3 * pct(0.95), 1e3 * pct(0.99),
                1e3 * e.back(), over1, over5, missed);
  };
  std::printf("\nsamples %zu (tried %d) in %.1f s, seed %u, tessellation tol %.3f mm\n", rows.size(), tried, secs, seed, tess_mm);
  std::vector<const Row *> all, c1, c2, c3;
  for (auto & r : rows)
  {
    all.push_back(&r);
    (r.category[0] == 'c' ? c1 : r.category[0] == 'n' ? c2 : c3).push_back(&r);
  }
  stats("ALL", all);
  stats("contact_or_<2cm", c1);
  stats("near_2_15cm", c2);
  stats("far", c3);

  // worst overestimates and which reference shape pairs they come from
  std::vector<const Row *> sorted = all;
  std::sort(sorted.begin(), sorted.end(), [](const Row * a, const Row * b) { return a->err > b->err; });
  std::printf("\nworst 12 overestimates (FCL - reference):\n");
  for (size_t i = 0; i < std::min<size_t>(12, sorted.size()); ++i)
  {
    const Row & r = *sorted[i];
    std::printf("  err %+8.3f mm  fcl %.4f ref %.4f  ref pair %s(%s) / %s(%s)   fcl pair %s / %s\n", 1e3 * r.err, r.fcl,
                r.ref, r.ref_a.c_str(), r.ref_ta.c_str(), r.ref_b.c_str(), r.ref_tb.c_str(), r.fcl_a.c_str(), r.fcl_b.c_str());
  }
  std::map<std::string, std::pair<int, double>> by_pair;  // type pair -> (count of err>1mm, max err)
  std::map<std::string, std::pair<int, double>> by_type;
  for (auto * r : all)
  {
    std::string k = r->ref_ta + "/" + r->ref_tb;
    auto & e = by_type[k];
    e.first += r->err > 0.001;
    e.second = std::max(e.second, r->err);
    auto & p = by_pair[r->ref_a + " / " + r->ref_b];
    p.first += r->err > 0.001;
    p.second = std::max(p.second, r->err);
  }
  std::printf("\nby reference shape-type pair (count err>1mm, max err mm):\n");
  for (auto & kv : by_type) std::printf("  %-14s %5d  %+8.3f\n", kv.first.c_str(), kv.second.first, 1e3 * kv.second.second);
  std::vector<std::pair<std::string, std::pair<int, double>>> pv(by_pair.begin(), by_pair.end());
  std::sort(pv.begin(), pv.end(), [](auto & a, auto & b) { return a.second.second > b.second.second; });
  std::printf("\nlink pairs with the largest overestimate (top 12):\n");
  for (size_t i = 0; i < std::min<size_t>(12, pv.size()); ++i)
    std::printf("  %-70s  n>1mm %4d   max %+8.3f mm\n", pv[i].first.c_str(), pv[i].second.first, 1e3 * pv[i].second.second);
  return 0;
}
