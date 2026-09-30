// servo_proximity_probe: READ-ONLY reproduction of MoveIt Servo's (Humble 2.5.10)
// CollisionCheck::run() distance computation, plus a per-pair breakdown.
//
// robot_description / robot_description_semantic:
//   default: fetched once from the parameters of /move_group (--from-node <name> to change,
//            e.g. --from-node /servo_node), then set on this node's own parameters.
//   --from-node ""  : use parameters given to this node (--ros-args -p robot_description:=... ).
// It never publishes, never modifies the scene, never commands motion.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_env.h>

namespace pm = planning_scene_monitor;
namespace cd = collision_detection;

struct Opts
{
  std::string from_node = "/move_group";
  std::string group = "igus_rebel_arm";  // servo.yaml move_group_name
  std::string joint_topic = "joint_states";
  std::string scene_topic = "/monitored_planning_scene";  // servo default monitored_planning_scene_topic
  double self_thr = 0.01;                                 // servo.yaml self_collision_proximity_threshold
  double scene_thr = 0.02;                                // servo.yaml scene_collision_proximity_threshold
  int top = 5;
  int loop = 1;
  double period = 0.5;
  bool global = false;  // DistanceRequestType::GLOBAL instead of LIMITED
  double wait = 10.0;
};

static double scale(double dist, double thr)
{
  if (dist >= thr)
    return 1.0;
  return std::exp((-std::log(0.001) / thr) * (dist - thr));
}

struct Pair
{
  double d;
  std::string a, b;
  cd::BodyType ta, tb;
  Eigen::Vector3d pa, pb;
};

static const char* bt(cd::BodyType t)
{
  switch (t)
  {
    case cd::BodyType::ROBOT_LINK: return "link";
    case cd::BodyType::ROBOT_ATTACHED: return "attached";
    default: return "world";
  }
}

static std::vector<Pair> collect(const cd::DistanceResult& res)
{
  std::vector<Pair> v;
  for (const auto& kv : res.distances)
    for (const auto& r : kv.second)
      v.push_back({ r.distance, r.link_names[0], r.link_names[1], r.body_types[0], r.body_types[1],
                    r.nearest_points[0], r.nearest_points[1] });
  std::sort(v.begin(), v.end(), [](const Pair& x, const Pair& y) { return x.d < y.d; });
  return v;
}

static void printPairs(const std::string& title, const cd::DistanceResult& res, const cd::AllowedCollisionMatrix& acm,
                       int top)
{
  auto v = collect(res);
  std::cout << "  " << title << ": in_collision=" << res.collision << " min_distance=";
  if (res.minimum_distance.distance > 1e9)
    std::cout << "inf";
  else
    std::cout << std::fixed << std::setprecision(2) << res.minimum_distance.distance * 1000 << " mm ("
              << res.minimum_distance.link_names[0] << " <-> " << res.minimum_distance.link_names[1] << ")";
  std::cout << "  pairs_reported=" << v.size() << "\n";
  for (int i = 0; i < top && i < static_cast<int>(v.size()); ++i)
  {
    cd::AllowedCollision::Type t;
    bool has = acm.getEntry(v[i].a, v[i].b, t);
    std::cout << "    #" << i + 1 << " " << std::setw(9) << std::fixed << std::setprecision(2) << v[i].d * 1000
              << " mm  " << v[i].a << "[" << bt(v[i].ta) << "] <-> " << v[i].b << "[" << bt(v[i].tb) << "]"
              << "  ACM=" << (has ? (t == cd::AllowedCollision::NEVER ? "NEVER(checked)" :
                                     t == cd::AllowedCollision::ALWAYS ? "ALWAYS(allowed)" : "CONDITIONAL") :
                                     "no-entry(checked)")
              << "  pt0=(" << std::setprecision(3) << v[i].pa.x() << "," << v[i].pa.y() << "," << v[i].pa.z()
              << ") pt1=(" << v[i].pb.x() << "," << v[i].pb.y() << "," << v[i].pb.z() << ")\n";
  }
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  Opts o;
  auto args = rclcpp::remove_ros_arguments(argc, argv);
  for (size_t i = 1; i < args.size(); ++i)
  {
    auto next = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : ""; };
    const std::string& a = args[i];
    if (a == "--from-node") o.from_node = next();
    else if (a == "--group") o.group = next();
    else if (a == "--joint-topic") o.joint_topic = next();
    else if (a == "--scene-topic") o.scene_topic = next();
    else if (a == "--self-threshold") o.self_thr = std::stod(next());
    else if (a == "--scene-threshold") o.scene_thr = std::stod(next());
    else if (a == "--top") o.top = std::stoi(next());
    else if (a == "--loop") o.loop = std::stoi(next());
    else if (a == "--period") o.period = std::stod(next());
    else if (a == "--wait") o.wait = std::stod(next());
    else if (a == "--global") o.global = true;
    else
    {
      std::cerr << "usage: servo_proximity_probe [--from-node /move_group|/servo_node|\"\"] [--group G] [--top N]\n"
                   "  [--loop N] [--period S] [--global] [--self-threshold m] [--scene-threshold m]\n"
                   "  [--joint-topic T] [--scene-topic T] [--wait S]\n";
      return a == "-h" || a == "--help" ? 0 : 2;
    }
  }

  rclcpp::NodeOptions no;
  no.automatically_declare_parameters_from_overrides(true);
  no.allow_undeclared_parameters(true);
  auto node = std::make_shared<rclcpp::Node>("servo_proximity_probe", no);
  auto log = node->get_logger();

  if (!o.from_node.empty())
  {
    auto pc = std::make_shared<rclcpp::SyncParametersClient>(node, o.from_node);
    if (!pc->wait_for_service(std::chrono::seconds(10)))
    {
      RCLCPP_ERROR(log, "parameter service of %s not available", o.from_node.c_str());
      return 1;
    }
    for (const char* n : { "robot_description", "robot_description_semantic" })
    {
      auto ps = pc->get_parameters({ n });
      if (ps.empty() || ps[0].get_type() != rclcpp::ParameterType::PARAMETER_STRING || ps[0].as_string().empty())
      {
        RCLCPP_ERROR(log, "%s has no string parameter '%s'", o.from_node.c_str(), n);
        return 1;
      }
      node->set_parameter(rclcpp::Parameter(n, ps[0].as_string()));
    }
  }

  // Same construction as servo_node.cpp (except no publishing / no world-geometry monitor / no scene service).
  auto psm = std::make_shared<pm::PlanningSceneMonitor>(node, "robot_description", "planning_scene_monitor");
  if (!psm->getPlanningScene())
  {
    RCLCPP_ERROR(log, "planning scene not configured (bad robot_description?)");
    return 1;
  }
  psm->startStateMonitor(o.joint_topic);
  psm->startSceneMonitor(o.scene_topic);
  psm->getStateMonitor()->enableCopyDynamics(true);
  bool got = psm->requestPlanningSceneState("/get_planning_scene");
  std::cout << "requestPlanningSceneState(/get_planning_scene): " << (got ? "ok" : "FAILED") << "\n";

  rclcpp::executors::MultiThreadedExecutor ex;
  ex.add_node(node);
  std::thread spin([&]() { ex.spin(); });

  int rc = 0;
  auto t0 = node->now();
  while (rclcpp::ok() && !psm->getStateMonitor()->waitForCompleteState(o.group, 1.0))
    if ((node->now() - t0).seconds() > o.wait)
    {
      RCLCPP_ERROR(log, "no complete joint state on %s within %.0f s", o.joint_topic.c_str(), o.wait);
      rc = 1;
      break;
    }
  // Let the scene topic deliver at least one update (move_group publishes at 2 Hz).
  if (rc == 0)
    rclcpp::sleep_for(std::chrono::milliseconds(1500));

  for (int it = 0; rc == 0 && rclcpp::ok() && it < o.loop; ++it)
  {
    auto state = psm->getStateMonitor()->getCurrentState();
    state->updateCollisionBodyTransforms();
    pm::LockedPlanningSceneRO scene(psm);
    const auto& acm = scene->getAllowedCollisionMatrix();
    auto model = state->getRobotModel();

    std::cout << "\n===== sample " << it + 1 << "/" << o.loop << " =====\n";
    if (it == 0)
    {
      std::cout << "robot model: " << model->getName() << "  root=" << model->getRootLinkName()
                << "  planning group: " << o.group
                << (model->hasJointModelGroup(o.group) ? "" : "  (NOT FOUND in model!)") << "\n";
      auto srdf = model->getSRDF();
      std::cout << "SRDF disabled collision pairs: " << srdf->getDisabledCollisionPairs().size()
                << "   ACM entries (scene): " << [&] { std::vector<std::string> n; acm.getAllEntryNames(n); return n.size(); }()
                << "\n";
      std::cout << "links (" << model->getLinkModelNames().size() << "):";
      for (const auto& l : model->getLinkModelNames()) std::cout << " " << l;
      std::cout << "\nworld objects:";
      auto ids = scene->getWorld()->getObjectIds();
      if (ids.empty()) std::cout << " (none)";
      for (const auto& id : ids)
      {
        auto obj = scene->getWorld()->getObject(id);
        std::cout << "\n  - " << id << " shapes=" << obj->shapes_.size();
        if (!obj->shape_poses_.empty())
        {
          auto p = obj->pose_.translation();
          std::cout << " origin=(" << std::fixed << std::setprecision(3) << p.x() << "," << p.y() << "," << p.z()
                    << ")";
        }
      }
      std::cout << "\nattached bodies:";
      std::vector<const moveit::core::AttachedBody*> ab;
      state->getAttachedBodies(ab);
      if (ab.empty()) std::cout << " (none)";
      for (auto* b : ab) std::cout << " " << b->getName() << "(on " << b->getAttachedLinkName() << ")";
      std::cout << "\n";
    }

    // --- 1. Exactly what Servo does ---------------------------------------------------------------
    cd::CollisionRequest creq;
    creq.group_name = o.group;
    creq.distance = true;
    creq.contacts = true;
    cd::CollisionResult cres;
    scene->getCollisionEnv()->checkRobotCollision(creq, cres, *state, acm);
    double scene_d = cres.distance;
    bool coll = cres.collision;
    std::cout << "[Servo-equivalent] world (padded env): distance=" << scene_d * 1000 << " mm collision=" << cres.collision;
    for (auto& c : cres.contacts) std::cout << " contact(" << c.first.first << "," << c.first.second << ")";
    std::cout << "\n";
    cres.clear();
    scene->getCollisionEnvUnpadded()->checkSelfCollision(creq, cres, *state, acm);
    double self_d = cres.distance;
    coll |= cres.collision;
    std::cout << "[Servo-equivalent] self (unpadded env): distance=" << self_d * 1000 << " mm collision=" << cres.collision;
    for (auto& c : cres.contacts) std::cout << " contact(" << c.first.first << "," << c.first.second << ")";
    std::cout << "\n";
    double vs = 1.0;
    if (coll) vs = 0.0;
    else
    {
      if (scene_d < o.scene_thr) vs = std::min(vs, scale(scene_d, o.scene_thr));
      if (self_d < o.self_thr) vs = std::min(vs, scale(self_d, o.self_thr));
    }
    std::cout << "[Servo-equivalent] implied collision_velocity_scale = " << std::setprecision(4) << vs
              << "  (thresholds self " << o.self_thr << " scene " << o.scene_thr << ")\n";
    if (vs < 1.0)
      std::cout << "   limiting: " << (scene_d < o.scene_thr && scale(scene_d, o.scene_thr) <= scale(self_d, o.self_thr) ? "WORLD/scene" : "SELF") << "\n";

    // --- 2. Per-pair distances ------------------------------------------------------------------------
    cd::DistanceRequest dreq;
    dreq.enable_nearest_points = true;
    dreq.type = o.global ? cd::DistanceRequestType::GLOBAL : cd::DistanceRequestType::LIMITED;
    dreq.max_contacts_per_body = 1;
    dreq.group_name = o.group;
    dreq.enableGroup(model);
    dreq.acm = &acm;
    std::cout << "[DistanceRequest] type=" << (o.global ? "GLOBAL" : "LIMITED") << " nearest_points=1 acm=scene\n";
    cd::DistanceResult dres;
    scene->getCollisionEnvUnpadded()->distanceSelf(dreq, dres, *state);
    printPairs("self (unpadded)", dres, acm, o.top);
    dres.clear();
    scene->getCollisionEnv()->distanceRobot(dreq, dres, *state);
    printPairs("world (padded)", dres, acm, o.top);
    dres.clear();
    scene->getCollisionEnvUnpadded()->distanceRobot(dreq, dres, *state);
    printPairs("world (unpadded)", dres, acm, o.top);

    if (it + 1 < o.loop)
      rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(o.period)));
  }

  ex.cancel();
  spin.join();
  rclcpp::shutdown();
  return rc;
}
