// Pure selection of the world objects that the executor's throwaway probe scenes copy from move_group's scene.
//
// The orientation-search probes of box_h_cut_executor (adaptive / parallel / varying left orientation, analytic IK
// branch filtering, hover and approach pose selection) build their own planning_scene::PlanningScene and used to
// add only the carton and the camera pole. The MTC tasks, however, start from move_group's scene, which also holds
// every other collision object (in the dual cell: the table, camera_head). A pose the probes accept could therefore
// be rejected later (GOAL_STATE_INVALID) because the probes never saw the object. 2026-10-22: the dual cell's table
// next to the right holder. This header only decides WHICH objects are copied; the executor fetches them with
// get_planning_scene and fails closed if it cannot.
//
// The carton (added by the probe itself from the snapshot) and the camera pole (added by the probe itself from the
// latched topic) are never copied, and the carton's scoped contact allowance is never extended to a copied object:
// it is an ACM entry between the contact links and the carton id only.
#pragma once

#include <moveit/collision_detection/collision_matrix.h>
#include <moveit_msgs/msg/collision_object.hpp>

#include <functional>

#include <set>
#include <string>
#include <vector>

namespace probe_scene
{
// Objects of `scene_objects` to add to a probe scene, in the scene's order, each as an ADD of the full geometry.
// Skipped: ids in `skip_ids`, empty ids, duplicates (first wins), objects without any geometry, and objects whose
// stored operation is not ADD (a get_planning_scene answer only holds ADD objects; anything else is not geometry).
inline std::vector<moveit_msgs::msg::CollisionObject> selectObjects(
  const std::vector<moveit_msgs::msg::CollisionObject>& scene_objects, const std::set<std::string>& skip_ids)
{
  std::vector<moveit_msgs::msg::CollisionObject> selected;
  std::set<std::string> seen;
  for (const auto& object : scene_objects) {
    if (object.id.empty() || skip_ids.count(object.id) != 0U || seen.count(object.id) != 0U) continue;
    if (object.operation != moveit_msgs::msg::CollisionObject::ADD) continue;
    if (object.primitives.empty() && object.meshes.empty() && object.planes.empty()) continue;
    seen.insert(object.id);
    selected.push_back(object);
  }
  return selected;
}

// Copies the ACM entries "copied object x robot link = ALWAYS" of move_group's scene into a probe scene's ACM
// (e.g. the padded table vs the arm's base link, written by table_collision.py). Only pairs of a copied object with
// a robot link (is_robot_link) are taken: nothing about the carton (the probe adds it itself and sets its scoped
// contact allowance on its own), the camera pole or any other name, and no "not allowed" entry. Returns the count.
inline std::size_t copyExemptions(
  const collision_detection::AllowedCollisionMatrix& monitored,
  const std::vector<moveit_msgs::msg::CollisionObject>& objects,
  const std::function<bool(const std::string&)>& is_robot_link, collision_detection::AllowedCollisionMatrix& probe)
{
  std::size_t copied = 0;
  std::vector<std::string> names;
  monitored.getAllEntryNames(names);
  for (const auto& object : objects) {
    for (const auto& other : names) {
      if (other == object.id || !is_robot_link(other)) continue;
      collision_detection::AllowedCollision::Type type;
      if (monitored.getEntry(object.id, other, type) && type == collision_detection::AllowedCollision::ALWAYS) {
        probe.setEntry(object.id, other, true);
        ++copied;
      }
    }
  }
  return copied;
}

inline std::string idList(const std::vector<moveit_msgs::msg::CollisionObject>& objects)
{
  std::string text;
  for (const auto& object : objects) text += (text.empty() ? "" : ", ") + object.id;
  return text.empty() ? std::string("none") : text;
}
}  // namespace probe_scene
