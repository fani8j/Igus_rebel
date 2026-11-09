// probe_scene::selectObjects: which world objects of move_group's scene the executor's probe scenes copy.
// With no object but the carton and the pole in the scene the selection is empty, i.e. the probe scene equals the
// former one (carton + pole only).
#include <gtest/gtest.h>

#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include "igus_rebel_moveit_config/probe_scene_objects.hpp"

namespace
{
using moveit_msgs::msg::CollisionObject;

CollisionObject box(const std::string& id, double x, double y, double z, double sx, double sy, double sz)
{
  CollisionObject object;
  object.id = id;
  object.header.frame_id = "world";
  object.operation = CollisionObject::ADD;
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {sx, sy, sz};
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.w = 1.0;
  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  return object;
}

const std::set<std::string> kSkip{"detected_carton", "camera_pole"};

std::vector<std::string> ids(const std::vector<CollisionObject>& objects)
{
  std::vector<std::string> out;
  for (const auto& o : objects) out.push_back(o.id);
  return out;
}
}  // namespace

TEST(ProbeSceneObjects, CopiesTheTableAndOtherObjectsButNotCartonOrPole)
{
  const std::vector<CollisionObject> scene = {
    box("camera_head", 0, 0, 0.5, 0.1, 0.1, 0.1), box("camera_pole", 0, 0, 0.5, 0.03, 0.03, 1.0),
    box("detected_carton", 0.4, 0, 0.1, 0.3, 0.3, 0.2), box("table", -0.2, -0.2, -0.02, 0.65, 0.65, 0.04)};
  EXPECT_EQ(ids(probe_scene::selectObjects(scene, kSkip)), (std::vector<std::string>{"camera_head", "table"}));
  EXPECT_EQ(probe_scene::idList(probe_scene::selectObjects(scene, kSkip)), "camera_head, table");
}

TEST(ProbeSceneObjects, NoTableMeansNoExtraObjectAndAnUnchangedProbeSceneList)
{
  const std::vector<CollisionObject> scene = {box("camera_pole", 0, 0, 0.5, 0.03, 0.03, 1.0),
                                              box("detected_carton", 0.4, 0, 0.1, 0.3, 0.3, 0.2)};
  EXPECT_TRUE(probe_scene::selectObjects(scene, kSkip).empty());
  EXPECT_TRUE(probe_scene::selectObjects({}, kSkip).empty());
  EXPECT_EQ(probe_scene::idList({}), "none");
}

TEST(ProbeSceneObjects, DropsEmptyIdsDuplicatesNonAddAndGeometryFreeObjects)
{
  CollisionObject removed = box("gone", 0, 0, 0, 1, 1, 1);
  removed.operation = CollisionObject::REMOVE;
  CollisionObject empty_geometry;
  empty_geometry.id = "no_geometry";
  empty_geometry.operation = CollisionObject::ADD;
  CollisionObject unnamed = box("", 0, 0, 0, 1, 1, 1);
  const std::vector<CollisionObject> scene = {box("table", 0, 0, 0, 1, 1, 1), box("table", 5, 5, 5, 1, 1, 1),
                                              removed, empty_geometry, unnamed};
  const auto selected = probe_scene::selectObjects(scene, kSkip);
  ASSERT_EQ(selected.size(), 1U);
  EXPECT_EQ(selected[0].id, "table");
  EXPECT_DOUBLE_EQ(selected[0].primitive_poses[0].position.x, 0.0);   // first wins
}

TEST(ProbeSceneObjects, SelectionDoesNotModifyTheInput)
{
  const std::vector<CollisionObject> scene = {box("table", 0, 0, 0, 1, 1, 1)};
  const auto selected = probe_scene::selectObjects(scene, kSkip);
  EXPECT_EQ(scene.size(), 1U);
  EXPECT_EQ(selected[0].operation, CollisionObject::ADD);
}

TEST(ProbeSceneObjects, CopiesOnlyAllowedObjectVsRobotLinkEntries)
{
  namespace AllowedCollision = collision_detection::AllowedCollision;
  using collision_detection::AllowedCollisionMatrix;
  AllowedCollisionMatrix monitored;
  monitored.setEntry("base_link", "table", true);        // the exemption table_collision.py writes
  monitored.setEntry("link1", "table", false);           // not allowed: must not be copied
  monitored.setEntry("link1", "base_link", true);        // robot-robot pair (SRDF): not an object entry
  monitored.setEntry("knife_c_holder", "detected_carton", true);   // carton allowance: never copied
  monitored.setEntry("table", "detected_carton", true);  // object-object pair: never copied
  const std::set<std::string> links{"base_link", "link1", "knife_c_holder"};
  const auto is_link = [&links](const std::string& n) { return links.count(n) != 0U; };
  const std::vector<CollisionObject> objects = {box("table", 0, 0, 0, 1, 1, 1)};

  AllowedCollisionMatrix probe;
  EXPECT_EQ(probe_scene::copyExemptions(monitored, objects, is_link, probe), 1U);
  AllowedCollision::Type type;
  ASSERT_TRUE(probe.getEntry("table", "base_link", type));
  EXPECT_EQ(type, AllowedCollision::ALWAYS);
  EXPECT_TRUE(!probe.getEntry("table", "link1", type) || type != AllowedCollision::ALWAYS);
  EXPECT_TRUE(!probe.getEntry("knife_c_holder", "detected_carton", type));
  EXPECT_TRUE(!probe.getEntry("table", "detected_carton", type));
  EXPECT_TRUE(!probe.getEntry("link1", "base_link", type));

  // no object copied (no table in the scene): the probe ACM is not touched
  AllowedCollisionMatrix untouched;
  EXPECT_EQ(probe_scene::copyExemptions(monitored, {}, is_link, untouched), 0U);
  std::vector<std::string> names;
  untouched.getAllEntryNames(names);
  EXPECT_TRUE(names.empty());
}
