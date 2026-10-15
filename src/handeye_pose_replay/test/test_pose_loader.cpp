#include <gtest/gtest.h>

#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "handeye_pose_replay/pose_loader.hpp"

using namespace handeye_pose_replay;

namespace
{

const std::vector<std::string> kGroup = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
const std::vector<JointBounds> kBounds(6, JointBounds{-3.0, 3.0});

// One samples.jsonl line. `prefix` is e.g. "" or "arm_right_"; v are 6 joint values.
std::string line(
    const std::string& arm, int capture, const std::vector<double>& v, const std::string& prefix = "",
    const std::string& extra = "")
{
  std::ostringstream s;
  s.precision(17);
  s << "{\"arm\": \"" << arm << "\", \"capture\": " << capture << ", \"stamp\": 1.5, \"joints\": {";
  // deliberately unordered, like the recorder's dicts
  const int order[] = {1, 0, 2, 3, 5, 4};
  bool first = true;
  for (int k : order)
  {
    if (static_cast<std::size_t>(k) >= v.size()) continue;
    s << (first ? "" : ", ") << "\"" << prefix << "joint" << (k + 1) << "\": " << v[k];
    first = false;
  }
  s << "}" << extra << "}";
  return s.str();
}

// Replace the value of "joint3" in a line with raw JSON text.
std::string withJoint3(const std::string& text, const std::string& raw)
{
  const std::regex re("\"joint3\": [^,}]+");
  EXPECT_TRUE(std::regex_search(text, re));
  return std::regex_replace(text, re, "\"joint3\": " + raw);
}

std::vector<double> pose(double base) { return {base, base + 0.1, base + 0.2, base + 0.3, base + 0.4, base + 0.5}; }

LoadResult parse(const std::string& text, const LoadOptions& o = LoadOptions{})
{
  std::istringstream in(text);
  return parsePoses(in, kGroup, kBounds, o);
}

std::string joined(const LoadResult& r)
{
  std::string s;
  for (const auto& e : r.errors) s += e + "\n";
  return s;
}

}  // namespace

TEST(PoseLoader, ValidSingleArmFile)
{
  const auto r = parse(line("left", 1, pose(0.0)) + "\n" + line("left", 2, pose(0.5)) + "\n");
  ASSERT_TRUE(r.ok()) << joined(r);
  ASSERT_EQ(r.poses.size(), 2u);
  EXPECT_EQ(r.filtered_total, 2u);
  EXPECT_EQ(r.poses[0].capture, 1);
  EXPECT_EQ(r.poses[1].capture, 2);
  EXPECT_EQ(r.poses[1].line, 2u);
  EXPECT_EQ(r.poses[1].index, 1u);
  for (int j = 0; j < 6; ++j) EXPECT_DOUBLE_EQ(r.poses[0].q[j], pose(0.0)[j]) << j;
}

TEST(PoseLoader, PrefixedDualFileWithArmFilter)
{
  const std::string text = line("left", 1, pose(0.0), "arm_left_") + "\n" +
                           line("right", 1, pose(1.0), "arm_right_") + "\n" +
                           line("right", 2, pose(1.5), "arm_right_") + "\n";
  LoadOptions o;
  o.arm = "right";
  const auto r = parse(text, o);
  ASSERT_TRUE(r.ok()) << joined(r);
  ASSERT_EQ(r.poses.size(), 2u);
  EXPECT_EQ(r.poses[0].arm, "right");
  // arm_right_joint3 maps to group joint3 (index 2)
  EXPECT_DOUBLE_EQ(r.poses[0].q[2], 1.2);
  EXPECT_DOUBLE_EQ(r.poses[1].q[0], 1.5);
}

TEST(PoseLoader, PrefixedGroupNamesAlsoMap)
{
  const std::vector<std::string> group = {"arm_right_joint1", "arm_right_joint2", "arm_right_joint3",
                                          "arm_right_joint4", "arm_right_joint5", "arm_right_joint6"};
  std::istringstream in(line("left", 1, pose(0.0)) + "\n");
  const auto r = parsePoses(in, group, kBounds, LoadOptions{});
  ASSERT_TRUE(r.ok()) << joined(r);
  EXPECT_DOUBLE_EQ(r.poses[0].q[5], 0.5);
}

TEST(PoseLoader, MixedArmsWithoutArmIsError)
{
  const auto r = parse(line("left", 1, pose(0.0)) + "\n" + line("right", 1, pose(1.0)) + "\n");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(joined(r).find("more than one arm"), std::string::npos) << joined(r);
  EXPECT_TRUE(r.poses.empty());
}

TEST(PoseLoader, UnknownArmFilterIsError)
{
  LoadOptions o;
  o.arm = "right";
  const auto r = parse(line("left", 1, pose(0.0)) + "\n", o);
  ASSERT_FALSE(r.ok());
  EXPECT_NE(joined(r).find("'right'"), std::string::npos);
}

TEST(PoseLoader, MissingJointIsError)
{
  auto v = pose(0.0);
  v.pop_back();  // joint6 omitted
  const auto r = parse(line("left", 7, v) + "\n");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(joined(r).find("line 1 (capture #7)"), std::string::npos) << joined(r);
  EXPECT_NE(joined(r).find("missing joint for joint6"), std::string::npos) << joined(r);
}

TEST(PoseLoader, UnmappedJointIsError)
{
  const auto r = parse(
      "{\"arm\": \"left\", \"capture\": 1, \"joints\": {\"joint1\": 0, \"joint2\": 0, \"joint3\": 0, "
      "\"joint4\": 0, \"joint5\": 0, \"joint6\": 0, \"gripper\": 0}}\n");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(joined(r).find("gripper"), std::string::npos);
}

TEST(PoseLoader, NonFiniteIsError)
{
  const auto r = parse(withJoint3(line("left", 1, pose(0.0)), "1e999") + "\n");
  ASSERT_FALSE(r.ok());
  // 1e999 overflows to inf; either "not finite" or a malformed-number error, never a pose
  EXPECT_TRUE(r.poses.empty());
  EXPECT_NE(joined(r).find("line 1"), std::string::npos) << joined(r);
}

TEST(PoseLoader, NaNTokenIsError)
{
  const auto r = parse(withJoint3(line("left", 1, pose(0.0)), "NaN") + "\n");
  ASSERT_FALSE(r.ok());
  EXPECT_TRUE(r.poses.empty());
}

TEST(PoseLoader, NonNumericJointIsError)
{
  const auto r = parse(withJoint3(line("left", 1, pose(0.0)), "\"x\"") + "\n");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(joined(r).find("not a number"), std::string::npos);
}

TEST(PoseLoader, OutOfBoundsReportsEveryOffender)
{
  auto bad1 = pose(0.0);
  bad1[3] = 3.5;
  auto bad2 = pose(1.0);
  bad2[0] = -3.2;
  const auto r = parse(
      line("left", 1, bad1) + "\n" + line("left", 2, pose(0.5)) + "\n" + line("left", 3, bad2) + "\n");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.errors.size(), 2u) << joined(r);
  EXPECT_NE(r.errors[0].find("line 1 (capture #1)"), std::string::npos);
  EXPECT_NE(r.errors[0].find("joint4"), std::string::npos);
  EXPECT_NE(r.errors[1].find("line 3 (capture #3)"), std::string::npos);
  EXPECT_TRUE(r.poses.empty());
}

TEST(PoseLoader, BoundsAreValidatedEvenBeyondMaxPoses)
{
  auto bad = pose(0.0);
  bad[1] = 9.0;
  LoadOptions o;
  o.max_poses = 1;
  const auto r = parse(line("left", 1, pose(0.0)) + "\n" + line("left", 2, bad) + "\n", o);
  EXPECT_FALSE(r.ok());
}

TEST(PoseLoader, DuplicatesAreSkipped)
{
  auto near = pose(0.0);
  near[2] += 0.009;  // within 0.01 of the previous kept pose
  auto far = pose(0.0);
  far[2] += 0.02;
  const auto r = parse(
      line("left", 1, pose(0.0)) + "\n" + line("left", 2, near) + "\n" + line("left", 3, pose(0.0)) + "\n" +
      line("left", 4, far) + "\n");
  ASSERT_TRUE(r.ok()) << joined(r);
  EXPECT_EQ(r.duplicates_skipped, 2u);
  ASSERT_EQ(r.poses.size(), 2u);
  EXPECT_EQ(r.poses[0].capture, 1);
  EXPECT_EQ(r.poses[1].capture, 4);
  EXPECT_EQ(r.filtered_total, 2u);
}

TEST(PoseLoader, StartIndexAndMaxPoses)
{
  std::string text;
  for (int i = 0; i < 6; ++i) text += line("left", i + 1, pose(0.2 * i)) + "\n";

  LoadOptions o;
  o.start_index = 2;
  auto r = parse(text, o);
  ASSERT_TRUE(r.ok()) << joined(r);
  ASSERT_EQ(r.poses.size(), 4u);
  EXPECT_EQ(r.poses.front().capture, 3);
  EXPECT_EQ(r.poses.front().index, 2u);
  EXPECT_EQ(r.filtered_total, 6u);

  o.max_poses = 2;
  r = parse(text, o);
  ASSERT_TRUE(r.ok());
  ASSERT_EQ(r.poses.size(), 2u);
  EXPECT_EQ(r.poses.back().capture, 4);

  o.start_index = 5;
  o.max_poses = 10;
  r = parse(text, o);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.poses.size(), 1u);

  o.start_index = 6;
  r = parse(text, o);
  ASSERT_FALSE(r.ok());
  EXPECT_NE(joined(r).find("start_index"), std::string::npos);
}

TEST(PoseLoader, BlankLinesTolerated)
{
  const auto r = parse("\n  \n" + line("left", 1, pose(0.0)) + "\n\n\t\n" + line("left", 2, pose(1.0)) + "\n\n");
  ASSERT_TRUE(r.ok()) << joined(r);
  ASSERT_EQ(r.poses.size(), 2u);
  EXPECT_EQ(r.poses[0].line, 3u);
  EXPECT_EQ(r.poses[1].line, 6u);
}

TEST(PoseLoader, MalformedJsonReportsLineNumber)
{
  const auto r = parse(line("left", 1, pose(0.0)) + "\n\n{\"arm\": \"left\", \"joints\": {\n" + line("left", 2, pose(1.0)) + "\n");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(joined(r).find("line 3: malformed JSON"), std::string::npos) << joined(r);
  EXPECT_TRUE(r.poses.empty());
}

TEST(PoseLoader, EmptyFileIsError)
{
  const auto r = parse("\n\n");
  EXPECT_FALSE(r.ok());
}

TEST(PoseLoader, JointSuffix)
{
  EXPECT_EQ(jointSuffixNumber("joint3"), 3);
  EXPECT_EQ(jointSuffixNumber("arm_right_joint3"), 3);
  EXPECT_EQ(jointSuffixNumber("joint"), -1);
  EXPECT_EQ(jointSuffixNumber("gripper3"), -1);
  EXPECT_EQ(jointSuffixNumber("joint3_x"), -1);
}

class PoseLoaderFiles : public ::testing::Test
{
protected:
  void SetUp() override
  {
    char tmpl[] = "/tmp/pose_loader_testXXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    dir_ = tmpl;
    std::ofstream(dir_ + "/samples.jsonl") << line("left", 1, pose(0.0)) << "\n" << line("left", 2, pose(1.0)) << "\n";
  }
  void TearDown() override
  {
    std::remove((dir_ + "/samples.jsonl").c_str());
    rmdir(dir_.c_str());
  }
  std::string dir_;
};

TEST_F(PoseLoaderFiles, DirectoryResolvesToSamplesJsonl)
{
  EXPECT_EQ(resolveSamplesPath(dir_), dir_ + "/samples.jsonl");
  EXPECT_EQ(resolveSamplesPath(dir_ + "/"), dir_ + "/samples.jsonl");
  EXPECT_EQ(resolveSamplesPath(dir_ + "/samples.jsonl"), dir_ + "/samples.jsonl");

  const auto from_dir = loadPoses(dir_, kGroup, kBounds, LoadOptions{});
  ASSERT_TRUE(from_dir.ok()) << joined(from_dir);
  EXPECT_EQ(from_dir.poses.size(), 2u);
  EXPECT_EQ(from_dir.source_path, dir_ + "/samples.jsonl");
  const auto from_file = loadPoses(dir_ + "/samples.jsonl", kGroup, kBounds, LoadOptions{});
  ASSERT_TRUE(from_file.ok());
  EXPECT_EQ(from_file.poses.size(), 2u);
}

TEST_F(PoseLoaderFiles, TildeIsExpanded)
{
  setenv("HOME", "/tmp", 1);
  const std::string base = dir_.substr(std::string("/tmp/").size());
  EXPECT_EQ(expandUser("~/" + base), dir_);
  const auto r = loadPoses("~/" + base, kGroup, kBounds, LoadOptions{});
  ASSERT_TRUE(r.ok()) << joined(r);
  EXPECT_EQ(r.poses.size(), 2u);
}

TEST_F(PoseLoaderFiles, MissingFileIsError)
{
  const auto r = loadPoses(dir_ + "/nope", kGroup, kBounds, LoadOptions{});
  ASSERT_FALSE(r.ok());
  EXPECT_NE(joined(r).find("cannot open"), std::string::npos);
}
