#pragma once

// Pure (no ROS) loader for the joint poses of a calibration-recorder session.
//
// A session directory holds samples.jsonl, one JSON object per line with at
// least {"arm": "left"|"right", "capture": int, "joints": {name: rad}}.
// Single-arm sessions use joint1..joint6, dual-arm sessions use prefixed names
// such as arm_left_joint1. Joints are mapped to the planning group's joints by
// the trailing "joint<N>" suffix, so both spellings map to the group's joint<N>.

#include <cstddef>
#include <istream>
#include <string>
#include <vector>

namespace handeye_pose_replay
{

struct JointBounds
{
  double lower;
  double upper;
};

struct LoadOptions
{
  // Keep only samples whose "arm" equals this. Empty = no filter; then a file
  // that contains more than one distinct arm is refused.
  std::string arm;
  // 0-based index into the filtered (arm + duplicate-skipped) pose list.
  std::size_t start_index = 0;
  // 0 = all poses from start_index on.
  std::size_t max_poses = 0;
  // A pose whose joints are all within this of the previous kept pose is a duplicate.
  double duplicate_tolerance_rad = 0.01;
  // Slack allowed outside the model bounds (recorded states may touch a limit).
  double bound_tolerance_rad = 1e-6;
};

struct ReplayPose
{
  std::size_t index = 0;   // 0-based index in the filtered list (before start/max slicing)
  std::size_t line = 0;    // 1-based line number in samples.jsonl
  int capture = -1;        // "capture" field, -1 if absent
  std::string arm;
  std::vector<double> q;   // ordered like the group joint names given to the loader
};

struct LoadResult
{
  std::vector<ReplayPose> poses;     // the poses to replay (after start_index / max_poses)
  std::size_t filtered_total = 0;    // size of the filtered list (after duplicate skipping)
  std::size_t duplicates_skipped = 0;
  std::string source_path;           // resolved samples.jsonl path (when loaded from a path)
  std::vector<std::string> errors;   // non-empty = refuse to move anything
  bool ok() const { return errors.empty(); }
};

// Expands a leading "~" using $HOME.
std::string expandUser(const std::string& path);

// A directory -> "<dir>/samples.jsonl"; anything else is returned unchanged
// (after "~" expansion).
std::string resolveSamplesPath(const std::string& path_or_dir);

// Trailing joint number of a name such as "joint3" / "arm_right_joint3";
// -1 if the name does not end in "joint<digits>".
int jointSuffixNumber(const std::string& name);

// Parse a whole samples.jsonl stream. group_joint_names / bounds describe the
// planning group's active joints (same order); the poses' q follow that order.
LoadResult parsePoses(
    std::istream& in,
    const std::vector<std::string>& group_joint_names,
    const std::vector<JointBounds>& bounds,
    const LoadOptions& options);

// Resolve (directory or file, "~" ok), open and parse.
LoadResult loadPoses(
    const std::string& path_or_dir,
    const std::vector<std::string>& group_joint_names,
    const std::vector<JointBounds>& bounds,
    const LoadOptions& options);

}  // namespace handeye_pose_replay
