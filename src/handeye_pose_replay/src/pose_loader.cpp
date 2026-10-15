#include "handeye_pose_replay/pose_loader.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>

namespace handeye_pose_replay
{

namespace
{

std::string where(std::size_t line, int capture)
{
  std::ostringstream s;
  s << "line " << line;
  if (capture >= 0) s << " (capture #" << capture << ")";
  return s.str();
}

struct Sample
{
  std::size_t line = 0;
  int capture = -1;
  std::string arm;
  std::map<std::string, nlohmann::json> joints;
};

}  // namespace

std::string expandUser(const std::string& path)
{
  if (!path.empty() && path[0] == '~' && (path.size() == 1 || path[1] == '/'))
  {
    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') return std::string(home) + path.substr(1);
  }
  return path;
}

std::string resolveSamplesPath(const std::string& path_or_dir)
{
  const std::string path = expandUser(path_or_dir);
  struct stat st;
  if (!path.empty() && ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
  {
    return path.back() == '/' ? path + "samples.jsonl" : path + "/samples.jsonl";
  }
  return path;
}

int jointSuffixNumber(const std::string& name)
{
  std::size_t end = name.size();
  std::size_t begin = end;
  while (begin > 0 && name[begin - 1] >= '0' && name[begin - 1] <= '9') --begin;
  if (begin == end || end - begin > 6) return -1;
  static const std::string kJoint = "joint";
  if (begin < kJoint.size() || name.compare(begin - kJoint.size(), kJoint.size(), kJoint) != 0)
    return -1;
  return std::atoi(name.substr(begin).c_str());
}

LoadResult parsePoses(
    std::istream& in,
    const std::vector<std::string>& group_joint_names,
    const std::vector<JointBounds>& bounds,
    const LoadOptions& options)
{
  LoadResult result;
  auto& errors = result.errors;

  // Map group joints by their joint<N> suffix.
  std::map<int, std::size_t> group_by_number;
  if (group_joint_names.empty() || bounds.size() != group_joint_names.size())
  {
    errors.push_back("internal: group joint names and bounds are empty or differ in size");
    return result;
  }
  for (std::size_t j = 0; j < group_joint_names.size(); ++j)
  {
    const int n = jointSuffixNumber(group_joint_names[j]);
    if (n < 0 || !group_by_number.emplace(n, j).second)
    {
      errors.push_back(
          "planning group joint '" + group_joint_names[j] +
          "' has no unique trailing joint<N> suffix");
    }
  }
  if (!errors.empty()) return result;

  if (options.duplicate_tolerance_rad < 0.0 || options.bound_tolerance_rad < 0.0)
  {
    errors.push_back("internal: tolerances must be non-negative");
    return result;
  }

  // 1. Parse every non-blank line.
  std::vector<Sample> samples;
  std::string text;
  std::size_t line_no = 0;
  while (std::getline(in, text))
  {
    ++line_no;
    if (text.find_first_not_of(" \t\r\n") == std::string::npos) continue;

    nlohmann::json j;
    try
    {
      j = nlohmann::json::parse(text);
    }
    catch (const nlohmann::json::exception& e)
    {
      errors.push_back("line " + std::to_string(line_no) + ": malformed JSON: " + e.what());
      continue;
    }
    if (!j.is_object())
    {
      errors.push_back("line " + std::to_string(line_no) + ": not a JSON object");
      continue;
    }

    Sample s;
    s.line = line_no;
    if (j.contains("capture"))
    {
      const auto& c = j["capture"];
      if (c.is_number_integer() && c.get<long long>() >= 0 && c.get<long long>() < 1000000000LL)
        s.capture = static_cast<int>(c.get<long long>());
      else
        errors.push_back(where(line_no, -1) + ": 'capture' is not a non-negative integer");
    }
    if (j.contains("arm"))
    {
      if (j["arm"].is_string())
        s.arm = j["arm"].get<std::string>();
      else
        errors.push_back(where(line_no, s.capture) + ": 'arm' is not a string");
    }
    if (!j.contains("joints") || !j["joints"].is_object())
    {
      errors.push_back(where(line_no, s.capture) + ": missing 'joints' object");
      continue;
    }
    for (auto it = j["joints"].begin(); it != j["joints"].end(); ++it) s.joints.emplace(it.key(), it.value());
    samples.push_back(std::move(s));
  }

  // 2. Arm selection.
  std::set<std::string> arms;
  for (const auto& s : samples) arms.insert(s.arm);
  if (options.arm.empty() && arms.size() > 1)
  {
    std::string list;
    for (const auto& a : arms) list += (list.empty() ? "'" : ", '") + a + "'";
    errors.push_back(
        "the file contains more than one arm (" + list + "); set the 'arm' parameter to choose one");
    return result;
  }
  std::vector<const Sample*> selected;
  for (const auto& s : samples)
    if (options.arm.empty() || s.arm == options.arm) selected.push_back(&s);
  if (selected.empty() && errors.empty())
  {
    errors.push_back(
        options.arm.empty() ? std::string("no poses found")
                            : "no samples with arm '" + options.arm + "' found");
    return result;
  }

  // 3. Map + validate every selected sample (the whole list, before any motion).
  struct Mapped
  {
    const Sample* sample;
    std::vector<double> q;
  };
  std::vector<Mapped> mapped;
  for (const Sample* s : selected)
  {
    const std::string at = where(s->line, s->capture);
    std::vector<double> q(group_joint_names.size(), 0.0);
    std::vector<bool> have(group_joint_names.size(), false);
    bool bad = false;
    for (const auto& kv : s->joints)
    {
      const int n = jointSuffixNumber(kv.first);
      const auto g = group_by_number.find(n);
      if (n < 0 || g == group_by_number.end())
      {
        errors.push_back(at + ": joint '" + kv.first + "' does not map to a joint of the planning group");
        bad = true;
        continue;
      }
      if (have[g->second])
      {
        errors.push_back(at + ": more than one joint maps to " + group_joint_names[g->second]);
        bad = true;
        continue;
      }
      if (!kv.second.is_number())
      {
        errors.push_back(at + ": joint '" + kv.first + "' is not a number");
        bad = true;
        continue;
      }
      const double v = kv.second.get<double>();
      if (!std::isfinite(v))
      {
        errors.push_back(at + ": joint '" + kv.first + "' is not finite");
        bad = true;
        continue;
      }
      q[g->second] = v;
      have[g->second] = true;
    }
    for (std::size_t j = 0; j < have.size(); ++j)
    {
      if (!have[j] && !bad)
      {
        errors.push_back(at + ": missing joint for " + group_joint_names[j]);
        bad = true;
      }
    }
    if (bad) continue;
    for (std::size_t j = 0; j < q.size(); ++j)
    {
      const double tol = options.bound_tolerance_rad;
      if (q[j] < bounds[j].lower - tol || q[j] > bounds[j].upper + tol)
      {
        std::ostringstream m;
        m << at << ": " << group_joint_names[j] << " = " << q[j] << " rad is outside the model bounds ["
          << bounds[j].lower << ", " << bounds[j].upper << "]";
        errors.push_back(m.str());
        bad = true;
      }
    }
    if (!bad) mapped.push_back({s, std::move(q)});
  }
  if (!errors.empty()) return result;

  // 4. Duplicate skipping against the previous kept pose.
  std::vector<ReplayPose> kept;
  for (const auto& m : mapped)
  {
    if (!kept.empty())
    {
      double largest = 0.0;
      for (std::size_t j = 0; j < m.q.size(); ++j)
        largest = std::max(largest, std::fabs(m.q[j] - kept.back().q[j]));
      if (largest <= options.duplicate_tolerance_rad)
      {
        ++result.duplicates_skipped;
        continue;
      }
    }
    ReplayPose p;
    p.index = kept.size();
    p.line = m.sample->line;
    p.capture = m.sample->capture;
    p.arm = m.sample->arm;
    p.q = m.q;
    kept.push_back(std::move(p));
  }
  result.filtered_total = kept.size();

  // 5. start_index / max_poses.
  if (options.start_index >= kept.size())
  {
    errors.push_back(
        "start_index " + std::to_string(options.start_index) + " is beyond the last pose (" +
        std::to_string(kept.size()) + " poses after filtering)");
    return result;
  }
  std::size_t end = kept.size();
  if (options.max_poses > 0) end = std::min(end, options.start_index + options.max_poses);
  result.poses.assign(
      kept.begin() + static_cast<std::ptrdiff_t>(options.start_index),
      kept.begin() + static_cast<std::ptrdiff_t>(end));
  return result;
}

LoadResult loadPoses(
    const std::string& path_or_dir,
    const std::vector<std::string>& group_joint_names,
    const std::vector<JointBounds>& bounds,
    const LoadOptions& options)
{
  const std::string path = resolveSamplesPath(path_or_dir);
  std::ifstream file(path);
  if (!file)
  {
    LoadResult r;
    r.source_path = path;
    r.errors.push_back("cannot open poses file '" + path + "'");
    return r;
  }
  LoadResult r = parsePoses(file, group_joint_names, bounds, options);
  r.source_path = path;
  return r;
}

}  // namespace handeye_pose_replay
