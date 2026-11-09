#include "dual_cut_coordinator/top_split.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <nlohmann/json.hpp>
#include <unistd.h>

namespace dual_cut
{

namespace
{
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

constexpr const char * kMarkerKey = "dual_cut_coordinator_top_split";

bool finite(double v) { return std::isfinite(v); }

Status pointFrom(const json & doc, const char * key, Eigen::Vector3d & out)
{
  if (!doc.is_object() || !doc.contains(key) || !doc[key].is_object())
    return Status::Error(std::string("snapshot has no point '") + key + "'");
  const json & p = doc[key];
  for (const char * c : {"x", "y", "z"})
    if (!p.contains(c) || !p[c].is_number()) return Status::Error(std::string("snapshot point '") + key + "' has no numeric " + c);
  out = Eigen::Vector3d(p["x"].get<double>(), p["y"].get<double>(), p["z"].get<double>());
  if (!out.allFinite()) return Status::Error(std::string("snapshot point '") + key + "' is not finite");
  return Status::Ok();
}

json pointJson(const json & like, const Eigen::Vector3d & v)
{
  json p = like;   // keeps any extra keys and the key order
  p["x"] = v.x();
  p["y"] = v.y();
  p["z"] = v.z();
  return p;
}

Status parse(const std::string & text, json & doc)
{
  try
  {
    doc = json::parse(text);
  }
  catch (const std::exception & e)
  {
    return Status::Error(std::string("snapshot is not valid JSON: ") + e.what());
  }
  if (!doc.is_object()) return Status::Error("snapshot is not a JSON object");
  return Status::Ok();
}

std::string mm(double m, int prec = 1)
{
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << m * 1000.0;
  return os.str();
}
std::string pct(double fraction)
{
  std::ostringstream os;
  os << std::fixed << std::setprecision(1) << fraction * 100.0;
  return os.str();
}
}  // namespace

const char * toString(TopSplitMode mode)
{
  return mode == TopSplitMode::LeftMax ? "left_max" : "half";
}

Status topSplitModeFromString(const std::string & text, TopSplitMode & out)
{
  if (text == "left_max") out = TopSplitMode::LeftMax;
  else if (text == "half") out = TopSplitMode::Half;
  else return Status::Error("top_split_mode '" + text + "' must be left_max or half");
  return Status::Ok();
}

double effectiveRightEndExtension(const TopSplitParams & p)
{
  return p.mode == TopSplitMode::LeftMax ? p.right_end_extension_m : 0.0;
}

Status validateTopSplitParams(const TopSplitParams & p)
{
  auto in = [](double v, double lo, double hi) { return finite(v) && v >= lo && v <= hi; };
  if (!in(p.overlap_m, 0.003, 0.05)) return Status::Error("top_split_overlap_m must be within [0.003, 0.05] m");
  if (!in(p.right_min_tail, 0.05, 0.5)) return Status::Error("top_split_right_min_tail must be within [0.05, 0.5]");
  if (!in(p.left_min_fraction, 0.05, 1.0)) return Status::Error("top_split_left_min_fraction must be within [0.05, 1.0]");
  if (!in(p.min_overlap_m, 0.0, 0.05)) return Status::Error("top_split_min_overlap_m must be within [0, 0.05] m");
  if (!(in(p.end_tolerance_m, 0.0, 0.05) && p.end_tolerance_m > 0.0))
    return Status::Error("top_split_end_tolerance_m must be within (0, 0.05] m");
  if (!in(p.skip_right_top_fraction, 0.9, 1.0)) return Status::Error("top_split_skip_right_top_fraction must be within [0.9, 1.0]");
  if (!in(p.right_end_extension_m, 0.0, 0.02)) return Status::Error("top_split_right_end_extension_m must be within [0, 0.02] m");
  const fs::path f(p.snapshot_file);
  if (!f.is_absolute() || f.extension() != ".json" || !f.has_filename())
    return Status::Error("top_split_snapshot_file must be an absolute .json path");
  return Status::Ok();
}

bool skipRightTop(double left_reach_fraction, double skip_fraction)
{
  return finite(left_reach_fraction) && finite(skip_fraction) && skip_fraction < 1.0 && left_reach_fraction >= skip_fraction;
}

Status topSplitStartFraction(double a, double seam_length_m, const TopSplitParams & p, double & out)
{
  Status s = validateTopSplitParams(p);
  if (!s) return s;
  if (!finite(a) || a < 0.0 || a > 1.0 + 1e-9) return Status::Error("left reach fraction " + std::to_string(a) + " is not within [0, 1]");
  if (!finite(seam_length_m) || !(seam_length_m > 0.05)) return Status::Error("the top seam is shorter than 50 mm or not finite");
  out = std::clamp(a - p.overlap_m / seam_length_m, 0.05, 1.0 - p.right_min_tail);
  return Status::Ok();
}

// ---- SHA-256 -------------------------------------------------------------------------------------

std::string sha256Hex(const std::string & data)
{
  static const std::array<uint32_t, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  std::array<uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::string msg = data;
  const uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8ULL;
  msg.push_back(static_cast<char>(0x80));
  while (msg.size() % 64 != 56) msg.push_back('\0');
  for (int i = 7; i >= 0; --i) msg.push_back(static_cast<char>((bit_len >> (8 * i)) & 0xff));
  auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (size_t off = 0; off < msg.size(); off += 64)
  {
    std::array<uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i)
      w[i] = (static_cast<uint32_t>(static_cast<unsigned char>(msg[off + 4 * i])) << 24) |
             (static_cast<uint32_t>(static_cast<unsigned char>(msg[off + 4 * i + 1])) << 16) |
             (static_cast<uint32_t>(static_cast<unsigned char>(msg[off + 4 * i + 2])) << 8) |
             static_cast<uint32_t>(static_cast<unsigned char>(msg[off + 4 * i + 3]));
    for (int i = 16; i < 64; ++i)
    {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i)
    {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  std::ostringstream os;
  os << std::hex << std::setfill('0');
  for (uint32_t v : h) os << std::setw(8) << v;
  return os.str();
}

Status readTextFile(const std::string & path, std::string & out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) return Status::Error("cannot read " + path);
  std::ostringstream os;
  os << in.rdbuf();
  out = os.str();
  return Status::Ok();
}

// ---- derived snapshot ----------------------------------------------------------------------------

Status seamFromSnapshotText(const std::string & json_text, Eigen::Vector3d & seam_start, Eigen::Vector3d & seam_end)
{
  json doc;
  Status s = parse(json_text, doc);
  if (!s) return s;
  s = pointFrom(doc, "seam_start", seam_start);
  if (!s) return s;
  return pointFrom(doc, "seam_end", seam_end);
}

Status sagGeometryFromSnapshotText(const std::string & json_text, SnapshotSagGeometry & out)
{
  json doc;
  Status s = parse(json_text, doc);
  if (!s) return s;
  for (auto [key, vec] : {std::pair<const char *, Eigen::Vector3d *>{"seam_start", &out.seam_start}, {"seam_end", &out.seam_end},
                          {"seam_center", &out.seam_center}, {"surface_normal", &out.surface_normal}, {"back_left", &out.back_left},
                          {"front_left", &out.front_left}, {"back_right", &out.back_right}, {"front_right", &out.front_right}})
  {
    s = pointFrom(doc, key, *vec);
    if (!s) return s;
  }
  return Status::Ok();
}

Status makeCutSag(const SnapshotSagGeometry & g, const Eigen::Isometry3d & T, double sag_m, CutSag & out)
{
  out = CutSag{};
  if (!finite(sag_m) || sag_m < 0.0) return Status::Error("middle_cut_sag_m must be finite and >= 0");
  if (sag_m == 0.0) return Status::Ok();
  const Eigen::Vector3d axis = T.linear() * (g.seam_end - g.seam_start);
  const Eigen::Vector3d normal = T.linear() * g.surface_normal;
  if (!(axis.norm() > 1e-6) || !(normal.norm() > 1e-6)) return Status::Error("sag: degenerate seam axis or surface normal");
  const Eigen::Vector3d u = axis.normalized();
  const Eigen::Vector3d bl = T * g.back_left, fl = T * g.front_left, br = T * g.back_right, fr = T * g.front_right;
  const double h = std::abs(0.5 * (bl + fl).dot(u) - 0.5 * (br + fr).dot(u)) / 2.0;
  if (!finite(h) || !(h > 0.01)) return Status::Error("sag: the carton half length along the seam axis is not positive");
  out.active = true;
  out.sag_m = sag_m;
  out.half_length_m = h;
  out.center = T * g.seam_center;
  out.u = u;
  out.down = -normal.normalized();
  if (!out.center.allFinite() || !out.down.allFinite())
  {
    out = CutSag{};
    return Status::Error("sag: non-finite geometry");
  }
  return Status::Ok();
}

Status deriveTopSnapshotText(
  const std::string & source_json, double f, const TopSplitMarkerInfo & info, std::string & out_json)
{
  if (!finite(f) || !(f > 0.0 && f < 1.0)) return Status::Error("start fraction must be within (0, 1)");
  const double ext = info.right_end_extension_m;
  if (!finite(ext) || ext < 0.0 || ext > 0.02) return Status::Error("right end extension must be within [0, 0.02] m");
  json doc;
  Status s = parse(source_json, doc);
  if (!s) return s;
  if (doc.contains(kMarkerKey)) return Status::Error("the source snapshot is already a top-split derived file; refusing to chain");
  Eigen::Vector3d seam_start, seam_end, back_left, front_left, back_right, front_right;
  for (auto [key, vec] : {std::pair<const char *, Eigen::Vector3d *>{"seam_start", &seam_start}, {"seam_end", &seam_end},
                          {"back_left", &back_left}, {"front_left", &front_left}})
  {
    s = pointFrom(doc, key, *vec);
    if (!s) return s;
  }
  if (ext > 0.0)
  {
    for (auto [key, vec] : {std::pair<const char *, Eigen::Vector3d *>{"back_right", &back_right}, {"front_right", &front_right}})
    {
      s = pointFrom(doc, key, *vec);
      if (!s) return s;
    }
  }
  const Eigen::Vector3d start2 = seam_start + f * (seam_end - seam_start);
  doc["seam_start"] = pointJson(doc["seam_start"], start2);
  doc["back_middle_left"] = pointJson(doc.contains("back_middle_left") && doc["back_middle_left"].is_object() ? doc["back_middle_left"] : json::object(), 0.5 * (back_left + start2));
  doc["front_middle_left"] = pointJson(doc.contains("front_middle_left") && doc["front_middle_left"].is_object() ? doc["front_middle_left"] : json::object(), 0.5 * (front_left + start2));
  if (ext > 0.0)
  {
    // the right cut finishes ext beyond the seam end: seam_end moves outward along the ORIGINAL seam direction
    // (start -> end of the source), and the two right side points keep their canonical midpoint formula
    const Eigen::Vector3d axis = seam_end - seam_start;
    if (!(axis.norm() > 1e-6)) return Status::Error("cannot extend a degenerate seam");
    const Eigen::Vector3d end2 = seam_end + ext * axis.normalized();
    doc["seam_end"] = pointJson(doc["seam_end"], end2);
    doc["back_middle_right"] = pointJson(doc.contains("back_middle_right") && doc["back_middle_right"].is_object() ? doc["back_middle_right"] : json::object(), 0.5 * (back_right + end2));
    doc["front_middle_right"] = pointJson(doc.contains("front_middle_right") && doc["front_middle_right"].is_object() ? doc["front_middle_right"] : json::object(), 0.5 * (front_right + end2));
  }
  json marker = json::object();
  marker["source_file"] = info.source_file;
  if (!info.source_sha256.empty()) marker["source_sha256"] = info.source_sha256;
  else
  {
    marker["source_size"] = info.source_size;
    marker["source_mtime_ns"] = info.source_mtime_ns;
  }
  marker["start_fraction"] = f;
  marker["left_reach_fraction"] = info.left_reach_fraction;
  if (ext > 0.0) marker["right_end_extension_m"] = ext;
  doc[kMarkerKey] = marker;
  out_json = doc.dump(2, ' ', true) + "\n";
  return Status::Ok();
}

Status writeDerivedTopSnapshot(
  const std::string & source_path, const std::string & derived_path, double start_fraction, double left_reach_fraction,
  const std::string & expected_source_sha256, double right_end_extension_m)
{
  std::error_code ec;
  const fs::path src(source_path), dst(derived_path);
  if (!src.is_absolute() || !dst.is_absolute() || dst.extension() != ".json")
    return Status::Error("snapshot paths must be absolute and the derived one a .json file");
  if (fs::equivalent(src, dst, ec)) return Status::Error("derived snapshot path equals the source path");
  std::string text;
  Status s = readTextFile(source_path, text);
  if (!s) return s;
  if (!expected_source_sha256.empty() && sha256Hex(text) != expected_source_sha256)
    return Status::Error("the source snapshot changed while planning; refusing to derive from a different file");
  const auto mtime = fs::last_write_time(src, ec);
  if (ec) return Status::Error("cannot read the mtime of " + source_path + ": " + ec.message());
  TopSplitMarkerInfo info;
  info.source_file = source_path;
  info.source_sha256 = sha256Hex(text);
  info.source_size = text.size();
  info.source_mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch()).count();
  info.left_reach_fraction = left_reach_fraction;
  info.right_end_extension_m = right_end_extension_m;
  std::string out;
  s = deriveTopSnapshotText(text, start_fraction, info, out);
  if (!s) return s;
  fs::create_directories(dst.parent_path(), ec);
  if (ec) return Status::Error("cannot create " + dst.parent_path().string() + ": " + ec.message());
  const fs::path tmp = dst.parent_path() / (dst.filename().string() + ".tmp." + std::to_string(::getpid()));
  {
    std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
    if (!o) return Status::Error("cannot write " + tmp.string());
    o << out;
    o.flush();
    if (!o) return Status::Error("write failed: " + tmp.string());
  }
  fs::last_write_time(tmp, mtime, ec);   // never look fresher than the capture
  if (ec)
  {
    fs::remove(tmp, ec);
    return Status::Error("cannot set the mtime of the derived snapshot");
  }
  fs::rename(tmp, dst, ec);
  if (ec)
  {
    std::error_code ec2;
    fs::remove(tmp, ec2);
    return Status::Error("cannot move the derived snapshot into place: " + ec.message());
  }
  return Status::Ok();
}

// ---- coverage ------------------------------------------------------------------------------------

Status projectCutLineOnSeam(
  const CutLine & line, const Eigen::Vector3d & seam_start, const Eigen::Vector3d & seam_end, SeamInterval & out)
{
  const Eigen::Vector3d axis = seam_end - seam_start;
  const double L = axis.norm();
  if (!(L > 1e-6) || !axis.allFinite() || !line.start.allFinite() || !line.end.allFinite())
    return Status::Error("cannot project a cut line onto a degenerate or non-finite seam");
  const Eigen::Vector3d u = axis / L;
  const double a = (line.start - seam_start).dot(u);
  const double b = (line.end - seam_start).dot(u);
  out.s0 = std::min(a, b);
  out.s1 = std::max(a, b);
  return Status::Ok();
}

std::string describeTopCoverage(const TopCoverage & c)
{
  const double L = c.seam_length_m;
  std::ostringstream os;
  os << "left " << pct(c.left.s0 / L) << ".." << pct(c.left.s1 / L) << " %, right " << pct(c.right.s0 / L) << ".."
     << pct(c.right.s1 / L) << " %, overlap " << mm(c.overlap_m) << " mm";
  if (c.right_end_extension_m > 0.0) os << ", right end " << mm(c.right_end_extension_m) << " mm beyond the seam end";
  return os.str();
}

std::string topCoverageJson(const TopCoverage & c)
{
  const double L = c.seam_length_m;
  std::ostringstream os;
  os << std::setprecision(9) << "{\"seam_length_mm\":" << L * 1e3 << ",\"left_mm\":[" << c.left.s0 * 1e3 << "," << c.left.s1 * 1e3
     << "],\"right_mm\":[" << c.right.s0 * 1e3 << "," << c.right.s1 * 1e3 << "],\"left_fraction\":[" << c.left.s0 / L << ","
     << c.left.s1 / L << "],\"right_fraction\":[" << c.right.s0 / L << "," << c.right.s1 / L << "],\"overlap_mm\":" << c.overlap_m * 1e3;
  if (c.right_end_extension_m > 0.0) os << ",\"right_end_extension_mm\":" << c.right_end_extension_m * 1e3;
  os << "}";
  return os.str();
}

Status checkTopCoverage(
  const SeamInterval & left, const SeamInterval & right, double L, double tol, double min_overlap, TopCoverage * out,
  double right_end_extension_m)
{
  TopCoverage c;
  c.seam_length_m = L;
  c.right_end_extension_m = right_end_extension_m;
  c.left = left;
  c.right = right;
  c.overlap_m = std::min(left.s1, right.s1) - std::max(left.s0, right.s0);
  if (out) *out = c;
  if (!finite(L) || !(L > 0.0) || !finite(left.s0) || !finite(left.s1) || !finite(right.s0) || !finite(right.s1) ||
      !finite(tol) || !finite(min_overlap) || !finite(right_end_extension_m) || right_end_extension_m < 0.0 || right_end_extension_m > 0.02)
    return Status::Error("top seam coverage: non-finite input or right end extension outside [0, 0.02] m");
  const std::string where = "left " + mm(left.s0) + ".." + mm(left.s1) + " mm (" + pct(left.s0 / L) + ".." + pct(left.s1 / L) +
                            " %), right " + mm(right.s0) + ".." + mm(right.s1) + " mm (" + pct(right.s0 / L) + ".." +
                            pct(right.s1 / L) + " %) of the " + mm(L) + " mm seam, overlap " + mm(c.overlap_m) + " mm";
  if (std::abs(left.s0) > tol)
    return Status::Error("top seam coverage refused: the left cut does not start at the seam's left end (tolerance " + mm(tol) + " mm): " + where);
  if (std::abs(right.s1 - (L + right_end_extension_m)) > tol)
    return Status::Error(
      "top seam coverage refused: the right cut does not end at the seam's right end" +
      (right_end_extension_m > 0.0 ? " + " + mm(right_end_extension_m) + " mm extension" : std::string()) + " (tolerance " + mm(tol) + " mm): " + where);
  if (!(c.overlap_m >= min_overlap))
    return Status::Error(
      "top seam coverage refused: " + std::string(c.overlap_m < 0.0 ? "GAP between the cuts" : "overlap too small") + " (need >= " +
      mm(min_overlap) + " mm): " + where);
  return Status::Ok();
}

Status checkLeftOnlyCoverage(const SeamInterval & left, double L, double tol, TopCoverage * out)
{
  TopCoverage c;
  c.seam_length_m = L;
  c.left = left;
  c.right.s0 = c.right.s1 = L;
  c.overlap_m = 0.0;
  if (out) *out = c;
  if (!finite(L) || !(L > 0.0) || !finite(left.s0) || !finite(left.s1) || !finite(tol)) return Status::Error("top seam coverage: non-finite input");
  const std::string where = "left " + mm(left.s0) + ".." + mm(left.s1) + " mm (" + pct(left.s0 / L) + ".." + pct(left.s1 / L) + " %) of the " + mm(L) + " mm seam";
  if (std::abs(left.s0) > tol)
    return Status::Error("top seam coverage refused (right top skipped): the left cut does not start at the seam's left end (tolerance " + mm(tol) + " mm): " + where);
  if (std::abs(left.s1 - L) > tol)
    return Status::Error("top seam coverage refused (right top skipped): the left cut does not reach the seam's right end (tolerance " + mm(tol) + " mm): " + where);
  return Status::Ok();
}

}  // namespace dual_cut
