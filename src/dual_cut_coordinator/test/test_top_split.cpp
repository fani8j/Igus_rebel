#include <gtest/gtest.h>

#include <unistd.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include "dual_cut_coordinator/top_split.hpp"

using namespace dual_cut;
namespace fs = std::filesystem;

namespace
{
// A snapshot shaped like a real capture (box_h_cut_snapshot/v1), numbers of /tmp/rh/capture_0556.json.
const char * kSnapshot = R"JSON({
 "schema": "box_h_cut_snapshot/v1",
 "snapshot_id": "capture_frame_000008",
 "capture_id": "frame_000008",
 "acquisition_stamp_ns": 1792447797748718000,
 "frame_id": "base_link",
 "seam_start": {"x": 0.45868071950053624, "y": -0.005479864899818193, "z": 0.02462807291517244},
 "seam_end": {"x": 0.4474669423246542, "y": -0.39875824840029334, "z": 0.03186467505241786},
 "seam_center": {"x": 0.45307383091259523, "y": -0.20211905665005575, "z": 0.028246373983795148},
 "back_left": {"x": 0.5999983881435447, "y": -0.010367345401932931, "z": 0.02596561752364268},
 "back_right": {"x": 0.5868777403750626, "y": -0.4027429734733289, "z": 0.03316856322575634},
 "front_left": {"x": 0.316410489861199, "y": -0.00047968905512144056, "z": 0.02328002496936521},
 "front_right": {"x": 0.3071726865327905, "y": -0.39493684861822875, "z": 0.030556041403429712},
 "surface_normal": {"x": -0.008817844120426742, "y": 0.018648224755218698, "z": 0.9997872220320416},
 "plane_rms_m": 0.0024573550077674903,
 "back_middle_left": {"x": 0.5293395538220405, "y": -0.007923605150875562, "z": 0.02529684521940756},
 "front_middle_left": {"x": 0.38754560468086763, "y": -0.002979776977469817, "z": 0.023954048942268824},
 "back_middle_right": {"x": 0.5171723413498583, "y": -0.4007506109368111, "z": 0.0325166191390871},
 "front_middle_right": {"x": 0.37731981442872237, "y": -0.39684754850926107, "z": 0.031210358227923785},
 "side_cutting_points": {"schema": "side_cutting_midpoints/v1", "source_mapping": {"back_middle_left": ["back_left", "seam_start"]}},
 "cut_permission": {"status": "allowed", "reasons": [], "flags": ["height_waived(support_hidden)"]},
 "mock_restamped": true
})JSON";

Eigen::Vector3d pt(const boost::property_tree::ptree & t, const std::string & key)
{
  const auto & c = t.get_child(key);
  return Eigen::Vector3d(c.get<double>("x"), c.get<double>("y"), c.get<double>("z"));
}

struct TempDir
{
  fs::path path;
  TempDir()
  {
    path = fs::temp_directory_path() / ("topsplit_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
    fs::create_directories(path);
  }
  ~TempDir()
  {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  static int counter;
};
int TempDir::counter = 0;

void writeText(const fs::path & p, const std::string & text)
{
  std::ofstream o(p, std::ios::binary);
  o << text;
}
}  // namespace

// ---- parameters ------------------------------------------------------------------------------

TEST(TopSplitParams, DefaultsAreValidAndModeParses)
{
  TopSplitParams p;
  EXPECT_TRUE(validateTopSplitParams(p));
  EXPECT_EQ(p.mode, TopSplitMode::LeftMax);
  TopSplitMode m;
  EXPECT_TRUE(topSplitModeFromString("half", m));
  EXPECT_EQ(m, TopSplitMode::Half);
  EXPECT_TRUE(topSplitModeFromString("left_max", m));
  EXPECT_EQ(m, TopSplitMode::LeftMax);
  EXPECT_FALSE(topSplitModeFromString("Left_Max", m));
  EXPECT_FALSE(topSplitModeFromString("", m));
  EXPECT_STREQ(toString(TopSplitMode::Half), "half");
}

TEST(TopSplitParams, RangesAreEnforced)
{
  auto bad = [](auto mutate)
  {
    TopSplitParams p;
    mutate(p);
    return !validateTopSplitParams(p).ok;
  };
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.overlap_m = 0.002; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.overlap_m = 0.051; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.overlap_m = std::nan(""); }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.right_min_tail = 0.04; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.right_min_tail = 0.51; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.left_min_fraction = 0.04; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.left_min_fraction = 1.01; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.min_overlap_m = -0.001; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.end_tolerance_m = 0.0; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.end_tolerance_m = 0.06; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.right_end_extension_m = -0.001; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.right_end_extension_m = 0.0201; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.right_end_extension_m = std::nan(""); }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.right_end_extension_m = std::numeric_limits<double>::infinity(); }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.snapshot_file = "relative/file.json"; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.snapshot_file = "/tmp/file.txt"; }));
  EXPECT_TRUE(bad([](TopSplitParams & p) { p.snapshot_file = ""; }));
  // the range edges are valid
  TopSplitParams e;
  e.overlap_m = 0.003;
  e.right_min_tail = 0.5;
  e.left_min_fraction = 1.0;
  e.min_overlap_m = 0.0;
  e.end_tolerance_m = 0.05;
  EXPECT_TRUE(validateTopSplitParams(e));
  e.right_end_extension_m = 0.02;
  EXPECT_TRUE(validateTopSplitParams(e));
  e.right_end_extension_m = 0.005;
  EXPECT_TRUE(validateTopSplitParams(e));
  EXPECT_EQ(TopSplitParams{}.right_end_extension_m, 0.0);   // default off
}

TEST(TopSplitParams, ExtensionOnlyAppliesInLeftMax)
{
  TopSplitParams p;
  p.right_end_extension_m = 0.005;
  p.mode = TopSplitMode::LeftMax;
  EXPECT_EQ(effectiveRightEndExtension(p), 0.005);
  p.mode = TopSplitMode::Half;   // no derived right snapshot in half mode
  EXPECT_EQ(effectiveRightEndExtension(p), 0.0);
}

// ---- fraction --------------------------------------------------------------------------------

TEST(TopSplitFraction, StartsOverlapBeforeTheLeftReach)
{
  TopSplitParams p;
  double f = 0.0;
  // 0.8 reach on a 0.4 m seam, 12 mm overlap -> 77 %
  ASSERT_TRUE(topSplitStartFraction(0.8, 0.4, p, f));
  EXPECT_NEAR(f, 0.77, 1e-12);
}

TEST(TopSplitFraction, Clamping)
{
  TopSplitParams p;
  double f = 0.0;
  ASSERT_TRUE(topSplitStartFraction(1.0, 0.4, p, f));   // would be 0.97: the right still cuts the last 10 %
  EXPECT_NEAR(f, 0.90, 1e-12);
  ASSERT_TRUE(topSplitStartFraction(0.96, 0.4, p, f));
  EXPECT_NEAR(f, 0.90, 1e-12);
  ASSERT_TRUE(topSplitStartFraction(0.06, 0.4, p, f));  // 0.03 -> lower clamp
  EXPECT_NEAR(f, 0.05, 1e-12);
  ASSERT_TRUE(topSplitStartFraction(0.0, 0.4, p, f));
  EXPECT_NEAR(f, 0.05, 1e-12);
  p.right_min_tail = 0.5;
  ASSERT_TRUE(topSplitStartFraction(0.9, 0.4, p, f));
  EXPECT_NEAR(f, 0.5, 1e-12);
  p.right_min_tail = 0.1;
  p.overlap_m = 0.05;
  ASSERT_TRUE(topSplitStartFraction(0.6, 0.25, p, f));   // 0.6 - 0.2
  EXPECT_NEAR(f, 0.4, 1e-12);
}

TEST(TopSplitFraction, RejectsNonsense)
{
  TopSplitParams p;
  double f = 0.0;
  EXPECT_FALSE(topSplitStartFraction(std::nan(""), 0.4, p, f));
  EXPECT_FALSE(topSplitStartFraction(-0.1, 0.4, p, f));
  EXPECT_FALSE(topSplitStartFraction(1.2, 0.4, p, f));
  EXPECT_FALSE(topSplitStartFraction(0.8, 0.0, p, f));
  EXPECT_FALSE(topSplitStartFraction(0.8, 0.04, p, f));
  EXPECT_FALSE(topSplitStartFraction(0.8, std::nan(""), p, f));
  p.overlap_m = 0.0;
  EXPECT_FALSE(topSplitStartFraction(0.8, 0.4, p, f));
}

// ---- sha256 ----------------------------------------------------------------------------------

TEST(Sha256, KnownVectors)
{
  EXPECT_EQ(sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(
    sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // 55 / 56 / 64 bytes: the padding boundaries
  EXPECT_EQ(sha256Hex(std::string(55, 'a')), "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
  EXPECT_EQ(sha256Hex(std::string(56, 'a')), "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
  EXPECT_EQ(sha256Hex(std::string(64, 'a')), "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
}

// ---- derived snapshot ------------------------------------------------------------------------

TEST(DerivedSnapshot, MovesSeamStartAndRecomputesTheTwoMidpoints)
{
  TopSplitMarkerInfo info;
  info.source_file = "/x/src.json";
  info.source_sha256 = "abc123";
  info.left_reach_fraction = 0.8;
  std::string out;
  ASSERT_TRUE(deriveTopSnapshotText(kSnapshot, 0.77, info, out));

  boost::property_tree::ptree src, dst;
  {
    std::istringstream a(kSnapshot), b(out);
    boost::property_tree::read_json(a, src);
    boost::property_tree::read_json(b, dst);   // strict JSON, as the executor parses it
  }
  const Eigen::Vector3d s = pt(src, "seam_start"), e = pt(src, "seam_end");
  const Eigen::Vector3d s2 = pt(dst, "seam_start");
  EXPECT_LT((s2 - (s + 0.77 * (e - s))).norm(), 1e-12);
  EXPECT_LT((pt(dst, "back_middle_left") - 0.5 * (pt(src, "back_left") + s2)).norm(), 1e-12);
  EXPECT_LT((pt(dst, "front_middle_left") - 0.5 * (pt(src, "front_left") + s2)).norm(), 1e-12);
  // untouched
  for (const char * k : {"seam_end", "seam_center", "back_left", "back_right", "front_left", "front_right", "back_middle_right",
                         "front_middle_right", "surface_normal"})
    EXPECT_EQ((pt(src, k) - pt(dst, k)).norm(), 0.0) << k;
  EXPECT_EQ(dst.get<std::string>("snapshot_id"), "capture_frame_000008");
  EXPECT_EQ(dst.get<std::string>("capture_id"), "frame_000008");
  EXPECT_EQ(dst.get<std::string>("frame_id"), "base_link");
  EXPECT_EQ(dst.get<double>("plane_rms_m"), src.get<double>("plane_rms_m"));
  EXPECT_EQ(dst.get<std::string>("cut_permission.status"), "allowed");
  EXPECT_EQ(dst.get<std::string>("mock_restamped"), "true");
  EXPECT_EQ(dst.get<std::string>("side_cutting_points.schema"), "side_cutting_midpoints/v1");
  // the acquisition stamp is kept bit for bit (text and value)
  EXPECT_EQ(dst.get<uint64_t>("acquisition_stamp_ns"), 1792447797748718000ULL);
  EXPECT_NE(out.find("\"acquisition_stamp_ns\": 1792447797748718000"), std::string::npos);
  // marker
  EXPECT_EQ(dst.get<std::string>("dual_cut_coordinator_top_split.source_file"), "/x/src.json");
  EXPECT_EQ(dst.get<std::string>("dual_cut_coordinator_top_split.source_sha256"), "abc123");
  EXPECT_DOUBLE_EQ(dst.get<double>("dual_cut_coordinator_top_split.start_fraction"), 0.77);
  EXPECT_DOUBLE_EQ(dst.get<double>("dual_cut_coordinator_top_split.left_reach_fraction"), 0.8);
  // the key order of the source is kept, the marker is last
  EXPECT_LT(out.find("\"schema\""), out.find("\"seam_start\""));
  EXPECT_LT(out.find("\"seam_end\""), out.find("\"dual_cut_coordinator_top_split\""));
}

TEST(DerivedSnapshot, ZeroExtensionIsIdenticalToTheDefaultAndLeavesTheRightEndAlone)
{
  TopSplitMarkerInfo info, info0;
  info0.right_end_extension_m = 0.0;
  info.source_sha256 = info0.source_sha256 = "abc";
  std::string a, b;
  ASSERT_TRUE(deriveTopSnapshotText(kSnapshot, 0.77, info, a));
  ASSERT_TRUE(deriveTopSnapshotText(kSnapshot, 0.77, info0, b));
  EXPECT_EQ(a, b);
  EXPECT_EQ(a.find("right_end_extension"), std::string::npos);   // marker and file unchanged at 0
  boost::property_tree::ptree src, dst;
  std::istringstream x(kSnapshot), y(a);
  boost::property_tree::read_json(x, src);
  boost::property_tree::read_json(y, dst);
  for (const char * k : {"seam_end", "back_middle_right", "front_middle_right"})
    EXPECT_EQ((pt(src, k) - pt(dst, k)).norm(), 0.0) << k;
}

TEST(DerivedSnapshot, ExtensionMovesSeamEndAlongTheOriginalSeamAndKeepsTheStartLogic)
{
  TopSplitMarkerInfo info;
  info.source_sha256 = "abc";
  info.right_end_extension_m = 0.005;
  std::string out, out0;
  ASSERT_TRUE(deriveTopSnapshotText(kSnapshot, 0.77, info, out));
  TopSplitMarkerInfo info0;
  info0.source_sha256 = "abc";
  ASSERT_TRUE(deriveTopSnapshotText(kSnapshot, 0.77, info0, out0));

  boost::property_tree::ptree src, dst, dst0;
  {
    std::istringstream a(kSnapshot), b(out), c(out0);
    boost::property_tree::read_json(a, src);
    boost::property_tree::read_json(b, dst);
    boost::property_tree::read_json(c, dst0);
  }
  const Eigen::Vector3d s = pt(src, "seam_start"), e = pt(src, "seam_end");
  const Eigen::Vector3d u = (e - s).normalized();   // the ORIGINAL seam direction
  const Eigen::Vector3d e2 = pt(dst, "seam_end");
  EXPECT_NEAR((e2 - e).norm(), 0.005, 1e-12);
  EXPECT_LT((e2 - (e + 0.005 * u)).norm(), 1e-12);
  // the seam start logic is exactly the one without the extension (fraction of the ORIGINAL seam)
  EXPECT_EQ((pt(dst, "seam_start") - pt(dst0, "seam_start")).norm(), 0.0);
  EXPECT_EQ((pt(dst, "back_middle_left") - pt(dst0, "back_middle_left")).norm(), 0.0);
  EXPECT_EQ((pt(dst, "front_middle_left") - pt(dst0, "front_middle_left")).norm(), 0.0);
  // the new seam still points the same way (the executor derives the seam axis, sag and carton axes from it)
  EXPECT_NEAR((e2 - pt(dst, "seam_start")).normalized().dot(u), 1.0, 1e-12);
  // the executor's canonical midpoint formulas (1 um) hold for the moved end
  EXPECT_LT((pt(dst, "back_middle_right") - 0.5 * (pt(src, "back_right") + e2)).norm(), 1e-12);
  EXPECT_LT((pt(dst, "front_middle_right") - 0.5 * (pt(src, "front_right") + e2)).norm(), 1e-12);
  // everything else is untouched
  for (const char * k : {"seam_center", "back_left", "back_right", "front_left", "front_right", "surface_normal"})
    EXPECT_EQ((pt(src, k) - pt(dst, k)).norm(), 0.0) << k;
  EXPECT_EQ(dst.get<uint64_t>("acquisition_stamp_ns"), 1792447797748718000ULL);
  EXPECT_DOUBLE_EQ(dst.get<double>("dual_cut_coordinator_top_split.right_end_extension_m"), 0.005);
  EXPECT_DOUBLE_EQ(dst.get<double>("dual_cut_coordinator_top_split.start_fraction"), 0.77);
}

TEST(DerivedSnapshot, ExtensionOutOfRangeRefusedAndIncompleteSourceFailsClosed)
{
  std::string out;
  for (double bad : {-0.001, 0.0201, std::nan(""), std::numeric_limits<double>::infinity()})
  {
    TopSplitMarkerInfo info;
    info.right_end_extension_m = bad;
    EXPECT_FALSE(deriveTopSnapshotText(kSnapshot, 0.5, info, out)) << bad;
  }
  TopSplitMarkerInfo ok;
  ok.right_end_extension_m = 0.02;
  EXPECT_TRUE(deriveTopSnapshotText(kSnapshot, 0.5, ok, out));
  // a source without back_right cannot be extended (the right side points could not be kept canonical)
  const std::string no_br = R"({"seam_start":{"x":0,"y":0,"z":0},"seam_end":{"x":1,"y":0,"z":0},"back_left":{"x":0,"y":1,"z":0},"front_left":{"x":0,"y":-1,"z":0}})";
  EXPECT_TRUE([&] { TopSplitMarkerInfo i; std::string o; return deriveTopSnapshotText(no_br, 0.5, i, o).ok; }());
  ok.right_end_extension_m = 0.005;
  EXPECT_FALSE(deriveTopSnapshotText(no_br, 0.5, ok, out));
}

TEST(DerivedSnapshot, ExtensionThroughTheFileWriter)
{
  TempDir d;
  const fs::path src = d.path / "src.json", dst = d.path / "out" / "derived.json";
  writeText(src, kSnapshot);
  ASSERT_TRUE(writeDerivedTopSnapshot(src.string(), dst.string(), 0.77, 0.8, std::string(), 0.005));
  std::string text;
  ASSERT_TRUE(readTextFile(dst.string(), text));
  Eigen::Vector3d s, e, s0, e0;
  ASSERT_TRUE(seamFromSnapshotText(text, s, e));
  ASSERT_TRUE(seamFromSnapshotText(kSnapshot, s0, e0));
  EXPECT_NEAR((e - e0).norm(), 0.005, 1e-12);
  EXPECT_NEAR((e - s).norm(), (1.0 - 0.77) * (e0 - s0).norm() + 0.005, 1e-9);
  EXPECT_EQ(fs::last_write_time(dst), fs::last_write_time(src));
}

TEST(DerivedSnapshot, NormalIsBitIdentical)
{
  TopSplitMarkerInfo info;
  std::string out;
  ASSERT_TRUE(deriveTopSnapshotText(kSnapshot, 0.5, info, out));
  EXPECT_NE(out.find("\"x\": -0.008817844120426742"), std::string::npos);
  EXPECT_NE(out.find("\"y\": 0.018648224755218698"), std::string::npos);
  EXPECT_NE(out.find("\"z\": 0.9997872220320416"), std::string::npos);
}

TEST(DerivedSnapshot, FractionZeroOneAndNonsenseRefused)
{
  TopSplitMarkerInfo info;
  std::string out;
  EXPECT_FALSE(deriveTopSnapshotText(kSnapshot, 0.0, info, out));
  EXPECT_FALSE(deriveTopSnapshotText(kSnapshot, 1.0, info, out));
  EXPECT_FALSE(deriveTopSnapshotText(kSnapshot, -0.2, info, out));
  EXPECT_FALSE(deriveTopSnapshotText(kSnapshot, std::nan(""), info, out));
  EXPECT_FALSE(deriveTopSnapshotText("not json", 0.5, info, out));
  EXPECT_FALSE(deriveTopSnapshotText("[1,2]", 0.5, info, out));
  EXPECT_FALSE(deriveTopSnapshotText(R"({"seam_start":{"x":0,"y":0,"z":0}})", 0.5, info, out));
}

TEST(DerivedSnapshot, DoesNotChainAndWithoutHashRecordsSizeAndMtime)
{
  TopSplitMarkerInfo info;
  info.source_size = 1234;
  info.source_mtime_ns = 99;
  std::string out, out2;
  ASSERT_TRUE(deriveTopSnapshotText(kSnapshot, 0.5, info, out));
  EXPECT_NE(out.find("\"source_size\": 1234"), std::string::npos);
  EXPECT_NE(out.find("\"source_mtime_ns\": 99"), std::string::npos);
  EXPECT_EQ(out.find("source_sha256"), std::string::npos);
  EXPECT_FALSE(deriveTopSnapshotText(out, 0.5, info, out2));   // a derived file is never a source
}

TEST(DerivedSnapshot, SeamFromText)
{
  Eigen::Vector3d s, e;
  ASSERT_TRUE(seamFromSnapshotText(kSnapshot, s, e));
  EXPECT_NEAR(s.x(), 0.45868071950053624, 1e-15);
  EXPECT_NEAR(e.y(), -0.39875824840029334, 1e-15);
  EXPECT_FALSE(seamFromSnapshotText("{}", s, e));
}

TEST(DerivedSnapshot, FileKeepsSourceMtimeAndHashAndIsAtomic)
{
  TempDir d;
  const fs::path src = d.path / "src.json", dst = d.path / "sub" / "dir" / "right_top.json";
  writeText(src, kSnapshot);
  const auto old = fs::file_time_type::clock::now() - std::chrono::seconds(100);
  fs::last_write_time(src, old);
  const std::string sha = sha256Hex(kSnapshot);
  ASSERT_TRUE(writeDerivedTopSnapshot(src.string(), dst.string(), 0.77, 0.8, sha));
  EXPECT_EQ(fs::last_write_time(dst), fs::last_write_time(src));   // not fresher than the capture
  std::string text;
  ASSERT_TRUE(readTextFile(dst.string(), text));
  boost::property_tree::ptree t;
  std::istringstream is(text);
  boost::property_tree::read_json(is, t);
  EXPECT_EQ(t.get<std::string>("dual_cut_coordinator_top_split.source_sha256"), sha);
  EXPECT_EQ(t.get<std::string>("dual_cut_coordinator_top_split.source_file"), src.string());
  // no temporary file is left behind
  size_t n = 0;
  for (const auto & e : fs::directory_iterator(dst.parent_path())) { (void)e; ++n; }
  EXPECT_EQ(n, 1u);
  // rewriting replaces it with the new fraction
  ASSERT_TRUE(writeDerivedTopSnapshot(src.string(), dst.string(), 0.5, 0.52, sha));
  ASSERT_TRUE(readTextFile(dst.string(), text));
  EXPECT_NE(text.find("\"start_fraction\": 0.5"), std::string::npos);
  // the source itself is untouched
  std::string s2;
  ASSERT_TRUE(readTextFile(src.string(), s2));
  EXPECT_EQ(s2, kSnapshot);
}

TEST(DerivedSnapshot, RefusesAChangedSourceAndBadPaths)
{
  TempDir d;
  const fs::path src = d.path / "src.json", dst = d.path / "out.json";
  writeText(src, kSnapshot);
  EXPECT_FALSE(writeDerivedTopSnapshot(src.string(), dst.string(), 0.5, 0.5, sha256Hex("something else")));
  EXPECT_FALSE(fs::exists(dst));
  EXPECT_FALSE(writeDerivedTopSnapshot(src.string(), src.string(), 0.5, 0.5));   // never overwrite the source
  std::string kept;
  ASSERT_TRUE(readTextFile(src.string(), kept));
  EXPECT_EQ(kept, kSnapshot);
  EXPECT_FALSE(writeDerivedTopSnapshot((d.path / "missing.json").string(), dst.string(), 0.5, 0.5));
  EXPECT_FALSE(writeDerivedTopSnapshot("rel.json", dst.string(), 0.5, 0.5));
  EXPECT_FALSE(writeDerivedTopSnapshot(src.string(), (d.path / "out.txt").string(), 0.5, 0.5));
}

// ---- coverage --------------------------------------------------------------------------------

TEST(TopCoverage, ProjectionIsDirectionIndependentAndIgnoresOffAxisOffsets)
{
  const Eigen::Vector3d a(0.0, 0.0, 0.0), b(0.0, -0.4, 0.0);   // seam along -y, 400 mm
  SeamInterval iv;
  CutLine fwd{Eigen::Vector3d(0.01, -0.0, 0.003), Eigen::Vector3d(0.02, -0.32, -0.004)};
  ASSERT_TRUE(projectCutLineOnSeam(fwd, a, b, iv));
  EXPECT_NEAR(iv.s0, 0.0, 1e-12);
  EXPECT_NEAR(iv.s1, 0.32, 1e-12);
  CutLine rev{Eigen::Vector3d(0.0, -0.4, 0.0), Eigen::Vector3d(0.0, -0.308, 0.0)};
  ASSERT_TRUE(projectCutLineOnSeam(rev, a, b, iv));
  EXPECT_NEAR(iv.s0, 0.308, 1e-12);
  EXPECT_NEAR(iv.s1, 0.4, 1e-12);
  EXPECT_FALSE(projectCutLineOnSeam(fwd, a, a, iv));
}

TEST(TopCoverage, LeftMaxExample)
{
  const double L = 0.4;
  TopCoverage c;
  // left 0..80 %, right 77..100 %: overlap 12 mm
  ASSERT_TRUE(checkTopCoverage({0.0, 0.32}, {0.308, 0.4}, L, 0.005, 0.005, &c));
  EXPECT_NEAR(c.overlap_m, 0.012, 1e-12);
  EXPECT_EQ(describeTopCoverage(c), "left 0.0..80.0 %, right 77.0..100.0 %, overlap 12.0 mm");
  const std::string j = topCoverageJson(c);
  EXPECT_NE(j.find("\"overlap_mm\":12"), std::string::npos);
}

TEST(TopCoverage, HalfModeShapeAlsoPasses)
{
  // half / half around the centre with 7.5 mm per arm
  EXPECT_TRUE(checkTopCoverage({0.0, 0.2075}, {0.1925, 0.4}, 0.4, 0.005, 0.005));
}

TEST(TopCoverage, GapIsRefusedAndNamesBothIntervals)
{
  TopCoverage c;
  const Status s = checkTopCoverage({0.0, 0.32}, {0.325, 0.4}, 0.4, 0.005, 0.005, &c);
  EXPECT_FALSE(s);
  EXPECT_LT(c.overlap_m, 0.0);
  EXPECT_NE(s.message.find("GAP"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("0.0..320.0 mm"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("325.0..400.0 mm"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("0.0..80.0 %"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("81.2..100.0 %"), std::string::npos) << s.message;
}

TEST(TopCoverage, ExactTouchPassesOnlyWithZeroRequiredOverlap)
{
  EXPECT_TRUE(checkTopCoverage({0.0, 0.32}, {0.32, 0.4}, 0.4, 0.005, 0.0));
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.32, 0.4}, 0.4, 0.005, 0.005));
  // overlap exactly the required value passes
  EXPECT_TRUE(checkTopCoverage({0.0, 0.32}, {0.315, 0.4}, 0.4, 0.005, 0.005 - 1e-12));
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.3151, 0.4}, 0.4, 0.005, 0.0051));
}

TEST(TopCoverage, EndsMustBeReached)
{
  const double L = 0.4;
  // left starts 8 mm in
  EXPECT_FALSE(checkTopCoverage({0.008, 0.32}, {0.30, 0.4}, L, 0.005, 0.005));
  // right stops 8 mm short
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.30, 0.392}, L, 0.005, 0.005));
  // right overshoots 8 mm
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.30, 0.408}, L, 0.005, 0.005));
  // inside the tolerance on both ends
  EXPECT_TRUE(checkTopCoverage({-0.004, 0.32}, {0.30, 0.403}, L, 0.005, 0.005));
}

TEST(TopCoverage, ExtendedRightEndIsExpectedAtSeamEndPlusExtension)
{
  const double L = 0.4, ext = 0.005;
  TopCoverage c;
  // the right now reaches 5 mm beyond the seam end: passes with the extension, and the report says so
  ASSERT_TRUE(checkTopCoverage({0.0, 0.32}, {0.308, L + ext}, L, 0.005, 0.005, &c, ext));
  EXPECT_NEAR(c.overlap_m, 0.012, 1e-12);
  EXPECT_NE(describeTopCoverage(c).find("right end 5.0 mm beyond the seam end"), std::string::npos) << describeTopCoverage(c);
  EXPECT_NE(topCoverageJson(c).find("\"right_end_extension_mm\":5"), std::string::npos);
  // overlap is unchanged by the extension
  TopCoverage c0;
  ASSERT_TRUE(checkTopCoverage({0.0, 0.32}, {0.308, L}, L, 0.005, 0.005, &c0));
  EXPECT_DOUBLE_EQ(c.overlap_m, c0.overlap_m);
  EXPECT_EQ(describeTopCoverage(c0), "left 0.0..80.0 %, right 77.0..100.0 %, overlap 12.0 mm");   // 0: text unchanged
  EXPECT_EQ(topCoverageJson(c0).find("extension"), std::string::npos);
  // still exact at the new end: 6 mm short of L + ext fails, 4 mm over passes (tolerance unchanged)
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.308, L + ext - 0.006}, L, 0.005, 0.005, nullptr, ext));
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.308, L + ext + 0.006}, L, 0.005, 0.005, nullptr, ext));
  EXPECT_TRUE(checkTopCoverage({0.0, 0.32}, {0.308, L + ext + 0.004}, L, 0.005, 0.005, nullptr, ext));
  // a right that ends 10 mm beyond the seam end does not pass for a 5 mm extension; nor an unextended 0 expectation
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.308, L + 0.010 + 0.001}, L, 0.005, 0.005, nullptr, ext));
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.308, L + ext}, L, 0.001, 0.005, nullptr, 0.0));
  const Status s = checkTopCoverage({0.0, 0.32}, {0.308, L + ext + 0.02}, L, 0.005, 0.005, nullptr, ext);
  EXPECT_FALSE(s);
  EXPECT_NE(s.message.find("extension"), std::string::npos) << s.message;
  // invalid extension fails closed
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.308, L}, L, 0.005, 0.005, nullptr, -0.001));
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.308, L}, L, 0.005, 0.005, nullptr, 0.03));
  EXPECT_FALSE(checkTopCoverage({0.0, 0.32}, {0.308, L}, L, 0.005, 0.005, nullptr, std::nan("")));
}

TEST(TopCoverage, NonFiniteFailsClosed)
{
  EXPECT_FALSE(checkTopCoverage({std::nan(""), 0.3}, {0.2, 0.4}, 0.4, 0.005, 0.005));
  EXPECT_FALSE(checkTopCoverage({0.0, 0.3}, {0.2, 0.4}, 0.0, 0.005, 0.005));
}

TEST(TopSag, GeometryFromSnapshotAndWorldTransform)
{
  // carton 0.4 m along x (back x=0.1..0.5), seam along +x at y=0, normal +z
  const std::string doc = R"({"seam_start":{"x":0.1,"y":0.0,"z":0.0},"seam_end":{"x":0.5,"y":0.0,"z":0.0},"seam_center":{"x":0.3,"y":0.0,"z":0.0},
    "surface_normal":{"x":0.0,"y":0.0,"z":1.0},"back_left":{"x":0.1,"y":0.1,"z":0.0},"front_left":{"x":0.1,"y":-0.1,"z":0.0},
    "back_right":{"x":0.5,"y":0.1,"z":0.0},"front_right":{"x":0.5,"y":-0.1,"z":0.0}})";
  SnapshotSagGeometry g;
  ASSERT_TRUE(sagGeometryFromSnapshotText(doc, g));
  CutSag sag;
  ASSERT_TRUE(makeCutSag(g, Eigen::Isometry3d::Identity(), 0.002, sag));
  EXPECT_TRUE(sag.active);
  EXPECT_NEAR(sag.half_length_m, 0.2, 1e-12);
  EXPECT_NEAR((sag.u - Eigen::Vector3d::UnitX()).norm(), 0.0, 1e-12);
  EXPECT_NEAR((sag.down + Eigen::Vector3d::UnitZ()).norm(), 0.0, 1e-12);
  EXPECT_NEAR((sag.center - Eigen::Vector3d(0.3, 0, 0)).norm(), 0.0, 1e-12);
  // the same profile after a rigid transform into the world (90 deg about z, shifted)
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T.translation() = Eigen::Vector3d(1, 2, 3);
  CutSag w;
  ASSERT_TRUE(makeCutSag(g, T, 0.002, w));
  EXPECT_NEAR(w.half_length_m, 0.2, 1e-12);
  EXPECT_NEAR((w.u - Eigen::Vector3d::UnitY()).norm(), 0.0, 1e-12);
  EXPECT_NEAR((w.center - T * Eigen::Vector3d(0.3, 0, 0)).norm(), 0.0, 1e-12);
  // sag 0 -> inactive; invalid sag and a degenerate carton fail closed
  CutSag z;
  ASSERT_TRUE(makeCutSag(g, Eigen::Isometry3d::Identity(), 0.0, z));
  EXPECT_FALSE(z.active);
  EXPECT_FALSE(makeCutSag(g, Eigen::Isometry3d::Identity(), -0.001, z));
  EXPECT_FALSE(makeCutSag(g, Eigen::Isometry3d::Identity(), std::nan(""), z));
  SnapshotSagGeometry flat = g;
  flat.back_right = flat.back_left;
  flat.front_right = flat.front_left;
  EXPECT_FALSE(makeCutSag(flat, Eigen::Isometry3d::Identity(), 0.002, z));
  EXPECT_FALSE(sagGeometryFromSnapshotText("{\"seam_start\":{\"x\":0,\"y\":0,\"z\":0}}", g));
}
