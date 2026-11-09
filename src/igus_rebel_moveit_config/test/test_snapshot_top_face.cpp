// Tests for the top-face check of validateSnapshot (snapshot_top_face.hpp), including the coordinator
// top-split right-end extension.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "igus_rebel_moveit_config/snapshot_top_face.hpp"

namespace
{
using Eigen::Vector3d;
using namespace snapshot_top_face;

// Carton top face 0.40 m (x, back_left -> back_right) by 0.30 m (y, back_left -> front_left), normal +z.
// The seam runs along +x at y = 0.15 from the left edge to the right edge.
struct Carton {
  Face face{Vector3d(0, 0, 0), Vector3d(0.4, 0, 0), Vector3d(0, 0.3, 0), Vector3d(0, 0, 1)};
  Vector3d seam_start{0.0, 0.15, 0.0};
  Vector3d seam_end{0.4, 0.15, 0.0};
  Vector3d seam_center{0.2, 0.15, 0.0};
  Vector3d back_right{0.4, 0.0, 0.0};
  Vector3d front_right{0.4, 0.3, 0.0};
  Vector3d back_left{0.0, 0.0, 0.0};
  Vector3d front_left{0.0, 0.3, 0.0};

  // The side points keep their canonical midpoint formulas, like the coordinator writes them.
  Result check(const Vector3d & start, const Vector3d & end, const Marker & marker) const
  {
    return checkSeamAndSidePoints(
      face, start, end, seam_center, 0.5 * (back_left + start), 0.5 * (front_left + start),
      0.5 * (back_right + end), 0.5 * (front_right + end), front_right, marker);
  }
};

Marker derived(double extension, ExtendedEnd end = ExtendedEnd::kSeamEnd)
{
  Marker m;
  m.present = true;
  m.extension_m = extension;
  m.extended = end;
  return m;
}
}  // namespace

TEST(SnapshotTopFace, NormalSnapshotKeepsTheTenMillimetreTolerance)
{
  const Carton c;
  EXPECT_EQ(c.check(c.seam_start, c.seam_end, Marker{}), Result::kOk);
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.409, 0.15, 0.0), Marker{}), Result::kOk);
  // 15 mm past the edge: refused (the side points at 7.5 mm would pass, the seam end does not)
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.415, 0.15, 0.0), Marker{}), Result::kOutsideTopFace);
  // plane distance is unchanged
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.4, 0.15, 0.011), Marker{}), Result::kOutsideTopFace);
}

TEST(SnapshotTopFace, DerivedSnapshotAllowsTheExtensionAlongTheSeamOnly)
{
  const Carton c;
  const Marker m = derived(0.015);
  // 15 mm along the seam beyond the edge: accepted (ordinary rule would refuse)
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.415, 0.15, 0.0), m), Result::kOk);
  // The seam end alone may reach 10 mm + extension = 25 mm past the edge (side points on the face here) ...
  const auto end_only = [&](double x, const Marker & marker) {
    return checkSeamAndSidePoints(
      c.face, c.seam_start, Vector3d(x, 0.15, 0.0), c.seam_center, Vector3d(0, 0.075, 0),
      Vector3d(0, 0.225, 0), Vector3d(0.39, 0.075, 0), Vector3d(0.39, 0.225, 0), c.front_right, marker);
  };
  EXPECT_EQ(end_only(0.4249, m), Result::kOk);
  // ... 26 mm is refused, and so is 15 mm without the marker
  EXPECT_EQ(end_only(0.426, m), Result::kOutsideTopFace);
  EXPECT_EQ(end_only(0.415, Marker{}), Result::kOutsideTopFace);
  // The canonical side points (midpoints of corner and seam end) follow the seam end but are judged at their
  // pre-extension position (end - 15 mm), so 25 mm is fine too; without the marker it is refused.
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.425, 0.15, 0.0), m), Result::kOk);
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.425, 0.15, 0.0), Marker{}), Result::kOutsideTopFace);
  // 15 mm SIDEWAYS (beyond the y edge, seam running close to that edge) is refused even with the marker
  const Vector3d edge_start(0.0, 0.28, 0.0);
  EXPECT_EQ(c.check(edge_start, Vector3d(0.4, 0.28, 0.0), m), Result::kOk);
  EXPECT_EQ(c.check(edge_start, Vector3d(0.4, 0.325, 0.0), m), Result::kOutsideTopFace);
  EXPECT_EQ(c.check(edge_start, Vector3d(0.415, 0.325, 0.0), m), Result::kOutsideTopFace);
  EXPECT_EQ(c.check(edge_start, Vector3d(0.4, 0.325, 0.0), Marker{}), Result::kOutsideTopFace);
  // off the plane is refused
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.415, 0.15, 0.011), m), Result::kOutsideTopFace);
}

TEST(SnapshotTopFace, ExtensionOnlyReachesTheExtendedEnd)
{
  const Carton c;
  const Marker m = derived(0.015);
  // seam_start 15 mm outside is NOT covered when seam_end is the extended end
  EXPECT_EQ(c.check(Vector3d(-0.015, 0.15, 0.0), c.seam_end, m), Result::kOutsideTopFace);
  // ... but it is when the relabelled snapshot names seam_start as the extended end
  EXPECT_EQ(c.check(Vector3d(-0.015, 0.15, 0.0), c.seam_end, derived(0.015, ExtendedEnd::kSeamStart)), Result::kOk);
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.415, 0.15, 0.0), derived(0.015, ExtendedEnd::kSeamStart)),
            Result::kOutsideTopFace);
  // a marker without an extension changes nothing
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.415, 0.15, 0.0), derived(0.0)), Result::kOutsideTopFace);
}

TEST(SnapshotTopFace, SideCuttingPointsKeepTheNormalRule)
{
  const Carton c;
  // the extended end is fine, but a side point 15 mm outside is not
  const Result r = checkSeamAndSidePoints(
    c.face, c.seam_start, Vector3d(0.415, 0.15, 0.0), c.seam_center, Vector3d(0, 0.075, 0),
    Vector3d(0, 0.225, 0), Vector3d(0.415, 0.075, 0), c.front_right, c.front_right, derived(0.015));
  EXPECT_EQ(r, Result::kOutsideTopFace);
}

TEST(SnapshotTopFace, AngledSeamIsExtendedAlongItsOwnDirection)
{
  Carton c;
  // seam 10 degrees off the x axis, leaving the right edge at y = 0.15 + 0.4 * tan(10 deg)
  const double a = 10.0 * M_PI / 180.0;
  const Vector3d start(0.0, 0.15, 0.0);
  const Vector3d dir(std::cos(a), std::sin(a), 0.0);
  const double edge_len = 0.4 / std::cos(a);  // reaches x = 0.4
  const Vector3d on_edge = start + edge_len * dir;
  // 15 mm along the seam past the edge point
  EXPECT_EQ(c.check(start, on_edge + 0.015 * dir, derived(0.015)), Result::kOk);
  // 15 mm sideways (perpendicular to the seam) from the edge point: not reachable by walking back along it
  const Vector3d perp(-dir.y(), dir.x(), 0.0);
  EXPECT_EQ(c.check(start, on_edge + 0.05 * dir + 0.015 * perp, derived(0.015)), Result::kOutsideTopFace);
}

TEST(SnapshotTopFace, InvalidMarkerExtensionIsRefused)
{
  const Carton c;
  // even a perfectly ordinary seam end: a marker with a bad extension fails closed
  EXPECT_EQ(c.check(c.seam_start, c.seam_end, derived(0.03)), Result::kInvalidExtension);
  EXPECT_EQ(c.check(c.seam_start, c.seam_end, derived(std::numeric_limits<double>::quiet_NaN())),
            Result::kInvalidExtension);
  EXPECT_EQ(c.check(c.seam_start, c.seam_end, derived(-0.001)), Result::kInvalidExtension);
  EXPECT_EQ(c.check(c.seam_start, c.seam_end, derived(std::numeric_limits<double>::infinity())),
            Result::kInvalidExtension);
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.415, 0.15, 0.0), derived(0.03)), Result::kInvalidExtension);
  EXPECT_TRUE(extensionValid(0.0));
  EXPECT_TRUE(extensionValid(0.02));
  EXPECT_FALSE(extensionValid(0.0201));
}

TEST(SnapshotTopFace, DegenerateSeamDirectionIsRefused)
{
  const Carton c;
  // the extended end coincides with the other end: no direction to extend along
  EXPECT_FALSE(extendedEndOnTopFace(c.face, Vector3d(0.415, 0.15, 0.0), Vector3d(0.415, 0.15, 0.0), 0.015));
}

// Side midpoints of the extended end are judged at their pre-extension position (shifted back by ext / 2).
TEST(SnapshotTopFace, ExtendedEndSideMidpointsAreCheckedBeforeTheExtension)
{
  const Carton c;
  const Marker m = derived(0.015);
  // seam end 21 mm past the edge (original 6 mm): the canonical midpoints sit at 10.5 mm, 0.5 mm over the tolerance
  const Vector3d end(0.421, 0.15, 0.0);
  EXPECT_EQ(c.check(c.seam_start, end, Marker{}), Result::kOutsideTopFace);  // the old behaviour of the midpoints
  EXPECT_EQ(c.check(c.seam_start, end, m), Result::kOk);
  // up to the limit of the seam end rule (25 mm past the edge, original 10 mm, midpoints originally 5 mm)
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.4249, 0.15, 0.0), m), Result::kOk);
  EXPECT_EQ(c.check(c.seam_start, Vector3d(0.426, 0.15, 0.0), m), Result::kOutsideTopFace);
  // the relabelled (rotate_180) file names seam_start as the extended end; its side points are the *_left ones
  const Marker ms = derived(0.015, ExtendedEnd::kSeamStart);
  const Vector3d start(-0.021, 0.15, 0.0);
  EXPECT_EQ(c.check(start, c.seam_end, Marker{}), Result::kOutsideTopFace);
  EXPECT_EQ(c.check(start, c.seam_end, ms), Result::kOk);
  // the marker names the other end: nothing is relaxed
  EXPECT_EQ(c.check(c.seam_start, end, ms), Result::kOutsideTopFace);
  // an angled seam: shifted back along the seam axis, not along a face axis
  const double a = 10.0 * M_PI / 180.0;
  const Vector3d dir(std::cos(a), std::sin(a), 0.0);
  const Vector3d on_edge = Vector3d(0.0, 0.15, 0.0) + (0.4 / std::cos(a)) * dir;
  EXPECT_EQ(c.check(Vector3d(0.0, 0.15, 0.0), on_edge + 0.021 * dir, m), Result::kOk);
}

TEST(SnapshotTopFace, ExtendedEndMidpointTrulyOutsideStillFails)
{
  const Carton c;
  const Marker m = derived(0.015);
  const Vector3d end(0.421, 0.15, 0.0);
  // front_right corner 40 mm outside the face: the pre-extension midpoint is 0.5 * (0.44 + 0.406) = 0.423, outside
  const Vector3d far_corner(0.44, 0.3, 0.0);
  const auto run = [&](const Marker & marker) {
    return checkSeamAndSidePoints(
      c.face, c.seam_start, end, c.seam_center, 0.5 * (c.back_left + c.seam_start), 0.5 * (c.front_left + c.seam_start),
      0.5 * (c.back_right + end), 0.5 * (far_corner + end), far_corner, marker);
  };
  EXPECT_EQ(run(m), Result::kOutsideTopFace);
  EXPECT_EQ(run(Marker{}), Result::kOutsideTopFace);
  // an off-plane midpoint is not helped by the shift
  const auto off_plane = [&](const Marker & marker) {
    return checkSeamAndSidePoints(
      c.face, c.seam_start, end, c.seam_center, 0.5 * (c.back_left + c.seam_start), 0.5 * (c.front_left + c.seam_start),
      0.5 * (c.back_right + end), 0.5 * (c.front_right + end) + Vector3d(0, 0, 0.012), c.front_right, marker);
  };
  EXPECT_EQ(off_plane(m), Result::kOutsideTopFace);
}

TEST(SnapshotTopFace, TamperedExtendedEndMidpointIsNotRelaxed)
{
  const Carton c;
  const Marker m = derived(0.015);
  const Vector3d end(0.421, 0.15, 0.0);
  // canonical midpoint is 0.4105; 0.4112 differs by 0.7 mm: checked as written (over the tolerance), not shifted back
  const auto run = [&](const Vector3d & back_middle_right, const Vector3d & front_middle_right) {
    return checkSeamAndSidePoints(
      c.face, c.seam_start, end, c.seam_center, 0.5 * (c.back_left + c.seam_start), 0.5 * (c.front_left + c.seam_start),
      back_middle_right, front_middle_right, c.front_right, m);
  };
  EXPECT_EQ(run(Vector3d(0.4105, 0.075, 0), Vector3d(0.4105, 0.225, 0)), Result::kOk);
  EXPECT_EQ(run(Vector3d(0.4112, 0.075, 0), Vector3d(0.4105, 0.225, 0)), Result::kOutsideTopFace);
  EXPECT_EQ(run(Vector3d(0.4105, 0.075, 0), Vector3d(0.4112, 0.225, 0)), Result::kOutsideTopFace);
  // a midpoint that is the canonical one for the ORIGINAL end (not the extended one) is checked as written
  EXPECT_EQ(run(Vector3d(0.403, 0.075, 0), Vector3d(0.403, 0.225, 0)), Result::kOk);
  // helper: shifted back only when canonical
  const Vector3d shifted = sideMidpointBeforeExtension(
    Vector3d(0.4105, 0.075, 0), c.back_right, end, c.seam_start, 0.015);
  EXPECT_NEAR(shifted.x(), 0.403, 1e-12);
  EXPECT_EQ(sideMidpointBeforeExtension(Vector3d(0.4112, 0.075, 0), c.back_right, end, c.seam_start, 0.015).x(), 0.4112);
}

TEST(SnapshotTopFace, MidpointRelaxationNeedsAValidPositiveExtension)
{
  const Carton c;
  const Vector3d end(0.421, 0.15, 0.0);
  EXPECT_EQ(c.check(c.seam_start, end, derived(0.0201)), Result::kInvalidExtension);
  EXPECT_EQ(c.check(c.seam_start, end, derived(0.03)), Result::kInvalidExtension);
  EXPECT_EQ(c.check(c.seam_start, end, derived(-0.015)), Result::kInvalidExtension);
  EXPECT_EQ(c.check(c.seam_start, end, derived(std::numeric_limits<double>::quiet_NaN())), Result::kInvalidExtension);
  // marker without an extension (0): behaves like an unmarked file
  EXPECT_EQ(c.check(c.seam_start, end, derived(0.0)), Result::kOutsideTopFace);
  // midpoints alone over the tolerance with ext = 0: unchanged
  const auto run = [&](const Marker & marker) {
    return checkSeamAndSidePoints(
      c.face, c.seam_start, c.seam_end, c.seam_center, 0.5 * (c.back_left + c.seam_start),
      0.5 * (c.front_left + c.seam_start), Vector3d(0.4105, 0.075, 0), Vector3d(0.4105, 0.225, 0), c.front_right, marker);
  };
  EXPECT_EQ(run(Marker{}), Result::kOutsideTopFace);
  EXPECT_EQ(run(derived(0.0)), Result::kOutsideTopFace);
}
