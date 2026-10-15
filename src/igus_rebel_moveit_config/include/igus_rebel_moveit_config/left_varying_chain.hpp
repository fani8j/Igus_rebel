// Pure (ROS/MoveIt/Eigen-free) helpers for the opt-in varying-orientation LEFT side cut
// (box_h_cut_executor parameter left_parallel_varying_orientation_enabled).
//
// The left cut line is sampled at stations s_0..s_N. At each station the executor evaluates
// the same orientation candidate set as the fixed left_parallel search and keeps, per
// feasible (candidate, IK branch), a Node. ChainDp then finds, over consecutive stations,
// the longest CONTIGUOUS chain of nodes such that consecutive nodes
//   - differ in tool orientation by at most the orientation step limit (the candidate
//     neighbour table, built by buildNeighborTable), and
//   - stay on one continuous joint branch (max per-joint change <= max_joint_step_rad, which
//     also excludes every joint4/joint6 wrap),
// and, among chains of equal length, the one with the smallest summed score (the per-candidate
// score plus a small smoothness term per orientation step). Nothing here decides feasibility
// of a node: the executor does that with collision-aware IK before calling addStation().
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace left_varying
{

// Number of intervals N so that N + 1 stations (both end points included) are at most
// `spacing_m` apart. Always at least 1. A non-finite/non-positive input yields 1.
inline std::size_t stationIntervals(double length_m, double spacing_m)
{
  if (!(length_m > 0.0) || !(spacing_m > 0.0) || !std::isfinite(length_m) ||
      !std::isfinite(spacing_m)) {
    return 1U;
  }
  const double raw = std::ceil(length_m / spacing_m - 1e-9);
  return raw < 1.0 ? 1U : static_cast<std::size_t>(raw);
}

// ---- left_cut_extend_to_corners geometry (pure scalars) ----------------------------------
// All coordinates are arc positions `s` (metres) along the NOMINAL left cut line, with s = 0 at
// back_middle_left and s = nominal_length_m at front_middle_left; the corners and the seam
// crossing are projected onto that line.
struct ExtendedSpan
{
  bool ok{false};
  std::string error;
  double s_start{0.0}, s_end{0.0};       // span of the extended candidate line
  double length_m{0.0};                  // s_end - s_start
  double band_lo{0.0}, band_hi{0.0};     // required seam band, clamped to the span
};
// The span runs from the back corner moved `margin_m` toward the front corner to the front
// corner moved `margin_m` toward the back corner. Fails closed (ok = false, error set) when an
// input is not finite, the corners are not ordered back -> front, the span is degenerate or
// shorter than the nominal segment or does not contain it, or the seam crossing lies outside
// the span. The seam band is [s_seam - seam_cover_m, s_seam + seam_cover_m] clamped to the span.
inline ExtendedSpan computeExtendedSpan(
  double nominal_length_m, double s_back_corner, double s_front_corner, double s_seam,
  double margin_m, double seam_cover_m)
{
  ExtendedSpan span;
  for (const double v : {nominal_length_m, s_back_corner, s_front_corner, s_seam, margin_m,
                         seam_cover_m}) {
    if (!std::isfinite(v)) {
      span.error = "non-finite extended-span input";
      return span;
    }
  }
  if (!(nominal_length_m > 0.0) || !(margin_m >= 0.0) || !(seam_cover_m >= 0.0)) {
    span.error = "degenerate extended-span input (non-positive nominal length or negative margin/cover)";
    return span;
  }
  if (!(s_front_corner > s_back_corner)) {
    span.error = "the back and front corners are not ordered along the nominal back -> front cut direction";
    return span;
  }
  span.s_start = s_back_corner + margin_m;
  span.s_end = s_front_corner - margin_m;
  span.length_m = span.s_end - span.s_start;
  if (!(span.length_m > 0.0)) {
    span.error = "the extended span is degenerate (corner margin leaves no length)";
    return span;
  }
  constexpr double kSlack = 1e-6;
  if (span.length_m < nominal_length_m - kSlack || span.s_start > kSlack ||
      span.s_end < nominal_length_m - kSlack) {
    span.error = "the extended span does not contain the nominal segment (extended span " +
      std::to_string(span.length_m * 1000.0) + " mm, nominal " +
      std::to_string(nominal_length_m * 1000.0) + " mm)";
    return span;
  }
  if (s_seam < span.s_start - kSlack || s_seam > span.s_end + kSlack) {
    span.error = "the seam crossing projects outside the extended span";
    return span;
  }
  span.band_lo = std::max(span.s_start, s_seam - seam_cover_m);
  span.band_hi = std::min(span.s_end, s_seam + seam_cover_m);
  span.ok = true;
  return span;
}

// Station indices (0..intervals, spaced length_m / intervals from the span start) that bound a
// chain which covers the arc band [band_lo_m, band_hi_m] (relative to the span start): the chain's
// FIRST station must be <= first_max and its LAST station >= last_min.
struct RequiredStations
{
  std::size_t first_max{0}, last_min{0};
};
inline RequiredStations requiredStationRange(
  double band_lo_m, double band_hi_m, double length_m, std::size_t intervals)
{
  RequiredStations range;
  if (intervals == 0U || !(length_m > 0.0)) return range;
  const double spacing = length_m / static_cast<double>(intervals);
  const double lo = std::floor(band_lo_m / spacing + 1e-9);
  const double hi = std::ceil(band_hi_m / spacing - 1e-9);
  range.first_max = lo < 0.0 ? 0U : std::min(static_cast<std::size_t>(lo), intervals);
  range.last_min = hi < 0.0 ? 0U : std::min(static_cast<std::size_t>(hi), intervals);
  return range;
}

// Rotation angle (degrees) between two unit quaternions given as {x, y, z, w}.
inline double quaternionAngleDeg(const std::array<double, 4>& a, const std::array<double, 4>& b)
{
  const double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const double clamped = std::min(1.0, std::abs(dot));
  return 2.0 * std::acos(clamped) * (180.0 / 3.14159265358979323846);
}

// neighbours[c] = every candidate c2 (c itself included, step 0) whose orientation differs
// from c by at most max_step_deg (+1e-6 deg slack for floating point), with that angle.
using NeighborTable = std::vector<std::vector<std::pair<std::size_t, double> > >;
// Quadratic in the candidate count, so the caller supplies an abort predicate that is polled
// once per candidate row; returns false (and leaves `table` incomplete) when it fires.
inline bool buildNeighborTableBounded(
  const std::vector<std::array<double, 4> >& orientations, double max_step_deg,
  NeighborTable& table, const std::function<bool()>& should_abort)
{
  table.assign(orientations.size(), {});
  const double limit = max_step_deg + 1e-6;
  // Cheap pre-filter on |dot| = cos(angle / 2) so acos only runs for real neighbours.
  const double min_dot = std::cos(0.5 * limit * 3.14159265358979323846 / 180.0) - 1e-9;
  for (std::size_t a = 0; a < orientations.size(); ++a) {
    if (should_abort && should_abort()) return false;
    for (std::size_t b = 0; b < orientations.size(); ++b) {
      if (a != b) {
        const auto& qa = orientations[a];
        const auto& qb = orientations[b];
        if (std::abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]) < min_dot) {
          continue;
        }
      }
      const double angle = a == b ? 0.0 : quaternionAngleDeg(orientations[a], orientations[b]);
      if (angle <= limit) table[a].emplace_back(b, angle);
    }
  }
  return true;
}
inline NeighborTable buildNeighborTable(
  const std::vector<std::array<double, 4> >& orientations, double max_step_deg)
{
  NeighborTable table;
  buildNeighborTableBounded(orientations, max_step_deg, table, {});
  return table;
}

struct Node
{
  std::size_t candidate{0};
  std::vector<double> q;  // joint values of this IK branch
  double score{0.0};      // per-candidate score (smaller is better)
};

class ChainDp
{
public:
  struct Config
  {
    double max_joint_step_rad{0.2};
    double smooth_weight{0.0};  // score added per degree of orientation step
  };
  // start_ok(station, node_index, node): may a chain begin at this node (only asked for nodes
  // that have no valid predecessor chain). edge_ok(station, from_node, to_node): the edge from
  // the node at `station` to the node at station + 1 is not banned. Either may be empty
  // (= always). Node indices are positions in the vector passed to addStation().
  using StartOkFn = std::function<bool(std::size_t, std::size_t, const Node&)>;
  using EdgeOkFn = std::function<bool(std::size_t, std::size_t, std::size_t)>;

  ChainDp(const Config& config, const NeighborTable& neighbors, StartOkFn start_ok, EdgeOkFn edge_ok)
  : config_(config), neighbors_(&neighbors), start_ok_(std::move(start_ok)),
    edge_ok_(std::move(edge_ok)) {}

  // Appends the next station (station index = stationCount() before the call). Relaxes from
  // the previous station; nodes left without a predecessor chain start a new chain when
  // allow_new_starts and start_ok() agree.
  void addStation(std::vector<Node> nodes, bool allow_new_starts)
  {
    const std::size_t station = stations_.size();
    Station current;
    current.nodes = std::move(nodes);
    current.len.assign(current.nodes.size(), 0U);
    current.score.assign(current.nodes.size(), 0.0);
    current.prev.assign(current.nodes.size(), kNone);
    for (std::size_t i = 0; i < current.nodes.size(); ++i) {
      current.by_candidate[current.nodes[i].candidate].push_back(i);
    }
    if (station > 0U) {
      const Station& previous = stations_[station - 1U];
      for (std::size_t a = 0; a < previous.nodes.size(); ++a) {
        if (previous.len[a] == 0U) continue;
        const Node& from = previous.nodes[a];
        if (from.candidate >= neighbors_->size()) continue;
        for (const auto& [candidate, step_deg] : (*neighbors_)[from.candidate]) {
          const auto found = current.by_candidate.find(candidate);
          if (found == current.by_candidate.end()) continue;
          for (const std::size_t b : found->second) {
            if (!jointsContinuous(from.q, current.nodes[b].q)) continue;
            if (edge_ok_ && !edge_ok_(station - 1U, a, b)) continue;
            const std::size_t length = previous.len[a] + 1U;
            const double score = previous.score[a] + current.nodes[b].score +
              config_.smooth_weight * step_deg;
            if (length > current.len[b] ||
                (length == current.len[b] && score < current.score[b] - 1e-12)) {
              current.len[b] = length;
              current.score[b] = score;
              current.prev[b] = a;
            }
          }
        }
      }
    }
    if (allow_new_starts) {
      for (std::size_t i = 0; i < current.nodes.size(); ++i) {
        if (current.len[i] != 0U) continue;
        if (start_ok_ && !start_ok_(station, i, current.nodes[i])) continue;
        current.len[i] = 1U;
        current.score[i] = current.nodes[i].score;
        current.prev[i] = kNone;
      }
    }
    for (const std::size_t length : current.len) best_length_ = std::max(best_length_, length);
    stations_.push_back(std::move(current));
  }

  std::size_t stationCount() const { return stations_.size(); }
  // Longest chain (in stations) found so far, ending at any station.
  std::size_t bestLength() const { return best_length_; }
  const Node& node(std::size_t station, std::size_t index) const
  {
    return stations_[station].nodes[index];
  }
  // Candidates that have a valid chain ending at the most recently added station.
  std::vector<std::size_t> aliveCandidates() const
  {
    std::vector<std::size_t> alive;
    if (stations_.empty()) return alive;
    const Station& last = stations_.back();
    for (std::size_t i = 0; i < last.nodes.size(); ++i) {
      if (last.len[i] > 0U) alive.push_back(last.nodes[i].candidate);
    }
    std::sort(alive.begin(), alive.end());
    alive.erase(std::unique(alive.begin(), alive.end()), alive.end());
    return alive;
  }

  struct End
  {
    std::size_t station{0}, node{0}, length{0};
    double score{0.0};
  };
  // Every node that ends a valid chain, best first: longest, then smallest score, then the
  // earliest station, then the lowest node index (deterministic).
  std::vector<End> rankedEnds() const
  {
    std::vector<End> ends;
    for (std::size_t s = 0; s < stations_.size(); ++s) {
      for (std::size_t i = 0; i < stations_[s].nodes.size(); ++i) {
        if (stations_[s].len[i] > 0U) ends.push_back({s, i, stations_[s].len[i], stations_[s].score[i]});
      }
    }
    std::sort(ends.begin(), ends.end(), [](const End& a, const End& b) {
        if (a.length != b.length) return a.length > b.length;
        if (std::abs(a.score - b.score) > 1e-12) return a.score < b.score;
        if (a.station != b.station) return a.station < b.station;
        return a.node < b.node;
      });
    return ends;
  }

  struct Chain
  {
    std::size_t first_station{0};
    std::vector<std::size_t> nodes;  // node index per station, first_station onward
    double score{0.0};
  };
  Chain extract(const End& end) const
  {
    Chain chain;
    chain.score = end.score;
    std::size_t station = end.station;
    std::size_t index = end.node;
    while (true) {
      chain.nodes.push_back(index);
      const std::size_t previous = stations_[station].prev[index];
      if (previous == kNone) break;
      --station;
      index = previous;
    }
    std::reverse(chain.nodes.begin(), chain.nodes.end());
    chain.first_station = station;
    return chain;
  }

private:
  static constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
  struct Station
  {
    std::vector<Node> nodes;
    std::vector<std::size_t> len;
    std::vector<double> score;
    std::vector<std::size_t> prev;
    std::unordered_map<std::size_t, std::vector<std::size_t> > by_candidate;
  };
  bool jointsContinuous(const std::vector<double>& a, const std::vector<double>& b) const
  {
    if (a.size() != b.size()) return false;
    for (std::size_t j = 0; j < a.size(); ++j) {
      if (!(std::abs(a[j] - b[j]) <= config_.max_joint_step_rad)) return false;
    }
    return true;
  }
  Config config_;
  const NeighborTable* neighbors_;
  StartOkFn start_ok_;
  EdgeOkFn edge_ok_;
  std::vector<Station> stations_;
  std::size_t best_length_{0};
};

// Longest chain whose END is accepted by end_ok(station, node_index), best first (rankedEnds
// order). Ends rejected by end_ok are skipped, never used to suppress other chains: every
// station is fed to the DP with new starts allowed, so a disconnected, fully verifiable chain
// (e.g. one that starts after an enterable but un-leavable chain) is still found.
// Returns false when no chain ends acceptably.
inline bool bestVerifiedEnd(
  const ChainDp& dp, const std::function<bool(std::size_t, std::size_t)>& end_ok,
  ChainDp::End& chosen, std::size_t min_length = 1U)
{
  for (const auto& end : dp.rankedEnds()) {
    if (end.length < min_length) break;
    if (end_ok && !end_ok(end.station, end.node)) continue;
    chosen = end;
    return true;
  }
  return false;
}

// bestVerifiedEnd restricted to chains that COVER a required station range: the chain's first
// station is <= required_first_max and its last station >= required_last_min (so the contiguous
// interval contains [required_first_max, required_last_min] when that is ordered, or at least
// reaches both bounds). Among those, longest first, then smallest score (rankedEnds order); ends
// rejected by end_ok are skipped. A node's longest chain starts as early as any chain ending at
// that node, so testing it is exact. Returns false when no verified chain covers the range. The
// unconstrained bestVerifiedEnd is unchanged.
inline bool bestVerifiedEndCovering(
  const ChainDp& dp, const std::function<bool(std::size_t, std::size_t)>& end_ok,
  ChainDp::End& chosen, std::size_t min_length, std::size_t required_first_max,
  std::size_t required_last_min)
{
  for (const auto& end : dp.rankedEnds()) {
    if (end.length < min_length) break;
    if (end.station < required_last_min) continue;
    const std::size_t first = end.station + 1U - end.length;
    if (first > required_first_max) continue;
    if (end_ok && !end_ok(end.station, end.node)) continue;
    chosen = end;
    return true;
  }
  return false;
}

// ---- left_varying_start_extra_inward_m taper (pure scalars) -----------------------------
// Horizontal inward offset (m) of a station at arc position `s` when the offset is `extra_m`
// at the window start `s_lo` and falls LINEARLY to 0 at the window end `s_hi`:
//   extra_m * (1 - clamp((s - s_lo) / (s_hi - s_lo), 0, 1)).
// Outside the window it is clamped (extra_m before s_lo, 0 after s_hi). A degenerate or
// non-finite window / non-finite or non-positive extra yields 0 (no offset).
inline double inwardTaperM(double extra_m, double s, double s_lo, double s_hi)
{
  if (!std::isfinite(extra_m) || !std::isfinite(s) || !std::isfinite(s_lo) ||
      !std::isfinite(s_hi) || !(extra_m > 0.0) || !(s_hi > s_lo)) {
    return 0.0;
  }
  const double fraction = std::min(1.0, std::max(0.0, (s - s_lo) / (s_hi - s_lo)));
  return extra_m * (1.0 - fraction);
}

}  // namespace left_varying
